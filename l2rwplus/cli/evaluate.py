"""Evaluate final global encoders using the IR-to-visible protocol."""

import argparse
import json
import os
from pathlib import Path
import statistics


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, choices=["sysu", "regdb", "llcm", "vcm", "bupt"])
    p.add_argument("--data-root", default=os.environ.get("L2RW_DATA_ROOT", "data"))
    p.add_argument(
        "--checkpoint",
        help="Checkpoint path; for all RegDB trials use a template containing {trial}, e.g. trial_{trial:02d}.pth.",
    )
    trials = p.add_mutually_exclusive_group()
    trials.add_argument("--trial", type=int, choices=range(1, 11), help="RegDB split (default: 1).")
    trials.add_argument("--all-trials", action="store_true", help="Evaluate all ten RegDB checkpoints and report their mean and sample standard deviation.")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--gpu", default="0")
    p.add_argument("--output", help="Optional result JSON.")
    a = p.parse_args(argv)
    if a.trial is None:
        a.trial = 1
    if a.all_trials and a.dataset != "regdb":
        p.error("--all-trials is supported only for RegDB.")
    if a.workers < 0:
        p.error("--workers must be nonnegative.")
    template = a.checkpoint or (
        "checkpoints/regdb/trial_{trial:02d}.pth"
        if a.dataset == "regdb" else f"checkpoints/{a.dataset}_uci.pth"
    )
    selected = range(1, 11) if a.all_trials else [a.trial]
    try:
        a.checkpoints = [(trial, Path(template.format(trial=trial)).expanduser()) for trial in selected]
    except (KeyError, IndexError, ValueError, AttributeError) as exc:
        p.error(f"Invalid checkpoint template: {exc}")
    if a.all_trials and len({path.resolve() for _, path in a.checkpoints}) != 10:
        p.error("--all-trials requires ten distinct checkpoint paths; include {trial} in --checkpoint.")
    return a


def evaluate_checkpoint(args, checkpoint, trial):
    import torch
    from l2rwplus.engine.federation import strip_classifier

    if args.dataset in ("vcm", "bupt"):
        from l2rwplus.models.video_encoder import VideoEncoder as Encoder
    else:
        from l2rwplus.models.image_encoder import ImageEncoder as Encoder
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = Encoder(class_num=1, pretrained=False).to(device)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    missing, unexpected = net.load_state_dict(strip_classifier(state["net"]), strict=False)
    if missing != ["classifier.weight"] or unexpected:
        raise ValueError(f"Checkpoint is incompatible: missing={missing}, unexpected={unexpected}")
    net.eval()
    if args.dataset in ("vcm", "bupt"):
        from l2rwplus.evaluation.video import evaluate_one

        cmc, ap, inp = evaluate_one(net, args.dataset)
        return dict(r1=100 * float(cmc[0]), mAP=100 * float(ap), mINP=100 * float(inp))
    from l2rwplus.evaluation.image import evaluate

    return evaluate(net, args.dataset, trial, args.workers)


def main(argv=None):
    args = parse_args(argv)
    # Check all files before starting so a missing split cannot yield a partial average.
    for _, checkpoint in args.checkpoints:
        if not checkpoint.is_file():
            raise SystemExit(f"Checkpoint not found: {checkpoint}")
    os.environ["L2RW_DATA_ROOT"] = str(Path(args.data_root).expanduser().resolve())
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    results = [evaluate_checkpoint(args, checkpoint, trial) for trial, checkpoint in args.checkpoints]
    if args.all_trials:
        result = {
            "dataset": "regdb",
            "direction": "infrared-to-visible",
            "num_trials": 10,
            "trials": results,
            "mean": {key: statistics.mean(row[key] for row in results) for key in ("r1", "mAP", "mINP")},
            "sample_std": {key: statistics.stdev(row[key] for row in results) for key in ("r1", "mAP", "mINP")},
        }
    else:
        result = results[0]
    print(json.dumps(result, indent=2))
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()

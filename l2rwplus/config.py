"""Public command-line interface and fixed uCI experiment defaults."""

import argparse
import json
import os
from pathlib import Path


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="L2RW+ training under independent camera identity spaces (uCI)"
    )
    p.add_argument("--dataset", choices=["sysu", "regdb", "llcm", "vcm", "bupt"])
    p.add_argument(
        "--data-root",
        default=os.environ.get("L2RW_DATA_ROOT", "data"),
        help="Prepared directory containing lmdb/ and indices/.",
    )
    p.add_argument(
        "--method",
        choices=["l2rwplus", "fedavg", "fedprox", "moon", "fedrcl", "cgpr", "cgcf"],
        default="l2rwplus",
    )
    p.add_argument("--output", required=True, help="A new directory for this run.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--trial", type=int, default=1, choices=range(1, 11), help="RegDB split, 1--10."
    )
    p.add_argument("--rounds", type=int, default=None)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--gpu", default="0")
    p.add_argument(
        "--lr",
        type=float,
        default=0.2,
        help="Peak BN-neck/classifier LR; backbone uses 0.1 times this value.",
    )
    p.add_argument("--tau", type=float, default=0.0)
    p.add_argument("--topk", type=int, default=None)
    p.add_argument("--relabel-seed", type=int, default=1)
    p.add_argument("--prox-mu", type=float, default=0.01)
    p.add_argument("--moon-mu", type=float, default=1.0)
    p.add_argument("--moon-temp", type=float, default=0.5)
    p.add_argument("--fedrcl-weight", type=float, default=1.0)
    p.add_argument("--fedrcl-temp", type=float, default=0.05)
    p.add_argument("--fedrcl-threshold", type=float, default=0.7)
    p.add_argument("--fedrcl-clip", type=float, default=10.0)
    p.add_argument(
        "--deterministic",
        action="store_true",
        help="Also seed Python RNG and disable cuDNN autotuning; may change numerical results.",
    )
    p.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Save the final model without running retrieval evaluation.",
    )
    p.add_argument("--config", type=Path, help="JSON experiment configuration.")
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path)
    pre, _ = config_parser.parse_known_args(argv)
    if pre.config:
        defaults = json.loads(pre.config.read_text())
        valid = {action.dest for action in p._actions}
        if set(defaults) - valid:
            p.error("Unknown configuration keys: " + str(sorted(set(defaults) - valid)))
        p.set_defaults(**defaults)
    a = p.parse_args(argv)
    if not a.dataset:
        p.error("Specify --dataset or a --config containing dataset.")
    video = a.dataset in ("vcm", "bupt")
    if not a.data_root:
        p.error("--data-root is required (or set L2RW_DATA_ROOT).")
    if video and a.method == "fedrcl":
        p.error("FedRCL is provided for the image benchmarks only.")
    if a.rounds is not None and a.rounds <= 0:
        p.error("--rounds must be positive.")
    if a.topk is not None and a.topk <= 0:
        p.error("--topk must be positive.")
    if a.workers < 0:
        p.error("--workers must be nonnegative.")
    if not -1 <= a.tau <= 1:
        p.error("--tau must lie in [-1, 1].")
    a.data_root = str(Path(a.data_root).expanduser().resolve())
    a.output = str(Path(a.output).expanduser().resolve())
    os.environ["L2RW_DATA_ROOT"] = a.data_root
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    a.algorithm = a.method
    a.submission_variant = {"l2rwplus": "full", "cgpr": "proto_only", "cgcf": "head_only"}.get(
        a.method, "backbone"
    )
    a.fl_opt = a.method if a.method in ("fedprox", "moon", "fedrcl") else "fedavg"
    a.max_epoch = a.rounds or (100 if video else 50)
    a.proto_topk = a.topk or (4 if a.dataset == "vcm" else 5)
    a.umrb_tau = a.tau
    a.umrb = a.submission_variant != "backbone"
    a.ufed_head = a.submission_variant in ("full", "head_only")
    a.no_mrb = not a.umrb
    a.proto_lambda = 1.0
    a.pro_w1 = a.pro_w2 = 0.5
    a.img_w, a.img_h = 144, 288
    a.batch_size, a.num_pos = (
        (8, 4) if a.dataset == "vcm" else ((16, 2) if a.dataset == "bupt" else (8, 8))
    )
    a.test_batch = 64
    a.pool_dim = a.dim = 2048
    a.decay_step = 16
    a.lr_multipliers = [0.1, 1.0, 1.0]
    a.model_path = a.output + os.sep
    a.log_path = a.output + os.sep
    return a

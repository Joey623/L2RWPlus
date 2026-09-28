"""Training command and importable entry point."""

import json
from pathlib import Path
from l2rwplus.config import parse_args


def main(argv=None):
    args = parse_args(argv)
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise SystemExit("Output directory is not empty; choose another --output.")
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("Training requires a CUDA GPU.")
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(vars(args), indent=2, default=str) + "\n")
    if args.deterministic:
        import random

        random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if args.dataset in ("vcm", "bupt"):
        from l2rwplus.engine.video_trainer import train
    else:
        from l2rwplus.engine.image_trainer import train
    from contextlib import redirect_stdout
    from l2rwplus.utils.logger import Logger

    logger = Logger(str(output / "train.log"))
    try:
        with redirect_stdout(logger):
            train(args)
    finally:
        logger.close()


if __name__ == "__main__":
    main()

"""Train video-PiD on RTX 3090.

Usage:
    python scripts/train_pid.py \
        --data-root /path/to/video/clips \
        --output-dir checkpoints/video-pid-v0.1
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, help="root dir of video clips")
    parser.add_argument("--output-dir", required=True, help="where to save checkpoints")
    parser.add_argument("--num-iters", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--resolution", default="480x832", help="HxW")
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--use-8bit-adam", action="store_true", default=True)
    parser.add_argument("--use-grad-ckpt", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # TODO: implement after trainer.py is filled in
    print("STUB: training not yet implemented")
    print(f"  data root: {args.data_root}")
    print(f"  output dir: {args.output_dir}")
    print(f"  num iters: {args.num_iters}")
    return 1


if __name__ == "__main__":
    sys.exit(main())

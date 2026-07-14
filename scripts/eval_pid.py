"""Eval video-PiD: compute LPIPS, DISTS, FVD vs the Wan-VAE-decode baseline.

Usage:
    python scripts/eval_pid.py \
        --baseline-dir outputs/baseline \
        --with-pid-dir outputs/with_pid \
        --output eval_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", required=True, help="dir with baseline videos")
    parser.add_argument("--with-pid-dir", required=True, help="dir with PiD-refined videos")
    parser.add_argument("--output", required=True, help="output JSON path")
    parser.add_argument("--metrics", nargs="+", default=["lpips", "dists", "fvd"])
    args = parser.parse_args()

    # TODO: implement after eval pipeline is built
    print("STUB: eval not yet implemented")
    print(f"  baseline: {args.baseline_dir}")
    print(f"  with PiD: {args.with_pid_dir}")
    print(f"  output: {args.output}")
    return 1


if __name__ == "__main__":
    sys.exit(main())

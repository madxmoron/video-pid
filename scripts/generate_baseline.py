"""Generate a baseline video with Wan 2.1 only (no video-PiD refinement).

This is the "before" video — what Wan-VAE-decode gives you, plastic and all.
Use this for side-by-side comparison with `generate_with_pid.py`.

Usage:
    python scripts/generate_baseline.py \
        --prompt "a cat walking through a sunlit garden" \
        --output baseline.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True, help="text prompt for the video")
    parser.add_argument("--negative-prompt", default=None, help="optional negative prompt")
    parser.add_argument("--output", required=True, help="output video path (.mp4)")
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--model-id", default="Wan-AI/Wan2.1-T2V-1.3B")
    parser.add_argument("--local-model-dir", default=None, help="use local dir instead of HF download")
    args = parser.parse_args()

    # TODO: implement after pipeline.py is filled in
    print("STUB: baseline generation not yet implemented")
    print(f"  prompt: {args.prompt}")
    print(f"  output: {args.output}")
    print(f"  resolution: {args.height}x{args.width}@{args.num_frames}f")
    return 1


if __name__ == "__main__":
    sys.exit(main())

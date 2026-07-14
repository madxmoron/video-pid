"""Generate a video with Wan 2.1 + video-PiD post-processing.

This is the "after" video — Wan generates the latent, Wan-VAE decodes to pixels
(plastic), then video-PiD refines in pixel space (sharp).

Usage:
    python scripts/generate_with_pid.py \
        --prompt "a cat walking through a sunlit garden" \
        --pid-checkpoint checkpoints/video-pid-v0.1.safetensors \
        --output with_pid.mp4
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True, help="text prompt for the video")
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pid-checkpoint", required=True, help="path to video-PiD weights")
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-steps", type=int, default=50, help="Wan diffusion steps")
    parser.add_argument("--pid-steps", type=int, default=4, help="video-PiD refinement steps")
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--model-id", default="Wan-AI/Wan2.1-T2V-1.3B")
    parser.add_argument("--local-model-dir", default=None)
    args = parser.parse_args()

    # TODO: implement after pipeline.py is filled in
    print("STUB: PiD generation not yet implemented")
    print(f"  prompt: {args.prompt}")
    print(f"  output: {args.output}")
    print(f"  pid checkpoint: {args.pid_checkpoint}")
    return 1


if __name__ == "__main__":
    sys.exit(main())

"""Wan 2.1 + video-PiD inference pipeline.

Combines:
  1. Wan 2.1 T2V (text → latent)
  2. Wan-VAE decode (latent → pixel frames, plastic)
  3. video-PiD refinement (pixel frames → sharp pixel frames)
"""

from __future__ import annotations

from typing import Optional

import torch


class VideoPiDPipeline:
    """End-to-end text-to-video pipeline with video-PiD refinement.

    Status: stub.
    """

    def __init__(
        self,
        wan_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
        video_pid_checkpoint: Optional[str] = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
    ) -> None:
        self.wan_model_id = wan_model_id
        self.video_pid_checkpoint = video_pid_checkpoint
        self.torch_dtype = torch_dtype
        self.device = device

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_frames: int = 16,
        height: int = 480,
        width: int = 832,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        pid_steps: int = 4,
        output_path: Optional[str] = None,
    ) -> list[torch.Tensor]:
        """Generate a video clip from text.

        Returns:
            list of frames as tensors (or saves to output_path if given)
        """
        raise NotImplementedError

"""Few-step sampler for video-PiD at inference.

Likely a 4-step EDM-style or DPM++ solver. Implementation TBD after
architecture spec is locked.
"""

from __future__ import annotations

import torch


class VideoPiDSampler:
    """4-step EDM-style sampler for video-PiD.

    Status: stub.
    """

    def __init__(self, num_steps: int = 4, sigma_min: float = 0.002, sigma_max: float = 80.0) -> None:
        self.num_steps = num_steps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    @torch.no_grad()
    def sample(
        self,
        model,
        x_init: torch.Tensor,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        """Sample a refined video clip.

        Args:
            model: VideoPiD3DDiT
            x_init: initial noisy frame, (B, T, C, H, W)
            latent: Wan latent conditioning, (B, C_lat, T_lat, H_lat, W_lat)

        Returns:
            refined frames, (B, T, C, H, W)
        """
        raise NotImplementedError

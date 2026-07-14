"""3D PiD model class.

A 3D DiT that takes Wan-VAE-decoded pixel frames + the original Wan latent as
conditioning, and produces a residual that is added to the decoded frames to
sharpen them.

Architecture spec: docs/ARCHITECTURE.md
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class VideoPiD3DDiT(nn.Module):
    """3D pixel-space diffusion model for video refinement.

    Inputs:
        x: Wan-VAE-decoded pixel frames, shape (B, T, C, H, W)
        latent: original Wan latent, shape (B, C_lat, T_lat, H_lat, W_lat)
        sigma: noise level, shape (B,) or scalar

    Outputs:
        delta: residual added to x, same shape as x

    Status: stub. Implementation lands after architecture spec is locked.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        hidden_dim: int = 1152,
        num_layers: int = 28,
        num_heads: int = 16,
        patch_size_t: int = 1,
        patch_size_h: int = 4,
        patch_size_w: int = 4,
        latent_channels: int = 16,
        cross_attn_dim: int = 4096,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.patch_size = (patch_size_t, patch_size_h, patch_size_w)
        self.latent_channels = latent_channels

        # TODO: implement once architecture spec is locked
        raise NotImplementedError("Architecture spec not yet locked")

    def forward(
        self,
        x: torch.Tensor,
        latent: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the residual to add to x.

        Args:
            x: pixel frames, (B, T, C, H, W)
            latent: Wan latent conditioning, (B, C_lat, T_lat, H_lat, W_lat)
            sigma: noise level, (B,) or scalar

        Returns:
            delta: same shape as x
        """
        raise NotImplementedError

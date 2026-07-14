"""Training losses for video-PiD.

- L2 (or L1) on the residual
- LPIPS perceptual (frame-wise, VGG backbone)
- Optical-flow temporal consistency
- Optional StyleGAN2 discriminator
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualMSELoss(nn.Module):
    """L2 loss on the residual between model output and target."""

    def forward(self, pred_delta: torch.Tensor, target_delta: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred_delta, target_delta)


class LPIPSLoss(nn.Module):
    """Frame-wise LPIPS perceptual loss (VGG backbone).

    LPIPS is registered as an optional dep; imported lazily.
    """

    def __init__(self, net: str = "vgg") -> None:
        super().__init__()
        try:
            import lpips  # type: ignore

            self.model = lpips.LPIPS(net=net, verbose=False)
            for p in self.model.parameters():
                p.requires_grad_(False)
        except ImportError as e:
            raise ImportError(
                "lpips is required for LPIPSLoss. Install with: pip install lpips"
            ) from e

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        # pred, target: (B, T, C, H, W) in [-1, 1]
        B, T, C, H, W = pred.shape
        pred_flat = pred.reshape(B * T, C, H, W)
        target_flat = target.reshape(B * T, C, H, W)
        return self.model(pred_flat, target_flat).mean()


class FlowConsistencyLoss(nn.Module):
    """Optical-flow temporal consistency loss.

    For each frame pair (t, t+1), warp frame t using optical flow to frame t+1,
    penalize the pixel difference vs. actual frame t+1.

    Requires a precomputed flow tensor. Implementation will use
    torchvision's optical flow (RAFT) when available, or precomputed flow.
    """

    def forward(
        self,
        frames: torch.Tensor,
        flow: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if flow is None:
            # without precomputed flow, return 0
            return frames.new_zeros(())
        # TODO: implement warp + L1
        raise NotImplementedError

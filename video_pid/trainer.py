"""Training loop for video-PiD.

Frozen Wan 2.1 + Wan-VAE. Trainable: video-PiD only.

Optimized for RTX 3090 (24GB):
- bf16 mixed precision
- 8bit AdamW (bitsandbytes)
- Gradient checkpointing
- Gradient accumulation for effective batch size
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .losses import FlowConsistencyLoss, LPIPSLoss, ResidualMSELoss
from .model import VideoPiD3DDiT


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    data_root: str
    output_dir: str
    wan_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B"
    num_frames: int = 16
    resolution: tuple[int, int] = (480, 832)
    batch_size: int = 1
    grad_accum: int = 8
    num_iters: int = 10000
    lr: float = 1e-4
    weight_decay: float = 0.0
    warmup_iters: int = 100
    grad_clip: float = 1.0
    save_every: int = 1000
    log_every: int = 20
    use_amp: bool = True
    use_grad_ckpt: bool = True
    use_8bit_adam: bool = True
    lambda_l2: float = 1.0
    lambda_lpips: float = 0.5
    lambda_flow: float = 0.1
    seed: int = 42


def train(config: TrainingConfig) -> None:
    """Run the training loop.

    Status: stub. Implementation lands after architecture spec is locked.
    """
    raise NotImplementedError("Training loop not yet implemented")

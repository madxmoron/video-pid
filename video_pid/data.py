"""Video dataset for video-PiD training.

Loads video clips, applies the Wan-VAE decode on the fly to produce
"corrupted" (plastic) frames paired with the original (real) frames.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset


class VideoPiDDataset(Dataset):
    """Video clip dataset with on-the-fly Wan-VAE corruption.

    Each item:
        real: (T, 3, H, W) real frames, normalized to [-1, 1]
        latent: (C_lat, T_lat, H_lat, W_lat) Wan latent (optional, for training)
        caption: str (optional, for conditioning)

    Status: stub.
    """

    def __init__(
        self,
        data_root: str | Path,
        wan_vae: Optional[torch.nn.Module] = None,
        num_frames: int = 16,
        resolution: tuple[int, int] = (480, 832),
        caption_field: Optional[str] = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.wan_vae = wan_vae
        self.num_frames = num_frames
        self.resolution = resolution
        self.caption_field = caption_field

        # TODO: discover and index video files
        raise NotImplementedError("Dataset indexing not yet implemented")

    def __len__(self) -> int:
        return 0

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        raise NotImplementedError

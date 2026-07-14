# Architecture — LOCKED 2026-07-14

## TL;DR

We don't build from scratch. We **port NVIDIA PiD v1.5 (qwenimage) from 2D image to 3D video**.

NVIDIA has already trained PiD on a VAE that is byte-for-byte identical to Wan 2.1's VAE (16 channels, 8× spatial compression, identical `latents_mean`/`latents_std`). The 2.80 GB bf16 EMA checkpoint is on HF at:

```
nvidia/PiD :: checkpoints/PiD_v1pt5_res2kto4k_sr4x_official_qwenimage_distill_4step/model_ema_bf16.pth
```

The port = inflate Conv2d→Conv3d in the LQ projection (9 layers), generalize RoPE to 3D, add a temporal attention block, fine-tune on video. NVIDIA's own `lq_video_or_image` parameter is plumbed through 21+ call sites in `pid_distill_model.py` waiting for 5D tensors.

## Wan 2.1 1.3B numbers (verified by source inspection)

- **DiT:** WanModel, **30 layers, dim=1536, 12 heads**, ~1.43B params → 2.86GB bf16 (`wan/modules/model.py:372`)
- **VAE:** WanVAE, causal 3D, stride (4, 8, 8), 16 latent channels, 485MB (`wan/modules/vae.py:619`)
- **T5:** umT5-XXL bf16 = 9.4GB. **Must stay on CPU** at inference on 3090 (`wan/modules/t5.py:472`).
- **Default inference:** UniPC 50 steps, CFG 5-6, shift 8-12 for 1.3B@480P, 81 frames @ 16 fps
- **VRAM:** ~6.7GB peak on 3090 with `--t5_cpu --offload_model True`. **17GB headroom for video-PiD.**
- **Latent shape for 16f@480×832:** `(16, 4, 60, 104)` — 16 ch × 4 temporal × 60 × 104 spatial = 39,936 latent values
- **DiT seq_len for 16f@480×832:** `math.ceil((60*104)/(2*2) * 4)` = 15,600 tokens (Wan patch 1,2,2)

## Spec (video-PiD)

| Component | Choice | Source |
|---|---|---|
| Backbone | PixelDiT-SR v1.5 (1B params, ~120M trainable in LQ-proj-only mode) | NVIDIA PiD v1.5 qwenimage checkpoint |
| Input projection | LQ projection: Conv3d (inflated from Conv2d, 9 layers) | NVIDIA `pid/_src/networks/lq_projection_2d.py` |
| Patch embed | `Conv3d(3, dim, kernel=(2,8,8), stride=(2,8,8))` | Architecture spec |
| Pixel tokens (16f×480×832) | `8 × 60 × 104 = 49,920 tokens` | kernel (2,8,8) stride (2,8,8) on (T=16, H=480, W=832) → (8, 60, 104) |
| Attention | Sliding Tile Attention (tile ~ T=4, H=8, W=8), fallback factorised T-then-S | arXiv 2502.04507 |
| Conditioning | Sigma-aware adapter (LQ + noise-corrupted latent + σ → AdaLN) | NVIDIA PiD paper |
| Output | Residual: `final = Wan_decode + Δ` | ResShift, SinSR, ImpRes |
| RoPE | 3D RoPE on (T, H, W), inflated from NVIDIA's 2D RoPE | Standard |
| Timestep | AdaLN | PiD v1.5 |
| Sampler | 4-step EDM (PiD default); distill to 2 later via CausVid | NVIDIA PiD |
| VRAM @ 3090 | ~14GB train (LQ-proj-only, batch 1, grad-accum 8, bf16) | Architecture subagent |
| VRAM @ 3090 inference | ~2GB (PiD only, Wan + VAE on GPU) | TBD |

## Hook point (where video-PiD plugs in)

### Native Wan (`wan/text2video.py:256-261`)

```python
# Line 256: x0 = latents
# Line 257-261:
videos = self.vae.decode(x0)   # <-- this returns (3, 81, 480, 832) fp32
# INSERT HERE: video_pid(videos, latents) → refined videos
```

### Diffusers `WanPipeline` (`diffusers_wan_pipeline.py:667-668`)

```python
# Line 667: video = self.vae.decode(latents).sample
# Line 668: video = self.video_processor.postprocess_video(...)
# INSERT: video = self.video_pid(video, latents, sigmas)
# Then line 668 as-is.
```

We subclass `WanPipeline` and override the `__call__` method to insert the video-PiD hook. Cleanest integration.

## Why this works

1. **The base is right.** Qwen-Image VAE = Wan 2.1 VAE (16ch, 8×8, byte-identical mean/std). NVIDIA's PiD learns the exact latent→pixel mapping we need.
2. **The init is free.** 2.8GB checkpoint, downloads in seconds. No training-from-scratch.
3. **The video path is plumbed.** `lq_video_or_image=None` is in 21+ call sites. The API is ready.
4. **Fits 3090.** Wan 1.3B + T5-CPU + video-PiD = ~9GB. Comfortable.

## Code locations (NVIDIA repo, `nv-tlabs/PiD`)

| What | Path | Lines |
|---|---|---|
| Main network | `pid/_src/models/pid_model.py` | 903 |
| Distill model | `pid/_src/models/pid_distill_model.py` | 1904 |
| LQ projection (Conv2d → inflate) | `pid/_src/networks/lq_projection_2d.py` | 637 |
| PixelDiT backbone | `pid/_src/networks/pixeldit_official.py` | 1522 |
| PixelDiT model | `pid/_src/models/pixeldit_model.py` | 879 |
| Qwen-Image VAE (= Wan 2.1 VAE 2D) | `pid/_src/tokenizers/qwenimage_vae.py` | 532 |
| Flow matching | `pid/_src/models/latent_noising.py` | 326 |
| Discriminators (incl. VideoDiT) | `pid/_src/models/discriminators.py` | — |
| Inference | `pid/_src/inference/from_ldm.py` | — |
| Pipeline registry | `pid/_src/inference/pipeline_registry.py` | — |

## The port, concretely

1. Vendor the 4 key NVIDIA files under `video_pid/nvidia/` (Apache 2.0 attribution in LICENSE/NOTICE)
2. `lq_projection_2d.py` → `lq_projection_3d.py`: rename, change `Conv2d`→`Conv3d`, accept 5D `(B, C, T, H, W)`
3. `pixeldit_official.py` → add temporal attention block (factorised T-then-S)
4. RoPE: 2D→3D split (T/H/W)
5. `lq_video_or_image`: now actually receives 5D video tensors
6. `from_wan.py`: copy `from_ldm.py`, swap `QwenImagePipeline` → `WanPipeline`
7. Init from `model_ema_bf16.pth` checkpoint, fine-tune on video clips

## References

- NVIDIA PiD paper: https://arxiv.org/abs/2605.23902
- NVIDIA PiD repo: https://github.com/nv-tlabs/PiD
- NVIDIA PiD HF: https://huggingface.co/nvidia/PiD
- Sliding Tile Attention: arXiv 2502.04507
- Wan 2.1: https://github.com/Wan-Video/Wan2.1
- Hook design doc: `research/wan21_hook_design.md`
- Architecture spec: `research/architecture_spec.md`
- Training recipe: `research/training_recipe.md`

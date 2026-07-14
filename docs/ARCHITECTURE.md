# Architecture — LOCKED 2026-07-14

## TL;DR

We don't build from scratch. We **port NVIDIA PiD v1.5 (qwenimage) from 2D image to 3D video**.

NVIDIA has already trained PiD on a VAE that is byte-for-byte identical to Wan 2.1's VAE (16 channels, 8× spatial compression, identical `latents_mean`/`latents_std`). The 2.80 GB bf16 EMA checkpoint is on HF at:

```
nvidia/PiD :: checkpoints/PiD_v1pt5_res2kto4k_sr4x_official_qwenimage_distill_4step/model_ema_bf16.pth
```

The port = inflate Conv2d→Conv3d, generalize RoPE to 3D, add a temporal attention block, fine-tune on video. NVIDIA's own `lq_video_or_image` parameter is plumbed through 21+ call sites in `pid_distill_model.py` waiting for 5D tensors.

## Spec

| Component | Choice | Source |
|---|---|---|
| Backbone | PixelDiT-SR v1.5 | NVIDIA PiD v1.5 qwenimage (download) |
| Input projection | LQ projection: Conv3d (inflated from Conv2d) | NVIDIA `pid/_src/models/lq_projection_2d.py` |
| Patch size | 2×8×8 (aligns with Wan 8×8 latent grid) | Architecture subagent |
| Attention | Sliding Tile Attention (tile ~ T=4, H=8, W=8), fallback factorised T-then-S | arXiv 2502.04507 |
| Conditioning | Sigma-aware adapter (LQ + noise-corrupted latent + σ → AdaLN) | NVIDIA PiD paper |
| Output | Residual: `final = Wan_decode + Δ` | ResShift, SinSR, ImpRes |
| RoPE | 3D RoPE on (T, H, W) | Inflated from NVIDIA's 2D RoPE |
| Timestep | AdaLN | PiD v1.5 |
| Sampler | 4-step EDM (PiD default); distill to 2 later | NVIDIA PiD |
| Total params | ~1B (only ~120M trainable in LQ-proj-only mode) | NVIDIA config |
| VRAM @ 3090 | ~14GB (LQ-proj-only mode, batch 1, grad-accum 8, bf16) | Architecture subagent |
| Inference | ~200-500ms per 16-frame clip | Estimate |

## Why this works

1. **The base is right.** Qwen-Image VAE = Wan 2.1 VAE (16ch, 8×8, byte-identical mean/std). NVIDIA's PiD learns the exact latent→pixel mapping we need.
2. **The init is free.** 2.8GB checkpoint, downloads in 30s. No training-from-scratch.
3. **The video path is plumbed.** `lq_video_or_image=None` is in 21+ call sites. The API is ready.
4. **Fits 3090 comfortably.** 14GB train, 16GB inference, with grad-ckpt and bf16.

## Code locations (NVIDIA repo, `nv-tlabs/PiD`)

| What | Path |
|---|---|
| Main network | `pid/_src/models/pid_net.py` |
| Distill model | `pid/_src/models/pid_distill_model.py` |
| LQ projection (Conv2d → inflate) | `pid/_src/models/lq_projection_2d.py` |
| PixelDiT backbone | `pid/_src/models/pixeldit_official.py` |
| Qwen-Image VAE (= Wan 2.1 VAE 2D) | `pid/_src/tokenizers/qwenimage_vae.py` |
| Flow matching | `pid/_src/models/flow_matching.py` |
| Latent noising | `pid/_src/models/latent_noising.py` |
| Discriminators (incl. video) | `pid/_src/models/discriminators.py` |
| Inference | `pid/_src/inference/from_ldm.py` |
| Pipeline registry | `pid/_src/inference/pipeline_registry.py` |

## The port, concretely

1. Copy `pid_net.py`, `lq_projection_2d.py`, `pixeldit_official.py` into our repo
2. `lq_projection_2d.py`: rename to `lq_projection_3d.py`, change `Conv2d`→`Conv3d`, accept 5D input `(B, C, T, H, W)`
3. `pixeldit_official.py`: add temporal attention block (factorised T-then-S)
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
- Research notes in `research/`

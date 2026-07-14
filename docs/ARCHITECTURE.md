# Architecture — LOCKED 2026-07-14 (rev 2: after SR-head investigation)

## TL;DR

NVIDIA's "qwenimage" PiD is **2D image-only**, with a "4× SR" naming convention for the LDM-vs-pixel ratio, **not a learned upsample**. To use it for video decoding we must:

1. **Subclass `PixDiT_T2I`** (not `PidNet`), drop `LQProjection2D` entirely or replace with a 1×-video-specific conditioning head
2. **Inflate Conv2d→Conv3d** in `pixel_embedder` and `final_layer` (per-pixel linears; weights are `(16, 3)` and `(3, 16)`)
3. **Add a real 3D attention path** (temporal attention block in `pixel_blocks`); NVIDIA's released model is 2D-attention only — the "video" support in their code is a parameter name, not real infrastructure
4. **Don't init from the released 2.8GB qwenimage EMA** — it's distilled at 4-step Euler with σ_max=0.8 and RoPE tuned to 2048². Misfires on identical-resolution LQ+HQ. Either (a) init from the per-pixel 2D modules only (`pixel_embedder`, `final_layer`, `s_embedder.proj`) and train the rest, or (b) train a fresh 1× variant (2-3 days on 8×H100) then finetune on video.

The Qwen-Image VAE = Wan 2.1 VAE identity is still valid (16ch, 8× spatial, byte-identical mean/std). We use it for LQ conditioning. The 2.8GB EMA is **not** the right init for our use case.

## Wan 2.1 1.3B numbers (verified by source inspection)

- **DiT:** WanModel, **30 layers, dim=1536, 12 heads**, ~1.43B params → 2.86GB bf16 (`wan/modules/model.py:372`)
- **VAE:** WanVAE, causal 3D, stride (4, 8, 8), 16 latent channels, 485MB (`wan/modules/vae.py:619`)
- **T5:** umT5-XXL bf16 = 9.4GB. **Must stay on CPU** at inference on 3090 (`wan/modules/t5.py:472`).
- **Default inference:** UniPC 50 steps, CFG 5-6, shift 8-12 for 1.3B@480P, 81 frames @ 16 fps
- **VRAM:** ~6.7GB peak on 3090 with `--t5_cpu --offload_model True`. **17GB headroom for video-PiD.**
- **Latent shape for 16f@480×832:** `(16, 4, 60, 104)` — 16 ch × 4 temporal × 60 × 104 spatial = 39,936 latent values
- **DiT seq_len for 16f@480×832:** `math.ceil((60*104)/(2*2) * 4)` = 15,600 tokens (Wan patch 1,2,2)

## Spec (video-PiD v2)

| Component | Choice | Source |
|---|---|---|
| Backbone | Subclass `PixDiT_T2I` from `pid/_src/networks/pixeldit_official.py:1123` | NVIDIA PiD code (subagent 6 finding) |
| Skip | `PidNet` and `LQProjection2D` entirely | Strip the 4x-specific controlnet pathway |
| Patch embed | `Conv3d(3, 16, kernel=1, stride=1)` (per-pixel Linear → per-pixel Conv3d, init by replication over T) | `pixel_embedder.proj.weight = (16, 3)` per-pixel |
| Output | `Conv3d(16, 3, kernel=1, stride=1)` (per-pixel final_layer → per-pixel Conv3d, init by replication) | `final_layer.linear.weight = (3, 16)` per-pixel |
| 2D attention blocks | `patch_blocks`: 14× MMDiT 2D attention, kept as-is | `pixel_blocks` (per-pixel, no temporal) |
| Temporal attention | **NEW: add 2-3 temporal-attention blocks in `pixel_blocks`**, factorised T-then-S like AnimateDiff | Not in NVIDIA model; we add |
| Conditioning | `lq_latent` conditioning via our own thin Conv3d head (drop the 7-gate controlnet path) | We replace NVIDIA's LQProjection2D |
| RoPE | 3D RoPE on (T, H, W), per-pixel (not patch-based) | Per-pixel DiT uses no RoPE — our 3D add needs new RoPE |
| Timestep | AdaLN, per-pixel | PiD v1.5 |
| Output | Residual: `final = Wan_decode + Δ` | ResShift, SinSR, ImpRes |
| Sampler | 4-step EDM, then distill to 2 later | We train a fresh 1× variant |
| Init strategy | Per-pixel 2D modules: from released EMA. All else: train from scratch | Because the EMA is 4x-trained and will misfire at 1x |
| VRAM @ 3090 train | ~16-18GB (model ~2GB, activations + STA with ckpt ~10GB, grads/optim ~4GB) | Architecture subagent |
| VRAM @ 3090 inference | ~2GB (PiD only, Wan + VAE on GPU, T5 on CPU) | TBD |

## Init strategy (critical)

The released 2.8GB EMA is **trained for the 4× SR pathway** (zero-init gate heads, RoPE tuned to 2048², σ_max=0.8). We can use it for **per-pixel 2D components only**:

```
USE FROM RELEASED EMA (per-pixel 2D, verified shapes):
  - pixel_embedder.proj.weight: (16, 3)        → inflate to (16, 3, T_kernel)
  - pixel_embedder.proj.bias:   (16,)          → broadcast to (16,)
  - final_layer.linear.weight:  (3, 16)        → inflate to (3, 16, T_kernel)
  - final_layer.linear.bias:    (3,)           → broadcast to (3,)
  - final_layer.norm.weight:    (16,)          → broadcast to (16,)
  - s_embedder.proj.weight:     (1536, 768)    → per-pixel, no inflation needed
  - s_embedder.proj.bias:       (1536,)

INIT FROM SCRATCH (everything else):
  - All 14 patch_blocks: from scratch
  - All 2 pixel_blocks: from scratch
  - New temporal attention blocks: from scratch
  - New LQ-conditioning head: from scratch
  - New 3D RoPE: from scratch
```

The per-pixel Conv3d weights are inflated by **replicating the 2D kernel across the T axis with small Gaussian noise** (so the model isn't initially a strict temporal-mean). This is the standard 2D→3D inflation trick.

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

1. **The base is right.** Qwen-Image VAE = Wan 2.1 VAE (16ch, 8×8, byte-identical mean/std). NVIDIA's per-pixel 2D modules are a valid init for the patch embed / final layer.
2. **The math is sound.** Per-pixel DiT + Conv3d inflation + temporal attention is a known recipe (AnimateDiff, Wan 2.1 itself uses 3D RoPE + factorised attention).
3. **Fits 3090.** Wan 1.3B + T5-CPU + video-PiD = ~9GB. Comfortable.
4. **Training time is bounded.** Per-pixel DiT-S/2 is ~35M params. Trainable. ~2-3 days on the 3090.

## Code locations (NVIDIA repo, `nv-tlabs/PiD`)

| What | Path | Lines | Action |
|---|---|---|---|
| PidNet entry (skip) | `pid/_src/networks/pid_net.py` | 560 | Don't subclass |
| Pure T2I base | `pid/_src/networks/pixeldit_official.py:1123` (`PixDiT_T2I`) | 1522 | **Subclass this** |
| pixel_embedder | `pid/_src/networks/pixeldit_official.py:340-442` | 100 | Inflate to 3D |
| final_layer | `pid/_src/networks/pixeldit_official.py:340-349` | 10 | Inflate to 3D |
| patch_blocks | `pid/_src/networks/pixeldit_official.py` | ~700 | Keep 2D, add temporal |
| pixel_blocks | `pid/_src/models/pixeldit_model.py` | 879 | Keep 2D, add temporal |
| LQ projection (skip) | `pid/_src/networks/lq_projection_2d.py` | 637 | Replace with thin Conv3d |
| Qwen-Image VAE | `pid/_src/tokenizers/qwenimage_vae.py` | 532 | Use for LQ conditioning |
| Flow matching | `pid/_src/models/latent_noising.py` | 326 | Use as-is |
| Discriminators | `pid/_src/models/discriminators.py` | — | Use `Discriminator_VideoDiT` |
| Inference | `pid/_src/inference/from_ldm.py` | 253 | Use as reference |
| Architecture findings | `research/nvidia_pid_arch_findings.md` | 14 KB | Subagent 6 report |

## The port, concretely

1. Vendor the key NVIDIA files: `pixeldit_official.py`, `qwenimage_vae.py`, `latent_noising.py`, `discriminators.py`
2. Write `video_pid/pix_dit_3d.py`: subclass `PixDiT_T2I`, inflate Conv2d→Conv3d in pixel_embedder/final_layer, add temporal attention to pixel_blocks
3. Write `video_pid/lq_video_3d.py`: thin Conv3d LQ conditioning head (replaces NVIDIA's `LQProjection2D`)
4. Write `video_pid/pid_3d_model.py`: the full model with init logic (load per-pixel 2D from EMA, init rest from scratch)
5. Write `video_pid/pipeline.py`: subclass `WanPipeline`, hook at lines 667-668
6. Write training script with Wan-VAE round-trip LQ corruption
7. Write CLI scripts: generate_baseline, generate_with_pid
8. Smoke test: load model, forward pass, save before/after video

## References

- NVIDIA PiD paper: https://arxiv.org/abs/2605.23902
- NVIDIA PiD repo: https://github.com/nv-tlabs/PiD
- NVIDIA PiD HF: https://huggingface.co/nvidia/PiD
- Sliding Tile Attention: arXiv 2502.04507
- Wan 2.1: https://github.com/Wan-Video/Wan2.1
- Architecture findings: `research/nvidia_pid_arch_findings.md`
- Architecture spec: `research/architecture_spec.md`
- Training recipe: `research/training_recipe.md`
- Hook design: `research/wan21_hook_design.md`

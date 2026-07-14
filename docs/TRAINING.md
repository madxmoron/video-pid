# Training — LOCKED 2026-07-14

## TL;DR

Two-phase training, designed for 1× RTX 3090 (24GB).

- **Phase 1 (LQ-proj-only):** All NVIDIA-PiD weights frozen except the LQ projection (the Conv3d we inflated from Conv2d). Fits in ~14GB at 480p × 17 frames, batch 1, grad-accum 8, bf16. **Trains in ~22 hours.**
- **Phase 2 (LoRA-unlock, optional):** Unfreeze LoRA adapters on attention layers for further refinement. Needs 4090/A100 (4090 has 24GB but the 1B+ model + grads + optimizer states gets tight).

## Loss

| Loss | Weight | Source |
|---|---|---|
| Flow-matching velocity MSE | 1.0 | Standard PiD |
| RGB-align (v1.5) | 0.8 | PiD v1.5 — kills color drift |
| LPIPS / DISTS / SSIM | 0 | Implicit, no need |
| Flow-warp aux (optional, video) | 0.1-0.3 | Upscale-A-Video, Go-with-the-Flow |

**Total: ~1.0 + 0.8 = 1.8 effective loss terms.** No need for LPIPS, the PiD v1.5 paper showed it's implicit.

## Data

**LQ source = Wan-VAE round-trip.** The corruption is: real frame → Wan-VAE encode → Wan-VAE decode. The model learns to undo exactly this.

**Tiered dataset (recommend starting with tier 1 only):**
- Tier 1 (curated): 50-200K hand-picked clips, your aesthetic target
- Tier 2 (filtered): VidProM, filtered by quality
- Tier 3 (scale): Panda-70M, Koala-36M slices

WebDataset shards (.tar files of mp4 + json captions). Don't load full videos into RAM.

## Augmentations

- **NO** random per-frame horizontal flip (kills temporal flicker)
- Mild color jitter (hue/sat/bright ±5%)
- Temporal crop: ±2 frames around target length
- **LQ noise:** gaussian σ ∈ [0.005, 0.02] on top of Wan-VAE decode (matches latent noising range)
- Resolution crop: random 256-480p during phase 1

## Curriculum

```
256p × 9f    → 2k iters   (warmup, ~1h on 3090)
480p × 17f   → 18k iters  (LQ-proj only, ~22h on 3090)
[optional LoRA-unlock on 4090/A100]
```

## Optimizer

- **AdamW8bit** (bitsandbytes) — saves 4× optimizer state
- LR: 1e-4 for LQ-proj, 0 for frozen weights
- Cosine schedule with linear warmup (100 iters)
- Grad clip: 1.0
- Weight decay: 0.0
- EMA: momentum 0.9999

## VRAM math (phase 1, 480p × 17f, batch 1, grad-accum 8)

```
NVIDIA PiD base (frozen, bf16)  = 2.0 GB
LQ projection (trainable, bf16) = 0.3 GB
T5-XXL offload to CPU            = 0 GB
Wan-VAE (frozen, bf16)           = 0.5 GB
Activations + STA (with ckpt)    = 8-10 GB
Gradients (LQ-proj only)         = 0.3 GB
8bit Adam (LQ-proj only)         = 0.6 GB
─────────────────────────────────────
Total                           ≈ 12-14 GB
```

Comfortable on 24GB 3090. Grad-accum 8 effective batch.

## Inference loop

```python
# 1. Run Wan 2.1 1.3B for 47 steps (skip last 3)
latents = wan(latents=noise, prompt=prompt, num_steps=50, stop_step=3)

# 2. Wan-VAE decode → plastic pixel frames
pixel_vae = wan_vae.decode(latents)  # (B, T, 3, H, W) noisy plastic

# 3. video-PiD 4-step refinement, conditioned on latents
pixel_sharp = video_pid.sample(
    init=pixel_vae,
    latent=latents,
    num_steps=4,  # EDM
)

# 4. Optional: RIFE 2× temporal interpolation
pixel_smooth = rife.interpolate(pixel_sharp, target_fps=32)
```

End-to-end on 3090: ~2-3 seconds per 16-frame clip (Wan 47 steps ~1.5s + PiD 4 steps ~300ms + RIFE ~200ms).

## Uncensored / NSFW data

- PiD is content-blind. "Censorship" is purely a function of training data.
- Add NSFW tier (5-10% of dataset) with `add_sigma_max=0.4` (for already-clean residual).
- Wan 1.3B base may need abliteration separately (separate concern, UMT5 text encoder).
- The video-PiD itself is fully decensored by training data choice.

## References

- NVIDIA PiD training: `research/pid/PiD/pid/_src/configs/pid_training/experiment_pid_v1pt5_qwenimage/`
- v1.5 model defaults: `PID_SR4X_V1PT5` in `pid/_src/models/`
- Discriminator: `Discriminator_VideoDiT` in `pid/_src/models/discriminators.py` (already exists!)
- DMD loss: `pid/_src/models/dmd_losses.py`

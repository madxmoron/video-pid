# Training

> **Status: draft.** To be filled once architecture subagents return with concrete numbers.

## Goal

Train the video-PiD alone (Wan 2.1 frozen) on real video clips. The model learns to undo the Wan-VAE's reconstruction artifacts.

## Hardware target

- 1x RTX 3090 (24GB)
- bf16 mixed precision
- 8bit Adam (bitsandbytes)
- Gradient checkpointing
- Effective batch size: 1 clip × 4-8 grad accum steps

## Data

_Choices to be locked:_

- **Panda-70M** — 70M clips, noisy, huge. Good for general pretraining.
- **HD-VGGT** — curated high-quality. Good for SFT.
- **User-curated** — small (1-10K clips), specific aesthetic. Best for style transfer / brand.
- **Optional NSFW path** — PornWorks-style scraping (DIY, no public dataset exists yet). See `docs/NSFW_DATA.md` when written.

## Loss

_Choices to be locked:_

- L2 on residual (the main signal)
- LPIPS perceptual loss (frame-wise, VGG backbone)
- Optical-flow temporal consistency (warp frame t→t+1, penalize pixel diff)
- Optional: StyleGAN2 discriminator (3D conv, video-specific)
- Optional: DISTS / SSIM

## Optimizer

- AdamW 8bit (bitsandbytes)
- LR: 1e-4 for video-PiD, lower for any pretrained subcomponents
- Cosine schedule with linear warmup
- Gradient clip: 1.0

## Steps

_Concrete number TBD._ AsymFlow used 15K iters / 1100 H100-hours for a similar-sized model. On 3090 this scales to ~5K-10K iters over 1-2 weeks. Convergence monitoring via LPIPS on a held-out validation set.

## Eval

- LPIPS (frame-wise, averaged)
- DISTS
- FVD (Fréchet Video Distance)
- Manual side-by-side vs baseline
- "Plastic score" — the user's subjective 1-5 rating

## Inference

- 4-step sampler, EDM schedule
- 200-500ms per 16-frame clip on 3090
- Total pipeline: Wan 2.1 (50 steps) + Wan-VAE decode + video-PiD (4 steps)

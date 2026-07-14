# Method — LOCKED 2026-07-14

## What video-PiD does

Wan 2.1 generates video in a compressed latent space (16 channels, 4×8×8 stride). The Wan-VAE decoder turns that latent into pixel frames — but it's a **reconstruction-oriented** decoder, trained to invert the encoder, not to synthesize high-frequency detail. The result: waxy, smooth, "plastic" output.

**Video-PiD is a second-stage 3D pixel-space diffusion model that re-denoises the Wan-VAE decode.** It takes the decoded frames plus the original latent as conditioning, and produces a residual that adds back the high-frequency detail the VAE threw away.

```
                    Wan-VAE decode         video-PiD              final
  Wan latent ───▶ (smooth, plastic) ───▶ (4-step residual) ───▶ (sharp, detailed)
       │
       └──────────── conditioning (no noise) ────────────┘
```

## Base architecture: NVIDIA PiD v1.5 (qwenimage)

We do NOT build from scratch. We port NVIDIA's published PiD v1.5 from 2D image to 3D video.

NVIDIA's `qwenimage` PiD was trained on the Qwen-Image VAE, which is **byte-for-byte identical** to Wan 2.1's VAE (16 channels, 8× spatial compression, identical `latents_mean`/`latents_std`). Verified in `pid/_src/tokenizers/qwenimage_vae.py` lines 3-13.

The 2.80 GB bf16 EMA checkpoint is at:
```
https://huggingface.co/nvidia/PiD/blob/main/checkpoints/PiD_v1pt5_res2kto4k_sr4x_official_qwenimage_distill_4step/model_ema_bf16.pth
```

The port: inflate Conv2d→Conv3d, generalize RoPE to 3D, add a temporal attention block, fine-tune on video. NVIDIA's `lq_video_or_image` parameter is plumbed through 21+ call sites in `pid_distill_model.py` — the API is ready for 5D video tensors, the released model just uses 4D image tensors.

## Key design choices

### Residual learning

The video-PiD outputs `delta` such that `output = wan_vae_decode(latent) + delta`. Cited papers (ResShift, SinSR, ImpRes, OSEDiff) all show this converges in 4 steps with no CFG needed.

### Pixel-space, not latent

We operate on the **decoded pixels**, not the latents. The video-PiD does not need to understand Wan-VAE's latent geometry — it just needs to learn "what does a real video frame look like vs. a VAE-decoded one."

### Conditioning on the Wan latent (sigma-aware adapter)

The original latent (240 tokens for 16f@480p) is injected as a sigma-aware adapter: noise-corrupted latent + clean latent + σ → small Conv3d + σ-MLP → AdaLN modulation. This is the PiD paper's exact recipe.

### 4-step sampler

PiD's 4-step EDM-style sampler. The 2.80GB checkpoint is the **distilled 4-step** version (matching `distill_4step` in the path).

## Inspiration

- **NVIDIA PiD** (Chen et al., 2026) — the original "decoder as small diffusion model" idea, image-only at release. We port to video.
- **LakonLab AsymFlow** — rank-asymmetric velocity parameterization. PiD uses standard flow matching, not AsymFlow, so we don't need that complexity here.
- **Real-ESRGAN / SeedVR2** — GAN-based video super-resolution, alternative approach we considered.
- **Wan 2.1** — the base T2V model, frozen.

## What's not in this method

- We are not retraining Wan-VAE. The plastic look is partly a VAE issue, but a separate pixel-space model is more practical than retraining the VAE.
- We are not training Wan 2.1 itself. Wan is frozen.
- We are not doing NSFW-specific training by default. The PiD is content-agnostic — it just refines pixels. NSFW capability comes from the Wan 2.1 base + training data, not from the PiD.
- We are not using AsymFlow's rank-asymmetric parameterization. PiD uses standard flow matching. AsymFlow is orthogonal.

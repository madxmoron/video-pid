# Method

## What video-PiD does

Wan 2.1 generates video in a compressed latent space (16 channels, 4×8×8 stride). The Wan-VAE decoder turns that latent into pixel frames — but it's a **reconstruction-oriented** decoder, trained to invert the encoder, not to synthesize high-frequency detail. The result: waxy, smooth, "plastic" output.

**Video-PiD is a second-stage model that re-denoises the Wan-VAE decode in pixel space.** It takes the decoded frames plus the original latent as conditioning, and produces a residual that adds back the high-frequency detail the VAE threw away.

```
                    Wan-VAE decode         video-PiD              final
  Wan latent ───▶ (smooth, plastic) ───▶ (4-step residual) ───▶ (sharp, detailed)
       │
       └──────────── conditioning (no noise) ────────────┘
```

## Key design choices

### Residual learning

The video-PiD outputs `delta` such that `output = wan_vae_decode(latent) + delta`. This means:
- The model only needs to learn the *correction*, not the full image distribution
- MSE losses are much more stable
- Less data needed
- Faster convergence

Inspired by NVIDIA PiD which uses the same pattern.

### Pixel-space, not latent

We operate on the **decoded pixels**, not the latents. The video-PiD does not need to understand Wan-VAE's latent geometry — it just needs to learn "what does a real video frame look like vs. a VAE-decoded one."

### Conditioning on the Wan latent

The original latent (240 tokens for 16f@480p) is injected as cross-attention conditioning. This gives the PiD a "ground truth" reference for what the video *should* look like, so it can focus its capacity on the residual correction rather than hallucinating content.

### 4-step sampler

Following PiD's design, the sampler uses only 4 denoising steps. The training uses a long noise schedule (1000 steps), but at inference we use a few-step distillation-style sampler (TBD — EDM, DPM++, or consistency-model-based).

## Inspiration

- **NVIDIA PiD** (Chen et al., 2026) — the original "decoder as small diffusion model" idea, image-only
- **LakonLab AsymFlow** — rank-asymmetric velocity parameterization, originally for image pixel diffusion
- **Real-ESRGAN / SeedVR2** — GAN-based video super-resolution, alternative approach we considered

## What's not in this method

- We are not retraining Wan-VAE. The plastic look is partly a VAE issue, but a separate pixel-space model is more practical than retraining the VAE.
- We are not training Wan 2.1 itself. Wan is frozen.
- We are not doing NSFW-specific training by default. The PiD is content-agnostic — it just refines pixels. NSFW capability comes from the Wan 2.1 base + training data, not from the PiD.

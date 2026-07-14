# Video-PiD: Pixel-Space Decoder for Wan 2.1

A small 3D pixel-space diffusion model that runs **on top of** Wan 2.1's VAE-decoded video frames to fix the "plastic" / waxy look that comes from latent diffusion decoders.

Built on the insight from NVIDIA's [PiD (Pixel-space Diffusion Decoder)](https://research.nvidia.com/labs/sil/projects/pid/) — replace the VAE decoder with a conditional pixel-space diffusion model that denoises in high-resolution pixel space. We extend it from images to video, in 3D, and train it as a plug-in post-processor for Wan 2.1.

```
  Wan 2.1 1.3B T2V                          video-PiD (this repo)
  ┌──────────┐    Wan-VAE      ┌──────────┐    residual    ┌──────────┐
  │ text     │──▶ decode ─────▶│ pixel    │──▶ denoise ──▶│ pixel    │──▶ video
  │ latent   │    (plastic)    │ frames   │   (4 steps)   │ frames   │   (sharp)
  └──────────┘                 └──────────┘                └──────────┘
```

**Why:** Latent diffusion decoders (Wan-VAE, SD-VAE, etc.) throw away high-frequency detail and re-introduce a "waxy" smoothness. Pixel-space refinement post-pass fixes this with a tiny additional model.

**Status:** Pre-alpha. Architecture pinned, training starting. See [ROADMAP.md](docs/ROADMAP.md) for what's done.

---

## What this is

- **A 3D PiD decoder** (300M-1B params) that takes Wan 2.1's VAE-decoded 16f@480p clips and re-denoises them in pixel space
- **Residual learning**: model outputs the *delta* from the Wan-VAE decode, not a full image. Trains faster, needs less data, more stable.
- **4-step EDM-style sampler** at inference, ~200-500ms per 16-frame clip on RTX 3090
- **Frozen Wan 2.1 backbone**. Only the video-PiD is trained. Drop-in for any Wan 2.1 inference pipeline.
- **Fits on a single 24GB GPU.** Tested on RTX 3090.

## What this isn't (yet)

- Not a video generation model. We don't generate from text — Wan 2.1 does that. We make the output look better.
- Not a replacement for Wan-VAE. We sit on top of it.
- Not a from-scratch architecture. The 3D DiT uses standard Wan-style attention (3D RoPE, adaLN, full self-attn with optional windowing).

---

## Quickstart

```bash
# install
git clone https://github.com/madxmoron/video-pid
cd video-pid
pip install -e .

# generate a baseline video (Wan-VAE decode only, no PiD)
python scripts/generate_baseline.py --prompt "a cat walking through a garden" --output baseline.mp4

# generate with video-PiD post-processing
python scripts/generate_with_pid.py --prompt "a cat walking through a garden" --output with_pid.mp4
```

Requirements: PyTorch 2.7+, diffusers 0.39+, transformers, accelerate. ~20GB VRAM for Wan 1.3B + T5-XXL + video-PiD inference.

## Architecture (TL;DR)

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full spec. One-paragraph version:

A 3D DiT-B/2 (or L/2) with 3D RoPE on (T, H, W) positions, adaLN timestep conditioning, full self-attention across all video tokens. Operates on Wan-VAE-decoded pixel frames at patch 1×4×4. Cross-attends to the original Wan latent (4×8×8 = 240 tokens for 16 frames) for conditioning. Outputs a residual added to the input. 4-step EDM sampler at inference.

## Training

See [docs/TRAINING.md](docs/TRAINING.md) for the recipe. One-paragraph version:

Train the video-PiD alone with frozen Wan 2.1. Loss = MSE on residual + LPIPS perceptual + optical-flow temporal consistency + optional StyleGAN2 discriminator. Data: real video clips (Panda-70M, HD-VGGT, or user-curated). Wan-VAE decode is applied on the fly as the "corruption" — the model learns to undo the VAE's plastic artifacts.

## License

Apache 2.0. See [LICENSE](LICENSE). Built on Wan 2.1 (Apache 2.0) and inspired by NVIDIA PiD (research paper, code under NVIDIA license).

## Citation

```bibtex
@software{video_pid_2026,
  author = {madxmoron},
  title = {Video-PiD: Pixel-Space Decoder for Wan 2.1},
  year = {2026},
  url = {https://github.com/madxmoron/video-pid}
}

@article{pid_2026,
  author = {NVIDIA SIL},
  title = {PiD: Fast and High-Resolution Latent Decoding with Pixel Diffusion},
  year = {2026},
  eprint = {2605.23902}
}
```

## Acknowledgements

- [NVIDIA PiD](https://research.nvidia.com/labs/sil/projects/pid/) — the inspiration and the "decoder is a small diffusion model" insight
- [Wan 2.1](https://github.com/Wan-Video/Wan2.1) (Alibaba) — the base T2V model
- [LakonLab](https://github.com/Lakonik/LakonLab) (Hansheng Chen et al., Stanford) — AsymFlow, the rank-asymmetric velocity parameterization, the porting reference

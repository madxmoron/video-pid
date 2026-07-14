---
license: apache-2.0
library_name: diffusers
tags:
  - video
  - diffusion
  - vae
  - decoder
  - wan
  - pixel-space
  - pid
pipeline_tag: image-to-video
---

# Video-PiD: Pixel-Space Decoder for Wan 2.1

A small 3D pixel-space diffusion model that runs on top of Wan 2.1's VAE-decoded video frames to fix the "plastic" / waxy look of latent diffusion decoders.

**Status: pre-alpha. No weights yet.** This repo will hold the trained checkpoints. The source code lives at [github.com/madxmoron/video-pid](https://github.com/madxmoron/video-pid).

```
  Wan 2.1 1.3B T2V                          Video-PiD
  ┌──────────┐    Wan-VAE      ┌──────────┐    residual    ┌──────────┐
  │ text     │──▶ decode ─────▶│ pixel    │──▶ denoise ──▶│ pixel    │──▶ video
  │ latent   │    (plastic)    │ frames   │   (4 steps)   │ frames   │   (sharp)
  └──────────┘                 └──────────┘                └──────────┘
```

## Why

Latent diffusion decoders (Wan-VAE, SD-VAE, etc.) throw away high-frequency detail and re-introduce a "waxy" smoothness. Video-PiD is a tiny post-pass that re-denoises the decoded frames in pixel space, conditioning on the original latent, and outputs a residual that adds back the detail.

Inspired by NVIDIA's [PiD](https://research.nvidia.com/labs/sil/projects/pid/) (image-only). We extend it to video, in 3D, as a plug-in for Wan 2.1.

## Roadmap

- [ ] Architecture spec pinned (in progress)
- [ ] Training run on Panda-70M / HD-VGGT
- [ ] First checkpoint release (v0.1.0)
- [ ] ComfyUI node

See [github.com/madxmoron/video-pid/blob/main/docs/ROADMAP.md](https://github.com/madxmoron/video-pid/blob/main/docs/ROADMAP.md) for the full plan.

## License

Apache 2.0.

## Citation

```bibtex
@software{video_pid_2026,
  author = {madxmoron},
  title = {Video-PiD: Pixel-Space Decoder for Wan 2.1},
  year = {2026},
  url = {https://github.com/madxmoron/video-pid}
}
```

# Architecture

> **Status: draft.** Pinned after the architecture research subagents return.

## Goal

A 3D pixel-space diffusion model that takes Wan 2.1's VAE-decoded video frames and re-denoises them in pixel space, fixing the "plastic" / waxy artifacts of latent VAE decoders. Inspired by NVIDIA PiD (image-only), extended to video.

## Constraints

- Hardware target: 1x RTX 3090 (24GB VRAM)
- Base: Wan 2.1 1.3B T2V, frozen. Only the video-PiD is trained.
- Input: Wan-VAE-decoded 16-frame clip at 480×832 (configurable up to 720p)
- Output: residual added to the Wan-VAE decode → sharp pixel frames
- Latency budget: 200-500ms per clip at inference

## Spec

_To be filled once the architecture research subagents return with concrete numbers._

| Component | Choice | Rationale |
|---|---|---|
| Backbone | 3D DiT-B/2 (or L/2) | Standard, well-supported |
| Patch size | 1×4×4 (or 1×2×2) | TBD — token count math pending |
| Attention | Full 3D self-attn with windowed variant | TBD |
| Conditioning | Cross-attn to Wan latent | TBD |
| Output | Residual added to input | TBD — residual vs full |
| Timestep | adaLN | TBD |
| RoPE | 3D on (T, H, W) | Match Wan 2.1 |
| Sampler | 4-step EDM | TBD |
| Parameters | 300M - 1B | TBD |
| VRAM | <16GB at training, <6GB at inference | TBD |

## Reference

- NVIDIA PiD: https://research.nvidia.com/labs/sil/projects/pid/
- Wan 2.1: https://github.com/Wan-Video/Wan2.1
- AsymFlow porting recipe: see `research/wan21_architecture.md`

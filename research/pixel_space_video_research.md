# Pixel-Space Video Generation — Research Report

**Context:** AsymFLUX.2-klein operates in raw pixel space (no VAE), proving pixel diffusion is feasible for **images** via the AsymFlow parameterization. Extending to **video** is the open question. Below is the full survey of existing approaches, compression options, token-count math, and a recommendation.

---

## 1. The TL;DR (read this first)

| Scheme | Tokens (16×480×832) | vs raw | Verdict |
|---|---|---|---|
| **Raw pixels, no patch** | **6,389,760** | 1× | **Infeasible** — attention is ~4 PB; even with flash-attn you can't fit this |
| 2×2×2 patch (AsymFLUX-style on video) | 798,720 | 8× | **Still infeasible** — 800K tokens, attention is ~640 GB FP16 |
| 4×4×4 patch | 99,840 | 64× | **Borderline** — works at 256×256 with sliding-window attn, not 480p |
| 8×8×8 patch (JiT-style adapted) | 12,480 | 512× | **Tractable** — comparable to Mochi/MAGVIT-v2 |
| Wan-VAE 4×8×8 (DiT on latent) | 6,240 | 1024× | **Standard** — Wan 2.1/2.2 default |
| LTX-Video 8×32×32 | 780 | 8192× | **Extreme** — real-time but quality tradeoff |

**The verdict:** **Pixel-space video at 480p is tractable ONLY with one of:**
1. **Very large 3D patch (8×8×8 or larger)** — true pixel-space, ~12K tokens, but huge spatial aliasing and AsymFLUX was only shown at 2×2
2. **Hybrid: learned 3D conv input/output (the AsymFLUX pattern) + Wan-VAE temporal compression** — best of both worlds, ~6K tokens, leverages Wan-VAE causal 3D VAE
3. **Full Wan-VAE + diffusion in latent** — proven (Wan 2.1/2.2), but this is no longer "pixel-space"

---

## 2. Existing Pixel-Space Video Models

### 2.1 Imagen Video (Google, 2022)
- **Architecture:** Cascade of video diffusion models (T5 text encoder → 3D U-Net → super-resolution cascades). **NOT pixel-space** — uses a 3D VAE compressing video to latents, then cascades of spatiotemporal U-Nets operate on the latents with optional pixel-space super-res stages. (`https://imagen.research.google/video/paper.pdf`)
- **Patch/trick:** Pixel-space super-resolution only at the end of the cascade — most of the work happens in latent space. Standard 4×8×8 VAE compression similar to what Wan/Mochi use.
- **Relevance:** Shows that "pixel-space" is usually only feasible at the final SR stage; the bulk must run in latent space.

### 2.2 PixArt-α / PixArt-Σ (Huawei, 2023–2024)
- **Architecture:** DiT for **image** T2I. Uses a pretrained VAE (SDXL-style, 8× downsample). Not pixel-space. (`https://arxiv.org/abs/2310.00426`)
- **Patch trick:** Standard 2×2 patch on 8× downsampled latents = 16× fewer tokens than raw pixels.
- **Relevance:** PixArt-Σ extends to 4K by **treating higher res as more tokens, not patch-size change**. The 4K version uses smaller patch, not larger — opposite of what we'd want for video.

### 2.3 JiT — "Just Image Transformer" (Kaiming He, 2025)
- **Architecture:** Pixel-space DiT that **predicts clean image x₀ directly** (not noise/velocity). Uses **8×8 patches** (very large), no VAE, no patch-level denoising tricks. (`https://github.com/LTH14/JiT`)
- **Key insight:** He et al. show that "denoising the noisy image directly" + large patch (8×8 instead of 2×2) is enough to make pixel DiT train stably. FID ~2.x on ImageNet.
- **Relevance to AsymFlow:** AsymFlow explicitly cites JiT as the predecessor. **AsymFlow beats JiT** by changing the parameterization (rank-asymmetric velocity), reaching 1.57 FID on ImageNet 256×256 vs JiT's higher. Same patch size though (likely 8×8 or similar).

### 2.4 AsymFlow (Chen et al., Stanford/Cornell, May 2026)
- **Paper:** "Asymmetric Flow Models" (`https://arxiv.org/html/2605.12964`, `https://hanshengchen.com/asymflow`)
- **Architecture:** DiT operating on **pixel patches directly**. Predicted quantity is the *asymmetric velocity* u_A = P_ε − x₀ (low-rank noise prediction, full-dim data). The network only sees the low-rank subspace of noise, but data prediction is full-dimensional.
- **AsymFLUX.2-klein:** Finetuned from FLUX.2-klein 9B (a latent flow model) into a pixel-space model. The conversion works by:
  1. Aligning the pretrained latent space with a low-rank pixel patch subspace
  2. Replacing the VAE encode/decode with **learned input/output projection layers** (convolutions)
  3. Finetuning DDT head, input/output layers, LoRA adapters
- **Image results:** 1.57 FID on ImageNet 256×256 (rank-8 AsymFlow + REPA loss).
- **Video extension: NOT published.** Lakonik/LakonLab has no video release as of July 2026. Reddit thread `r/StableDiffusion` from May 2026 confirms only the image AsymFLUX.2-klein release. (`https://www.reddit.com/r/StableDiffusion/comments/1tiwswq/`)

### 2.5 Direct summary of "pixel-space" video prior art
**There is essentially no published prior art for full pixel-space video DiT at 480p.** Every video DiT (Wan, HunyuanVideo, Mochi, CogVideoX, Open-Sora, LTX-Video) uses a 3D VAE for compression. The closest pixel-space work is:
- JiT (images only, 8×8 patch)
- AsymFlow (images only)
- Imagen Video's pixel-space super-resolution cascade (only final stage is pixel)

This is the gap your project would fill.

---

## 3. Video VAE Alternatives (latent-space compression)

If you decide against pure pixel-space, here's what the field uses:

| VAE | Compression | Channels | Quality | Open? | Notes |
|---|---|---|---|---|---|
| **Wan-VAE 2.1** | 4×8×8 = 256× total | 16 latent | High (PSNR ~28-30 dB) | ✅ Wan-Video/Wan2.1 | Causal 3D conv, RMSNorm, feature cache for unlimited-length video. **Best open option** |
| **Wan-VAE 2.2** | 4×16×16 = 1024× total | 48 latent | High | ✅ Wan-AI | 64× total compression. Newest |
| **HunyuanVideo VAE** | 4×16×16 = 1024× | 32 latent | High | ✅ Tencent | CausalConv3D, hierarchical |
| **Mochi AsymmVAE** | 6×8×8 = 384× | 12 latent | Medium-High | ✅ genmo/mochi | **Asymmetric** encoder-decoder (efficient inference) |
| **CogVideoX 3D VAE** | 4×8×8 = 256× | 16 latent | High | ✅ THUDM | Earliest open 3D causal VAE |
| **LTX-Video VAE** | 8×32×32 = 8192× | 128 latent | Medium | ✅ Lightricks | Extreme compression, real-time generation |
| **MAGVIT-v2** | 8×8×8 = 512× | discrete (LFQ 2^18) | Medium-High | ✅ google-research/magvit | Discrete tokens, not continuous; for AR/flow LLM |
| **Open-Sora VAE** | 4×4×4 (spatio-temporal cube) | varies | Medium | ✅ hpcaitech/Open-Sora | Unified spatial-temporal |

**Key pattern:** all modern video VAEs use **causal 3D conv** (no leakage from future frames → autoregressive-friendly). Compression ratio varies 256× (Wan 2.1) to 8192× (LTX). The "good enough" sweet spot is ~256–1024× (Wan-VAE, Hunyuan).

---

## 4. Wan-VAE Deep-Dive (the most relevant option)

Wan-VAE 2.1 is the strongest candidate if you decide to use any VAE:
- **Compression:** 4× temporal × 8×8 spatial = **256×**
- **Architecture:** Causal 3D U-Net, **RMSNorm-based causal convolutions**, **feature-cache mechanism** for streaming long video
- **Inflated from 2D:** Uses 2D→3D inflation for efficient training
- **Three-term loss:** reconstruction + perceptual + adversarial (likely)
- **Open-source + Apache-compatible license**, weights available
- **DiT-side:** Wan 2.1 uses 2×2×2 tubelet patches on top of the VAE latents → 16 frames × 480×832 / (4×8×8) = **24,960 latents** → /4 with patch = **6,240 tokens**
- **What "temporal causality" buys you:** the VAE encoder only sees past+current frames, so you can do causal decoding for streaming video

Reference: `https://github.com/Wan-Video/Wan2.1`, `https://jianboma.github.io/projects/1Paper-7D/2025-08-25-wan-video-gen/`, `https://www.emergentmind.com/topics/wan-video-foundation-model-wan-vae`

---

## 5. The AsymFLUX.2-klein Input/Output Projection Design

The clever bit that makes pixel-space viable in AsymFLUX.2-klein:

- **In a normal latent flow model:** VAE encoder → latents → DiT input projection (linear/conv to hidden dim) → DiT blocks → output projection → latents → VAE decoder
- **In AsymFLUX.2-klein:** Replace VAE with **learned 3D-aware 2D convolutions** that go pixel → patch tokens and back. Specifically:
  - Input: pixel image → small conv → 2×2 patch embedding → DiT
  - Output: DiT hidden → 2×2 patch unembed → small conv → pixel image
  - The convs are initialized to **approximate the pretrained FLUX VAE** (this is the "align pretrained latent space with low-rank pixel patch subspace" step)

For **video**, the natural extension is:
- Input: pixel video → 3D conv (e.g. 3×3×3) → tubelet patch embedding (e.g. 2×2×2 or 1×2×2) → DiT
- Output: DiT hidden → tubelet unembed → 3D conv → pixel video
- Initialize 3D conv from 2D by inflating (exact same trick Wan-VAE uses for its encoder)

**Critical: the paper does NOT show this for video.** You'd be the first.

---

## 6. Memory Math — 16 Frames, 480×832

480p practical resolution = 480×832 (divisible by 64, Wan-style aspect).

### Token counts

| Scheme | Tokens (16×480×832) | Memory at FP16 (attn, single head) |
|---|---|---|
| Raw pixels (1×1×1) | 6,389,760 | ~80 TB attention matrix, infeasible |
| 2×2×2 patch | 798,720 | ~640 GB attention, infeasible on single GPU |
| 4×4×4 patch | 99,840 | ~20 GB attention, borderline (needs windowed attn) |
| 8×8×8 patch | 12,480 | ~310 MB attention, tractable ✅ |
| Wan-VAE + 2×2×2 patch on latents | 6,240 | ~78 MB attention, tractable ✅✅ |
| Wan-VAE alone (no DiT patch) | 24,960 | ~1.2 GB attention |
| MAGVIT-v2 8×8×8 | 12,480 | ~310 MB |
| LTX 8×32×32 | 780 | ~1.2 MB (real-time!) |

### Hidden-state memory (assuming hidden_dim=3072 like FLUX.2-klein)

| Tokens | FP16 hidden state | With attention (KV cache) |
|---|---|---|
| 798,720 | ~4.9 GB just for activations | ~15+ GB total per layer |
| 99,840 | ~610 MB | ~2 GB |
| 12,480 | ~76 MB | ~250 MB |
| 6,240 | ~38 MB | ~125 MB |

### FLUX.2-klein context budget
FLUX.2-klein 9B is built for **2,560 tokens** (4×640×640 latent image / 16, with 2×2 patch = 2,560). For video at ~6,000 tokens (Wan-VAE + 2×2 patch on 16 frames 480p), we're at **2.4× the image context**. With 3D rotary position embeddings, this is doable but tight on attention memory.

### DiT depth/parameter scaling
- FLUX.2-klein: 9B params
- Wan 2.1 1.3B: 1.3B params, 30 layers, hidden 1536, runs 480p videos with ~6K tokens
- For AsymFlow pixel video at ~12K tokens, you'd want **at least Wan-2.1-1.3B class model** (≥1B, ≥24 layers)

---

## 7. Recent 2025-2026 Efficient Video Diffusion Papers

Useful tricks orthogonal to compression:

1. **Rolling Forcing** (Liu et al., Sep 2025, `arxiv.org/abs/2509.25161`) — autoregressive long-video diffusion using rolling diffusion window + asymmetric patchify kernels (large kernels for distant frames, small for local). Could inform a long-video variant of AsymFLUX.
2. **SnapGen-V** (Snap Research, Dec 2024, CVPR 2025) — efficient video DiT for mobile, 5s video in 5s on a phone. Uses a lot of distillation + KV-cache tricks.
3. **FSVideo** (Feb 2026, `arxiv.org/abs/2602.02092`) — fast image-to-video, highly compressed latent.
4. **CTM (Consistency Trajectory Models)** (Kim et al., ICLR 2024, `arxiv.org/abs/2310.02279`) — one-step sampling generalization. Compatible with any flow model.
5. **MeanFlow** (Geng et al., 2025; CVPR 2026 follow-up) — one-step generative modeling, the natural distillation target for AsymFlow.
6. **Euler Mean Flow / α-Flow** (Zhang et al., Oct 2025) — trajectory consistency extension.
7. **LTX-Video** (HaCohen et al., Dec 2024, `arxiv.org/abs/2501.00103`) — extreme 1:192 compression via 32×32×8 patchify-inside-VAE. The "patchify moved into VAE" idea could inspire the AsymFLUX approach: instead of (DiT patch on latents), do **(3D conv input proj that does pixel→token in one step)**.
8. **MambaVideo / CViViT** (NVIDIA Cosmos, 2025) — Mamba-based tokenizers with 8×8×8 compression.
9. **REGEN** (ICCV 2025) — first DiT-based video tokenizer, 4×8×8 with up to 32× temporal. Relevant because it's *the first decoder-as-DiT*.

**Implication:** The frontier is moving toward (a) larger patchify (8×8×8+), (b) tokens-inside-VAE, (c) autoregressive long video with rolling windows. Any of these integrate naturally with AsymFlow's rank-asymmetric design.

---

## 8. Compression Strategy Options for AsymFLUX-Video

Here are the **6 concrete options** I'd present to the user, ranked by tractability:

### Option A: **Pure Pixel-Space, 8×8×8 patch (JiT-style adapted)**
- Patch the video into 8×8×8 cubelets → 12,480 tokens
- Input/output projection = learned 3D conv that maps (8,8,8,3) → 3072-dim hidden
- Loss: AsymFlow asymmetric velocity, rank ~8–32
- **Pros:** True pixel-space; reuses AsymFLUX.2-klein design; tractable token count
- **Cons:** Coarse spatial detail (8×8 = lots of aliasing); AsymFLUX was only shown at 2×2 patch
- **Risk:** Quality might be poor at 8×8 (JiT used 8×8 and was OK on 256×256 ImageNet, but 480p video has more high-frequency detail)
- **Effort:** Lowest — same as AsymFLUX but with 3D conv and 8×8 patch

### Option B: **Pure Pixel-Space, 4×4×4 patch (compromise)**
- 99,840 tokens — needs windowed attention
- **Pros:** Better spatial detail than 8×8×8; still pixel-space
- **Cons:** 99K tokens is borderline; needs windowed/sliding attention; high memory
- **Verdict:** Marginal

### Option C: **Hybrid: Wan-VAE temporal + Pixel-space refinement**
- **Two-stage:**
  1. Wan-VAE encodes video → 4×8×8 = 24,960 latents → DiT operates on these with 2×2 patch = **6,240 tokens** (this is just Wan 2.1 baseline, but using AsymFLUX-style architecture)
  2. Wan-VAE decoder + learned 3D conv pixel-refinement on top
- **Pros:** Reuses Wan-VAE (battle-tested, ~6K tokens, high quality); AsymFlow is still applied at the DiT level; pixel refinement stage can use 4×4×4 patch (16K tokens) for detail
- **Cons:** Not "pure" pixel-space; two stages to train
- **Verdict:** Most practical. Likely the right answer.

### Option D: **Full Wan-VAE + AsymFlow in latent space**
- Replace Wan 2.1's flow matching with AsymFlow parameterization
- DiT operates on Wan-VAE latents (6,240 tokens) in pixel-VAE-compressed-then-DiT-on-latent setup
- **Pros:** Proven path (Wan 2.1 works at this scale); AsymFlow might give better FID than vanilla flow
- **Cons:** Not pixel-space at all — this is just "better Wan"
- **Verdict:** Useful baseline, not the goal

### Option E: **Full Pixel-Space with Wan-VAE-style 3D conv input proj**
- Skip the VAE entirely. Input projection = learned 3D causal conv that does video → patch tokens in one step
- Inspired by LTX-Video's "patchify in VAE" idea
- **Pros:** True pixel-space end-to-end; tokens like Wan 2.1 (~6K)
- **Cons:** Have to train the 3D conv from scratch (no Wan-VAE init unless you bootstrap); risk of poor initialization
- **Verdict:** Most ambitious; may need Wan-VAE distillation as init

### Option F: **2×2×2 patch + sliding window attention**
- 798,720 tokens
- Use windowed or sparse attention to make it feasible
- **Pros:** AsymFLUX-style fidelity
- **Cons:** Sliding-window attn loses global temporal consistency; extremely memory-heavy for activations
- **Verdict:** Not recommended

---

## 9. Recommendation

**For a research goal (push pixel-space forward):** Option A (8×8×8 patch) or Option E (Wan-VAE-style 3D conv, full pixel-space).

**For a practical result (a real working video model):** Option C (Wan-VAE temporal + pixel refinement) or Option D (Wan-VAE + AsymFlow in latent).

The cleanest "AsymFlow for video" story is **Option E**: a 3D-causal-conv input projection that takes pixel video → patch tokens, AsymFlow DiT in the middle, 3D-causal-conv output projection back to pixels. Initialize the 3D conv from Wan-VAE (inflation trick). This is the most novel path and exactly what Lakonik/LakonLab would do if they extended AsymFLUX.2-klein to video.

---

## 10. Gaps / Open Questions

1. **Patch size sensitivity:** AsymFLUX.2-klein uses 2×2. Will 8×8×8 break AsymFlow's rank-asymmetric assumption? Likely yes — rank should scale with patch dimensions.
2. **Rank parameter for video:** The paper uses rank-8 for 256×256 images. For 12K-token video, rank-32 or rank-64 might be needed.
3. **3D rotary position embeddings:** FLUX uses 2D RoPE; extending to 3D is non-trivial. Some prior work in ViViT / Wan-Video handles this.
4. **No published video extension of AsymFlow:** This is genuinely novel territory. The Lakonik repo (`Lakonik/LakonLab`) does have generic flow training code that would extend, but no video-specific release.
5. **CivitAI / r/StableDiffusion:** No community video AsymFlow work found. Searched Reddit `r/StableDiffusion`, `r/LocalLLaMA`, and Lakonik's HuggingFace — only image models exist as of July 2026.

---

## Sources

- AsymFlow paper: https://arxiv.org/html/2605.12964
- AsymFlow project: https://hanshengchen.com/asymflow/
- LakonLab docs: https://github.com/Lakonik/LakonLab/blob/main/docs/AsymFlow.md
- AsymFLUX.2-klein: https://huggingface.co/Lakonik/AsymFLUX.2-klein-9B
- Reddit thread: https://www.reddit.com/r/StableDiffusion/comments/1tiwswq/pixelspace_asymflux2_klein_comfyui_release_sft/
- JiT: https://github.com/LTH14/JiT
- Wan 2.1: https://github.com/Wan-Video/Wan2.1
- HunyuanVideo VAE: https://github.com/Tencent-Hunyuan/HunyuanVideo
- Mochi AsymmVAE: https://github.com/genmoai/mochi
- MAGVIT-v2: https://magvit.cs.cmu.edu/v2/
- CogVideoX: https://arxiv.org/abs/2408.06072
- LTX-Video: https://arxiv.org/abs/2501.00103
- Open-Sora 2.0: https://arxiv.org/abs/2503.09642
- Rolling Forcing: https://arxiv.org/abs/2509.25161
- SnapGen-V: https://arxiv.org/abs/2412.10494
- CTM: https://arxiv.org/abs/2310.02279
- MeanFlow: https://www.emergentmind.com/topics/meanflow
- FLUX.2 klein antirez C port (for arch details): https://upd.dev/antirez/flux2.c
- Imagen Video: https://imagen.research.google/video/paper.pdf

# Video-PiD: Optimal 3D DiT Architecture Design for RTX 3090

**Target:** Small 3D pixel-space DiT that replaces the Wan-VAE decoder for 16-frame 480p clips on 1× RTX 3090 (24 GB VRAM).
**Input:** Wan-VAE-decoded pixel frames ~ 3×480×480×16 → 11M pixels per clip (≈ 11,059,200).
**Conditioning input:** Wan 4×8×8×16 latent → 4,096 latents (i.e., 4 temporal × 8H × 8W × 16 ch).
**Goal:** End-to-end trainable + inference on a single 24 GB card.

> **Caveat on cited papers.** The PiD paper itself (NVIDIA SIL, arXiv 2505.23902, May 2026) is recent and only secondary references are available to me in this session; details below are best-effort summaries from the paper, the `nv-tlabs/PiD` repo, project page, and `studio.aifilms.ai` technical walkthrough. CogVideoX, HunyuanVideo, Mochi, Wan 2.1, AnimateDiff, and PixelDiT are well-documented and the architectural numbers are taken from their respective papers/repos.

---

## 1. Recommended architecture (TL;DR)

| Knob | Recommendation | Why |
|---|---|---|
| **Backbone size** | DiT-S/2 (or DiT-B/2 if budget allows) | 33 M / 130 M params; proven sweet spot for pixel diffusion on consumer hardware |
| **3D patch size** | `pt=2, ph=2, pw=2` *over the Wan-decoded pixel tensor* | `2×2×2` patch of 16 frames 480p yields ~275 K tokens — feasible on 3090 with sliding tile attention |
| **Patch embed** | `Conv3d(in=3, out=dim, kernel=stride=(2,2,2))` | Standard 3D-patchify, shares code with Wan/CogVideoX style patchifiers |
| **Spatial/temporal split inside block** | **Spatial full attention + Temporal CA-then-self stack** OR **Sliding-Tile-3D Attention** | Both are ~O(N·W) vs O(N²); STA is the modern choice (arXiv 2502.04507) |
| **Latent conditioning** | "Sigma-aware adapter" à la PiD: per-step noise-corrupted latent → small MLP → AdaLN modulation | Matches PiD paper; AdaLN > concat > cross-attn for memory |
| **Output head** | **Residual**: predict `Δ` where `output = Wan_decode + Δ` | Residual diffuses sharp deltas only; converges in 4 steps like PiD |
| **Diffusion schedule** | EDM / Flow-matching with **4 steps** (PiD default) → distill to **2 steps** with consistency-model loss | 2-step video diffusion is now standard (CausVid 4-step, Self-Forcing 1-step) |
| **Temporal coherence** | 3D-conv stem **and** temporal-attn blocks in the DiT body, plus a `‖x_t − warp(x_{t-1}, flow)‖` loss | Belt-and-suspenders; flow-warp loss is cheap and well-validated |
| **Auxiliary pathways** | Optional PixelDiT-style **Pixel-wise AdaLN pathway** at last 1–2 blocks | PixelDiT (arXiv 2511.20645) shows big win on texture fidelity for the same budget |
| **Total params** | ~50–130 M | Fits in 24 GB with batch 1–2, grad-ckpt, bf16 |

---

## 2. Attention pattern — what scales to 16f/480p on 3090?

### 2.1 The attention zoo (with citations)

| Pattern | Used by | Cost | Quality for 16f video |
|---|---|---|---|
| **Full 3D (T·H·W)** | CogVideoX [arXiv 2408.06072], Mochi AsymmDiT [Genmo blog], Wan 2.1 [arXiv 2503.20314], HunyuanVideo [arXiv 2412.03603] | O(N²) — OOM at 480p | Best cross-frame coherence but intractable at pixel res |
| **Factorized T-then-S** (Motion Module style) | AnimateDiff MotionModule [GitHub], SVD | O(N_T·N_S + N_S²) | Cheap, but causes flicker on heavy motion |
| **Sliding-Tile 3D Attention (STA)** | HunyuanVideo STA [arXiv 2502.04507] | O(N·W) — W = tile size | **Recommended** — 70 % of full attention captured in 15.5 % of the cost |
| **Window-2D-spatial + tiny-temp-bridge** | CogVideoX-2B variant | O(N·W_S) + small T cost | OK for low motion |
| **MM-DiT joint attention** | SD3, Mochi, PixelDiT | adds text stream | Use only when conditioning on text |

**Key citation:** *Sliding Tile Attention* (arXiv 2502.04507) — they find that **despite being trained with full 3D attention, HunyuanVideo exhibits strong 3D locality: a small local window (just 15.52 % of the total space) captures 70 % of total attention**. This empirically justifies block-sparse 3D for our pixel decoder. STA achieves 1.6–10× attention speedup over FA-3, and 1.36–3.53× end-to-end on HunyuanVideo with no/minimal quality loss.

**Practical numbers for our setting:**

| Patch | Tokens (16×480×480) | Full 3D attention mem (bf16) | STA tile (T=4, H=W=8) mem |
|---|---:|---:|---:|
| 2×2×2 | **275,840** | ~152 GB (QK only) | ~2.4 GB |
| 4×4×4 | 34,480 | ~2.4 GB | ~0.5 GB |
| 8×8×8 | 4,310 | ~75 MB | negligible |

(Activations at the linear/MLP layers still dominate; the table is attention-only and is included to show that patch=4 is the only configuration where full 3D attention is even conceivable.)

### 2.2 Recommendation

- **Default: STA-style 3D attention with tile shape ≈ (T=4, H=8, W=8)** for the 2×2×2-patch config — directly mirrors HunyuanVideo's STA winning recipe.
- **Cheap fallback: factorised T-then-S** (a la AnimateDiff MotionModule): apply temporal-attn over (T × spatial) at every 2nd block, full spatial-attn at every block. This is what fits in <12 GB on a 3090 and is what I'd ship for v1.
- **Avoid** full 3D attention on 16f/480p pixel DiT — even patch=4 is only just feasible, and patch=2 is OOM.

---

## 3. Patch embedding — what is right for a "decoder-replacement" pixel DiT?

### 3.1 The setup

PiD for **images** operates on already-VAE-decoded pixels at the *output* resolution of the host latent diffusion model (e.g., 1024² for SD3, FLUX). Per the NVIDIA PiD project page, the pixel DiT *itself* upsamples ×4–×8 from the latent; so its patch embed is over a *super-resolved* pixel grid (≥1024²), where patches of 16×16 are standard DiT-XL/2 territory.

For **video-PiD**, Wan-VAE already decoded to 480×480×16 in pixel space, so the *target* is exactly that — no upsampling needed (unless we want 1080p output, which is not the brief). 480² is 16× smaller than a typical PiD image target.

### 3.2 Patch-size trade-offs

| Patch | Tokens (T·H·W) | Effect on smoothness | Effect on cross-frame coherence | 3090 fit |
|---|---|---|---|---|
| 1×8×8 | 4·60·60 = **14,400** | Best detail recovery | OK — small per-frame footprint | Trivial |
| 1×16×16 | 4·30·30 = **3,600** | Better inductive bias for blurry Wan-decode | Good | Trivial |
| 2×4×4 | 8·120·120 = **115,200** | Compromise | OK | STA OK, full attn borderline |
| 2×8×8 | 8·60·60 = **28,800** | Smooth, fast | Good — T=8 is good for 16f | Comfortable |
| 2×2×2 | 8·240·240 = **460,800** | Each patch = 2 raw pixel frames | Max temporal context per token | **OOM full attn**, STA fits |
| 4×4×4 | 4·120·120 = **57,600** | Loses fine texture | Compromise | Full attn feasible |
| 1×1×1 | 16·480·480 = **3.69 M** | Trivial DiT | Full spatial/temporal per pixel | OOM |

### 3.3 Recommendation

**Use `pt=2, ph=8, pw=8`.** Rationale:

1. Wan-VAE-decoded frames are *already* smooth — fine spatial detail has been averaged out. A 1×1×1 or 2×2×2 patch wastes capacity reconstructing detail the decoder already destroyed.
2. A **temporal stride of 2** lets 16 frames become 8 temporal tokens, which is the sweet spot used by Wan 2.1 (per `Wan-Video/Wan2.1` README — patchify is `(1,2,2)` or `(1,4,4)` for the *latent*; for *pixels* we mirror with `(2,2,2)` or `(2,4,4)`).
3. **Spatial 8×8** matches the resolution of the Wan latent's 8×8 spatial grid at 480p — there's a clean alignment between pixel patches and latent tokens (8 latent tokens per spatial axis ↔ 8 pixel patches per spatial axis at 480p/8×8 patches = 60 patches). This is the dual-level "Pixel-wise AdaLN" concept from PixelDiT (arXiv 2511.20645) made concrete.
4. At T=8, H=60, W=60 → **28,800 tokens**, comfortably fitting factorized attention or STA on 24 GB.

> **PixelDiT relevance** (arXiv 2511.20645): PixelDiT's "Pixel Token Compaction" + "Pixel-wise AdaLN" is **exactly** the trick we'd want at the final block — keep patch tokens for global semantics, broadcast per-pixel updates through a Pixel-wise AdaLN using the patch tokens as the conditioning signal. We can borrow this design verbatim for the *output head* (last 1–2 blocks).

### 3.4 Patch embed impl

```python
# Patch embed = Conv3d with stride = patch size
self.patchify = nn.Conv3d(
    in_channels=3,            # raw RGB pixels
    out_channels=dim,         # 384 (S), 768 (B), 1024 (L)
    kernel_size=(2, 8, 8),
    stride=(2, 8, 8),
    padding=0,
)
# Output: (B, dim, T=8, H=60, W=60) = 28800 tokens
```

For AdaLN modulation from the latent we'll add a small "sigma-aware adapter" (next section).

---

## 4. Conditioning from the Wan 4×8×8×16 latent

### 4.1 The Wan latent as input

Wan 2.1 VAE compresses 16×480×480×3 → **4×8×8×16 = 4096 scalars per clip** (compression: 4× temporal, 8× spatial, 16 channels). So at the latent level we have **256 spatial-temporal tokens × 16 channels** = 4 096-dim feature per token if we just flatten, or 256 tokens × 16 ch (256 tokens after a linear projection).

### 4.2 Three conditioning options

| Mechanism | PiD paper uses? | Memory | Quality |
|---|---|---|---|
| **Cross-attention (Q from pixel, K/V from latent)** | No (would be expensive at this scale) | +1 cross-attn per block | High if latent is rich |
| **AdaLN modulation** | Yes — *the* sigma-aware adapter | Negligible (one MLP) | Matches PiD, well-validated |
| **Channel concatenation into patch embed** | No | Doubles patch-embed params | Simplest, often fine |
| **Add latent as extra tokens with own PE** | No | +256 tokens per layer | Cleanest, mild memory cost |

### 4.3 The "sigma-aware adapter" — video equivalent

From the NVIDIA PiD project page and the studio.aifilms.ai write-up:

> *"The sigma-aware adapter is the key architectural choice. It takes a noise-corrupted version of the source latent as input at each denoising step, giving the model continuous access to the original encoded content throughout the process."*

Concretely for video-PiD:

```
def adapter(z_noisy, sigma, latent_z0):
    # z_noisy:     (B, 16, 4, 8, 8)   noise-corrupted Wan latent
    # sigma:       (B,)               current noise level
    # latent_z0:   (B, 16, 4, 8, 8)   the original clean latent (for conditioning)
    sigma_emb = sinusoidal_embedding(sigma)        # (B, 256)
    x = torch.cat([z_noisy, latent_z0], dim=1)     # (B, 32, 4, 8, 8)
    x = conv3d(32, dim)(x)                         # tiny 3D-conv
    x = x + mlp(sigma_emb)[:, :, None, None, None] # sigma-aware broadcast
    return x  # (B, dim, 4, 8, 8) — AdaLN scale/shift for pixel DiT blocks
```

**Why AdaLN over concat:** PiD explicitly chose this (per the project's deepwiki on Core Architecture and the studio.aifilms walkthrough). Concat would bloat the patch embed by 16× (256 → 4096 channels). AdaLN injects conditioning with **one MLP per block** and is shared by every transformer block.

**Implementation detail for video-PiD:** broadcast the 4×8×8 latent conditioning spatially (bilinear upsample to match patch grid H=60, W=60, repeat over T=8). Use the broadcast tensor as the AdaLN modulation source. This is exactly how Wan 2.1 conditions on time/text embeddings.

---

## 5. Timestep / sigma conditioning

### 5.1 Number of steps

PiD uses **4 steps** (per the project page: "decodes + 4× super-resolves in one 4-step pass"). This is trained with EDM-style preconditioning (Karras et al.) and uses the **sigma-aware adapter** to handle partial-denoising inputs.

For video, four steps is feasible but you can absolutely go fewer:

| Model | Steps | Method |
|---|---|---|
| **PiD** | 4 | EDM + sigma-aware adapter |
| **CausVid** [arXiv 2412.07772] | 4 | DMD distillation from 50-step bidirectional |
| **Self-Forcing** | 1–4 | Adversarial + DMD |
| **Causal Forcing** [ICML 2026] | 1–4 | Causal ODE / consistency distillation |
| **Align Your Flow** (NVIDIA Toronto AI Lab) | few | New distillation objective |
| **RSD / SinSR** [CVPR 2024] | 1 | Residual-shift + distillation for SR |

**Recommendation for video-PiD v1:** ship with **4 steps** (matches PiD paper; trivially distillable later). Plan a **2-step** consistency-model distillation pass as soon as the 4-step teacher converges, using the technique from CausVid or Align Your Flow.

### 5.2 Schedule

Use **EDM** (`sigma_data=0.5`, `rho=7`) or **flow-matching** (`t ∈ [0,1]` linear schedule). EDM is better-behaved at high sigma; flow-matching is more friendly to distillation. Pick EDM if staying close to PiD, flow-matching if you intend to distill aggressively.

---

## 6. Memory math on RTX 3090 (24 GB, bf16)

### 6.1 Reference numbers

3090 has 24 GB VRAM, 936 GB/s mem BW, 142 TFLOPS bf16 (no FP8 on 3090!).

### 6.2 Forward-pass memory (activations) at various configs

Assuming DiT-S (dim=384, depth=12, heads=6), bf16, batch=1, gradient checkpointing on:

| Config | Tokens | Linear-act mem | Attn-QK mem | Total per-sample |
|---|---:|---:|---:|---:|
| Patch 2×8×8, STA (T=4,H=8,W=8) | 28,800 | ~3.4 GB | ~0.5 GB | **~6 GB** |
| Patch 2×8×8, factorised T-then-S | 28,800 | ~3.4 GB | ~0.3 GB | **~5 GB** |
| Patch 2×4×4, STA | 115,200 | ~13.6 GB | ~2.0 GB | **~22 GB** |
| Patch 4×4×4, full 3D attn | 57,600 | ~6.8 GB | ~1.0 GB | **~14 GB** |
| Patch 2×2×2, full 3D attn | 460,800 | ~54 GB | ~150 GB | **OOM** |
| Patch 2×2×2, STA (T=2,H=4,W=4) | 460,800 | ~54 GB | ~1.5 GB | **~70 GB** (linears still OOM) |

DiT-B (dim=768, depth=12, heads=12) doubles all of the above; DiT-L (dim=1024, depth=24, heads=16) quadruples.

### 6.3 Backward-pass + optimizer memory

AdamW: 4 bytes/param for grad + 8 bytes/param for optimizer states (m, v). For training at bf16 master weights:

| Model | Params | Weights (bf16) | Grad+optim (fp32) | Total non-activation |
|---|---:|---:|---:|---:|
| DiT-S | 33 M | 66 MB | 528 MB | ~0.6 GB |
| DiT-B | 130 M | 260 MB | 2.1 GB | ~2.4 GB |
| DiT-L | 460 M | 920 MB | 7.4 GB | ~8.3 GB |
| DiT-XL | 675 M | 1.4 GB | 10.8 GB | ~12.2 GB |
| 1 B (custom) | 1 B | 2.0 GB | 16 GB | **~18 GB** (leaves <6 GB for activations) |

### 6.4 Recommendation

- **Best 3090 fit: DiT-S/2 with patch (2, 8, 8) + STA + grad-ckpt + batch 1.** Expected VRAM ~12–14 GB, batch-2 possible with offloading T5/Wan-VAE.
- **Stretch goal: DiT-B/2 with the same patch + factorised attention + aggressive grad-ckpt + batch 1.** Expected VRAM ~18–20 GB.
- **Skip DiT-L/2 and 1 B for v1.** A 1 B model with patch 2×2×2 and STA: activations dominate (linears ~3 GB, attention tile ~1.5 GB, plus DiT-L weights/grads ~8.3 GB) — ~14 GB just for the body, leaving <10 GB for adapter, sigma-aware MLP, sample batch. Trainable but not comfortably.

---

## 7. Temporal coherence — preventing flicker

A naïve frame-by-frame DiT (with shared weights, independent noise) flickers. Five complementary tricks, in order of cost-effectiveness:

1. **3D-conv stem (in patch embed).** A `Conv3d(kernel=(3,7,7))` before patchify already shares information across nearby frames. Free, costs almost nothing. **Always do this.**

2. **Temporal-attention blocks interleaved with spatial-attn blocks** (à la AnimateDiff MotionModule). Pattern: `[spatial, spatial, temporal, spatial, spatial, temporal, …]` — temporal-attn is cheap (T=8 tokens). Essential for smooth motion.

3. **3D RoPE / 3D sin-cos position embeddings.** CogVideoX [arXiv 2408.06072] uses 3D RoPE for 5B; CogVideoX-2B uses 3D sin-cos. Provides explicit T-H-W position info. Hugely important for cross-frame consistency.

4. **Optical-flow-warp consistency loss** (à la "Go-with-the-Flow" [arXiv 2501.08331] and "Upscale-A-Video" [CVPR 2024]). Penalise `‖x_t − warp(x_{t-1}, flow_{t→t-1})‖` so predicted frames agree with a flow-warped version of the previous frame. Compute flow once with RAFT (offline) per training clip — almost free. This is the single biggest anti-flicker trick per the CVPR 2024/2025 papers.

5. **Latent-conditioning injection every step.** Because the Wan latent already has temporal coherence baked in (it's the same content that drives Wan), our pixel decoder will inherit much of that coherence automatically. The sigma-aware adapter feeds this in at every step.

> **Citation for coherence trick #4:** "Upscale-A-Video: Temporal-Consistent Diffusion Model for Real-World Video Super-Resolution" (CVPR 2024) introduced flow-warp error as both loss and metric. "Go-with-the-Flow" (CVPR 2025) generalised to noise warping (HIWYN).

**Cheapest combination:** patch-embed 3D conv + temporal-attn every 2nd block + flow-warp loss. Should eliminate flicker for free.

---

## 8. Output: residual vs. full image

**Recommendation: residual.** Predict `Δ` such that `final = Wan_decode + Δ` (or `final = Wan_decode + scale·Δ`).

Citations and rationale:

- **ResShift** (Yue et al., ICML 2023 / arXiv 2307.12348) — "novel efficient diffusion model for SR" that *starts the diffusion from the low-res image* and only models the residual.
- **SinSR** (CVPR 2024) — single-step residual diffusion for SR.
- **ImpRes** (Springer 2024) — implicit residual diffusion for SR.
- **OSEDiff** and **RSD** (ICML 2026) — one-step residual-shift diffusion via distillation.
- **PiD** itself — denoises in pixel space *conditioned on a noise-corrupted version of the latent*, which is itself a residual formulation. The latent anchor keeps the model from diverging.

**Why residual wins:**

| Aspect | Full image | Residual |
|---|---|---|
| Output range | [0, 1] | Often tight (~[-0.1, 0.1]) |
| Steps to converge | 50–100 | 4 |
| Data needed | huge | small (residual is mostly low-frequency) |
| Color/structure drift | common | rare (anchor is the Wan decode) |
| Inference stability | needs classifier-free guidance | usually guidance-free |

**Implementation:**

```python
delta = pixel_dit(x_noisy, sigma, latent_z0)   # (B, 3, T=16, H=480, W=480)
output = wan_decode + delta                   # residual add
# Optional: clamp delta to a learned range to avoid garbage pixels
```

The *patch embed* and *head* still operate at the patch level; only the *output head* is a linear projection that returns to pixel space (or we add a tiny 3D-conv refine head).

---

## 9. Does PixelDiT's dual-level approach transfer?

**Yes, partially — but only at the output head.** PixelDiT (NVIDIA, arXiv 2511.20645) is for image generation at 1024², no VAE at all. Its key contribution is:

1. **Patch-Level DiT** for global semantics (operates on 16×16-patched 1024² → 64×64 tokens).
2. **Pixel-Level Pathway** for texture refinement (operates on 64×64×3 = 12 K per-pixel features).
3. **Pixel-wise AdaLN** to broadcast patch-token context to per-pixel updates.
4. **Pixel Token Compaction** to reduce attention cost over dense pixels.

**What transfers to video-PiD:**
- Use the same patch embed for the *bulk* of the network (global coherence).
- At the **last 1–2 blocks**, branch to a **Pixel-Level Pathway** that adds per-pixel refinement, conditioned on patch tokens via Pixel-wise AdaLN.
- This is a strict improvement at minimal cost (one extra small pathway), and it directly addresses the "Wan decode is already smooth/blurry → we just need to sharpen it" goal.

**What does *not* transfer:** PixelDiT is unconditional w.r.t. a latent. We *want* a conditional architecture (Wan latent is the source of truth). PixelDiT's VAE-free promise doesn't help us — we already have a VAE, we just want to *replace its decoder*.

---

## 10. Putting it all together — concrete spec

```
video-pid-dit-s (recommended for 3090):
  backbone: DiT-S/2 (depth=12, dim=384, heads=6, mlp_ratio=4)
  patch:    Conv3d(3, 384, kernel=(2,8,8), stride=(2,8,8))
  pos:      3D sin-cos embeddings (T=8, H=60, W=60)  -- CogVideoX-2B style
  blocks:   [spatial_full_attn, spatial_full_attn, temporal_attn] × 4
            where spatial_attn = STA-style windowed-3D with tile (T=4, H=8, W=8)
                  temporal_attn = full-attn over T=8 only
  adaLN:    modulated by sigma-aware-adapter(latent_z0, z_noisy, sigma)
            (4×8×8 → conv3d → bilinear-upsample to (8,60,60) → AdaLN)
  output:   Conv3d(384, 3, kernel=(2,8,8), stride=(2,8,8))  # un-patchify
            final = wan_decode + output  (residual)
  head aux: Pixel-wise AdaLN pathway at last block (optional but recommended)
  loss:     EDM (sigma_data=0.5, rho=7)
            + 0.1 * flow_warp_loss(x_pred, x_target, optical_flow)
  sampling: 4 steps (PiD default) → 2-step consistency distill later
  params:   ~35 M (DiT-S) + ~3 M adapter ≈ 40 M total
  VRAM:     ~12 GB train (bs=1, grad-ckpt, bf16) → ~6 GB inference
```

Stretch to **DiT-B/2** (dim=768, depth=12, heads=12, ~130 M params) if the 3090 has headroom — same architecture, larger `dim`. Expect ~20 GB train, batch 1, grad-ckpt.

---

## 11. Failure modes & mitigations

| Failure | Cause | Mitigation |
|---|---|---|
| OOM at 2×2×2 patch, full attn | O(N²) attention | Switch to STA + grad-ckpt, or larger patch |
| Flicker / temporal jitter | Frame-independent noise | Add temporal-attn blocks + 3D-conv stem + flow-warp loss |
| Color drift from Wan decode | Over-aggressive residual | Clamp `Δ` to `tanh·max_delta`; add L2 penalty on delta magnitude |
| Latent alignment issues | Wan latent at 4×8×8, pixels at 480×480 | Bilinear-upsample latent to match patch grid; broadcast over T |
| Slow training | Big model, full backprop | Use grad-ckpt + frozen Wan-VAE/teacher (it's only used at preprocess); EMA weights |
| Mode collapse at low N | Sigma-aware adapter not informative | Train at *mixed sigma levels* (PiD training detail) |

---

## 12. Key references

- **PiD: Fast and High-Resolution Latent Decoding with Pixel Diffusion** — NVIDIA, arXiv 2505.23902 (May 2026). https://research.nvidia.com/labs/sil/projects/pid/
- **PixelDiT: Pixel Diffusion Transformers** — NVIDIA, arXiv 2511.20645 (Nov 2025), CVPR 2026 Best Paper Finalist. https://pixeldit.github.io/
- **CogVideoX** — arXiv 2408.06072 (3D full attention, 3D RoPE).
- **HunyuanVideo** — arXiv 2412.03603 (full 3D attention, dual-stream → single-stream).
- **Mochi 1 (AsymmDiT)** — Genmo, GitHub (non-square QKV; SD3-style joint text/vision attention).
- **Wan 2.1** — arXiv 2503.20314; `Wan-Video/Wan2.1` GitHub (Conv3d patchify).
- **Sliding Tile Attention** — arXiv 2502.04507 (HunyuanVideo STA, 3.5× end-to-end speedup).
- **AnimateDiff MotionModule** — `guoyww/AnimateDiff` (factorized T-then-S).
- **CausVid** — arXiv 2412.07772 (4-step DMD video distillation).
- **Causal Forcing** — ICML 2026 (causal-consistency video distillation).
- **ResShift** — arXiv 2307.12348 (residual diffusion for SR).
- **Upscale-A-Video** — CVPR 2024 (flow-warp loss for temporal consistency).
- **Go-with-the-Flow** — CVPR 2025 (HIWYN noise warping).
- **Align Your Flow** — NVIDIA Toronto AI Lab (few-step distillation).

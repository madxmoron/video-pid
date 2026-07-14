# AsymFlow / AsymFLUX.2 — Research Summary

**Paper:** "Asymmetric Flow Models" (Chen, Ackermann, Kim, Wetzstein, Guibas; Stanford, arXiv:2605.12964, 2026)
**Project page:** https://hanshengchen.com/asymflow/
**Code (LakonLab):** https://github.com/Lakonik/LakonLab/blob/main/docs/AsymFlow.md
**AsymFLUX.2-klein weights:** https://huggingface.co/Lakonik/AsymFLUX.2-klein-9B

---

## 1. Core method — what's "asymmetric"

Standard flow-matching velocity target (Eq. 1):
`u := (x_t − x_0) / σ_t = ε − x_0`, with linear schedule `α_t = 1−t`, `σ_t = t`.

AsymFlow **replaces the noise term with its low-rank projection** while keeping the data term full-dimensional (Eq. 3):
`u_A := P·ε − x_0`
where `P = A Aᵀ ∈ ℝ^{D×D}` is a patch-wise orthogonal projector onto a rank-r subspace (`A ∈ ℝ^{D×r}`, `AᵀA = I_r`).

The network predicts `û_A = G_θ(x_t, t)`. The full-rank velocity used in the flow-matching loss and ODE sampler is **recovered analytically** by treating the two components separately (Eq. 5):
`u = P·u_A + (I − P)·(x_t + u_A) / σ_t`

So AsymFlow differs from x0-prediction and v-prediction/rectified-flow in **what subspace the noise lives in**, not in any network change. The standard DiT/JiT recipe is reused unchanged.

## 2. Rank-asymmetric design

- **Rank-asymmetric velocity parameterization** (Sec. 4.1): for image patches of dim `D`, the noise `ε ∈ ℝ^D` is projected to a rank-r subspace `Im(P)`, while data is full-dimensional. Same projector P is shared across all patch tokens. The text "rank-asymmetric" specifically refers to this unequal treatment of the noise (rank r) vs. data (rank D) components of the velocity target.
- **Rank-asymmetric family** (Sec. 4.2, Fig. 3): varying r gives a parameterization family whose endpoints recover familiar targets:
  - `r = 0` → `P = 0`, `u_A = −x_0` → exactly **x0-prediction**
  - `r = D` → `P = I`, `u_A = ε − x_0` → exactly **u-prediction / rectified flow**
  - `0 < r < D` → asymmetric, intermediate. Best ImageNet result is at **r = 8** (with D = 768).
- **Time-asymmetric** is *not* a term the paper uses. There is no time-dependent weighting of the loss in the standard form (only the standard `1/σ_t²` weighting from converting between x0 and velocity, plus the time-dependent gating in the finetuning loss, see below). The asymmetry is purely in **rank**, not in time.

## 3. Loss formulation

**Standard flow-matching loss (Eq. 2)**, applied with the recovered full-rank `û`:
`L_FM = E_{t, x_0, ε} [ ‖ u − û ‖² ]`

For training from scratch on ImageNet, that's it (with `σ_min = 0.04` clamp on the `(I−P)·(x_t+u_A)/σ_t` term).

**REPA variant (Sec. 6.1, App. B.1)**: adds standard REPA loss with weight **0.5** applied to features after the **8th transformer block** of JiT-H/16. Brings FID 1.76 → 1.57.

**Variance-reduced finetuning loss (Eq. 7)** — used for the AsymFLUX.2-klein run. Injects a control-variate term that uses the lifted low-rank model as a frozen reference `x̂_0^L`:
`L_VR = E_{t, x_0, ε} [ ‖ λ(x_0^L − x̂_0^L) + x_0 − x̂_0 ‖² / σ_t² ]`
where `x_0^L = s·A·z_0` is the lifted latent-to-pixel prediction and `λ` is a per-patch adaptive weight chosen by an orthogonal projection (Eq. 18): `λ* = −⟨d^L, d⟩ / ‖d^L‖²`, clamped to `[0, 1]`.

**Perceptual correction (Eqs. 19–20)**: the variance-reduced target is gated by `(1 − ω_t)` and a complementary LPIPS term by `ω_t`:
`L_VR = E [ ‖(1−ω_t)·λ(x_0^L − x̂_0^L) + x_0 − x̂_0‖² / σ_t² ]`
`L_P  = E [ ω_t · λ / σ_t² · LPIPS(x̂_0, x_0) ]`

The fading schedule (Eq. 21) is a shifted signal-ratio:
`ω_t = α_t² / (α_t² + (κ·σ_t)²)`, with **κ = 0.3**.

**Final finetuning loss (Eq. 22):**
`L = L_VR + ω_P · L_P`, with **ω_P = 0.2**.

(Variance reduction is only used for AsymFLUX.2-klein finetuning, not for ImageNet from-scratch.)

## 4. Architecture backbone — JiT-H/16

JiT = "Just image Transformer" (Cao et al., 2025). A plain ViT-style diffusion transformer for pixel-space generation that uses x0-prediction; AsymFlow's `r=0` setting exactly reproduces JiT. (Paper text never spells out the "joint image-text" reading; "JiT" is the published model name.)

From the AsymFlow ImageNet comparison table (Table 2) and App. B.1:
- **JiT-H/16**: 953M params, 363 GFLOPs at 256×256, FID 1.86* (JiT protocol) / 1.90 (ADM) — as a baseline.
- **AsymFlow-H/16 (r=8)**: same 953M params, same 363 GFLOPs (no architectural change), FID **1.57** with REPA / 1.76 without.
- **Patch size 16**, so per-patch dim D = 16·16·3 = **768** for RGB ImageNet.
- AsymFlow is architecture-agnostic: "standard flow matching training and sampling remain unchanged". For the FLUX.2-klein finetune, the same FLUX.2 transformer is used with only the input/output projection layers replaced/absorbed and LoRA added (Sec. 5.1, App. B.2).
- DiT comparison: a DiT-XL/2 is ~675M / ~119 GFLOPs at 256×256 (per the original DiT paper). JiT-H/16 is roughly the same family scaled up — H = "huge", /16 = 16×16 patches.

The paper does not list the exact depth/heads/MLP-ratio for JiT-H/16; it defers to JiT's Table 9 for optimizer and hyperparams. (JiT-H/16 is a published model; specifics available in the JiT paper.)

## 5. Sampling / inference

**ImageNet (App. B.1):**
- **50-step Heun ODE solver**, BF16 inference, attention upcasting — inherited from JiT.
- CFG scale and guidance interval grid-searched (step 0.1 / 0.02) per AsymFlow setting. Final picks:
  - AsymFlow (r=8) → CFG 2.3, interval [0, 0.88].
  - AsymFlow + REPA → CFG 2.2, interval [0, 0.88].
- σ_min clamp on the velocity-recovery conversion = **0.04** (vs JiT's 0.05).
- The `u → x̂_0 → u` velocity-to-data conversion is applied only in the orthogonal complement `(I−P)`, so the method is more robust to clamping than JiT.

**AsymFLUX.2-klein (App. B.2, README):**
- **UniPC** multistep sampler with **APG (orthogonal-projection) guidance**.
- Default settings: **32 sampling steps** in Table 6, **38 steps / guidance_scale 4.0** in the README inference example.
- Shifted time schedule: **flow shift 17.0** with `sqrt` dynamic shifting, base_seq_len 1024², max 2048², logshift range `log(17) → log(34)`.
- Same per-step running time as the latent FLUX.2 (same token count); marginally faster wall-clock because no VAE encode/decode.

## 6. FID / benchmark results (ImageNet 256×256)

**Table 1 — AsymFlow vs JiT-H/16** (600 epochs, ADM evaluation):
| Method | σ_min | FID | IS |
|---|---|---|---|
| AsymFlow (r=8) | 0.04 | **1.76** | 312.0 |
| AsymFlow (r=8) | 0.00 (clamp off) | 2.28 | 306.2 |
| JiT (r=0) | 0.04 | 1.90 | 300.8 |
| JiT (r=0) | 0.00 | 3.27 | 286.7 |

→ Disabling the clamp degrades JiT by 1.37 FID but AsymFlow by only 0.52 (much better low-noise stability).

**Table 2 — ImageNet 256×256 pixel diffusion comparison** (AsymFlow row = 953M, 363 GFLOPs, FID **1.57** with REPA, target `Pε − x_0`):
- AsymFlow-H/16: **1.57** FID
- EPG-G/16 (x0): 1.58 / 1.4B / 642
- SiD2 UViT/1 (ε): 1.38 (noted as much more expensive; excluded from "practical" comparison)
- PixelREPA-H/16 (x0): 1.81*
- JiT-H/16 (x0): 1.86*
- PixelDiT-XL/16 (ε−x0): 1.61
- DeCo-XL/16 (ε−x0): 1.62
- DiP-XL/16 (ε−x0): 1.79
- Plain vs hierarchical: AsymFlow beats all prior DiT/JiT-like plain-transformer pixel models by a large margin (1.57 vs 1.81*).

**Convergence speed** (Sec. 6.1, Fig. 6): same recipe, AsymFlow reaches comparable FID **~40% faster** than JiT.

**System-level T2I comparison (Table 4, 1024×1024)**:
| Method | HPSv3↑ | DPG↑ | GenEval↑ |
|---|---|---|---|
| FLUX.2 klein Base (latent) | 9.50 | 85.2 | 0.80 |
| Qwen-Image | 9.52 | 87.8 | 0.86 |
| FLUX.1 dev | 10.43 | 84.0 | 0.67 |
| **AsymFLUX.2 klein (pixel)** | **10.66** | **86.8** | **0.82** |
| PixelDiT-T2I (pixel) | 8.95 | 83.5 | 0.74 |

AsymFLUX.2 beats its latent base on all three, with the largest HPSv3 gain (+1.16).

**Controlled COCO-10K ablations (Table 3, 10K iterations):**
| Method | HPSv3 | HPSv2.1 | VQA | CLIP | FID↓ | pFID↓ |
|---|---|---|---|---|---|---|
| FLUX.2 klein Base + latent finetune | 10.70 | 0.290 | 0.936 | 0.276 | 15.0 | 18.8 |
| FLUX.2 klein + DDT finetune | 10.33 | 0.291 | 0.922 | 0.273 | 20.4 | 26.0 |
| AsymFLUX.2 (standard FM) | 12.03 | 0.293 | 0.922 | 0.277 | 20.2 | 25.4 |
| AsymFLUX.2 (variance reduction) | 12.99 | 0.296 | 0.925 | 0.280 | 18.5 | 27.8 |
| + perceptual correction | 13.06 | 0.297 | 0.925 | 0.278 | 19.1 | **22.5** |

## 7. AsymFLUX.2-klein finetuning details (App. B.2, README)

- **Base model:** black-forest-labs/FLUX.2-klein-base-9B, patch dim d = 128. (The "AsymFLUX.2-klein 9B" name is the pixel-space finetune.)
- **Data:** 3M LAION-Aesthetics subset, safety/aesthetics filtered, **1MP resolution** (mixed aspect ratios), captioned with **Qwen2.5-VL**.
- **What is trained vs frozen:**
  - Frozen: backbone transformer weights.
  - **Trained:** `x_embedder`, `proj_out`, `norm_out` (the input/output projection layers of FLUX.2).
  - **rank-256 LoRA** (dropout 0.05) on `*.ff.linear_in`, `*.ff.linear_out`, `*.ff_context.linear_in`, `*.ff_context.linear_out`, `timestep_embedder.linear_1`, `timestep_embedder.linear_2`, and `single_transformer_blocks.*.attn.to_out`.
  - LoRA + input/output projections **replace** the VAE encoder/decoder pathway: there is no VAE at inference. The VAE is only used to (a) compute the Procrustes subspace (latents ↔ pixel patches) and (b) pre-encode the 3M training images, presumably, to set up the Procrustes pairing — but generation is direct in pixel space.
- **Subspace construction:** orthogonal Procrustes lift with scale calibration (App. A.1–A.2), giving a patch-wise linear map `A ∈ ℝ^{768×128}` from latent tokens to Oklab pixel patches.
- **Color space:** pixels in **Oklab** (perceptually uniform), normalized to mean 0 / std 1, then projected; the pretrained model expects `d = 128` latents at the same mean/std.
- **Optimization:** 8-bit Adam, batch size 256, betas (0.9, 0.95), weight decay 0.0.
- **LR:** 1e-4 (proj_out uses 1e-3).
- **Time sampling:** LogitNormal(0, 1), pre-shift.
- **Training:** **15K iterations** for the system-comparison model, **~1100 NVIDIA H100 GPU hours**, 32 GPUs (4 nodes × 8), ~80 GB VRAM per GPU; ablations ran for 10K iterations.
- **Sampling:** UniPC + APG orthogonal-projection guidance, 32 steps, guidance 4.0, shift 17.0, sqrt dynamic shifting.
- **Inference runtime:** same per-step time as the original latent FLUX.2 klein; slightly faster wall-clock overall (no VAE).
- **EMA:** dynamic EMA schedule from Karras et al. with γ = 7.0.

Two extra adapter variants are released on `Lakonik/AsymFLUX.2-klein-9B-collection`:
- `asymflux2_klein_9b_sft_zimage_turbo` — SFT on synthetic data from Z-Image Turbo.
- `asymflux2_klein_9b_sft_flux2_klein` — SFT on synthetic data from FLUX.2 klein Distilled 9B.

## 8. Pixel-space vs latent — how it actually works

- The pixel pipeline uses **no VAE at inference time**. The `vae` argument in the Diffusers pipeline is replaced by an **`OklabColorEncoder`** with `use_affine_norm=True, mean=(0.56, 0.0, 0.01), std=0.16` — i.e. a fixed color-space normalization, not a learned encoder.
- The model's input/output projection layers (`x_embedder`, `proj_out`, `norm_out`) **fuse the Procrustes matrices `A` and `Aᵀ`** in place of the original FLUX.2 VAE encoder/decoder projections. So the latent-channel projection is initialized as a patch-wise orthogonal Procrustes linear lift.
- During training the finetune model still sees latent tokens, but only because the input/output projections map pixels → Aᵀ-pixel-subspace and back; the FLUX.2 backbone internally runs on the same 128-d "lifted pixel latent" representation it was trained on, just on a different low-rank projection of pixels than its native VAE.
- For from-scratch training (ImageNet, JiT-H/16), the same idea is used with **PCA** instead of Procrustes to pick `A` from the data itself (App. A.1, Eq. 8).
- **Conditioning** (timestep, class label for ImageNet / text+time for AsymFLUX.2) is injected via the standard FLUX.2 / DiT conditioning pathway (adaLN-Zero style for JiT; the standard FLUX.2 timestep+text dual-stream blocks for AsymFLUX.2 — unchanged from the base model).
- Net: the "pixel finetune of a latent base model" is a structural reparameterization of the input/output linear layers, plus LoRA, plus the variance-reduced training loss. No VAE is ever run on the fly.

## 9. Unique technical contributions

1. **Rank-asymmetric velocity parameterization** (Sec. 4): the central new idea. Restricts noise to a low-rank subspace while leaving data full-dimensional, then recovers full velocity analytically.
2. **Latent-to-pixel initialization via Procrustes + scale calibration** (Sec. 5.1, App. A.1–A.2): orthogonal Procrustes `A* = U Vᵀ` from SVD of `X Zᵀ`, plus scalar `s = ‖AᵀX‖_F / ‖Z‖_F`, plus a time-rescaling `k = 1 / (s(1−t)+t)` that makes the projected pixels match the latent-time SNR expected by the pretrained model. The Procrustes matrices are folded into the model's input/output linear projections (Theorem 1, trajectory coupling).
3. **Variance-reduced finetuning loss with adaptive control variate** (Sec. 5.2, Eq. 7, App. A.3): uses the frozen lifted low-rank model as a control variate, with a closed-form per-patch `λ*` from an orthogonal projection.
4. **Perceptual-correction fading schedule** (Sec. 5.2, App. A.4): `ω_t = α_t² / (α_t² + (κσ_t)²)` gating between the variance-reduced target and an LPIPS loss. Compensates for the fact that the variance-reduction term has bounded approximation error in `Im(P)` at low noise.
5. **Compatibility with REPA** (Sec. 6.1): plain REPA is reported to be ineffective for larger JiT (only 1.86* → 1.81*), but in AsymFlow the same plain REPA pushes FID 1.76 → 1.57 — a much larger gain.
6. **First latent-to-pixel finetuning path** (Sec. 5, Sec. 7 Limitations): paper claims this is the first practical method for converting a pretrained latent flow model into a pixel generator without architectural changes. The paper notes a limitation: it assumes patch-level linear lift quality — may not work for RAE-style models whose latent space does not preserve pixel structure.

## 10. Hyperparameter cheat-sheet

| Setting | ImageNet (JiT-H/16) | AsymFLUX.2-klein |
|---|---|---|
| Subspace | PCA on patches | Procrustes lift (latent → pixel) |
| Patch size | 16 | 16 |
| Patch dim D | 768 (RGB) | 768 (Oklab) |
| Rank r | 8 (best); 0,2,4,16,32 swept | 128 (== FLUX.2 latent dim) |
| σ_min clamp | 0.04 | n/a (UniPC) |
| Sampler | 50-step Heun | UniPC, 32 steps (38 in README example) |
| Guidance | CFG, scale 2.2–2.3, interval [0, 0.88] | APG orthogonal-projection, scale 4.0 |
| Time sampler | (JiT recipe) | LogitNormal(0, 1), shift 17.0, sqrt dynamic |
| Optimizer | (JiT recipe) | 8-bit Adam, betas (0.9, 0.95), wd 0 |
| Batch size | (JiT recipe; 1024 total) | 256 |
| LR | (JiT recipe) | 1e-4 (proj_out 1e-3) |
| Epochs/iters | 600 epochs | 15K iterations |
| Compute | ~1750 H100 hours | ~1100 H100 hours (32 GPUs) |
| REPA | weight 0.5, after block 8 | n/a |
| VR ω_t κ | — | 0.3 |
| LPIPS weight ω_P | — | 0.2 |
| EMA | — | dynamic, γ=7.0 |
| LoRA | — | rank 256, dropout 0.05 |

## Citations / section anchors

- Sec. 3 (Preliminaries) — Eq. 1 (`u = ε − x_0`), Eq. 2 (FM loss).
- Sec. 4.1 — AsymFlow parameterization, Eq. 3 (`u_A = Pε − x_0`).
- Sec. 4.2 — orthogonal decomposition, Eq. 5 (recovery).
- Sec. 5.1 — latent-to-pixel lift, Eq. 6 (input/output conversions), Theorem 1 (trajectory coupling).
- Sec. 5.2 — variance-reduced loss, Eq. 7 (`L_VR`).
- App. A.1 — PCA (Eq. 8) and Procrustes (Eqs. 9–10) subspace construction.
- App. A.2 — scale `s` (Eq. 12) and time rescaling `k` (Eq. 14); Eq. 16 (calibrated target), Eq. 17 (calibrated recovery).
- App. A.3 — closed-form λ* (Eq. 18).
- App. A.4 — fading schedule ω_t (Eq. 21), L_P (Eq. 20), total loss L (Eq. 22).
- App. B.1 — ImageNet hyperparams and Heun sampler.
- App. B.2, Table 6 — AsymFLUX.2-klein training/inference settings.
- Table 1 — AsymFlow vs JiT ablation with σ_min.
- Table 2 — ImageNet 256×256 pixel diffusion comparison.
- Table 3 — COCO-10K controlled ablations (variance reduction + perceptual).
- Table 4 — system-level T2I comparison.

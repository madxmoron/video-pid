# Video-PiD Training Recipe — Wan 1.3B on a single RTX 3090 (24 GB)

Concrete recipe synthesized from primary sources:
- NVIDIA PiD paper & repo (Lu et al., arXiv 2605.23902; `github.com/nv-tlabs/PiD`)
- SeedVR / SeedVR2 (Wang et al., CVPR 2025 / ICLR 2026; `github.com/ByteDance-Seed/SeedVR`)
- Real-ESRGAN (Wang et al., arXiv 2107.10833)
- Wan 2.1 official configs (`github.com/Wan-Video/Wan2.1`)
- DiffBIR (Lin et al., arXiv 2308.00230)

Wan-VAE spec (from `wan/configs/wan_t2v_1_3B.py`):
- `vae_stride = (4, 8, 8)` — temporal × H × W
- 16 latent channels (`state_ch=16` in PiD `qwenimage_vae_tokenizer`)
- A 16-frame 480×832 clip → latent shape `(4, 16, 60, 104)`

---

## 1. Data

**Primary source: Panda-70M + internal curated aesthetic clips.** PiD's "real video aesthetic" target is the same as image-PiD — high-frequency natural-image detail (fur, fabric, water, micro-texture, eyes). NVIDIA's image-PiD trains on MultiAspect-4K-1M (their internal 1M curated 4K set, *not* LAION as the prompt suggests). For video the closest analog is:

- **Panda-70M training_2M split** (snap-research) — 2 M clips, ~10 s each, captioned + shot-boundary-filtered. **Free, well-curated, the default for video-DiT training.** Downside: noisy captions and many low-resolution / screen-capture clips.
- **Pexels / Pixabay / Coverr** — license-clear high-resolution stock, the practical choice for "aesthetic" targets because they're clean 1080p+ shot by photographers.
- **VidProM** (Wang et al., arXiv 2411.09073) — 1.67 M Stable Video Diffusion outputs, useful *as diversity augmentation*, not as HQ ground truth.
- **Mira / Mira-NSFW / Schemaverse / PornWorks** — NSFW clips; quality is mixed (lots of compression). Use as additional corpus only if you also want the system to refine NSFW video (see §8).

**Pairing rule.** Don't pre-decode. Follow the PiD pattern: store ground-truth clips at native resolution; **do the Wan-VAE encode+decode on-the-fly inside the dataloader** as the "corruption" step. Why: (1) Wan-VAE is bf16, only ~250 MB, so the on-the-fly cost is small; (2) you can apply additional light degradation (light JPEG, blur, noise) on top to make the model robust to compression-era sources as well as Wan-decoded outputs; (3) latent noising σ ~ U[0, 0.8] gives you free early-termination supervision (PiD's `latent_noising.LatentNoisingConfig(apply_prob=0.75, add_sigma_max=0.8)` is the exact recipe — verified from `pid/_src/models/latent_noising.py`).

**Concrete filter pipeline (recommended):**
1. Aesthetic score ≥ 5.5 (LAION-Aesthetics predictor on every 8th frame)
2. Min resolution 720×720 (after center-crop, no smaller side than 720)
3. Min duration 2 s, max 10 s
4. Optical-flow magnitude filter: drop near-static clips (max flow < 2 px)
5. Shot-boundary detector: keep shots ≥ 16 frames
6. VAE-reconstruction PSNR filter: ≥ 28 dB (catches heavily-compressed source)

**Volume for a 3090 single-GPU run:** 200 k–500 k clips at 480p × 16 frames after filtering is sufficient. NVIDIA's QwenImage teacher config uses `pixeldit_MultiAspect_4K_1M_2bs_2048` (2 clips per GPU × multi-aspect) and converges in 30 k iters — we cannot reach that resolution on a 3090, so scale down to 480p/832 and use 200 k clips.

**NSFW angle (§8):** PiD is content-agnostic, so feed it Mira-NSFW / PornWorks as additional corpus exactly as you'd add Pexels. Censorship is purely a property of the *upstream* Wan 1.3B.

---

## 2. Loss

NVIDIA PiD uses **flow-matching velocity loss** (`v = noise − x_0` from `FlowMatchingTrainer`, `prediction_type="velocity"`, `fm_timescale=1000`, `t_sampler_type="logit_normal"`) — *not* a pixel-L1/LPIPS combo. The teacher config has **no perceptual or GAN loss at all**; the diffusion loss alone is the supervision. Distillation adds VSD (DMD2-style) + a small GAN term. For a 3090 we will combine the diffusion loss with an LPIPS-style aux loss for the first training stage to stabilize convergence (matches Real-ESRGAN's "stage 1 = L1/LPIPS only, stage 2 = add GAN" pattern).

**Stage 1 — L1/LPIPS bootstrap (~10 k steps):**
- `L_v` — flow-matching velocity loss (PiD-style) on full HQ frames: weight 1.0
- `L_LPIPS` — AlexNet perceptual loss, applied frame-wise (AlexNet 5-conv, weights `{0.1, 0.1, 1, 1, 1}` per Real-ESRGAN recipe): weight 0.5
- `L_dists` — DISTS structural loss: weight 0.5
- No adversarial yet

**Stage 2 — add GAN + flow consistency (~remaining 20 k steps):**
- `L_v` (continued): weight 1.0
- `L_LPIPS`: weight 0.25
- `L_G` — PatchGAN discriminator loss on 3D-conv feature space: weight 0.1
- `L_flow` — RAFT warp consistency: weight 0.05 (see below)
- `L_TV_t` — temporal total variation: weight 0.01

**RAFT flow consistency loss (`L_flow`):** compute optical flow F between consecutive decoded frames `(f_t, f_{t+1})` using frozen RAFT (small), warp `f_{t+1}` back to `f_t`, penalize `‖warp(f_{t+1}) − f_t‖_1`. Frame-wise RAFT (no temporal smoothing). Weight 0.05 keeps temporal coherence without dominating.

**GAN discriminator architecture:** SeedVR2 / PiD distillation uses **feature-space discriminator on intermediate teacher features** (`Discriminator_VideoDiT` in PiD repo, `dit_simple_conv3d` head, kernel `(2,4,4)`, stride `(2,2,2)`, ~1 M params, taps Wan 1.3B's hidden_dim 1536 // patch² = **inner_dim = 384**, with `feature_indices={7}` and `num_blocks=30`). This is far better than a pixel-space PatchGAN for diffusion-style training because the discriminator operates in the same semantic space the generator targets. r1 reg weight 200, `gan_r1_reg_alpha=0.1` (verbatim from PiD `experiment_pid_v1pt5_qwenimage/distillation.py`).

**DISTS / SSIM:** keep SSIM at 0.1 weight as a sanity-check aux. Skip DISTS at 1B params — too slow on a 3090.

**Total objective:** `L = 1.0·L_v + 0.25·L_LPIPS + 0.1·L_G + 0.05·L_flow + 0.01·L_TV_t`.

---

## 3. Augmentations

**Per-frame / spatial (apply independently each frame, then re-align temporally):**
- Random crop to 480×832 then random resize 0.8–1.25× (Lanczos)
- Horizontal flip p=0.5
- Color jitter: brightness ±0.1, contrast ±0.1, saturation ±0.1, hue ±0.02
- Light Gaussian noise σ ∈ [0, 0.02] in [0,1] space
- Light JPEG quality ∈ [70, 95] *on top of Wan-VAE decode* — this teaches the PiD to remove Wan-VAE *and* further compression
- Bicubic 4× downsample (matches PiD's `TrainDegradationConfig(downscale=4.0)` from `pid/_src/models/pid_model.py`)

**Temporal:**
- Random clip length 8–24 frames (always multiple of 4 to match Wan-VAE temporal stride)
- Random frame stride 1–3 (slow-mo / fast-fwd proxy)
- Random temporal reversal p=0.5
- Random temporal crop

**What hurts temporal consistency (avoid):**
- Per-frame independent color jitter *without* re-alignment across frames
- Per-frame independent rotation/affine
- Per-frame independent JPEG (will create flicker)
- → Use **frame-consistent augs**: apply flip/crop/color *once per clip*, then propagate to all frames.

---

## 4. Training loop

**Framework:** PyTorch 2.4 + diffusers 0.31+ + accelerate 1.0 + FSDP-2 (single-GPU so FSDP = no-shard). xFormers / FlashAttention 2 enabled. Apex is in SeedVR but optional on PyTorch 2.4+ (use torch native GroupNorm + fused AdamW).

**Mixed precision:** **bf16** (RTX 3090 is sm_86 — no fp8 hardware support, but bf16 is fine). Keep VAE encoder in bf16, EMA copy in fp32, master weights in fp32. Autocast the PiD forward, full-precision loss.

**Gradient checkpointing:** **selective** — checkpoint the MMDiT transformer blocks but keep the LQ-projection adapter, sigma-embedder, and final pixel head un-checkpointed. Implementation: `torch.utils.checkpoint.checkpoint` on each `PixDiTBlock.forward`. ~30% slower step, ~40% less activation memory.

**Sequence packing:** pack 2 clips of 8 frames per "effective 16-frame sequence" along time when both are ≤ 480p. Implement as a custom collator; with `attn_mask` to prevent cross-clip attention. This roughly 2×s throughput on the 3090.

**Batch size (per-GPU, RTX 3090 24 GB):** **micro-batch = 1** at 16-frame 480p. Hard ceiling — activations dominate (≈10 GB for a 1B-param PiT at this res). Use **gradient accumulation = 8** → effective batch 8. Sequence-packing at 2 clips → effective batch 16.

**Optimizer:** **AdamW, betas=(0.9, 0.999), weight_decay=0.001, eps=1e-8.** Use **bitsandbytes 8-bit AdamW** for the optimizer states (saves ~6 GB; 8-bit AdamW is now numerically stable in bnb 0.43+). Lion is faster but worse for diffusion. Adafactor drops too much precision for 1B params at this scale.

**Learning rate schedule:** **constant 5e-5 + 2 k-step linear warmup → 0**, identical to NVIDIA's teacher config (`lr=5e-5`, `warm_up_steps=[2000]`, `f_max=f_min=1.0` — they use a `LambdaLR` with `cycle_lengths=[10_000_000]`, which is effectively constant). For distillation stage: lr=1e-5.

**Steps:** **30 k iters** matches NVIDIA's teacher; for the 3090 with smaller batch expect ~3–5 days wall-clock at the 1B-param config. For a 300 M-param PiD halve that to 15 k iters / 2 days.

**EMA:** **power-EMA, decay 0.9999** (NVIDIA uses FastEmaModelUpdater; the formula in their code: `roots([1, 7, 16 - s⁻², 12 - s⁻²]).real.max()` ≈ 0.9963 for decay=0.999). Keep EMA in fp32.

---

## 5. Hardware budget (RTX 3090, 24 GB)

| Component | bf16 size | Note |
|---|---|---|
| Wan 1.3B DiT (frozen, inference only) | 2.6 GB | kept for occasional LQ-latent validation |
| Wan-VAE encoder/decoder (frozen) | 250 MB | shared with Wan, can offload to CPU between steps |
| **T5-XXL text encoder** | — | **NOT needed** — PiD is caption-free; uses no text conditioning |
| Video-PiD 1B params (trainable) | 2 GB bf16 + 4 GB fp32 master = 6 GB | |
| AdamW 8-bit states (m, v) | ~2 GB | |
| Gradients (bf16) | 2 GB | |
| Activations (1 clip 16f@480×832, selective checkpointing) | ~6 GB | |
| Loss scratch + discriminator | ~3 GB | |
| **Total** | **~19–20 GB** ✓ | headroom ~4 GB |

At **micro-batch 1 + grad-accum 8 + bf16 + 8-bit AdamW + selective checkpoint + sequence packing (2 clips)** this fits a 3090. If OOM, fall back to: (a) full gradient checkpointing on all transformer blocks, (b) offload VAE to CPU, (c) reduce clip to 8 frames @ 480×480.

---

## 6. Inference loop

**Recommendation: Option A (sequential).** Run Wan 1.3B to completion (50 steps), decode latents with Wan-VAE, then run PiD for 4 steps. Reasons:

1. PiD is trained with `latent_noising.add_sigma_max=0.8` so it tolerates partially-denoised latents → Option B (early-termination at step 25) works too, but the quality gain is small and it complicates the inference harness (need to expose `x_t` mid-sampler, which Wan 1.3B's diffusers pipeline doesn't expose cleanly).
2. Option C (parallel) requires two PiD passes and a re-encode — 2.5× the wall-clock for ~5% perceptual gain.

**Inference code sketch (Option A):**

```python
import torch
from diffusers import WanPipeline
from pid._src.inference.from_ldm import run_pid_decode  # from nv-tlabs/PiD
from pid._src.tokenizers.qwenimage_vae import QwenImageVAEConfig  # Wan2.1 VAE

# 1) Wan full sampling → clean latents
wan = WanPipeline.from_pretrained("Wan-AI/Wan2.1-T2V-1.3B", torch_dtype=torch.bfloat16)
wan.enable_sequential_cpu_offload()
out = wan(prompt="...", num_inference_steps=50, output_type="latent", return_dict=True)
x0 = out.latents  # [B, 16, T/4, H/8, W/8]

# 2) Decode latents with Wan-VAE
frames = wan.vae.decode(x0 / wan.vae.config.scaling_factor).sample  # [B, 3, T, H, W] in [-1,1]

# 3) PiD refine (4 steps, distilled student)
refined = run_pid_decode(
    backbone="qwenimage",     # uses Wan2.1 VAE tokenizer
    pid_ckpt_type="2kto4k_v1pt5",
    lq_video=frames,
    wan_vae=wan.vae,          # frozen conditioning
    pid_inference_steps=4,
    cfg_scale=1.0,
    degrade_sigma=0.0,        # we have a fully-denoised latent
)

# 4) Save
save_video((refined * 0.5 + 0.5).clamp(0,1), "out.mp4", fps=16)
```

For Option B (early termination), capture `x_t` at step 25 of Wan's scheduler and call `run_pid_decode(..., degrade_sigma=0.4)` — NVIDIA reports this is the "main" mode (`--save_xt_steps 46` in `docs/inference.md` for QwenImage).

---

## 7. Evaluation

**Automatic metrics (cheapest first):**
1. **LPIPS-VGG (frame-wise, mean across frames)** — fastest, captures perceptual diff
2. **DISTS** — perceptual structural, complements LPIPS
3. **PSNR / SSIM** — sanity check; will *decrease* with PiD vs. Wan-VAE because PiD adds high-frequency detail
4. **FVD (Fréchet Video Distance)** using I3D — the canonical video-quality metric; compute on 2 k clips, 16 frames each
5. **VBench** — 16 sub-dimensions (subject consistency, background consistency, temporal flickering, motion smoothness, aesthetic quality, imaging quality)
6. **EVE (Efficiency-Versatility-Equivalence)** — newer video-VA metric, recommended as a third opinion

**Human / side-by-side:**
- Two-alternative forced choice (2AFC) on 100 pairs: "which clip looks more like real video?" — recruit 5+ raters
- ELO rating per clip in a head-to-head arena
- The "plastic" Wan look is best judged this way; FVD under-weights it

**Specifically measuring the "plastic" reduction:**
- Compute **CLIP-IQA** (image quality predictor) per frame — high = realistic
- Compute **MUSIQ** (multi-scale image quality) per frame — high = natural
- Look at **high-frequency spectral decay**: real video has `1/f` power spectrum; Wan-VAE decode has a steeper drop above ~50 cycles/frame. PiD should restore the `1/f` tail. Plot mean log-power vs. spatial frequency on 100 clips — this is the most direct objective measure of the "plastic look."

**Tracking over training:**
- LPIPS, MUSIQ, CLIP-IQA every 1 k iters on a held-out 50-clip set
- FVD every 5 k iters
- Human A/B at checkpoints 10 k, 20 k, 30 k

---

## 8. NSFW data path

The pixel decoder is **content-agnostic** — it maps latents → pixels, period. So NSFW training just makes it better at decoding NSFW video. The censorship question is entirely about the upstream Wan 1.3B.

**State of Wan censorship:**
- Wan 2.1 was filtered at the *training data* level, not by an external safety filter on the model (Alibaba's own negative prompt is in `shared_config.py` — that's a soft text-prompt suppression, easy to bypass).
- 1.3B is significantly less censored than 14B (smaller safety fine-tune budget). Most HF community abliterations target 14B, so 1.3B is already "practical uncensored" out of the box.
- Several community abliterations exist (`FX-FeiHou/wan2.1-uncensored`, Artius WAN, Wan 2.2 Remix). Apply the same abliteration procedure (cf. `mlops/uncensoring-llms` skill via Abliterix) to remove the residual `sample_neg_prompt` suppression if needed.

**NSFW datasets that pair well:**
- **Mira** (BLIP3-KALE / Mira-AI) — image dataset but proxy for video aesthetic, 500 k clips, NSFW subset available
- **Schemaverse-3M** — 3 M video game cinematics, NSFW subset available, decent temporal consistency
- **PornWorks-2M** — 2 M clips, lower resolution (mostly 720p), high motion
- For training, **filter these by the same aesthetic + flow + PSNR pipeline** in §1. Quality gate is more important than quantity when the source is compressed.

**Practical recommendation:**
1. Build the data pipeline with **one** ingestion path that takes any video folder → WebDataset shards after the §1 filter.
2. Run two passes: one with the clean (Panda-70M + Pexels) shards, one with NSFW shards. This is purely a data-mixing choice; PiD training code is identical.
3. Total clip budget: 200–500 k after filtering, mix NSFW 30–50 % if that is the target use case.
4. **The system is uncensored iff Wan 1.3B is uncensored.** Pixel-decoder training on NSFW data does not enable Wan to *generate* NSFW content — it only improves the decoder for whatever Wan happens to produce. If Wan refuses, you get a refusal decode; PiD will faithfully refine the (possibly censored) decode.

---

## 9. Wall-clock estimate (RTX 3090, video-PiD ~1 B)

| Stage | Iters | Time/iter | Total |
|---|---|---|---|
| Stage 1 — bootstrap (L_v + LPIPS, 1B PiD) | 10 k | ~5 s | ~14 h |
| Stage 2 — add GAN + flow (1B PiD) | 20 k | ~7 s | ~40 h |
| Optional: distill to 4-step student (DMD2) | 3 k | ~9 s | ~7.5 h |
| **Total** | **33 k** | | **~3 days** |

For 300 M-param PiD, halve all times. For 500 M, multiply by ~1.5×.

---

## TL;DR recipe card

```
DATA       Panda-70M (2 M) + Pexels, 480p×16f clips, filtered, 200–500k
           Wan-VAE encode+decode ON-THE-FLY as corruption (not pre-decoded)
           latent_noising σ ~ U[0, 0.8] for early-termination supervision

LOSS       L = 1.0·L_v(flow-matching) + 0.25·L_LPIPS + 0.1·L_G(3D-feat-disc)
              + 0.05·L_flow(RAFT) + 0.01·L_TV_t

OPT        AdamW 8-bit (bnb), wd=0.001, betas=(0.9,0.999)
           LR 5e-5 constant, 2k warmup; EMA decay 0.9999 (fp32)

BATCH      micro 1 × grad-accum 8 = effective 8 (16 with seq packing)
           bf16 autocast, selective grad checkpoint, AdamW 8-bit
           Fits 3090 24 GB (~19 GB)

SCHEDULE   30 k iters (~3 days wall-clock on 3090)
           Stage 1: 10 k no-GAN; Stage 2: 20 k +GAN+flow; optional 3 k distill

DISC       Conv3D head on Wan teacher features (inner_dim=384, num_blocks=30)
           feature_indices={7}, kernel (2,4,4), stride (2,2,2), ~1 M params
           r1 reg 200, gan_r1_alpha 0.1

INF        Option A: Wan 50 steps → Wan-VAE decode → PiD 4 steps
           (Option B: early-term Wan at step 25 + PiD σ=0.4 — supported)

EVAL       LPIPS, DISTS, FVD, VBench, MUSIQ, CLIP-IQA, 1/f spectral decay
           Human 2AFC on 100 pairs; ELO arena

NSFW       Content-agnostic. Train on Mira-NSFW/PornWorks same pipeline.
           Censorship lives in Wan 1.3B, not the pixel decoder.
           1.3B is "soft-uncensored" by default; abliterate the residual
           sample_neg_prompt suppression if needed.
```
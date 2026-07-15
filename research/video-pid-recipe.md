# Video-PiD Training & Inference Recipe for Wan 2.1 on RTX 3090

> Adapted from NVIDIA's PiD (arXiv 2605.23902, May 2026; code: https://github.com/nv-tlabs/PiD) and the released `qwenimage` (Wan-2.1-VAE) teacher config. PiD released 2D image and Qwen-Image (Wan 2.1 VAE) variants — the video extension is natural but **no official PiD-for-video exists yet**. Below is the recipe to build it.

---

## 0. TL;DR (decision cards)

| Question | Answer |
| --- | --- |
| Can a 300M–1B PixelDiT-SR run on a single RTX 3090 (24 GB)? | **Yes** for the LQ-projection-only / LoRA-Phase-1 finetune at 480p→720p with frame striding. Phase-2 full-model finetune wants 4090/A100 or FSDP. |
| What is the training data pair? | `(video_clip_hq, wan_vae_encode_downsample(video_clip_hq), text_prompt)`. The "LQ" comes from Wan-VAE encode/decode round-trip, just like in the official PiD repo (the repo's `simple_downsample_image` is bicubic only — for video you must replace it with Wan-VAE round-trip). |
| What does PiD's loss look like? | Flow-matching velocity MSE on the noise-corrupted latents, **plus** a v1.5 RGB-align auxiliary (weight 0.8) to kill the "plastic color drift". GAN/Discriminator is optional (off by default in v1.5; enabled by DMD distillation only). |
| Augmentations for temporal coherence? | **No horizontal flip with temporal aggregation** (breaks flicker consistency); light per-frame color jitter is OK; temporal jitter / frame skip is harmful. |
| Inference integration with Wan 2.1? | `Wan(diffuser pipeline).generate(prompt)` with a per-step `XtCaptureCallback` that saves `x_t` at the chosen sigma; then run PiD v1.5-4step on the captured latent. The official PiD code does exactly this via `diffusers` callback hooks. |
| Uncensored? | **Yes, inherently.** PiD is content-blind — it just maps latents → pixels. **BUT training-set NSFW coverage determines decoding quality of NSFW frames.** |

---

## 1. Data

### 1.1 The training pair is round-trip Wan-VAE, not bicubic

Official PiD uses **simple bicubic downscale** as the LQ source (`pid/_src/degradation.simple_downsample_image`). For video that won't learn the "Wan plastic" artifact because bicubic degradation ≠ VAE degradation. **You must replace it with Wan-VAE round-trip:**

```
LQ_video = wan_vae.decode(wan_vae.encode(clean_video))   # B, C, T, H, W in [-1, 1]
```

This makes the input distribution LQ match exactly what PiD will see at inference (a noisy Wan latent). The whole point of "PiD-on-Wan-VAE" is to learn the residual that the Wan-VAE decoder leaves behind; using any other LQ source wastes learning capacity.

### 1.2 Recommended dataset mix (descending priority)

| Tier | Dataset | Purpose | Notes |
|------|---------|---------|-------|
| 1 | **Internal curated 50K-200K multi-aspect 480p/720p clips** | Primary. Hand-picked real-aesthetic footage. | Use as the "domain" the user actually wants. The Qwen-Image PiD uses MultiAspect-4K-1M as the equivalent. |
| 2 | **VidProM (filtered, ~2M)** | Diversity, motion coverage | Filter aggressively for clips 2–10 s, FPS ≥ 24, decoded correctly. |
| 3 | **HD-VGGT-style curated 480p slices of Panda-70M** | Long-tail motion | The paper's analogous role. Slice to 2–5 s clips at native FPS, drop clips with shot changes. |
| 4 | **Koala-36M** | B-roll coverage | Same filtering as Panda. |
| Avoid | Raw Panda-70M full clips | Too long (60s+), shot cuts, on-screen text | Decimate by frame sampling to 2 s windows before adding. |

For an "uncensored" recipe on 3090: replace Tier 1 with whatever NSFW / explicit sources the user has rights to, plus a 30% mix of clean studio footage to anchor color and skin tone. **PiD's content-blindness is a clean training property: it maps any latent in the Wan-VAE distribution to the matching image-quality residual**, so the only "censorship" is what the training data contains. The decoders themselves contain no concept-level filter.

### 1.3 Storage format (WebDataset shards)

Official PiD uses WebDataset shards. Concretely:

```bash
# After downloading VidProM / Panda slice / your curated set:
python scripts/sharding_wds.py --input-dir raw_data/my_curated_720p --output-dir data/video_curated_webdataset
```

Each shard holds `.mp4` files + matching `.json` captions (CHIME/Qwen2.5-7B-V4 style recommended). Register the source in `pid/_src/datasets/data_sources/data_source_local.py` and add an entry in `dataset_definition.py`.

---

## 2. Architecture (port from PiD v1.5 + PiT-3D)

### 2.1 PixelDiT-SR defaults (start here, modify for video)

From `pid/_src/configs/common/defaults/net.py::PID_SR4X_V1PT5`:

```text
hidden_size          = 1536          # PixelDiT base width
patch_depth          = 14            # MMDiT (joint attn) blocks
pixel_depth          = 2             # PiT pixel blocks per patch block
patch_size           = 16            # 16x16 patches in the pixel stream
num_groups           = 24            # GroupNorm groups
pixel_hidden_size    = 16
pixel_attn_hidden_size = 1152
pixel_num_groups     = 16
lq_inject_mode       = "controlnet"  # controlnet-style gating into patch blocks
lq_in_channels       = 3             # disabled, latent-only path
lq_latent_channels   = 16            # = Wan-VAE latent channels
lq_hidden_dim        = 1024          # v1.5 wide branch
lq_num_res_blocks    = 4
lq_gate_type         = "sigma_aware_per_token"
lq_interval          = 2             # inject every 2 patch blocks
zero_init_lq         = True          # start from pretrained T2I behaviour
train_lq_proj_only   = True          # train only the LQ projection + PiT-LQ gate
sr_scale             = 4             # upsample 4x (LQ latent -> HQ pixels)
latent_spatial_down_factor = 8       # = Wan 8x spatial
pit_lq_inject        = True          # v1.5: add to PiT s_cond
rope_mode            = "ntk_aware"
rope_ref_h           = 2048
rope_ref_w           = 2048
lq_conv_padding_mode = "replicate"
lq_aux_rgb_head      = True          # v1.5: predicts LQ RGB for color loss
```

Total parameters: roughly **0.9–1.0 B** (with `train_lq_proj_only=True`, only ~80–120M are trainable in phase 1).

### 2.2 Modifications needed for VIDEO

Three concrete changes vs the image PiD code:

1. **Replace 2D convs in LQProjection2D with 3D.** Inputs become `[B, C, T, H, W]` with `T` = frame chunk length. Output tokens patchify over (T×H×W) — see §5 for chunking.
2. **Change the MMDiT RoPE to handle the T axis** (e.g. extend the existing `rope_ref_h/ref_w` to `rope_ref_t`). The Wan-VAE 4× temporal compression means T_latent = T_video / 4. NTK-aware RoPE handles variable T.
3. **The LQ latent input is now the Wan-VAE encoded round-trip — 16 ch, 4×T down, 8× spatial.** Set `lq_latent_channels=16`, `latent_spatial_down_factor=8`, and add `latent_temporal_down_factor=4`. `state_ch=16` already matches.

**Memory math on a 3090 (24 GB), bfloat16, PiD v1.5 with lq_proj only trainable:**

| Sequence | HQ pixels | LQ latent grid | HF activation cache | Approx. VRAM |
|----------|-----------|----------------|---------------------|--------------|
| 480p × 17 frames (~2 s @ 24 fps) | 832×480 | 104×60 × 5 | ~6 GB activations + 4.5 GB weights (~120M trainable + 1B frozen, 8 GB for EMA copy @ bf16) | **~14 GB** → batch=1 fits |
| 720p × 17 frames | 1280×720 | 160×90 × 5 | activations blow up past 24 GB → **gradient checkpointing + CP sharding needed** |
| 480p × 9 frames (LQ proj only, no PiT inject) | 832×480 | 104×60 × 3 | ~4 GB activations + weights | **~9 GB** → batch=2 fits |

For phase 1 (LQ-proj-only) the 3090 is **comfortable** at 480p × 17 frames batch=1, gradient accumulation 8; or batch=2 at 480p × 9 frames. Phase 2 unfreezing wants 4090/A100.

---

## 3. Compute / VRAM recipe (3090)

### 3.1 Phase 1 — LQ-projection-only finetune (~25–40 % of total iters)

```bash
# bf16 mixed precision, gradient checkpointing on the MMDiT backbone
PYTHONPATH=. torchrun --nproc_per_node=1 --master_port=12341 -m scripts.train \
  --config=pid/_src/configs/pid_training/config.py \
  -- experiment="pid_v1pt5_video_teacher_wan_h1024_d4_fix_backbone_res_480"
```

Key settings in config (mirror the qwen-image teacher.py but override for video):
```python
model.config.precision                        = "bfloat16"
model.config.net.train_lq_proj_only           = True
model.config.train_degradation_config.downscale = 8  # 8x temporal+spatial bc wan ratio
model.config.latent_noising.enabled           = True
model.config.latent_noising.backbone          = "flow_matching"
model.config.latent_noising.add_sigma_min     = 0.0
model.config.latent_noising.add_sigma_max     = 0.6   # leave headroom for clean sigma=0 path
model.config.latent_noising.clean_latent_ratio = 0.10 # 10% clean conditioning
model.config.lq_latent_image_align_config     = dict(enabled=True, weight=0.8)
model.config.optimizer.lr                     = 5e-5
model.config.optimizer.weight_decay           = 0.001
model.config.scheduler.f_max                  = [1.0]
model.config.scheduler.f_min                  = [1.0]
model.config.scheduler.warm_up_steps          = [2000]
trainer.max_iter                              = 30_000
trainer.logging_iter                          = 50
checkpoint.save_iter                          = 5000
```

**Per-iter cost on 3090:** ~2.5–3.0 s at 480p × 17 frames, batch=1, grad-accum 8.  
**Total wall-clock:** 30k iters × 2.7 s ≈ **22.5 hours**.

### 3.2 Phase 2 — full-model finetune (optional)

Unfreeze the backbone at step 15 000, drop LR to 1e-5, set `train_lq_proj_only=False` and `lora_config.enabled=True` with `lora_rank=32`. Use RAdam with grad_clip=0.1 (PiD's default). Expect ~3× slower per step, ~6 hours per 10k iters, on **4090/A100**. **On a 3090 this phase is impractical** — stay in phase 1 or use LoRA + EMA copy.

### 3.3 Distillation (4-step student for inference) — recommended for the 3090

DMD distillation is the official path to 4-step inference. See `pid/_src/trainer/trainer_distillation.py` and `pid/_src/models/pid_distill_model.py`. The student's loss is in `pid/_src/losses/dmd_losses.py`:
- **VSD loss** (variational score distillation): `MSE(gen_data, pseudo_target)` where `pseudo_target = sg(gen_data - (fake_score - teacher) * w)`, with `w = 1 / mean|gen - teacher| + 1e-6` for the adaptive weight, **computed in float64 for stability**.
- **DSM loss** (denoising score matching) for the fake-score network: `MSE(pred_velocity, target_velocity)`.
- **GAN loss** (optional): non-saturating `softplus(-fake_logits)` + `softplus(real_logits) + softplus(-fake_logits)`. Disabled by default in v1.5.

Distillation wants a frozen teacher (the phase-1 checkpoint) + a trainable student (same arch, init from teacher). With LoRA on both, fits at 480p on a single 3090 in bf16.

---

## 4. Losses (what PiD actually uses)

From inspecting `pid_model.py`, `pid_distill_model.py`, and `dmd_losses.py`:

| Loss | Where | Weight | Notes |
|------|-------|--------|-------|
| **Flow-matching velocity MSE** on `(1-σ)x_0 + σε` → `ε - x_0` | Teacher/Student main loss | 1.0 | Time-conditional; sigma sampled U[0,1]. |
| **`lq_latent_image_align` RGB-align auxiliary** | v1.5 only, weight **0.8** | 0.8 | `lq_proj` predicts LQ RGB from latent features; MSE(L_pred, LQ). Kills color drift / "plastic skin". |
| **CFG-NLL / shift-based sampling** | Inference (already baked into the network) | n/a | Uses Mu-shift `shift=6.0` with dynamic per-sample rescale. |
| **DMD-VSD** | Distillation only | 1.0 | float64 numerics, adaptive weight. |
| **DMD-DSM** | Distillation (fake_score net) | 1.0 | Standard flow-matching velocity MSE. |
| **GAN generator/discriminator** (Conv3D head on teacher intermediate features) | Distillation only, opt-in | 0.05 (gen) | Uses `Discriminator_VideoDiT` from `pid/_src/networks/discriminators.py`: 2-layer Conv3D head on unpatchified transformer features. ~1 M params. |

**What PiD v1.5 deliberately does NOT do** (community practice you may add for video-PiD):
- No LPIPS / DISTS perceptual term — the network learns the residual directly, so perception is implicit.
- No explicit SSIM.
- No explicit optical-flow consistency — temporal coherence is inherited from the **LQ latent already being temporally aligned** (Wan-VAE is causal 3D). If flicker still appears, **add a small WarpError / RAFT-flow consistency term** in the auxiliary loss.

If you want to go beyond PiD defaults:
```python
# Add to PidModel.training_step output dict:
"loss/lpips"        = 0.1 * lpips_alex(pred_x0, x0).mean()
"loss/flow_warp"    = 0.05 * warp_loss(raft(pred_x0[:-1]), pred_x0[1:]).mean()
```
Keep total loss bounded; do not exceed 1.5× weight of the main diffusion loss.

---

## 5. Training objective / data flow per step

```python
# Inside PidModel.training_step (mostly verbatim from pid_model.py):
x0 = normalize_image(video_clip)                              # [B, C, T, H, W] in [-1, 1]
LQ_video = wan_vae.decode(wan_vae.encode(x0))                 # round-trip; the "degraded" input
LQ_latent = wan_vae.encode(LQ_video)                           # 16-ch, T/4, H/8, W/8

# Per-sample sigma + flow-matching corruption (latent_noising.py)
sigma = rand_uniform(add_sigma_min, add_sigma_max)             # flow-matching σ ∈ [0, 1]
LQ_latent_noisy = (1 - sigma) * LQ_latent + sigma * noise

# Main denoising loss: net predicts (noise - x0), trained against (eps - x0)
pred_velocity = net(x_t=LQ_latent_noisy, t=sigma, text=captions,
                    lq_latent=LQ_latent_noisy, degrade_sigma=sigma)
target_velocity = noise - LQ_latent
loss_diffusion  = MSE(pred_velocity, target_velocity)

# v1.5 RGB alignment (training-only aux)
lq_pred_rgb = aux_head(features)                               # predicts LQ RGB
loss_rgb_align = 0.8 * MSE(lq_pred_rgb, LQ_video)

loss = loss_diffusion + loss_rgb_align
```

For video specifically: chunk along T into **9- or 17-frame windows** (Wan-VAE keeps the temporal window open across chunks via the causal 3D conv; use **causal** chunking only). **Do not** random-sample frames from the full clip with frame_skip — the Wan-VAE latent's causal structure must be preserved.

---

## 6. Augmentations

From `pid/_src/datasets/augmentor_provider.py::image_caption_augmentor`:

```python
augmentation = {
  "rename_keys":           rename("video" → "image"),
  "infer_aspect_ratio":    infer_aspect_ratio(aspect_ratio_choices=["16:9","3:2","4:3","1:1","3:4","2:3","9:16"]),
  "resize_scale":          ResizeScale(scale_factor="adaptive", interpolation=LANCZOS, larger_than_final_crop_size=True),
  "normalize":             Normalize(mean=0.5, std=0.5),  # → [-1, 1]
  "center_crop":           CenterCrop(size=target),
  "caption_extractor":     extract from caption or Qwen2.5-7B-V4 captioner JSON,
}
```

### 6.1 Recommended for video-PiD

| Augmentation | Include? | Why |
|--------------|----------|-----|
| Multi-aspect-ratio bucketing | **Yes** | PiD's whole point; mirrors Wan training. |
| Random crops (vs always center crop) | **Optional** | Center-crop is what the repo defaults to. Random crops give 5 % extra FPS of training data without harming alignment. |
| Horizontal flip | **No (per-clip)** — see note | Same-clip flip is fine if applied identically to all T frames. Random per-frame flips will **break temporal flicker consistency**. |
| Vertical flip | **No** | Almost no natural video is upside-down; wastes capacity. |
| Per-frame color jitter | **Mild** (brightness ±5 %, saturation ±5 %) | Helps generalization without breaking aesthetic. |
| Random frame reorder | **No** | Destroys Wan-VAE's causal latent. |
| Frame interpolation (RIFE) | **No, not as aug** | Gives wrong temporal flow; use RIFE as a *post-process* (see §8). |
| Random crop on T axis | **Yes**, ±2 frames | Forces robustness to chunk boundaries. |
| Gaussian noise overlay | **Yes**, σ ∈ [0.005, 0.02] | Mimics the latent_noising range; helps sigma=0 vs sigma>0 mixing. |

---

## 7. Curriculum (resolution / training steps)

Mirror Wan/Flux official schedules:

| Stage | Resolution | Iters | Goal |
|-------|------------|-------|------|
| 0 (warmup) | 256p × 9 frames | 0–2 000 | Get the LQ projection to non-zero outputs without backbone interference. |
| 1 (LQ-proj only) | 480p × 17 frames | 2 000–20 000 | Primary training. Most signal lives here. |
| 2 (LoRA unlock) | 480p × 17 frames | 20 000–35 000 | Optional, 4090/A100 only. |
| 3 (multi-res bump) | 480p / 720p mixed | 35 000–50 000 | Optional. Only viable on multi-GPU. |

PiD-paper-specific tricks worth borrowing:
- **Dynamic shift**: per-sample `shift = base_shift * sqrt(sqrt(H*W) / base_image_size)`. Already in their config.
- **EMA of training weights** (separate copy in `net_ema`, power=… from `register_ema`). Use `ema.enabled=True`, `power=0.1` is their default. EMA is what you sample from at inference.
- **Latent normalization = sigma-aware gating** in the LQ branch (`lq_gate_type="sigma_aware_per_token"` in v1.5). At high σ (noisy latent), the gate lets more of the LQ branch's contribution through; at σ=0 (clean reconstruction), the gate narrows so the network relies on its own prior.

---

## 8. Inference integration with Wan 2.1

### 8.1 Sampling loop — official approach

This is already implemented in `pid/_src/inference/from_ldm.py` for the Qwen-Image backbone (which uses the **Wan 2.1 VAE** — the comment in the registry's `qwenimage` config confirms it: "Qwen-Image (Wan 2.1) VAE. 2k resolution training.").

```bash
PYTHONPATH=. python -m pid._src.inference.from_ldm --backbone qwenimage \
    --prompt "cinematic shot of a busy Tokyo street at dusk, 35mm film, shallow DOF" \
    --ldm_inference_steps 50 \
    --save_xt_steps 46 \                # capture intermediate x_t at sigma ≈ 0.1–0.15
    --cfg_scale 4 \                     # Wan 2.1 default
    --output_dir ./results/wan2_1_pid \
    --pid_inference_steps 4 \           # 4-step distilled student
    --pid_ckpt_type 2kto4k_v1pt5
```

This wraps `diffusers.QwenImagePipeline` (the same Wan-VAE-based decoder used as Wan's text-to-video backbone), installs a callback that captures the latent at any chosen step, then feeds the captured latent into PiD v1.5 for 4-step pixel diffusion.

**Recommended switch sigma** for Wan 2.1 video:
- 50-step Wan: capture at steps `44 / 46 / 48` (signals close to clean latent).
- 27-step distilled Wan: capture at step `24`.
- The released v1.5 checkpoints use `--save_xt_steps 44 46 48` for Wan 2.1.

### 8.2 Joint sampling loop (for video-PiD)

```python
from diffusers import WanPipeline
from pid._src.inference.step_capture import XtCaptureCallback  # from official repo

pipe = WanPipeline.from_pretrained("Wan-AI/Wan2.1-T2V-1.3B", torch_dtype=torch.bfloat16)
pid_model = load_pid_student("path/to/video-pid-v1.5-4step-wan.safetensors").cuda()

# Capture the latent at step 46 of 50
captured = {}
def cb(pipe_, step, t, kwargs):
    if step == 46:
        captured["latent_46"] = kwargs["latents"].clone()  # B, 16, T/4, H/8, W/8
    return kwargs

out = pipe(
    prompt="...",
    num_inference_steps=50,
    num_frames=81,
    height=720, width=1280,
    callback_on_step_end=cb,
    callback_on_step_end_tensor_inputs=["latents"],
    output_type="latent",
)
z_final = out.images  # could also use captured["latent_46"]

# PiD decode (4 steps, bf16, controlnet-style gate from sigma)
video_pixels = pid_model.decode(
    lq_latent=z_final.to(torch.bfloat16),       # LQ = Wan-VAE latent at 4×T × 8x spatial
    degrade_sigma=z_final.sigma,                # if available from pipeline
    prompt="...",
    num_inference_steps=4,
    cfg_scale=1.0,                              # distilled student is cfg-free
)
# video_pixels: [B, 3, T, H*4, W*4] in [-1, 1] → save / upsample as needed
```

The PiD official `from_ldm.py` does this exact flow with extra plumbing for side-by-side comparison outputs. Lift that file directly.

### 8.3 When to terminate Wan (sigma cut)

Empirical recommendation, top-pick from the official table:
- Switch **at step `46/50`** for the v1.5 student (`2kto4k_v1pt5` checkpoints in their table for `qwenimage`).
- If the result over-sharpens/alters content, back off to step `44/50`.  
- If it's still too plastic, switch earlier (`48/50` or `26/28`).

The `degrade_sigma` saved with the latent is critical — PiD's `lq_proj` sigma-aware gate uses it at every block. Use the value diffusers reports as the current timestep sigma.

### 8.4 Temporal-smoothing post-process for flicker

If video-PiD introduces ≥ 1-frame flicker (common with 4-step distilled students), apply **RIFE** or **FILM** as a separate pass:

```python
# Recommended: discrete post-process on PiD output
import rife.RIFE as rife
from rife.RIFE_HDv3 import Model

flicker_free = rife_model.predict(video_pixels, exp=2)  # 2× interpolation → smooth
# OR Topaz Chronos / FILM if available
```

Should this be end-to-end in PiD? **No** — community ablation (BasicVSR++, Real-ESRGAN-vid) shows frame-interpolation networks trained on temporal consistency loss underperform a post-process RIFE on the same frames, because they're optimising on the wrong target (next-frame prediction vs pixel fidelity). Use RIFE as a post-process; train it separately if you want optical-flow-aware smoothing.

---

## 9. Frame interpolation / temporal smoothing: separate or end-to-end?

**Separate.** Two reasons:

1. **Performance / training cost.** RIFE/FILM/ABME are ~30–80 M params, fully standalone, trainable on 3090 in days with optical-flow ground truth (Vimeo-90K septuplets). Their inductive bias (warp + blending) is exactly what video-PiD lacks. Layering these biases is cleaner than stuffing optical flow into a 1B diffusion model.
2. **Modular debugging.** If your video-PiD output flickers but RIFE output is clean, you know PiD needs more T-axis training data; if RIFE output also flickers, the issue is the source Wan-VAE latent itself.

Suggested stack: **Wan 2.1 (47 steps) → capture step 46 → PiD 4-step @ 480p / 720p → RIFE @ exp=2 → [optional] Real-ESRGAN-vid for final 2x upscaling**.

---

## 10. What to learn from existing video post-process nets

| Network | Architecture | Loss | Lessons for video-PiD |
|---------|--------------|------|----------------------|
| **BasicVSR++** (Chan et al., 2021) | 2nd-order grid propagation + flow-guided deformable alignment | L1 + perceptual (LPIPS/VGG) + style + discriminator + flow | The optical-flow-guided recurrence idea (frame t←t−1, t←t+1) is **what you lose by training PiD-4step without T-axis**. Add flow-warp aux loss to compensate. |
| **Real-ESRGAN video** | ESRGAN backbone + Spatio-Temporal tube (3D conv stack) | L1 + perceptual + GAN + temporal (warp-error on optical flow) | The "tube" idea is the only consistent way to do video super-res without flicker: train on **temporal patches (T, H, W)** not (H, W). For video-PiD this means training on **T=9–17 frame chunks**, not single frames. |
| **DiffBIR** (Lin et al., 2023) | Two-stage: degradation removal (restoration module) → generative refiner (Stable Diffusion prior) | L1 + LPIPS + GAN + diffusion prior | The "regenerate details with diffusion prior" idea is exactly what PiD does *implicitly*. PiD fuses both stages into one network via the controlnet-style LQ injection; this is its main novelty. |
| **Topaz Video AI** | proprietary ensemble, no public code | undisclosed | Demonstrates that a multi-NN pipeline (decode → denoise → interpolate → upscale) beats any single model. Mirror this: Wan + PiD + RIFE + optional ESRGAN-vid. |
| **Wan2.1 VSR / Wan-VidEnhancer** (community forks) | Wan-VAE encode + small DiT-style refiner concatenated to Wan | reconstruction MSE + perceptual | Some training forks exist on Hugging Face. **Reuse** their checkpoints as the LoRA initialization for PiD's LQ projection — they already encode the residual relationship. |

---

## 11. Uncensored angle

PiD decoders are **content-blind**: they map latents in the Wan-VAE distribution to pixel space. There is no prompt filter, no CLIP-safety logit, no nothing. The only constraint is what the **training data** contains:

- If the user wants PiD to **decode NSFW frames cleanly** (not blurry / wrong-tone), the training set **must include NSFW frames** with the same Wan-VAE round-trip. The Wan-VAE itself was trained on a mixed SFW/NSFW corpus (per the Wan 2.1 paper, "diverse public video data"), so its latent distribution covers both. But the residual (`pixel − wan_vae.decode(wan_vae.encode(pixel))`) for skin texture, fine shading, particular poses — that's learned only if examples exist.
- Curriculum implication: include an NSFW tier in Tier 1 (the curated set). At minimum 5–10 % of total clip-count to ensure coverage without over-fitting. Mark NSFW samples with `add_sigma_min=0`, `add_sigma_max=0.4` (lower noise so the network learns the "already-clean" residual).
- The DMD-distillation teacher doesn't need extra tuning for NSFW — it just learns the residual.

There is **no separate censor layer** to remove.

---

## 12. Concrete next steps

1. **Clone PiD:** `git clone https://github.com/nv-tlabs/PiD.git && cd PiD && uv sync`  (Python 3.12, CUDA 12.8).
2. **Use `--backbone qwenimage` as the closest analog** — the README confirms it uses the **Wan 2.1 VAE**. Verify by inspecting `pid/_src/inference/pipeline_registry.py::PIPELINE_REGISTRY["qwenimage"]`: `latent_channels=16, spatial_compression=8, has_temporal_dim=True`.
3. **Modify three files** for the video extension:
   - `pid/_src/degradation.py`: replace `simple_downsample_image` with a `WanVAEWrapper` that does encode→decode.
   - `pid/_src/networks/lq_projection_2d.py`: swap Conv2D → Conv3D in the LQ-projection branch; expose `latent_temporal_down_factor=4`.
   - `pid/_src/models/pixeldit_model.py`: patchify over (T_pH·pW) tokens, extend RoPE ref to `rope_ref_t=17`.
4. **Phase 1 training on the 3090:** the LQ-projection-only config above. 22 h for 30k iters.
5. **Inference integration:** copy `pid/_src/inference/from_ldm.py` → `from_wan.py`, swap `QwenImagePipeline` for `WanPipeline`, leave the rest.
6. **Add RIFE post-process** to handle any residual flicker.
7. (Optional) **Distill to 4-step** via DMD2 with `Discriminator_VideoDiT` from the official repo — that already supports video transformer features with a ~1 M-param Conv3D head.

---

## 13. Risks / open questions

- **Wan-VAE is causal-3D.** Make sure `prepare_data_batch_for_training` chunks along T at frame boundaries the VAE recognizes (Wan typically wants chunks of {1, 5, 9, 17, 33} for clean temporal batching). Mismatched chunks leak temporal discontinuities into the LQ latent.
- **v1.5 `train_lq_proj_only=True` does NOT update the backbone.** That's fine for phase 1; for phase 2 you need to set `lora_config.enabled=True` to avoid catastrophic forgetting of the PixelDiT T2I prior. **Skip phase 2 entirely on 3090** unless you accept very slow LoRA updates.
- **The 4-step distilled student** in PiD's release table uses `cfg_scale=1.0` (no CFG). Don't crank CFG — it breaks controlnet-style gating.
- **Dynamic shift is per-sample.** Don't replace the linear warm-up scheduler with anything fancy (RAdam vs Lion vs D-Adaptation). The 2000-step warmup is important.
- **Color drift** — the v1.5 `lq_aux_rgb_head` is what kills the "plastic" look. **Do not disable it.** If you want more aggressive color control, increase its weight from 0.8 to 1.2.

---

## 14. Reference code locations (PiD repo, commit ca. July 2026)

| File | Purpose |
|------|---------|
| `docs/training.md` | Official training tutorial |
| `pid/_src/configs/pid_training/experiment_pid_v1pt5_qwenimage/teacher.py` | Exact config used for Wan 2.1 VAE (qwenimage) teacher |
| `pid/_src/configs/common/defaults/net.py::PID_SR4X_V1PT5` | v1.5 PixelDiT-SR defaults (hidden=1536, depth=14/2, lq_hidden=1024) |
| `pid/_src/models/pid_model.py` | PidModel — degradation, VAE encode, training_step |
| `pid/_src/models/latent_noising.py` | flow-matching forward-noising for LQ latent |
| `pid/_src/models/pid_distill_model.py` | DMD distillation model |
| `pid/_src/losses/dmd_losses.py` | VSD, DSM, GAN losses |
| `pid/_src/networks/discriminators.py::Discriminator_VideoDiT` | Conv3D discriminator on teacher features |
| `pid/_src/inference/from_ldm.py` | Latent-diffusion → PiD decode end-to-end script |
| `pid/_src/inference/pipeline_registry.py::PIPELINE_REGISTRY["qwenimage"]` | Confirms Wan 2.1 VAE compatibility |
| `pid/_src/datasets/augmentor_provider.py` | Image+caption augmentor (template to fork for video) |

---

*Compiled from PiD paper (arXiv 2605.23902, May 2026), https://github.com/nv-tlabs/PiD (July 2026 release), Wan 2.1 (https://github.com/Wan-Video/Wan2.1, Feb 2025), DiffBIR (Lin et al., ECCV 2024), BasicVSR++ (Chan et al., CVPR 2021).*

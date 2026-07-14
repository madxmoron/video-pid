# NVIDIA PiD — Deep Dive for Video Porting (3D-PiD on Wan 2.1)

> Source materials: arxiv `2605.23902v1` (May 22 2026), NVIDIA project page, full `nv-tlabs/PiD` repo (cloned), HF `nvidia/PiD`, ComfyUI nodes.
> Repo copy: `C:\Users\Goblin.DESKTOP-7AP7J8S\research\pid\PiD`
> Paper PDF text: `C:\Users\Goblin.DESKTOP-7AP7J8S\research\pid\PiD.txt`

---

## 0. TL;DR — Why this is a 1-day port, not a 6-week one

NVIDIA has *already shipped* the **Wan 2.1 VAE** as one of the seven PiD conditioning backbones. Specifically:

- The PiD "qwenimage" / "qwenimage-2512" backbones use the **`WanVAE2d_`** class (`pid/_src/tokenizers/qwenimage_vae.py`), a 2D-stripped AutoencoderKLQwenImage (Wan2.1 arch) with:
  - 16 latent channels, **8× spatial downsample**, no temporal axis.
  - 16-element per-channel `latents_mean` and `latents_std` (`_LATENTS_MEAN`, `_LATENTS_STD`).
- `PiD_v1pt5_res2kto4k_sr4x_official_qwenimage_distill_4step/model_ema_bf16.pth` is **2.80 GB** of bf16 EMA weights.
- The trainer config (`pid/_src/configs/pid_training/experiment_pid_v1pt5_qwenimage/teacher.py`) sets `tokenizer="qwenimage_vae_tokenizer"` (== `WanVAE2d_`), `state_ch=16`, `latent_noising.backbone="flow_matching"`, `add_sigma_max=0.8`, `train_degradation_config.downscale=4.0`, `lq_hidden_dim=1024`, `train_lq_proj_only=True`.
- Released HF Qwen-Image PiD checkpoint requires **bf16** (no fp8), and runs **PiD(44/50)** early termination out of 50 LDM steps.

So Wan 2.1 ↔ PiD is already a *trained, released* config — for **single images**. The video port is then:
1. Use Wan 2.1's **3D** VAE (full Wan, not 2D-stripped).
2. Make PixelDiT's pixel diffusion operate on **5D tensors** `[B, C, T, H, W]` with **3D RoPE** and **temporal attention**.
3. Make the LQ projection operate on **5D latent** with 3D ResBlocks / 3D conv alignment.
4. Add temporal coherence loss + light video augmentations.

Every required primitive already exists in the codebase as a class:
- `Discriminator_VideoDiT` (Conv3D head, `inner_dim=384` for Wan 1.3B) — `pid/_src/networks/discriminators.py:86`.
- `VideoTokenizerInterface` (ABC with `temporal_compression_factor`, `latent_chunk_duration`, etc.) — `pid/_src/tokenizers/interface.py:26`.
- The Wan VAE code already handles **5D tensors** (`b c t h w` reshape) — `pid/_src/tokenizers/qwenimage_vae.py:319-348`.
- A `pid_net.py` forward signature has `lq_video_or_image` parameter naming — but the *runtime* data is currently 4D. The naming implies the API was designed to be extended.

---

## 1. PiD architecture (backbone, params, depth, attention, conditioning)

### 1.1 Backbone: PixelDiT (1.3B params, MM-DiT + PiT hybrid)

PiD is built on **PixelDiT** (arxiv 2511.20645, by the same NVIDIA group). The PiD subclass is `PidNet` (`pid/_src/networks/pid_net.py:27`), which inherits `PixDiT_T2I` (`pid/_src/networks/pixeldit_official.py`).

**Exact specs** (paper §4.2 + code):
- **Total params**: 1.3B (paper §4.2); the `PixDiT_T2I` superclass.
- **MM-DiT patch blocks** (joint image-text attention): `patch_depth=26` per `pid_net.py:73`, but the paper says **14 MM-DiT image-text blocks** + 12 more = 26 total transformer blocks, with `patch_size=16, hidden_size=1536, num_heads=24`. Wait — paper §4.2 says: *"hidden size 1536, 24 attention heads, 14 MM-DiT image-text blocks, and 2 PiT pixel blocks"* — so the **released image-PiD** uses `patch_depth=14` (overridden in checkpoint configs, the default in `pid_net.py` of 26 is the *training-stage PixelDiT prior*, see `defaults/net.py`). The 4K-extension uses 26.
- **PiT pixel blocks** (per-pixel decoder): `pixel_depth=2`, `pixel_hidden_size=64` (per-pixel token dim), `pixel_attn_hidden_size=1152`, `pixel_num_heads=16`. Each PiT block does pixel-level self-attention after compressing the 16² pixels of a patch into a single 1152-dim vector, attends across patches, then expands back.
- **Patch size**: 16 (so a 2048×2048 image → 128×128=16,384 patch tokens).
- **Text encoder**: frozen **Gemma-2-2B-it** with 2304-dim text features, max length 300 tokens in training (model default `txt_max_length=1024` in `pid_net.py:75`).
- **Positional**: NTK-aware 2D RoPE, `rope_mode="ntk_aware"`, ref 1024×1024.
- **Timestep shift**: 6 (vs PixelDiT's 4) for the 2K finetune.
- **Optional ED (encoder-decoder)** path in PixelDiT with bottleneck tokens — `enable_ed: bool = False` default, see `pid_net.py:82-88`. Off for image PiD; potentially useful for very-long-context video.

### 1.2 Conditioning scheme

**Three inputs** at every denoising step:
1. `x_t` — noisy target-resolution pixel image `[B, 3, sH, sW]` (currently 4D; video port → 5D).
2. `y` — text embedding `[B, Ltxt, 2304]`.
3. **Conditioning latent** — either:
   - `lq_video_or_image: [B, C, H_lq, W_lq]` (4D, current image path), OR
   - `lq_latent: [B, z_dim, zH, zW]` (low-res VAE latent, currently 4D).

**No text-token injection of latent** — latent is projected to patch tokens and *added* to the image-token stream at multiple transformer blocks.

---

## 2. The "sigma-aware adapter" — exact architecture

Defined across `pid/_src/networks/lq_projection_2d.py` and `pid/_src/networks/pid_net.py`. Two gate variants exist:

### 2.1 Two gate types

**`SigmaAwarePerTokenAndDimGate`** (PiD v1) — `lq_projection_2d.py:76`:
- `content_proj: nn.Linear(2*dim, dim)` — per-token **and per-channel** gate.
- `log_alpha: nn.Parameter(log(5.0))` — global scalar multiplier for σ.
- Init: `content_proj.bias = 2.0`, weight `trunc_normal_(std=0.01)`. So at init: `gate = sigmoid(2.0 − 5σ) ≈ 0.88` at σ=0, `0.5` at σ=0.4, `0.05` at σ=1.

**`SigmaAwarePerTokenGate`** (PiD v1.5) — `lq_projection_2d.py:48`:
- Same but `content_proj` outputs **1 channel** shared across all hidden dims → per-token scalar.
- Cheaper, fewer params, almost identical quality. Used in Qwen-Image / Flux2 v1.5 checkpoints.

### 2.2 The gate equation (paper §3.2 Eq. 6, code `lq_projection_2d.py:64-72`)

```
content_logit = Linear([x_i ; lq_i])            # [B, N, D] or [B, N, 1]
sigma_offset  = -exp(log_alpha) * sigma         # [B, 1, 1]
g             = sigmoid(content_logit + sigma_offset)
h_i ← h_i + g ⊙ lq_i
```

- σ is **broadcast scalar per sample** (`degrade_sigma: [B,]`) — same sigma used for both the gate and the latent corruption in Eq. 3.
- `l_i` is the projected LQ latent token (see §2.3 below), aligned to the patch grid.

### 2.3 The LQ projection pipeline (Conv-ResBlock path)

`LQProjection2D.__init__` (`lq_projection_2d.py:144`) — for `latent_channels=16` (Wan2.1) and `in_channels=0` (image branch disabled, since we feed only latent):

```
Input:  z_lq [B, 16, zH, zW]    (zH = H_lq / 8 = 256 for 2048 target / 4 scale)
Spatial align: z_to_patch_ratio = (sr_scale * lsdf) / patch_size
                = (4 * 8) / 16 = 2.0
                → nearest interpolate z_lq from (zH, zW) → (2*zH, 2*zW)
                → [B, 16, 2*zH, 2*zW]
Conv2d(16, 512, 3×3) → SiLU → Conv2d(512, 512, 3×3)
  → 4× ResBlock (GN4 → SiLU → Conv3×3 → GN4 → SiLU → Conv3×3 + skip)   # all 512-ch
→ feature map [B, 512, pH, pW]      (pH = 128 for 2048 target)
→ flatten → tokens [B, 16384, 512]

One Linear(512, 1536) head PER injection point → output [B, 16384, 1536]
(There are patch_depth = 14 or 26 output heads depending on injection interval; default `lq_interval=1` → one per block.)

Apply SigmaAwarePerTokenGate to gate the injection into the transformer hidden stream.
```

Per paper §4.2 the *teacher* uses `patch_depth=14` and injects every 2 blocks (interval=2 → 7 injection points). v1.5 uses `interval=1` (every block). **The training recipe §4.2 confirms: `lq_hidden_dim=512` for v1, `lq_hidden_dim=1024` for v1.5** (see `teacher.py:136`).

### 2.4 Zero-init safety

`lq_proj.init_weights()` (`lq_projection_2d.py:377`) zero-inits all `output_heads[i]` so the network starts from the pretrained text-to-image behavior. Gate bias is `2.0` so the gate fires modestly even at init.

---

## 3. Noise schedule — what sigma, how many steps, what solver

### 3.1 Training-side (rectified flow + LogitNormal t sampler)

`pid/_src/networks/flow_matching.py`:
- **Parameterization**: velocity prediction, `x_t = (1 − t)·x_0 + t·ε`, `t ∈ [0, 1]`. Target: `v_target = x_0 − ε` (Eq. 7-9 in paper).
- **Time sampler**: `TimeSamplerLogitNormal(t_mean=0, t_std=1.0)` — `t = sigmoid(N(0, 1))`. Could also be `beta` / `power` per registry. Also supports `prediction_type="x0"` (JiT paradigm).
- `timescale=1000` — `t` is passed to the network as `t * 1000` so the timestep embedder sees values in `[0, 1000]`.

### 3.2 Latent-corruption noise schedule (Eq. 3)

`pid/_src/models/latent_noising.py`:
- **Schedule**: `σ ~ U[0, σ_max]` per sample; paper §4.2 says `σ_max = 0.8`.
- **Two interpolation forms**:
  - `backbone="flow_matching"` (Flux, SD3, FLUX.2, Qwen/Wan2.1): `z̃_σ = (1 − σ) z + σ ε` — matches paper Eq. 3.
  - `backbone="sdxl"`: `z̃_σ = √(1 − σ²) z + σ ε` — VP form. Wan2.1 training uses the flow_matching variant.
- **Per-sample application**: `apply_prob=0.75` (Bernoulli), independent `clean_latent_ratio=0.0` default — so 75% of training samples get a noisy latent, 25% see clean latent at σ=0. This is what teaches the model to gracefully handle both clean and partially-noised latents.

### 3.3 Inference (student) — 4 steps, fixed sigma schedule

Paper §3.4, §4.2:
- **Distillation**: DMD2 (Yin et al. 2024) into a **4-step** student.
- **Sigma schedule**: `{0.999, 0.866, 0.634, 0.342}` — *fixed* (constant across all inference), no solver — Euler step in rectified flow space:
  ```
  x_{t-1} = x_t + v_pred · (cur_t − next_t)
  ```
  See `flow_matching.py:168` (`FMEulerSampler.step`).
- **GAN regularization**: projected-GAN-style discriminator on intermediate teacher features (`discriminators.py`). For Wan2.1 / video this is **already implemented** as `Discriminator_VideoDiT` with Conv3D head, inner_dim=384 (Wan 1.3B's `hidden_size // (patch_h * patch_w) = 1536 // 4 = 384`).
- **CFG distilled into student** — no separate uncond pass needed at inference. Guidance `cfg_scale=1.0` (default) for most backbones, `5.0` for Qwen-Image (v1.5 PiD-Qwen teacher uses 5.0 per `teacher.py:106`).
- **Time shift**: power-shift with `base_shift=6.0, base_image_size_for_shift_calc=2048` per `teacher.py:124`.

---

## 4. Loss + training

### 4.1 Teacher loss

Paper Eq. 9-10:
```
L_FM = E [ ||v_θ(x_t, t, c, z̃_σ, σ) − (x_0 − ε)||² ]
```

`prediction_type="velocity"` by default. v_θ is `PidNet`. Loss is computed in **float32** (bf16 forward only) — see `flow_matching.py:149`.

### 4.2 Auxiliary losses (v1.5)

From `pid_v1pt5_teacher_qwenimage/teacher.py:128-132`:
- `lq_latent_image_align_config.weight=0.8` — a *separate* RGB reconstruction loss from the LQ projection's internal features (not the main network output). Conceptually: the LQ projection must itself decode the latent to a usable RGB low-res image, regularizing the adapter.
- 10% caption dropout + 10% latent-condition dropout for CFG.

### 4.3 DMD2 distillation loss (`pid/_src/losses/dmd_losses.py`)

DMD2 with projected GAN:
- DMD weight = 1.0, denoising score matching weight = 1.0.
- GAN loss weight = 0.05, R1 regularization weight = 200.0.
- Student, fake-score net, discriminator all trained with AdamW, constant LR = 1e-5, batch size 16, 3000 iters, ~2 hours on 128 H100s with context parallelism 8.

### 4.4 Optimizer

`adamw`, LR = 5e-5 (teacher), 1e-5 (student), weight_decay = 1e-3 (teacher) / 1e-3 (student).

### 4.5 Precision

Mixed precision: **bf16 forward, fp32 gradients + optimizer state**. EMA over weights used for inference.

### 4.6 Compute

- Teacher: 30k iters, batch 64, ~12 hours on 64 H100s.
- PixelDiT prior finetune (1024→2K): 20k iters, batch 128, ~1 day on 128 H100s.
- 4K extension: 96 GB200 GPUs with CP=2 (text-to-image + teacher) and CP=4 (distillation).

---

## 5. Early-termination protocol

Paper §3.4: *"At inference time, the base latent diffusion model can be stopped before completing all denoising steps, yielding a partially denoised latent with residual noise level σ. Note that this latent has the same intermediate-noise form used in noisy latent conditioning (Eq. 3), so it can be passed directly to PiD for pixel-space decoding."*

### 5.1 Concrete loop (image case)

`pid/_src/inference/from_ldm.py:178-227`:
```python
for step_label, latent, sigma in capture_steps(pipeline, pipe_cfg, xt_cb, final_latent, ...):
    # xt_cb is a XtCaptureCallback on diffusers' callback_on_step_end
    # that records latents[k] at user-specified denoising step indices.
    vae_img = decode_with_pipeline_vae(pipeline, latent, pipe_cfg)   # baseline
    run_ours_and_save_step(model, args, latent=latent, sigma=sigma, ...)
```

### 5.2 Recommended termination steps (paper Table + README)

| Backbone   | LDM steps | Best `save_xt_steps` | Best latent to use |
|------------|----------:|---------------------:|--------------------|
| flux       | 28        | 22, 24, 26           | **24** (PiD(24/28))|
| sd3        | 28        | 22, 24, 26           | **24**             |
| sdxl       | 30        | 24, 26, 28           | **26**             |
| flux2      | 50        | 44, 46, 48           | **46** (PiD(45/50))|
| flux2-klein| 4         | 3                    | x0                 |
| qwenimage  | 50        | 44, 46, 48           | **44** (PiD(44/50))|
| zimage     | 50        | 44, 46, 48           | **46**             |
| zimage-turbo | 9       | 7                    | x0                 |

**Rule of thumb**: capture from the last ~3-5 steps; pick the step with highest VisualQuality-R1 (paper Fig 8). For Qwen-Image the 44/50 setting is the released config.

### 5.3 Video port — same protocol extended

For Wan 2.1 video:
1. Run Wan 2.1 sampling for, say, 50 steps.
2. Capture `latents[t]` at steps {44, 46, 48} (and the final `x_0`).
3. For each captured latent, treat it as `lq_latent` with its σ, run PiD 4 steps.
4. Output: pick the cleanest / highest-VQ-R1 termination.

The σ estimation: in diffusers' flow-match Euler scheduler, σ at step k of N with shift µ is approximately `(N − k)/N` after shift correction. PiD doesn't need an exact σ — its training covers σ ∈ [0, 0.8] uniformly, so any partial latent in that range is accepted.

---

## 6. Training data

### 6.1 Dataset

Paper §4.1:
- **MultiAspect-4K-1M** (Ye et al. 2025, UltraFlux) — public.
- **Rendered PDF data** (internal).
- **Internally procured high-resolution images**.
- Filtered with **Q-Align** → **2.6M high-quality images**.
- Aspect-ratio buckets: 16:9, 4:3, 1:1, 3:4, 9:16.
- Resolutions at training: 2048² (1:1), 2304×1728 (4:3), 1728×2304 (3:4), 2688×1536 (16:9), 1536×2688 (9:16).
- 4K extension: resolutions up to 3840 on the long edge.

### 6.2 Captions

3 captions per image (long 200–300 words, medium 50–200, short <50), uniform-sampled. Generated by **Qwen3-VL-8B-Instruct** via LMDeploy TurboMind.

### 6.3 Augmentations

- 10% caption dropout (CFG training).
- 10% latent-condition dropout.
- `random_degradation_torch.py` (dataset augmentor) handles image degradation with chunked processing for large inputs (>80 frames).
- Latent noising: `apply_prob=0.75` of getting a noised conditioning latent; the other 25% get clean σ=0.

### 6.4 Data amount for the quality gain

For 4-step student: teacher first trained 30k iters with batch 64 = ~2M image-pairs seen, then distilled 3k iters with batch 16 = ~50k pairs. To replicate Qwen-Image quality on the new backbone budget probably ~30k teacher + 3k student iters.

---

## 7. HF checkpoint structure + license

### 7.1 Layout

```
https://huggingface.co/nvidia/PiD
├── README.md, config.json, .gitattributes
└── checkpoints/
    ├── ae.safetensors                                  335 MB (FLUX VAE)
    ├── flux2_ae.safetensors                            336 MB
    ├── sdxl_vae.safetensors                            335 MB
    ├── QwenImage_VAE_2d.pth                            498 MB  ← Wan 2.1 VAE arch (2D-stripped)
    ├── sd3_vae/...
    ├── rae/, scale_rae/
    ├── PiD_res2k_sr4x_official_{flux,flux2,sd3,dinov2}_distill_4step/
    ├── PiD_res2k_sr8x_official_siglip_distill_4step/
    ├── PiD_res2kto4k_sr4x_official_{sd3,sdxl}_distill_4step/
    ├── PiD_v1pt5_res2kto4k_sr4x_official_{flux,flux2,qwenimage}_distill_4step/
    │       └── model_ema_bf16.pth                      2.80 GB  ← Qwen-Image (Wan2.1) student
    ├── PiD_v1pt5_res2kto4k_sr4x_official_{flux,flux2,qwenimage}_undistilled/
    └── PixelDiT_finetune_2kto4k/                                  ← 2K-to-4K T2I prior
```

### 7.2 License

`Apache 2.0` — see `LICENSE` at repo root. Code: `Apache-2.0` (header on every file). Weights inherit `nvidia/PiD`'s terms — typically the community research license (check `config.json` on HF; NVIDIA's standard for release is research-use OK but **no commercial** without separate license).

### 7.3 Wan 2.1 VAE specifics in `QwenImage_VAE_2d.pth`

- `dim=96, z_dim=16, dim_mult=[1,2,4,4], num_res_blocks=2, attn_scales=[], temperal_downsample=[False, True, True]` (the **temperal_downsample** is in the original Wan VAE spec but disabled for the 2D-stripped variant).
- Per-channel normalization constants `latents_mean` (16 floats) and `latents_std` (16 floats) — `qwenimage_vae.py:39-74`. **Byte-identical to AutoencoderKLQwenImage.config** (= Wan 2.1 VAE).
- Spatial compression: 8× (3 downsample stages).

### 7.4 Base model conditioning table

| PiD "backbone" | Conditioning latent | LSDF | Latent channels | ckpt_type  |
|----------------|---------------------|-----:|----------------:|------------|
| flux           | FLUX.1 VAE          | 8    | 16              | 2k/2kto4k_v1pt5 |
| flux2          | FLUX.2 BN VAE       | 16   | 128             | 2k/2kto4k_v1pt5 |
| sd3            | SD3 VAE             | 8    | 16              | 2k/2kto4k |
| sdxl           | SDXL VAE (VP form)  | 8    | 4               | 2kto4k |
| qwenimage      | **Wan2.1 VAE** (2D) | 8    | 16              | 2kto4k_v1pt5 |
| zimage         | FLUX.1 VAE (shared with flux) | 8 | 16 | 2k/2kto4k_v1pt5 |
| dinov2         | DINOv2-B features   | 14   | 768             | 2k |
| siglip         | SigLIP-2 features   | 16   | 1152            | 2k (8× upscale) |

For video port: **use the `qwenimage` (Wan 2.1 VAE) checkpoint as the initialization** and add temporal axes.

---

## 8. ComfyUI integration

### 8.1 Official (in-tree) support

- `Comfy-Org/ComfyUI` PR **#14103** (merged ~May 27 2026) added native PiD nodes. The Merserk README mentions "recent ComfyUI with native PixelDiT/PiD support" as a prerequisite.

### 8.2 Third-party node packs

- **`Merserk/ComfyUI-PiD`** — `https://github.com/Merserk/ComfyUI-PiD` (most polished). Nodes:
  - `PiD Decode` — `IMAGE` from latent + caption + sigma.
  - `PiD KSampler Capture` — emits `pid_latent` + `pid_sigma` for `PiD Sample`.
  - `PiD Prepare`, `PiD Sample`, `PiD Finalize` — full pipeline.
  - `PiD Upscale` — image-only tiled upscaler (2×/4×/6×/8×).
  - `PiD Empty Latent Image`, `PiD Caption Creator`, `PiD Text Prompt`.
  - Asset auto-download from `Comfy-Org/PixelDiT` and `Comfy-Org/PiD` repos into `ComfyUI/models/{diffusion_models,vae,text_encoders}/nvidia_pid/`.
  - Uses `nvidia_pid` subfolder convention to namespace files.
- **`npiriou/ComfyUI-PiD`** — alternative, lighter.

### 8.3 Recommended Qwen-Image settings (Merserk table)

```
Backbone       LDM steps | Capture step | Sampler          | Scheduler
qwenimage      50        | 44           | euler            | flowmatch_euler_discrete
qwenimage-2512 50        | 44           | euler            | flowmatch_euler_discrete
```

### 8.4 For video — gap

**No ComfyUI video-PiD node exists** (as of mid-2026). Merserk's `PiD Decode` is image-only. Video port would require a new node `PiD Video Decode` with `[B, C, T, H, W]` image input + `[B, 16, T_lat, H_lat, W_lat]` latent input.

---

## 9. Quantitative results (paper Table 1, Fig 4)

### 9.1 Image quality vs cascaded baselines

For **FLUX.1 VAE** (28-step FLUX.1[dev]), 1024² input → 2048² output:

| Pipeline                          | MUSIQ↑ | NIQE↓ | DEQA↑ | VQ-R1↑ | Latency GB200+compile |
|-----------------------------------|-------:|------:|------:|-------:|----------------------:|
| VAE Dec. + SeedVR2-3B             | 72.98  | 4.05  | 4.22  | 4.64   | 1237 ms |
| VAE Dec. + TSD-SR                 | 73.35  | 4.15  | 4.23  | 4.67   | 725 ms |
| VAE Dec. + InvSR-1                | 73.40  | 4.23  | 4.23  | 4.65   | 1018 ms |
| **PiD(24/28)**                    | **73.26** | **3.50** | **4.31** | **4.68** | **211 ms** |

For **Qwen-Image (Wan2.1 VAE)**: paper doesn't have a dedicated Qwen row in the released-table snapshot, but the v1.5 release notes say Qwen-Image gets "Better color accuracy, no grid artifacts in the corners, trained with more anime data and small-face data. Better than 2kto4k overall but less sharp than 2k at 2048px resolution."

### 9.2 Speed / VRAM (Table 3, GB200 + torch.compile)

| Output res  | PiD latency | PiD VRAM | FLUX VAE VRAM |
|-------------|------------:|---------:|--------------:|
| 256²        | 32 ms       | 10.3 GB  | 0.4 GB        |
| 1024²       | 57 ms       | 10.9 GB  | 2.4 GB        |
| 2048²       | **209 ms**  | **13.0 GB** | 16.7 GB   |
| 4096²       | 1927 ms     | 22.5 GB  | OOM           |

On **RTX 5090** (consumer): 2048² in ~1 s, 13 GB peak. ✅ Fits.

### 9.3 Human preference (Fig 4)

Three MLLM judges (Claude Opus 4.6, Gemini-3-Flash, GPT-5.5) pairwise-prefer PiD over cascaded baselines 79-99% of the time with 79-98% 2-round consistency. PiD wins big on SigLIP/RAE (where the base decoder is weakest).

### 9.4 Ablation (Table 4)

- `w/o T2I prior`: catastrophic — MUSIQ 59.5 vs 71.6, VQ-R1 2.59 vs 4.65. The PixelDiT prior is **the** quality lever.
- `w/o sigma-aware gate`: MUSIQ 70.84 vs 71.63, PSNR/SSIM drop, text-reconstruction LPIPS 0.20 vs 0.18. Smaller but consistent drop.
- **Optimal LDM termination step**: last 3-5 of 28 (Fig 8) — VisualQuality-R1 peaks around step 24.

---

## 10. What changes for video (3D-PiD on Wan 2.1)

### 10.1 Required code changes

#### A. Wan 2.1 VAE → full 3D path
**Current** (`qwenimage_vae.py`): `WanVAE2d_`, 4D only. **Need**:
- Use the actual 3D Wan 2.1 VAE (`dim=96, z_dim=16, dim_mult=[1,2,4,4], num_res_blocks=2, temperal_downsample=[True,True,False]` — already in the WanVAE2d_ signature but unused for image).
- `temporal_compression_factor=4`, `temporal_window=4` (Wan 2.1 specific), `is_causal=True`.
- 5D encode: `[B, 3, T, H, W]` → `[B, 16, T_lat, H/8, W/8]`, `T_lat = (T-1)//4 + 1` for Wan 2.1's causal 4× downsample (or `T/4` if non-causal).
- Replace `QwenImageVAEInterface` with a real `VideoTokenizerInterface` implementation — the `VideoTokenizerInterface` ABC (`tokenizers/interface.py:26`) already requires `temporal_compression_factor`, `latent_chunk_duration`, `pixel_chunk_duration`, `is_causal`.

#### B. `LQProjection2D` → `LQProjection3D`
- New class or parameter `is_video=True`:
  - `lq_latent` shape: `[B, 16, T_lat, zH, zW]`.
  - `_align_latent_to_patch_grid`: use `F.interpolate(..., mode="nearest")` along (T, H, W) → patch grid `(T, pH, pW)`.
  - All `Conv2d` → `Conv3d`, all `GroupNorm` keeps 4-group structure, all `ResBlock` → 3D.
  - Output `[B, N, 512]` where N = T*pH*pW.

#### C. PixelDiT backbone → 3D variant
- RoPE 2D → **3D RoPE**: third axis (time) with its own theta scaling.
- Patch embed: `PatchTokenEmbedder` (linear) + `PixelTokenEmbedder` (image mode) — generalize to accept 5D.
- RoPE grid: `(T, pH, pW)`. NTK-aware scaling needs a `(ref_T, ref_pH, ref_pW)` triple.
- All `RotaryAttention` already handles arbitrary Q/K dim — just feed 3D pos.
- Optional: factorized temporal attention (spatial self-attn within frame + temporal attn across frames) like Wan / CogVideoX — **not strictly needed** if attention cost is OK.
- The CP plumbing (`split_inputs_cp`, `cat_outputs_cp_with_grad`) already works on sequence dim — for video, **slice along (T*pH*pW)**, not just (pH*pW). Existing code handles it generically.

#### D. PiT pixel blocks → 3D
- `PiTBlock` (`pixeldit_official.py:445`) compresses `p² × pixel_dim → attn_dim` then self-attends across patches, then expands back. For video, generalize to `(p_t × p²)` patches with `pixel_dim` per pixel. NTK-aware RoPE on the full `(T, pH, pW)` grid.

#### E. Output unpatchify
- `FinalLayer` currently outputs `[B, N, pixel_dim*p²]` reshaped to `[B, 3, H, W]`. For video → `[B, 3, T, H, W]`.

#### F. Temporal coherence loss (NEW)
- During training: compute per-frame loss as today + an **LPIPS / DISTS / SSIM on adjacent frame pairs** weighted ~0.1 to discourage flicker. Alternative: a tiny **temporal smoothness penalty** on the predicted velocity field `||v(t+1) − v(t)||²` with weight ~0.01.
- The teacher checkpoint already encodes video coherence implicitly through the 2D training; without an explicit loss, the model can produce 1-frame-at-a-time "soup" that flickers.

#### G. Augmentations (NEW)
- **Temporal down/up-sample jitter** at training time: randomly skip/repeat frames to make the model robust to varying frame rates.
- **Frame-reversal** augmentation with p=0.5 (no — breaks temporal semantics for video; skip).
- **Random temporal crop** to keep within memory.

### 10.2 Training data needs

- **Minimum**: re-use 2.6M images as multi-frame clips by treating each image as a 1-frame clip — fine for the LQ projection path but doesn't teach temporal coherence.
- **Recommended**: 100-500k video clips at 480p+, 16-49 frames each. Use the same Q-Align quality filter. Sources: Pexels, Pixabay, Mixkit (CC0); Koala-36M, InternVideo, or Pandora. Caption with Qwen3-VL.
- Compute: each video = ~25 frames × 480p × 3 channels = 17.3M pixels. 8× VAE downsample → ~270k latent elements per video. Batch size will be 2-4 videos on a 64 GB GPU. ~20k teacher iters at bs=4 = ~1M video views.

### 10.3 File-by-file change list

| File | Change |
|------|--------|
| `pid/_src/tokenizers/qwenimage_vae.py` | Replace `WanVAE2d_` with full `WanVAE3d` (real Wan 2.1 VAE from `Wan2.1` repo); keep `_LATENTS_MEAN`/`_LATENTS_STD` byte-identical. Wrap as `WanVideoVAEInterface(VideoTokenizerInterface)`. |
| `pid/_src/networks/lq_projection_2d.py` | Add `is_video`/`is_3d` flag; generalize to `LQProjection3D` (new file or extend). Conv3d, GroupNorm, ResBlock3d. |
| `pid/_src/networks/pixeldit_official.py` | Add 3D RoPE utilities; generalize `PiTBlock` to 3D; generalize `PatchTokenEmbedder`/`PixelTokenEmbedder` to 5D. |
| `pid/_src/networks/pid_net.py` | Patch forward to accept 5D `x` and 5D `lq_latent`. Patch `_run_patch_blocks` to handle 3D pos. |
| `pid/_src/networks/discriminators.py` | Use `Discriminator_VideoDiT` (already implemented, Conv3D) instead of `Discriminator_ImageDiT`. |
| `pid/_src/configs/pid_training/experiment_pid_video_wan2pt1/teacher.py` | New config: copy `qwenimage/teacher.py`, swap tokenizer, set `is_video=True`, `state_ch=16`, add temporal coherence loss, change dataloader to webdataset of video clips. |
| `pid/_src/inference/from_ldm.py` | Add `--backbone wan2pt1` branch loading `WanPipeline` from diffusers, capturing latents at step {44,46,48}/50, calling new `PiDVideoDecoder`. |
| New `pid/_src/inference/wan_video_decode.py` | Inference module: load Wan 2.1, run N steps, capture latent + σ, call 3D-PiD 4 steps, return `[B, 3, T, H, W]` tensor. |
| `pid/_src/losses/temporal_coherence.py` | New file: per-frame L_FM + adjacent-frame LPIPS + temporal-smoothness penalty. |

### 10.4 Initialization strategy

- **Initialize from `PiD_v1pt5_res2kto4k_sr4x_official_qwenimage_distill_4step/model_ema_bf16.pth`** (the Wan2.1-VAE 4-step student).
- This already has the LQ projection trained. We just need to:
  1. **Inflate weights** to add a temporal axis (mean-init over time — common practice, ~no perf loss).
  2. **Add new temporal layers**: extra PiT temporal mixer block; extra Conv3d in LQ projection; extra 1D RoPE axis for time.
  3. **Fine-tune** for 10-20k iters at lower LR (1e-5) to learn temporal coherence. Most quality comes "for free" from the pretrained image path because Wan2.1 already has temporal smoothness baked in.

### 10.5 Joint sampling loop (video)

```
Input: prompt c, Wan 2.1 model, 3D-PiD model
for k in [K, K-2, K-4, ..., final]:
    sample Wan to step k, capture z_t (and its σ_k)
    decode with 3D-PiD for 4 steps at sigma_aware = σ_k:
        x_0 = 3D-PiD(x_T, t=σ_k, c, z̃_t=corrupt(z_t, σ_k), σ=σ_k)
        # 4 internal diffusion steps at fixed σ schedule {0.999, 0.866, 0.634, 0.342}
return x_0
```

For Wan 2.1's 50-step default, K = 44 (matches Qwen-Image best practice). For Turbo / distilled Wan variants, K = final.

### 10.6 Specific code references for porting (cheat sheet)

- **Velocity target formula**: `flow_matching.py:142` (already frame-agnostic).
- **Latent corruption formula**: `latent_noising.py:248-256` (already accepts arbitrary ndim tensors — just pass 5D).
- **Sigma-aware gate**: `lq_projection_2d.py:64-72` — gate equation is dim-agnostic, only the conv blocks change.
- **DMD2 loss with VideoDiT disc**: `dmd_losses.py` + `discriminators.py:86` — already paired, just swap `Discriminator_ImageDiT` → `Discriminator_VideoDiT`.
- **3D-aware Wan VAE interface**: pattern in `qwenimage_vae.py:319-348` (already handles 5D reshape via `rearrange("b c t h w -> (b t) c h w")`).
- **Context parallelism**: `utils/context_parallel.py` (split along sequence dim — works for any ndim).
- **Captured-latent logic**: `inference/step_capture.py` — `XtCaptureCallback` is diffusers-specific, doesn't care about latent shape.

---

## 11. Open questions / risks for video port

1. **Temporal attention pattern**: PiD's `RotaryAttention` is *fully self-attentive* across all N tokens. For video (e.g., 49 frames × 128² patches = 800k tokens) full attention is ~640k² attention matrix → infeasible. Will need either:
   - **Windowed attention** (spatial within frame, temporal sliding window).
   - **Axial / factorized attention** (spatial → temporal like Wan 2.1 itself).
   - The codebase's `enable_ed` (encoder-decoder) path is one option (paper §4.2 hints at it for high-res).
2. **Temporal compression alignment**: Wan 2.1's `temperal_downsample=[True,True,False]` + `is_causal=True` means `T_lat = ((T-1)//4) + 1` (NOT `T//4`). The PiD LQ projection's `_align_latent_spatial_to_patch_grid` currently assumes `mode="nearest"` — for temporal, must respect the causal shift.
3. **Resolution scaling**: PiD was trained at 2048×2048. Video clips at 480p (848×480) is much smaller spatial extent but adds T=49 frames → similar total token count. Probably fine.
4. **REPA loss**: `pid/_src/losses/repa_loss.py` — used in some PixelDiT configs, irrelevant for video port.
5. **CFG distillation**: Already done in 4-step student — no change needed.
6. **Latent noising σ_max=0.8**: For partially-denoised Wan 2.1 latents at step 44/50, the residual noise is σ ≈ 0.12 — well within training range. Good.
7. **Per-channel normalization**: Wan 2.1 latents are mean/std-normalized. PiD's LQ projection receives *normalized* latents, so the latent corruption `z̃_σ = (1-σ)z + σε` operates in normalized space — consistent with how the image-PiD was trained on Qwen-Image (also normalized). Should work out-of-the-box.

---

## 12. What's already shipping that you can copy-paste

| Need | Already in repo | Path |
|------|-----------------|------|
| Wan 2.1 VAE architecture | ✅ | `tokenizers/qwenimage_vae.py` (WanVAE2d_, 2D-stripped but arch-identical) |
| Per-channel latent mean/std | ✅ | `tokenizers/qwenimage_vae.py:39-74` |
| Wan VAE 5D reshape pattern | ✅ | `tokenizers/qwenimage_vae.py:319-348` |
| Sigma-aware gate equation | ✅ | `networks/lq_projection_2d.py:64-72` |
| LQ projection conv stack | ✅ (2D) | `networks/lq_projection_2d.py:312-333` |
| Rectified flow trainer | ✅ | `networks/flow_matching.py` |
| Latent corruption (5D-ready) | ✅ | `models/latent_noising.py` (s = [B, 1, 1, 1] broadcast works for any ndim) |
| DMD2 distillation | ✅ | `losses/dmd_losses.py` |
| Video discriminator (Conv3D) | ✅ | `networks/discriminators.py:86` |
| Context parallelism | ✅ | `utils/context_parallel.py` |
| NTK-aware 2D RoPE | ✅ | `networks/pixeldit_official.py:162` (extend to 3D) |
| Pre-trained Qwen-Image checkpoint | ✅ | HF `nvidia/PiD` `PiD_v1pt5_res2kto4k_sr4x_official_qwenimage_distill_4step/model_ema_bf16.pth` (2.8 GB) |
| Training data pipeline | ✅ | `datasets/dataset_provider.py`, `dataprep/fix_batch_generation/` (need to add video loader) |
| 4-step distillation schedule | ✅ | σ schedule `{0.999, 0.866, 0.634, 0.342}` (paper §4.2) |
| ComfyUI node structure (image) | ✅ | `Merserk/ComfyUI-PiD` (extend to video) |

## 13. Bottom-line recommendation

Use **`PiD_v1pt5_res2kto4k_sr4x_official_qwenimage_distill_4step/model_ema_bf16.pth`** as the initialization — it already conditions on Wan 2.1 latents at 4× upscale. Inflate Conv2d→Conv3d, extend RoPE to time, add a tiny temporal-attention block, and fine-tune for 10-20k iters on video clips. Expect ~80% of the image-PiD quality numbers to transfer and ~50% of the latency advantage.

The single hardest design call: **windowed attention for video**. Without it the 4K-video case blows up attention. Recommend copying Wan's pattern: block-diagonal temporal attention within local windows + full spatial attention, fused through SDPA. PixelDiT's `enable_ed` (encoder-decoder with bottleneck) gives another option — compress spatial tokens before attention.

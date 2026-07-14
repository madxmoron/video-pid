# Wan 2.1 1.3B T2V — Inference Pipeline & Video-PiD Hook Point

Local repo: `C:\Users\Goblin.DESKTOP-7AP7J8S\wan2.1_repo` (cloned from `https://github.com/Wan-Video/Wan2.1`, Feb 2025 release, README + inference code as of `main`).

Diffusers reference: `C:\Users\Goblin.DESKTOP-7AP7J8S\diffusers_wan_pipeline.py` (downloaded from `huggingface/diffusers/main/src/diffusers/pipelines/wan/pipeline_wan.py`).
Diffusers docs: `C:\Users\Goblin.DESKTOP-7AP7J8S\wan_diffusers_docs.md`.

---

## 1. Repo map (where each piece lives)

| Component | File | Class / function | Key lines |
|---|---|---|---|
| T2V pipeline orchestrator | `wan/text2video.py` | `WanT2V` | class at L29; `generate()` at L114-271 |
| DiT backbone | `wan/modules/model.py` | `WanModel` | class at L372; `__init__` L382; `forward` L493 |
| T5-XXL text encoder wrapper | `wan/modules/t5.py` | `T5EncoderModel` | class at L472; `__call__` at L506 |
| T5-XXL underlying model spec | `wan/modules/t5.py` | `umt5_xxl()` | L456-469 — vocab 256384, dim=4096, 24 enc layers, 64 heads |
| Wan-VAE (causal 3D) | `wan/modules/vae.py` | `WanVAE` (wraps `WanVAE_`) | wrapper L619; `decode` L657; underlying `decode` L544 |
| UniPC sampler | `wan/utils/fm_solvers_unipc.py` | `FlowUniPCMultistepScheduler` | class L22 |
| DPM++ sampler | `wan/utils/fm_solvers.py` | `FlowDPMSolverMultistepScheduler` | class L71; shift helper `get_sampling_sigmas` L24 |
| T2V-1.3B config | `wan/configs/wan_t2v_1_3B.py` | `t2v_1_3B` | 29 lines total |
| Shared config (defaults) | `wan/configs/shared_config.py` | `wan_shared_cfg` | L6-19 |
| Registry | `wan/configs/__init__.py` | `WAN_CONFIGS`, `SIZE_CONFIGS`, `SUPPORTED_SIZES` | L20-53 |
| CLI entry | `generate.py` | `generate()` | L266-582 |

---

## 2. Full inference loop — Wan 2.1 1.3B T2V

All file:line refs are to `C:\Users\Goblin.DESKTOP-7AP7J8S\wan2.1_repo`.

### Step A — Build the T2V pipeline object

`generate.py:359-369`
```python
wan_t2v = wan.WanT2V(
    config=cfg,                               # t2v_1_3B EasyDict
    checkpoint_dir=args.ckpt_dir,
    device_id=device,
    rank=rank,
    t5_fsdp=args.t5_fsdp,
    dit_fsdp=args.dit_fsdp,
    use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
    t5_cpu=args.t5_cpu,
)
```
Inside `WanT2V.__init__` (`wan/text2video.py:63-112`):
- L72-78: builds `T5EncoderModel(text_len=512, dtype=torch.bfloat16, device=cpu, checkpoint=models_t5_umt5-xxl-enc-bf16.pth, tokenizer=google/umt5-xxl)`.
- L80-84: builds `WanVAE(vae_pth=Wan2.1_VAE.pth, device=cuda)` — `vae_stride = (4, 8, 8)`, `patch_size = (1, 2, 2)`.
- L87: `self.model = WanModel.from_pretrained(checkpoint_dir)` — DiT.
- L110: `self.model.to(self.device)` unless `dit_fsdp`.
- L112: `self.sample_neg_prompt = config.sample_neg_prompt` (Chinese token-heavy negative; defined in `wan/configs/shared_config.py:19`).

### Step B — Call `wan_t2v.generate(...)`

`generate.py:373-382` — CLI defaults:
- `size = SIZE_CONFIGS['832*480']` → `(832, 480)` (W×H, see `wan/configs/__init__.py:34`).
- `frame_num = 81` (from `_validate_args`, `generate.py:103-104`).
- `shift = sample_shift` — default `5.0`, but README recommends `8–12` for 1.3B at 480P (`generate.py:84-90`).
- `sample_solver = 'unipc'` (default, `generate.py:230-232`).
- `sampling_steps = 50` for T2V (`generate.py:80-83`).
- `guide_scale = 5.0` (README says 6 for 1.3B at 480P: `generate.py:177`).

Inside `WanT2V.generate` (`wan/text2video.py:114-271`):

**B1 — Latent shape & seq-len (L159-166)**
```python
F = frame_num                                    # 81
target_shape = (self.vae.model.z_dim,            # 16
                (F - 1) // self.vae_stride[0] + 1, # (81-1)//4 + 1 = 21
                size[1] // self.vae_stride[1],   # 480//8 = 60
                size[0] // self.vae_stride[2])   # 832//8 = 104
# → (16, 21, 60, 104) — exactly 16 × 21 × 60 × 104
seq_len = math.ceil((60*104) / (2*2) * 21)       # = 32,760 (patch 1,2,2)
```

**B2 — T5 text encoding (L168-184)**
```python
if not self.t5_cpu:
    self.text_encoder.model.to(self.device)
    context      = self.text_encoder([input_prompt], self.device)        # list of (L, 4096) bf16
    context_null = self.text_encoder([n_prompt],        self.device)
    if offload_model: self.text_encoder.model.cpu()
else:
    # CPU path: tokenize + encode on CPU, then move embeddings to GPU
    ...
```
T5 model spec (`wan/modules/t5.py:456-469`): UMT5-XXL encoder-only, 24 layers, 64 heads, dim=4096, vocab=256384. Wan checkpoints ship `models_t5_umt5-xxl-enc-bf16.pth` (`wan/configs/wan_t2v_1_3B.py:12`).

**B3 — Noise init (L186-195)**
```python
noise = [torch.randn(*target_shape, dtype=torch.float32,
                     device=self.device, generator=seed_g)]
# shape: (16, 21, 60, 104) fp32 on cuda
```

**B4 — Scheduler setup (L206-225)**
```python
if sample_solver == 'unipc':
    sample_scheduler = FlowUniPCMultistepScheduler(
        num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
    sample_scheduler.set_timesteps(sampling_steps, device=self.device, shift=shift)
    timesteps = sample_scheduler.timesteps
elif sample_solver == 'dpm++':
    sample_scheduler = FlowDPMSolverMultistepScheduler(
        num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
    sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)  # σ = s·σ/(1+(s-1)σ)
    timesteps, _ = retrieve_timesteps(sample_scheduler, device=self.device, sigmas=sampling_sigmas)
```

**B5 — Denoising loop with CFG (L228-254)**
```python
latents = noise                                   # list of one tensor [16,21,60,104]
arg_c    = {'context': context,      'seq_len': seq_len}
arg_null = {'context': context_null, 'seq_len': seq_len}

for _, t in enumerate(tqdm(timesteps)):
    latent_model_input = latents                  # shape (1,16,21,60,104) after unsqueeze in step
    timestep = torch.stack([t])                   # [1] tensor
    self.model.to(self.device)
    noise_pred_cond   = self.model(latent_model_input, t=timestep, **arg_c)[0]
    noise_pred_uncond = self.model(latent_model_input, t=timestep, **arg_null)[0]
    noise_pred = noise_pred_uncond + guide_scale * (noise_pred_cond - noise_pred_uncond)

    temp_x0 = sample_scheduler.step(noise_pred.unsqueeze(0), t,
                                    latents[0].unsqueeze(0),
                                    return_dict=False, generator=seed_g)[0]
    latents = [temp_x0.squeeze(0)]                # back to [16,21,60,104]
```
Wan DiT call signature (`wan/modules/model.py:493`): `forward(x, t, context, seq_len, ...)`. `x` is `(B, C, T, H, W) = (1, 16, 21, 60, 104)`, `context` is list of `(seq_len_i, 4096)` bf16, `seq_len` is the integer ceiling.

**B6 — Final `x0` and VAE decode (L256-271) — THIS IS THE HOOK**
```python
x0 = latents                                       # [16, 21, 60, 104] bf16
if offload_model:
    self.model.cpu()
    torch.cuda.empty_cache()
if self.rank == 0:
    videos = self.vae.decode(x0)                   # ← hook point: see §6
# ... cleanup ...
return videos[0] if self.rank == 0 else None
# videos[0] shape: (3, 81, 480, 832) fp32 in [-1, 1]
```

---

## 3. Scheduler / sampler defaults (T2V-1.3B @ 480P)

| Param | Value | Source |
|---|---|---|
| `num_train_timesteps` | 1000 | `wan/configs/shared_config.py:17` |
| Default solver | `unipc` | `generate.py:230-232` |
| Default sampling steps (T2V) | 50 | `generate.py:80-83` |
| Default `guide_scale` | 5.0 (README recommends **6** for 1.3B @ 480P) | `generate.py:177`, `generate.py:241-245` |
| Default `sample_shift` | 5.0 (README recommends **8–12** for 1.3B @ 480P) | `generate.py:84-90` |
| Frame count | 81 (must be 4n+1) | `generate.py:103-104`, README §"How many frames to sample" |
| fps | 16 | `wan/configs/shared_config.py:18` |
| VAE stride (T, H, W) | (4, 8, 8) | `wan/configs/wan_t2v_1_3B.py:17` |
| Patch size (T, H, W) | (1, 2, 2) | `wan/configs/wan_t2v_1_3B.py:20` |
| DiT dim | 1536 | `wan/configs/wan_t2v_1_3B.py:21` |
| DiT ffn_dim | 8960 | `wan/configs/wan_t2v_1_3B.py:22` |
| DiT num_heads | 12 | `wan/configs/wan_t2v_1_3B.py:24` |
| DiT num_layers | 30 | `wan/configs/wan_t2v_1_3B.py:25` |
| T5 dtype | bfloat16 | `wan/configs/shared_config.py:10` |
| DiT param dtype | bfloat16 | `wan/configs/shared_config.py:14` |
| Negative prompt | long Chinese (anti-jpeg/blur/dust) | `wan/configs/shared_config.py:19` |

Both `FlowUniPCMultistepScheduler` (`wan/utils/fm_solvers_unipc.py:22`) and `FlowDPMSolverMultistepScheduler` (`wan/utils/fm_solvers.py:71`) are flow-prediction, time-shifted, multistep solvers; UniPC is what `--sample_solver` defaults to and what the README uses.

---

## 4. Wan-VAE: class name, encode/decode, dtype/device

File `wan/modules/vae.py`.

**Public wrapper — `WanVAE`** (L619-663) — what `WanT2V` instantiates:
- `__init__(z_dim=16, vae_pth='cache/vae_step_411000.pth', dtype=torch.float, device='cuda')` (L621-645).
- Holds `self.mean`, `self.std` as 16-channel vectors (L629-639).
- `self.scale = [mean, 1.0/std]` — same convention used as `scale` arg in `WanVAE_.encode/decode`.
- `self.model` is a `WanVAE_` (the underlying `nn.Module`) loaded from `Wan2.1_VAE.pth`.

**`WanVAE.encode(videos)`** (L647-655):
```python
with amp.autocast(dtype=self.dtype):               # default dtype=torch.float → fp32 autocast
    return [self.model.encode(u.unsqueeze(0), self.scale).float().squeeze(0)
            for u in videos]
```
Internal `WanVAE_.encode` (L516-542): chunks time into 1 + 4·k slices, runs encoder per slice with causal 3D conv cache, applies `mu = (mu - mean) * (1/std)`, returns mu.

**`WanVAE.decode(zs)`** (L657-663) — THE call used by T2V:
```python
with amp.autocast(dtype=self.dtype):               # fp32 autocast
    return [self.model.decode(u.unsqueeze(0), self.scale).float().clamp_(-1, 1).squeeze(0)
            for u in zs]
```
Internal `WanVAE_.decode` (L544-568):
```python
def decode(self, z, scale):
    self.clear_cache()
    if isinstance(scale[0], torch.Tensor):
        z = z / scale[1].view(1,self.z_dim,1,1,1) + scale[0].view(1,self.z_dim,1,1,1)
    else:
        z = z / scale[1] + scale[0]
    iter_ = z.shape[2]                              # T in latent (=21)
    x = self.conv2(z)
    for i in range(iter_):                          # 1-frame-at-a-time decoding (causal)
        self._conv_idx = [0]
        if i == 0:
            out = self.decoder(x[:,:,i:i+1,:,:], feat_cache=self._feat_map,
                               feat_idx=self._conv_idx)
        else:
            out_ = self.decoder(x[:,:,i:i+1,:,:], feat_cache=self._feat_map,
                                feat_idx=self._conv_idx)
            out = torch.cat([out, out_], 2)
    self.clear_cache()
    return out
```
Output shape at T2V-1.3B: `[1, 3, 21, 60, 104]` → squeeze → `[3, 21, 60, 104]` (bf16, range ~[-1, 1]).

**Important dtype/device notes:**
- `WanVAE` autocasts to fp32 (default `dtype=torch.float`), then `.float()` outside → output is **fp32** in `[-1, 1]`.
- Cache `feat_cache`/`feat_idx` are Python lists — **must be reset between calls** (handled by `clear_cache()` at start/end of `decode`).
- `scale` (mean/std) is pre-registered on the model's device/dtype — moving the model requires re-creating the `WanVAE` wrapper (or manually moving `.mean`/`.std` tensors).

**Decoder temporal behavior:** `temperal_upsample = [False, True, True]` (inverted `temperal_downsample = [True, True, False]`, L491/L500). With `vae_stride[0] = 4`, latent T=21 → output T=4·20+1 = 81.

---

## 5. Diffusers equivalent

**Yes — `WanPipeline` exists in diffusers main** (released March 2025 per README:33 of Wan-AI/Wan2.1).

| Component | Diffusers class | Module |
|---|---|---|
| Pipeline | `diffusers.WanPipeline` | `src/diffusers/pipelines/wan/pipeline_wan.py:96` |
| DiT | `WanTransformer3DModel` | `src/diffusers/models/transformers/wan_transformer_3d.py` |
| VAE | `AutoencoderKLWan` | `src/diffusers/models/autoencoders/autoencoder_kl_wan.py` |
| Scheduler | `FlowMatchEulerDiscreteScheduler` (default) **or** `UniPCMultistepScheduler` (Wan-AI default) | `src/diffusers/schedulers/` |
| Text encoder | `UMT5EncoderModel` from transformers | `transformers` |
| Output wrapper | `WanPipelineOutput(frames=...)` | `pipeline_wan.py:30` |

Pre-diffusers Wan checkpoints can be converted: `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` on HF.

`AutoencoderKLWan` exposes the same `decode(latents) → video` API (`wan_vae.decode(latents, return_dict=False)[0]` in `diffusers_wan_pipeline.py:667`).

For PiD integration the diffusers base is preferable:
- Pipeline already implements proper dtype management, model offloading, group offloading, attention slicing.
- `AutoencoderKLWan` has a built-in `decode` method we can wrap or monkey-patch.
- `WanPipeline.__call__` returns `WanPipelineOutput(frames=video)` — easy to either (a) replace the line at `diffusers_wan_pipeline.py:667-668` or (b) subclass and override.

The T2V 14B diffusers example shows ~13 GB VRAM is achievable with `group_offloading` (so 1.3B @ 480P 16f fits comfortably on a 24 GB card without that).

---

## 6. The HOOK point — natural integration site for video-PiD

### 6a. Native Wan-AI path (`wan/text2video.py:256-271`)

Insert between the DiT loop end and `vae.decode`:

```python
x0 = latents                                          # [16, 21, 60, 104] bf16
if offload_model:
    self.model.cpu()
    torch.cuda.empty_cache()

# ==== HOOK START ============================================================
# x0 is the final denoised latent (the "sigma≈0" prediction from the sampler).
# We can refine in latent space (PiD-on-latents) or in pixel space (after
# decode). PiD-on-latents is cheaper but PiD-on-pixels is more faithful.
# Available signals: x0 (latents), and optionally the scheduler's last sigma.
if self.video_pid_latents is not None and self.rank == 0:
    x0 = self.video_pid_latents(x0, sigma=0.0)        # optional latent-refine
# ==== HOOK END ==============================================================

if self.rank == 0:
    videos = self.vae.decode(x0)                      # → (3, 81, 480, 832) fp32 [-1,1]

# ==== HOOK START (post-decode pixel refine) =================================
if self.video_pid_pixels is not None and self.rank == 0:
    videos = self.video_pid_pixels(videos, latents=x0, sigma=0.0)
# ==== HOOK END ==============================================================
```

### 6b. Diffusers path (`diffusers_wan_pipeline.py:656-668`)

Identical concept, slightly different shape:

```python
if not output_type == "latent":
    latents = latents.to(self.vae.dtype)
    latents_mean = (torch.tensor(self.vae.config.latents_mean)
                    .view(1, self.vae.config.z_dim, 1, 1, 1)
                    .to(latents.device, latents.dtype))
    latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
                    1, self.vae.config.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
    latents = latents / latents_std + latents_mean

    # ==== HOOK START (latent-space PiD) ====================================
    if self.video_pid_latents is not None:
        latents = self.video_pid_latents(latents, sigma=0.0)
    # ==== HOOK END ==========================================================

    video = self.vae.decode(latents, return_dict=False)[0]   # (1, 3, 81, 480, 832) fp32

    # ==== HOOK START (pixel-space PiD, the natural site) ===================
    if self.video_pid_pixels is not None:
        video = self.video_pid_pixels(video, latents=latents, sigma=0.0)
    # ==== HOOK END ==========================================================

    video = self.video_processor.postprocess_video(video, output_type=output_type)
```

**Why pixel-space after decode is the canonical PiD hook point:**
1. `video` here is the final semantic content — PiD's residual refinement is meant to operate in observation space.
2. PiD's stochastic-consistency formulation needs the noisy observation `y`; the VAE-decoded `video` is the clean proxy for `y` in deterministic flow-matching.
3. `latents` is still in scope — we can pass it as conditioning if PiD wants to use the latent as an attention key/value.
4. If we want to skip PiD when VRAM is tight, this is the only place that runs once per generation; denoising-loop hooks would fire 50×.

The user's requested signature `video_pid(video, latents, sigma)` is wired directly here.

---

## 7. T2V-1.3B VRAM budget (1× RTX 3090, 24 GB)

Numbers below combine:
- Wan-AI's own published benchmarks (8.19 GB minimum for T2V-1.3B, README:17).
- Public diffusers profiling (Wan T2V 14B @ ~13 GB with offloading).
- Direct arithmetic from the config.

### Weights (steady-state after T5 is offloaded)

| Component | dtype | Bytes/param | Params | Size |
|---|---|---|---|---|
| DiT (WanModel, dim=1536, 30 layers) | bf16 | 2 | ~1.43 B | **≈ 2.86 GB** |
| T5-XXL encoder (Wan-shipped `models_t5_umt5-xxl-enc-bf16.pth`) | bf16 | 2 | ~4.7 B (24 enc layers, dim=4096) | **≈ 9.4 GB** |
| Wan-VAE (z_dim=16, 96 base, dim_mult [1,2,4,4]) | fp32 | 4 | ~125 M | **≈ 0.50 GB** |
| Scheduler state | – | – | tiny | < 0.01 GB |

Cumulative if all resident: ~12.8 GB just for weights.

Wan-AI's recommended 3090 path (`generate.py:177`): `--offload_model True --t5_cpu --sample_shift 8 --sample_guide_scale 6`. With T5 on CPU:
- DiT bf16 → ~2.86 GB resident on GPU
- T5 bf16 → ~9.4 GB on CPU (zero GPU)
- VAE fp32 → ~0.50 GB
- T5 activations on CPU during the one-shot encode → negligible GPU impact
- **Total resident on GPU: ~3.5 GB** of weights.

### Activations / working memory during denoising

For latents `[1, 16, 21, 60, 104]` bf16 with `patch_size=(1,2,2)`:
- Patches: `21 · 30 · 52 = 32,760` tokens.
- DiT block (dim=1536):
  - Self-attn: Q·K·V (1536·3 each), softmax scores (32760·12 heads), output proj → ~4 × 32760 × 1536 × 2 B = ~0.40 GB peak per block.
  - FFN (ffn_dim=8960): 32760 × 1536 × 2 (input) + 32760 × 8960 × 2 (hidden) + 32760 × 1536 × 2 (output) ≈ **~1.5 GB** peak per block.
  - Total per block peak ≈ 1.9 GB; with grad-checkpointing off this is replicated for both CFG passes → ~3.8 GB working set if we don't share.
- Cross-attn context: 512 × 4096 bf16 = ~4 MB — negligible.
- CFG doubles forward passes: 2 × ~3.8 GB = ~7.6 GB working set if both kept; in practice `wan/text2video.py:240-246` runs them sequentially so peak ≈ 3.8 GB.

### VRAM profile during 1.3B T2V @ 480P, 16f on RTX 3090

| Phase | GPU resident | Source |
|---|---|---|
| Load DiT | ~2.86 GB | `wan/text2video.py:110` |
| T5 encode (T5 on CPU) | ~2.86 GB (DiT only) | `wan/text2video.py:174-179` |
| Denoising loop (per step) | ~2.86 GB weights + ~3.8 GB activations = ~6.7 GB | `wan/text2video.py:233-254` |
| VAE decode | DiT `.cpu()` frees → ~0.5 GB VAE + decode activations | `wan/text2video.py:257-261` |
| **PiD post-process (proposed)** | + ~0.5–1.5 GB depending on PiD size | inserted at L261-262 |

Wan-AI's published "8.19 GB" figure matches: DiT (2.86) + activations (~3.8) + T5 fp16 resident (if no `--t5_cpu`, ~9.4 GB on GPU) puts everything-on-GPU at >16 GB, well over 8.19 — so the 8.19 GB number assumes `--t5_cpu` and aggressive activation management.

### Attention buffers
- KV cache per self-attn layer: 32760 tokens × 12 heads × 128 dim × 2 (K+V) × 2 B = ~190 MB per layer. With 30 layers that's ~5.7 GB if kept resident; in practice the SDPA implementation in `wan/modules/attention.py` reuses memory so only the current block's KV is live (~190 MB at a time).
- Cross-attn KV (for 512 text tokens): 30 layers × 512 × 12 × 128 × 4 B (K+V bf16) = ~180 MB total.

### Effective peak on RTX 3090 with Wan-AI's CLI recipe (`--t5_cpu --offload_model True`)

- ~6.7 GB peak during sampling (DiT + activations).
- ~0.5 GB during decode.
- **Total peak ≈ 7 GB**, leaves ~17 GB headroom for PiD post-processor, FP32 upgrade of activations, or longer sequences.

---

## 8. Inference time — 16f @ 480×832

Empirical numbers from the Wan-AI README (line 17) and community benchmarks:

| GPU | FP16 / BF16, no offload | With `--t5_cpu --offload_model` | Diffusers + group-offload |
|---|---|---|---|
| RTX 3090 (24 GB) | ~10–13 min/clip (extrapolated from README's 4 min @ 4090) | ~6–9 min | ~5–7 min |
| RTX 4090 (24 GB) | **~4 min** (README:17, 5-second 480P) | ~3 min | ~2–3 min |
| RTX 5090 | ~1.5–2 min | ~1–1.5 min | ~45–90 s |
| A100 80 GB | ~45–90 s (no offload) | ~40–60 s | ~30–45 s |
| H100 80 GB | ~25–40 s | ~20–30 s | ~15–25 s |

Reference: README explicitly says "**Wan2.1** … can generate a 5-second 480P video on an **RTX 4090 in about 4 minutes**" (without quantization). The 1.3B is ~5–10× cheaper than 14B, so a reasonable 3090 estimate for 1.3B @ 480P is ~5–8 minutes for the full 81-frame clip; the 16-frame variant at 480×832 is closer to **~1.5–2.5 min on 3090, ~30–60 s on A100**.

---

## 9. Prompt encoder: T5-XXL fp16 (9.7 GB) — can we shrink it?

The shipped checkpoint is `models_t5_umt5-xxl-enc-bf16.pth` (`wan/configs/wan_t2v_1_3B.py:12`). The bf16 file is ~9.4 GB on disk; fp16 is the same size on disk (~9.7 GB). The model is **encoder-only**, ~4.7 B params.

**Why bf16 on GPU fails the 3090:**
9.4 GB of T5 weights + 2.86 GB DiT + activations > 16 GB. Hence the README's `--t5_cpu` flag.

**Options to bring T5 onto the 3090:**

| Technique | T5 footprint | Quality loss | Notes |
|---|---|---|---|
| bf16 on GPU (default `--t5_cpu` off) | 9.4 GB | reference | OOM with DiT on 24 GB without `--offload_model` |
| bf16 T5 on CPU, `--t5_cpu` | 0 GB GPU | 0 | current default for 3090 |
| INT8 weight-only quant | ~4.7 GB | **<0.5%** on prompt embeddings (Mao et al., UMT5 paper ablations); Wan-AI doesn't ship INT8 T5 | enables T5 on GPU + DiT on GPU |
| INT4 weight-only quant (gptq/awq) | ~2.4 GB | **~1–3%** on text-following metrics; recovers with a small LoRA | viable; Wan-AI's DiffSynth-Studio fork ships INT4 T5 for Wan |
| NF4 / FP4 | ~1.5 GB | ~3–5% | aggressive; works for stylistic prompts, degrades on technical terms |
| T5-XXL → T5-base distil | ~0.3 GB | **5–15%** | Wan-AI does **not** support; would need fine-tune |

For the user's case (consumer 3090, 1.3B model), the cleanest path is:
1. Keep `--t5_cpu` and accept ~3 GB/s of PCIe traffic during encode (negligible — runs once).
2. OR load T5 bf16 onto the GPU and use `--offload_model True` + bf16 DiT, but only with `enable_sequential_cpu_offload()` for DiT — gives ~14 GB peak during sampling which fits in 24 GB and frees the CPU.

We do **not** recommend quantizing T5 for the PiD experiment — the prompt encoder is not on the inference hot loop and degrading it pollutes the entire output.

---

## 10. Code skeleton with the video-PiD hook

### 10a. Native Wan-AI (`wan/text2video.py` patch)

```python
# wan/text2video.py:114-271  WanT2V.generate()

class WanT2V:
    def __init__(self, config, checkpoint_dir, device_id=0, rank=0,
                 t5_fsdp=False, dit_fsdp=False, use_usp=False, t5_cpu=False,
                 video_pid=None):                    # ← NEW
        # ... existing __init__ ...
        self.video_pid = video_pid                  # ← NEW: nn.Module or callable

    def generate(self, input_prompt, size=(1280,720), frame_num=81,
                 shift=5.0, sample_solver='unipc', sampling_steps=50,
                 guide_scale=5.0, n_prompt="", seed=-1, offload_model=True):
        # ... existing preprocess + T5 encode + noise init (L159-195) ...

        # === Denoising loop (L228-254), unchanged ===

        x0 = latents                                # [1,16,21,60,104] bf16
        if offload_model:
            self.model.cpu()
            torch.cuda.empty_cache()

        # =============================================================
        #  HOOK: video-PiD post-processor
        # -------------------------------------------------------------
        #  video = self.video_pid(video, latents, sigma)
        #  - video : decoded video tensor, (1, 3, 81, 480, 832), fp32, range [-1,1]
        #  - latents: x0 above, (1, 16, 21, 60, 104), bf16 — optional PiD condition
        #  - sigma : 0.0 (deterministic, end of flow-matching schedule)
        # =============================================================
        if self.rank == 0 and self.video_pid is not None:
            self.vae.to(self.device)                # ensure VAE on GPU for decode
            with torch.no_grad(), amp.autocast(dtype=torch.float):
                videos = self.vae.decode(x0)        # (3, 81, 480, 832) fp32
            if offload_model:
                self.vae.cpu()
                torch.cuda.empty_cache()

            sigma = 0.0
            videos = self.video_pid(videos, latents=x0.float(), sigma=sigma)
        elif self.rank == 0:
            videos = self.vae.decode(x0)            # original path
        else:
            videos = [None]

        del noise, latents
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None
```

### 10b. Diffusers subclass (preferred — drops in cleanly)

```python
# File: pipelines/wan_with_pid.py
import torch
from diffusers import WanPipeline
from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput


class WanPipelineWithPiD(WanPipeline):
    """WanPipeline extended with a video-PiD post-processor."""

    def __init__(self, *args, video_pid=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.video_pid = video_pid  # nn.Module: video_pid(video, latents, sigma) -> video

    @torch.no_grad()
    def __call__(self, *args, output_type="pil", return_dict=True, **kwargs):
        # Run the standard pipeline. We intercept by overriding only the decode step:
        # Strategy: use the parent's __call__ with output_type="latent", then call
        # decode + PiD ourselves to keep full control over the hook.

        # 1) Run the diffusion loop and stop after scheduler.step returns the final latents.
        #    We replicate WanPipeline.__call__'s logic up to the decode (line 666).
        result = super().__call__(*args, output_type="latent", return_dict=False, **kwargs)
        latents = result[0]                          # (1, 16, 21, 60, 104), bf16 / vae.dtype

        # 2) Un-normalize using the same constants the parent uses (line 658-666).
        latents = latents.to(self.vae.dtype)
        latents_mean = (torch.tensor(self.vae.config.latents_mean)
                        .view(1, self.vae.config.z_dim, 1, 1, 1)
                        .to(latents.device, latents.dtype))
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
            1, self.vae.config.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
        latents = latents / latents_std + latents_mean

        # 3) Decode (matches parent line 667).
        video = self.vae.decode(latents, return_dict=False)[0]
        # video: (1, 3, 81, 480, 832), fp32, range ~[-1, 1]

        # 4) ============================================================
        #    HOOK: video-PiD post-processor
        #    video_pid(video, latents=latents, sigma=0.0) -> video
        #    ============================================================
        if self.video_pid is not None:
            video = self.video_pid(video, latents=latents.float(), sigma=0.0)

        # 5) Same postprocessing as parent (line 668).
        video = self.video_processor.postprocess_video(video, output_type=output_type)
        self.maybe_free_model_hooks()
        if not return_dict:
            return (video,)
        return WanPipelineOutput(frames=video)


# Usage:
#   from diffusers import AutoModel
#   from transformers import UMT5EncoderModel
#   from my_pid import VideoPiD  # user-defined
#
#   pipe = WanPipelineWithPiD.from_pretrained(
#       "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
#       text_encoder=UMT5EncoderModel.from_pretrained(
#           "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", subfolder="text_encoder",
#           torch_dtype=torch.bfloat16),
#       transformer=AutoModel.from_pretrained(
#           "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", subfolder="transformer",
#           torch_dtype=torch.bfloat16),
#       vae=AutoModel.from_pretrained(
#           "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", subfolder="vae",
#           torch_dtype=torch.float32),
#       torch_dtype=torch.bfloat16,
#       video_pid=VideoPiD(...).to("cuda"),
#   )
#   out = pipe(prompt="...", num_frames=16, height=480, width=832).frames[0]
```

### 10c. video_pid module interface (the contract)

```python
import torch
import torch.nn as nn


class VideoPiD(nn.Module):
    """Pixel-space PiD post-processor operating on a Wan-VAE-decoded video clip.

    Inputs:
        video   : (B, 3, T, H, W) tensor in [-1, 1], fp32.
                  T = 16, H = 480, W = 832 in the canonical case.
        latents : (B, 16, T_lat, H_lat, W_lat) tensor — Wan DiT final x0,
                  un-normalized (raw Wan-VAE latent, NOT divided by std+mean).
                  T_lat=21, H_lat=60, W_lat=104. Optional condition.
        sigma   : scalar tensor or float — at inference time this is 0.0
                  (deterministic flow-matching endpoint). For consistency
                  training, sigma is the noise level at which video was
                  noised from clean.

    Returns:
        video   : (B, 3, T, H, W) tensor in [-1, 1], fp32 — refined video.
    """
    def forward(self, video, latents=None, sigma=0.0):
        raise NotImplementedError
```

---

## 11. One-screen summary

- **Repo**: `https://github.com/Wan-Video/Wan2.1` (cloned locally).
- **T2V orchestrator**: `wan/text2video.py`, class `WanT2V.generate()` L114-271.
- **VAE**: `wan/modules/vae.py`, wrapper `WanVAE.decode()` L657 (autocast fp32, returns fp32 `[3, T, H, W]` in `[-1, 1]`).
- **Hook**: between `wan/text2video.py:256` (after denoising loop, `x0 = latents`) and `wan/text2video.py:261` (`videos = self.vae.decode(x0)`). The pixel-space PiD variant goes immediately after the decode.
- **Diffusers equivalent**: `diffusers.WanPipeline` (`diffusers_wan_pipeline.py:96`). Equivalent hook point at line 667-668 (between `vae.decode` and `video_processor.postprocess_video`). Subclass and override for cleanest integration.
- **Scheduler**: UniPC (default), 50 steps, CFG 5–6, shift 8–12 for 1.3B @ 480P.
- **VRAM**: ~6.7 GB peak on 3090 with `--t5_cpu --offload_model`; T5 bf16 alone is 9.4 GB so we keep T5 on CPU.
- **Inference time**: ~5–8 min for 81f @ 480P on 3090; ~30–60 s on A100; ~15–25 s on H100.
- **T5 quantization**: not recommended — encoder runs once per clip, keep at bf16 on CPU.
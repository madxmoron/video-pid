# Wan 2.1 14B — Architecture Deep Dive

Compiled from source: `Wan-Video/Wan2.1` (main, fetched 2026-07-14) +
`huggingface/diffusers` `WanTransformer3DModel`.

**Important framing:** Wan 2.1 is **NOT** MM-DiT. It is a vanilla **DiT
(self-attn) + cross-attn to T5** backbone — same family as DiT XL, NOT
Flux/SD3-style joint attention. It uses AdaLN with 6 modulation
parameters per block (DiT-style), not adaLN-zero.

---

## 1. Backbone architecture

Source: `wan/configs/wan_t2v_14B.py`, `wan/modules/model.py:WanModel` (line 372).

| Property                     | 14B (T2V / I2V)         | 1.3B (T2V)              |
|------------------------------|-------------------------|-------------------------|
| `dim` (hidden)               | **5120**                | 1536                    |
| `ffn_dim` (MLP)              | **13824**               | 8960                    |
| `num_heads`                  | **40**                  | 12                      |
| `head_dim`                   | 128 (=5120/40)          | 128                     |
| `num_layers`                 | **40**                  | 30                      |
| `in_dim` (input channels)    | 16                      | 16                      |
| `out_dim` (output channels)  | 16                      | 16                      |
| `patch_size` (Conv3d)        | (1, 2, 2)               | (1, 2, 2)               |
| `text_len` (T5 tokens)       | 512                     | 512                     |
| `freq_dim` (time emb)        | 256                     | 256                     |
| `qk_norm`                    | True (RMSNorm)          | True                    |
| `cross_attn_norm`            | True (LayerNorm affine) | True                    |
| `eps`                        | 1e-6                    | 1e-6                    |
| `window_size`                | (-1, -1) = full global  | (-1, -1)                |

**Activation:** GELU(approximate='tanh') in FFN; SiLU in time MLP;
GELU(approximate='tanh') in text-embedding MLP. (source:
`WanAttentionBlock.ffn` and `WanModel.time_embedding`)

**HF config.json on disk (Wan-AI/Wan2.1-T2V-14B):** confirms exactly the
same `dim=5120, ffn_dim=13824, num_heads=40, num_layers=40, in_dim=16,
out_dim=16, freq_dim=256, text_len=512`. (source:
`https://huggingface.co/Wan-AI/Wan2.1-T2V-14B/raw/main/config.json`)

### Block structure: `WanAttentionBlock` (`wan/modules/model.py:238`)

Per block (in order):
1. `norm1 = WanLayerNorm(dim)` (no affine)
2. **Self-attention** `WanSelfAttention(dim, num_heads)` — pre-RoPE Q,K
3. `norm3 = WanLayerNorm(dim, elementwise_affine=True)` if
   `cross_attn_norm` else `Identity`
4. **Cross-attention** `WanT2VCrossAttention` / `WanI2VCrossAttention` —
   Q from video, K,V from text context
5. `norm2 = WanLayerNorm(dim)` (no affine)
6. **FFN**: `Linear(dim→ffn_dim) → GELU(tanh) → Linear(ffn_dim→dim)`
7. **AdaLN modulation**: `modulation = Parameter(1,6,dim)` — 6 vectors
   per block (shift_sa, scale_sa, gate_sa, shift_ffn, scale_ffn,
   gate_ffn) — chunked from `e = time_embedding(t) →
   time_projection → (B,6,dim)` plus the learnable per-block
   `modulation` parameter.

### Per-block parameter count (14B, computed)

```
self-attn  Q+K+V+O:  4·(5120²) = 104.86 M
  norm_q + norm_k (RMS, weight only):  10.24 K
cross-attn Q+O    :  2·(5120²) =  52.43 M
cross-attn K+V    :  2·(5120²) =  52.43 M
  norm_q + norm_k (RMS)            :  10.24 K
FFN              :  5120·13824 + 13824·5120 = 141.56 M
norm3 (cross, affine): 2·5120    = 10.24 K
modulation table : 6·5120        = 30.72 K
                                          ──────────
                                   per block ≈ 351.3 M
40 blocks                              ≈ 14.05 B
```

### Total params breakdown (DiT only, 14B, computed)

| Component            | Params     |
|----------------------|------------|
| 40× WanAttentionBlock | **14.05 B** |
| `patch_embedding` Conv3d (16→5120, k=(1,2,2)) | 0.33 M |
| `text_embedding` MLP (4096→5120→5120) | 47.19 M |
| `time_embedding` MLP (256→5120→5120) | 26.42 M |
| `time_projection` Linear (5120→5120·6) | 157.29 M |
| `head.norm` LayerNorm (no affine) | 0 |
| `head.head` Linear (5120→16·4) | 0.33 M |
| `head.modulation` (1,2,5120) | 10.24 K |
| **TOTAL DiT**        | **~14.29 B** |

(T5-XXL encoder is a separate ~4.8 B params loaded independently;
Wan-VAE is a separate ~125 M params.)

---

## 2. 3D RoPE — temporal + spatial

Source: `wan/modules/model.py:rope_params` (line 32), `rope_apply` (42),
and the buffer init at `WanModel.__init__` line 480.

```python
# In WanModel.__init__ (line 480), constructed once as a non-P buffer:
d = dim // num_heads                                # 128 for 14B
self.freqs = torch.cat([
    rope_params(1024, d - 4*(d//6)),                # T-axis: 104 dims
    rope_params(1024, 2*(d//6)),                    # H-axis:  40 dims
    rope_params(1024, 2*(d//6)),                    # W-axis:  40 dims
], dim=1)                                           # total 184 per head/2
```

Important subtlety: **the head-dim split is `104 / 40 / 40` for the
`dim // 2 = 64`-side, but the split happens on `c = head_dim // 2 = 64`
in `rope_apply` (line 47)**:

```python
def rope_apply(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2                # c = head_dim//2 = 64
    freqs = freqs.split([c - 2*(c//3), c//3, c//3], dim=1)  # 22, 21, 21
    ...
```

So inside the per-head dimension the actual **per-axis allocation is
22 / 21 / 21** in `head_dim//2` space. The wider split (`104/40/40`) at
init just makes the buffer; the **runtime split is `c - 2(c//3)` / `c//3`
/ `c//3` where `c = head_dim // 2`**.

- **Base theta: 10000** (standard RoPE), no per-axis scaling.
- **Max seq len per axis: 1024** (the `1024` in `rope_params`).
- **Temporal axis is encoded identically** to spatial axes — same
  RoPE frequency formula; just split into a separate sub-band of the
  head dim. No special "time" theta. Frame index is implicit in the
  **position of the token along T within the flattened `F·H·W` token
  sequence**; `grid_sizes` (B,3) holds `(F,H,W)` per sample and
  `rope_apply` indexes `freqs[0][:f]`, `freqs[1][:h]`, `freqs[2][:w]`
  and broadcasts to `(f,h,w,-1)`.
- **Causal temporal mask: NONE.** The model is non-causal in time; it
  attends across all T·H·W tokens bidirectionally. (See §3.)
- No rotary on the cross-attention K/V (only self-attention Q,K get
  RoPE).

---

## 3. Attention pattern

Source: `wan/modules/attention.py:flash_attention`, `model.py:WanSelfAttention`.

- **Full global self-attention** across the flattened `T·H·W` token
  sequence per sample. `window_size=(-1,-1)` is hardcoded in config;
  the call passes it to `flash_attn_varlen_func(window_size=(-1,-1))`
  meaning "no sliding window".
- **No causal mask** (`causal=False` everywhere).
- **No block-causal / sparse pattern** — every token sees every token.
- **Implementation:** Flash Attention 2 or 3 (varlen), dispatched from
  `flash_attention` in `attention.py`. Falls back to
  `scaled_dot_product_attention` if neither is installed (with a perf
  warning). Heads dim ≤ 256 required.
- **QK-norm: RMSNorm** applied **before** RoPE on Q and K (per head
  dim), weights of shape `dim`. (Source: `WanSelfAttention.norm_q`,
  `norm_k`.)
- **Cross-attention:** separate `WanT2VCrossAttention` class, full
  attention over the 512 T5 tokens, no RoPE, no QK norm dim issue.
  For **I2V**: `WanI2VCrossAttention` splits context into
  `[image_embeds(257), t5_embeds(512)]`, runs **two** cross-attentions
  (text + image) and adds the outputs before `o` projection. Image
  branch has its own `k_img`/`v_img` linears + RMS-norm on k_img.

**Token counts** (computed for typical inference, post-VAE, post-patch):
- 480p 16 frames → T_lat=4, H_lat=60, W_lat=104 → **24,960 tokens**
- 720p 16 frames → T_lat=4, H_lat=90, W_lat=156 → **56,160 tokens**

That is large — full O(N²) attention in self-attn: 480p ≈ 6.2e8
entries, 720p ≈ 3.2e9 entries per layer, x40 layers. This is why
the repo ships Ulysses + Ring Attention sequence parallelism in
`wan/distributed/xdit_context_parallel.py`.

---

## 4. Text conditioning — T5-XXL (NOT CLIP)

Source: `wan/modules/t5.py:umt5_xxl` (line 456),
`wan/configs/shared_config.py`, `wan/modules/model.py:text_embedding`.

**Encoder:** `umt5-xxl` (the mT5-style multilingual T5 from Google).
Wan loads it as encoder-only with weights from
`models_t5_umt5-xxl-enc-bf16.pth` and the HF tokenizer
`google/umt5-xxl`.

| Property             | Value                           |
|----------------------|---------------------------------|
| `vocab_size`         | 256,384                         |
| `dim`                | **4096**                        |
| `dim_attn`           | 4096                            |
| `dim_ffn`            | 10,240                          |
| `num_heads`          | 64                              |
| `head_dim`           | 64                              |
| `encoder_layers`     | 24                              |
| `num_buckets` (rel pos) | 32                            |
| `shared_pos`         | False (sinusoidal, learned rel emb) |
| `dropout`            | 0.1                             |
| Fixed text length    | **512 tokens** (padded/truncated) |
| dtype                | bfloat16                        |
| T5 attention scaling | None (no 1/√d) — T5 quirk       |

**No CLIP-L** in T2V. Only the **I2V-14B variant** adds CLIP:
- `clip_model = 'clip_xlm_roberta_vit_h_14'` (SigLIP-style via
  OpenCLIP, ViT-H/14, image encoder)
- `clip_checkpoint = 'models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth'`
- `clip_dtype = float16`
- `clip_tokenizer = 'xlm-roberta-large'` (for the caption side)
- Image branch produces **257 tokens** (16×16 patches + 1 CLS),
  dim **1280**.

**Fusion with video tokens — this is NOT MM-DiT / Flux-style joint
attention.** It is plain **cross-attention**:

```python
# WanModel.forward() (model.py:553)
context = self.text_embedding(
    torch.stack([... padded 512-token sequences ...])
)                                    # (B, 512, dim=5120)
if clip_fea is not None:
    context_clip = self.img_emb(clip_fea)   # (B, 257, dim=5120)
    context = torch.cat([context_clip, context], dim=1)  # (B, 769, 5120)

# per block:
x = x + self.self_attn(self.norm1(x)*scale_sa + shift_sa, ...)
x = x + self.cross_attn(self.norm3(x), context, context_lens)
x = x + self.ffn(self.norm2(x)*scale_ffn + shift_ffn) * gate_ffn
```

- `text_embedding`: `Linear(4096, 5120) → GELU(tanh) → Linear(5120, 5120)`
- Each `WanAttentionBlock` owns its own `q, k, v, o` for cross-attn —
  text context goes through a **per-block** K/V projection (not shared).
- I2V: per-block has **two pairs** of K/V projections: `k,v` (text)
  and `k_img, v_img` (CLIP image), and two cross-attention calls whose
  outputs are summed.
- The MLP projector for CLIP image features is `MLPProj(1280 →
  dim)` with optional learned 2-frame position embedding (FLF2V only).

---

## 5. Wan-VAE

Source: `wan/modules/vae.py`, `wan/configs/wan_t2v_14B.py`.

| Property          | Value                                                   |
|-------------------|---------------------------------------------------------|
| Compression (T,H,W) | **(4, 8, 8)** — from `vae_stride = (4, 8, 8)`       |
| Latent channels `z_dim` | **16**                                              |
| Type              | **KL-VAE** (encoder outputs `mu, log_var`, sampled with reparam) — see `WanVAE_.reparameterize` |
| Encoder dim base  | 96 (`dim=96` in `_video_vae` cfg)                       |
| `dim_mult`        | `[1, 2, 4, 4]` → channel widths [96, 192, 384, 384, 384] |
| `num_res_blocks`  | 2 per stage                                             |
| `attn_scales`     | **[] (no attention blocks in VAE)**                     |
| `temperal_downsample` | `[True, True, False]` for encoder; reversed for decoder |
| Causal temporal   | **YES — `CausalConv3d`** (asymmetric left-padding on T) |
| Activation        | SiLU; RMS_norm (channel-first)                         |
| Latent shift/scale | Hardcoded `mean` (16,) and `std` (16,) tensors used in `encode`/`decode`: |

```
mean = [-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
        0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921]
std  = [ 2.8184,  1.4541,  2.3275,  2.6558,  1.2196,  1.7708,  2.6052,  2.0743,
        3.2687,  2.1526,  2.8652,  1.5579,  1.6382,  1.1253,  2.8251,  1.9160]
```

**Causal Conv3d** = Conv3d with asymmetric T-padding
(`padding=(0,0,0,0, 2*pad_T, 0)`), so each latent frame only depends
on past + current frames, not future. This makes the VAE suitable
for **streaming / autoregressive decoding** and chunked inference
(see `feat_cache` in `WanVAE_.encode` / `decode`).

**Chunked inference pattern:** the encoder is run on
`[1, 4, 4, ..., 4]` frame chunks; the decoder runs one latent frame
at a time, passing the last `CACHE_T=2` frames as cache.

**Output shape:** for a video of shape `[C=3, T, H, W]`, latent is
`[16, T//4, H//8, W//8]`.

**Total params:** Wan-VAE is small — ~125 M (mostly the 4-level
encoder/decoder with Conv3d). No flash attn inside VAE.

---

## 6. Timestep / flow-matching conditioning

Source: `model.py:sinusoidal_embedding_1d` (line 18), `time_embedding`,
`time_projection` (line 462), `WanAttentionBlock.forward` (line 278).

This is **flow matching**, not classical DDPM. The scheduler is in
`wan/utils/fm_solvers.py` and `fm_solvers_unipc.py` (UniPC +
DPM-Solver flow-matching variants).

```python
# WanModel.forward()
e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).float())
#   = SiLU(Linear(256, 5120)) → Linear(5120, 5120)
e0 = self.time_projection(e).unflatten(1, (6, dim))
#   = SiLU() → Linear(5120, 5120 * 6)

# Per block (WanAttentionBlock.forward):
e = (self.modulation + e0).chunk(6, dim=1)
# 6 modulation vectors: shift_sa, scale_sa, gate_sa, shift_msa, scale_msa, gate_msa
y = self.self_attn(self.norm1(x) * (1 + e[1]) + e[0], ...)   # scale_sa & shift_sa
x = x + y * e[2]                                            # gate_sa

x = x + self.cross_attn(self.norm3(x), context, context_lens)  # cross-attn has no modulation

y = self.ffn(self.norm2(x) * (1 + e[4]) + e[3])             # scale_msa & shift_msa
x = x + y * e[5]                                            # gate_msa
```

- **Sinusoidal timestep embedding** of dim `freq_dim=256`, **base
  10000** (`pow(10000, -arange/half)`).
- One MLP `Linear(256→5120) → SiLU → Linear(5120→5120)` produces the
  base timestep vector.
- One projection `Linear(5120→5120·6)` produces the per-block
  modulation; **this is shared across all 40 blocks** (single
  `e0` of shape `(B, 6, dim)` is broadcast).
- Per-block learnable `modulation = Parameter(1, 6, dim)` is **added**
  to the broadcast (DiT-style; not adaLN-zero where the per-block
  params are zero-initialized and a residual scale is learned).
- The output `Head` (line 320) has its own `modulation = Parameter(1,
  2, dim)` for shift/scale before final `Linear(dim → out_dim ·
  patch_prod)`.
- **No bias addition**, no adaLN-zero, no class-embedding
  conditioning. Only `t` + per-block `modulation`.

`num_train_timesteps = 1000` (from `wan/configs/shared_config.py`).

---

## 7. Frame packing / position embeddings

There is **no explicit per-frame learned position embedding** in the
video DiT. Frame identity comes purely from 3D RoPE on the flattened
`(F, H, W)` token grid:

- `x = patch_embedding(u.unsqueeze(0))` → shape `(1, dim, F', H', W')`
  where `F' = F/t_patch, H' = H/h_patch, W' = W/w_patch`
- `grid_sizes = stack([(F', H', W') for u in x])` → `(B, 3)`
- Flatten to `(1, F'·H'·W', dim)`, pad to fixed `seq_len` with zeros
- Pass through `WanSelfAttention.forward(x, seq_lens, grid_sizes,
  freqs)` which calls `rope_apply(q, grid_sizes, freqs)` — RoPE
  indexes `freqs[0][:f]`, `freqs[1][:h]`, `freqs[2][:w]` and
  broadcasts.
- For batched training, sequences are padded to `seq_len` and
  `seq_lens` is the per-sample actual length for FlashAttention's
  varlen cu_seqlens.

The only **learned** position-like embedding in the whole pipeline is
`MLPProj.emb_pos` (only in `flf2v` mode for first+last frame context,
shape `(1, 514, 1280)`).

---

## 8. Training / inference defaults

Source: `wan/configs/shared_config.py`, `generate.py`, HF `model_index.json`.

### Defaults from `generate.py` (CLI)

| Arg                  | Default         | Notes                                   |
|----------------------|-----------------|-----------------------------------------|
| `--sample_solver`    | `unipc`         | also `dpm++`                            |
| `--sample_steps`     | 50 (T2V/T2I), 40 (I2V/FLF) |                                  |
| `--sample_shift`     | 5.0 (T2V 720p), 3.0 (I2V 480p), 16.0 (FLF/VACE) | flow-match shift |
| `--sample_guide_scale` | 5.0 (T2V), 6.0 (1.3B T2V) | CFG                          |
| `--frame_num`        | 81 (≈5s @ 16fps) | 1 for T2I                              |
| `--size`             | `1280*720`      | `720*1280`, `1280*720`, `480*832`, `832*480` |
| `sample_fps`         | 16              | from `shared_config.py`                 |

### Supported sizes (from `wan/configs/__init__.py`)

```
t2v-14B : 720*1280, 1280*720, 480*832, 832*480    (both 480p & 720p)
t2v-1.3B: 480*832, 832*480                        (480p only)
i2v-14B : 720*1280, 1280*720, 480*832, 832*480
flf2v-14B: same 4 sizes
vace-14B / 1.3B: same
```

### Scheduler

HF `model_index.json`:
```
scheduler: UniPCMultistepScheduler
text_encoder: UMT5EncoderModel
vae: AutoencoderKLWan
transformer: WanTransformer3DModel
```

The HF model's HF-API card lists `num_inference_steps: 10` as a
quick-test value; the official Wan repo defaults to 50/40.

### Flow-matching shift

```python
# generate.py:77-80
if "i2v" in args.task and args.size in ["832*480", "480*832"]:
    args.sample_shift = 3.0
elif "flf2v" in args.task or "vace" in args.task:
    args.sample_shift = 16
# else (T2V): 5.0
```

This is the standard rectified-flow `shift = exp(s)` rescaling of the
noise schedule (s=log(5)≈1.61 → 5x at sigma=0.5).

---

## 9. Memory footprint at inference

Source: `Wan-Video/Wan2.1 README.md` claims, repo inference scripts
(`--offload_model`, `--t5_cpu`, `--dit_fsdp`, `--ulysses_size`,
`--ring_size`).

### Static (params + persistent state) bf16 weights only

| Component        | Params | bf16 size |
|------------------|--------|-----------|
| DiT 14B           | 14.29 B | 28.6 GB   |
| T5-XXL encoder    | ~4.8 B  | ~9.7 GB   |
| Wan-VAE           | ~125 M  | ~250 MB   |
| **TOTAL bf16**    | ~19.2 B | **~38.6 GB** |

### Activation-dominated (training / single-GPU inference)

For 14B at bf16, **no offload**: ≥ 80 GB HBM needed for 480p 16f.
The README states T2V-14B 480p is **infeasible on a single 4090
(24 GB)** without offload. T2V-1.3B 480p fits in **8.19 GB**.

### Practical inference recipes (from `generate.py` + README)

- **T2V-14B 480p on 24 GB consumer GPU:** not possible with full
  pipeline; needs `--offload_model --t5_cpu --dit_fsdp` and multiple
  GPUs, OR fp8 quant (LightX2V / DiffSynth), OR Ulysses sequence
  parallel with 2+ GPUs.
- **T2V-14B 720p 16 frames (56k tokens):** 8×H100 or 8×A100 with
  Ulysses/ring sequence parallel (`ulysses_size=8` shown in README).
- **T2V-1.3B 480p:** 8.19 GB on RTX 4090, ~4 min for 5s video.
- Activations scale as `B·N²·head_dim·num_layers` for self-attn.
  720p/16f ≈ 56k² ≈ 3.1 G attention matrix per layer × 40 layers
  (in bf16: ~250 GB just for the attention softmax matrices if
  materialized — FlashAttn varlen is mandatory).

---

## 10. Code structure — what to override for AsymFlow port

### Canonical repo classes (Wan-Video/Wan2.1)

| File / class                                | Purpose                          | Override point for AsymFlow? |
|---------------------------------------------|----------------------------------|-----------------------------|
| `wan/modules/model.py:WanModel`             | Main DiT backbone                | **YES — subclass / wrap** |
| `WanModel.__init__` (line 382)              | Builds everything                | Keep as is; build full Wan, then surgically replace I/O |
| `WanModel.patch_embedding` (line 456)       | `Conv3d(16, dim, (1,2,2))`       | **YES — AsymFlow input projection** |
| `WanModel.head` (line 320, 475)             | Output `Linear(dim→16·4)` + 2-vec AdaLN | **YES — AsymFlow output projection** |
| `WanModel.text_embedding` (line 458)        | T5 → dim MLP                     | Optional (depends on cond design) |
| `WanModel.time_embedding` + `time_projection` (line 462-4) | Time AdaLN       | Likely keep |
| `WanModel.blocks` (line 468)                | 40× `WanAttentionBlock`          | **YES if AsymFlow changes block internals** |
| `WanModel.freqs` buffer (line 480)          | RoPE table                       | Keep (or extend if new axes) |
| `WanModel.unpatchify` (line 584)            | `einsum 'fhwpqrc→cfphqwr'`        | Likely keep |
| `WanModel.forward` (line 493)               | Whole forward: pad, embed, loop, head | **YES — entry point** |
| `WanAttentionBlock` (line 238)              | SA → CA → FFN with 6-vec AdaLN    | Optional to subclass |
| `WanSelfAttention` (line 105)               | Full self-attn + RoPE            | Subclass if you need different pattern |
| `WanT2VCrossAttention` / `WanI2VCrossAttention` (162, 187) | CA to T5 / CA to T5+CLIP | Subclass for new cond modality |
| `WanRMSNorm` (line 73), `WanLayerNorm` (92) | Normalizations                   | Keep |
| `MLPProj` (line 350)                        | I2V CLIP image → dim             | Optional |
| `wan/modules/attention.py:flash_attention`  | FA2/FA3 varlen dispatch          | Keep (or replace with SDPA fallback) |
| `wan/modules/vae.py:WanVAE`                 | Video VAE                        | Keep (or replace with your own pixel VAE) |
| `wan/modules/t5.py:T5EncoderModel`          | umT5-XXL wrapper                 | Keep or replace encoder |
| `wan/configs/wan_t2v_14B.py`                | All hyperparams                  | Read-only reference |

### Equivalent diffusers classes (cleaner API)

```
diffusers/models/transformers/transformer_wan.py
  WanAttnProcessor       - attention call (dispatch_attention_fn)
  WanAttention           - fused/unfused QKV + added_kv projections
  WanImageEmbedding      - I2V CLIP image feature projector
  WanTimeTextImageEmbedding - time + text + image embeds
  WanRotaryPosEmbed      - 3D RoPE module
  WanTransformerBlock    - 1 block: SA → CA → FFN
  WanTransformer3DModel  - the full DiT
```

Diffusers version names the I/O projections `patch_embedding` and
`proj_out` (vs Wan's `patch_embedding` and `head.head`); same
semantics.

### Recommended AsymFlow port strategy

The cleanest minimal-touch port for "use Wan 2.1 14B as a backbone
for a custom pixel-space video model":

1. **Load pretrained `WanModel`** (`model_type='t2v'` for T2V, or
   `'i2v'` if you keep image conditioning) from
   `Wan-AI/Wan2.1-T2V-14B` weights. All keys already exist and
   match — verified by HF `config.json` + Wan repo constants.

2. **Override `WanModel.patch_embedding`** — replace `Conv3d(16,
   dim, (1,2,2))` with a custom `AsymFlowPatchEmbed` that ingests
   whatever AsymFlow's input representation is (e.g., extra channels
   for pixel-space conditioning, optical flow, depth). Keep output
   dim = 5120.

3. **Override `WanModel.head.head`** — replace `Linear(5120, 16·4)`
   with `Linear(5120, your_out_channels · 4)` to match AsymFlow's
   output. Keep `head.modulation` and `head.norm` to preserve AdaLN
   conditioning.

4. **Inject new conditioning by overloading `WanModel.forward`** —
   the current signature accepts `x: List[Tensor[Ci,F,H,W]]`,
   `t`, `context: List[Tensor[L,C]]`, optional `clip_fea`, `y`. To
   add new conditioning, either (a) prepend tokens to `context` or
   (b) wrap with a forward that calls a small `AsymFlowConditioner`
   then `super().forward(...)`.

5. **Reuse all 40 blocks, RoPE, T5 encoder, Wan-VAE verbatim** —
   they are not Wan-specific architecture, they're standard
   DiT-with-cross-attn + flow-matching + KL-VAE + T5.

6. **Keep `WanVAE` for pixel <-> latent conversion** if AsymFlow
   still operates in Wan-VAE latent space; replace with your own
   VAE if you want pure pixel-space.

7. **Flow-matching training/inference:** borrow
   `wan/utils/fm_solvers_unipc.py` (UniPC) and `fm_solvers.py`
   (DPM-Solver) directly; the `FlowMatchingScheduler` API mirrors
   `diffusers.FlowMatchEulerDiscreteScheduler` and diffusers ships
   its own equivalent (`scheduling_flow_match_euler_discrete.py`).

### File map for the port

```
C:\Users\Goblin.DESKTOP-7AP7J8S\wan21-deepdive\
├── WAN21_14B_DEEPDIVE.md              <- this file
├── diffusers_transformer_wan.py      <- diffusers reference impl
├── diffusers_attention_processor.py  <- FA dispatch
├── wan/
│   ├── configs/
│   │   ├── wan_t2v_14B.py             <- 14B hyperparams (dim=5120 etc)
│   │   ├── wan_t2v_1_3B.py
│   │   ├── wan_i2v_14B.py             <- +CLIP image cond
│   │   └── shared_config.py          <- text_len=512, fps=16, t5=umt5
│   └── modules/
│       ├── model.py                  <- WanModel (the backbone)
│       ├── attention.py              <- FA2/FA3 varlen dispatch
│       ├── t5.py                     <- umT5-XXL wrapper (dim=4096, 24 enc layers)
│       ├── vae.py                    <- WanVAE (16ch, 4×8×8 stride, causal)
│       └── clip.py                   <- CLIP for I2V
└── generate.py                       <- inference CLI defaults
```
# PiD v1.5 qwenimage Architecture Investigation — 4× SR Head Strippability

Subagent report (2026-07-14).

## TL;DR

**The PiD v1.5 qwenimage model has NO learned 4× upsampling head.** The network
operates in pixel space at the full output resolution (e.g. 2048×2048 for the
`2kto4k` variant; LDM runs at 512×512). Everything in the network is
stride-1, 3×3 conv or per-pixel/patch linear — confirmed by exhaustive tensor
shape inspection of the released checkpoint.

The "4× SR" label is purely a **data-pipeline convention**:
1. LDM samples at `H_ldm = H_out / 4`, `W_ldm = W_out / 4`.
2. `lq_latent` (the LDM output) is fed to the network.
3. `_resolve_inference_image_size` (pid_model.py:317-339) sets
   `img_h = latent_grid * vae_scale * sr_scale` → network runs at
   `latent_grid * 8 * 4` = upscaled.

For a **1× video-decoder use case** (where you already have Wan 2.1 latents at
full 480p pixel resolution), the right call is to **NOT use the released
checkpoint at all**. Subclass `PixDiT_T2I` (the parent of `PidNet`) and drop the
LQ projection entirely. All weights in the released `model_ema_bf16.pth` are
trained with the `LQProjection2D`-conditioned controlnet pathway on a fixed
4×-style data pipeline; they will likely misfire when fed identical-resolution
LQ + HQ.

## 1. Architecture dataflow (text diagram)

```
input x_t : [B, 3, H, W]      (H = output resolution, e.g. 2048)
                              ↓
s_embedder : unfold+Linear → [B, L, 1536]  (L = (H/16)²)
                              ↓
y_embedder : linear           → caption tokens [B, L_txt, 1536]
                              ↓
patch_blocks : 14× MMDiT with ControlNet-style LQ injection
   ┌─────────────────────────────────────────────────────────────────┐
   │  LQ injection (one per lq_interval=2 blocks → 7 gates total)    │
   │  s_main = sigma_aware_gate(s_main, lq_feat[i], sigma)            │
   └─────────────────────────────────────────────────────────────────┘
                              ↓
s = silu(t_emb + s_main)     → [B, L, 1536]
                              ↓
pixel_embedder : per-pixel Linear → [B*L, 16, 16]  (pixel tokens, full res)
                              ↓
pixel_blocks : 2× PiTBlock   (note: NO temporal dim, just 2D attention)
                              ↓
final_layer : RMSNorm + Linear → [B*L, 256, 3]    (3 RGB chans per pixel)
                              ↓
fold((H, W), 16, 16)         → [B, 3, H, W] = PIXEL SPACE, SAME H,W AS INPUT
```

**The 4× SR happens ZERO times in this diagram.** The `PidNet.forward` returns
an image at `(H, W)` — exactly the spatial size of the noisy input `x_t`.

## 2. Where "4× SR" actually lives

Three places — none are learned upsampling inside the network:

### 2.1 Inference-time image_size resolution

`pid/_src/inference/decoder.py:122-123`:
```python
lq_h, lq_w = baseline_01.shape[-2], baseline_01.shape[-1]
infer_image_size = (lq_h * args.scale, lq_w * args.scale)   # scale=4 → ×4
```

`pid/_src/models/pid_model.py:317-339` (`_resolve_inference_image_size`):
```python
sr_scale = int(net.sr_scale)                                # 4
img_h = int(lq_latent.shape[-2]) * vae_scale * sr_scale    # zH * 8 * 4
img_w = int(lq_latent.shape[-1]) * vae_scale * sr_scale
```

So `lq_latent` shape `[B, 16, 64, 64]` → output `[B, 3, 2048, 2048]`.

The PiD model runs in PIXEL SPACE at 2048×2048, and the input `x_t` is a noisy
2048×2048 *image* (not a latent). Sampling initialization is per-pixel at full
output resolution:

`pixeldit_model.py:636` (in `generate_samples_from_batch`):
```python
z = torch.randn(B, 3, img_h, img_w, device="cuda", generator=gen)
```
This noise `z` is also at 2048×2048 — confirming PiD samples live in pixel
space at the post-SR resolution.

### 2.2 LQ projection's "z_to_patch_ratio" math (alignment only, no learnable upsample)

`pid/_src/networks/lq_projection_2d.py:289-310`:
```python
z_to_patch_ratio = (sr_scale * effective_lsdf) / patch_size
#                  = (4 * 8) / 16 = 2.0     for v1.5 qwenimage
if z_to_patch_ratio > 1:                 # latent is smaller than patch grid
    z_aligned = F.interpolate(lq_latent, size=(pH, pW), mode="nearest")
elif z_to_patch_ratio == 1:             # exact alignment
    z_aligned = lq_latent
else:                                   # latent is bigger (1×-style)
    fold into channels (PixelUnshuffle-like)
```

For `sr_scale=4`, `lsdf=8`, `patch_size=16` → ratio=2.0 → NEAREST 2× UPSAMPLE.
The Conv layers after this nearest are all stride-1, 3×3 (no learned SR).

**This nearest interpolate is the ONLY place spatial upsampling happens for
qwenimage v1.5**, and it operates only on the conditioning `lq_latent` going
INTO the network, not on the network's output.

### 2.3 LQProjection2D lays out the latent in patch-grid form

With `lq_latent: [B, 16, 64, 64]` → nearest 2× → `[B, 16, 128, 128]` →
`Conv2d(16, 512, 3×3)` (this is `latent_proj.0`) + `Conv2d(512, 512, 3×3)`
+ 4 ResBlocks → feature map `[B, 512, 128, 128]` = `[B, 512, pH, pW]` where
`pH = 2048/16 = 128`.

## 3. Input/output shapes (verified from released checkpoint)

`PixDiT_T2I.__init__` (`pixeldit_official.py:1156-1224`):
- `in_channels = out_channels = 3`
- `hidden_size = 1536`, `pixel_hidden_size = 16`, `patch_size = 16`
- `patch_depth = 14`, `pixel_depth = 2`
- `final_layer = FinalLayer(pixel_hidden_size=16, out_channels=3)`

Checkpoint tensor shapes (verified by loading
`PiD_res2kto4k_sr4x_official_flux_distill_4step/model_ema_bf16.pth`, same
PidNet arch as qwenimage — only the upstack text encoder differs):

```
net.pixel_embedder.proj.weight    -> (16, 3)                    per-pixel RGB→16-dim
net.pixel_embedder.proj.bias      -> (16,)
net.final_layer.linear.weight     -> (3, 16)                    16-dim pixel → RGB
net.final_layer.linear.bias       -> (3,)
net.final_layer.norm.weight       -> (16,)
net.s_embedder.proj.weight        -> (1536, 768)                unfold(3, 16, 16) → 768 = 3·16²
net.s_embedder.proj.bias          -> (1536,)
```

**Every 4-D weight in the checkpoint is 3×3 stride-1.** No ConvTranspose2d, no
PixelShuffle, no upsample conv, no kernel ≠ 3.

## 4. LQ projection structure (qwenimage v1.5 config)

For qwenimage v1.5:
- `lq_in_channels=0` (image branch DISABLED)
- `lq_latent_channels=16` (Wan 2.1 VAE latent)
- `lq_hidden_dim=1024` (vs 512 in v1)
- `lq_num_res_blocks=4`
- `sr_scale=4`, `latent_spatial_down_factor=8`, `lq_latent_unpatchify_factor=1`

`latent_proj` (Sequential: Conv → SiLU → Conv → 4 ResBlocks):
```
latent_proj.0 : Conv2d(16, 1024, 3×3, stride=1)       # align 16 → 1024
latent_proj.1 : SiLU
latent_proj.2 : Conv2d(1024, 1024, 3×3, stride=1)     # project
latent_proj.3 : ResBlock(1024)         (Conv2d, Conv2d inside)
latent_proj.4 : ResBlock(1024)
latent_proj.5 : ResBlock(1024)
latent_proj.6 : ResBlock(1024)
```
The task says "9 Conv2d layers" but it's actually 10 Conv2d tensors
(`latent_proj.{0, 2, 3.block.{2,5}, 4.block.{2,5}, 5.block.{2,5}, 6.block.{2,5}}`).
I confirmed all 10 exist in the v1.5 architecture definition.

After Conv/ResBlock, `latent_proj` outputs `[B, 1024, pH, pW]`. Then:
- 7 `output_heads` (Linear 1024→1536) for controlnet injection
- 1 `pit_head` (Linear 1024→1536) for PiT (pixel-block) injection
- 7 `gate_modules` (SigmaAwarePerTokenGate, content_proj 3072→1)

**The role**: convert the noisy LQ `lq_latent: [B, 16, 64, 64]` into
per-patch conditioning tokens, one set per transformer block, that get
GATED INTO the main DiT stream (added with a sigmoid-gated scalar that
depends on the residual noise level σ). It does NOT do the 4× upsample
of the network's output.

## 5. Can the "SR head" be stripped?

**There is no "SR head" in the network itself** to strip. What can be
stripped (without losing the network's residual-denoising power):

1. **Skip the LQ projection controlnet injection entirely.** Call
   `PidNet.forward(x, t, y, lq_latent=None, lq_video_or_image=None,
   degrade_sigma=None)`. Output will still be at full output resolution.
   But: the model was trained with LQ conditioning (75% noisy / 25%
   clean σ=0). Without that conditioning, fidelity will collapse
   (the gate output heads are zero-init; see below).

2. **Use a 1× sr_scale.** The math is parameterised; setting
   `sr_scale=1` makes `z_to_patch_ratio = 8/16 = 0.5`, triggering the
   "fold into channels" path. But the LQ conv weights were trained for
   ratio=2.0 (1024-dim projections on nearest-upsampled input). With
   ratio=0.5 you'd be feeding the network `Conv2d(64, 1024, 3×3)`
   features instead of the trained distribution.

3. **Replace LQProjection2D with a different conditioning head** that
   projects Wan latents AT THE SAME RESOLUTION as the patch grid
   (ratio=1.0). This is the cleanest path for a 1× video decoder.

The "video decoder" use case (Wan 2.1 → Wan 2.1 latent → 480p pixels with
HF detail) means the LATENT is already at the right size. So:
- `output_image_size = latent_grid * vae_scale * 1 = latent_grid * 8`
- `z_to_patch_ratio = (1 * 8) / 16 = 0.5` (fold path)
- NO nearest interpolate needed

## 6. Recommended approach

**Do not use the released `PiD_v1pt5_res2kto4k_sr4x_official_qwenimage_distill_4step/model_ema_bf16.pth` for 1× video decoding.**

Why:
- It's distilled at 4-step Euler with fixed σ schedule `{0.999, 0.866, 0.634, 0.342}`
  tied to the Qwen/Wan image-decode use case (σ_max≈0.8).
- Gate heads are zero-init (so without LQ-injection training, the path
  was a no-op — the weights reflect the SR-data training only).
- Image-size ref is 2048 (NTK-aware RoPE tuned to 2048×2048).

**Recommended**:

1. Subclass `PixDiT_T2I` (the parent of `PidNet`,
   `pid/_src/networks/pixeldit_official.py:1123`) directly, NOT `PidNet`.
2. Drop `LQProjection2D` entirely (or replace with a thin,
   1×-projection head trained for video).
3. Inflate Conv2d→Conv3d in `pixel_embedder`, `final_layer`,
   `pixel_blocks`, `lq_proj` (if you keep it) — same as DEEP_DIVE_VIDEO_PID.md
   says.
4. Initialize from a VIDEO-AWARE checkpoint if available; if not, then
   from the released image ckpt ONLY for the per-pixel 2D components
   (pixel_embedder, final_layer, s_embedder.proj) — verified-shape:
   `(16, 3)`, `(3, 16)`, `(1536, 768)`. The transformer blocks are
   2D-attention-based so they're OK for video frame-by-frame init but
   NOT for joint temporal reasoning.

## 7. Concrete code paths

| Concern | File:line |
|---|---|
| PidNet entry class | `pid/_src/networks/pid_net.py:27` (inherits `PixDiT_T2I`) |
| Pure T2I base (no LQ) | `pid/_src/networks/pixeldit_official.py:1123` (`PixDiT_T2I`) |
| Forward returns (H, W)-shape image | `pid/_src/networks/pid_net.py:540-550` (`final_layer → fold((H,W))`) |
| pixel_embedder (per-pixel Linear) | `pid/_src/networks/pixeldit_official.py:377-442` |
| final_layer (per-pixel Linear) | `pid/_src/networks/pixeldit_official.py:340-349` |
| LQ projection class | `pid/_src/networks/lq_projection_2d.py:144-422` |
| LQ conv layers (9-10 stride-1) | `pid/_src/networks/lq_projection_2d.py:312-333` |
| LQ z_to_patch_ratio math | `pid/_src/networks/lq_projection_2d.py:289-310` |
| 4× size inference resolution | `pid/_src/models/pid_model.py:317-339` (`_resolve_inference_image_size`) |
| Pixel-space z noise init | `pid/_src/models/pid_model.py:694` (`z = torch.randn(B, 3, img_h, img_w)`) |
| Inference scale=4 entry | `pid/_src/inference/decoder.py:122-123` |
| v1.5 qwenimage config defaults | `pid/_src/configs/pid/experiment_2kto4k_v1pt5/shared_config.py:28-46` |
| Net config (sr_scale=4, lq_hidden_dim=1024) | `pid/_src/configs/common/defaults/net.py:62-74` |
| sigma-aware gate | `pid/_src/networks/lq_projection_2d.py:48-102` |
| Zero-init output heads | `pid/_src/networks/lq_projection_2d.py:377-413` |

## 8. Direct answers to task questions

- **Q: Where does the 4× SR happen?**
  **A:** NOT inside the learned network. It's an inference convention:
  the network is sampled in pixel space at H×W and the LDM runs at H/4 × W/4.
  The only learned "spatial alignment" is a `F.interpolate(nearest)` on the
  16-channel LQ latent that targets the patch grid (no learned params).

- **Q: Is the SR head separable?**
  **A:** There is no SR head to separate. The conditioning comes from
  `lq_proj`, and the OUTPUT always lives at H×W = same spatial size as
  the noisy input `x_t`. You CAN strip the `LQProjection2D` injection
  (call `net.forward(x, t, y, lq_latent=None, ...)` ) and still get an
  image at the same H×W — but at severe fidelity cost.

- **Q: Which model class to subclass for a 1× video decoder?**
  **A:** `PixDiT_T2I` directly (`pid/_src/networks/pixeldit_official.py:1123`),
  not `PidNet`. Skip `LQProjection2D`. For video, additionally add Conv3d
  inflation in `pixel_embedder`, `final_layer`, and `pixel_blocks`.

- **Q: Alternative if not separable?**
  **A:** It's separable. But the released EMA checkpoint is 4×-trained;
  we need either a fresh 1× distil run (~2-3 days on 8×H100) or
  continued finetune for ~10k iters on Wan 2.1 video pairs at 480p.

## 9. Files I read / wrote

Read:
- `pid/_src/models/pid_model.py` (903 lines)
- `pid/_src/models/pixeldit_model.py` (879 lines)
- `pid/_src/networks/pid_net.py` (560 lines)
- `pid/_src/networks/pixeldit_official.py:340-1282` (FinalLayer, PixelTokenEmbedder, PixDiT_T2I)
- `pid/_src/networks/lq_projection_2d.py` (637 lines)
- `pid/_src/inference/from_ldm.py` (253 lines)
- `pid/_src/inference/decoder.py` (190 lines)
- `pid/_src/configs/pid/experiment_2kto4k_v1pt5/qwenimage.py` (50 lines)
- `pid/_src/configs/pid/experiment_2kto4k_v1pt5/shared_config.py` (46 lines)
- `pid/_src/configs/common/defaults/net.py` (105 lines)
- `DEEP_DIVE_VIDEO_PID.md` (excerpts, prior work)
- Checkpoint `/Documents/comfy/ComfyUI/models/nvidia_pid/checkpoints/PiD_res2kto4k_sr4x_official_flux_distill_4step/model_ema_bf16.pth` (verified tensor shapes).

Wrote:
- This file: `C:\Users\Goblin.DESKTOP-7AP7J8S\research\pid\PiD\FINDINGS_v1pt5_qwenimage_arch.md`

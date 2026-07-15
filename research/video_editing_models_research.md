# Open Video Editing Models for RTX 3090 (24GB) — Research Summary

**Target hardware:** RTX 3090 (24 GB GDDR6X, Ampere)
**Use cases:** style transfer, color grading, inpainting, object removal
**Constraints:** uncensored, no plastic/waxy look, runs locally in 24 GB

---

## 1. Comparison Table — Editing Models

| Model | Params | VRAM @ 480p / 16f (3090) | License | Censorship | Edits Supported | Plastic Look? |
|---|---|---|---|---|---|---|
| **Wan 2.1 VACE-1.3B** | 1.3B | ~8.2 GB (8 GB cards OK) | Apache 2.0 | Mild built-in filter; bypass via LoRA / abliteration | Ref→Vid, V2V, masked V2V, outpaint, depth/pose/edge control, swap-anything, move-anything | Yes — speckles/polka-dots, **fixable** with `spacepxl/Wan2.1-VAE-upscale2x` |
| **Wan 2.1 VACE-14B** | 14B | ~24 GB FP8, fits with offload otherwise 40 GB+ | Apache 2.0 | Same as 1.3B | Same suite, much higher quality | Same VAE artifacts, same fix |
| **VACE-LTX-Video-0.9** | ~2B (LTX core) | 8–10 GB | Apache 2.0 (LTX-Video) | Filtered; LTX is more censored than Wan | Same VACE tasks, lower quality than Wan-VACE | Less speckle than Wan, but softer/waxy output |
| **CogVideoX (2B / 5B)** — `THUDM`/`zai-org` | 2B / 5B | 2B: ~6 GB / 5B: ~16 GB | Apache 2.0 | Filtered, "no LoRA-free NSFW" | Mostly **I2V** + video continuation, not real "editing" | Less plastic than Wan, but weaker instruction-following |
| **HunyuanVideo-Edit** | not officially released as a separate edit model (Sept 2025); bundled into HunyuanVideo | n/a | Tencent license (open) | Filtered | Limited public docs | — |
| **HunyuanVideo 1.5** (not Edit, but relevant) | 8.3B | 14 GB min | Tencent open license | Filtered but lighter | T2V/I2V; no dedicated edit model yet | Reports: "beats Wan2.2 and Kling2.1 in clarity" |
| **Wan2.2-Animate-14B** | 14B | 24 GB (FP8 / offload) | Apache 2.0 | Filtered | **Character animation + character replacement** (not general editing) | Wan VAE artifacts apply, same fix |
| **DragAnything** (ShowLab, ECCV 2024) | ~2.7B (built on SVD) | ~10–12 GB @ 320×576 / 14 frames | Apache 2.0 (per repo) | None (no text prompt path) | Trajectory-based motion control only — not generic editing | No "plastic" — uses SVD decoder |
| **DiffuEraser** (Alibaba Tongyi, Jan 2025) | ~1.5B (SD-1.5 backbone) | 8–10 GB | Research-only (Alibaba license) | N/A (inpainting) | Video object removal / inpainting only | SD-1.5 waxiness |
| **ProPainter** (ICCV 2023) | propagation+Transformer, ~70M params | 4–6 GB | S-Lab / academic | N/A (no text) | Object removal, video completion, outpainting | No diffusion — no plastic, classic inpainter |
| **ROSE** (Aug 2025) | DiT-based, not fully released | Unknown | Paper only (arxiv 2508.18633) | N/A | **Object removal with side-effect handling** (shadows, reflections) | Best-in-class for removal quality |
| **DiffBIR** | SD-based | 8 GB | Apache 2.0 | N/A | Blind image restoration (image only, not video natively) | Mild |
| **LTX-2 / LTX-Video 0.9.5 → 2.3** | 2B (LTX-Video); newer LTX-2 undisclosed | 8–12 GB | OpenRAIL (Lightricks) | Filtered but abliterations exist on CivitAI | T2V/I2V + LTX-2.3 added R2V LoRAs for ref-based style/inpaint | LTX VAE acts as denoising decoder — generally cleaner than Wan |
| **Stable Video Diffusion + edits** | SVD 1.1B / SVD-XT 1.1B | 8–10 GB | SAI Community License | Filtered | Multi-frame editing via fine-tuning; tight ecosystem | Mildly waxy |

---

## 2. Headline Recommendation for 3090 (24 GB)

### Primary pick: **Wan 2.1 VACE-1.3B** + **`spacepxl/Wan2.1-VAE-upscale2x`**

**Why:**
- **Apache 2.0** — fully permissive commercial use.
- **8.2 GB VRAM** at 480×832, 16 frames — leaves 16 GB headroom for inpainting masks, ControlNet, longer sequences.
- All-in-one: reference-to-video, V2V, masked V2V, outpaint, depth/pose/edge, swap-anything, move-anything.
- **No built-in image filter** (text-to-video has weak content moderation; easy to abliterate via LoRA).
- **Plastic fix exists**: `spacepxl/Wan2.1-VAE-upscale2x` is a decoder-only finetune that explicitly removes Wan speckles/grain and doubles resolution in decode — this is the single biggest mitigation for the "waxy VAE" problem.
- For higher fidelity when 16+ GB free: **Wan2.1-VACE-14B in FP8** fits comfortably in 24 GB.

### Pairing for object removal / inpainting: **ProPainter** (lightweight, no diffusion waxy look) or **ROSE** (when quality matters more than speed)

### For pure style transfer / color grading: V2V mode of VACE-1.3B with reference image (FusioniX workflow)

---

## 3. The "Plastic / Waxy VAE" Problem — Solutions

### Root cause
Latent video diffusion models encode video → small latent → decode back. The decoder (often an inflated image VAE with a temporal head) loses high-frequency detail and smooths over skin, hair, textures → "plastic" look. Wan 2.1 specifically also exhibits **speckles / polka-dot grain** due to its VAE's quantization residuals.

### Fixes (ranked by ease / impact)

1. **`spacepxl/Wan2.1-VAE-upscale2x` (HuggingFace)** — *the killer app for Wan.*
   - Decoder-only finetune of the Wan 2.1 VAE.
   - **Integrated 2× upscaling** in the decoder.
   - Primary purpose: kill Wan speckles/polka-dots/grain.
   - Pairs with `spacepxl/ComfyUI-VAE-Utils` custom node (loads via ComfyUI Manager).
   - Also works with Qwen Image (same VAE architecture).
   - Sharper than Wan FP32 VAE in user reports.

2. **Diffusion-decoder replacement: NVIDIA PiD (Pixel Diffusion)** (announced 2026).
   - Plug-and-play diffusion decoder replacing VAE/RAE.
   - Latent → 2K–4K pixels directly. Designed to eliminate VAE smoothing artifacts.
   - Still maturing; community wrappers emerging.

3. **LTX-Video-style denoising decoder** — LTX-Video's VAE is *conditioned on diffusion timestep* and takes over the last sampling step. Conceptually a denoising decoder that recovers high-frequency detail. This is why LTX outputs look less plastic than Wan, but the model is otherwise weaker.

4. **Two-stage: render + Real-ESRGAN / SeedVR / SUPIR upscale**
   - Universal fallback. Apply video-Restoration model after VACE decode.
   - SeedVR (ByteDance) and SUPIR are state-of-art for high-frequency detail recovery.

5. **`VideoVAE+` (ICCV 2025)** — cross-modal joint video-image VAE training. Decouples spatial and temporal compression to reduce motion blur and detail distortion. Open weights on HF (`VideoVerses/VideoVAEPlus`). Worth trying as a swap-in.

6. **Generic VAE fine-tune (Leminhbinh0209/FinetuneVAE-SD)** — recipe to fine-tune any SD-style VAE with L1+LPIPS then L2+LPIPS — relevant if you want to do your own VAE finetune on Wan.

### Practical stack for 3090, no plastic
```
Wan2.1-VACE-1.3B  →  decode with Wan2.1-VAE-upscale2x (2× upscale, kills speckles)
                                          ↓
                              optional: SeedVR / SUPIR final pass
```
This is the canonical "uncensored + clean" workflow on CivitAI / Reddit (mid-2025 → 2026).

---

## 4. Uncensored / Abliterated Forks

The honest state: **there is no clean "Wan 2.1 NSFW-pretrain" release**. Censorship in video models is much weaker than in image models (no NSFW classifier baked in), but Wan 2.1's T2V does refuse some concepts.

Practical workarounds (community consensus, mid-2026):

1. **LoRA training** — the dominant path. Train short-clip LoRAs on uncensored video (CivitAI guides exist; `WAN 2.1 Video Lora Training Guide` on Civitai by kijai/others). Wan is "extremely censored" out of the box — you need action data.
2. **Image-to-video (I2V) instead of T2V** — bypass text moderation entirely by feeding your own start frame. Most VACE workflows default to I2V.
3. **VACE-14B reference-to-video with an explicit reference image** — even less filtering because the prompt has less work to do.
4. **Use `NSFW-API/NSFW_Wan_1.3b`** (community fork on HF) — exists but is lightly documented; primarily a finetuned 1.3B variant.
5. **Locally Uncensored app** (`PurpleDoubleD/locally-uncensored`) bundles Wan 2.1 / 2.2 alongside abliterated LLMs and image models — useful as a one-stop stack.
6. **LTX-2.3** has been discussed as potentially less filtered than Wan, but multiple Reddit threads (r/StableDiffusion "Can LTX 2.3 do uncensored spicy videos") conclude Wan still wins for NSFW generation.

### Image-mod only abliterations
- `Krea 2` VAE fix thread on Reddit (Jun 2026) discusses PiD replacing terrible VAEs — same toolkit applies to video.

---

## 5. Model-by-Model Practical Notes

### Wan 2.1 VACE (1.3B vs 14B)
- **1.3B** is genuinely the workhorse: 8.2 GB VRAM, runs on RTX 3060 12 GB, leaves massive headroom on 3090.
- **14B** is the quality ceiling but eats 40 GB+ in FP16; on 3090 use **FP8** or offload (--offload_model, --t5_cpu).
- Both support the full VACE task suite: ref-to-video, video-to-video, masked V2V, outpainting, depth/pose/edge control, swap-anything, move-anything. Resolutions up to 720p with appropriate flags.
- 1.3B-Preview was released March 31, 2025; full 1.3B + 14B released May 14, 2025 (ICCV 2025 paper).

### CogVideoX (the "CogVideoX-Edit" question)
- **There is no canonical "CogVideoX-Edit" model.** CogVideoX is a generation model (T2V/I2V/video-continuation), not a dedicated editing model.
- "FunEditor" doesn't appear as a major open release — likely refers to LLM-agent video editing papers, not a released model.
- Use CogVideoX-2B (~6 GB) only for **I2V continuation** if you want continuity. For real editing, prefer VACE.

### DragAnything (ShowLab, ECCV 2024)
- Trajectory-based motion control. You draw a line on a frame; the model animates the dragged entity.
- Best for: precise object motion control, "move the person left" style edits.
- Not for: style transfer, color grading, generic edits.
- Built on SVD; no text censorship; mild plastic look from SVD decoder.

### DiffuEraser (Alibaba Tongyi, Jan 2025)
- Video inpainting built on Stable Diffusion 1.5.
- Higher detail than ProPainter's transformer approach; lower temporal coherence.
- ~8 GB VRAM, runs comfortably on 3090.

### ProPainter (ICCV 2023)
- Not a diffusion model — uses optical-flow propagation + Transformer.
- Tiny (~70M params), 4–6 GB VRAM, very fast.
- Best for: clean, sharp object removal without diffusion artifacts (no plastic!).
- Limitation: struggles with large masks / complex backgrounds.

### ROSE (Remove Objects with Side Effects, Aug 2025)
- Built on a diffusion transformer; handles **shadows, reflections, smoke** that other erasers miss.
- Best-in-class removal quality as of late 2025.
- Model not fully publicly released yet (paper + dataset only).

### DiffBIR
- Image-only restoration, but trivially extendable to per-frame video.
- Useful as a final polish pass for old / compressed video.
- Mild plastic look vs SOTA, but fast and reliable.

### Wan2.2-Animate-14B (Sept 2025)
- Specialized: character animation + replacement, not general editing.
- 24 GB VRAM minimum on RTX 4090; tight on 3090 but workable with offload.
- Same Wan VAE artifacts → pair with `Wan2.1-VAE-upscale2x`.

### LTX-Video / LTX-2 (Lightricks)
- LTX-Video (2B): clean VAE w/ denoising decoder; 8 GB VRAM.
- LTX-2.x: latest, larger; less public detail on params.
- LTX-2.3 added R2V LoRAs for reference-based style transfer / inpainting — the closest competitor to VACE on speed.
- More restricted than Wan for NSFW (per r/StableDiffusion consensus).

---

## 6. Stack Recommendation (3090, 24 GB)

```
Primary editing model:  Wan2.1-VACE-1.3B  (Apache 2.0, 8.2 GB VRAM)
    ↳ For highest quality:  Wan2.1-VACE-14B in FP8 (24 GB fits)
VAE upgrade:            spacepxl/Wan2.1-VAE-upscale2x (kills speckles, 2× upscale in decode)
Object removal:         ProPainter  (cheap, sharp)  OR  DiffuEraser (detailed)
Optional polish:        SeedVR / SUPIR (high-freq detail recovery)
Uncensoring:            I2V mode + LoRA training on community datasets
Restoration:            DiffBIR (per-frame) + RIFE for temporal interp
```

This stack hits all four constraints (3090 VRAM, uncensored via I2V/LoRA, editing capability, no plastic via VAE-upscale2x) with open licenses throughout.

---

## 7. Sources

- Wan-AI/Wan2.1-VACE-14B & Wan2.1-VACE-1.3B — huggingface.co/Wan-AI
- ali-vilab/VACE GitHub & UserGuide (ICCV 2025)
- spacepxl/Wan2.1-VAE-upscale2x (HF) + ComfyUI-VAE-Utils
- sczhou/ProPainter GitHub (ICCV 2023)
- lixiaowen-xw/DiffuEraser (arXiv 2501.10018)
- ROSE paper (arXiv 2508.18633)
- DragAnything paper (arXiv 2403.07420)
- showlab/DragAnything GitHub
- wan-animate/wananimate GitHub + humanaigc.github.io/wan-animate
- Tencent-Hunyuan/HunyuanVideo-1.5 GitHub (arXiv 2511.18870)
- Lightricks/ltx-video GitHub
- NVIDIA PiD pixel diffusion decoder (2026)
- VideoVAE+ (ICCV 2025) — VideoVerses/VideoVAEPlus
- "Toward Lightweight and Fast Decoders" (arXiv 2503.04871)
- Improved Video VAE (arXiv 2411.06449)
- r/StableDiffusion threads: "Can LTX 2.3 do Uncensored Spicy Videos", "Krea 2 terrible VAE / PID fix"
- CivitAI: WAN 2.1 Video LoRA Training Guide
- Locally Uncensored desktop app
- Apatero / LocalAIMaster / Spheron 2026 model comparisons

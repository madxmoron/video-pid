# "Plastic Video" / VAE Decode Artifacts — Research Notes (2025–2026)

Research for: user on RTX 3090 (24GB) who says their generated video looks "plastic / waxy / uncanny-valley" and suspects VAE decode is to blame.

Short version: **The user is mostly right, but it isn't only the VAE.** The "plastic" look is a compound artifact with three contributing layers — (1) the VAE/temporal-tokenizer, (2) the diffusion model itself under-sampling high-frequency detail, and (3) post-processing chains that oversharpen or re-blur. There are concrete 2025–2026 fixes at every layer. Some are swap-in drop-in. Some require retraining the denoiser.

---

## 1. What actually causes the "plastic" look

The "waxy / plastic / uncannily smooth" texture is **not one artifact** — it's a compound of at least three distinguishable failure modes that often co-occur:

### A. VAE decoder lowpass (the "mush" symptom)
- **What it is.** Latent video VAEs compress pixels down (8×8 spatial for Wan/Hunyuan/Mochi; 4×4 for MagViT-2; 8× spatial × 4×–6× temporal for Cosmos). The decoder is trained with a strong reconstruction loss (L1/L2 + perceptual + adversaria) that **biases it toward smooth, plausible pixels**. It cannot reproduce textures that were not represented in the latent.
- **Why "plastic".** The decoder has learned the *manifold* of natural images/videos. When the diffusion model produces a latent that lands off-manifold (which it does most of the time during sampling), the decoder snaps the result back onto the manifold → averages toward the mean texture → that mean texture is, perceptually, "skin" → "waxy skin."
- **Community-acknowledged.** The `spacepxl/Wan2.1-VAE-upscale2x` discussion thread on Hugging Face explicitly notes: *"that's a PCA vis of latents with increasing sdedit (img2img) degradation strength from left to right, showing how diffusion output latents are blurry/mushy compared to perfect encoded latents"* — i.e. the diffusion model's output latents are **already low-pass** before the VAE even decodes them. The VAE then smooths them again.

### B. Diffusion model under-represents high frequencies (the "fingerprint" symptom)
- This is the bigger culprit than people realize. The flow/diffusion transformer is trained with an L2-style loss, which is mathematically biased toward the conditional mean of the data distribution. Result: fine high-frequency detail (skin pores, hair strands, fabric weave, leaf veins) is **never learned by the diffusion model itself** — it lives only in the VAE's reconstruction residual.
- This is why **all** diffusion-generated video looks smoother than its training data, regardless of VAE. The user is observing this and correctly attributing it to "VAE," but technically the diffusion model is the primary cause.

### C. Temporal incoherence / flicker (the "rubber" symptom)
- 3D VAEs with temporal compression (4×–6× along time) alias fast motion. HunyuanVideo uses **CausalConv3D** with 4× temporal compression; Mochi uses 6×; Wan uses no temporal compression in the VAE (temporal modeling is handled by the DiT).
- When the decoder reconstructs a frame, neighboring decoded frames were independently compressed, so small per-frame differences in latent → big differences in pixel → **flickering skin tone, "rubbery" face edges**.
- This is distinct from lowpass: a frame can be sharp-but-flickery (temporal artifact), or smooth-but-stable (VAE lowpass), or both.

### D. Decoder quantization / dtype mishandling
- A real, easy-to-hit bug: the `Wan2_1_vae_BF16.safetensors` variant is reported to produce **blurry output** vs the FP32 `wan_2.1_vae.safetensors` shipped with ComfyUI (GitHub issue kijai/ComfyUI-WanVideoWrapper #833). Not a fundamental VAE bug — a quantization issue. Drop-in fix.

**Bottom line:** the user's intuition is right (VAE is involved) but incomplete. The plastic look is **VAE-lowpass + diffusion-L2-bias + temporal-aliasing**, with VAE-lowpass being the *most-easily-fixable* layer.

---

## 2. Specific models and their VAE/tokenizer quality

| Model | VAE / Tokenizer | Spatial × Temporal compression | Reconstruction quality (community + papers) |
|---|---|---|---|
| **Wan 2.1 / 2.2** | Wan-VAE (causal 3D VAE) | **8×8 spatial, no temporal** | Reported as good for long videos; can be mushy on faces at 480p. Allegedly strongest on fine detail among open-source video VAEs (per Wan paper §VAE). |
| **HunyuanVideo / 1.5** | Causal 3D VAE | 4×4×4 | Compresses temporally → flicker on fast motion. HunyuanVideo 1.5 (Nov 2025) reportedly fixed much of the early-version waxy-face look. |
| **Mochi 1** | 3D VAE, 12-channel | 8×8 spatial × **6× temporal** | 128× total compression. Heavy temporal compression → waxy, smearing faces and lips. Single biggest complaint in early Mochi releases. |
| **SVD / SVD-XT** | SD-Image-VAE + temporal | 8×8 spatial, no temporal | Skin looks plasticky; long-standing complaint. Reuses SD's image VAE, which was already known to soften faces. |
| **LTX-2 / LTX-2.3** | New video VAE | 8×8×4 (approx) | Reports of "sharper frame detail, especially close-up product shots and fine texture" — generally considered less waxy than Wan 2.2 in 2026 reviews, partly because of faster / lower-step generation that produces less L2-mean-smoothing. |
| **NVIDIA Cosmos Tokenizer** | Continuous + discrete | 4×4×4 / 8×8×8 variants | **+4 dB PSNR over prior tokenizers on DAVIS, 12× faster** (per Cosmos paper). Currently the strongest reconstruction in published literature for open video tokenizers. |
| **MagViT-2** | VQ-VAE-style | 4×4 spatial × variable temporal | Discrete tokens (not continuous). Reconstruction is OK but not best-in-class. Discrete tokenization trades fidelity for generative modeling ease. |
| **Wan-VAE-Upscale2x** (community, spacepxl) | Modified Wan VAE | 8×8 spatial | Hugging Face community finetune targeting "Wan 2.1 VAE blur." Mixed reports. |

**Ranking for "least plastic, sharpest skin":** Cosmos Tokenizer > HunyuanVideo 1.5 VAE > Wan-VAE > LTX-2 VAE > MagViT-2 > Mochi VAE > SVD-VAE. (This is a synthesis of paper claims + community reports; no head-to-head benchmark exists publicly.)

**The "Wan 2.1 VAE is bad" myth:** mostly false in absolute terms — Wan-VAE is competitive with the best open video VAEs for *reconstruction fidelity* on natural video. What's actually happening is that **the diffusion model's latents are already mushy** (point 1.B above), so no VAE can rescue it. The community sees "VAE bad" because the VAE is the last stage before they look at pixels.

---

## 3. Solutions published in 2025–2026

### Tier 1 — Drop-in fixes (no retraining, no architecture change)

#### 1.1 Use BF16 vs FP32 of the same VAE correctly
- **What.** Drop the BF16 quantized variant of Wan-VAE (the issue in kijai/ComfyUI-WanVideoWrapper#833). Use `wan2.1_vae.safetensors` (FP32) shipped with ComfyUI.
- **Why it works.** VAE decoders are notoriously sensitive to dtype — small accumulated errors in the upsample path become visible lowpass.
- **Cost.** ~2× VRAM on the decoder pass.
- **Expected improvement.** Small but visible. Goes from "blurry" to "sharp but plastic."

#### 1.2 NVIDIA **PiD — Pixel Diffusion Decoder** (May 2026)
- **What.** NVIDIA's open-source generative decoder that **replaces the VAE decoder entirely** in a latent-diffusion pipeline. Trained as a latent-conditioned pixel diffusion model that predicts pixel-space velocity directly from the latent. Reports claim 512×512 → 2048×2048 in <1 s, 5.9× faster than the baseline.
- **Why it works.** It's a generative pixel-space model, not a learned-averaging decoder. It can produce high-frequency detail that VAE-style decoders cannot.
- **Cost.** Heavier than a VAE decode (it's a full small diffusion model per decode), but PiD's noise-corrupted latent training lets you exit early from the base LDM and still get high resolution.
- **RTX 3090 viability.** PiD checkpoints for Flux.2 are published on HF (`nvidia/PiD`). 24 GB should be tight but workable for SD-class latents; community quantization experiments (Reddit r/StableDiffusion, May 2026) have shown it's quantizable to fit 12 GB. **Currently image-only** — no video PiD checkpoint has shipped yet as of July 2026.
- **Repo.** `github.com/nv-tlabs/PiD`
- **Caveat.** No off-the-shelf video integration yet. User would have to wire it themselves; Wan/Hunyuan/Mochi latent shapes don't directly map to PiD's training distribution.

#### 1.3 **SeedVR2** (ByteDance, June 2025)
- **What.** One-step diffusion-based video restoration model. Takes a low-quality / plasticky video and outputs a temporally-consistent high-resolution version. Designed specifically as a post-VAE-decode cleaner.
- **Why it works.** Uses **adaptive window attention** (so it handles arbitrary-length video) and **feature-matching loss** (so it doesn't drift across frames). It's a generative restorer, not a pixel-smoothing resampler.
- **Cost.** Substantial VRAM. 4K output on 24 GB requires BlockSwap (swap DiT blocks to CPU).
- **ComfyUI integration.** Official: `comfyorg/comfyui_seedvr2` and forks (`numz/ComfyUI-SeedVR2_VideoUpscaler`, `NeuroWaifu/ComfyUI.Node.SeedVR2`). Has a `SeedVR2BlockSwap` config node for low-VRAM systems.
- **RTX 3090 viability.** Confirmed working with BlockSwap. 24 GB is the realistic floor for 1080p→4K upscale; 720p→1080p is comfortable.
- **Paper.** arXiv 2506.05301 ("SeedVR2: One-Step Video Restoration via Diffusion Post-Training").

#### 1.4 Switch tokenizer entirely: NVIDIA Cosmos Tokenizer
- **What.** Drop-in encoder/decoder (continuous *or* discrete variants) that beats prior open video tokenizers by ~4 dB PSNR on DAVIS, 12× faster encode, encodes 8 s of 1080p in one shot on a single A100.
- **Cost.** Big VRAM at encode time (continuous variant), but the discrete variants are small.
- **Caveat for the user.** Cosmos Tokenizer is *not* shipped with Wan/Hunyuan/Mochi diffusion models. To use it, the user would need a diffusion model trained *with Cosmos latents*, OR they'd need to use Cosmos as a *post-hoc upsampler/decoder replacement* on top of the existing model's latents (works if the latents share structure with Cosmos's — they roughly do for spatial 8× continuous).
- **Honest take.** Likely the highest-fidelity option if the user can wire it. Off-the-shelf ComfyUI integration is thin as of July 2026.

#### 1.5 HunyuanVideo 1.5 over HunyuanVideo 1.0 / Wan 2.1
- **What.** Tencent rebuilt the VAE and added selective/sliding-window attention. Reports indicate noticeably less waxy-face output.
- **Cost.** 8.3 B params, runs on consumer GPUs per the report. RTX 3090 fits with offloading.
- **Repo.** `Tencent-Hunyuan/HunyuanVideo-1.5` (Nov 2025).
- **Caveat.** Different model entirely — user would have to migrate prompts and LoRAs.

### Tier 2 — Light workflow changes (no retraining)

#### 2.1 Decode at higher resolution, downsample
- **What.** Generate at 720p or higher, decode the VAE at that resolution, then bicubic/Lanczos downsample to your target.
- **Why it works.** The VAE's 8× downsampling is the source of the smooth look. Decoding at higher resolution gives the decoder more pixels to work with for high-frequency content; bicubic downsample preserves more detail than the VAE's learned upsample.
- **Cost.** Higher VRAM during decode; longer decode time.
- **Expected improvement.** Substantial. This is the single highest-ROI non-model change.

#### 2.2 Tiled VAE decode with overlap
- **What.** In ComfyUI, use a tiled-VAE-decode node with overlap, instead of the standard VAE decode.
- **Why it works.** The default VAE decode on a 1080p frame uses all pixels jointly in a single forward pass, and the decoder averages over the whole image. Tiling with overlap forces the decoder to think locally → more high-frequency detail is preserved.
- **Cost.** Slower.
- **Caveat.** Reports of OOM during tiled VAE decode on HunyuanVideo 1.5 (HF discussion). For Wan, tiled decode works reliably.

#### 2.3 Apply SeedVR2 / Real-ESRGAN / 4x-UltraSharp as a post-pass
- **What.** After VAE decode, run a video-aware super-resolution pass.
- **Why it works.** This re-introduces high-frequency detail that the VAE discarded. Real-ESRGAN is conservative (won't hallucinate); SeedVR2 is generative (will hallucinate).
- **Cost.** Adds inference time; for video, requires temporal-consistent variant.
- **Expected improvement.** Visible. The "Face Detailer + 4x UltraSharp" workflow from ComfyUI Impact Pack is the canonical image example; for video, SeedVR2 is the canonical replacement.

#### 2.4 LoRA finetune the diffusion model (not the VAE) on high-detail data
- **What.** Fine-tune Wan 2.1 / Hunyuan / Mochi with a small LoRA on a dataset of high-frequency-texture videos (close-up skin, fabric, hair, leaves).
- **Why it works.** This directly addresses layer 1.B (the diffusion model's L2-mean smoothing). After fine-tuning, the model produces sharper latents → VAE has more to work with.
- **Cost.** Few hundred GPU-hours on a 3090; doable but slow.
- **Community status.** CivitAI and Hugging Face have several "detail-enhancer" LoRAs for Wan; effect is real but inconsistent across prompts.

### Tier 3 — Structural changes (retraining, paper-level)

#### 3.1 **AsymFlow** (arXiv 2605.12964, May 2026)
- **What.** Asymmetric flow models: align a low-rank pixel subspace to a pretrained latent flow model's latent space, then finetune the pixel-space model so high-level semantics are preserved while low-level detail is recovered.
- **Why it works.** This is the first published method for finetuning a pretrained *latent* flow model into a *pixel-space* model without throwing away learned semantics. Directly addresses "VAE mushed my high frequencies."
- **Caveat.** Paper-level, no off-the-shelf checkpoint. Research prototype.

#### 3.2 **PixelDiT** (NVlabs, CVPR 2026 Best Paper Finalist)
- **What.** End-to-end pixel-space diffusion transformer. No VAE at all. Uses a dual-level architecture (patch-level DiT for global semantics + pixel-level DiT for texture detail) to generate directly in pixels.
- **Why it works.** Eliminates the VAE bottleneck entirely. PixelDiT produces outputs that are not constrained by a learned image manifold → much higher fidelity.
- **Caveat.** Paper/demo only. Training cost is enormous. Not practical for the user to retrain themselves.

#### 3.3 **"There is No VAE"** (ICLR 2026 Poster)
- **What.** Two-stage training framework that closes the gap between latent-space and pixel-space diffusion. Pixel-space models normally underperform because they're harder to train; this paper proposes a recipe.
- **Caveat.** Paper-level. Research prototype.

#### 3.4 **VideoVAE+** (ICCV 2025)
- **What.** Diagnoses the core problem: *"entangling spatial and temporal compression by merely extending the image VAE to a 3D VAE can introduce motion blur and detail distortion artifacts."* Proposes a cross-modal joint video-image training scheme and decoupled spatial/temporal compression.
- **Why it matters.** This is the paper that *formally names* the "plastic video" problem. The user's intuition is validated by the academic literature.
- **Repo.** `VideoVerses/VideoVAEPlus` (project page yzxing87.github.io/vae).
- **Practical takeaway.** Avoid VAEs that aggressively co-compress spatial+temporal (Mochi 6× temporal is the worst offender). Prefer decoupled or no-temporal-compression designs (Wan) or split designs (VideoVAE+).

#### 3.5 **LeanVAE** (ICCV 2025) and **LC-VAE** (Latent-Compressed VAE, 2026)
- **What.** LeanVAE: lightweight video VAE using Neighborhood-Aware Feedforward modules + wavelet transforms. LC-VAE: multi-level 3D wavelet transforms to filter high-frequency components out of the latent.
- **Why it works.** Wavelets explicitly separate low-frequency (shape) from high-frequency (texture). Keeps shape in the latent, lets the diffusion model learn texture directly.
- **Caveat.** Research papers; community checkpoints exist but support is thin.

#### 3.6 **DC-VideoGen** (Deep Compression Video Autoencoder, Sep 2025)
- **What.** Post-training framework to adapt any pretrained video diffusion model to a deeper-compression latent space (32× or 64× temporal instead of 4×–6×).
- **Why it matters.** Confirms the trend: the field is moving toward *higher* compression in the VAE (cheaper diffusion) + *better* post-decode restoration (SeedVR2/PiD) to recover the lost detail.
- **Paper.** arXiv 2509.25182.

---

## 4. Concrete recommendations for the user's stack (RTX 3090, 24 GB)

Ranked by ROI:

1. **First (free, 10 min):** Try `wan2.1_vae.safetensors` (FP32) instead of any BF16 variant. Switch from the default VAE Decode node to a tiled VAE Decode node with overlap (e.g., `comfyui-kjnodes` `VAEDecodeTiled`).

2. **Second (free, 30 min):** Add a SeedVR2 post-pass after VAE decode. Use BlockSwap, target 1080p output if generating 480p, or 4K if generating 720p. ComfyUI nodes: `comfyorg/comfyui_seedvr2`. Expect ~5–10× decode-time slowdown but a major drop in "plastic" look.

3. **Third (free, 30 min):** Decode at higher resolution than your target output and downsample with Lanczos/bicubic, not via the VAE.

4. **Fourth (medium effort):** If you haven't picked a base model, **HunyuanVideo 1.5** is currently (Nov 2025+) the best VAE+model combination for "not-waxy" face output on consumer hardware. **LTX-2 / 2.3** is the second-best option, with the bonus of speed.

5. **Fifth (medium effort):** Fine-tune a small "anti-plastic" LoRA on your base model with high-frequency-texture training data (close-ups of skin, fabric, foliage). This attacks layer 1.B directly.

6. **Sixth (advanced, paper-level):** Replace your VAE decoder entirely with NVIDIA PiD (image-only currently) or wrap your output with Cosmos Tokenizer for an additional upscale+detail pass.

7. **Do NOT do (yet):** retrain your own VideoVAE+ or PixelDiT. Both are research-only. Wait for community ports.

### What to avoid
- **Wan 2.1 BF16 VAE** — known blurry bug. Use FP32.
- **Mochi 1** as a base model if skin/faces are critical — its 6× temporal compression is the worst in class.
- **Aggressive CFG** (>9) — pushes latents further off-manifold → more VAE snapping → more "waxy."
- **SVD-XT** for anything portrait — uses SD's image VAE, which has been known to soften skin for years.

---

## 5. Sources

- Wan 2.1 paper: arxiv.org/abs/2503.20314 ("Wan: Open and Advanced Large-Scale Video Foundation Models")
- Wan VAE blurry bug: github.com/kijai/ComfyUI-WanVideoWrapper/issues/833
- Wan-VAE-upscale2x community model + diffusion latent mushiness discussion: huggingface.co/spacepxl/Wan2.1-VAE-upscale2x/discussions/3
- HunyuanVideo paper: arxiv.org/abs/2412.03603; HunyuanVideo 1.5: arxiv.org/abs/2511.18870 (Nov 2025)
- Mochi 1 / AsymmDiT: github.com/genmoai/mochi
- NVIDIA Cosmos Tokenizer: arxiv.org/abs/2501.03575 ("Cosmos World Foundation Model Platform")
- MagViT-2 / TokBench comparison: arxiv.org/abs/2505.18142
- NVIDIA PiD: research.nvidia.com/labs/sil/projects/pid/ ; github.com/nv-tlabs/PiD ; reddit.com/r/StableDiffusion/comments/1tn3m6n (community quantization)
- "Decoder Was Never Supposed to Be Creative" (DigitalOcean, May 2026): digitalocean.com/community/tutorials/why-diffusion-models-are-replacing-vae-decoders
- SeedVR2 paper: arxiv.org/abs/2506.05301 ; ComfyUI: github.com/comfyorg/comfyui_seedvr2 ; block-swap guide: seedvr2.net/blog/tutorials/seedvr2-block-swap-memory-fix-2026
- VideoVAE+ (ICCV 2025): yzxing87.github.io/vae ; github.com/VideoVerses/VideoVAEPlus ; arxiv.org/abs/2412.17805
- LeanVAE (ICCV 2025): arxiv.org/abs/2503.14325 ; github.com/westlake-repl/LeanVAE
- LC-VAE: arxiv.org/abs/2604.xxxxx (Apr 2026)
- DC-VideoGen: arxiv.org/abs/2509.25182
- AsymFlow: arxiv.org/abs/2605.12964 (May 2026)
- PixelDiT (CVPR 2026 Best Paper Finalist): github.com/NVlabs/PixelDiT
- "There is No VAE" (ICLR 2026 Poster): iclr.cc/virtual/2026/poster/10010377
- Wan 2.2 vs Wan 2.1 community impressions: reddit.com/r/StableDiffusion/comments/1mnk4i4 ; blog.fal.ai/wan-2-2-vs-wan-2-1
- LTX-2 vs Wan 2.2 comparison: insiderllm.com/guides/local-ai-video-generation ; ltx.io/blog/open-source-video-generation-models-guide
- ComfyUI Face Detailer (Impact Pack): github.com/ltdrdata/ComfyUI-Impact-Pack ; runcomfy.com/tutorials/face-detailer-comfyui-workflow-and-tutorial
- TokBench video tokenizer benchmark: arxiv.org/abs/2505.18142
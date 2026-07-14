# Roadmap

## Phase 0: Research (DONE)

- [x] AsymFlow paper deep dive → `research/asymflow_*`
- [x] Wan 2.1 architecture deep dive → `research/wan21_architecture.md`
- [x] NVIDIA PiD paper deep dive → `research/pid_paper.txt`
- [x] **NVIDIA PiD code deep dive + port plan** → `research/nvidia_pid_ported.md`
- [x] Architecture spec for video-PiD → `research/architecture_spec.md`
- [x] Training recipe for video-PiD → `research/training_recipe.md`
- [x] Open-source repo + HF model card (live)
- [x] Wan 2.1 1.3B T2V weights downloaded

## Phase 1: MVP (in progress)

**Critical insight:** NVIDIA already trained a PiD on Wan 2.1's VAE (the "qwenimage" checkpoint). We **port** it from 2D image to 3D video, not build from scratch.

- [x] Architecture locked → `docs/ARCHITECTURE.md`
- [x] Method locked → `docs/METHOD.md`
- [x] Training recipe locked → `docs/TRAINING.md`
- [ ] **Download NVIDIA PiD v1.5 qwenimage checkpoint** (~2.8GB, in progress)
- [ ] Vendor NVIDIA PiD code under `video_pid/nvidia/` (Apache 2.0 attribution)
- [ ] Write `video_pid/lq_projection_3d.py` (inflate Conv2d→Conv3d)
- [ ] Write `video_pid/pid_3d_net.py` (3D PixelDiT, STA attention)
- [ ] Write `video_pid/lq_video_or_image.py` glue (handle 5D video tensors)
- [ ] Write `scripts/train_pid.py` with Wan-VAE round-trip LQ corruption
- [ ] Write `video_pid/pipeline.py` Wan + PiD inference
- [ ] Write `scripts/generate_baseline.py` and `generate_with_pid.py`
- [ ] Smoke test: 5-min training, model improves LPIPS

## Phase 2: First release

- [ ] Full training run (256p → 480p curriculum, ~22h on 3090)
- [ ] Eval: LPIPS, DISTS, FVD vs Wan-VAE-decode baseline
- [ ] Side-by-side comparison videos
- [ ] HuggingFace model card + weights upload
- [ ] GitHub release v0.1.0
- [ ] Reddit r/StableDiffusion + r/LocalLLaMA post
- [ ] HF Spaces demo

## Phase 3: Iterate

- [ ] NSFW data tier (5-10% of dataset, add_sigma_max=0.4)
- [ ] Different aesthetic checkpoints (cinematic, anime, etc.)
- [ ] 2-step distilled student (CausVid / Align Your Flow distillation)
- [ ] RIFE temporal interpolation as a post-process
- [ ] Hunyuan / Mochi / LTX ports
- [ ] Optional LoRA-unlock phase on rented A100

## The moat (if this works)

1. **The training data** — curated aesthetic datasets, hard to copy
2. **The base model** — Wan 2.1 is Apache 2.0, freely usable
3. **The architecture** — published (NVIDIA PiD paper + code)
4. **The port** — first video-PiD, even though the math was published
5. **The brand** — consistent aesthetic, recognizable output

In a year, anyone can copy the architecture and the port. Nobody can copy the taste.

## Open questions

- Does inflating Conv2d→Conv3d break the NVIDIA checkpoint's learned weights? (init from conv2d → repeat over T axis, then fine-tune)
- Does 4-step sampler work for video or do we need 8?
- Does STA work as well in 3D as the 2D-PiD's full attention?
- Does adding temporal attention help or hurt?
- Can the video-PiD run jointly with Wan during the diffusion (early termination trick)?

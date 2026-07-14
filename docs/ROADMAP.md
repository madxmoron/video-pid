# Roadmap

## Phase 0: Research (DONE)

- [x] AsymFlow paper deep dive
- [x] Wan 2.1 architecture deep dive
- [x] PiD paper deep dive
- [x] Porting recipe from LakonLab (AsymQwen)
- [x] Video editing landscape survey
- [x] Plastic-video / VAE artifact analysis
- [x] Uncensored video model survey

## Phase 1: MVP (in progress)

- [ ] Pin video-PiD architecture (patch size, attention, conditioning, output)
- [ ] Lock training recipe (data, losses, optimizer, schedule)
- [ ] Set up the open-source repo (GitHub, HF, CI)
- [ ] Write the video-PiD model class (`video_pid/model.py`)
- [ ] Write the training script (`scripts/train_pid.py`)
- [ ] Write the inference pipeline (`video_pid/pipeline.py`)
- [ ] Generate "before" baseline video (Wan-VAE decode only)
- [ ] Generate "after" video (Wan + video-PiD)
- [ ] Smoke test: 5-min training run, model improves LPIPS

## Phase 2: First release

- [ ] Full training run on Panda-70M (or smaller curated set)
- [ ] Eval: LPIPS, DISTS, FVD vs baseline
- [ ] HuggingFace model card + weights upload
- [ ] GitHub release v0.1.0
- [ ] ComfyUI node (separate repo)
- [ ] Reddit + HF Spaces demo

## Phase 3: Iterate

- [ ] NSFW data path (DIY scraping, see `docs/NSFW_DATA.md` when ready)
- [ ] Different aesthetic checkpoints (cinematic, anime, etc.)
- [ ] 2-step distilled student
- [ ] Hunyuan / Mochi / LTX ports
- [ ] Real-time inference (1-step)

## Open questions

- Is residual learning the right call, or should we go full-image?
- Does the 4-step sampler actually work for video, or do we need 8?
- Is 300M params enough, or do we need 1B+?
- Does the PiD degrade Wan 2.1's prompt adherence, or improve it?

## Long-term moat

If this works, the moat is:
1. The training data (curated aesthetic datasets, hard to copy)
2. The base model (Wan 2.1 is Apache 2.0, freely usable)
3. The architecture is published (NVIDIA PiD paper)
4. The brand (consistent aesthetic, recognizable output)

In a year, anyone can copy the architecture. Nobody can copy the taste.

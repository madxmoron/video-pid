# Contributing

Open PRs welcome. This is a solo-started project, so anything you contribute is meaningful.

## What we'd love help with

- **More video-PiD checkpoints** trained on different data (anime, cinematic, NSFW, etc.) — the architecture is the hard part, the data is the easy part to add
- **Better samplers** — 4-step is good, 2-step consistency would be better
- **Different base backbones** — currently Wan 2.1, but the same pattern works for Hunyuan, Mochi, LTX
- **Eval work** — FVD, VBench, side-by-side with baselines
- **ComfyUI node** — `ComfyUI-VideoPiD` or similar

## Dev setup

```bash
git clone https://github.com/madxmoron/video-pid
cd video-pid
pip install -e ".[dev,train]"
pre-commit install  # if you want
```

Run tests:

```bash
pytest tests/
```

Lint:

```bash
ruff check video_pid/ scripts/ tests/
black video_pid/ scripts/ tests/
```

## PRs

- Keep changes small and focused
- One feature / bug fix per PR
- Add tests if you're adding a feature
- Update docs if you're changing user-facing behavior

## Issues

- Use the GitHub issue templates
- Include the prompt, the model checkpoint used, and the expected vs actual output for generation bugs
- For training issues, include your GPU, VRAM, batch size, and the traceback

## Code of conduct

Standard: don't be an asshole. This is a research project, not a corporation.

"""Smoke test: does the package import + does a forward pass work?

This is a stub-test until the model is implemented. It verifies the
package structure and module imports are healthy.
"""


def test_imports():
    """All public modules should import without error."""
    from video_pid import __version__
    from video_pid.losses import ResidualMSELoss
    from video_pid.data import VideoPiDDataset
    from video_pid.sampler import VideoPiDSampler
    from video_pid.pipeline import VideoPiDPipeline
    from video_pid.trainer import TrainingConfig, train

    assert __version__ == "0.0.1"
    assert ResidualMSELoss is not None
    assert VideoPiDDataset is not None
    assert VideoPiDSampler is not None
    assert VideoPiDPipeline is not None
    assert TrainingConfig is not None
    assert train is not None


def test_model_imports():
    """Model class should import, but forward pass is not yet implemented."""
    from video_pid.model import VideoPiD3DDiT

    assert VideoPiD3DDiT is not None


def test_cli_help():
    """All CLI scripts should be invokable with --help."""
    import subprocess
    import sys

    for script in [
        "scripts/generate_baseline.py",
        "scripts/generate_with_pid.py",
        "scripts/train_pid.py",
        "scripts/eval_pid.py",
    ]:
        result = subprocess.run(
            [sys.executable, script, "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, f"{script} --help failed: {result.stderr}"


if __name__ == "__main__":
    test_imports()
    test_model_imports()
    print("All smoke tests passed")

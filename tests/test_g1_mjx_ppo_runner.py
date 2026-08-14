import hashlib
from pathlib import Path

import pytest

from tools.run_g1_mjx_ppo import (
    build_environment_kwargs,
    build_training_kwargs,
    validate_asset,
    validate_clean_commit,
)


def test_runner_pins_exact_mjx_positive_control_contract() -> None:
    reference = Path("/tmp/walk.npz")
    environment = build_environment_kwargs(
        profile_name="g1-4x5",
        reference_path=reference,
    )
    training = build_training_kwargs(seed=42, output_dir=Path("/tmp/run"))

    assert environment == {
        "reference_path": str(reference.resolve()),
        "reference_stride": 1,
        "actor_history_len": 1,
        "actor_observation_noise": False,
        "domain_randomization": False,
        "friction_range": (1.0, 1.0),
        "mass_range": (1.0, 1.0),
        "kp_range": (35.0, 35.0),
        "kd_range": (0.5, 0.5),
        "com_offset_range": (0.0, 0.0, 0.0),
        "reference_reset_noise_scale": 0.0,
        "reference_residual_control": True,
        "reference_residual_scale": 1.0,
        "solver_iterations": 4,
        "solver_ls_iterations": 5,
    }
    assert training["total_iterations"] == 32
    assert training["num_envs"] == 4096
    assert training["horizon"] == 24
    assert training["initial_action_std"] == 0.2
    assert training["actor_learning_rate"] == 3e-4
    assert training["critic_learning_rate"] == 3e-4
    assert training["num_epochs"] == 4
    assert training["num_minibatches"] == 8
    assert training["seed"] == 42


def test_validate_asset_rejects_wrong_hash(tmp_path: Path) -> None:
    asset = tmp_path / "asset.bin"
    asset.write_bytes(b"exact")
    digest = hashlib.sha256(b"exact").hexdigest()

    assert validate_asset(asset, digest) == asset.resolve()
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate_asset(asset, "0" * 64)


def test_validate_clean_commit_rejects_dirty_or_wrong_revision(
    tmp_path: Path,
) -> None:
    import subprocess

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"], cwd=tmp_path, check=True
    )
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("clean\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=tmp_path, check=True)
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True
    ).strip()

    assert validate_clean_commit(tmp_path, head) == head
    with pytest.raises(ValueError, match="commit mismatch"):
        validate_clean_commit(tmp_path, "0" * 40)
    tracked.write_text("dirty\n", encoding="utf-8")
    with pytest.raises(ValueError, match="dirty"):
        validate_clean_commit(tmp_path, head)

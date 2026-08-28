import inspect
from pathlib import Path

from src.algorithms.shac.algorithm import train


def test_shac_jave_is_default_off_and_resume_start_requires_authority():
    parameters = inspect.signature(train).parameters

    assert parameters["jave_vg_weight"].default == 0.0
    assert parameters["jave_vg_warmup_steps"].default == 5_000
    assert parameters["jave_ldm_hidden"].default == (256, 256)
    assert parameters["jave_ldm_batch_size"].default == 256
    assert parameters["jave_vg_batch_size"].default == 256
    assert parameters["jave_ldm_buffer_capacity"].default == 100_000
    assert parameters["jave_reward_feature_scale"].default == 8.0
    assert parameters["jave_collect_transitions"].default is False
    assert parameters["allow_resume_jave_start"].default is False


def test_shac_jave_uses_the_g1_reward_sufficient_pre_reset_transition():
    source = Path(inspect.getsourcefile(train)).read_text()

    assert '"jave_enabled": jave_enabled' in source
    assert '"jave_obs": env._get_jave_obs(' in source
    assert 'candidate_unreplayed_state.info[' in source
    assert '"bootstrap_jave_obs"' in source
    assert "gradient_bellman_targets(" in source
    assert "gradient_bellman_loss(" in source
    assert "state.jave_vg_weight" in source
    assert "jave_start_step + jave_vg_warmup_steps" in source
    assert "allow_resume_jave_start=True" in source

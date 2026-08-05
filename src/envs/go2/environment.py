"""Variant factory for Go2 locomotion environments."""

from importlib import import_module


DEFAULT_VARIANT = "blind_nolinvel_nokinref"
VARIANTS = {
    "blind_nolinvel_nokinref": (
        "src.envs.go2.variants.blind_nolinvel_nokinref.environment"
    ),
    "blind_linvel_nokinref": (
        "src.envs.go2.variants.blind_linvel_nokinref.environment"
    ),
    "blind_linvel_kinref": "src.envs.go2.variants.blind_linvel_kinref.environment",
    "highspeed_nokinref": "src.envs.go2.variants.highspeed_nokinref.environment",
    "humanoid_blind_linvel_nokinref": "src.envs.humanoid.environment",
    "g1_tracking": "src.envs.g1_tracking.environment",
    "g1_tracking_rmr_50hz": "src.envs.g1_tracking.environment",
    "g1_tracking_rmr_50hz_unbounded": "src.envs.g1_tracking.environment",
    "g1_tracking_rmr_50hz_source_step": "src.envs.g1_tracking.environment",
    "g1_tracking_rmr_50hz_source_step_robust": (
        "src.envs.g1_tracking.environment"
    ),
    "g1_tracking_rmr_50hz_validated": (
        "src.envs.g1_tracking.environment"
    ),
}


def get_go2_env_class(variant: str = DEFAULT_VARIANT):
    """Return the Go2Env class for a named environment variant."""
    if variant is None:
        variant = DEFAULT_VARIANT
    try:
        module_name = VARIANTS[variant]
    except KeyError as exc:
        valid = ", ".join(sorted(VARIANTS))
        raise ValueError(f"Unknown Go2 variant '{variant}'. Valid variants: {valid}") from exc
    module = import_module(module_name)
    if variant == "humanoid_blind_linvel_nokinref":
        return module.HumanoidEnv
    if variant == "g1_tracking":
        return module.G1TrackingEnv
    if variant == "g1_tracking_rmr_50hz":
        return module.G1TrackingRMR50HzEnv
    if variant == "g1_tracking_rmr_50hz_unbounded":
        return module.G1TrackingRMR50HzUnboundedEnv
    if variant == "g1_tracking_rmr_50hz_source_step":
        return module.G1TrackingRMR50HzSourceStepEnv
    if variant == "g1_tracking_rmr_50hz_source_step_robust":
        return module.G1TrackingRMR50HzSourceStepRobustEnv
    if variant == "g1_tracking_rmr_50hz_validated":
        return module.G1TrackingRMR50HzValidatedEnv
    return module.Go2Env


def Go2Env(*args, variant: str = DEFAULT_VARIANT, **kwargs):
    """Instantiate a named Go2 environment variant."""
    env_cls = get_go2_env_class(variant)
    return env_cls(*args, **kwargs)


__all__ = ["DEFAULT_VARIANT", "VARIANTS", "Go2Env", "get_go2_env_class"]

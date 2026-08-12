"""Pure schedule and resume contract for analytic torso assistance."""

from __future__ import annotations

import math

import jax
import jax.numpy as jp


DISABLED_ASSISTANCE_SETTINGS = (False, 0, 1, 0.0)
DISABLED_CONDITIONING_SETTINGS = (False, False, False)


def assistance_scale_at_step(
    step: jax.Array | int,
    *,
    start_step: int,
    end_step: int,
) -> jax.Array:
    """Return the clipped linear one-to-zero assistance schedule."""
    progress = (jp.asarray(step, dtype=jp.float32) - float(start_step)) / float(
        end_step - start_step
    )
    return jp.clip(1.0 - progress, 0.0, 1.0)


def sample_assistance_scales(
    key: jax.Array,
    *,
    num_envs: int,
    scheduled_scale: jax.Array | float,
    zero_fraction: float,
    continuous: bool = False,
) -> jax.Array:
    """Sample one fixed assistance scale per environment and unroll."""
    mask_key, magnitude_key = jax.random.split(key)
    held_out = jax.random.uniform(mask_key, (num_envs,)) < zero_fraction
    scale = jp.asarray(scheduled_scale, dtype=jp.float32)
    if continuous:
        scale = scale * jax.random.uniform(
            magnitude_key, (num_envs,), dtype=jp.float32
        )
    return jp.where(held_out, jp.zeros_like(scale), scale)


def torso_wrench_assistance_diagnostics(
    wrenches: jax.Array,
    *,
    assistance_scales: jax.Array,
    force_cap: float,
    torque_cap: float,
) -> dict[str, jax.Array]:
    """Summarize a population rollout and validate its assistance contract."""
    values = jp.asarray(wrenches)
    scales = jp.asarray(assistance_scales)
    force_norms = jp.linalg.norm(values[..., :3], axis=-1)
    torque_norms = jp.linalg.norm(values[..., 3:], axis=-1)
    zero_environments = scales == 0.0
    zero_wrenches = jp.all(
        jp.where(
            zero_environments[..., None, None],
            values == 0.0,
            jp.ones_like(values, dtype=bool),
        )
    )
    finite = jp.all(jp.isfinite(values)) & jp.all(jp.isfinite(scales))
    scales_bounded = jp.all((scales >= 0.0) & (scales <= 1.0))
    max_force = jp.max(force_norms)
    max_torque = jp.max(torque_norms)
    cap_compliant = (max_force <= force_cap + 1e-5) & (max_torque <= torque_cap + 1e-5)
    return {
        "active_fraction": jp.mean(scales > 0.0),
        "rms_force": jp.sqrt(jp.mean(jp.square(force_norms))),
        "rms_torque": jp.sqrt(jp.mean(jp.square(torque_norms))),
        "max_force": max_force,
        "max_torque": max_torque,
        "finite": finite,
        "scales_bounded": scales_bounded,
        "cap_compliant": cap_compliant,
        "zero_environments_exact": zero_wrenches,
        "valid": finite & scales_bounded & cap_compliant & zero_wrenches,
    }


def validate_torso_wrench_assistance_configuration(
    *,
    enabled: bool,
    start_step: int,
    end_step: int,
    zero_fraction: float,
    env_variant: str,
) -> None:
    """Fail closed on malformed or unsupported assistance settings."""
    if not isinstance(enabled, bool):
        raise ValueError("torso_wrench_assistance must be boolean")
    if (
        isinstance(start_step, bool)
        or not isinstance(start_step, int)
        or start_step < 0
    ):
        raise ValueError("start_step must be a non-negative integer")
    if isinstance(end_step, bool) or not isinstance(end_step, int):
        raise ValueError("end_step must be an integer")
    if end_step <= start_step:
        raise ValueError("end_step must be greater than start_step")
    if (
        isinstance(zero_fraction, bool)
        or not isinstance(zero_fraction, (int, float))
        or not math.isfinite(zero_fraction)
        or not 0.0 <= zero_fraction <= 1.0
    ):
        raise ValueError("zero_fraction must be finite and in [0, 1]")
    if enabled and not env_variant.startswith("g1_tracking"):
        raise ValueError("torso wrench assistance requires a G1 tracking environment")


def validate_assistance_conditioning_configuration(
    *,
    assistance_enabled: bool,
    continuous: bool,
    conditioning: bool,
    observed: bool,
    residual_adapter_enabled: bool,
) -> None:
    """Validate the scalar-only assistance observation boundary."""
    for name, value in (
        ("torso_wrench_assistance_continuous", continuous),
        ("actor_torso_wrench_assistance_conditioning", conditioning),
        ("actor_observe_torso_wrench_assistance", observed),
    ):
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be boolean")
    if observed and not conditioning:
        raise ValueError("observed assistance requires scalar conditioning")
    if conditioning and not assistance_enabled:
        raise ValueError("assistance conditioning requires torso wrench assistance")
    if conditioning and not residual_adapter_enabled:
        raise ValueError("assistance conditioning requires residual preview adapter")


def resolve_assistance_conditioning_resume_settings(
    resumed_hparams: dict[str, object] | None,
    *,
    requested_continuous: bool,
    requested_conditioning: bool,
    requested_observed: bool,
    allow_change: bool,
) -> tuple[bool, bool, bool]:
    """Restore scalar-conditioning metadata unless change is authorized."""
    if not isinstance(allow_change, bool):
        raise ValueError("allow_resume_assistance_conditioning_change must be boolean")
    requested = (
        requested_continuous,
        requested_conditioning,
        requested_observed,
    )
    resumed_hparams = resumed_hparams or {}
    resumed = (
        resumed_hparams.get("torso_wrench_assistance_continuous", False),
        resumed_hparams.get("actor_torso_wrench_assistance_conditioning", False),
        resumed_hparams.get("actor_observe_torso_wrench_assistance", False),
    )
    if not all(isinstance(value, bool) for value in resumed):
        raise ValueError("assistance conditioning checkpoint metadata is invalid")
    if requested == DISABLED_CONDITIONING_SETTINGS and not allow_change:
        return resumed
    if requested != resumed and not allow_change:
        raise ValueError(
            "assistance conditioning settings must match the checkpoint unless "
            "allow_resume_assistance_conditioning_change is enabled"
        )
    return requested


def resolve_torso_wrench_assistance_resume_settings(
    resumed_hparams: dict[str, object] | None,
    *,
    requested_enabled: bool,
    requested_start_step: int,
    requested_end_step: int,
    requested_zero_fraction: float,
    allow_change: bool,
) -> tuple[bool, int, int, float]:
    """Restore an active curriculum or admit one explicitly authorized change."""
    if not isinstance(allow_change, bool):
        raise ValueError("allow_resume_torso_wrench_assistance_change must be boolean")
    requested = (
        requested_enabled,
        requested_start_step,
        requested_end_step,
        float(requested_zero_fraction),
    )
    resumed_hparams = resumed_hparams or {}

    resumed_enabled = resumed_hparams.get("torso_wrench_assistance", False)
    if not isinstance(resumed_enabled, bool):
        raise ValueError(
            "torso wrench assistance checkpoint contains invalid resume metadata"
        )
    if resumed_enabled:
        required = {
            "torso_wrench_assistance_start_step",
            "torso_wrench_assistance_end_step",
            "torso_wrench_assistance_zero_fraction",
        }
        if not required.issubset(resumed_hparams):
            raise ValueError(
                "active torso wrench assistance requires complete resume metadata"
            )
    resumed = (
        resumed_enabled,
        resumed_hparams.get("torso_wrench_assistance_start_step", 0),
        resumed_hparams.get("torso_wrench_assistance_end_step", 1),
        resumed_hparams.get("torso_wrench_assistance_zero_fraction", 0.0),
    )
    validate_torso_wrench_assistance_configuration(
        enabled=resumed[0],
        start_step=resumed[1],
        end_step=resumed[2],
        zero_fraction=resumed[3],
        env_variant="g1_tracking_resume",
    )
    resumed = (resumed[0], resumed[1], resumed[2], float(resumed[3]))
    if requested == DISABLED_ASSISTANCE_SETTINGS and not allow_change:
        return resumed
    if requested != resumed and not allow_change:
        raise ValueError(
            "torso wrench assistance settings must match the checkpoint unless "
            "allow_resume_torso_wrench_assistance_change is enabled"
        )
    return requested

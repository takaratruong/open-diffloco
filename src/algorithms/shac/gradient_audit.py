"""Pure score-estimator primitives for the SHAC gradient audit."""

import jax
import jax.numpy as jp


def discounted_return_to_go(
    rewards: jax.Array,
    dones: jax.Array,
    *,
    gamma: float,
) -> jax.Array:
    """Returns actor-loss score coefficients through episode boundaries.

    ``dones[t]`` marks that transition ``t`` as terminal, so its immediate
    reward remains in the return while rewards from later transitions do not.
    Each reward retains the outer discount accumulated from the start of its
    episode, matching the production actor loss; that discount resets after a
    done transition.
    The leading axis is time; all remaining axes are independent trajectories.
    """

    def advance_discount(discount, done):
        next_discount = jp.where(done, 1.0, discount * gamma)
        return next_discount, discount

    _, outer_discounts = jax.lax.scan(
        advance_discount,
        jp.ones_like(rewards[0]),
        dones,
    )
    discounted_rewards = outer_discounts * rewards

    def accumulate(next_return, transition):
        reward, done = transition
        current_return = reward + jp.where(done, 0.0, next_return)
        return current_return, current_return

    _, reversed_returns = jax.lax.scan(
        accumulate,
        jp.zeros_like(rewards[0]),
        (jp.flip(discounted_rewards, axis=0), jp.flip(dones, axis=0)),
    )
    return jp.flip(reversed_returns, axis=0)


def detached_gaussian_score_loss(
    means: jax.Array,
    actions: jax.Array,
    returns_to_go: jax.Array,
    *,
    std: float | jax.Array,
) -> jax.Array:
    """Computes a stopped-data Gaussian likelihood-ratio loss.

    The leading axis is time and the final axis is the action dimension.  Any
    axes between them are preserved, allowing callers to receive one loss per
    environment by passing arrays shaped ``(time, environment, action)``.
    """
    stopped_actions = jax.lax.stop_gradient(actions)
    stopped_returns = jax.lax.stop_gradient(returns_to_go)
    standard_deviation = jp.asarray(std)
    normalized_error = (stopped_actions - means) / standard_deviation
    log_probability = -0.5 * jp.sum(
        jp.square(normalized_error)
        + 2.0 * jp.log(standard_deviation)
        + jp.log(2.0 * jp.pi),
        axis=-1,
    )
    return -jp.mean(stopped_returns * log_probability, axis=0)

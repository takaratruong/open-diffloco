import unittest

from flax.core import freeze, unfreeze
import jax
import jax.numpy as jnp
import numpy as np

from src.core.networks import Actor


class ActorActionSupportTest(unittest.TestCase):
    def test_linear_rmr_actor_is_not_tanh_bounded(self):
        linear_actor = Actor(action_dim=1, hidden=(), squash=False)
        bounded_actor = Actor(action_dim=1, hidden=(), squash=True)
        params = unfreeze(
            linear_actor.init(jax.random.PRNGKey(0), jnp.ones((1, 1)))
        )
        params["params"]["Dense_0"]["bias"] = jnp.array([2.0])
        params = freeze(params)

        linear = linear_actor.apply(params, jnp.ones((1, 1)))
        bounded = bounded_actor.apply(params, jnp.ones((1, 1)))

        np.testing.assert_allclose(linear, np.array([[2.0]]))
        np.testing.assert_allclose(bounded, np.tanh(np.array([[2.0]])))


if __name__ == "__main__":
    unittest.main()

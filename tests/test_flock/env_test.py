import chex
import jax
from jumanji.testing.env_not_smoke import (
    check_env_does_not_smoke,
    check_env_specs_does_not_smoke,
)

from covey.flock.env import Flock
from covey.flock.types import Observation


def test_env_does_not_smoke(env: Flock) -> None:
    """Test that we can run an episode without any errors."""
    env.time_limit = 10

    def select_action(action_key: chex.PRNGKey, _state: Observation) -> chex.Array:
        return jax.random.uniform(
            action_key, (env.generator.num_boids, 2), minval=-1.0, maxval=1.0
        )

    check_env_does_not_smoke(env, select_action=select_action)


def test_env_specs_do_not_smoke(env: Flock) -> None:
    """Test that we can access specs without any errors."""
    check_env_specs_does_not_smoke(env)

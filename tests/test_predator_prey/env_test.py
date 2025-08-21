import chex
import jax
from jumanji.testing.env_not_smoke import (
    check_env_does_not_smoke,
    check_env_specs_does_not_smoke,
)

from floxs.predator_prey.env import PredatorPrey
from floxs.predator_prey.types import Actions, Observation


def test_env_does_not_smoke(env: PredatorPrey) -> None:
    """Test that we can run an episode without any errors."""
    env.time_limit = 10

    def select_action(action_key: chex.PRNGKey, _state: Observation) -> Actions:
        k1, k2 = jax.random.split(action_key, 2)
        return Actions(
            predators=jax.random.uniform(
                k1, (env.generator.num_predators, 2), minval=-1.0, maxval=1.0
            ),
            prey=jax.random.uniform(
                k2, (env.generator.num_prey, 2), minval=-1.0, maxval=1.0
            ),
        )

    check_env_does_not_smoke(env, select_action=select_action)


def test_env_specs_do_not_smoke(env: PredatorPrey) -> None:
    """Test that we can access specs without any errors."""
    check_env_specs_does_not_smoke(env)

import chex
import jax.numpy as jnp
import pytest
from jumanji.environments.swarms.common.types import AgentParams

from covey.predator_prey.generator import Generator, RandomGenerator
from covey.predator_prey.types import State


@pytest.mark.parametrize("env_size", [1.0, 0.5])
def test_random_generator(key: chex.PRNGKey, env_size: float) -> None:
    predator_params = AgentParams(
        max_rotate=0.5,
        max_accelerate=0.01,
        min_speed=0.01,
        max_speed=0.05,
    )
    prey_params = AgentParams(
        max_rotate=0.5,
        max_accelerate=0.01,
        min_speed=0.1,
        max_speed=0.15,
    )
    generator = RandomGenerator(num_predators=10, num_prey=2, env_size=env_size)

    assert isinstance(generator, Generator)

    state = generator(key, predator_params, prey_params)

    assert isinstance(state, State)
    assert state.predators.pos.shape == (generator.num_predators, 2)
    assert state.prey.pos.shape == (generator.num_prey, 2)
    assert jnp.all(0.0 <= state.predators.pos) and jnp.all(
        state.predators.pos <= env_size
    )
    assert jnp.all(0.0 <= state.prey.pos) and jnp.all(state.prey.pos <= env_size)
    assert state.step == 0

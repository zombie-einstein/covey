import chex
import jax.numpy as jnp
import pytest
from jumanji.environments.swarms.common.types import AgentParams

from covey.flock.generator import Generator, RandomGenerator
from covey.flock.types import State


@pytest.mark.parametrize("env_size", [1.0, 0.5])
def test_random_generator(key: chex.PRNGKey, env_size: float) -> None:
    params = AgentParams(
        max_rotate=0.5,
        max_accelerate=0.01,
        min_speed=0.01,
        max_speed=0.05,
    )
    generator = RandomGenerator(num_boids=50, env_size=env_size)

    assert isinstance(generator, Generator)

    state = generator(key, params)

    assert isinstance(state, State)
    assert state.boids.pos.shape == (generator.num_boids, 2)
    assert jnp.all(0.0 <= state.boids.pos) and jnp.all(state.boids.pos <= env_size)
    assert state.step == 0

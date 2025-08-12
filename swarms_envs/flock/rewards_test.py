import jax.numpy as jnp
import pytest
from jumanji.environments.swarms.common.types import AgentState

from .rewards import ExponentialRewardFn, RewardFn
from .types import State


@pytest.mark.parametrize(
    "positions, env_size, expected",
    [
        ([(0.5, 0.25), (0.5, 0.75)], 1.0, [0.0, 0.0]),
        ([(0.5, 0.1), (0.5, 0.15)], 1.0, [-0.5, -0.5]),
        ([(0.5, 0.1), (0.5, 0.3)], 1.0, [0.0, 0.0]),
    ],
)
def test_exponential_rewards(
    positions: list[tuple[float, float]],
    env_size: float,
    expected: list[float],
) -> None:
    positions = jnp.array(positions)
    n_agents = positions.shape[0]
    headings = jnp.zeros((n_agents,))
    speed = jnp.zeros((n_agents,))

    state = State(
        boids=AgentState(pos=positions, heading=headings, speed=speed),
    )

    rewards = ExponentialRewardFn(
        boid_radius=0.1,
        collision_penalty=0.5,
        i_range=0.2,
    )

    assert isinstance(rewards, RewardFn)

    rewards = rewards(state)
    expected = jnp.array(expected)

    assert jnp.allclose(rewards, expected)

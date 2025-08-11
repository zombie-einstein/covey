import jax.numpy as jnp
import pytest
from jumanji.environments.swarms.common.types import AgentState

from .rewards import SparseRewards
from .types import Rewards, State

PREDATOR_REWARD = 1.1
PREY_PENALTY = 0.9


@pytest.mark.parametrize(
    "predator_positions, prey_positions, env_size, " "predator_expected, prey_expected",
    [
        ([(0.2, 0.5)], [(0.8, 0.5)], 1.0, [0.0], [0.0]),
        ([(0.2, 0.5)], [(0.3, 0.5)], 1.0, [PREDATOR_REWARD], [-PREY_PENALTY]),
        (
            [(0.2, 0.5), (0.4, 0.5)],
            [(0.3, 0.5)],
            1.0,
            [PREDATOR_REWARD, PREDATOR_REWARD],
            [-2 * PREY_PENALTY],
        ),
        (
            [(0.3, 0.5)],
            [(0.2, 0.5), (0.4, 0.5)],
            1.0,
            [PREDATOR_REWARD],
            [-PREY_PENALTY, -PREY_PENALTY],
        ),
    ],
)
def test_sparse_rewards(
    predator_positions: list[tuple[float, float]],
    prey_positions: list[tuple[float, float]],
    env_size: float,
    predator_expected: float,
    prey_expected: list[float],
) -> None:

    predator_positions = jnp.array(predator_positions)
    prey_positions = jnp.array(prey_positions)

    state = State(
        predators=AgentState(
            pos=predator_positions,
            heading=jnp.zeros((1,)),
            speed=jnp.zeros((1,)),
        ),
        prey=AgentState(
            pos=prey_positions,
            heading=jnp.zeros((1,)),
            speed=jnp.zeros((1,)),
        ),
        step=0,
    )

    reward_fn = SparseRewards(
        capture_radius=0.2, predator_reward=PREDATOR_REWARD, prey_penalty=PREY_PENALTY
    )

    rewards = reward_fn(state)

    assert isinstance(rewards, Rewards)

    predator_expected = jnp.array(predator_expected)
    prey_expected = jnp.array(prey_expected)

    assert jnp.allclose(rewards.predators, predator_expected)
    assert jnp.allclose(rewards.prey, prey_expected)

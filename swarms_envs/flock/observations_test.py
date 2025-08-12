import jax.numpy as jnp
import pytest
from jumanji.environments.swarms.common.types import AgentState

from .observations import ObservationFn
from .types import State

VISION_RANGE = 0.2
VIEW_ANGLE = 0.5


@pytest.mark.parametrize(
    "positions, headings, env_size, view",
    [
        # Both out of view range
        ([(0.8, 0.5), (0.2, 0.5)], [jnp.pi, 0.0], 1.0, []),
        # Both view each other
        ([(0.25, 0.5), (0.2, 0.5)], [jnp.pi, 0.0], 1.0, [(0, 5, 0.25), (1, 5, 0.25)]),
        # One facing wrong direction
        (
            [(0.25, 0.5), (0.2, 0.5)],
            [jnp.pi, jnp.pi],
            1.0,
            [(0, 5, 0.25)],
        ),
        # Only see closest neighbour
        (
            [(0.35, 0.5), (0.25, 0.5), (0.2, 0.5)],
            [jnp.pi, 0.0, 0.0],
            1.0,
            [(0, 5, 0.5), (1, 5, 0.5), (2, 5, 0.25)],
        ),
        # Observed around wrapped edge
        (
            [(0.025, 0.5), (0.975, 0.5)],
            [jnp.pi, 0.0],
            1.0,
            [(0, 5, 0.25), (1, 5, 0.25)],
        ),
        # Observed around wrapped edge of smaller env
        (
            [(0.025, 0.25), (0.475, 0.25)],
            [jnp.pi, 0.0],
            0.5,
            [(0, 5, 0.25), (1, 5, 0.25)],
        ),
    ],
)
def test_vision(
    positions: list[tuple[float, float]],
    headings: list[float],
    env_size: float,
    view: list[tuple[int, int, float]],
) -> None:
    n_agents = len(headings)
    positions = jnp.array(positions)
    headings = jnp.array(headings)
    speed = jnp.zeros(headings.shape)

    state = State(
        boids=AgentState(pos=positions, heading=headings, speed=speed),
    )

    observe_fn = ObservationFn(
        num_vision=11,
        vision_range=VISION_RANGE,
        view_angle=VIEW_ANGLE,
        boid_radius=0.01,
        env_size=env_size,
    )

    obs = observe_fn(state)
    assert obs.shape == (n_agents, observe_fn.num_vision)

    expected = jnp.full((n_agents, observe_fn.num_vision), -1.0)

    for i, idx, val in view:
        expected = expected.at[i, idx].set(val)

    assert jnp.all(jnp.isclose(obs, expected))

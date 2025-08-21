import jax.numpy as jnp
import pytest
from jumanji.environments.swarms.common.types import AgentState

from floxs.predator_prey.env import PredatorPrey
from floxs.predator_prey.observations import ObservationFn
from floxs.predator_prey.types import State


@pytest.mark.parametrize(
    "predator_positions, predator_headings, env_size, view",
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
def test_predator_view_predator(
    env: PredatorPrey,
    predator_positions: list[tuple[float, float]],
    predator_headings: list[float],
    env_size: float,
    view: list[tuple[int, int, float]],
) -> None:
    n_agents = len(predator_headings)
    predator_positions = jnp.array(predator_positions)
    predator_headings = jnp.array(predator_headings)
    speed = jnp.zeros(predator_headings.shape)

    predator_state = AgentState(
        pos=predator_positions, heading=predator_headings, speed=speed
    )

    prey_state = AgentState(
        pos=jnp.array([(0.0, 0.0)]),
        heading=jnp.array([0.0]),
        speed=jnp.array([0.0]),
    )

    state = State(
        predators=predator_state,
        prey=prey_state,
        step=0,
    )

    observe_fn = ObservationFn(
        num_vision=11,
        predator_vision_range=0.2,
        prey_vision_range=0.2,
        predator_view_angle=0.5,
        prey_view_angle=0.5,
        agent_radius=0.01,
        env_size=env_size,
    )

    predator_obs, prey_obs = observe_fn(state)

    assert predator_obs.shape == (n_agents, 2, observe_fn.num_vision)

    expected = jnp.full((n_agents, 2, observe_fn.num_vision), -1.0)

    for i, idx, val in view:
        expected = expected.at[i, 0, idx].set(val)

    assert jnp.all(jnp.isclose(predator_obs, expected))


@pytest.mark.parametrize(
    "prey_positions, prey_headings, env_size, view",
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
def test_prey_view_prey(
    env: PredatorPrey,
    prey_positions: list[tuple[float, float]],
    prey_headings: list[float],
    env_size: float,
    view: list[tuple[int, int, float]],
) -> None:
    n_agents = len(prey_headings)
    prey_positions = jnp.array(prey_positions)
    prey_headings = jnp.array(prey_headings)
    speed = jnp.zeros(prey_headings.shape)

    predator_state = AgentState(
        pos=jnp.array([(0.0, 0.0)]),
        heading=jnp.array([0.0]),
        speed=jnp.array([0.0]),
    )

    prey_state = AgentState(pos=prey_positions, heading=prey_headings, speed=speed)

    state = State(
        predators=predator_state,
        prey=prey_state,
        step=0,
    )

    observe_fn = ObservationFn(
        num_vision=11,
        predator_vision_range=0.2,
        prey_vision_range=0.2,
        predator_view_angle=0.5,
        prey_view_angle=0.5,
        agent_radius=0.01,
        env_size=env_size,
    )

    prey_obs, prey_obs = observe_fn(state)

    assert prey_obs.shape == (n_agents, 2, observe_fn.num_vision)

    expected = jnp.full((n_agents, 2, observe_fn.num_vision), -1.0)

    for i, idx, val in view:
        expected = expected.at[i, 0, idx].set(val)

    assert jnp.all(jnp.isclose(prey_obs, expected))


@pytest.mark.parametrize(
    (
        "predator_position, predator_heading, prey_position, "
        "prey_heading, env_size, predator_view, prey_view"
    ),
    [
        # Both out of view range
        ((0.8, 0.5), jnp.pi, (0.2, 0.5), 0.0, 1.0, [], []),
        # # Both view each other
        ((0.25, 0.5), jnp.pi, (0.2, 0.5), 0.0, 1.0, [(5, 0.25)], [(5, 0.25)]),
        # # Prey facing wrong direction
        ((0.25, 0.5), jnp.pi, (0.2, 0.5), jnp.pi, 1.0, [(5, 0.25)], []),
        # Observed around wrapped edge
        ((0.025, 0.5), jnp.pi, (0.975, 0.5), 0.0, 1.0, [(5, 0.25)], [(5, 0.25)]),
        # # Observed around wrapped edge of smaller env
        ((0.025, 0.25), jnp.pi, (0.475, 0.25), 0.0, 0.5, [(5, 0.25)], [(5, 0.25)]),
    ],
)
def test_predator_view_prey(
    env: PredatorPrey,
    predator_position: tuple[float, float],
    predator_heading: float,
    prey_position: tuple[float, float],
    prey_heading: float,
    env_size: float,
    predator_view: tuple[int, int, float],
    prey_view: tuple[int, int, float],
) -> None:
    predator_state = AgentState(
        pos=jnp.array([predator_position]),
        heading=jnp.array([predator_heading]),
        speed=jnp.zeros((1,)),
    )

    prey_state = AgentState(
        pos=jnp.array([prey_position]),
        heading=jnp.array([prey_heading]),
        speed=jnp.zeros((1,)),
    )

    state = State(
        predators=predator_state,
        prey=prey_state,
        step=0,
    )

    observe_fn = ObservationFn(
        num_vision=11,
        predator_vision_range=0.2,
        prey_vision_range=0.2,
        predator_view_angle=0.5,
        prey_view_angle=0.5,
        agent_radius=0.01,
        env_size=env_size,
    )

    predator_obs, prey_obs = observe_fn(state)

    assert predator_obs.shape == (1, 2, observe_fn.num_vision)
    assert prey_obs.shape == (1, 2, observe_fn.num_vision)

    predator_expected = jnp.full((1, 2, observe_fn.num_vision), -1.0)

    for idx, val in predator_view:
        predator_expected = predator_expected.at[0, 1, idx].set(val)

    prey_expected = jnp.full((1, 2, observe_fn.num_vision), -1.0)

    for idx, val in prey_view:
        prey_expected = prey_expected.at[0, 1, idx].set(val)

    assert jnp.all(jnp.isclose(predator_obs, predator_expected))
    assert jnp.all(jnp.isclose(prey_obs, prey_expected))

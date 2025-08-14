import pytest

from covey.predator_prey.env import PredatorPrey
from covey.predator_prey.observations import ObservationFn


@pytest.fixture
def env() -> PredatorPrey:
    observation_fn = ObservationFn(
        num_vision=5,
        predator_vision_range=0.2,
        prey_vision_range=0.1,
        predator_view_angle=0.2,
        prey_view_angle=0.1,
        agent_radius=0.01,
        env_size=1.0,
    )
    return PredatorPrey(observation=observation_fn)

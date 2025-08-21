import pytest

from floxs.flock.env import Flock
from floxs.flock.observations import ObservationFn


@pytest.fixture
def env() -> Flock:
    observation_fn = ObservationFn(
        num_vision=5,
        vision_range=0.2,
        view_angle=0.5,
        boid_radius=0.01,
        env_size=1.0,
    )
    return Flock(observation=observation_fn)

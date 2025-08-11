import abc

import chex
from jumanji.environments.swarms.common.types import AgentParams

from swarms_envs.common.utils import random_agent_state

from .types import State


class Generator(abc.ABC):
    def __init__(self, num_boids: int, env_size: float = 1.0) -> None:
        self.num_boids = num_boids
        self.env_size = env_size

    @abc.abstractmethod
    def __call__(self, key: chex.PRNGKey, boid_params: AgentParams) -> State:
        """Generate initial agent positions and velocities.

        Args:
            key: random key.
            boid_params: Searcher aagent `AgentParams`.

        Returns:
            Initial agent `State`.
        """


class RandomGenerator(Generator):
    def __call__(self, key: chex.PRNGKey, boid_params: AgentParams) -> State:
        boid_states = random_agent_state(
            key, boid_params, self.num_boids, self.env_size
        )
        state = State(
            boids=boid_states,
        )
        return state

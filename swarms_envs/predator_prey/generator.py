import abc

import chex
import jax.random
from jumanji.environments.swarms.common.types import AgentParams

from swarms_envs.common.utils import random_agent_state

from .types import State


class Generator(abc.ABC):
    def __init__(
        self, num_predators: int, num_prey: int, env_size: float = 1.0
    ) -> None:
        self.num_predators = num_predators
        self.num_prey = num_prey
        self.env_size = env_size

    @abc.abstractmethod
    def __call__(
        self, key: chex.PRNGKey, predator_params: AgentParams, prey_params: AgentParams
    ) -> State:
        """Generate initial agent positions and velocities.

        Args:
            key: random key.
            boid_params: Searcher aagent `AgentParams`.

        Returns:
            Initial agent `State`.
        """


class RandomGenerator(Generator):
    def __call__(
        self, key: chex.PRNGKey, predator_params: AgentParams, prey_params: AgentParams
    ) -> State:
        k_predator, k_prey = jax.random.split(key)
        predator_states = random_agent_state(
            k_predator, predator_params, self.num_predators, self.env_size
        )
        prey_states = random_agent_state(
            k_prey, prey_params, self.num_prey, self.env_size
        )
        state = State(
            predators=predator_states,
            prey=prey_states,
            step=0,
        )
        return state

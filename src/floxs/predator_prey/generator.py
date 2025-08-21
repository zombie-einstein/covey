"""Predator-prey environment state generator"""
import abc

import chex
import jax.random
from jumanji.environments.swarms.common.types import AgentParams

from floxs.common.utils import random_agent_state

from .types import State


class Generator(abc.ABC):
    def __init__(
        self, num_predators: int, num_prey: int, env_size: float = 1.0
    ) -> None:
        """
        State generator abstract class

        Parameters
        ----------
        num_predators
            Number of predator agents
        num_prey
            Number of prey agents
        env_size
            Environment size, default 1.0
        """
        self.num_predators = num_predators
        self.num_prey = num_prey
        self.env_size = env_size

    @abc.abstractmethod
    def __call__(
        self, key: chex.PRNGKey, predator_params: AgentParams, prey_params: AgentParams
    ) -> State:
        """Generate initial agent positions and velocities

        Parameters
        ----------
        key
            JAX random key
        predator_params
            Predator agent parameters
        prey_params
            Prey agent parameters

        Returns
        -------
        State
            Initial environment state
        """


class RandomGenerator(Generator):
    def __call__(
        self, key: chex.PRNGKey, predator_params: AgentParams, prey_params: AgentParams
    ) -> State:
        """
        Initial state generator that random samples agent states

        Random generator that samples predator and prey agent positions
        and velocities from a uniform distribution

        Parameters
        ----------
        key
            JAX random key
        predator_params
            Predator agent parameters
        prey_params
            Prey agent parameters

        Returns
        -------
        State
            Randomly sampled initial environment state
        """
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

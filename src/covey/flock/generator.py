"""Flock environment initial state generator"""
import abc

import chex
from jumanji.environments.swarms.common.types import AgentParams

from covey.common.utils import random_agent_state

from .types import State


class Generator(abc.ABC):
    """
    Initial state generation interface
    """

    def __init__(self, num_boids: int, env_size: float = 1.0) -> None:
        """
        Base generator initialiser

        Parameters
        ----------
        num_boids
            Number of boid agents in the flock
        env_size
            Size of the environment
        """
        self.num_boids = num_boids
        self.env_size = env_size

    @abc.abstractmethod
    def __call__(self, key: chex.PRNGKey, boid_params: AgentParams) -> State:
        """
        Generate initial state

        Parameters
        ----------
        key
            JAX random key
        boid_params
            Boid agent parameters

        Returns
        -------
        State
            Initial environment state
        """
        """Generate initial agent positions and velocities.

        Args:
            key: JAX random key
            boid_params: Agent `AgentParams` parameters

        Returns:
            Initial agent `State`.
        """


class RandomGenerator(Generator):
    """
    Random generator, sampling from a uniform distribution of positons and velocities
    """

    def __call__(self, key: chex.PRNGKey, boid_params: AgentParams) -> State:
        """
        Generate a randomly sampled state

        Parameters
        ----------
        key
            JAX random key
        boid_params
            Boid agent parameters

        Returns
        -------
        State
            Randomly sampled initial state
        """
        boid_states = random_agent_state(
            key, boid_params, self.num_boids, self.env_size
        )
        state = State(
            boids=boid_states,
        )
        return state

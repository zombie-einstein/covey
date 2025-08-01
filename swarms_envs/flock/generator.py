import abc

import chex
import jax
import jax.numpy as jnp
from jumanji.environments.swarms.common.types import AgentParams, AgentState

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

        k_pos, k_head, k_speed = jax.random.split(key, 3)
        positions = jax.random.uniform(
            k_pos, (self.num_boids, 2), minval=0.0, maxval=self.env_size
        )
        headings = jax.random.uniform(
            k_head, (self.num_boids,), minval=0.0, maxval=2.0 * jnp.pi
        )
        speeds = jax.random.uniform(
            k_speed,
            (self.num_boids,),
            minval=boid_params.min_speed,
            maxval=boid_params.max_speed,
        )
        boid_states = AgentState(
            pos=positions,
            speed=speeds,
            heading=headings,
        )

        state = State(
            boids=boid_states,
        )
        return state

import abc
from typing import Callable

import chex
import esquilax
import jax
import jax.numpy as jnp
from jumanji.environments.swarms.common.types import AgentState

from .types import State


def reward_fn(
    boid_radius: float,
    boid_a: AgentState,
    boid_b: AgentState,
    *,
    f: Callable,
    i_range: float,
):
    d = esquilax.utils.shortest_distance(boid_a.pos, boid_b.pos, norm=True)
    reward = f(d / i_range)
    return jax.lax.cond(d < 2 * boid_radius, lambda: (1, reward), lambda: (0, reward))


class RewardFn(abc.ABC):
    @abc.abstractmethod
    def __call__(self, state: State) -> chex.Array:
        """The reward function.

        Args:
            state: Env state

        Returns:
            Individual reward for each agent.
        """


class ExponentialRewardFn(RewardFn):
    def __init__(
        self,
        boid_radius: float,
        collision_penalty: float,
        i_range: float,
    ) -> None:
        self.boid_radius = boid_radius
        self.collision_penalty = collision_penalty
        self.i_range = i_range
        self.reward_fn = lambda d: jnp.exp(-5 * d)

    def __call__(self, state: State) -> chex.Array:
        collisions, rewards = esquilax.transforms.spatial(
            reward_fn,
            reduction=esquilax.reductions.Reduction((jnp.add, jnp.add), (0, 0.0)),
            include_self=False,
            topology="moore",
            i_range=self.i_range,
        )(
            self.boid_radius,
            state.boids,
            state.boids,
            pos=state.boids.pos,
            f=self.reward_fn,
            i_range=self.i_range,
        )
        rewards = jnp.where(collisions > 0, -self.collision_penalty, rewards)
        return rewards

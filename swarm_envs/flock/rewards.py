import abc
from typing import Callable, Union

import chex
import esquilax
import jax
import jax.numpy as jnp
from jumanji.environments.swarms.common.types import AgentState

from .types import State


def _apply_reward_fn(
    boid_radius: float,
    boid_a: AgentState,
    boid_b: AgentState,
    *,
    f: Callable,
    i_range: float,
) -> chex.Array:
    d = esquilax.utils.shortest_distance(boid_a.pos, boid_b.pos, norm=True)
    reward = f(d / i_range)
    return jax.lax.cond(d < 2 * boid_radius, lambda: (1, reward), lambda: (0, reward))


class RewardFn(abc.ABC):
    """
    Reward function interface
    """

    @abc.abstractmethod
    def __call__(self, state: State) -> chex.Array:
        """The reward function

        Parameters
        ----------
        state
            Env state

        Returns
        -------
        Array
            Individual reward for each agent
        """


class BaseDistanceRewardFn(RewardFn):
    """
    Base distance based reward function

    Base reward function that generated rewards based on the
    distance between individual agents (up to a given range).

    A negative penalty is applied if the agents collide.

    Rewards are accumulated between all pairs of agents within range.
    """

    def __init__(
        self,
        boid_radius: float,
        collision_penalty: float,
        i_range: float,
        reward_fn: Callable[[float], Union[float, chex.Array]],
    ) -> None:
        """
        Initialise distance reward function

        Parameters
        ----------
        boid_radius
            Radius of agents
        collision_penalty
            Penalty returned in case of colliding agents
        i_range
            Interaction range
        reward_fn
            Distance reward function
        """
        self.boid_radius = boid_radius
        self.collision_penalty = collision_penalty
        self.i_range = i_range
        self.reward_fn = reward_fn

    def __call__(self, state: State) -> chex.Array:
        """The reward function

        Parameters
        ----------
        state
            Env state

        Returns
        -------
            Individual reward for each agent
        """
        collisions, rewards = esquilax.transforms.spatial(
            _apply_reward_fn,
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


def _exponential_rewards(d: float) -> chex.Array:
    return jnp.exp(-5 * d)


class ExponentialRewardFn(BaseDistanceRewardFn):
    """
    Rewards that drop of exponentially with distance
    """

    def __init__(
        self,
        boid_radius: float,
        collision_penalty: float,
        i_range: float,
    ) -> None:
        """The reward function

        Parameters
        ----------
        state
            Env state

        Returns
        -------
            Individual reward for each agent
        """
        super().__init__(boid_radius, collision_penalty, i_range, _exponential_rewards)

"""Predator-prey environment reward functions"""
import abc

import esquilax
from jumanji.environments.swarms.common.types import AgentState

from .types import Rewards, State


class RewardFn(abc.ABC):
    """Predator-prey reward function interface"""

    @abc.abstractmethod
    def __call__(self, state: State) -> Rewards:
        """Generate predator and prey rewards from state

        Parameters
        ----------
        state
            Env state

        Returns
        -------
        Rewards
            Individual reward for each predator and prey agent
        """


class SparseRewards(RewardFn):
    def __init__(
        self,
        capture_radius: float,
        predator_reward: float,
        prey_penalty: float,
    ) -> None:
        """
        Sparse reward function

        Sparse reward function that provides a fixed reward/penalty
        to predators/prey when in capture range, irrespective of
        distance.

        Predator agents are provided a fixed reward if within capture range
        of a prey agent. Prey agents are provide a fixed reward summed
        over all prey agents within capture range.

        Parameters
        ----------
        capture_radius
            Range from predators at which prey are considered captured
        predator_reward
            Fixed reward provided to predators
        prey_penalty
            Fixed penalty provided to prey
        """
        self.capture_radius = capture_radius
        self.predator_reward = predator_reward
        self.prey_penalty = prey_penalty

    def __call__(self, state: State) -> Rewards:
        """
        Generate rewards

        Parameters
        ----------
        state
            Current environment state

        Returns
        -------
        Rewards
            Struct containing predator and prey reward arrays
        """
        prey_rewards = esquilax.transforms.spatial(
            lambda reward, _a, _b: -reward,
            reduction=esquilax.reductions.add(),
            include_self=False,
            i_range=self.capture_radius,
        )(
            self.prey_penalty,
            None,
            None,
            pos=state.prey.pos,
            pos_b=state.predators.pos,
        )
        predator_rewards = esquilax.transforms.nearest_neighbour(
            lambda reward, _a, _b: reward,
            default=0.0,
            i_range=self.capture_radius,
        )(
            self.predator_reward,
            None,
            None,
            pos=state.predators.pos,
            pos_b=state.prey.pos,
        )
        return Rewards(
            predators=predator_rewards,
            prey=prey_rewards,
        )


def _distance_reward(
    reward: float, a: AgentState, b: AgentState, *, i_range, env_size
) -> float:
    d = esquilax.utils.shortest_distance(a.pos, b.pos, env_size) / i_range
    return reward * (d - 1.0)


class LinearRewards(RewardFn):
    def __init__(
        self,
        capture_radius: float,
        predator_reward: float,
        prey_penalty: float,
        env_size: float,
    ) -> None:
        """
        Rewards function that scales reward by linear distance

        Reward function that assigns rewards/penalties to
        predator/prey if within capture range. The rewards
        are scaled linearly based on the distance between the
        predator and prey.

        Predator agents are provided a reward if within capture range
        of a prey agent. Prey agents are provide a reward summed
        over all prey agents within capture range.

        Parameters
        ----------
        capture_radius
            Agent capture radius
        predator_reward
            Max predator reward
        prey_penalty
            Max prey penalty
        env_size
            Environment size
        """
        self.capture_radius = capture_radius
        self.predator_reward = predator_reward
        self.prey_penalty = prey_penalty
        self.env_size = env_size

    def __call__(self, state: State) -> Rewards:
        """
        Generate rewards

        Parameters
        ----------
        state
            Current environment state

        Returns
        -------
        Rewards
            Struct containing predator and prey reward arrays
        """
        prey_rewards = esquilax.transforms.spatial(
            _distance_reward,
            reduction=esquilax.reductions.add(),
            include_self=False,
            i_range=2 * self.capture_radius,
        )(
            -self.prey_penalty,
            None,
            None,
            pos=state.prey.pos,
            pos_b=state.predators.pos,
            i_range=2 * self.capture_radius,
            env_size=self.env_size,
        )
        predator_rewards = esquilax.transforms.nearest_neighbour(
            _distance_reward,
            default=0.0,
            i_range=2 * self.capture_radius,
        )(
            self.predator_reward,
            None,
            None,
            pos=state.predators.pos,
            pos_b=state.prey.pos,
            i_range=2 * self.capture_radius,
            env_size=self.env_size,
        )
        return Rewards(
            predators=predator_rewards,
            prey=prey_rewards,
        )

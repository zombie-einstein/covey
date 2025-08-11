import abc

import esquilax
from jumanji.environments.swarms.common.types import AgentState

from .types import Rewards, State


class RewardFn(abc.ABC):
    @abc.abstractmethod
    def __call__(self, state: State) -> Rewards:
        """The reward function.

        Args:
            state: Env state

        Returns:
            Individual reward for each agent.
        """


class SparseRewards(RewardFn):
    def __init__(
        self,
        capture_radius: float,
        predator_reward: float,
        prey_penalty: float,
    ) -> None:
        self.capture_radius = capture_radius
        self.predator_reward = predator_reward
        self.prey_penalty = prey_penalty

    def __call__(self, state: State) -> Rewards:
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
        self.capture_radius = capture_radius
        self.predator_reward = predator_reward
        self.prey_penalty = prey_penalty
        self.env_size = env_size

    def __call__(self, state: State) -> Rewards:
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

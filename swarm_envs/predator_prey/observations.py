"""Predator-prey observation function"""
from typing import Tuple

import chex
import esquilax
import jax.numpy as jnp
from jumanji.environments.swarms.common.updates import view, view_reduction_fn

from .types import State


def _view_reduction(view_shape: Tuple[int, ...]) -> esquilax.reductions.Reduction:
    return esquilax.reductions.Reduction(
        fn=view_reduction_fn,
        id=-jnp.ones(view_shape),
    )


class ObservationFn:
    """
    Default predator-prey observation function

    Produces an individual local view of other agents in the
    environment for each agent. Each agents view is a
    `[2, num_vision]` array, where the values represent the
    distance along a ray to the nearest agent, with -1
    representing the case no agent is in range. The two
    rows represent the view of different types of agents.
    """

    def __init__(
        self,
        num_vision: int,
        predator_vision_range: float,
        prey_vision_range: float,
        predator_view_angle: float,
        prey_view_angle: float,
        agent_radius: float,
        env_size: float,
    ) -> None:
        """
        Initialise the observation function

        Parameters
        ----------
        num_vision
            Number of cells/values in the segmented view
        predator_vision_range
            Predator agent vision range
        prey_vision_range
            Prey vision range
        predator_view_angle
            Predator view angle, as a fraction of π
        prey_view_angle
            Prey view angle, as a fraction of π
        agent_radius
            Agent visual radius
        env_size
            Environment size
        """
        self.num_vision = num_vision
        self.predator_vision_range = predator_vision_range
        self.prey_vision_range = prey_vision_range
        self.predator_view_angle = predator_view_angle
        self.prey_view_angle = prey_view_angle
        self.agent_radius = agent_radius
        self.env_size = env_size

    def __call__(self, state: State) -> tuple[chex.Array, chex.Array]:
        """
        Generate predator and prey views from the current state

        Parameters
        ----------
        state
            Environment state

        Returns
        -------
        Array, Array
            Tuple containing predator and prey views respectively
        """
        prey_view_predators = esquilax.transforms.spatial(
            view,
            reduction=_view_reduction((self.num_vision,)),
            include_self=False,
            i_range=self.prey_vision_range,
            dims=self.env_size,
        )(
            (self.prey_view_angle, self.agent_radius),
            state.prey,
            state.predators,
            pos=state.prey.pos,
            pos_b=state.predators.pos,
            n_view=self.num_vision,
            i_range=self.prey_vision_range,
            env_size=self.env_size,
        )
        prey_view_prey = esquilax.transforms.spatial(
            view,
            reduction=_view_reduction((self.num_vision,)),
            include_self=False,
            i_range=self.prey_vision_range,
            dims=self.env_size,
        )(
            (self.prey_view_angle, self.agent_radius),
            state.prey,
            state.prey,
            pos=state.prey.pos,
            n_view=self.num_vision,
            i_range=self.prey_vision_range,
            env_size=self.env_size,
        )
        predators_view_prey = esquilax.transforms.spatial(
            view,
            reduction=_view_reduction((self.num_vision,)),
            include_self=False,
            i_range=self.predator_vision_range,
            dims=self.env_size,
        )(
            (self.predator_view_angle, self.agent_radius),
            state.predators,
            state.prey,
            pos=state.predators.pos,
            pos_b=state.prey.pos,
            n_view=self.num_vision,
            i_range=self.predator_vision_range,
            env_size=self.env_size,
        )
        predators_view_predators = esquilax.transforms.spatial(
            view,
            reduction=_view_reduction((self.num_vision,)),
            include_self=False,
            i_range=self.predator_vision_range,
            dims=self.env_size,
        )(
            (self.predator_view_angle, self.agent_radius),
            state.predators,
            state.predators,
            pos=state.predators.pos,
            n_view=self.num_vision,
            i_range=self.predator_vision_range,
            env_size=self.env_size,
        )

        prey_views = jnp.stack((prey_view_prey, prey_view_predators), axis=1)
        predator_views = jnp.stack(
            (predators_view_predators, predators_view_prey), axis=1
        )

        return predator_views, prey_views

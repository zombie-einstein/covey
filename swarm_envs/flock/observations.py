"""Agent observation function"""
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
    Base agent environment observation function
    """

    def __init__(
        self,
        num_vision: int,
        vision_range: float,
        view_angle: float,
        boid_radius: float,
        env_size: float,
    ) -> None:
        """
        Initialise base observation function

        Parameters
        ----------
        num_vision
            Number of segments/cells in the observation
        vision_range
            Vision of range of individual agents
        view_angle
            View angle of individual agents (as a fraction
            of pi from the agents heading)
        boid_radius
            Visual radius of the agents
        env_size
            Size of the environment
        """
        self.num_vision = num_vision
        self.vision_range = vision_range
        self.view_angle = view_angle
        self.boid_radius = boid_radius
        self.env_size = env_size

    def __call__(self, state: State) -> chex.Array:
        """

        Parameters
        ----------
        state
            Current environment state

        Returns
        -------
        Array
            Array of individual agent observation arrays, in
            shape `[n-agents, n-observation]`
        """
        views = esquilax.transforms.spatial(
            view,
            reduction=_view_reduction((self.num_vision,)),
            include_self=False,
            i_range=self.vision_range,
            dims=self.env_size,
        )(
            (self.view_angle, self.boid_radius),
            state.boids,
            state.boids,
            pos=state.boids.pos,
            n_view=self.num_vision,
            i_range=self.vision_range,
            env_size=self.env_size,
        )
        return views

from typing import Tuple

import chex
import esquilax
import jax.numpy as jnp
from jumanji.environments.swarms.common.updates import view, view_reduction_fn

from .types import State


def view_reduction(view_shape: Tuple[int, ...]) -> esquilax.reductions.Reduction:
    return esquilax.reductions.Reduction(
        fn=view_reduction_fn,
        id=-jnp.ones(view_shape),
    )


class ObservationFn:
    def __init__(
        self,
        num_vision: int,
        vision_range: float,
        view_angle: float,
        boid_radius: float,
        env_size: float,
    ) -> None:
        self.num_vision = num_vision
        self.vision_range = vision_range
        self.view_angle = view_angle
        self.boid_radius = boid_radius
        self.env_size = env_size

    def __call__(self, state: State) -> chex.Array:
        views = esquilax.transforms.spatial(
            view,
            reduction=view_reduction((self.num_vision,)),
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

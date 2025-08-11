from functools import cached_property, partial
from typing import Optional, Sequence

import chex
import jax
import jax.numpy as jnp
from jumanji import specs
from jumanji.env import Environment
from jumanji.environments.swarms.common.types import AgentParams
from jumanji.environments.swarms.common.updates import update_state
from jumanji.types import TimeStep, restart, termination, transition
from jumanji.viewer import Viewer
from matplotlib.animation import FuncAnimation

from .generator import Generator, RandomGenerator
from .observations import ObservationFn
from .rewards import ExponentialRewardFn, RewardFn
from .types import Observation, State
from .viewer import FlockViewer


class Flock(Environment):
    def __init__(
        self,
        boid_max_rotate: float = 0.025,
        boid_max_accelerate: float = 0.001,
        boid_min_speed: float = 0.015,
        boid_max_speed: float = 0.025,
        time_limit: int = 400,
        boid_radius: float = 0.01,
        viewer: Optional[Viewer[State]] = None,
        generator: Optional[Generator] = None,
        reward_fn: Optional[RewardFn] = None,
        observation: Optional[ObservationFn] = None,
    ) -> None:
        self.boid_params = AgentParams(
            max_rotate=boid_max_rotate,
            max_accelerate=boid_max_accelerate,
            min_speed=boid_min_speed,
            max_speed=boid_max_speed,
        )
        self.time_limit = time_limit
        self.generator = generator or RandomGenerator(num_boids=50)
        self._viewer = viewer or FlockViewer()
        self._reward_fn = reward_fn or ExponentialRewardFn(
            boid_radius=boid_radius,
            collision_penalty=0.1,
            i_range=0.2,
        )
        self._observation_fn = observation or ObservationFn(
            num_vision=128,
            vision_range=0.2,
            view_angle=0.5,
            env_size=self.generator.env_size,
            boid_radius=boid_radius,
        )
        super().__init__()

    def __repr__(self) -> str:
        return "\n".join(
            [
                "Boid flock multi-agent environment:",
                f" - num boids: {self.generator.num_boids}",
                f" - boid max rotation: {self.boid_params.max_rotate}",
                f" - boid max acceleration: {self.boid_params.max_accelerate}",
                f" - boid min speed: {self.boid_params.min_speed}",
                f" - boid max speed: {self.boid_params.max_speed}",
                f" - boid vision range: {self._observation_fn.vision_range}",
                f" - boid view angle: {self._observation_fn.view_angle}",
                f" - num vision: {self._observation_fn.num_vision}",
                f" - boid radius: {self._observation_fn.boid_radius}",
                f" - time limit: {self.time_limit},"
                f" - env size: {self.generator.env_size}"
                f" - generator: {self.generator.__class__.__name__}",
                f" - reward fn: {self._reward_fn.__class__.__name__}",
                f" - observation fn: {self._observation_fn.__class__.__name__}",
            ]
        )

    def reset(self, key: chex.PRNGKey) -> tuple[State, TimeStep[Observation]]:
        state = self.generator(key, self.boid_params)
        timestep = restart(
            observation=self._state_to_observation(state), shape=(self.num_agents,)
        )
        return state, timestep

    def step(
        self, state: State, actions: chex.Array
    ) -> tuple[State, TimeStep[Observation]]:
        boids = update_state(
            self.generator.env_size, self.boid_params, state.boids, actions
        )
        state = State(boids=boids, step=state.step + 1)
        rewards = self._reward_fn(state)
        observation = self._state_to_observation(state)
        observation = jax.lax.stop_gradient(observation)
        timestep = jax.lax.cond(
            state.step >= self.time_limit,
            partial(termination, shape=(self.num_agents,)),
            partial(transition, shape=(self.num_agents,)),
            rewards,
            observation,
        )
        return state, timestep

    def _state_to_observation(self, state: State) -> Observation:
        boid_views = self._observation_fn(state)
        return Observation(
            views=boid_views,
            step=state.step,
            positions=state.boids.pos / self.generator.env_size,
        )

    @cached_property
    def num_agents(self) -> int:
        return self.generator.num_boids

    @cached_property
    def observation_spec(self) -> specs.Spec[Observation]:
        """Returns the observation spec.

        Local searcher agent views representing the distance to the
        closest neighbouring agents and targets in the environment.

        Returns:
            observation_spec: Search-and-rescue observation spec
        """
        boid_views = specs.BoundedArray(
            shape=(
                self.num_agents,
                self._observation_fn.num_vision,
            ),
            minimum=-1.0,
            maximum=1.0,
            dtype=float,
            name="boid_views",
        )
        return specs.Spec(
            Observation,
            "ObservationSpec",
            views=boid_views,
            step=specs.BoundedArray(
                shape=(),
                minimum=0,
                maximum=self.time_limit,
                name="step",
                dtype=jnp.int32,
            ),
            positions=specs.BoundedArray(
                shape=(self.num_agents, 2),
                minimum=0.0,
                maximum=1.0,
                name="positions",
                dtype=jnp.float32,
            ),
        )

    @cached_property
    def action_spec(self) -> specs.BoundedArray:
        """Returns the action spec.

        2d array of individual agent actions. Each agents action is
        an array representing [rotation, acceleration] in the range
        [-1, 1].

        Returns:
            action_spec: Action array spec
        """
        return specs.BoundedArray(
            shape=(self.generator.num_boids, 2),
            minimum=-1.0,
            maximum=1.0,
            dtype=float,
        )

    @cached_property
    def reward_spec(self) -> specs.BoundedArray:
        """Returns the reward spec.

        Array of individual rewards for each agent.

        Returns:
            reward_spec: Reward array spec.
        """
        return specs.BoundedArray(
            shape=(self.generator.num_boids,),
            minimum=0.0,
            maximum=1.0,
            dtype=float,
        )

    def render(self, state: State) -> None:
        """Render a frame of the environment for a given state using matplotlib.

        Args:
            state: State object.
        """
        self._viewer.render(state)

    def animate(
        self,
        states: Sequence[State],
        interval: int = 100,
        save_path: Optional[str] = None,
    ) -> FuncAnimation:
        """Create an animation from a sequence of environment states.

        Args:
            states: sequence of environment states corresponding to consecutive
                timesteps.
            interval: delay between frames in milliseconds.
            save_path: the path where the animation file should be saved. If it
                is None, the plot will not be saved.

        Returns:
            Animation that can be saved as a GIF, MP4, or rendered with HTML.
        """
        return self._viewer.animate(states, interval=interval, save_path=save_path)

    def close(self) -> None:
        """Perform any necessary cleanup."""
        self._viewer.close()

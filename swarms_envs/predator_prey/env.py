"""Predator-prey multi-agent environment"""
from functools import cached_property
from typing import Optional, Sequence

import chex
import jax
import jax.numpy as jnp
from jumanji import specs
from jumanji.env import Environment
from jumanji.environments.swarms.common.types import AgentParams
from jumanji.environments.swarms.common.updates import update_state
from jumanji.types import StepType, TimeStep
from jumanji.viewer import Viewer
from matplotlib.animation import FuncAnimation

from .generator import Generator, RandomGenerator
from .observations import ObservationFn
from .rewards import RewardFn, SparseRewards
from .types import Actions, Discounts, Observation, Rewards, State
from .viewer import PredatorPreyViewer


class PredatorPrey(Environment):
    """
    Predator-prey multi-agent RL environment

    Environment containing two distinct agent types, predators and prey.
    The predator agents are rewarded for coming within capture range of
    the prey agents. Conversely, the prey agents are penalised if within
    capture range of a prey agent.
    """

    def __init__(
        self,
        prey_max_rotate: float = 0.025,
        prey_max_accelerate: float = 0.001,
        prey_min_speed: float = 0.015,
        prey_max_speed: float = 0.025,
        predator_max_rotate: float = 0.025,
        predator_max_accelerate: float = 0.001,
        predator_min_speed: float = 0.015,
        predator_max_speed: float = 0.025,
        time_limit: int = 400,
        viewer: Optional[Viewer[State]] = None,
        generator: Optional[Generator] = None,
        reward_fn: Optional[RewardFn] = None,
        observation: Optional[ObservationFn] = None,
    ) -> None:
        """
        Initialise a predator-prey environment

        Parameters
        ----------
        prey_max_rotate
            Max prey agent change in heading in a single
            step, as a fraction of π
        prey_max_accelerate
            Max prey agent change in speed in a single
            step, as a fraction of π
        prey_min_speed
            Prey agent minimum speed
        prey_max_speed
            Prey agent maximum speed
        predator_max_rotate
            Max predator agent change in heading in a single
            step, as a fraction of π
        predator_max_accelerate
            Max predator agent change in speed in a single
            step, as a fraction of π
        predator_min_speed
            Predator agent minimum speed
        predator_max_speed
            Predator agent maximum speed
        time_limit
            Environment time limit
        viewer
            Plot/image generator, default uses the Matplotlib backend
        generator
            Initial state generator, default generates a uniform random
            distribution of predator and prey agents
        reward_fn
            Reward function, default provides a reward of 0.1 to
            predators if within range of a prey agent, and a -0.1
            penalty to prey for each predator within capture range
        observation
            Agent observation function, default generates a segmented
            view with 2 channels, containing the distance to agents
            of the same-type, and opposite type in different channels
        """
        self.predator_params = AgentParams(
            max_rotate=predator_max_rotate,
            max_accelerate=predator_max_accelerate,
            min_speed=predator_min_speed,
            max_speed=predator_max_speed,
        )
        self.prey_params = AgentParams(
            max_rotate=prey_max_rotate,
            max_accelerate=prey_max_accelerate,
            min_speed=prey_min_speed,
            max_speed=prey_max_speed,
        )
        self.time_limit = time_limit
        self.generator = generator or RandomGenerator(num_predators=2, num_prey=20)
        self._viewer = viewer or PredatorPreyViewer()
        self._reward_fn = reward_fn or SparseRewards(
            capture_radius=0.02,
            predator_reward=0.1,
            prey_penalty=0.1,
        )
        self._observation_fn = observation or ObservationFn(
            num_vision=128,
            predator_vision_range=0.2,
            prey_vision_range=0.1,
            predator_view_angle=0.25,
            prey_view_angle=0.6,
            env_size=self.generator.env_size,
            agent_radius=0.01,
        )
        super().__init__()

    def __repr__(self) -> str:
        predator_vision_range = self._observation_fn.predator_vision_range
        return "\n".join(
            [
                "Boid flock multi-agent environment:",
                f" - num predators: {self.generator.num_predators}",
                f" - num prey: {self.generator.num_prey}",
                f" - predator max rotation: {self.predator_params.max_rotate}",
                f" - predator max acceleration: {self.predator_params.max_accelerate}",
                f" - predator min speed: {self.predator_params.min_speed}",
                f" - predator max speed: {self.predator_params.max_speed}",
                f" - prey max rotation: {self.prey_params.max_rotate}",
                f" - prey max acceleration: {self.prey_params.max_accelerate}",
                f" - prey min speed: {self.prey_params.min_speed}",
                f" - prey max speed: {self.prey_params.max_speed}",
                f" - predator vision range: {predator_vision_range}",
                f" - prey vision range: {self._observation_fn.prey_vision_range}",
                f" - predator view angle: {self._observation_fn.predator_view_angle}",
                f" - prey view angle: {self._observation_fn.prey_view_angle}",
                f" - num vision: {self._observation_fn.num_vision}",
                f" - agent radius: {self._observation_fn.agent_radius}",
                f" - time limit: {self.time_limit},"
                f" - env size: {self.generator.env_size}"
                f" - generator: {self.generator.__class__.__name__}",
                f" - reward fn: {self._reward_fn.__class__.__name__}",
                f" - observation fn: {self._observation_fn.__class__.__name__}",
            ]
        )

    def reset(self, key: chex.PRNGKey) -> tuple[State, TimeStep[Observation]]:
        """
        Reset the environment

        Parameters
        ----------
        key
            JAX random key

        Returns
        -------
        State, TimeStep
            New initial state and initial timestep
        """
        state = self.generator(key, self.predator_params, self.prey_params)
        timestep = TimeStep(
            step_type=StepType.FIRST,
            reward=Rewards(
                predators=jnp.zeros(self.generator.num_predators, dtype=float),
                prey=jnp.zeros(self.generator.num_prey, dtype=float),
            ),
            discount=Discounts(
                predators=jnp.ones(self.generator.num_predators, dtype=float),
                prey=jnp.ones(self.generator.num_prey, dtype=float),
            ),
            observation=self._state_to_observation(state),
            extras={},
        )
        return state, timestep

    def step(
        self, state: State, actions: Actions
    ) -> tuple[State, TimeStep[Observation]]:
        """
        Apply actions and update the state of the environment

        The environment is updated in several steps:

        - Apply predator and prey actions
        - Update predator and prey positions
        - Calculate new individual rewards for each agent type
        - Generate individual views for each agent and type

        Parameters
        ----------
        state
            Current environment state
        actions
            Actions struct containing predator and prey action arrays

        Returns
        -------
        State, TimeStep
            Updated environment state and new timestep
        """
        predators = update_state(
            self.generator.env_size,
            self.predator_params,
            state.predators,
            actions.predators,
        )
        prey = update_state(
            self.generator.env_size, self.prey_params, state.prey, actions.prey
        )
        state = State(predators=predators, prey=prey, step=state.step + 1)
        rewards = self._reward_fn(state)
        observation = self._state_to_observation(state)
        observation = jax.lax.stop_gradient(observation)
        step_type, discounts = jax.lax.cond(
            state.step >= self.time_limit,
            lambda: (
                StepType.LAST,
                Discounts(
                    predators=jnp.zeros(self.generator.num_predators, dtype=float),
                    prey=jnp.zeros(self.generator.num_prey, dtype=float),
                ),
            ),
            lambda: (
                StepType.MID,
                Discounts(
                    predators=jnp.ones(self.generator.num_predators, dtype=float),
                    prey=jnp.ones(self.generator.num_prey, dtype=float),
                ),
            ),
        )
        timestep = TimeStep(
            step_type=step_type,
            reward=rewards,
            discount=discounts,
            observation=observation,
            extras={},
        )
        return state, timestep

    def _state_to_observation(self, state: State) -> Observation:
        predator_views, prey_views = self._observation_fn(state)
        return Observation(
            predator_views=predator_views,
            prey_views=prey_views,
            step=state.step,
            predator_positions=state.predators.pos / self.generator.env_size,
            prey_positions=state.prey.pos / self.generator.env_size,
        )

    @cached_property
    def observation_spec(self) -> specs.Spec[Observation]:
        """Returns the observation spec

        Local predator and prey agent views representing the distance to the
        closest neighbouring agents of each type

        Returns
        -------
        Spec
            Predator-prey environment observation spec
        """
        predator_views = specs.BoundedArray(
            shape=(
                self.generator.num_predators,
                2,
                self._observation_fn.num_vision,
            ),
            minimum=-1.0,
            maximum=1.0,
            dtype=float,
            name="predator_view",
        )
        prey_views = specs.BoundedArray(
            shape=(
                self.generator.num_prey,
                2,
                self._observation_fn.num_vision,
            ),
            minimum=-1.0,
            maximum=1.0,
            dtype=float,
            name="prey_view",
        )
        return specs.Spec(
            Observation,
            "ObservationSpec",
            predator_views=predator_views,
            prey_views=prey_views,
            step=specs.BoundedArray(
                shape=(),
                minimum=0,
                maximum=self.time_limit,
                name="step",
                dtype=jnp.int32,
            ),
            predator_positions=specs.BoundedArray(
                shape=(self.generator.num_predators, 2),
                minimum=0.0,
                maximum=1.0,
                name="positions",
                dtype=jnp.float32,
            ),
            prey_positions=specs.BoundedArray(
                shape=(self.generator.num_prey, 2),
                minimum=0.0,
                maximum=1.0,
                name="positions",
                dtype=jnp.float32,
            ),
        )

    @cached_property
    def action_spec(self) -> specs.Spec[Actions]:
        """Returns the action spec

        Actions struct containing predator and prey action arrays

        Returns
        -------
        Spec
            Action array spec
        """
        return specs.Spec(
            Actions,
            predators=specs.BoundedArray(
                shape=(self.generator.num_predators, 2),
                minimum=-1.0,
                maximum=1.0,
                dtype=float,
            ),
            prey=specs.BoundedArray(
                shape=(self.generator.num_prey, 2),
                minimum=-1.0,
                maximum=1.0,
                dtype=float,
            ),
        )

    @cached_property
    def reward_spec(self) -> specs.Spec[Rewards]:
        """Returns the reward spec

        Rewards struct, containing individual rewards for
        predator and prey agents

        Returns
        -------
        Spec
            Reward spec
        """
        return specs.Spec(
            Rewards,
            predators=specs.Array(
                shape=(self.generator.num_predators,),
                dtype=float,
            ),
            prey=specs.Array(
                shape=(self.generator.num_prey,),
                dtype=float,
            ),
        )

    def render(self, state: State) -> None:
        """Render a frame of the environment for a given state using matplotlib.

        Parameters
        ----------
        state
            State object
        """
        self._viewer.render(state)

    def animate(
        self,
        states: Sequence[State],
        interval: int = 100,
        save_path: Optional[str] = None,
    ) -> FuncAnimation:
        """Create an animation from a sequence of environment states.

        Parameters
        ----------
        states
            Sequence of environment states corresponding to consecutive
            timesteps.
        interval
            Delay between frames in milliseconds.
        save_path
            The path where the animation file should be saved. If it
            is None, the plot will not be saved.

        Returns
        -------
        FuncAnimation
            Animation that can be saved as a GIF, MP4, or rendered with HTML
        """
        return self._viewer.animate(states, interval=interval, save_path=save_path)

    def close(self) -> None:
        """Perform any necessary cleanup."""
        self._viewer.close()

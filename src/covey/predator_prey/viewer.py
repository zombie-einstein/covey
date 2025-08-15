"""Predator-prey environment visualisation"""
from typing import Optional, Sequence, Tuple

import jax.numpy as jnp
import matplotlib.animation
import matplotlib.pyplot as plt
from jumanji.environments.swarms.common.viewer import draw_agents, format_plot
from jumanji.viewer import MatplotlibViewer
from matplotlib.artist import Artist
from numpy.typing import NDArray

from .types import State


class PredatorPreyViewer(MatplotlibViewer[State]):
    def __init__(
        self,
        name: str = "predator_prey",
        env_size: Tuple[float, float] = (1.0, 1.0),
        prey_color: str = "#5ec962",
        predator_color: str = "#440154",
        render_mode: str = "human",
    ) -> None:
        """
        Predator-prey environment visualiser

        Parameters
        ----------
        name
            Plot name, default ``predator_prey``
        env_size
            Tuple containing the dimensions of the environment
        prey_color
            Color applied to prey agents
        predator_color
            Color applied to predator agents
        render_mode
            Default ``human``
        """
        self.env_size = env_size
        self.prey_color = prey_color
        self.predator_color = predator_color
        super().__init__(name, render_mode)

    def render(
        self, state: State, save_path: Optional[str] = None
    ) -> Optional[NDArray]:
        """Render a frame of the environment for a given state using matplotlib.

        Parameters
        ----------
        state
            State object containing the current dynamics of the environment.
        save_path
            Optional path to save the rendered environment image to.

        Returns
        -------
        Array
            RGB array if the render_mode is 'rgb_array'.
        """
        self._clear_display()
        fig, ax = self._get_fig_ax()
        self._draw(ax, state)

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", pad_inches=0.2)

        return self._display(fig)

    def animate(
        self, states: Sequence[State], interval: int, save_path: Optional[str] = None
    ) -> matplotlib.animation.FuncAnimation:
        """Create an animation from a sequence of states.

        Parameters
        ----------
        states
            Sequence of ``State`` objects corresponding to subsequent timesteps.
        interval
            Delay between frames in milliseconds, default to 200.
        save_path
            The path where the animation file should be saved. If it is None,
            the plot will not be saved

        Returns
        -------
        FuncAnimation
            Animation object that can be saved as a GIF, MP4, or rendered with HTML
        """
        if not states:
            raise ValueError(f"The states argument has to be non-empty, got {states}.")
        fig, ax = self._get_fig_ax(name_suffix="_animation", show=False)
        fig, ax = format_plot(fig, ax, self.env_size)
        plt.close(fig=fig)

        prey_quiver = draw_agents(ax, states[0].prey, self.prey_color)
        predator_quiver = draw_agents(ax, states[0].predators, self.predator_color)

        def make_frame(state: State) -> tuple[Artist, Artist]:
            prey_quiver.set_offsets(state.prey.pos)
            prey_quiver.set_UVC(
                jnp.cos(state.prey.heading), jnp.sin(state.prey.heading)
            )
            predator_quiver.set_offsets(state.predators.pos)
            predator_quiver.set_UVC(
                jnp.cos(state.predators.heading), jnp.sin(state.predators.heading)
            )
            return prey_quiver, predator_quiver

        self._animation = matplotlib.animation.FuncAnimation(
            fig,
            make_frame,
            frames=states,
            interval=interval,
            blit=True,
        )

        if save_path:
            self._animation.save(save_path)

        return self._animation

    def _draw(self, ax: plt.Axes, state: State) -> None:
        ax.clear()
        draw_agents(ax, state.prey, self.prey_color)
        draw_agents(ax, state.predators, self.predator_color)

    def _get_fig_ax(
        self,
        name_suffix: Optional[str] = None,
        show: bool = True,
        padding: float = 0.05,
        **fig_kwargs: str,
    ) -> Tuple[plt.Figure, plt.Axes]:
        fig, ax = super()._get_fig_ax(
            name_suffix=name_suffix, show=show, padding=padding
        )
        fig, ax = format_plot(fig, ax, self.env_size)
        return fig, ax

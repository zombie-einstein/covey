from typing import TYPE_CHECKING, NamedTuple

import chex
from jumanji.environments.swarms.common.types import AgentState

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from chex import dataclass


@dataclass
class State:
    """
    Environment state

    boids
        State of flock agents
    step
        Current simulation step
    """

    boids: AgentState
    step: int = 0


class Observation(NamedTuple):
    """
    Agent observations

    views
        Array of individual agent views on the environment
    step
        Current environment step
    positions
        Positions of individual agents in the environment
    """

    views: chex.Array  # (num_boids, num_vision)
    step: chex.Numeric  # ()
    positions: chex.Array  # (num_boids, 2)

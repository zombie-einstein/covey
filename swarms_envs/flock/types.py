from typing import TYPE_CHECKING, NamedTuple

import chex
from jumanji.environments.swarms.common.types import AgentState

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from chex import dataclass


@dataclass
class State:
    boids: AgentState
    step: int = 0


class Observation(NamedTuple):
    views: chex.Array  # (num_boids, num_vision)
    step: chex.Numeric  # ()
    positions: chex.Array  # (num_boids, 2)

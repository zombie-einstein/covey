from typing import TYPE_CHECKING, NamedTuple

import chex
from jumanji.environments.swarms.common.types import AgentState

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from chex import dataclass


@dataclass
class Actions:
    predators: chex.Array
    prey: chex.Array


@dataclass
class Rewards:
    predators: chex.Array
    prey: chex.Array


@dataclass
class Discounts:
    predators: chex.Array
    prey: chex.Array


@dataclass
class State:
    predators: AgentState
    prey: AgentState
    step: int = 0


class Observation(NamedTuple):
    predator_views: chex.Array  # (num_predators, 2, num_vision)
    prey_views: chex.Array  # (num_prey, 2, num_vision)
    step: chex.Numeric  # ()
    predator_positions: chex.Array  # (num_predators, 2)
    prey_positions: chex.Array  # (num_prey, 2)

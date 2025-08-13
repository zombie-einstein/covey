"""Predator-prey environment data types"""
from typing import TYPE_CHECKING, NamedTuple

import chex
from jumanji.environments.swarms.common.types import AgentState

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from chex import dataclass


@dataclass
class Actions:
    """
    Predator-prey actions

    predators
        Array of predator actions
    prey
        Array of prey actions
    """

    predators: chex.Array  # (num_predators, 2)
    prey: chex.Array  # (num_prey, 2)


@dataclass
class Rewards:
    """
    Predator-prey rewards

    predators
        Array of predator rewards
    prey
        Array of prey rewards
    """

    predators: chex.Array  # (num_predators,)
    prey: chex.Array  # (num_prey,)


@dataclass
class Discounts:
    """
    Predator-prey discounts

    predators
        Array of predator discounts
    prey
        Array of prey discounts
    """

    predators: chex.Array  # (num_predators,)
    prey: chex.Array  # (num_prey,)


@dataclass
class State:
    """
    Predator-prey environment state

    predators
        Predator agent states
    prey
        Prey agent states
    step
        Simulation step
    """

    predators: AgentState
    prey: AgentState
    step: int = 0


class Observation(NamedTuple):
    """
    Predator-prey observations

    predator_views
        Array of individual predator agent views
    prey_views
        Array of individual prey agent views
    step
        Simulation step
    predator_positions
        Array of predator agent positions in the environment
    prey_positions
        Array of prey agent positions in the environment
    """

    predator_views: chex.Array  # (num_predators, 2, num_vision)
    prey_views: chex.Array  # (num_prey, 2, num_vision)
    step: chex.Numeric  # ()
    predator_positions: chex.Array  # (num_predators, 2)
    prey_positions: chex.Array  # (num_prey, 2)

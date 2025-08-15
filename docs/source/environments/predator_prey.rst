Predator-Prey
=============

Multi-agent RL environment with two competing flocks/teams of agents:

- Predator agents attempt to capture prey agents
- Prey agents attempt to evade predator agents

See :py:class:`covey.predator_prey.PredatorPrey` for details of the
environment API.

Dynamics
--------
All agent states consist of their position and velocity
(represented in polar coordinates as a heading an speed).
Each step all agent positions are updated from their current
velocity, and consequently their new rewards and observations
generated.

The space is wrapped at the edges (i.e. it forms a torus).

Actions
-------
Each agent can individually updated their velocity each step. Each agents
actions is an array of two continuous values in the range ``[-1, 1]``,
where the values represent ``[rotation, acceleration]``. The action values
are then scaled by the maximum rotation and acceleration parameters for
each agent type. In total the actions for the flock are given by arrays of shape
``[n-predators, 2]`` and ``[n-prey, 2]``, representing the velocity update
for each individual agent.

Rewards
-------
Agents are individually rewarded on their proximity to other agents.

- Predator agents are positively rewarded for coming within capture
  range of a prey agent
- Prey agents are penalised if within range of a prey agent, with penalties
  accumulated over all predators in range

By default rewards are independent of distance, i.e. they are a binary
fixed rewards when predator/prey are in range.

Rewards can be customised by implementing the
:py:class:`covey.predator_prey.rewards.RewardFn` interface.

Observations
------------
By default each agent individually observes their local neighbourhood
of the environment, as a segmented view. Each agents view is a 2d array
of shape ``[2, n_vision]`` where each row represents a view of each
agent type (i.e. predator or prey). The view cone of each agent
is divided into segments, with values representing the distance to the closest
neighbour along a ray cast from the agent. In the case that no agent lies within
range, then the default value is -1.

Observations can be customized by extending the default
:py:class:`covey.predator_prey.observations.ObservationFn` observation class.

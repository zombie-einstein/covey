Flock
=====

Multi-agent RL environment inspired by Reynolds popular `boids model`_
of animal flocks and swarms.

Agents attempt to remain close to other members of the flock whilst
avoiding colliding with other agents.

See :py:class:`covey.flock.Flock` for details of the environment API.

Dynamics
--------
The agent state consists of their position on the space, and velocity
(represented in polar co-ordinates as a heading and speed). Each step
agent positions are updated from their current velocity, and
consequently their new rewards and observations generated.

The space is wrapped at the edges (i.e. it forms a torus).

Actions
-------
Each agent can individually updated their velocity each step. Each agents
actions is an array of two continuous values in the range ``[-1, 1]``,
where the values represent ``[rotation, acceleration]``. The action values
are then scaled by the maximum rotation and acceleration parameters.
In total the actions for the flock are given by an array of shape
``[n-agents, 2]``, representing the velocity update for each individual
agent.

Rewards
-------
Agents are individually rewarded based on their proximity to other agents
in the flock:

- A positive reward when a neighbour is within a fixed neighbourhood, summed
  over contributing neighbours
- A fixed negative penalty when any agent collides

By default the reward provided by in range neighbours decrease exponentially
with distance.

Rewards can be customised by implementing the :py:class:`covey.flock.rewards.RewardFn`
interface.

Observations
------------

By default each agent individually observes their local neighbourhood
of the environment, as a segmented view. The view cone of each agent
is divided into segments, with values representing the distance to the closest
neighbour along a ray cast from the agent. In the case that no agent lies within
range, then the default value is -1.

Observations can be customized by extending the default
:py:class:`covey.flock.observations.ObservationFn` observation class.

.. _boids model: https://en.wikipedia.org/wiki/Boids

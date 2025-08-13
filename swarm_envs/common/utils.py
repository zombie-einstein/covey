"""Common utilities"""
import chex
import jax
import jax.numpy as jnp
from jumanji.environments.swarms.common.types import AgentParams, AgentState


def random_agent_state(
    key: chex.PRNGKey, agent_params: AgentParams, n_agents: int, env_size: float
) -> AgentState:
    """
    Initialise a random agent state

    Sample agent position, heading, and speed from a uniform distribution.

    Parameters
    ----------
    key
        JAX random key
    agent_params
        Agent parameters
    n_agents
        Number of agents to generate
    env_size
        Size of the environment

    Returns
    -------
    AgentState
        Random agent states
    """
    k_pos, k_head, k_speed = jax.random.split(key, 3)
    positions = jax.random.uniform(k_pos, (n_agents, 2), minval=0.0, maxval=env_size)
    headings = jax.random.uniform(k_head, (n_agents,), minval=0.0, maxval=2.0 * jnp.pi)
    speeds = jax.random.uniform(
        k_speed,
        (n_agents,),
        minval=agent_params.min_speed,
        maxval=agent_params.max_speed,
    )
    return AgentState(
        pos=positions,
        speed=speeds,
        heading=headings,
    )

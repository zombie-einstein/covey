import chex
import jax
import jax.numpy as jnp
from jumanji.environments.swarms.common.types import AgentParams, AgentState


def random_agent_state(
    key: chex.PRNGKey, agent_params: AgentParams, n_agents: int, env_size: float
) -> AgentState:
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

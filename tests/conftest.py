import chex
import jax
import pytest


@pytest.fixture
def key() -> chex.PRNGKey:
    return jax.random.PRNGKey(101)

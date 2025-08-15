# Covey

**Swarm and Flock Multi-agent RL environments, implemented with [JAX](https://github.com/google/jax) using [Esquilax](https://zombie-einstein.github.io/esquilax/)**

<p float="left">
  <img src=".github/images/rl_boids001.gif?raw=true" width="300" />
  <img src=".github/images/rl_boids002.gif?raw=true" width="300" />
</p>

Multi-agent RL environments with a focus on larger scale flocks and swarms on continuos
spaces. Implemented using [Esquilax](https://zombie-einstein.github.io/esquilax/) a JAX multi-agent simulation and RL
library, and the [Jumanji](https://github.com/instadeepai/jumanji) RL environment API.

```python
from covey.flock.env import Flock
import jax


env = Flock()

key = jax.random.PRNGKey(101)
state, ts = env.reset(key)
states = [state]

for _ in range(100):
    key, k = jax.random.split(key)
    actions = jax.random.uniform(k, (env.generator.num_boids, 2), minval=0.5, maxval=1.0)
    state, ts = env.step(state, actions)
    states.append(state)

# Save an animation of the environment
env.animate(states, interval=100, save_path="animation.gif")
```

See the [Jumanji docs](https://instadeepai.github.io/jumanji/) for more usage information.

## Usage

The package and requirements can be installed using [poetry](https://python-poetry.org/docs/)
by running

```shell
poetry install
```

## Developers

### Pre-Commit Hooks

Pre commit hooks can be installed by running

```bash
pre-commit install
```

Pre-commit checks can then be run using

```bash
task lint
```

### Tests

Tests can be run with

```bash
task test
```

### Build Documentation

Docs can be built using
[Sphinx](https://www.sphinx-doc.org/en/master/)
by running

```bash
task docs
```

Built docs will be generated in the `docs/build` folder.

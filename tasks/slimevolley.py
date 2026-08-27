"""Slime Volleyball: evaluating genomes against the built-in AI.

The environment comes from EvoJAX and is fully vectorised, so an entire
population plays simultaneously inside one ``jax.lax.scan`` -- the population
axis of the stacked genome pytree lines up with the environment's batch axis.
"""

import functools
import glob

import jax
import jax.numpy as jnp
import numpy as np
from evojax.task.slimevolley import SlimeVolley

from neat import config as cfg
from neat.genome import forwardpass, load_genome

N_IN, N_OUT = 12, 3  # 12-dim observation, 3 binary actions (forward, back, jump)

_TASKS = {}


def get_task(test=False):
    """The shared environment instance, built on first use.

    Creating it lazily matters: episode length is a hyperparameter, so the task
    must not be constructed at import time, before the config preset is chosen.
    """
    max_steps = cfg.TEST_MAX_STEPS if test else cfg.MAX_STEPS
    cache_key = (test, max_steps)
    if cache_key not in _TASKS:
        _TASKS[cache_key] = SlimeVolley(max_steps=max_steps, test=test)
    return _TASKS[cache_key]


def policy_action(ind, obs):
    """One agent's move: a 12-dim observation to 3 binary actions.

    The observation is padded to the genome's node capacity, the network is
    run, and the three output slots are thresholded at zero -- the action
    encoding Slime Volleyball expects.
    """
    out_slots = jnp.arange(N_IN + 1, N_IN + 1 + N_OUT)  # bias sits at slot N_IN
    inp = jnp.zeros(cfg.MAX_NODES, jnp.float32).at[:N_IN].set(obs)
    return (forwardpass(ind, inp)[out_slots] > 0.0).astype(jnp.float32)


@functools.partial(jax.jit, static_argnums=(2, 3))
def eval_pop_jit(pop, key, pop_size, n_episodes):
    """Mean episode reward per genome over ``n_episodes`` games.

    Within an episode every genome faces the *same* initial conditions, so
    score differences reflect the policy rather than the luck of the draw;
    averaging over several episodes then cancels what luck remains.
    """
    task = get_task(test=False)

    def one_episode(ep_key):
        keys = jnp.broadcast_to(ep_key, (pop_size,) + ep_key.shape)
        state = task.reset(keys)

        def one_step(carry, _):
            state, total = carry
            actions = jax.vmap(policy_action)(pop, state.obs)
            state, reward, done = task.step(state, actions)
            return (state, total + reward), None

        (_, total), _ = jax.lax.scan(
            one_step, (state, jnp.zeros(pop_size)), None, length=task.max_steps
        )
        return total

    ep_keys = jax.random.split(key, n_episodes)
    return jax.vmap(one_episode)(ep_keys).mean(axis=0)


def make_fitness_fn(n_episodes=None):
    """Build the ``fitness_fn`` that :func:`neat.evolve` expects."""
    n_episodes = cfg.N_EPISODES if n_episodes is None else n_episodes

    def fitness_fn(pop, pop_size, gen, key):
        rewards = eval_pop_jit(pop, jax.random.fold_in(key, gen), pop_size, n_episodes)
        return np.asarray(rewards), pop  # genomes are unchanged by evaluation

    return fitness_fn


@functools.partial(jax.jit, static_argnums=(2,))
def eval_one(ind, key, n_episodes):
    """Per-episode rewards for a single genome. Returns an array of length ``n_episodes``."""
    task = get_task(test=False)

    def one_episode(ep_key):
        state = task.reset(ep_key[None])  # a batch of one

        def one_step(carry, _):
            state, total = carry
            action = policy_action(ind, state.obs[0])[None]
            state, reward, done = task.step(state, action)
            return (state, total + reward[0]), None

        (_, total), _ = jax.lax.scan(
            one_step, (state, jnp.zeros(())), None, length=task.max_steps
        )
        return total

    return jax.vmap(one_episode)(jax.random.split(key, n_episodes))


def report_champion(ind, n_episodes=20, seed=12345, label="champion"):
    """Score one genome over fresh episodes and print a summary with a standard error."""
    r = np.asarray(eval_one(ind, jax.random.PRNGKey(seed), n_episodes))
    se = r.std(ddof=1) / np.sqrt(n_episodes) if n_episodes > 1 else 0.0
    print(
        f"  {label}:  mean {r.mean():+.3f} +/- {se:.3f} (SE)   "
        f"min {r.min():+.2f}  max {r.max():+.2f}   "
        f"positive-reward episodes: {(r > 0).sum()}/{n_episodes}"
    )
    return r


def match_record(ind, n_matches=20, seed=999):
    """Play full matches in test mode, which end when a side loses five lives."""
    task = get_task(test=True)
    totals = []
    for m in range(n_matches):
        state = task.reset(jax.random.PRNGKey(seed + m)[None])
        total = 0.0
        for _ in range(task.max_steps):
            action = policy_action(ind, state.obs[0])[None]
            state, reward, done = task.step(state, action)
            total += float(reward[0])
            if bool(done[0]):
                break
        totals.append(total)
    r = np.array(totals)
    print(
        f"  {n_matches} matches:  mean net points {r.mean():+.2f}   "
        f"won {(r > 0).sum()}/{n_matches}"
    )
    return r


def make_gif(ind, path="agent.gif", seed=0, frame_skip=4, max_frames=400):
    """Render one match of ``ind`` against the built-in AI as an animated GIF."""
    task = get_task(test=True)
    state = task.reset(jax.random.PRNGKey(seed)[None])
    frames = []
    for t in range(task.max_steps):
        action = policy_action(ind, state.obs[0])[None]
        state, reward, done = task.step(state, action)
        if t % frame_skip == 0:
            single = jax.tree_util.tree_map(lambda x: x[0], state)  # drop batch axis
            frames.append(task.render(single, task_id=0))
        if bool(done[0]) or len(frames) >= max_frames:
            break
    if not frames:
        return None
    frames[0].save(
        path, save_all=True, append_images=frames[1:], duration=33, loop=0
    )
    print(f"  saved {path}  ({len(frames)} frames)")
    return path


def evaluate_all_checkpoints(pattern="best_gen*.npz", n_episodes=20, seed=12345):
    """Re-score every saved checkpoint on the same episodes and report the best.

    Worth doing: the last generation is not reliably the strongest one, because
    a champion selected on three noisy episodes can be a lucky draw.
    """
    results = []
    for path in sorted(glob.glob(pattern)):
        r = report_champion(load_genome(path), n_episodes, seed, label=path)
        results.append((path, float(r.mean())))
    if results:
        best_path, best_mean = max(results, key=lambda t: t[1])
        print(
            f"\n  BEST OVERALL: {best_path}   "
            f"mean reward {best_mean:+.3f} over {n_episodes} episodes"
        )
    return results

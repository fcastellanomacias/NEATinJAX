"""Backprop NEAT on 2-D binary classification."""

import functools

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from neat import config as cfg
from neat.genome import ACT_NAMES, HIDDEN, forwardpass, n_connections
from neat.viz import draw_genome

N_IN_BP, N_OUT_BP = 2, 1  # inputs at slots 0,1 | bias at 2 | output at 3
OUT_SLOT_BP = N_IN_BP + 1
INPUT_LABELS = ("x", "y")


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


def make_dataset(name="xor", n=None, noise=None, seed=0):
    """A 2-D binary classification set, mirroring the TensorFlow Playground tasks.

    ``gauss`` is linearly separable, ``circle`` and ``xor`` are not, and
    ``spiral`` needs real depth.
    """
    n = cfg.N_POINTS if n is None else n
    noise = cfg.DATA_NOISE if noise is None else noise
    rng = np.random.default_rng(seed)

    if name == "xor":
        X = rng.uniform(-1, 1, size=(n, 2))
        y = (X[:, 0] * X[:, 1] > 0).astype(np.float32)
    elif name == "circle":
        X = rng.uniform(-1, 1, size=(n, 2))
        y = ((X**2).sum(1) < 0.35).astype(np.float32)
    elif name == "gauss":
        half = n // 2
        X = np.vstack(
            [
                rng.normal([-0.5, -0.5], 0.25, (half, 2)),
                rng.normal([0.5, 0.5], 0.25, (n - half, 2)),
            ]
        )
        y = np.concatenate([np.zeros(half), np.ones(n - half)]).astype(np.float32)
    elif name == "spiral":
        half = n // 2
        t = np.sqrt(rng.uniform(0, 1, half)) * 3 * np.pi
        arm = np.c_[t * np.cos(t), t * np.sin(t)] / (3 * np.pi)
        X = np.vstack([arm, -arm])
        y = np.concatenate([np.zeros(half), np.ones(half)]).astype(np.float32)
    else:
        raise ValueError(f"unknown dataset {name!r}; choose from {cfg.DATASETS}")

    X = X + rng.normal(0, noise, X.shape)
    return jnp.asarray(X, jnp.float32), jnp.asarray(y, jnp.float32)


# ---------------------------------------------------------------------------
# Loss and gradient training
# ---------------------------------------------------------------------------


def predict_one(w, ind, xy):
    """Probability that the point ``xy`` belongs to class 1."""
    inp = jnp.zeros(cfg.MAX_NODES, jnp.float32).at[:N_IN_BP].set(xy)
    out = forwardpass(ind._replace(conn_w=w), inp)[OUT_SLOT_BP]
    return (out + 1.0) / 2.0  # the output node is tanh: [-1,1] -> [0,1]


def loss_fn(w, ind, X, y):
    """Binary cross-entropy, plus an L2 penalty on the active weights."""
    p = jax.vmap(lambda xy: predict_one(w, ind, xy))(X)
    p = jnp.clip(p, 1e-7, 1.0 - 1e-7)  # guard against log(0)
    bce = -jnp.mean(y * jnp.log(p) + (1 - y) * jnp.log(1 - p))
    l2 = jnp.sum(w**2 * ind.conn_on)
    return bce + cfg.L2_REG * l2


@functools.partial(jax.jit, static_argnums=(4,))
def train_genome(w0, ind, X, y, n_steps, lr):
    """Gradient descent on one genome's weights. Returns ``(weights, final_loss)``."""

    def one(w, _):
        g = jax.grad(loss_fn)(w, ind, X, y)
        return jnp.clip(w - lr * g, -10.0, 10.0), None  # clip: some topologies blow up

    w, _ = jax.lax.scan(one, w0, None, length=n_steps)
    return w, loss_fn(w, ind, X, y)


@functools.partial(jax.jit, static_argnums=(3,))
def train_pop(pop, X, y, n_steps, lr):
    """Train every genome in the population in parallel."""
    return jax.vmap(lambda ind: train_genome(ind.conn_w, ind, X, y, n_steps, lr))(pop)


def accuracy(w, ind, X, y):
    """Fraction of points classified correctly."""
    p = jax.vmap(lambda xy: predict_one(w, ind, xy))(X)
    return float(jnp.mean((p > 0.5) == (y > 0.5)))


def make_fitness_fn(X, y, bp_steps=None, lr=None, verbose=True):
    """Build the ``fitness_fn``."""
    bp_steps = cfg.BP_STEPS if bp_steps is None else bp_steps
    lr = cfg.LEARN_RATE if lr is None else lr

    def fitness_fn(pop, pop_size, gen, key):
        trained_w, losses = train_pop(pop, X, y, bp_steps, lr)
        pop = pop._replace(conn_w=trained_w)  # Lamarckian write-back

        losses = np.nan_to_num(np.asarray(losses), nan=1e3, posinf=1e3, neginf=1e3)
        n_bad = int((losses >= 1e3).sum())
        if n_bad and verbose:
            print(
                f"     [{n_bad}/{pop_size} genomes produced a non-finite loss]",
                flush=True,
            )

        fitness = -losses - cfg.BP_PENALTY * np.sqrt(n_connections(pop))
        return fitness, pop

    return fitness_fn


def activation_counts(ind):
    """How many hidden nodes use each activation function."""
    node_type = np.asarray(ind.node_type)
    node_act = np.asarray(ind.node_act)
    return np.bincount(node_act[node_type == HIDDEN], minlength=len(ACT_NAMES))


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def plot_decision_boundary(ind, X, y, ax=None, title="", save_path=None, res=120):
    lo, hi = -1.7, 1.7
    gx, gy = np.meshgrid(np.linspace(lo, hi, res), np.linspace(lo, hi, res))
    pts = jnp.asarray(np.c_[gx.ravel(), gy.ravel()], jnp.float32)
    p = np.asarray(jax.vmap(lambda xy: predict_one(ind.conn_w, ind, xy))(pts)).reshape(
        res, res
    )

    own_figure = ax is None
    if own_figure:
        _, ax = plt.subplots(figsize=(4.2, 4.2))

    ax.contourf(gx, gy, p, levels=np.linspace(0, 1, 21), cmap="RdBu", alpha=0.8)
    ax.contour(gx, gy, p, levels=[0.5], colors="k", linewidths=1.2)
    Xn, yn = np.asarray(X), np.asarray(y)
    ax.scatter(
        Xn[yn == 0, 0], Xn[yn == 0, 1], s=9, c="#b2182b", edgecolors="k", linewidths=0.3
    )
    ax.scatter(
        Xn[yn == 1, 0], Xn[yn == 1, 1], s=9, c="#2166ac", edgecolors="k", linewidths=0.3
    )
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10)

    if own_figure:
        ax.figure.tight_layout()
        if save_path:
            ax.figure.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(ax.figure)
    return ax


def plot_summary_grid(results, save_path=None):
    names = list(results)
    fig, axes = plt.subplots(
        2, len(names), figsize=(3.75 * len(names), 7.2), squeeze=False
    )
    for j, name in enumerate(names):
        r = results[name]
        best = r["best"]
        plot_decision_boundary(
            best, r["X"], r["y"], ax=axes[0, j], title=f"{name}   acc {r['acc']:.0%}"
        )
        draw_genome(
            best,
            ax=axes[1, j],
            title="",
            label_activations=True,
            input_labels=INPUT_LABELS,
        )
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig

"""Genome representation and the forward pass.

A genome is a fixed-size bundle of arrays, which is what lets a whole
population be stacked into a single pytree and evaluated under ``jax.vmap``.
Unused capacity is carried around as inactive slots rather than being
allocated on demand, so every genome in the population has identical shapes no
matter how much structure it has grown.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from . import config as cfg

# Node type codes stored in ``node_type``.
UNUSED, INPUT, OUTPUT, HIDDEN, BIAS = 0, 1, 2, 3, 4


class Ind(NamedTuple):
    """One individual, as seven immutable arrays.

    Node genes, each of length ``cfg.MAX_NODES``:
        node_type  -- 0 unused, 1 input, 2 output, 3 hidden, 4 bias
        node_act   -- index into :data:`ACTIVATIONS`

    Connection genes, each of length ``cfg.MAX_CONNS``:
        conn_in    -- source node index
        conn_out   -- destination node index
        conn_w     -- weight
        conn_on    -- enabled?
        conn_innov -- innovation number (-1 for an empty slot)

    A single individual holds 1-D arrays; a stacked population holds 2-D
    arrays with the population along axis 0.
    """

    node_type: jnp.ndarray
    node_act: jnp.ndarray
    conn_in: jnp.ndarray
    conn_out: jnp.ndarray
    conn_w: jnp.ndarray
    conn_on: jnp.ndarray
    conn_innov: jnp.ndarray


# ---------------------------------------------------------------------------
# Activation functions
# ---------------------------------------------------------------------------


def _linear(x):
    return x


def _step(x):
    return 1.0 * (x > 0.0)  # not differentiable; excluded from backprop NEAT


def _sin(x):
    return jnp.sin(jnp.pi * x)


def _gauss(x):
    return jnp.exp(-(x * x) / 2.0)


def _tanh(x):
    return jnp.tanh(x)


def _sigmoid(x):
    return (jnp.tanh(x / 2.0) + 1.0) / 2.0


def _inverse(x):
    return -x


def _abs(x):
    return jnp.abs(x)


def _relu(x):
    return jnp.maximum(0.0, x)


def _cos(x):
    return jnp.cos(jnp.pi * x)


def _square(x):
    return x**2


ACTIVATIONS = (
    _linear,  # 0
    _step,  # 1
    _sin,  # 2
    _gauss,  # 3
    _tanh,  # 4
    _sigmoid,  # 5
    _inverse,  # 6
    _abs,  # 7
    _relu,  # 8
    _cos,  # 9
    _square,  # 10
)

ACT_NAMES = (
    "linear",
    "step",
    "sin",
    "gauss",
    "tanh",
    "sigmoid",
    "inverse",
    "abs",
    "relu",
    "cos",
    "square",
)

TANH = 4  # output nodes use tanh, so raw outputs live in [-1, 1]

# Every activation except the step function, which has zero gradient
# everywhere it is defined and so is useless to backprop NEAT.
DIFFERENTIABLE_ACTS = np.array([i for i in range(len(ACTIVATIONS)) if i != 1])


def apply_act(act_id, x):
    """Apply activation ``act_id`` to the scalar ``x``."""
    return jax.lax.switch(act_id, ACTIVATIONS, x)


def apply_acts(act_ids, xs):
    """Apply each node's own activation to its own value."""
    return jax.vmap(apply_act)(act_ids, xs)


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------


def forwardpass(ind, inputs):
    """Evaluate one genome. Returns the settled activation of every node slot.

    ``inputs`` is a vector of length ``cfg.MAX_NODES``; only its first few
    entries -- as many as the genome has input nodes -- are read.

    The network is not layered explicitly. Instead the whole activation vector
    is updated ``MAX_NODES`` times in a scan: because mutation only ever adds
    connections that point strictly forward, the graph is acyclic, so a network
    of n nodes has settled after at most n propagation rounds.
    """
    n = cfg.MAX_NODES
    is_input = ind.node_type == INPUT
    is_bias = ind.node_type == BIAS
    is_used = ind.node_type != UNUSED

    # Write the supplied values into the input slots. `input_slots` lists the
    # input node indices first and pads with n; the pad entries are dropped.
    act = jnp.zeros(n, jnp.float32)
    input_slots = jnp.nonzero(is_input, size=n, fill_value=n)[0]
    valid = input_slots < n
    vals = jnp.where(valid, inputs[jnp.arange(n)], 0.0)
    act = act.at[input_slots].set(vals, mode="drop")
    act = jnp.where(is_bias, 1.0, act)

    # Disabled connections contribute nothing.
    w_eff = jnp.where(ind.conn_on, ind.conn_w, 0.0)

    def one_step(act, _):
        incoming = w_eff * act[ind.conn_in]  # weighted signal along each edge
        agg = jnp.zeros(n, jnp.float32).at[ind.conn_out].add(incoming)
        new = apply_acts(ind.node_act, agg)
        new = jnp.where(is_input | is_bias, act, new)  # sources hold their value
        new = jnp.where(is_used, new, 0.0)
        return new, None

    act, _ = jax.lax.scan(one_step, act, None, length=n)
    return act


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def init_ind(key, n_in, n_out):
    """One minimal genome: bias plus inputs fully connected to outputs.

    Slot layout, fixed for the whole run so that node indices mean the same
    thing in every genome::

        0 .. n_in-1                  input nodes
        n_in                         bias node
        n_in+1 .. n_in+n_out         output nodes
        the rest                     unused, available for hidden nodes
    """
    if n_in + 1 + n_out > cfg.MAX_NODES:
        raise ValueError(
            f"MAX_NODES={cfg.MAX_NODES} too small: need {n_in + 1 + n_out} "
            f"(inputs + bias + outputs). Raise MAX_NODES in neat/config.py."
        )
    n_conns = (n_in + 1) * n_out
    if n_conns > cfg.MAX_CONNS:
        raise ValueError(
            f"MAX_CONNS={cfg.MAX_CONNS} too small: need {n_conns} initial "
            f"connections for {n_in} inputs + bias -> {n_out} outputs. "
            f"Raise MAX_CONNS in neat/config.py."
        )

    node_type = jnp.full(cfg.MAX_NODES, UNUSED, jnp.int8)
    node_type = node_type.at[:n_in].set(INPUT)
    node_type = node_type.at[n_in].set(BIAS)
    node_type = node_type.at[n_in + 1 : n_in + 1 + n_out].set(OUTPUT)

    # Outputs squash with tanh; everything else starts linear.
    node_act = jnp.full(cfg.MAX_NODES, 0, jnp.int8)
    node_act = node_act.at[n_in + 1 : n_in + 1 + n_out].set(TANH)

    conn_in, conn_out = [], []
    for src in range(n_in + 1):  # every input, and the bias
        for dst in range(n_in + 1, n_in + 1 + n_out):
            conn_in.append(src)
            conn_out.append(dst)
    pad = [0] * (cfg.MAX_CONNS - n_conns)
    conn_in = jnp.array(conn_in + pad, jnp.int32)
    conn_out = jnp.array(conn_out + pad, jnp.int32)

    idx = jnp.arange(cfg.MAX_CONNS)
    is_real = idx < n_conns
    w = jax.random.uniform(key, (cfg.MAX_CONNS,), minval=-cfg.W_CAP, maxval=cfg.W_CAP)
    conn_w = jnp.where(is_real, w, 0.0)
    conn_innov = jnp.where(is_real, idx, -1)
    return Ind(node_type, node_act, conn_in, conn_out, conn_w, is_real, conn_innov)


def init_pop(key, pop_size, n_in, n_out):
    """A population of ``pop_size`` minimal genomes, stacked into one pytree."""
    inds = [init_ind(k, n_in, n_out) for k in jax.random.split(key, pop_size)]
    return stack_inds(inds)


def get_ind(pop, i):
    """Extract individual ``i`` from a stacked population."""
    return jax.tree_util.tree_map(lambda leaf: leaf[i], pop)


def stack_inds(inds):
    """Stack a list of individuals into one population pytree."""
    return Ind(*[jnp.stack([getattr(g, f) for g in inds]) for f in Ind._fields])


# ---------------------------------------------------------------------------
# Small host-side helpers
# ---------------------------------------------------------------------------


def free_conn_slot(conn_on):
    """Index of the first inactive connection slot, or None if full."""
    empty = np.where(~np.asarray(conn_on, dtype=bool))[0]
    return int(empty[0]) if len(empty) else None


def free_node_slot(node_type):
    """Index of the first unused node slot, or None if full."""
    empty = np.where(np.asarray(node_type) == UNUSED)[0]
    return int(empty[0]) if len(empty) else None


def n_connections(pop):
    """Active connection count. Scalar for an individual, array for a population."""
    return np.asarray(pop.conn_on, dtype=bool).sum(axis=-1)


def n_hidden(pop):
    """Hidden node count. Scalar for an individual, array for a population."""
    return (np.asarray(pop.node_type) == HIDDEN).sum(axis=-1)


def node_layers(ind):
    """Depth of each node: inputs and bias at 0, others at 1 + max incoming depth.

    Outputs are then pushed to the deepest layer so they always draw on the
    right-hand side. Unused slots get -1. Used both for laying out topology
    figures and for deciding which new connections point forward.
    """
    node_type = np.asarray(ind.node_type)
    conn_on = np.asarray(ind.conn_on, dtype=bool)
    src = np.asarray(ind.conn_in)[conn_on]
    dst = np.asarray(ind.conn_out)[conn_on]

    layer = np.zeros(cfg.MAX_NODES, dtype=int)
    for _ in range(cfg.MAX_NODES):  # the graph is acyclic, so this converges
        nxt = layer.copy()
        np.maximum.at(nxt, dst, layer[src] + 1)
        if np.array_equal(nxt, layer):
            break
        layer = nxt

    is_used = node_type != UNUSED
    is_out = node_type == OUTPUT
    if is_out.any():
        non_out = is_used & ~is_out
        layer[is_out] = layer[non_out].max() + 1 if non_out.any() else 0
    layer[~is_used] = -1
    return layer


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_genome(ind, path):
    """Write one genome's seven arrays to a .npz file."""
    np.savez(path, **{f: np.asarray(getattr(ind, f)) for f in Ind._fields})


def load_genome(path):
    """Read back a genome written by :func:`save_genome`."""
    data = np.load(path)
    return Ind(**{f: jnp.asarray(data[f]) for f in Ind._fields})

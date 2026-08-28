"""Mutation and crossover."""

import jax.numpy as jnp
import numpy as np

from . import config as cfg
from .genome import (
    ACTIVATIONS,
    HIDDEN,
    free_conn_slot,
    free_node_slot,
    node_layers,
)


def get_innov(record, key, counter):
    """Look up, or mint, the innovation number for a structural change.
    Returns ``(innovation_number, new_counter)``.
    """
    if key in record:
        return record[key], counter
    record[key] = counter
    return counter, counter + 1


def mut_weights(ind, rng):
    """Perturb connection weights with Gaussian noise, then clip."""
    w = np.asarray(ind.conn_w, dtype=np.float64)
    mutate = rng.random(cfg.MAX_CONNS) < cfg.PROB_MUT_CONN
    w = w + cfg.MUT_SIGMA * rng.normal(size=cfg.MAX_CONNS) * mutate
    w = np.clip(w, -cfg.W_CAP, cfg.W_CAP)
    return ind._replace(conn_w=jnp.asarray(w, jnp.float32))


def mut_add_conn(ind, rng, innov_counter, innov_record):
    """Add one connection between two existing nodes.
    Only pairs that point from a shallower to a deeper node are eligible, which
    keeps the network acyclic.

    Returns ``(new_ind, new_counter)``, unchanged if there is no room or no
    legal pair left.
    """
    slot = free_conn_slot(ind.conn_on)
    if slot is None:
        return ind, innov_counter

    conn_in = np.asarray(ind.conn_in).copy()
    conn_out = np.asarray(ind.conn_out).copy()
    conn_w = np.asarray(ind.conn_w).copy()
    conn_on = np.asarray(ind.conn_on, dtype=bool).copy()
    conn_innov = np.asarray(ind.conn_innov).copy()
    layer = node_layers(ind)

    # Legal candidates: both nodes in use, strictly forward, not already wired.
    used = layer >= 0
    legal = used[:, None] & used[None, :] & (layer[None, :] > layer[:, None])
    for k in np.where(conn_innov >= 0)[0]:
        legal[int(conn_in[k]), int(conn_out[k])] = False
    candidates = np.argwhere(legal)
    if len(candidates) == 0:
        return ind, innov_counter

    src, dst = (int(v) for v in candidates[rng.integers(len(candidates))])
    innov, innov_counter = get_innov(innov_record, (src, dst), innov_counter)

    conn_in[slot] = src
    conn_out[slot] = dst
    conn_w[slot] = rng.uniform(-cfg.W_CAP, cfg.W_CAP)
    conn_on[slot] = True
    conn_innov[slot] = innov

    new_ind = ind._replace(
        conn_in=jnp.asarray(conn_in, jnp.int32),
        conn_out=jnp.asarray(conn_out, jnp.int32),
        conn_w=jnp.asarray(conn_w, jnp.float32),
        conn_on=jnp.asarray(conn_on, bool),
        conn_innov=jnp.asarray(conn_innov, jnp.int32),
    )
    return new_ind, innov_counter


def mut_add_node(ind, rng, innov_counter, innov_record, act_choices=None):
    """Split an active connection by inserting a hidden node.

    ``src --w--> dst`` becomes ``src --1--> new --w--> dst`` with the original
    connection disabled.

    Returns ``(new_ind, new_counter)``, unchanged if there is no free node slot
    or fewer than two free connection slots.
    """
    node_type = np.asarray(ind.node_type).copy()
    node_act = np.asarray(ind.node_act).copy()
    conn_in = np.asarray(ind.conn_in).copy()
    conn_out = np.asarray(ind.conn_out).copy()
    conn_w = np.asarray(ind.conn_w).copy()
    conn_on = np.asarray(ind.conn_on, dtype=bool).copy()
    conn_innov = np.asarray(ind.conn_innov).copy()

    active = np.where(conn_on)[0]
    if len(active) == 0:
        return ind, innov_counter
    split = int(active[rng.integers(len(active))])

    new_node = free_node_slot(node_type)
    free = np.where(~conn_on)[0]
    if new_node is None or len(free) < 2:
        return ind, innov_counter
    left, right = int(free[0]), int(free[1])

    if act_choices is None:
        node_act[new_node] = rng.integers(len(ACTIVATIONS))
    else:
        node_act[new_node] = int(rng.choice(act_choices))
    node_type[new_node] = HIDDEN

    # Read the original endpoints before overwriting anything.
    src, dst, w = int(conn_in[split]), int(conn_out[split]), conn_w[split]
    split_innov = int(conn_innov[split])
    innov_l, innov_counter = get_innov(
        innov_record, ("left", split_innov), innov_counter
    )
    innov_r, innov_counter = get_innov(
        innov_record, ("right", split_innov), innov_counter
    )

    conn_on[split] = False
    for slot, (a, b, weight, innov) in zip(
        (left, right),
        ((src, new_node, 1.0, innov_l), (new_node, dst, w, innov_r)),
    ):
        conn_in[slot] = a
        conn_out[slot] = b
        conn_w[slot] = weight
        conn_innov[slot] = innov
        conn_on[slot] = True

    new_ind = ind._replace(
        node_type=jnp.asarray(node_type, jnp.int8),
        node_act=jnp.asarray(node_act, jnp.int8),
        conn_in=jnp.asarray(conn_in, jnp.int32),
        conn_out=jnp.asarray(conn_out, jnp.int32),
        conn_w=jnp.asarray(conn_w, jnp.float32),
        conn_on=jnp.asarray(conn_on, bool),
        conn_innov=jnp.asarray(conn_innov, jnp.int32),
    )
    return new_ind, innov_counter


def crossover(parent_a, parent_b, rng):
    """Combine two parents, ``parent_a`` being the fitter one.

    The child inherits A's topology outright and, for genes both parents share
    (identified by innovation number) takes B's weight with probability
    ``PROB_INHERIT_B``.
    """
    a_innov = np.asarray(parent_a.conn_innov)
    a_on = np.asarray(parent_a.conn_on, dtype=bool)
    b_innov = np.asarray(parent_b.conn_innov)
    b_on = np.asarray(parent_b.conn_on, dtype=bool)
    b_w = np.asarray(parent_b.conn_w)

    child_w = np.asarray(parent_a.conn_w).copy()
    b_slot_of_innov = {int(b_innov[k]): int(k) for k in np.where(b_on)[0]}

    for k in np.where(a_on)[0]:
        b_slot = b_slot_of_innov.get(int(a_innov[k]))
        if b_slot is not None and rng.random() < cfg.PROB_INHERIT_B:
            child_w[k] = b_w[b_slot]

    return parent_a._replace(conn_w=jnp.asarray(child_w, jnp.float32))

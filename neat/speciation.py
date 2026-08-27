"""Speciation and fitness sharing.

A structural mutation almost always hurts before it helps: a fresh hidden node
arrives with untuned weights and loses to the streamlined genomes around it.
Speciation buys it time by making genomes compete mainly against relatives, and
fitness sharing stops any one species from taking over the population.
"""

import numpy as np

from . import config as cfg
from .genome import get_ind


def compat_distance(ind_a, ind_b):
    """Compatibility distance between two genomes.

    Two terms, as in the original NEAT: the fraction of genes present in only
    one of the two (structural difference), and the mean absolute weight
    difference over the genes they share (parametric difference).
    """
    a_innov = np.asarray(ind_a.conn_innov)
    a_on = np.asarray(ind_a.conn_on, dtype=bool)
    b_innov = np.asarray(ind_b.conn_innov)
    b_on = np.asarray(ind_b.conn_on, dtype=bool)
    a_w = np.asarray(ind_a.conn_w)
    b_w = np.asarray(ind_b.conn_w)

    a_map = {int(a_innov[k]): int(k) for k in np.where(a_on)[0]}
    b_map = {int(b_innov[k]): int(k) for k in np.where(b_on)[0]}

    matching = a_map.keys() & b_map.keys()
    n_unmatched = len(a_map.keys() ^ b_map.keys())

    if matching:
        mean_w_diff = sum(
            abs(a_w[a_map[i]] - b_w[b_map[i]]) for i in matching
        ) / len(matching)
    else:
        mean_w_diff = 0.0

    n = max(len(a_map), len(b_map), 1)  # normalise by the larger genome
    return cfg.C_UNMATCHED * (n_unmatched / n) + cfg.C_WEIGHT * mean_w_diff


def speciate(pop, pop_size, reps):
    """Assign every individual to a species.

    Each genome joins the first species whose representative it is closer to
    than ``COMPAT_THRESHOLD``, and founds a new species otherwise. Carrying the
    representatives over from the previous generation keeps species identities
    stable across time.

    Args:
        pop: stacked population.
        pop_size: number of individuals.
        reps: representatives from the previous generation (empty at first).

    Returns:
        ``(species_of, new_reps)`` -- an int array of length ``pop_size``, and
        one representative per surviving species.
    """
    species_of = np.full(pop_size, -1, dtype=int)
    reps = list(reps)

    for i in range(pop_size):
        ind_i = get_ind(pop, i)
        for j, rep in enumerate(reps):
            if compat_distance(ind_i, rep) < cfg.COMPAT_THRESHOLD:
                species_of[i] = j
                break
        else:
            reps.append(ind_i)
            species_of[i] = len(reps) - 1

    # Drop species that ended up empty and renumber the rest from zero.
    used = sorted(set(int(s) for s in species_of))
    remap = {old: new for new, old in enumerate(used)}
    species_of = np.array([remap[int(s)] for s in species_of])
    new_reps = [
        get_ind(pop, int(np.where(species_of == remap[old])[0][0])) for old in used
    ]
    return species_of, new_reps


def shared_fitness(fitness, species_of):
    """Divide each individual's fitness by the size of its species."""
    return fitness / np.bincount(species_of)[species_of]

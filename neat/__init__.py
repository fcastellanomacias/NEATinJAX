"""A JAX implementation of NEAT (NeuroEvolution of Augmenting Topologies)."""

from . import config
from .config import use
from .evolution import Result, evolve
from .genome import (
    ACT_NAMES,
    ACTIVATIONS,
    BIAS,
    DIFFERENTIABLE_ACTS,
    HIDDEN,
    INPUT,
    OUTPUT,
    UNUSED,
    Ind,
    forwardpass,
    get_ind,
    init_ind,
    init_pop,
    load_genome,
    n_connections,
    n_hidden,
    node_layers,
    save_genome,
    stack_inds,
)
from .mutation import crossover, mut_add_conn, mut_add_node, mut_weights
from .speciation import compat_distance, shared_fitness, speciate
from .viz import draw_genome, plot_fitness, plot_fitness_and_complexity

__all__ = [
    "config",
    "use",
    "evolve",
    "Result",
    "Ind",
    "forwardpass",
    "init_ind",
    "init_pop",
    "get_ind",
    "stack_inds",
    "save_genome",
    "load_genome",
    "node_layers",
    "n_hidden",
    "n_connections",
    "ACTIVATIONS",
    "ACT_NAMES",
    "DIFFERENTIABLE_ACTS",
    "UNUSED",
    "INPUT",
    "OUTPUT",
    "HIDDEN",
    "BIAS",
    "mut_weights",
    "mut_add_conn",
    "mut_add_node",
    "crossover",
    "compat_distance",
    "speciate",
    "shared_fitness",
    "draw_genome",
    "plot_fitness",
    "plot_fitness_and_complexity",
]

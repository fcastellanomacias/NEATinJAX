"""The evolutionary loop.

Both experiments in this repository use the same loop; they differ only in how
a genome is scored. That difference is passed in as ``fitness_fn``, so the
selection machinery (speciation, fitness sharing, offspring allocation,
elitism, mutation) is written once.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, List, Tuple

import numpy as np

from . import config as cfg
from .genome import get_ind, init_pop, n_connections, n_hidden, stack_inds
from .mutation import crossover, mut_add_conn, mut_add_node, mut_weights
from .speciation import shared_fitness, speciate


@dataclass
class Result:
    """What a run leaves behind."""

    pop: Any  # final stacked population, best genome first
    history: List[float] = field(default_factory=list)  # best raw fitness per gen
    hidden_hist: List[float] = field(default_factory=list)  # mean hidden nodes per gen
    snapshots: List[Tuple[int, Any]] = field(default_factory=list)  # (gen, champion)

    @property
    def champion(self):
        return get_ind(self.pop, 0)


def evolve(
    fitness_fn: Callable,
    key,
    rng,
    n_in: int,
    n_out: int,
    pop_size: int = None,
    n_gens: int = None,
    act_choices=None,
    structural: bool = True,
    use_crossover: bool = True,
    use_speciation: bool = True,
    snapshot_every: int = None,
    callback: Callable = None,
    verbose: bool = True,
) -> Result:
    """Run NEAT.

    Args:
        fitness_fn: ``f(pop, pop_size, gen, key) -> (raw_fitness, pop)``.
            Returns one score per genome, plus the population, which it may
            return modified, as backprop NEAT does when it writes trained
            weights back into the genomes.
        key: JAX PRNG key, used for genome initialisation and folded per
            generation for evaluation.
        rng: NumPy ``Generator``, used for every mutation and selection draw.
        n_in, n_out: input and output node counts for the task.
        pop_size, n_gens, snapshot_every: default to the active config values.
        act_choices: activation ids a new hidden node may take; ``None`` allows
            all of them.
        structural: allow add-connection and add-node mutations.
        use_crossover: allow two-parent offspring.
        use_speciation: if false, the whole population is one species.
        callback: ``f(gen, champion, raw_fitness)``, called once per generation
            after the new population is assembled. Used for checkpoints, GIFs
            and anything else task-specific.
        verbose: print a one-line summary per generation.
    """
    pop_size = cfg.POP_SIZE if pop_size is None else pop_size
    n_gens = cfg.N_GENS if n_gens is None else n_gens
    snapshot_every = cfg.SNAPSHOT_EVERY if snapshot_every is None else snapshot_every

    if pop_size <= cfg.N_ELITES:
        raise ValueError(f"pop_size ({pop_size}) must exceed N_ELITES ({cfg.N_ELITES})")

    pop = init_pop(key, pop_size, n_in, n_out)
    innov_counter = (n_in + 1) * n_out  # the initial genes used 0 .. counter-1
    innov_record = {}
    reps = []  # species representatives, carried across generations
    result = Result(pop=pop)

    for gen in range(n_gens):
        raw_fitness, pop = fitness_fn(pop, pop_size, gen, key)
        raw_fitness = np.asarray(raw_fitness, dtype=np.float64)
        result.history.append(float(raw_fitness.max()))

        # Pressure against bloat, optionally annealed to zero
        n_conns = n_connections(pop)
        penalty = cfg.COMPLEXITY_PENALTY
        if cfg.ANNEAL_PENALTY and n_gens > 0:
            penalty *= 1.0 - gen / n_gens
        fitness = raw_fitness - penalty * n_conns

        # Fitness sharing needs positive values, so shift before dividing.
        if use_speciation:
            species_of, reps = speciate(pop, pop_size, reps)
            sel_fitness = shared_fitness(fitness - fitness.min() + 1e-3, species_of)
        else:
            species_of = np.zeros(pop_size, dtype=int)
            sel_fitness = fitness - fitness.min() + 1e-3

        # Each species breeds in proportion to its total shared fitness.
        n_sp = species_of.max() + 1
        sp_score = np.maximum(
            np.array([sel_fitness[species_of == s].sum() for s in range(n_sp)]), 1e-9
        )
        alloc = np.floor(sp_score / sp_score.sum() * (pop_size - cfg.N_ELITES))
        alloc = alloc.astype(int)
        while alloc.sum() < pop_size - cfg.N_ELITES:  # hand out the remainder
            alloc[np.argmax(sp_score)] += 1

        # Elites and parent ranking use raw fitness, never the shared value
        elite_idx = np.argsort(raw_fitness)[-cfg.N_ELITES :][::-1]
        next_inds = [get_ind(pop, int(i)) for i in elite_idx]

        for s in range(n_sp):
            members = np.where(species_of == s)[0]
            members = members[np.argsort(raw_fitness[members])[::-1]]
            n_keep = max(1, int(round((1.0 - cfg.CULL_RATIO) * len(members))))
            pool = members[:n_keep]

            for _ in range(alloc[s]):
                if (
                    use_crossover
                    and rng.random() < cfg.PROB_CROSSOVER
                    and len(pool) > 1
                ):
                    p1, p2 = rng.choice(pool, size=2, replace=True)
                    a, b = (p1, p2) if raw_fitness[p1] >= raw_fitness[p2] else (p2, p1)
                    child = crossover(get_ind(pop, int(a)), get_ind(pop, int(b)), rng)
                else:
                    child = get_ind(pop, int(rng.choice(pool)))

                child = mut_weights(child, rng)
                if structural:
                    if rng.random() < cfg.PROB_ADD_CONN:
                        child, innov_counter = mut_add_conn(
                            child, rng, innov_counter, innov_record
                        )
                    if rng.random() < cfg.PROB_ADD_NODE:
                        child, innov_counter = mut_add_node(
                            child, rng, innov_counter, innov_record, act_choices
                        )
                next_inds.append(child)

        assert len(next_inds) == pop_size, f"got {len(next_inds)}, want {pop_size}"
        pop = stack_inds(next_inds)
        result.pop = pop

        hidden = n_hidden(pop)
        result.hidden_hist.append(float(hidden.mean()))
        champion = get_ind(pop, 0)  # elites come first, best of them at index 0

        if gen % snapshot_every == 0 or gen == n_gens - 1:
            result.snapshots.append((gen, champion))
        if callback is not None:
            callback(gen, champion, raw_fitness)

        if verbose:
            print(
                f"  gen {gen:4d}  best {raw_fitness.max():+.4f}  "
                f"mean {raw_fitness.mean():+.4f}  "
                f"species {n_sp:3d}  hidden {int(hidden[0]):2d}  "
                f"avg_hidden {hidden.mean():4.1f}  avg_conns {n_conns.mean():5.1f}",
                flush=True,
            )

    return result

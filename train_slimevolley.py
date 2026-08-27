#!/usr/bin/env python3
"""Evolve a NEAT agent to play Slime Volleyball against the built-in AI.

    python train_slimevolley.py                     # full run, config defaults
    python train_slimevolley.py --gens 200 --pop 50  # something quicker
    python train_slimevolley.py --set MUT_SIGMA=0.5 --set PROB_ADD_NODE=0.2

Outputs, all written to --out:
    fitness.png              best episode reward against generation
    topology_gen####.png     the champion's network, every SNAPSHOT_EVERY gens
    best_gen####.gif         the champion playing a match
    best_gen####.npz         the champion's genome, reloadable
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")  # write files without needing a display

import jax
import numpy as np

from neat import config as cfg


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gens", type=int, default=None, help="generations (default: config N_GENS)")
    p.add_argument("--pop", type=int, default=None, help="population size (default: config POP_SIZE)")
    p.add_argument("--episodes", type=int, default=None, help="episodes averaged per genome")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="outputs/slimevolley", help="directory for figures and checkpoints")
    p.add_argument("--eval-episodes", type=int, default=20, help="episodes used to score the final champion")
    p.add_argument("--no-speciation", action="store_true")
    p.add_argument("--no-crossover", action="store_true")
    p.add_argument("--no-structural", action="store_true", help="fix the topology; evolve weights only")
    p.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="override any hyperparameter from neat/config.py, repeatable",
    )
    return p.parse_args()


def parse_overrides(pairs):
    """Turn ``--set NAME=VALUE`` strings into typed keyword overrides."""
    out = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--set expects NAME=VALUE, got {pair!r}")
        name, _, value = pair.partition("=")
        current = getattr(cfg, name.strip(), None)
        if current is None:
            raise SystemExit(f"unknown hyperparameter {name.strip()!r}")
        caster = type(current)
        out[name.strip()] = value.strip() == "True" if caster is bool else caster(value)
    return out


def main():
    args = parse_args()
    overrides = parse_overrides(args.set)
    if args.gens is not None:
        overrides["N_GENS"] = args.gens
    if args.pop is not None:
        overrides["POP_SIZE"] = args.pop
    if args.episodes is not None:
        overrides["N_EPISODES"] = args.episodes

    # Choose hyperparameters before anything builds a genome or an environment.
    cfg.use("slimevolley", **overrides)

    # Imported after the config is set, so module-level defaults see the right values.
    from neat.evolution import evolve
    from neat.genome import save_genome
    from neat.viz import draw_genome, plot_fitness
    from tasks import slimevolley as sv

    os.makedirs(args.out, exist_ok=True)
    path = lambda name: os.path.join(args.out, name)

    print("Slime Volleyball -- NEAT")
    print(cfg.summary())
    print()

    def checkpoint(gen, champion, raw_fitness):
        """Save a GIF and the genome itself at regular intervals."""
        if gen % cfg.GIF_EVERY == 0 or gen == cfg.N_GENS - 1:
            sv.make_gif(champion, path=path(f"best_gen{gen:04d}.gif"), seed=args.seed)
            save_genome(champion, path(f"best_gen{gen:04d}.npz"))

    result = evolve(
        fitness_fn=sv.make_fitness_fn(),
        key=jax.random.PRNGKey(args.seed),
        rng=np.random.default_rng(args.seed),
        n_in=sv.N_IN,
        n_out=sv.N_OUT,
        structural=not args.no_structural,
        use_crossover=not args.no_crossover,
        use_speciation=not args.no_speciation,
        callback=checkpoint,
    )

    plot_fitness(
        result.history,
        title="NEAT on Slime Volleyball",
        ylabel="best episode reward",
        save_path=path("fitness.png"),
    )
    for gen, genome in result.snapshots:
        draw_genome(
            genome,
            title=f"Generation {gen}",
            save_path=path(f"topology_gen{gen:04d}.png"),
        )

    print("\nFinal champion, on episodes it was never selected on:")
    sv.report_champion(result.champion, n_episodes=args.eval_episodes, label="champion")

    # The last generation is not reliably the best one: a champion picked on a
    # handful of noisy episodes can simply have been lucky. Re-score them all.
    print("\nRe-scoring every checkpoint on identical episodes:")
    sv.evaluate_all_checkpoints(
        pattern=path("best_gen*.npz"), n_episodes=args.eval_episodes
    )

    print(f"\nWrote figures, GIFs and genomes to {args.out}/")


if __name__ == "__main__":
    main()

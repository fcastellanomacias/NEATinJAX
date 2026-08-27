#!/usr/bin/env python3
"""Evolve architectures with backprop NEAT on 2-D classification tasks.

    python train_classification.py                          # all four datasets
    python train_classification.py --datasets xor spiral
    python train_classification.py --gens 80 --pop 50
    python train_classification.py --set BP_STEPS=300 --set LEARN_RATE=0.2

Outputs, all written to --out:
    summary.png                  every dataset: boundary above, topology below
    boundary_<dataset>.png       decision boundary on its own
    topology_<dataset>.png       the evolved network, hidden nodes labelled
                                 with their activation function
    fitness_<dataset>.png        fitness and mean hidden-node count per generation
    best_<dataset>.npz           the winning genome, reloadable
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
    p.add_argument("--points", type=int, default=None, help="samples per dataset")
    p.add_argument("--datasets", nargs="+", default=None, choices=list(cfg.DATASETS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="outputs/classification")
    p.add_argument("--quiet", action="store_true", help="suppress the per-generation log")
    p.add_argument("--no-speciation", action="store_true")
    p.add_argument("--no-crossover", action="store_true")
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


def report_activation_usage(totals, act_names):
    """Which activation functions evolution actually kept, pooled over datasets."""
    n = int(totals.sum())
    if not n:
        return
    print(
        f"\nActivations chosen across all datasets ({n} hidden nodes; "
        f"uniform choice would be {100 / len(act_names):.1f}% each)"
    )
    for a in np.argsort(-totals):
        if totals[a]:
            bar = "#" * int(30 * totals[a] / totals.max())
            print(f"   {act_names[a]:>8s} {totals[a]:3d}   {100 * totals[a] / n:5.1f}%   {bar}")
    never = [act_names[a] for a in range(len(act_names)) if totals[a] == 0]
    if never:
        print(f"   never used: {', '.join(never)}")


def main():
    args = parse_args()
    overrides = parse_overrides(args.set)
    if args.gens is not None:
        overrides["N_GENS"] = args.gens
    if args.pop is not None:
        overrides["POP_SIZE"] = args.pop
    if args.points is not None:
        overrides["N_POINTS"] = args.points

    # Choose hyperparameters before anything builds a genome.
    cfg.use("classification", **overrides)

    from neat.evolution import evolve
    from neat.genome import ACT_NAMES, DIFFERENTIABLE_ACTS, n_hidden, save_genome
    from neat.viz import draw_genome, plot_fitness_and_complexity
    from tasks import classification as clf

    datasets = args.datasets or list(cfg.DATASETS)
    os.makedirs(args.out, exist_ok=True)
    path = lambda name: os.path.join(args.out, name)

    print("Backprop NEAT -- 2-D classification")
    print(cfg.summary())
    print()

    results = {}
    totals = np.zeros(len(ACT_NAMES), dtype=int)

    for name in datasets:
        print(f"=== {name} " + "=" * (56 - len(name)))
        X, y = clf.make_dataset(name, seed=args.seed)

        result = evolve(
            # Only differentiable activations: the step function has zero
            # gradient, so a node using it would be dead weight to backprop.
            fitness_fn=clf.make_fitness_fn(X, y, verbose=not args.quiet),
            key=jax.random.PRNGKey(args.seed),
            rng=np.random.default_rng(args.seed),
            n_in=clf.N_IN_BP,
            n_out=clf.N_OUT_BP,
            act_choices=DIFFERENTIABLE_ACTS,
            use_crossover=not args.no_crossover,
            use_speciation=not args.no_speciation,
            verbose=not args.quiet,
        )

        best = result.champion
        acc = clf.accuracy(best.conn_w, best, X, y)
        counts = clf.activation_counts(best)
        totals += counts
        results[name] = dict(best=best, X=X, y=y, acc=acc, result=result)

        used = ", ".join(
            f"{ACT_NAMES[a]} {counts[a]}" for a in np.argsort(-counts) if counts[a] > 0
        )
        print(
            f"  -> {name}: accuracy {acc:.1%}, {int(n_hidden(best))} hidden nodes, "
            f"activations: {used or 'none'}\n"
        )

        save_genome(best, path(f"best_{name}.npz"))
        clf.plot_decision_boundary(
            best,
            X,
            y,
            title=f"{name}  ({int(n_hidden(best))} hidden, acc {acc:.0%})",
            save_path=path(f"boundary_{name}.png"),
        )
        draw_genome(
            best,
            title=f"{name}",
            label_activations=True,
            input_labels=clf.INPUT_LABELS,
            figsize=(6, 4),
            save_path=path(f"topology_{name}.png"),
        )
        plot_fitness_and_complexity(
            result.history,
            result.hidden_hist,
            title=f"Backprop NEAT on {name}",
            save_path=path(f"fitness_{name}.png"),
        )

    clf.plot_summary_grid(results, save_path=path("summary.png"))
    report_activation_usage(totals, ACT_NAMES)

    print(f"\n{'dataset':10s} {'accuracy':>9s} {'hidden':>7s}")
    print("-" * 28)
    for name, r in results.items():
        print(f"{name:10s} {r['acc']:9.1%} {int(n_hidden(r['best'])):7d}")

    print(f"\nWrote figures and genomes to {args.out}/")


if __name__ == "__main__":
    main()

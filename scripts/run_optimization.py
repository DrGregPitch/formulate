#!/usr/bin/env python3
"""Run the active-learning campaigns and draw the money plot.

    python scripts/run_optimization.py --outdir results

Two figures:

* ``money_plot.png`` -- best formulation found vs. number of experiments, active
  learning against random screening, averaged over restarts with an interquartile
  band. The horizontal gap between the curves is how many experiments active
  learning saves.
* ``cost_aware.png`` -- best found vs. cumulative experimental *cost*, standard vs.
  cost-aware acquisition (which penalises exotic comonomers). Shows you can reach
  the same property for less spend, not just fewer runs.

Controlled oracle (copolymer Tg via copolybench) -- the loop treats each cheap
oracle call as an expensive experiment and counts them, which is the whole point.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from polytools import silence_rdkit

from formulate import (
    GPSurrogate,
    build_design_space,
    experiments_to_target,
    run_experiment,
)

STRATEGIES = ("random", "greedy", "ucb", "ei")
COLORS = {"random": "#888888", "greedy": "#e69f00",
          "ucb": "#2b6cb0", "ei": "#009e73"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", default="results", type=Path)
    p.add_argument("--n-pairs", default=120, type=int)
    p.add_argument("--points-per-pair", default=12, type=int)
    p.add_argument("--budget", default=60, type=int)
    p.add_argument("--n-seed", default=5, type=int)
    p.add_argument("--restarts", default=20, type=int)
    p.add_argument("--target-frac", default=0.98, type=float)
    p.add_argument("--seed", default=0, type=int)
    p.add_argument("--no-figures", action="store_true")
    args = p.parse_args()

    silence_rdkit()
    args.outdir.mkdir(parents=True, exist_ok=True)

    space = build_design_space(args.n_pairs, args.points_per_pair, seed=args.seed)
    print(f"Design space: {len(space)} candidate formulations | "
          f"best achievable Tg {space.best_value:.0f} C")

    summ = run_experiment(
        space, lambda: GPSurrogate(), strategies=STRATEGIES,
        budget=args.budget, n_seed=args.n_seed, n_restarts=args.restarts,
    )
    ett = experiments_to_target(summ, args.target_frac)

    print(f"\nExperiments to reach {args.target_frac:.0%} of the optimum "
          f"(median over {args.restarts} restarts):")
    for s in STRATEGIES:
        v = ett[s]
        shown = f"{v:.0f}" if v <= args.budget else f">{args.budget}"
        print(f"  {s:8s} {shown}")
    best_al = min(v for s, v in ett.items() if s != "random")
    if ett["random"] > args.budget:
        print(f"\nActive learning reaches the target in ~{best_al:.0f} experiments; "
              f"random screening does not reach it within {args.budget}.")
    else:
        print(f"\nActive learning: ~{best_al:.0f} experiments vs random "
              f"{ett['random']:.0f} -- a {ett['random'] / best_al:.1f}x reduction.")

    # write the summary table
    with open(args.outdir / "experiments_to_target.md", "w") as fh:
        fh.write("| strategy | experiments to target |\n|:---|---:|\n")
        for s in STRATEGIES:
            v = ett[s]
            cell = f"{v:.0f}" if v <= args.budget else f">{args.budget}"
            fh.write(f"| {s} | {cell} |\n")

    if args.no_figures:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # --- money plot ---
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    xs = np.arange(1, args.budget + 1)
    for s in STRATEGIES:
        ax.plot(xs, summ.mean(s), color=COLORS[s], lw=2, label=s)
        lo, hi = summ.band(s)
        ax.fill_between(xs, lo, hi, color=COLORS[s], alpha=0.15)
    target = args.target_frac * space.best_value
    ax.axhline(target, color="crimson", ls="--", lw=1,
               label=f"{args.target_frac:.0%} of optimum")
    ax.axhline(space.best_value, color="k", ls=":", lw=1, label="optimum")
    ax.set_xlabel("number of experiments")
    ax.set_ylabel("best Tg found (C)")
    ax.set_title("Active learning finds the best formulation in far fewer experiments")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.outdir / "money_plot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- cost-aware extension: best found vs cumulative cost ---
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    # plot best-found against the median cumulative cost across restarts
    from formulate.loop import run_campaign
    for label, ca, col in (("UCB", False, "#2b6cb0"),
                           ("cost-aware UCB", True, "#009e73")):
        costs, bests = [], []
        for r in range(args.restarts):
            res = run_campaign(space, "ucb", lambda: GPSurrogate(), budget=args.budget,
                               n_seed=args.n_seed, cost_aware=ca, seed=r)
            costs.append(res.cost_so_far)
            bests.append(res.best_so_far)
        costs = np.vstack(costs).mean(axis=0)
        bests = np.vstack(bests).mean(axis=0)
        ax.plot(costs, bests, color=col, lw=2, label=label)
    ax.axhline(target, color="crimson", ls="--", lw=1, label=f"{args.target_frac:.0%} of optimum")
    ax.set_xlabel("cumulative experimental cost (arb. units)")
    ax.set_ylabel("best Tg found (C)")
    ax.set_title("Cost-aware acquisition reaches the target for less spend")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.outdir / "cost_aware.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigures written to {args.outdir}/")
    print("\nNOTE: controlled oracle (copolymer Tg), not experimental data. This "
          "demonstrates\nthe active-learning loop and the experiments it saves.")


if __name__ == "__main__":
    main()

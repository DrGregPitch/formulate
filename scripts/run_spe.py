#!/usr/bin/env python3
"""Active learning on REAL data: discover high-conductivity polymer electrolytes.

    python scripts/fetch_spe.py            # download the data first
    python scripts/run_spe.py --outdir results

Builds the design space from the solid polymer electrolyte conductivity dataset
(CheMixHub, MIT) and runs the same surrogate + acquisition loop as the synthetic
benchmark -- the only thing that changed is the design space. The money plot shows
how many real electrolyte experiments active learning saves against random
screening.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from formulate import (
    GPSurrogate,
    build_spe_design_space,
    experiments_to_target,
    run_experiment,
)

STRATEGIES = ("random", "greedy", "ucb", "ei")
COLORS = {"random": "#888888", "greedy": "#e69f00",
          "ucb": "#2b6cb0", "ei": "#009e73"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", default="results", type=Path)
    p.add_argument("--cache", default="data_cache", type=Path)
    p.add_argument("--budget", default=60, type=int)
    p.add_argument("--n-seed", default=5, type=int)
    p.add_argument("--restarts", default=20, type=int)
    p.add_argument("--target-frac", default=0.90, type=float)
    p.add_argument("--no-figures", action="store_true")
    args = p.parse_args()

    proc = args.cache / "processed_PolymerElectrolyteData.csv"
    comp = args.cache / "compounds.csv"
    if not proc.exists() or not comp.exists():
        raise SystemExit("Data not found. Run:  python scripts/fetch_spe.py")

    args.outdir.mkdir(parents=True, exist_ok=True)
    space = build_spe_design_space(str(proc), str(comp))
    best_scm = 10 ** space.best_value
    print(f"Design space: {len(space)} real polymer-electrolyte formulations")
    print(f"Best achievable log-conductivity {space.best_value:.2f} "
          f"({best_scm:.3f} S/cm)")

    summ = run_experiment(
        space, lambda: GPSurrogate(), strategies=STRATEGIES,
        budget=args.budget, n_seed=args.n_seed, n_restarts=args.restarts,
    )
    ett = experiments_to_target(summ, args.target_frac)
    pct = int(round((1 - args.target_frac) * 100))
    print(f"\nExperiments to reach the top {pct}% of the conductivity range "
          f"(median over {args.restarts} restarts):")
    for s in STRATEGIES:
        v = ett[s]
        print(f"  {s:8s} {v:.0f}" if v <= args.budget else f"  {s:8s} >{args.budget}")
    best_al = min(v for s, v in ett.items() if s != "random")
    if ett["random"] <= args.budget and best_al > 0:
        print(f"\nActive learning: ~{best_al:.0f} experiments vs random "
              f"{ett['random']:.0f} -- a {ett['random'] / best_al:.1f}x reduction.")

    with open(args.outdir / "spe_experiments_to_target.md", "w") as fh:
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

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    xs = np.arange(1, args.budget + 1)
    for s in STRATEGIES:
        ax.plot(xs, summ.mean(s), color=COLORS[s], lw=2, label=s)
        lo, hi = summ.band(s)
        ax.fill_between(xs, lo, hi, color=COLORS[s], alpha=0.15)
    target = summ.worst_value + args.target_frac * (summ.best_value - summ.worst_value)
    ax.axhline(target, color="crimson", ls="--", lw=1, label=f"top {pct}% of range")
    ax.axhline(space.best_value, color="k", ls=":", lw=1, label="best in pool")
    ax.set_xlabel("number of experiments")
    ax.set_ylabel("best log-conductivity found (log S/cm)")
    ax.set_title("Real data: active learning finds better electrolytes, faster\n"
                 "(CheMixHub solid polymer electrolytes)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.outdir / "spe_money_plot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure written to {args.outdir}/spe_money_plot.png")
    print("\nReal measured data (curated from the literature via CheMixHub). This "
          "is the\nbattery-electrolyte formulation problem; the loop counts each "
          "measurement as an experiment.")


if __name__ == "__main__":
    main()

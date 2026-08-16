"""Run many campaigns and aggregate them into the money plot.

A single active-learning run is noisy -- it depends on which formulations happened
to seed it. The honest result averages over many restarts and shows the spread, so
"active learning wins" is a distribution statement, not a lucky seed. This module
runs each strategy over N restarts and reports the mean best-found trace with an
interquartile band, plus the headline number: how many experiments each strategy
needs to reach a target property.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .design_space import DesignSpace
from .loop import run_campaign

__all__ = ["ExperimentSummary", "run_experiment", "experiments_to_target"]


@dataclass
class ExperimentSummary:
    """Aggregated best-found traces per strategy over restarts."""

    budget: int
    best_value: float
    traces: dict[str, np.ndarray] = field(default_factory=dict)  # strategy -> (restarts, budget)

    def mean(self, strategy: str) -> np.ndarray:
        return self.traces[strategy].mean(axis=0)

    def band(self, strategy: str) -> tuple[np.ndarray, np.ndarray]:
        t = self.traces[strategy]
        return np.percentile(t, 25, axis=0), np.percentile(t, 75, axis=0)


def run_experiment(
    space: DesignSpace,
    surrogate_factory,
    strategies=("random", "ucb", "ei"),
    budget: int = 60,
    n_seed: int = 5,
    n_restarts: int = 20,
    beta: float = 2.0,
    cost_aware: bool = False,
) -> ExperimentSummary:
    """Run each strategy over ``n_restarts`` seeds; collect best-found traces."""
    summary = ExperimentSummary(budget=budget, best_value=space.best_value)
    for strat in strategies:
        runs = []
        for r in range(n_restarts):
            res = run_campaign(
                space, strat, surrogate_factory, budget=budget, n_seed=n_seed,
                beta=beta, cost_aware=cost_aware, seed=r,
            )
            runs.append(res.best_so_far)
        summary.traces[strat] = np.vstack(runs)
    return summary


def experiments_to_target(
    summary: ExperimentSummary,
    target_frac: float = 0.98,
) -> dict[str, float]:
    """Median number of experiments each strategy needs to reach the target.

    ``target_frac`` is a fraction of the global best property in the pool (0.98 =
    within 2% of the optimum). Returns NaN for a strategy that fails to reach it
    within budget in over half its restarts -- which is itself the finding for
    random search on a hard space.
    """
    target = target_frac * summary.best_value
    out = {}
    for strat, t in summary.traces.items():
        reached = []
        for row in t:
            hit = int(np.argmax(row >= target))
            # censor restarts that never reach the target at budget+1, so a
            # strategy that usually fails is not flattered by its lucky runs.
            reached.append(hit + 1 if row[hit] >= target else summary.budget + 1)
        out[strat] = float(np.median(reached))
    return out

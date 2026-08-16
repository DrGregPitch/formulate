"""The pool-based active-learning loop.

Start from a few random "experiments", fit a surrogate, let the acquisition
function choose the next formulation to run from the remaining pool, reveal its
true property, retrain, repeat. Track the best property found after each
experiment -- that trace, averaged over restarts, is the money plot.

This is the shape of real experimental campaigns: a closed loop of propose ->
measure -> update, under a hard budget on how many measurements you can afford.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .acquisition import acquisition_scores
from .design_space import DesignSpace

__all__ = ["CampaignResult", "run_campaign"]


@dataclass
class CampaignResult:
    """One active-learning run's trace."""

    strategy: str
    best_so_far: np.ndarray     # best true value after each experiment
    queried: np.ndarray         # candidate index queried at each step
    cost_so_far: np.ndarray     # cumulative experimental cost after each step

    @property
    def n_experiments(self) -> int:
        return len(self.best_so_far)


def run_campaign(
    space: DesignSpace,
    strategy: str,
    surrogate_factory,
    budget: int = 60,
    n_seed: int = 5,
    beta: float = 2.0,
    cost_aware: bool = False,
    seed: int = 0,
) -> CampaignResult:
    """Run one campaign: seed, then acquire until the budget is spent.

    Parameters
    ----------
    surrogate_factory
        Zero-arg callable returning a fresh surrogate with ``fit`` /
        ``predict(return_std=True)``.
    cost_aware
        If True, divide the acquisition score by candidate cost, so the loop
        prefers cheaper experiments of comparable promise.
    """
    rng = np.random.default_rng(seed)
    n = len(space)
    budget = min(budget, n)

    queried: list[int] = list(rng.choice(n, size=min(n_seed, n), replace=False))
    remaining = set(range(n)) - set(queried)

    def trace(idx_list):
        vals = space.y_true[idx_list]
        return np.maximum.accumulate(vals)

    best_so_far = list(trace(queried))
    cost_so_far = list(np.cumsum(space.cost[queried]))

    while len(queried) < budget and remaining:
        rem = np.fromiter(remaining, dtype=int)
        if strategy == "random":
            scores = rng.random(len(rem))
        else:
            surrogate = surrogate_factory().fit(space.X[queried], space.y_true[queried])
            mu, sigma = surrogate.predict(space.X[rem], return_std=True)
            best = float(np.max(space.y_true[queried]))
            scores = acquisition_scores(strategy, mu, sigma, best, rng, beta)
        if cost_aware:
            scores = scores / space.cost[rem]

        pick = int(rem[int(np.argmax(scores))])
        queried.append(pick)
        remaining.discard(pick)
        best_so_far.append(max(best_so_far[-1], float(space.y_true[pick])))
        cost_so_far.append(cost_so_far[-1] + float(space.cost[pick]))

    return CampaignResult(
        strategy=strategy,
        best_so_far=np.asarray(best_so_far),
        queried=np.asarray(queried),
        cost_so_far=np.asarray(cost_so_far),
    )

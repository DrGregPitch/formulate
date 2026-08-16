"""Acquisition functions: given surrogate predictions, which candidate next?

Each takes the surrogate mean ``mu`` and uncertainty ``sigma`` over the unqueried
pool and returns a score per candidate; the loop queries the argmax. The contrast
between them is the lesson:

* ``random`` -- ignore the surrogate entirely. The baseline every active-learning
  result must beat, and the one a lot of real screening still uses.
* ``greedy`` -- pure exploitation: query the highest predicted value. Fast early,
  but gets stuck in whatever basin the seed points suggested.
* ``ucb`` -- ``mu + beta * sigma``. Explicitly trades exploitation against
  exploration; ``beta`` tunes how adventurous.
* ``ei`` -- expected improvement over the best value seen so far. The classic
  Bayesian-optimisation choice; explores automatically where improvement is
  plausible, without a ``beta`` to hand-tune.

All are written for *maximisation* (find the highest-property formulation).
"""

from __future__ import annotations

import numpy as np
from scipy import stats

__all__ = ["ACQUISITIONS", "acquisition_scores"]


def _random(mu, sigma, best, rng, beta):
    return rng.random(len(mu))


def _greedy(mu, sigma, best, rng, beta):
    return mu


def _ucb(mu, sigma, best, rng, beta):
    return mu + beta * sigma


def _ei(mu, sigma, best, rng, beta):
    sigma = np.clip(sigma, 1e-9, None)
    z = (mu - best) / sigma
    return (mu - best) * stats.norm.cdf(z) + sigma * stats.norm.pdf(z)


#: Registry of acquisition strategies.
ACQUISITIONS = {
    "random": _random,
    "greedy": _greedy,
    "ucb": _ucb,
    "ei": _ei,
}


def acquisition_scores(
    kind: str,
    mu: np.ndarray,
    sigma: np.ndarray,
    best: float,
    rng: np.random.Generator,
    beta: float = 2.0,
) -> np.ndarray:
    """Score each candidate under the named acquisition (higher = query sooner)."""
    if kind not in ACQUISITIONS:
        raise ValueError(f"unknown acquisition {kind!r}; choose from {list(ACQUISITIONS)}")
    return ACQUISITIONS[kind](mu, sigma, best, rng, beta)

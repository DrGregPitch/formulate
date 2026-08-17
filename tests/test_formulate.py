"""Tests for formulate, organised by invariant.

Kept fast: a small design space, short budgets, few restarts. The headline test
asserts the whole point -- active learning reaches a good formulation in fewer
experiments than random -- on a small but sufficient pool.
"""

from __future__ import annotations

import numpy as np
import pytest

from formulate import (
    GPSurrogate,
    acquisition_scores,
    build_design_space,
    build_spe_design_space,
    experiments_to_target,
    run_campaign,
    run_experiment,
)
from formulate.acquisition import ACQUISITIONS


@pytest.fixture(scope="module")
def space():
    return build_design_space(n_pairs=40, points_per_pair=10, seed=0)


# --------------------------------------------------------------------------
# design space
# --------------------------------------------------------------------------

def test_design_space_shapes_and_oracle(space):
    assert len(space) == 40 * 10
    assert space.X.shape == (len(space), len(space.feature_names))
    assert np.isfinite(space.X).all() and np.isfinite(space.y_true).all()
    assert space.best_value == pytest.approx(space.y_true.max())


def test_cost_is_positive_and_varies(space):
    assert (space.cost > 0).all()
    assert space.cost.std() > 0    # exotic vs common comonomers differ in cost


# --------------------------------------------------------------------------
# surrogate
# --------------------------------------------------------------------------

def test_gp_surrogate_predicts_mean_and_positive_std(space):
    idx = np.arange(30)
    gp = GPSurrogate().fit(space.X[idx], space.y_true[idx])
    mu, sd = gp.predict(space.X[30:60], return_std=True)
    assert mu.shape == sd.shape == (30,)
    assert np.isfinite(mu).all() and (sd > 0).all()


# --------------------------------------------------------------------------
# acquisition
# --------------------------------------------------------------------------

def test_all_acquisitions_return_scores():
    rng = np.random.default_rng(0)
    mu = np.array([1.0, 2.0, 3.0])
    sigma = np.array([1.0, 1.0, 0.1])
    for kind in ACQUISITIONS:
        s = acquisition_scores(kind, mu, sigma, best=2.0, rng=rng)
        assert s.shape == (3,) and np.isfinite(s).all()


def test_ei_is_non_negative_and_rewards_uncertainty():
    rng = np.random.default_rng(0)
    mu = np.array([1.9, 1.9])
    sigma = np.array([0.1, 1.0])    # same mean below best, different uncertainty
    ei = acquisition_scores("ei", mu, sigma, best=2.0, rng=rng)
    assert (ei >= 0).all()
    assert ei[1] > ei[0]            # more uncertainty -> more expected improvement


def test_unknown_acquisition_raises():
    with pytest.raises(ValueError):
        acquisition_scores("bogus", np.zeros(2), np.ones(2), 0.0, np.random.default_rng(0))


# --------------------------------------------------------------------------
# loop
# --------------------------------------------------------------------------

def test_campaign_best_is_monotonic_and_right_length(space):
    res = run_campaign(space, "ucb", lambda: GPSurrogate(), budget=20, n_seed=4, seed=0)
    assert res.n_experiments == 20
    assert np.all(np.diff(res.best_so_far) >= 0)          # best-so-far never decreases
    assert np.all(np.diff(res.cost_so_far) >= 0)          # cost accumulates
    assert res.best_so_far[-1] <= space.best_value + 1e-9  # cannot beat the oracle max


# --------------------------------------------------------------------------
# experiment (the headline)
# --------------------------------------------------------------------------

def test_active_learning_beats_random(space):
    summ = run_experiment(
        space, lambda: GPSurrogate(), strategies=("random", "ucb"),
        budget=30, n_seed=4, n_restarts=6,
    )
    # after the same number of experiments, UCB has found a better formulation
    assert summ.mean("ucb")[-1] > summ.mean("random")[-1]


def test_experiments_to_target_censors_non_reachers(space):
    summ = run_experiment(
        space, lambda: GPSurrogate(), strategies=("random", "ucb"),
        budget=30, n_seed=4, n_restarts=6,
    )
    ett = experiments_to_target(summ, target_frac=0.98)
    # UCB should reach the target in no more experiments than random (usually far fewer)
    assert ett["ucb"] <= ett["random"]


def test_experiments_to_target_handles_negative_values():
    """Range-relative target must work when the property is negative (e.g. log
    conductivity), where the old `frac * best` definition was nonsense."""
    from formulate.experiment import ExperimentSummary
    # two strategies, values run from -15 (worst) to -1 (best); one reaches high fast
    s = ExperimentSummary(budget=5, best_value=-1.0, worst_value=-15.0)
    s.traces = {
        "fast": np.array([[-15, -10, -2, -2, -2]], dtype=float),
        "slow": np.array([[-15, -14, -13, -12, -11]], dtype=float),
    }
    ett = experiments_to_target(s, target_frac=0.9)  # target = -15 + 0.9*14 = -2.4
    assert ett["fast"] == 3        # reaches -2 at experiment 3
    assert ett["slow"] > s.budget  # never gets close


# --------------------------------------------------------------------------
# real data (solid polymer electrolytes)
# --------------------------------------------------------------------------

def test_spe_design_space_from_fixture(tmp_path):
    """The SPE loader converts [Cu]/[Au] to [*], keeps binary polymer+salt rows,
    and yields a usable design space. Uses a tiny in-test fixture (no network)."""
    import pandas as pd
    compounds = pd.DataFrame({
        "compound_id": [1, 2, 10, 11],
        "smiles": ["CC", "CCC", "[Li+].[F-]", "[Li+].[Cl-]"],
        "salt": [0, 0, 1, 1],
        "polymer": [1, 1, 0, 0],
        "monomeric_unit": ["[Cu]CC[Au]", "[Cu]CC(C)[Au]", None, None],
    })
    processed = pd.DataFrame({
        "cmp1_mn_or_mw": [50000.0, 50000.0, 80000.0, 80000.0],
        "value": [-3.0, -4.0, -2.0, -5.0],       # log conductivity
        "property": ["log Conductivity"] * 4,
        "unit": ["S/cm"] * 4,
        "Temperature, K": [300.0, 320.0, 300.0, 350.0],
        "cmp_ids": ["[1, 10]", "[1, 11]", "[2, 10]", "[2, 11]"],
        "cmp_mole_fractions": ["[0.8, 0.2]"] * 4,
        "cmp_mws": ["[44.0, 100.0]"] * 4,
    })
    pc = tmp_path / "compounds.csv"
    pp = tmp_path / "processed.csv"
    compounds.to_csv(pc, index=False)
    processed.to_csv(pp, index=False)

    space = build_spe_design_space(str(pp), str(pc))
    assert len(space) == 4                         # all four are binary polymer+salt
    assert space.X.shape[1] == len(space.feature_names)
    assert np.isfinite(space.X).all()
    np.testing.assert_allclose(sorted(space.y_true), [-5.0, -4.0, -3.0, -2.0])
    assert space.best_value == pytest.approx(-2.0)

"""Surrogate models: cheap stand-ins for the expensive oracle, with uncertainty.

An acquisition function needs two things from a surrogate at every candidate: a
predicted mean and a predicted uncertainty. Point predictions are not enough -- the
whole idea of active learning is to trade off *exploiting* high predicted values
against *exploring* where the model is unsure.

Two surrogates, same ``fit`` / ``predict(return_std=True)`` interface:

* :class:`GPSurrogate` -- a Gaussian process (scikit-learn). The textbook choice for
  Bayesian optimisation: principled, closed-form uncertainty, excellent in the
  small-data regime an experimental budget lives in. Features and target are
  standardised internally because a GP's length scales assume it.
* :class:`EnsembleSurrogate` -- the polytools gradient-boosting ensemble, whose
  spread gives the uncertainty. More robust in higher dimensions; a useful contrast
  to show the loop is not wedded to a GP.
"""

from __future__ import annotations

import numpy as np
from polytools import EnsembleRegressor, GBMRegressor

__all__ = ["GPSurrogate", "EnsembleSurrogate"]


class GPSurrogate:
    """Gaussian-process surrogate with standardised inputs and target."""

    name = "gp"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self._gp = None
        self._x_mean = None
        self._x_std = None
        self._y_mean = 0.0
        self._y_std = 1.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> GPSurrogate:
        import warnings

        from sklearn.exceptions import ConvergenceWarning
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self._x_mean = X.mean(axis=0)
        self._x_std = X.std(axis=0)
        self._x_std[self._x_std == 0] = 1.0
        self._y_mean = float(y.mean())
        self._y_std = float(y.std()) or 1.0

        Xs = (X - self._x_mean) / self._x_std
        ys = (y - self._y_mean) / self._y_std

        kernel = (
            ConstantKernel(1.0, (1e-3, 1e3))
            * RBF(length_scale=np.ones(X.shape[1]), length_scale_bounds=(1e-2, 1e4))
            + WhiteKernel(1e-2, (1e-6, 1e1))
        )
        self._gp = GaussianProcessRegressor(
            kernel=kernel, normalize_y=False, n_restarts_optimizer=0,
            random_state=self.seed,
        )
        # a hit-the-bound length scale is fine for acquisition ranking and just
        # spams the log across hundreds of refits; silence that one warning.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self._gp.fit(Xs, ys)
        return self

    def predict(self, X: np.ndarray, return_std: bool = False):
        Xs = (np.asarray(X, dtype=float) - self._x_mean) / self._x_std
        mu, sd = self._gp.predict(Xs, return_std=True)
        mu = mu * self._y_std + self._y_mean
        if return_std:
            return mu, np.clip(sd * self._y_std, 1e-9, None)
        return mu


class EnsembleSurrogate:
    """Gradient-boosting ensemble surrogate; uncertainty is the member spread."""

    name = "ensemble"

    def __init__(self, n_members: int = 5, seed: int = 0) -> None:
        self.n_members = n_members
        self.seed = seed
        self._model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> EnsembleSurrogate:
        self._model = EnsembleRegressor(
            factory=lambda i: GBMRegressor(
                n_estimators=200, random_state=self.seed + i,
                quantile_uncertainty=False, backend="sklearn",
            ),
            n_members=self.n_members, bootstrap=True, include_member_sigma=False,
        ).fit(np.asarray(X, dtype=float), np.asarray(y, dtype=float))
        return self

    def predict(self, X: np.ndarray, return_std: bool = False):
        return self._model.predict(np.asarray(X, dtype=float), return_std=return_std)

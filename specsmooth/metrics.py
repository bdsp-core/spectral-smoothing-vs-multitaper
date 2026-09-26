"""Variance bookkeeping for quadratic spectral estimators."""
import numpy as np


def equivalent_dof(est):
    """Chi-square equivalent degrees of freedom from a sample of estimates: 2 * mean^2 / var."""
    est = np.asarray(est, float)
    return 2.0 / np.var(est / est.mean())


def dof_quadratic(Q):
    """Exact equivalent dof of x^H Q x for real unit white Gaussian x: tr(Re Q)^2 / tr((Re Q)^2)."""
    R = np.real(np.asarray(Q))
    R = (R + R.T) / 2
    return np.trace(R) ** 2 / np.trace(R @ R)


def mc_white_noise(est_fn, N, M, rng):
    """Apply a scalar-valued estimator to M white-noise records of length N."""
    X = rng.standard_normal((N, M))
    return np.array([est_fn(X[:, m]) for m in range(M)])

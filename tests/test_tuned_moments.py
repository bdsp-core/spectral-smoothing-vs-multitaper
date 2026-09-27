"""The exact mean and variance of a quadratic estimator (demos/tuned_comparison.moments), which shortlist the settings of the
tuned comparison (Table IV). For real Gaussian x with covariance Sigma and a real symmetric Q, x_f^* Q x_f = x' Q_f x with
(Q_f)_{k,l} = Q_{k,l} cos(2 pi (k - l) f), so its mean is tr(Q_f Sigma) and its variance 2 tr(Q_f Sigma Q_f Sigma), exactly."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "demos"))
import numpy as np
from scipy.linalg import toeplitz, cholesky
import specsmooth as ss
import tuned_comparison as tc


def _banks(N):
    t = np.arange(N)
    tuk = ss.unit_taper("tukey", N, alpha=0.25)
    V, _ = ss.dpss(N, 2, 3)
    Qp = np.outer(tuk, tuk) * ss.parabolic_lag_window(N, np.sqrt(2) * 2 / N)[np.abs(t[:, None] - t[None, :])]
    return {"multitaper": (V, np.full(3, 1 / 3)), "recipe (b')": tc.bank(Qp)}


def test_exact_moments_match_the_closed_form():
    N = 48; t = np.arange(N)
    r = tc.acf(lambda f: ss.ar_psd(ss.AR4, f), N); Sig = toeplitz(r); T = tc.Toeplitz(r)
    for V, c in _banks(N).values():
        Q = (V * c) @ V.T
        for f in (0.0, 0.07, 0.25, 0.5):
            m, v = tc.moments(V, c, T, [f])
            Qf = Q * np.cos(2 * np.pi * f * (t[:, None] - t[None, :]))
            mean, var = np.trace(Qf @ Sig), 2 * np.trace(Qf @ Sig @ Qf @ Sig)
            assert abs(m[0] - mean) < 1e-10 * mean and abs(v[0] - var) < 1e-9 * var


def test_exact_moments_match_monte_carlo():
    N, M = 48, 20000
    r = tc.acf(lambda f: ss.ar_psd(ss.AR4, f), N); T = tc.Toeplitz(r)
    X = (cholesky(toeplitz(r), lower=True) @ np.random.default_rng(0).standard_normal((N, M))).T
    f = 0.07; nfft = 8 * N; j = int(round(f * nfft)); f = j / nfft
    for V, c in _banks(N).values():
        m, v = tc.moments(V, c, T, [f])
        est = tc.mc_estimates(V, c, X, nfft, np.array([j]))[:, 0]
        assert abs(est.mean() / m[0] - 1) < 4 * np.sqrt(v[0] / M) / m[0]            # four standard errors of the mean
        assert abs(est.var() / v[0] - 1) < 0.05

"""Averaging periodograms, smoothing a periodogram and multitaper estimation are one family of quadratic estimators.

Tested: (1) sliding a window with unit step and overhang is exactly a lag-window (smoothed-periodogram) estimate whose lag
window is the window's autocorrelation; (2) with a sinc window it converges (as 1/L) to Thomson's eigenvalue-weighted
all-taper estimate, i.e. the box-smoothed periodogram; (3) the eigen-tapers of the box-smoothed periodogram are the
Slepian sequences with weights lambda_k / 2NW; (4) the dof of any estimator is 2 (sum c)^2 / sum c^2 over its eigen-weights.
"""
import numpy as np
import specsmooth as ss

N, NW = 256, 4
W = NW / N


def test_sliding_window_is_lag_window_estimate():
    rng = np.random.default_rng(0)
    x = ss.ar_process(ss.AR4, N, rng)
    w = ss.unit_taper("hann", 64)
    r = np.correlate(w, w, "full")[len(w) - 1:]            # autocorrelation, tau = 0..L-1
    h = np.zeros(N); h[:len(r)] = r / r[0]
    S_slide, _ = ss.welch_sliding(x, w, 2048, step=1, overhang=True)
    S_lag, _ = ss.lag_window_estimate(x, h, 2048)
    assert np.max(np.abs(S_slide / S_lag - 1)) < 1e-10


def test_sliding_sinc_converges_to_thomson_all_tapers():
    A = ss.sinc_toeplitz(N, W) / (2 * N * W)
    errs = []
    for L in [N, 2 * N, 4 * N, 16 * N, 32 * N]:
        Q = ss.quadratic_matrix(N, 0.0, "welch", taper=ss.sinc_window(L, W), step=1, overhang=True).real
        errs.append(np.linalg.norm(Q - A) / np.linalg.norm(A))
    assert errs[0] < 0.15 and errs[1] < 0.04 and errs[2] < 0.02 and errs[3] < 0.005 and errs[4] < 0.0025
    assert all(a > b for a, b in zip(errs, errs[1:]))
    assert 1.5 < errs[1] / errs[2] < 2.7 and 1.5 < errs[3] / errs[4] < 2.7      # ~1/L beyond L = 2N (halving L's error per doubling)
    rng = np.random.default_rng(1)
    x = ss.ar_process(ss.AR4, N, rng)
    V, lam = ss.dpss_all(N, W)
    S_mt = ss.multitaper(x, V, 2048, weights=lam)[0]
    S_box = ss.lag_window_estimate(x, ss.box_lag_window(N, W), 2048)[0]
    S_sl = ss.welch_sliding(x, ss.sinc_window(16 * N, W), 2048)[0]
    assert np.max(np.abs(S_mt / S_box - 1)) < 1e-9
    assert np.max(np.abs(S_sl / S_box - 1)) < 0.02


def test_eigen_tapers_of_box_smoothed_periodogram_are_slepians():
    Q = ss.quadratic_matrix(N, 0.0, "lagwindow", h=ss.box_lag_window(N, W)).real
    c, U = ss.eigen_tapers(Q)
    V, lam = ss.dpss_all(N, W)
    assert np.allclose(c, lam / (2 * N * W), atol=1e-10)
    for k in range(6):                                          # non-degenerate leading eigenvectors, up to sign
        assert min(np.linalg.norm(U[:, k] - V[:, k]), np.linalg.norm(U[:, k] + V[:, k])) < 1e-6


def test_dof_from_eigen_weights():
    hann = ss.unit_taper("hann", N)
    Vk, _ = ss.dpss(N, NW)
    for Q0, Qf in [(ss.quadratic_matrix(N, 0.0, "multitaper", tapers=Vk), ss.quadratic_matrix(N, 0.25, "multitaper", tapers=Vk)),
                   (ss.quadratic_matrix(N, 0.0, "lagwindow", h=ss.box_lag_window(N, W), taper=hann),
                    ss.quadratic_matrix(N, 0.25, "lagwindow", h=ss.box_lag_window(N, W), taper=hann)),
                   (ss.quadratic_matrix(N, 0.0, "welch", taper=ss.unit_taper("hann", 64), overlap=0.5),
                    ss.quadratic_matrix(N, 0.25, "welch", taper=ss.unit_taper("hann", 64), overlap=0.5))]:
        c, _ = ss.eigen_tapers(Q0)
        assert abs(ss.dof_from_weights(c) / ss.dof_quadratic(Qf) - 1) < 0.03
    assert abs(ss.dof_from_weights(np.ones(7)) - 14) < 1e-12


def test_dof_formula_matches_monte_carlo_for_all_three_families():
    """The trace formula for nu (Appendix B) against Monte Carlo on white noise, at an interior frequency."""
    rng = np.random.default_rng(7)
    hann = ss.unit_taper("hann", N); hb = ss.box_lag_window(N, W); Vk, _ = ss.dpss(N, NW)
    hseg = ss.unit_taper("hann", 64); f0 = 0.25; nfft = 1024; j = int(round(f0 * nfft))
    cases = {
        "multitaper": (lambda x: ss.multitaper(x, Vk, nfft)[0][j], ss.quadratic_matrix(N, f0, "multitaper", tapers=Vk)),
        "Hann+box": (lambda x: ss.lag_window_estimate(x, hb, nfft, taper=hann)[0][j], ss.quadratic_matrix(N, f0, "lagwindow", h=hb, taper=hann)),
        "Welch": (lambda x: ss.welch_sliding(x, hseg, nfft, step=32, overhang=False)[0][j], ss.quadratic_matrix(N, f0, "welch", taper=hseg, step=32)),
    }
    for name, (fn, Q) in cases.items():
        est = ss.mc_white_noise(fn, N, 3000, rng)
        assert abs(ss.equivalent_dof(est) / ss.dof_quadratic(Q) - 1) < 0.15, name

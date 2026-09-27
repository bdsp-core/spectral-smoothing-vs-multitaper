"""The central claims, as tests."""
import numpy as np
import specsmooth as ss


def _setup(N=128, NW=3, seed=0):
    rng = np.random.default_rng(seed)
    W = NW / N
    x = ss.ar_process(ss.AR4, N, rng)
    x = x / x.std()
    return N, NW, W, x


def test_exact_identity_all_tapers_equals_box_smoothed_periodogram():
    N, NW, W, x = _setup()
    nfft = 8 * N
    V, lam = ss.dpss_all(N, W)
    Sk = np.abs(np.fft.fft(V * x[:, None], nfft, axis=0)) ** 2
    lhs = Sk @ lam                                              # sum_k lambda_k |J_k(f)|^2
    S_box, _ = ss.lag_window_estimate(x, ss.box_lag_window(N, W), nfft)   # (1/2W) int_{f-W}^{f+W} I/N
    rhs = 2 * W * N * S_box                                     # = int_{f-W}^{f+W} I(f') df'
    assert np.max(np.abs(lhs - rhs)) / np.max(rhs) < 1e-10


def test_eigenvalues_sum_to_about_2NW():
    N, NW, W, _ = _setup()
    _, lam = ss.dpss_all(N, W)
    assert abs(lam.sum() - 2 * NW) < 1e-8


def test_standard_multitaper_vs_box_smoothed_raw_periodogram():
    """Claim 2 is conditional: K-taper MT ~ box-smoothed RAW periodogram only when leakage is negligible."""
    N, NW, W, _ = _setup()
    nfft = 8 * N
    V, lam = ss.dpss(N, NW)
    rng = np.random.default_rng(5)
    # (a) white noise + a tone (tiny dynamic range): the two agree closely
    x = rng.standard_normal(N) + np.cos(2 * np.pi * 0.2 * np.arange(N))
    d = np.abs(10 * np.log10(ss.multitaper(x, V, nfft)[0] / ss.lag_window_estimate(x, ss.box_lag_window(N, W), nfft)[0]))
    assert np.median(d) < 0.5
    # (b) AR(4), 65 dB dynamic range: smoothing cannot undo periodogram leakage. Median |dB| error vs the true PSD,
    #     over 200 realizations (demos/outputs/short_record_leakage.txt): raw periodogram + box 13.4 dB, equal-weight
    #     MT K=2NW-1 7.1 dB (its last taper leaks), K=2NW-2 3.5 dB, Hann periodogram + box 2.1 dB. The single
    #     realization below is more extreme (about 20, 12, 6 and 2 dB); only the ordering is asserted.
    x = ss.ar_process(ss.AR4, N, rng)
    f = np.arange(nfft) / nfft
    truth = ss.ar_psd(ss.AR4, f)
    err = lambda S: np.median(np.abs(10 * np.log10(S / truth)))
    hb = ss.box_lag_window(N, W)
    e_raw = err(ss.lag_window_estimate(x, hb, nfft)[0])
    e_hann = err(ss.lag_window_estimate(x, hb, nfft, taper=ss.unit_taper("hann", N))[0])
    e_mt = err(ss.multitaper(x, V, nfft)[0])
    e_mt2 = err(ss.multitaper(x, V[:, :-1], nfft)[0])
    assert e_raw > 10
    assert e_hann < 4
    assert e_hann < e_mt2 < e_mt < e_raw


def test_variance_ordering_at_equal_design_bandwidth():
    N, NW, W, _ = _setup()
    Vk, _ = ss.dpss(N, NW)
    hb = ss.box_lag_window(N, W)
    dof_mt = ss.dof_quadratic(ss.quadratic_matrix(N, 0.25, "multitaper", tapers=Vk))
    dof_raw = ss.dof_quadratic(ss.quadratic_matrix(N, 0.25, "lagwindow", h=hb))
    dof_hann = ss.dof_quadratic(ss.quadratic_matrix(N, 0.25, "lagwindow", h=hb, taper=ss.unit_taper("hann", N)))
    V, lam = ss.dpss_all(N, W)
    dof_full = ss.dof_quadratic(ss.quadratic_matrix(N, 0.25, "multitaper", tapers=V, weights=lam))
    assert abs(dof_full - dof_raw) < 1e-6          # the exact identity, seen through the variance
    assert dof_raw > dof_mt > dof_hann              # raw+box (all data, no taper) > MT (2K) > single taper + box


def test_kernels_unit_area_and_multitaper_kernel_is_box_convolved_fejer():
    N, NW, W, _ = _setup()
    nfft = 8 * N
    V, lam = ss.dpss_all(N, W)
    H_full = ss.kernel_multitaper(V, nfft, weights=lam)
    H_fej = ss.kernel_smoothed(np.ones(N) / np.sqrt(N), ss.box_lag_window(N, W), nfft)
    assert abs(H_full.sum() / nfft - 1) < 1e-12
    assert np.max(np.abs(H_full - H_fej)) / H_fej.max() < 1e-9


def test_grid_smoothing_matches_lag_window_for_gaussian_kernel():
    N, NW, W, x = _setup()
    nfft = 16 * N
    sig = W / 2
    I, _ = ss.periodogram(x, nfft)
    S_grid = ss.smooth_on_grid(I, ss.kernel_gaussian(nfft, sig))
    S_lag, _ = ss.lag_window_estimate(x, ss.gaussian_lag_window(N, sig), nfft)
    assert np.max(np.abs(S_grid - S_lag)) / S_lag.max() < 1e-6


def test_dof_formula_matches_monte_carlo():
    N, NW, W, _ = _setup()
    rng = np.random.default_rng(1)
    V, lam = ss.dpss(N, NW)
    f0 = 0.25
    Q = ss.quadratic_matrix(N, f0, "multitaper", tapers=V)
    exact = ss.dof_quadratic(Q)
    nfft = 4 * N
    j = int(round(f0 * nfft))
    mc = ss.equivalent_dof(ss.mc_white_noise(lambda x: ss.multitaper(x, V, nfft)[0][j], N, 4000, rng))
    assert abs(exact - 2 * V.shape[1]) < 0.05 * 2 * V.shape[1]
    assert abs(mc - exact) / exact < 0.15

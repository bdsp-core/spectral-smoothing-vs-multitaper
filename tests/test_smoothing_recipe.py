"""The light taper and parabolic kernel of recipe (b'), and the kernel-matched lag window."""
import numpy as np
import specsmooth as ss


def test_parabolic_kernel_shape():
    N, W = 256, 6 / 256; nfft = 16 * N
    H = ss.kernel_from_lag(ss.parabolic_lag_window(N, W), nfft)
    f = ss.signed_freq(nfft)
    target = np.where(np.abs(f) <= W, 0.75 / W * (1 - (f / W) ** 2), 0.0)
    assert abs(H.sum() / nfft - 1) < 1e-9
    assert np.abs(H * 1.0 - target).max() < 0.05 * target.max()              # a truncated lag window ripples slightly
    assert abs(ss.kernel_stats(H)["bw3"] - np.sqrt(2) * W) < 2 / nfft + 0.03 * W


def test_tukey_taper_is_between_rectangular_and_hann():
    N = 256; hb = ss.box_lag_window(N, 4 / N)
    nu = [ss.dof_quadratic(ss.quadratic_matrix(N, 0.25, "lagwindow", h=hb, taper=w))
          for w in (None, ss.unit_taper("tukey", N, alpha=0.25), ss.unit_taper("hann", N))]
    assert nu[0] > nu[1] > nu[2]
    assert abs(np.linalg.norm(ss.unit_taper("tukey", N, alpha=0.25)) - 1) < 1e-12


def test_matched_lag_window_reproduces_the_multitaper_kernel():
    N, NW, K = 128, 4, 7
    V, _ = ss.dpss(N, NW, K); Q = V @ V.T / K
    H_mt = ss.kernel_multitaper(V, 8 * N)
    for taper in (None, ss.unit_taper("tukey", N, alpha=0.25)):
        g = ss.matched_lag_window(Q, taper)
        w = np.ones(N) / np.sqrt(N) if taper is None else taper
        H = ss.kernel_smoothed(w, g, 8 * N)
        assert np.abs(H - H_mt).max() < 1e-3 * H_mt.max()
    # same kernel, different estimator
    g = ss.matched_lag_window(Q)
    t = np.arange(N); Qs = g[np.abs(t[:, None] - t[None, :])] / N
    assert 0.1 < np.linalg.norm(Qs - Q) / np.linalg.norm(Q) < 0.4


def test_smoothed_periodogram_has_least_variance_for_a_given_kernel():
    """Same kernel as the K-taper estimate, but constant along each diagonal: at least as many degrees of freedom."""
    N, NW = 128, 4; t = np.arange(N); e = np.exp(-2j * np.pi * 0.25 * t)
    for K in (2, 4, 7):
        V, _ = ss.dpss(N, NW, K); Q = V @ V.T / K
        Qs = ss.matched_lag_window(Q)[np.abs(t[:, None] - t[None, :])] / N
        assert np.sum(Qs ** 2) <= np.sum(Q ** 2) + 1e-12
        nu_mt = ss.dof_quadratic(e[:, None] * Q * e.conj()[None, :]); nu_s = ss.dof_quadratic(e[:, None] * Qs * e.conj()[None, :])
        assert abs(nu_mt - 2 * K) < 0.05 and nu_s > nu_mt


def test_multitaper_is_smoothing_of_the_complex_transform_before_squaring():
    rng = np.random.default_rng(0); N = 48; x = rng.standard_normal(N); V, _ = ss.dpss(N, 3, 5)
    S_time = ss.multitaper(x, V, 2 * N)[0]
    S_freq = ss.multitaper_from_fft(x, V, 2 * N)[0]
    assert np.abs(S_time - S_freq).max() < 1e-10 * S_time.max()


def test_sine_multitaper_from_one_fft():
    rng = np.random.default_rng(1); N, K = 200, 6; x = rng.standard_normal(N)
    S, f = ss.multitaper_sine_from_fft(x, K, oversample=2)
    S_ref = ss.multitaper(x, ss.sine_tapers(N, K), len(S))[0]
    assert np.abs(S - S_ref).max() < 1e-10 * S_ref.max()


def test_smoothed_periodogram_needs_a_toeplitz_matrix():
    """Averaging the diagonals of the multitaper matrix gives the kernel-matched smoothed periodogram, and no Toeplitz matrix is closer."""
    N, NW, K = 128, 4, 7; t = np.arange(N)
    V, _ = ss.dpss(N, NW, K); Q = V @ V.T / K
    T = ss.toeplitz_part(Q)
    Qs = ss.matched_lag_window(Q)[np.abs(t[:, None] - t[None, :])] / N
    assert np.abs(T - Qs * np.trace(T) / np.trace(Qs)).max() < 1e-12
    d = np.linalg.norm(Q - T)
    rng = np.random.default_rng(2)
    for _ in range(5):
        g = rng.standard_normal(N) * 1e-3
        assert np.linalg.norm(Q - (T + g[np.abs(t[:, None] - t[None, :])])) >= d
    A = ss.sinc_toeplitz(N, NW / N)                                   # the all-taper, eigenvalue-weighted matrix is Toeplitz already
    assert np.abs(A - ss.toeplitz_part(A)).max() < 1e-12
    assert 0.1 < d / np.linalg.norm(Q) < 0.3

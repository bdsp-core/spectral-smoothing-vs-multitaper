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


def test_multitaper_is_a_smoothed_periodogram_plus_a_cross_term():
    """S_mt = (periodogram smoothed with H_K) + cross term; for white noise the cross term has zero mean, is uncorrelated with the
    smoothed term, and carries the share d^2 = 1 - (K/N^2) sum_j H_K(j/N)^2 of the variance."""
    N, NW, K = 128, 4, 7; V, _ = ss.dpss(N, NW, K); j = np.arange(N)
    rng = np.random.default_rng(4); x = rng.standard_normal(N)
    sm, cr, _ = ss.multitaper_split(x, V)
    assert np.abs(sm + cr - ss.multitaper(x, V, N)[0]).max() < 1e-12
    HK = (np.abs(np.fft.fft(V, axis=0)) ** 2).mean(axis=1); I = np.abs(np.fft.fft(x)) ** 2 / N
    direct = np.array([(HK[(m - j) % N] * I).sum() / N for m in range(N)])       # the smoothed term, written as a plain sum
    assert np.abs(sm - direct).max() < 1e-12 and abs(HK.sum() / N - 1) < 1e-12
    d2 = 1 - K * (HK ** 2).sum() / N ** 2
    X = rng.standard_normal((6000, N)); m0 = N // 4
    parts = np.array([[a[m0], b[m0]] for a, b, _ in (ss.multitaper_split(r, V) for r in X)])
    tot = parts.sum(axis=1)
    assert abs(parts[:, 1].mean()) < 0.02                                           # zero mean
    assert abs(np.corrcoef(parts[:, 0], parts[:, 1])[0, 1]) < 0.05                  # uncorrelated
    assert abs(parts[:, 1].var() / tot.var() - d2) < 0.02                           # share of the variance
    assert abs(np.corrcoef(parts[:, 0], tot)[0, 1] - np.sqrt(1 - d2)) < 0.01


def test_parabola_is_the_stationary_point_of_the_mean_square_error():
    """Appendix D: minimizing (c^2/4) mu2^2 + (kappa/N) int G^2 over non-negative unit-area kernels gives the parabola of
    half-width b = (15 kappa / (N c^2))^(1/5). Check on a grid that no non-negative perturbation of it lowers the functional."""
    N, kappa, c = 400.0, 1.15, 30.0
    b = (15 * kappa / (N * c ** 2)) ** 0.2
    u = np.linspace(-2.5 * b, 2.5 * b, 4001); du = u[1] - u[0]
    J = lambda G: (c ** 2 / 4) * (np.sum(u ** 2 * G) * du) ** 2 + (kappa / N) * np.sum(G ** 2) * du
    par = np.where(np.abs(u) <= b, 0.75 / b * (1 - (u / b) ** 2), 0.0)
    J0 = J(par)
    rng = np.random.default_rng(5)
    shapes = [np.where(np.abs(u) <= w, 0.5 / w, 0.0) for w in (0.6 * b, 0.8 * b, b)] + \
             [np.exp(-0.5 * (u / s) ** 2) / (s * np.sqrt(2 * np.pi)) for s in (0.3 * b, 0.45 * b, 0.6 * b)] + \
             [np.where(np.abs(u) <= w, 0.75 / w * (1 - (u / w) ** 2), 0.0) for w in (0.8 * b, 0.9 * b, 1.1 * b, 1.25 * b)]
    for G in shapes:
        G = G / (G.sum() * du)
        assert J(G) > J0
        for eps in (0.05, 0.3):                                                     # convexity along the segment
            assert J((1 - eps) * par + eps * G) > J0 - 1e-15
    for _ in range(20):
        G = np.maximum(par + 0.05 * par.max() * np.convolve(rng.standard_normal(len(u)), np.ones(41) / 41, "same"), 0)
        assert J(G / (G.sum() * du)) > J0

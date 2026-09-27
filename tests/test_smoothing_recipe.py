"""The light taper and parabolic kernel of recipe (b'), and the kernel-matched lag window."""
import numpy as np
import pytest
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
    g = ss.matched_lag_window(Q)                                                 # no taper: r_w > 0 at every lag, exact
    assert np.abs(ss.kernel_smoothed(np.ones(N) / np.sqrt(N), g, 8 * N) - H_mt).max() < 1e-12 * H_mt.max()
    w = ss.unit_taper("tukey", N, alpha=0.25)                                    # zero end samples: the last lags are not matched
    with pytest.warns(UserWarning, match="not matched"):
        H = ss.kernel_smoothed(w, ss.matched_lag_window(Q, w), 8 * N)
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


def test_least_variance_is_a_white_noise_result():
    """For real white noise, var(x_f^* Q x_f) = sum_tau (1 + cos 4 pi f tau) e_tau, where e_tau is the sum of squares of Q along
    the diagonal at lag tau, so the untapered kernel-matched (Toeplitz) estimator has the least variance at every f, including
    0 and 1/2. For a coloured spectrum the ordering reverses where the spectrum is low. Most of its eigen-weights are negative."""
    N, NW, K = 256, 4, 7; t = np.arange(N); lag = np.abs(t[:, None] - t[None, :])
    V, _ = ss.dpss(N, NW, K); Q = V @ V.T / K
    Qs = ss.matched_lag_window(Q)[lag] / N

    def moments(M, f, Sig):
        e = np.exp(-2j * np.pi * f * t); R = np.real(e[:, None] * M * e.conj()[None, :])
        return np.trace(R @ Sig), 2 * np.trace(R @ Sig @ R @ Sig)

    tau = np.arange(-(N - 1), N); e_tau = np.array([np.sum(np.diagonal(Q, k) ** 2) for k in tau])
    for f in (0.0, 0.1, 0.25, 0.37, 0.5):
        v_mt = moments(Q, f, np.eye(N))[1]; v_s = moments(Qs, f, np.eye(N))[1]
        assert abs(v_mt - np.sum((1 + np.cos(4 * np.pi * f * tau)) * e_tau)) < 1e-10 * v_mt
        assert v_s < v_mt
    nfft = 1 << 15; acv = np.real(np.fft.ifft(ss.ar_psd(ss.AR4, np.arange(nfft) / nfft)))[:N]
    for f in np.arange(0.2, 0.501, 0.05):                                        # AR(4): same mean, > 10x the variance
        (m_mt, v_mt), (m_s, v_s) = moments(Q, f, acv[lag]), moments(Qs, f, acv[lag])
        assert abs(m_s / m_mt - 1) < 1e-9 and v_s > 10 * v_mt
    c = np.linalg.eigvalsh(Qs)
    assert (c < 0).sum() > N // 2 and -c[c < 0].sum() < 0.05 * c[c > 0].sum()


def test_matched_lag_window_exactness_conditions():
    """Exact when r_w != 0 wherever q != 0, including tapers whose autocorrelation is negative at some lags; the symmetric Hann and
    cosine tapers (zero end samples) miss the last lags by less than 3e-4 of the peak at N = 256; the kernel keeps the area tr Q."""
    N, NW, K = 256, 4, 7; nfft = 8 * N; t = np.arange(N)
    V, _ = ss.dpss(N, NW, K); Q = V @ V.T / K
    H_mt = ss.kernel_multitaper(V, nfft)
    w2 = ss.dpss(N, 2, 2)[0][:, 1]                                               # second Slepian taper: r_w changes sign
    assert (np.correlate(w2, w2, "full")[N - 1:] < 0).any()
    assert np.abs(ss.kernel_smoothed(w2, ss.matched_lag_window(Q, w2), nfft) - H_mt).max() < 1e-10 * H_mt.max()
    for w in (ss.unit_taper("tukey", N, alpha=0.25), ss.unit_taper("hann", N)):
        with pytest.warns(UserWarning, match="not matched"):
            err = np.abs(ss.kernel_smoothed(w, ss.matched_lag_window(Q, w), nfft) - H_mt).max() / H_mt.max()
        assert 0 < err < 3e-4
    w = 7.0 * ss.unit_taper("gaussian", N, std=0.3)                               # no zero end samples, not unit norm
    g = ss.matched_lag_window(Q, w)
    assert abs(np.trace(w[:, None] * g[np.abs(t[:, None] - t[None, :])] * w[None, :]) - np.trace(Q)) < 1e-12


def test_nu_of_recipe_b_prime_splits_between_taper_and_kernel():
    """nu of recipe (b') at W = 4/N: the 25% cosine taper alone (with the box) already exceeds multitaper K = 7, and the parabola of
    the same half-power width adds the rest through its wider base. Values quoted in Sections I and III."""
    for N in (256, 1024):
        W = 4 / N; hb = ss.box_lag_window(N, W); hp = ss.parabolic_lag_window(N, np.sqrt(2) * W)
        tuk = ss.unit_taper("tukey", N, alpha=0.25)
        nu = lambda **p: ss.dof_quadratic(ss.quadratic_matrix(N, 0.25, **p))
        nu_mt = nu(method="multitaper", tapers=ss.dpss(N, 4)[0])
        nu_hann_box = nu(method="lagwindow", h=hb, taper=ss.unit_taper("hann", N))
        nu_tuk_box = nu(method="lagwindow", h=hb, taper=tuk)
        nu_tuk_par = nu(method="lagwindow", h=hp, taper=tuk)
        assert (round(nu_hann_box, 1), round(nu_mt, 1), round(nu_tuk_box, 1), round(nu_tuk_par, 1)) == (8.9, 14.0, 14.8, 16.7)


def test_recipe_b_prime_skirts_depend_on_the_time_bandwidth_product():
    """Recipe (b'): the peak of the kernel beyond 3W is -25, -31 and -38 dB for 2NW = 4, 6 and 8, independent of N, and the
    Hann-then-box kernel is at least 25 dB lower at each setting (Section IV, recipe (b'))."""
    for N in (256, 1024):
        nfft = 64 * N; f = np.abs(ss.signed_freq(nfft))
        for NW, level in ((2, -25), (3, -31), (4, -38)):
            W = NW / N
            Hb = ss.kernel_smoothed(ss.unit_taper("tukey", N, alpha=0.25), ss.parabolic_lag_window(N, np.sqrt(2) * W), nfft)
            Hh = ss.kernel_smoothed(ss.unit_taper("hann", N), ss.box_lag_window(N, W), nfft)
            db = lambda H: 10 * np.log10(H[f > 3 * W].max() / H.max())
            assert level - 1 < db(Hb) < level and db(Hh) < db(Hb) - 25


def test_parabola_versus_box_at_equal_half_power_width():
    """At the same half-power width w the parabola (half-width w / sqrt 2) has second moment w^2/10 against w^2/12 for the box
    (20% more leading bias) and int G^2 = 3 sqrt(2) / (5 w) against 1 / w (15% less variance), so it is not optimal at a fixed width.
    Its asymptotic optimality holds with each kernel at its own best width: the classical efficiency constant
    (mu2^(1/2) int G^2)^(4/5) is smallest for the parabola, and the box and Gaussian RMS errors are 3% and 2% larger."""
    w = 1.0; u = np.linspace(-3, 3, 600001); du = u[1] - u[0]
    b = w / np.sqrt(2); par = np.where(np.abs(u) <= b, 0.75 / b * (1 - (u / b) ** 2), 0.0)
    box = np.where(np.abs(u) <= w / 2, 1 / w, 0.0)
    sg = w / (2 * np.sqrt(2 * np.log(2))); gau = np.exp(-0.5 * (u / sg) ** 2) / (sg * np.sqrt(2 * np.pi))
    mom = lambda G: (np.sum(u ** 2 * G) * du, np.sum(G ** 2) * du)
    (m_p, r_p), (m_b, r_b), (m_g, r_g) = mom(par), mom(box), mom(gau)
    assert abs(m_p / m_b - 1.2) < 1e-3 and abs(r_p / r_b - 3 * np.sqrt(2) / 5) < 1e-3
    eff = lambda m, r: (np.sqrt(m) * r) ** 0.8                                  # minimal AMSE is proportional to this
    rms = lambda m, r: np.sqrt(eff(m, r) / eff(m_p, r_p))
    assert abs(rms(m_b, r_b) - 1.0297) < 1e-3 and abs(rms(m_g, r_g) - 1.0202) < 1e-3

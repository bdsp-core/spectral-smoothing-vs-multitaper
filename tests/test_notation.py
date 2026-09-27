"""The paper's formulas in the notation of Babadi and Brown (2014), with the sampling interval Delta kept explicit.

x_k = x(k Delta), k = 0..N-1; X(f) := Delta sum_k x_k e^{-i2 pi k f Delta}; periodogram S^p(f) := |X(f)|^2 / (N Delta);
eigenspectra S^(i)(f) := Delta |sum_k h^(i)_k x_k e^{-i2 pi k f Delta}|^2 with unit-energy tapers; resolution R = 2 alpha / (N Delta);
(Phi_R)_{k,l} := sin(pi R Delta (k - l)) / (pi (k - l)). Each test evaluates both sides from these definitions at Delta = 1/200 s,
so that a missing or extra factor of Delta would fail."""
import numpy as np
from scipy.integrate import quad
import specsmooth as ss

DELTA = 1 / 200.0


def _setup(N=64, alpha=3, seed=1):
    x = np.random.default_rng(seed).standard_normal(N)
    R = 2 * alpha / (N * DELTA)
    return N, alpha, R, x, np.arange(N)


def _X(x, f, shift=0):
    k = np.arange(len(x))
    return DELTA * np.sum(x * np.exp(-2j * np.pi * (k + shift) * f * DELTA))


def _Sp(x, f):
    return abs(_X(x, f)) ** 2 / (len(x) * DELTA)


def _Si(h, x, f):
    return DELTA * abs(np.sum(h * x * np.exp(-2j * np.pi * np.arange(len(x)) * f * DELTA))) ** 2


def _Phi(N, R):
    d = np.subtract.outer(np.arange(N), np.arange(N)).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        P = np.sin(np.pi * R * DELTA * d) / (np.pi * d)
    P[d == 0] = R * DELTA
    return P


def test_phi_R_eigenvalues_sum_to_2alpha_and_eigenvectors_are_slepians():
    N, alpha, R, _, _ = _setup()
    lam, V = np.linalg.eigh(_Phi(N, R)); lam, V = lam[::-1], V[:, ::-1]
    V0, lam0 = ss.dpss_all(N, alpha / N)
    assert abs(lam.sum() - 2 * alpha) < 1e-10 and np.abs(lam - lam0).max() < 1e-10
    assert np.abs(np.abs(np.sum(V[:, :6] * V0[:, :6], axis=0)) - 1).max() < 1e-8


def test_fejer_kernel_is_the_rectangular_taper_kernel_with_unit_area():
    N = 64
    D2 = lambda f: np.sin(N * np.pi * f * DELTA) ** 2 / (N ** 2 * np.sin(np.pi * f * DELTA) ** 2)
    H2 = lambda f: abs(DELTA * np.sum(np.exp(-2j * np.pi * np.arange(N) * f * DELTA)) / np.sqrt(N)) ** 2
    for f in (0.37, 3.1, 41.0):
        assert abs(N * DELTA * D2(f) - H2(f) / DELTA) < 1e-12 * N * DELTA
    area = quad(lambda f: N * DELTA * D2(f), -0.5 / DELTA, 0.5 / DELTA, limit=400, points=[0.0])[0]
    assert abs(area - 1) < 1e-8


def test_thomson_identity_in_physical_units():
    """(1 / 2 alpha) sum_{i=1}^N lambda_i S^(i)(f) = (1 / R) int_{f - R/2}^{f + R/2} S^p(f') df'."""
    N, alpha, R, x, _ = _setup()
    lam, V = np.linalg.eigh(_Phi(N, R))
    for f in (0.0, 7.3, 55.0):
        lhs = sum(lam[i] * _Si(V[:, i], x, f) for i in range(N)) / (2 * alpha)
        rhs = quad(lambda u: _Sp(x, u), f - R / 2, f + R / 2, limit=400, epsabs=0, epsrel=1e-12)[0] / R
        assert abs(lhs - rhs) < 1e-9 * rhs


def test_multitaper_before_squaring_and_split_into_smoothed_periodogram_plus_cross_term():
    """S^mt(f) = (1 / L Delta) sum_i |int H^(i)(f - f') X(f') df'|^2, the integral equals (1 / N Delta) sum_j H^(i)(f - f_j) X_j at
    f_j = j / (N Delta), and S^mt = (1/N Delta) sum_j K_L(f - f_j) S^p_j + (1/(N Delta)^2) sum_{j != l} B_L(f - f_j, f - f_l) X_j X_l^*."""
    N, alpha, R, x, k = _setup()
    L = 2 * alpha - 1; V, _ = ss.dpss(N, alpha, L)
    fj = np.arange(N) / (N * DELTA); Xj = np.array([_X(x, g) for g in fj]); Spj = np.abs(Xj) ** 2 / (N * DELTA)
    H = lambda i, f: DELTA * np.sum(V[:, i] * np.exp(-2j * np.pi * k * f * DELTA))
    for f in (3.3, 20.0, 71.7):
        S_mt = np.mean([_Si(V[:, i], x, f) for i in range(L)])
        conv = [sum(H(i, f - g) * Xg for g, Xg in zip(fj, Xj)) / (N * DELTA) for i in range(L)]
        re = quad(lambda u: (H(0, f - u) * _X(x, u)).real, -0.5 / DELTA, 0.5 / DELTA, limit=800)[0]
        im = quad(lambda u: (H(0, f - u) * _X(x, u)).imag, -0.5 / DELTA, 0.5 / DELTA, limit=800)[0]
        assert abs(conv[0] - (re + 1j * im)) < 1e-7 * abs(conv[0])
        assert abs(np.sum(np.abs(conv) ** 2) / (L * DELTA) - S_mt) < 1e-10 * S_mt
        Hf = np.array([[H(i, f - g) for g in fj] for i in range(L)])                   # L x N
        B = (Hf.T @ Hf.conj()) / (L * DELTA)                                          # B_L(f - f_j, f - f_l)
        smooth = np.real(np.diag(B)) @ Spj / (N * DELTA)
        cross = (np.sum(B * np.outer(Xj, Xj.conj())) - np.sum(np.diag(B) * np.abs(Xj) ** 2)) / (N * DELTA) ** 2
        assert abs(smooth + cross.real - S_mt) < 1e-10 * S_mt and abs(cross.imag) < 1e-10 * S_mt


def test_cross_term_share_does_not_depend_on_delta():
    """d^2 = 1 - (L / (N Delta)^2) sum_j K_L(f_j)^2 with K_L = (1 / L Delta) sum_i |H^(i)|^2: 0.08 for alpha = 4, L = 7."""
    N, alpha = 256, 4; L = 2 * alpha - 1; V, _ = ss.dpss(N, alpha, L)
    for delta in (1.0, DELTA):
        Hj = delta * np.fft.fft(V, axis=0)                                           # H^(i)(f_j), f_j = j / (N delta)
        K = np.sum(np.abs(Hj) ** 2, axis=1) / (L * delta)
        d2 = 1 - L / (N * delta) ** 2 * np.sum(K ** 2)
        assert round(d2, 2) == 0.08


def test_sine_multitaper_from_the_transform_with_delta():
    """S(f) = (1 / (2 L (N + 1) Delta)) sum_{i=1}^L |X~(f - d_i) - X~(f + d_i)|^2, d_i = i / ((2N + 2) Delta), X~ with the time
    origin one sample before the record."""
    N, _, _, x, _ = _setup(); L = 5; St = ss.sine_tapers(N, L)
    for f in (1.0, 33.3):
        direct = np.mean([_Si(St[:, i], x, f) for i in range(L)])
        d = np.arange(1, L + 1) / ((2 * N + 2) * DELTA)
        via = sum(abs(_X(x, f - di, 1) - _X(x, f + di, 1)) ** 2 for di in d) / (2 * L * (N + 1) * DELTA)
        assert abs(via - direct) < 1e-10 * direct


def test_kernel_of_a_quadratic_form_and_the_parabola_lag_window():
    """E{Delta x_f^* Q x_f} = int K(f - f') S(f') df' with K(f) = Delta sum_tau q_tau e^{-i2 pi tau f Delta}, S(f) = Delta sum_k s_k
    e^{-i2 pi k f Delta}; and the parabola (3 / 4b)(1 - f^2 / b^2) has lag window 3 (sin a - a cos a) / a^3, a = 2 pi b tau Delta."""
    N, alpha, R, _, k = _setup()
    h = ss.unit_taper("tukey", N, alpha=0.25); b = R / np.sqrt(2)
    tau = np.arange(N); a = 2 * np.pi * b * tau * DELTA
    g = np.ones(N); g[1:] = 3 * (np.sin(a[1:]) - a[1:] * np.cos(a[1:])) / a[1:] ** 3
    for t in (1, 5, 17):
        num = quad(lambda f: 0.75 / b * (1 - (f / b) ** 2) * np.cos(2 * np.pi * f * t * DELTA), -b, b)[0]
        assert abs(num - g[t]) < 1e-10
    Q = h[:, None] * g[np.abs(k[:, None] - k[None, :])] * h[None, :]
    phi = 0.8; s = phi ** np.arange(N) / (1 - phi ** 2)                             # AR(1) autocovariance, unit innovations
    S = lambda f: DELTA / abs(1 - phi * np.exp(-2j * np.pi * f * DELTA)) ** 2
    q = np.array([np.trace(Q, offset=t) for t in range(N)])
    K = lambda f: DELTA * (q[0] + 2 * np.sum(q[1:] * np.cos(2 * np.pi * np.arange(1, N) * f * DELTA)))
    Sig = s[np.abs(k[:, None] - k[None, :])]
    for f in (0.0, 12.5, 60.0):
        e = np.exp(-2j * np.pi * k * f * DELTA)
        mean = DELTA * np.real(e.conj() @ (Q * Sig) @ e)
        conv = quad(lambda u: K(f - u) * S(u), -0.5 / DELTA, 0.5 / DELTA, limit=800, points=[f])[0]
        assert abs(mean - conv) < 1e-7 * mean

"""Nonparametric spectral estimators on the grid f_j = j/nfft (cycles/sample).

Conventions: tapers have unit norm, so every estimator here has E[S(f)] ~ S(f) (the true PSD)
for a process with unit-variance white noise giving S = 1.
"""
import warnings

import numpy as np


def _fgrid(nfft):
    return np.arange(nfft) / nfft


def periodogram(x, nfft=None, taper=None):
    """Direct estimate |sum_t w_t x_t e^{-i 2 pi f t}|^2 with a unit-norm taper (default rectangular)."""
    x = np.asarray(x, float)
    N = len(x)
    nfft = nfft or 4 * N
    w = np.ones(N) / np.sqrt(N) if taper is None else np.asarray(taper, float)
    return np.abs(np.fft.fft(w * x, nfft)) ** 2, _fgrid(nfft)


def multitaper(x, tapers, nfft=None, weights=None):
    """(Weighted) average of tapered periodograms. tapers: (N, K) unit-norm columns."""
    x = np.asarray(x, float)
    N = len(x)
    nfft = nfft or 4 * N
    Sk = np.abs(np.fft.fft(tapers * x[:, None], nfft, axis=0)) ** 2
    if weights is None:
        return Sk.mean(axis=1), _fgrid(nfft)
    w = np.asarray(weights, float)
    return Sk @ w / w.sum(), _fgrid(nfft)


def multitaper_adaptive(x, tapers, lam, nfft=None, max_iter=200, tol=1e-5):
    """Thomson's adaptively weighted multitaper estimate (Thomson 1982, Sec. V; Percival & Walden 1993, Sec. 7.4).

    S(f) = sum_k b_k^2 lam_k S_k(f) / sum_k b_k^2 lam_k,   b_k(f) = S(f) / (lam_k S(f) + (1 - lam_k) sigma^2),
    iterated from the average of the first two eigenspectra. Where the spectrum is far below the process variance
    sigma^2 the leaky high-order tapers are down-weighted. Returns (S, grid, nu) with
    nu(f) = 2 (sum_k b_k^2 lam_k)^2 / sum_k b_k^4 lam_k^2 the equivalent degrees of freedom at each frequency.
    """
    x = np.asarray(x, float)
    N = len(x)
    nfft = nfft or 4 * N
    lam = np.asarray(lam, float)[None, :]
    Sk = np.abs(np.fft.fft(tapers * x[:, None], nfft, axis=0)) ** 2
    sig2 = np.mean(x ** 2)
    S = Sk[:, :2].mean(axis=1)
    for _ in range(max_iter):
        b = S[:, None] / (lam * S[:, None] + (1 - lam) * sig2)
        w = b ** 2 * lam
        Snew = (w * Sk).sum(axis=1) / w.sum(axis=1)
        done = np.max(np.abs(Snew - S) / S) < tol
        S = Snew
        if done:
            break
    b = S[:, None] / (lam * S[:, None] + (1 - lam) * sig2)
    w = b ** 2 * lam
    nu = 2 * w.sum(axis=1) ** 2 / (w ** 2).sum(axis=1)
    return S, _fgrid(nfft), nu


def hybrid_estimate(x, tapers, h, nfft=None):
    """'A few tapers, then smooth' (Riedel, Sidorenko & Thomson 1994): the average over tapers of the tapered periodogram
    smoothed with the lag window h."""
    x = np.asarray(x, float)
    nfft = nfft or 4 * len(x)
    return np.mean([lag_window_estimate(x, h, nfft, taper=tapers[:, k])[0] for k in range(tapers.shape[1])], axis=0), _fgrid(nfft)


def acs(y):
    """Unnormalized sample autocovariance r_tau = sum_t y_t y_{t+tau}, tau = -(N-1)..(N-1) (lag 0 at index N-1)."""
    y = np.asarray(y, float)
    N = len(y)
    r = np.fft.ifft(np.abs(np.fft.fft(y, 2 * N)) ** 2).real
    return np.concatenate([r[N + 1:2 * N], r[:N]])


def _place_lags(g, N, nfft):
    """Put a symmetric lag sequence g (lags -(N-1)..(N-1)) into an nfft-periodic array with lag 0 at index 0."""
    if nfft < 2 * N - 1:
        raise ValueError("nfft must be >= 2N-1 for an exact lag-domain evaluation")
    q = np.zeros(nfft)
    q[:N] = g[N - 1:]
    q[nfft - N + 1:] = g[:N - 1]
    return q


def lag_window_estimate(x, h, nfft=None, taper=None):
    """Quadratic estimate S(f) = sum_tau h_tau r_tau e^{-i 2 pi f tau}, r = ACS of the tapered data.

    This equals the tapered periodogram convolved (in continuous frequency, exactly) with the
    kernel H(f) = FT{h}. h is given for tau = 0..N-1 and is assumed symmetric; h[0] = 1 makes H unit-area.
    """
    x = np.asarray(x, float)
    N = len(x)
    nfft = nfft or 4 * N
    w = np.ones(N) / np.sqrt(N) if taper is None else np.asarray(taper, float)
    h = np.asarray(h, float)[:N]
    hfull = np.concatenate([h[:0:-1], h])
    q = _place_lags(acs(w * x) * hfull, N, nfft)
    return np.fft.fft(q).real, _fgrid(nfft)


def box_lag_window(N, W):
    """Lag window of the unit-area box kernel of half-width W: h_tau = sinc(2 W tau). Smoothing with it is (1/2W) int_{f-W}^{f+W}."""
    return np.sinc(2 * W * np.arange(N))


def gaussian_lag_window(N, sigma_f):
    """Lag window of a unit-area Gaussian kernel with standard deviation sigma_f (cycles/sample)."""
    return np.exp(-2 * (np.pi * sigma_f * np.arange(N)) ** 2)


def parabolic_lag_window(N, W):
    """Lag window of the unit-area parabolic kernel (3 / 4W)(1 - f^2 / W^2) on |f| <= W: h = 3 (sin a - a cos a) / a^3, a = 2 pi W tau.
    Its half-power width is sqrt(2) W, so W = sqrt(2) W0 matches the half-power width 2 W0 of a box of half-width W0."""
    a = 2 * np.pi * W * np.arange(N, dtype=float)
    h = np.ones(N)
    nz = a > 1e-4
    h[nz] = 3 * (np.sin(a[nz]) - a[nz] * np.cos(a[nz])) / a[nz] ** 3
    return h


def matched_lag_window(Q, taper=None):
    """Lag window g that gives 'taper, then smooth' the same kernel as the quadratic estimator with real symmetric matrix Q.

    The kernel of Q has lag sequence q_tau = sum_t Q[t, t+tau]; the kernel of the smoothed tapered periodogram has lag sequence
    g_tau r_w(tau), with r_w the autocorrelation of the taper. So g_tau = q_tau / r_w(tau), which also keeps the kernel's area
    (g_0 = tr Q / ||w||^2). The match is exact when r_w(tau) != 0 at every lag where q_tau != 0, as it is without a taper. Tapers
    that vanish at their end samples (scipy's symmetric Hann and Tukey windows) have r_w = 0 at the last lags; those lags are left
    unmatched, with a warning. Where r_w is small g is large, so g must be applied in the lag domain, not as a deconvolved kernel
    on a frequency grid. With an exact match the two estimators have the same expected value for every spectrum. They are not
    the same estimator: their matrices differ, and so do their variances."""
    Q = np.real(np.asarray(Q)); N = Q.shape[0]
    w = np.ones(N) / np.sqrt(N) if taper is None else np.asarray(taper, float)
    q = np.array([np.trace(Q, offset=k) for k in range(N)])
    rw = np.array([(w[:N - k] * w[k:]).sum() for k in range(N)])
    ok = np.abs(rw) > 1e-9 * rw[0]                 # |r_w|: an autocorrelation may be negative at some lags
    if np.any(np.abs(q[~ok]) > 1e-12 * np.abs(q).max()):
        warnings.warn("the taper's autocorrelation vanishes at lags where Q has nonzero lag sums; those lags are not matched",
                      stacklevel=2)
    return np.where(ok, q / np.where(ok, rw, 1.0), 0.0)


def smooth_on_grid(S, H):
    """What a practitioner does: circularly convolve a spectrum with a kernel on the same FFT grid (kernel normalized to unit sum)."""
    H = np.asarray(H, float)
    return np.fft.ifft(np.fft.fft(S) * np.fft.fft(H)).real / H.sum()


def welch(x, seg_len, overlap=0.5, nfft=None, taper=None):
    """Welch / WOSA: average of tapered periodograms of overlapping segments (taper unit-norm, length seg_len)."""
    x = np.asarray(x, float)
    N = len(x)
    nfft = nfft or 4 * seg_len
    step = max(1, int(round(seg_len * (1 - overlap))))
    w = np.ones(seg_len) / np.sqrt(seg_len) if taper is None else np.asarray(taper, float)
    segs = [np.abs(np.fft.fft(w * x[s:s + seg_len], nfft)) ** 2 for s in range(0, N - seg_len + 1, step)]
    return np.mean(segs, axis=0), _fgrid(nfft)


def _placements(N, L, step, overhang):
    """Offsets s at which a length-L window is placed on a length-N record; with overhang the window may run off either end."""
    return range(-L + 1, N, step) if overhang else range(0, N - L + 1, step)


def _placed(w, N, s):
    """The length-N vector holding window w at offset s (zero elsewhere; parts of w outside the record are dropped)."""
    L = len(w)
    u = np.zeros(N)
    lo, hi = max(0, s), min(N, s + L)
    u[lo:hi] = w[lo - s:hi - s]
    return u


def welch_sliding(x, w, nfft=None, step=1, overhang=True):
    """Averaged periodograms of a window slid across the record: sum_s |FT(u_s x)|^2 / sum_s ||u_s||^2.

    u_s is the window w placed at offset s. With step=1 and overhang=True (the window may run off the zero-padded ends)
    the quadratic form is exactly Toeplitz with lags r_w(tau) = sum_u w[u] w[u+tau], so this equals the raw periodogram
    smoothed with the kernel |W(f)|^2 (Welch 1967 in the limit; Nuttall & Carter 1982). With w = sinc_window(L, W),
    L -> inf, it is the box-smoothed periodogram = Thomson's eigenvalue-weighted all-taper estimate (his eq. 8.3).
    """
    x = np.asarray(x, float)
    N = len(x)
    nfft = nfft or 4 * N
    S = np.zeros(nfft)
    norm = 0.0
    for s in _placements(N, len(w), step, overhang):
        u = _placed(w, N, s)
        S += np.abs(np.fft.fft(u * x, nfft)) ** 2
        norm += u @ u
    return S / norm, _fgrid(nfft)


def quadratic_matrix(N, f0, method, **p):
    """Hermitian matrix Q with estimate(f0) = x^H Q x, for variance/dof bookkeeping (see metrics.dof_quadratic).

    method: 'multitaper' (tapers=(N,K), weights=None), 'lagwindow' (h=lag window, taper=None), 'hybrid' (tapers, h),
            'welch' (seg_len, overlap, taper=None; or step=..., overhang=True for the sliding form of welch_sliding).
    """
    t = np.arange(N)
    e = np.exp(-2j * np.pi * f0 * t)
    if method == "multitaper":
        V = np.asarray(p["tapers"], float)
        wts = p.get("weights")
        wts = np.ones(V.shape[1]) if wts is None else np.asarray(wts, float)
        U = V * e[:, None]
        return (U * wts) @ U.conj().T / wts.sum()
    if method == "lagwindow":
        h = np.asarray(p["h"], float)[:N]
        tau = t[:, None] - t[None, :]
        Hm = h[np.abs(tau)] * np.exp(-2j * np.pi * f0 * tau)
        w = np.ones(N) / np.sqrt(N) if p.get("taper") is None else np.asarray(p["taper"], float)
        return (w[:, None] * Hm * w[None, :])
    if method == "hybrid":
        V = np.asarray(p["tapers"], float)
        h = np.asarray(p["h"], float)[:N]
        tau = t[:, None] - t[None, :]
        Hm = h[np.abs(tau)] * np.exp(-2j * np.pi * f0 * tau)
        return sum(V[:, k][:, None] * Hm * V[:, k][None, :] for k in range(V.shape[1])) / V.shape[1]
    if method == "welch":
        w = np.ones(int(p["seg_len"])) / np.sqrt(int(p["seg_len"])) if p.get("taper") is None else np.asarray(p["taper"], float)
        L = len(w)
        step = p["step"] if "step" in p else max(1, int(round(L * (1 - p.get("overlap", 0.5)))))
        Q = np.zeros((N, N), complex)
        norm = 0.0
        for s in _placements(N, L, step, p.get("overhang", False)):
            u = _placed(w, N, s) * e
            Q += np.outer(u, u.conj())
            norm += (u.conj() @ u).real
        return Q / norm
    raise ValueError(method)

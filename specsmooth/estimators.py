"""Nonparametric spectral estimators on the grid f_j = j/nfft (cycles/sample).

Conventions: tapers have unit norm, so every estimator here has E[S(f)] ~ S(f) (the true PSD)
for a process with unit-variance white noise giving S = 1.
"""
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


def quadratic_matrix(N, f0, method, **p):
    """Hermitian matrix Q with estimate(f0) = x^H Q x, for variance/dof bookkeeping (see metrics.dof_quadratic).

    method: 'multitaper' (tapers=(N,K), weights=None), 'lagwindow' (h=lag window, taper=None),
            'welch' (seg_len, overlap, taper=None).
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
    if method == "welch":
        L = int(p["seg_len"])
        step = max(1, int(round(L * (1 - p.get("overlap", 0.5)))))
        w = np.ones(L) / np.sqrt(L) if p.get("taper") is None else np.asarray(p["taper"], float)
        starts = list(range(0, N - L + 1, step))
        Q = np.zeros((N, N), complex)
        for s in starts:
            u = np.zeros(N, complex)
            u[s:s + L] = w * e[s:s + L]
            Q += np.outer(u, u.conj())
        return Q / len(starts)
    raise ValueError(method)

"""Tapers: Slepian (DPSS) sequences, the sinc Toeplitz matrix they diagonalize, and classical single tapers."""
import numpy as np
from scipy.linalg import eigh
from scipy.signal import windows


def sinc_toeplitz(N, W):
    """A[t,t'] = sin(2*pi*W*(t-t')) / (pi*(t-t')), with 2W on the diagonal.

    Its eigenvectors are the DPSS and its eigenvalues their spectral concentrations.
    Its lag sequence is the inverse Fourier transform of a box of half-width W, which is
    what makes the eigenvalue-weighted multitaper estimate a box-smoothed periodogram.
    """
    t = np.arange(N, dtype=float)
    tau = t[:, None] - t[None, :]
    A = np.full((N, N), 2.0 * W)
    nz = tau != 0
    A[nz] = np.sin(2 * np.pi * W * tau[nz]) / (np.pi * tau[nz])
    return A


def dpss_all(N, W):
    """All N Slepian sequences (unit-norm columns) and eigenvalues, sorted by decreasing concentration."""
    lam, V = eigh(sinc_toeplitz(N, W))
    idx = np.argsort(lam)[::-1]
    return V[:, idx], lam[idx]


def dpss(N, NW, K=None):
    """The usual K = 2NW-1 Slepian tapers (columns, unit norm) and their concentrations, via scipy."""
    K = int(round(2 * NW - 1)) if K is None else int(K)
    v, lam = windows.dpss(N, NW, Kmax=K, return_ratios=True)
    return np.asarray(v).T, np.asarray(lam)


def unit_taper(name, N, **kw):
    """A unit-norm single taper: 'rect', 'hann', 'hamming', 'bohman' (= Papoulis minimum-bias), 'gaussian' (std= fraction of N),
    'tukey' (alpha= fraction of the record that is cosine-tapered, half at each end; 0 is rectangular, 1 is Hann)."""
    if name in (None, "rect", "boxcar"):
        w = np.ones(N)
    elif name == "hann":
        w = windows.hann(N)
    elif name == "hamming":
        w = windows.hamming(N)
    elif name in ("bohman", "papoulis"):
        w = windows.bohman(N)
    elif name == "tukey":
        w = windows.tukey(N, kw.get("alpha", 0.25))
    elif name == "gaussian":
        w = windows.gaussian(N, std=kw.get("std", 0.2) * N)
    else:
        raise ValueError(f"unknown taper {name!r}")
    return w / np.linalg.norm(w)


def sine_tapers(N, K):
    """Riedel-Sidorenko minimum-bias (sinusoidal) tapers, unit norm, as columns."""
    t = np.arange(1, N + 1)
    return np.stack([np.sqrt(2 / (N + 1)) * np.sin(np.pi * k * t / (N + 1)) for k in range(1, K + 1)], axis=1)


def sinc_window(L, W):
    """The centered sinc window w[t] = 2W sinc(2W t), t = -(L-1)/2 .. (L-1)/2: its spectral window |W(f)|^2 tends to the
    box of half-width W as L grows, and its autocorrelation is the box's lag sequence 2W sinc(2W tau) (exactly as L -> inf).

    Sliding it one sample at a time over a zero-padded record and summing the squared transforms gives the box-smoothed
    periodogram, i.e. Thomson's eigenvalue-weighted multitaper estimate with all N tapers (see estimators.welch_sliding).
    """
    t = np.arange(L) - (L - 1) / 2
    return 2 * W * np.sinc(2 * W * t)

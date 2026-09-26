"""Test signals with known spectra."""
import numpy as np
from scipy.signal import lfilter

# Percival & Walden's AR(4): two sharp peaks with ~70 dB dynamic range; a classic leakage stress test.
AR4 = np.array([1.0, -2.7607, 3.8106, -2.6535, 0.9238])


def ar_process(a, N, rng, burn=2000, sigma=1.0):
    e = sigma * rng.standard_normal(N + burn)
    return lfilter([1.0], a, e)[burn:]


def ar_psd(a, f, sigma2=1.0):
    """PSD of an AR process at frequencies f (cycles/sample), per-unit-frequency normalization."""
    z = np.exp(-2j * np.pi * np.outer(f, np.arange(len(a))))
    return sigma2 / np.abs(z @ a) ** 2


def signal_from_psd(S, N, rng, nfft=None):
    """Random-phase synthesis of a real Gaussian signal with (approximately) the PSD S given on an nfft grid."""
    S = np.asarray(S, float)
    nfft = len(S)
    Z = np.sqrt(S) * (rng.standard_normal(nfft) + 1j * rng.standard_normal(nfft)) / np.sqrt(2)
    x = np.fft.ifft(Z).real * np.sqrt(2 * nfft)
    return x[:N]


def two_tones(N, f1, f2, a1=1.0, a2=1.0, noise=0.0, rng=None):
    t = np.arange(N)
    x = a1 * np.cos(2 * np.pi * f1 * t) + a2 * np.cos(2 * np.pi * f2 * t + 1.0)
    if noise:
        x = x + noise * rng.standard_normal(N)
    return x

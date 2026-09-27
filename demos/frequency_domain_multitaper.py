"""The multitaper recipe in the frequency domain, and its split into a smoothed periodogram and a cross term.

One FFT of the untapered record gives X_j. Taper k acts as a filter on the complex FFT, y_k(f) = (1/N) sum_j V_k(f - j/N) X_j,
and S(f) = (1/K) sum_k |y_k(f)|^2. Expanding the square,
    S(f) = (1/N) sum_j H_K(f - j/N) I_j  +  (1/N^2) sum_{j != l} B_K(f - j/N, f - l/N) X_j conj(X_l),
with I_j = |X_j|^2 / N, H_K = (1/K) sum_k |V_k|^2 and B_K(a, b) = (1/K) sum_k V_k(a) conj(V_k(b)).
For white noise the cross term has zero mean, is uncorrelated with the first term, and carries the share
d^2 = 1 - (K/N^2) sum_j H_K(j/N)^2 of the variance. This script checks the identity and these formulas, and shows the
cross term removing leakage on coloured spectra.
Run: python demos/frequency_domain_multitaper.py
"""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "demos"))
import numpy as np, specsmooth as ss
from scipy.linalg import toeplitz, cholesky
import tuned_comparison as tc
DB = 10 / np.log(10)
rng = np.random.default_rng(0)

def check(N, NW, K, M=4000):
    V = ss.dpss(N, NW, K)[0]; t = np.arange(N)
    Vh = np.fft.fft(V, axis=0)                                   # V_k at the Fourier frequencies j/N
    HK = (np.abs(Vh) ** 2).mean(axis=1)                          # rounded-corner kernel H_K(j/N) = (1/K) sum_k |V_k(j/N)|^2 ; sums to N
    # --- identity on one record, at the Fourier frequencies f_m = m/N
    x = rng.standard_normal(N); X = np.fft.fft(x); I = np.abs(X) ** 2 / N
    S_time = (np.abs(np.fft.fft(V * x[:, None], axis=0)) ** 2).mean(axis=1)                      # taper, FFT, square, average
    j = np.arange(N)
    Y = np.array([[(Vh[(m - j) % N, k] * X).sum() / N for k in range(K)] for m in range(N)])      # one FFT, then filter it K ways
    S_freq = (np.abs(Y) ** 2).mean(axis=1)
    T1 = np.array([(HK[(m - j) % N] * I).sum() / N for m in range(N)])                            # periodogram smoothed with H_K
    print(f"N={N} NW={NW} K={K}: max |time-domain - frequency-domain multitaper| = {np.abs(S_time - S_freq).max():.1e}; "
          f"weights of the smoothing kernel sum to {HK.sum() / N:.6f}; bins with more than 1% of the peak weight: {(HK > 0.01 * HK.max()).sum()}")
    # --- the cross term: mean, share of variance, correlation (white noise), and the derived formulas
    d2 = 1 - K * (HK ** 2).sum() / N ** 2
    Xs = rng.standard_normal((M, N)); m0 = N // 4
    S_mt = (np.abs(np.fft.fft(V[None, :, :] * Xs[:, :, None], axis=1)[:, m0, :]) ** 2).mean(axis=1)
    Is = np.abs(np.fft.fft(Xs, axis=1)) ** 2 / N
    S_sm = (HK[(m0 - j) % N][None, :] * Is).sum(axis=1) / N
    cross = S_mt - S_sm
    print(f"   white noise, {M} records, f = 1/4: mean of cross term {cross.mean():+.4f} (estimate mean {S_mt.mean():.3f}); "
          f"var(cross)/var(multitaper) = {cross.var() / S_mt.var():.4f}, derived 1 - (K/N^2) sum_j H_K(j/N)^2 = {d2:.4f}; "
          f"correlation {np.corrcoef(S_mt, S_sm)[0, 1]:.4f}, derived sqrt(1 - d^2) = {np.sqrt(1 - d2):.4f}; "
          f"nu: multitaper {2 * S_mt.mean() ** 2 / S_mt.var():.1f}, smoothed {2 * S_sm.mean() ** 2 / S_sm.var():.1f}, derived 2K/(1-d^2) = {2 * K / (1 - d2):.1f}")
    # --- on coloured spectra: how far apart are the two on the same record
    for pn, psd in (("EEG-like", tc.eeg_psd), ("AR(4)", lambda f: ss.ar_psd(ss.AR4, f))):
        Xc = (cholesky(toeplitz(tc.acf(psd, N)), lower=True) @ rng.standard_normal((N, 1000))).T
        mm = np.arange(int(0.02 * N), int(0.48 * N)); St = psd(mm / N)
        Smt = (np.abs(np.fft.fft(V[None, :, :] * Xc[:, :, None], axis=1)[:, mm, :]) ** 2).mean(axis=2)
        Ic = np.abs(np.fft.fft(Xc, axis=1)) ** 2 / N
        Ssm = np.stack([(HK[(m - j) % N][None, :] * Ic).sum(axis=1) / N for m in mm], axis=1)
        print(f"   {pn:9s}: RMS dB difference between the two {np.sqrt(np.mean((DB * np.log(Ssm / Smt)) ** 2)):.2f}; "
              f"RMS dB error: multitaper {np.sqrt(np.mean((DB * np.log(Smt / St)) ** 2)):.2f}, smoothed periodogram {np.sqrt(np.mean((DB * np.log(Ssm / St)) ** 2)):.2f}; "
              f"mean cross term / true spectrum where the spectrum is low: {np.mean(((Smt - Ssm) / St)[:, mm / N > 0.3]):+.2f}")

for N, NW, K in ((256, 4, 7), (256, 4, 5), (400, 2, 3), (1024, 8, 15)):
    check(N, NW, K)

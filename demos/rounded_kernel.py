"""Smoothing a periodogram with the rounded-corner kernel built from K Slepian tapers: how close to multitaper?

Four smoothed periodograms are compared with the K-taper multitaper estimate on the same simulated records:
  the kernel H_K = (1/K) sum_k |V_k|^2 used directly on the untapered periodogram;
  the same with the Fejer kernel divided out (the kernel-matched lag window), so that the kernels agree exactly;
  the same after a taper built from the K tapers, w_t = sqrt((1/K) sum_k v_k(t)^2), which also matches the diagonal of Q;
  the box of half-width W on the untapered periodogram (the all-taper identity).
Reports the kernel mismatch, the distance between the matrices, nu for white noise, and per process the RMS dB
difference from multitaper, the correlation with it, the RMS dB error of each, and the share of negative estimates.
Run: python demos/rounded_kernel.py
"""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "demos"))
import numpy as np, specsmooth as ss
from scipy.linalg import toeplitz, cholesky, eigh
import tuned_comparison as tc
DB = 10 / np.log(10)

def est(Q, X, nfft, idx):
    c, U = eigh((Q + Q.T) / 2); keep = np.abs(c) > 1e-10 * np.abs(c).max()
    return tc.mc_estimates(U[:, keep], c[keep] / c.sum(), X, nfft, idx)

def run(N, NW, K, M=1000, seed=3):
    t = np.arange(N); lag = np.abs(t[:, None] - t[None, :]); nfft = 4 * N
    V = ss.dpss(N, NW, K)[0]; QK = V @ V.T / K
    q = np.array([np.trace(QK, offset=k) for k in range(N)])                      # lag sequence of the rounded-corner kernel (1/K) sum |V_k|^2
    rect = np.ones(N) / np.sqrt(N)
    wK = np.sqrt((V ** 2).mean(axis=1)); wK = wK / np.linalg.norm(wK)               # a taper built from the same tapers: root mean square of the K tapers
    rw = lambda w: np.array([(w[:N - k] * w[k:]).sum() for k in range(N)])
    def lagQ(w, g): return np.outer(w, w) * g[lag]
    cands = {
        "rounded kernel used directly, no taper": lagQ(rect, q / q[0]),
        "rounded kernel, Fejer kernel divided out, no taper": lagQ(rect, (q / rw(rect)) / (q[0] / rw(rect)[0])),
        "rounded kernel divided out, taper = rms of the K tapers": lagQ(wK, np.where(rw(wK) > 1e-9, q / np.maximum(rw(wK), 1e-300), 0)),
        "box of half-width W, no taper (the all-taper identity)": lagQ(rect, ss.box_lag_window(N, NW / N)),
    }
    HK = ss.kernel_quadratic(QK, 8 * N)
    procs = {"white": lambda f: np.ones_like(f), "EEG-like": tc.eeg_psd, "AR(4)": lambda f: ss.ar_psd(ss.AR4, f)}
    idx = np.arange(int(0.01 * nfft), int(0.49 * nfft), 4); freqs = idx / nfft
    rng = np.random.default_rng(seed)
    recs = {pn: ((cholesky(toeplitz(tc.acf(psd, N)), lower=True) @ rng.standard_normal((N, M))).T, psd(freqs)) for pn, psd in procs.items()}
    e = np.exp(-2j * np.pi * 0.25 * t)
    print(f"\nN = {N}, NW = {NW}, K = {K}: multitaper nu = {2 * K}; taper built from the K tapers has max/mean = {wK.max() / wK.mean():.2f}")
    print(f"{'smoothed periodogram':58s} {'kernel':>8s} {'matrix':>7s} {'nu':>5s} | per process: RMS dB difference from multitaper / correlation with multitaper / RMS dB error [multitaper's] / negative")
    for name, Q in cands.items():
        Q = Q / np.trace(Q); H = ss.kernel_quadratic(Q, 8 * N)
        line = f"{name:58s} {np.abs(H - HK).max() / HK.max():8.1e} {np.linalg.norm(Q - QK) / np.linalg.norm(QK):7.3f} {ss.dof_quadratic(e[:, None] * Q * e.conj()[None, :]):5.1f} |"
        for pn, (X, S) in recs.items():
            Sk = est(QK, X, nfft, idx); Sq = est(Q, X, nfft, idx); ok = Sq > 0
            d = DB * np.log(np.where(ok, Sq, np.nan) / Sk)
            r = np.mean([np.corrcoef(Sk[:, j], Sq[:, j])[0, 1] for j in range(0, Sk.shape[1], 5)])
            line += f" {pn}: {np.sqrt(np.nanmean(d ** 2)):4.2f} / {r:.3f} / {np.sqrt(np.nanmean((DB * np.log(np.where(ok, Sq, np.nan) / S)) ** 2)):4.2f} [{np.sqrt(np.mean((DB * np.log(Sk / S)) ** 2)):4.2f}] / {100 * (~ok).mean():3.1f}% |"
        print(line)

for N, NW, K in ((256, 4, 7), (256, 4, 8), (256, 4, 5), (400, 2, 3), (1024, 8, 15)):
    run(N, NW, K)

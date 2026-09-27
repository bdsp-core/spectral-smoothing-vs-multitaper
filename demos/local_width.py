"""How much would a width chosen separately at every frequency gain?

For each family the settings are scored at every frequency on 1000 simulated records. "One width" is the setting with the
smallest error over the band. "Best width at each frequency" picks a setting separately at every frequency. The choice is
made on one set of records and scored on a second, independent set, so that the selection does not flatter the result.
The selection uses the true spectrum: it shows what a frequency-dependent width could gain, and is not a procedure.
Families: a 25% cosine taper then a parabola (19 half-widths); multitaper with Slepian tapers and equal weights (every
K for 13 time-bandwidth products); adaptively weighted multitaper (K = 2NW - 1, 8 time-bandwidth products, 300 records).
Run: python demos/local_width.py
"""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "demos"))
import numpy as np, specsmooth as ss
from scipy.linalg import toeplitz, cholesky
import tuned_comparison as tc, taper_and_kernel as tk
DB = 10 / np.log(10)

def records(proc, N, M, seed):
    P = tc.PROCESSES[proc]
    return (cholesky(toeplitz(tc.acf(P["psd"], N)), lower=True) @ np.random.default_rng(seed).standard_normal((N, M))).T

def run(proc, N, M=1000, Ma=300):
    P = tc.PROCESSES[proc]; nfft = 4 * N; lo, hi = P["band"]; step = max(1, int(round((hi - lo) * nfft / 200)))
    idx = np.arange(int(round(lo * nfft)), int(round(hi * nfft)), step); freqs = idx / nfft; S = P["psd"](freqs)
    XA, XB = records(proc, N, M, 5), records(proc, N, M, 6)
    mse = lambda Sh: np.mean((DB * np.log(Sh / S)) ** 2, axis=0)                    # per frequency
    fam = {}
    w = ss.unit_taper("tukey", N, alpha=0.25)
    widths = [0.5, 0.75, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24]
    fam["cosine taper, then parabola"] = [(f"b={b:g}/N", mse(tk.smooth_est(XA, w, ss.parabolic_lag_window(N, b / N), nfft, idx)), mse(tk.smooth_est(XB, w, ss.parabolic_lag_window(N, b / N), nfft, idx))) for b in widths]
    rows = []
    for NW in (1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 10, 12):
        Vall = ss.dpss(N, NW, int(2 * NW))[0]
        for K in range(1, int(2 * NW) + 1):
            V = Vall[:, :K]; c = np.full(K, 1 / K)
            rows.append((f"NW={NW:g},K={K}", mse(tc.mc_estimates(V, c, XA, nfft, idx)), mse(tc.mc_estimates(V, c, XB, nfft, idx))))
    fam["multitaper, equal weights"] = rows
    rows = []
    for NW in (1.5, 2, 3, 4, 5, 6, 8, 10):
        K = int(2 * NW) - 1; V, lam = ss.dpss(N, NW, K)
        ad = lambda X: np.array([ss.multitaper_adaptive(x, V, lam, nfft)[0][idx] for x in X])
        rows.append((f"NW={NW:g},K={K}", mse(ad(XA[:Ma])), mse(ad(XB[:Ma]))))
    fam["multitaper, adaptive weights"] = rows
    print(f"\n{P['name']}, N = {N}: RMS dB error over the band")
    print(f"{'family':32s} {'one width for all frequencies':>32s} {'best width at each frequency':>30s} {'gain':>6s}")
    for name, rows in fam.items():
        A = np.array([r[1] for r in rows]); B = np.array([r[2] for r in rows])
        i = int(np.argmin(A.mean(axis=1))); fixed = np.sqrt(B[i].mean())
        j = np.argmin(A, axis=0); local = np.sqrt(B[j, np.arange(B.shape[1])].mean())
        print(f"{name:32s} {fixed:14.2f}  ({rows[i][0]:>14s}) {local:30.2f} {100 * (1 - local / fixed):5.0f}%")

for proc, N in (("ar4", 256), ("ar4", 1024), ("eeg", 400), ("eeg", 1024)):
    run(proc, N)

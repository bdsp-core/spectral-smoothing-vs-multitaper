"""Can one taper followed by a special smoothing kernel reproduce the K-taper multitaper estimate?

The lag window g_tau = q_tau / r_w(tau) (q: lag sums of the multitaper matrix, r_w: autocorrelation of the taper) gives the
smoothed tapered periodogram the multitaper kernel (exactly when r_w has no zeros, as without a taper), hence the same
expected value for every spectrum. The two are
still different estimators. This script reports, for N = 256, NW = 4, K = 7:
  the kernel mismatch, the distance between the matrices, the most negative eigen-weight, nu for white noise, and, on
  1000 simulated records of three processes, the RMS dB difference from the multitaper estimate of the same record,
  the RMS dB error of each, and the share of estimates that are negative. The two RMS dB figures are computed over the
  positive estimates only, since a negative estimate has no value in decibels.
It also prints the kernel properties of recipe (b'), for Table II.
Run: python demos/kernel_matched.py
"""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "demos"))
import numpy as np
from scipy.linalg import toeplitz, cholesky, eigh
import specsmooth as ss
import tuned_comparison as tc
import fit_single_window as fsw

DB = 10 / np.log(10)


def lag_Q(w, g):
    N = len(w); t = np.arange(N)
    return np.outer(w, w) * g[np.abs(t[:, None] - t[None, :])]


def estimates(Q, X, nfft, idx):
    c, U = eigh((Q + Q.T) / 2); keep = np.abs(c) > 1e-10 * np.abs(c).max()
    return tc.mc_estimates(U[:, keep], c[keep] / c.sum(), X, nfft, idx)


def main(N=256, NW=4, K=7, M=1000, seed=3):
    V, _ = ss.dpss(N, NW, K); QK = V @ V.T / K; t = np.arange(N); nfft = 4 * N
    tapers = {"no taper": np.ones(N) / np.sqrt(N), "Tukey 25%": ss.unit_taper("tukey", N, alpha=0.25), "Hann": ss.unit_taper("hann", N)}
    cands = {f"{n}, kernel matched to multitaper": lag_Q(w, ss.matched_lag_window(QK, w)) for n, w in tapers.items()}
    w, h, _ = fsw.fit_taper_kernel(QK)
    cands["closest taper and kernel (least squares)"] = lag_Q(w, h)
    cands["Tukey 25%, parabola of half-power width 2W"] = lag_Q(tapers["Tukey 25%"], ss.parabolic_lag_window(N, np.sqrt(2) * NW / N))
    cands["Hann, box of half-width W"] = lag_Q(tapers["Hann"], ss.box_lag_window(N, NW / N))
    cands["no taper, box of half-width W"] = lag_Q(tapers["no taper"], ss.box_lag_window(N, NW / N))
    HK = ss.kernel_quadratic(QK, 8 * N)
    procs = {"white noise": lambda f: np.ones_like(f), "EEG-like": tc.eeg_psd, "AR(4)": lambda f: ss.ar_psd(ss.AR4, f)}
    idx = np.arange(int(0.005 * nfft), int(0.495 * nfft), 4); freqs = idx / nfft
    rng = np.random.default_rng(seed); recs = {}
    for pn, psd in procs.items():
        recs[pn] = ((cholesky(toeplitz(tc.acf(psd, N)), lower=True) @ rng.standard_normal((N, M))).T, psd(freqs))
    e = np.exp(-2j * np.pi * 0.25 * t)
    print(f"N = {N}, NW = {NW}, K = {K}; {M} records per process")
    print(f"{'estimator':46s} {'kernel':>8s} {'matrix':>7s} {'neg. w':>8s} {'nu':>5s} | per process: RMS dB difference from multitaper / RMS dB error [multitaper] / negative estimates")
    for name, Q in cands.items():
        Q = Q / np.trace(Q); c = np.linalg.eigvalsh((Q + Q.T) / 2)
        H = ss.kernel_quadratic(Q, 8 * N); nu = ss.dof_quadratic(e[:, None] * Q * e.conj()[None, :])
        line = f"{name:46s} {np.abs(H - HK).max() / HK.max():8.1e} {np.linalg.norm(Q - QK) / np.linalg.norm(QK):7.3f} {c.min() / c.max():8.1e} {nu:5.1f} |"
        for pn, (X, S) in recs.items():
            Sk = estimates(QK, X, nfft, idx); Sq = estimates(Q, X, nfft, idx); ok = Sq > 0
            d = DB * np.log(np.where(ok, Sq, np.nan) / Sk); er = DB * np.log(np.where(ok, Sq, np.nan) / S); ek = DB * np.log(Sk / S)
            line += f" {pn}: {np.sqrt(np.nanmean(d ** 2)):5.2f} / {np.sqrt(np.nanmean(er ** 2)):5.2f} [{np.sqrt((ek ** 2).mean()):4.2f}] / {100 * (~ok).mean():4.1f}% |"
        print(line)
    print("\nKernel properties at N = 256, W = 4/N (Table II):")
    for name in ("Tukey 25%, parabola of half-power width 2W", "Hann, box of half-width W"):
        Q = cands[name] / np.trace(cands[name]); st = ss.kernel_stats(ss.kernel_quadratic(Q, 16 * N), W=NW / N)
        print(f"  {name:46s} nu {ss.dof_quadratic(e[:, None] * Q * e.conj()[None, :]):5.1f}  width {st['bw3'] * N:5.2f}  leakage(>2W) {st['mass_out_2W']:.5f}  "
              f"peak side lobe {st['peak_sidelobe_db']:6.1f} dB  two-tone {st['resolution'] * N:4.1f}")


if __name__ == "__main__":
    main()

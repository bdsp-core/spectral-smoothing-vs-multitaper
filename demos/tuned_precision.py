"""Monte Carlo precision of the tuned comparison (Table IV).

Each row of Table IV is rebuilt from its setting (SETTINGS, copied from demos/outputs/tuned_comparison.txt and
demos/outputs/taper_and_kernel.txt) and scored on the table's own 1000 records (seed 5; the adaptive rows on all 1000 here,
not 500), which reproduces the printed entries. Then:
  1. Paired bootstrap over records (2000 resamples): the standard error of each entry, and the 95% percentile interval of
     each entry relative to a reference row in the same block (tuned rows: "One taper, then smooth"; rows at the fixed
     resolution: recipe (b')). The rows share records, so the intervals of the differences are narrower than the entries' own.
  2. Rescoring on 2000 new records (seed 2026): the change in each entry, and whether the order of the rows changes within
     each block.
An entry is the RMS over records and frequencies of 10 log10(S_hat / S); for the matched kernel it is taken over the positive
estimates, as in Table IV.
Run: python demos/tuned_precision.py
"""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "demos"))
import numpy as np
from scipy.linalg import toeplitz, cholesky
import specsmooth as ss
import tuned_comparison as tc
import taper_and_kernel as tk

# the best setting of each family, per case, and the Table IV entry it reproduces (band RMS dB error)
SETTINGS = {
    ("ar4", 256): dict(one=(0.25, "parabola", 3.5), few=(3, 3, 2), welch=84, sine=5, slep=(4.5, 5), adapt=(5, 5),
                       matched=(0.25, 10, 2), none=1.5),
    ("ar4", 1024): dict(one=(0.1, "parabola", 8), few=(3, 3, 7), welch=128, sine=14, slep=(8.5, 13), adapt=(9, 13),
                        matched=(0.1, 12, 9), none=6),
    ("eeg", 400): dict(one=(0.1, "parabola", 6), few=(5, 3, 3.5), welch=80, sine=9, slep=(4.5, 9), adapt=(5, 9),
                       matched=(0.0, 12, 4), none=5),
    ("eeg", 1024): dict(one=(0.1, "parabola", 10), few=(5, 3, 7), welch=128, sine=15, slep=(7.5, 15), adapt=(8, 15),
                        matched=(0.0, 12, 9), none=10),
}
TUNED = ["One taper, then smooth", "A few Slepian tapers, then smooth", "Welch, Hann segments", "Multitaper, sine tapers, equal weights",
         "Multitaper, Slepian tapers, equal weights", "Multitaper, Slepian tapers, adaptive weights",
         "One taper, then kernel matched to a multitaper estimate", "No taper, then smooth"]
FIXED = ["Recipe (b')", "Hann periodogram, then box", "Multitaper, L = 7, adaptive weights", "Multitaper, L = 7, equal weights"]
LEAKY = {"No taper, then smooth", "Multitaper, L = 7, equal weights"}         # errors several times the others'; summarized apart


def taper(N, a):
    return np.ones(N) / np.sqrt(N) if a == 0 else ss.unit_taper("tukey", N, alpha=a)


def estimators(N, s):
    """Row name -> function of the records X (M x N) returning the estimates at the grid indices idx."""
    lag = {"box": ss.box_lag_window, "parabola": ss.parabolic_lag_window}

    def smooth(w, h):
        return lambda X, nfft, idx: tk.smooth_est(X, w, h, nfft, idx)

    def mt(V):
        return lambda X, nfft, idx: tc.mc_estimates(V, np.full(V.shape[1], 1 / V.shape[1]), X, nfft, idx)

    def adaptive(NW, K):
        V, lam = ss.dpss(N, NW, K)
        return lambda X, nfft, idx: np.array([ss.multitaper_adaptive(x, V, lam, nfft)[0][idx] for x in X])

    def few(Kh, NW0, Wn):
        Vh = ss.dpss(N, NW0, Kh)[0]; h = ss.box_lag_window(N, Wn / N)
        return lambda X, nfft, idx: sum(tk.smooth_est(X, Vh[:, k], h, nfft, idx) for k in range(Kh)) / Kh

    def welch(L):
        w = ss.unit_taper("hann", L)
        U = np.stack([np.r_[np.zeros(p), w, np.zeros(N - L - p)] for p in range(0, N - L + 1, max(1, L // 8))], axis=1)
        return lambda X, nfft, idx: tc.mc_estimates(U, np.full(U.shape[1], 1 / (U ** 2).sum()), X, nfft, idx)

    a, kern, Wn = s["one"]; am, NWm, Km = s["matched"]; wm = taper(N, am)
    return {
        "One taper, then smooth": smooth(taper(N, a), lag[kern](N, Wn / N)),
        "A few Slepian tapers, then smooth": few(*s["few"]),
        "Welch, Hann segments": welch(s["welch"]),
        "Multitaper, sine tapers, equal weights": mt(ss.sine_tapers(N, s["sine"])),
        "Multitaper, Slepian tapers, equal weights": mt(ss.dpss(N, *s["slep"])[0]),
        "Multitaper, Slepian tapers, adaptive weights": adaptive(*s["adapt"]),
        "One taper, then kernel matched to a multitaper estimate": smooth(wm, tk.matched_window(tk.multitaper_lag_sums(N, NWm, Km), wm)),
        "No taper, then smooth": smooth(taper(N, 0), ss.parabolic_lag_window(N, s["none"] / N)),
        "Recipe (b')": smooth(taper(N, 0.25), ss.parabolic_lag_window(N, np.sqrt(2) * 4 / N)),
        "Hann periodogram, then box": smooth(ss.unit_taper("hann", N), ss.box_lag_window(N, 4 / N)),
        "Multitaper, L = 7, adaptive weights": adaptive(4, 7),
        "Multitaper, L = 7, equal weights": mt(ss.dpss(N, 4, 7)[0]),
    }


def per_record(Sh, S):
    """Sum over frequencies of the squared dB error of the positive estimates, and their number, for each record."""
    ok = Sh > 0
    E2 = np.where(ok, (tc.DB * np.log(np.where(ok, Sh, 1.0) / S)) ** 2, 0.0)
    return E2.sum(axis=1), ok.sum(axis=1)


def records(proc, N, M, seed):
    P = tc.PROCESSES[proc]
    return (cholesky(toeplitz(tc.acf(P["psd"], N)), lower=True) @ np.random.default_rng(seed).standard_normal((N, M))).T


def run(proc, N, B=2000):
    P = tc.PROCESSES[proc]; nfft = 4 * N; lo, hi = P["band"]; step = max(1, int(round((hi - lo) * nfft / 200)))
    idx = np.arange(int(round(lo * nfft)), int(round(hi * nfft)), step); S = P["psd"](idx / nfft)
    XA, XB = records(proc, N, 1000, 5), records(proc, N, 2000, 2026)
    est = estimators(N, SETTINGS[(proc, N)])
    A = {k: per_record(f(XA, nfft, idx), S) for k, f in est.items()}
    Bn = {k: per_record(f(XB, nfft, idx), S) for k, f in est.items()}
    rms = lambda sc, r=slice(None): np.sqrt(sc[0][r].sum(-1) / sc[1][r].sum(-1))
    boot = np.random.default_rng(1).integers(0, 1000, (B, 1000))
    print(f"\n{P['name']}, N = {N}: entry on the table's 1000 records (bootstrap SE); 95% interval of the entry relative to the block's"
          f" first row; entry on 2000 new records")
    worst = {"ci": 0.0, "change": 0.0, "ci leaky": 0.0, "change leaky": 0.0}
    for block in (TUNED, FIXED):
        ref = np.sqrt(A[block[0]][0][boot].sum(1) / A[block[0]][1][boot].sum(1))
        for k in block:
            eb = np.sqrt(A[k][0][boot].sum(1) / A[k][1][boot].sum(1)); rel = 100 * (eb / ref - 1)
            lo_, hi_ = np.percentile(rel, [2.5, 97.5]); point = 100 * (rms(A[k]) / rms(A[block[0]]) - 1)
            sfx = " leaky" if k in LEAKY else ""
            if k != block[0]:
                worst["ci" + sfx] = max(worst["ci" + sfx], (hi_ - lo_) / 2)
            worst["change" + sfx] = max(worst["change" + sfx], abs(rms(Bn[k]) - rms(A[k])))
            print(f"  {k:56s} {rms(A[k]):6.3f} ({eb.std():.3f})   {point:+6.2f}% [{lo_:+6.2f}, {hi_:+6.2f}]   {rms(Bn[k]):6.3f}")
        oa = sorted(block, key=lambda k: rms(A[k])); ob = sorted(block, key=lambda k: rms(Bn[k]))
        swaps = [(u, v) for i, u in enumerate(oa) for v in oa[i + 1:] if ob.index(u) > ob.index(v)]
        print("  order on new records: " + ("unchanged" if not swaps else "; ".join(f"{u} and {v} swap ({rms(A[u]):.3f} vs {rms(A[v]):.3f}"
                                                                                       f" on the table's records)" for u, v in swaps)))
    NWs, Ks = SETTINGS[(proc, N)]["slep"]; w0 = taper(N, 0)
    Sh = tk.smooth_est(XA, w0, tk.matched_window(tk.multitaper_lag_sums(N, NWs, Ks), w0), nfft, idx)
    print(f"  kernel matched, without a taper, to the Slepian row (NW={NWs:g}, K={Ks}): {rms(per_record(Sh, S)):.3f} dB,"
          f" against {rms(A['Multitaper, Slepian tapers, equal weights']):.3f}; negative estimates {100 * (Sh <= 0).mean():.2f}%")
    print(f"  tapered rows: largest half-width of a 95% interval {worst['ci']:.2f}%, largest change on new records {worst['change']:.3f} dB;"
          f" untapered and L = 7 equal-weight rows: {worst['ci leaky']:.2f}% and {worst['change leaky']:.3f} dB")


if __name__ == "__main__":
    for proc, N in SETTINGS:
        run(proc, N)

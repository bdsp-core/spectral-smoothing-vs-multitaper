"""Each route tuned to do its best: is the multitaper estimate the best trade-off of bias against variance?

Every estimator family is searched over its parameters on a process with known spectrum, and the best setting of each
family is reported. The search has two stages.

1. Exact screening. For a Gaussian process the mean and variance of a quadratic estimator are exact: with
   z_k = sum_t v_k(t) exp(-i 2 pi f t) x_t and S_hat = sum_k c_k |z_k|^2,
       E S_hat   = sum_k c_k E|z_k|^2,
       var S_hat = sum_kl c_k c_l ( |E z_k conj(z_l)|^2 + |E z_k z_l|^2 ).
   Every estimator is first written as a bank of tapers v_k with weights c_k (its eigendecomposition). A dB error follows
   from the chi-square approximation with nu' = 2 mean^2 / var. The approximation overstates the error of leaky
   estimators, so it is used only to shortlist the five best settings of each family.
2. Monte Carlo scoring. The shortlisted settings are scored on M simulated records (the same records for every family),
   as the RMS over records and frequencies of 10 log10(S_hat / S). The adaptively weighted multitaper estimate is not
   quadratic, so all of its settings are scored this way.

Two processes: the AR(4) process of the paper (65 dB range, two narrow peaks) and an EEG-like spectrum sampled at
200 Hz (a 1/f^2 background with a knee at 3 Hz, an alpha peak at 10 Hz that is 1.4 Hz wide, and a noise floor; 30 dB range).

Run: python demos/tuned_comparison.py [ar4|eeg] [N ...]   (default: ar4 at 256 and 1024, eeg at 400 and 1024)
"""
import sys, pathlib, time
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.linalg import eigh, toeplitz, cholesky
from scipy.special import digamma, polygamma
from scipy.signal import windows
import specsmooth as ss

DB = 10 / np.log(10)


def eeg_psd(f, fs=200.0):
    """EEG-like spectrum, f in cycles/sample: 1/f^2 background with a knee at 3 Hz, alpha peak at 10 Hz (FWHM 1.4 Hz), noise floor."""
    hz = fs * np.minimum(np.mod(f, 1.0), 1.0 - np.mod(f, 1.0))
    return 100.0 / (1.0 + (hz / 3.0) ** 2) + 60.0 * np.exp(-(hz - 10.0) ** 2 / (2 * 0.6 ** 2)) + 0.01


PROCESSES = {
    "ar4": dict(name="AR(4) process", psd=lambda f: ss.ar_psd(ss.AR4, f), band=(0.005, 0.495), peaks=(0.09, 0.16), low=(0.3, 0.495), sizes=(256, 1024)),
    "eeg": dict(name="EEG-like spectrum at 200 Hz, scored from 1 to 40 Hz", psd=eeg_psd, band=(0.005, 0.2), peaks=(0.04, 0.06), low=(0.125, 0.2), sizes=(400, 1024)),
}


def acf(psd, N, nfft=1 << 18):
    """Autocovariance at lags 0..N-1, by inverse transform of the spectrum on a fine grid."""
    return np.fft.ifft(psd(np.arange(nfft) / nfft)).real[:N]


class Toeplitz:
    """Multiply the N x N covariance matrix with lags r by the columns of U, using the FFT."""
    def __init__(self, r):
        self.N = N = len(r); self.nfft = 1 << int(np.ceil(np.log2(2 * N)))
        q = np.zeros(self.nfft); q[:N] = r; q[self.nfft - N + 1:] = r[1:][::-1]
        self.R = np.fft.fft(q)

    def mul(self, U):
        return np.fft.ifft(self.R[:, None] * np.fft.fft(U, self.nfft, axis=0), axis=0)[:self.N]


def moments(V, c, T, freqs):
    """Exact mean and variance of S_hat(f) = sum_k c_k |v_k^T D_f x|^2 for real Gaussian x with covariance T."""
    N = V.shape[0]; t = np.arange(N); cc = np.outer(c, c)
    m = np.empty(len(freqs)); v = np.empty(len(freqs))
    for j, f in enumerate(freqs):
        U = V * np.exp(-2j * np.pi * f * t)[:, None]
        SU = T.mul(U)
        A = U.T @ SU.conj(); B = U.T @ SU
        m[j] = (c * np.diag(A).real).sum(); v[j] = (cc * (np.abs(A) ** 2 + np.abs(B) ** 2)).sum()
    return m, v


def bank(Q0, tol=1e-10):
    """Tapers and weights of a real symmetric matrix, weights normalized to sum to one."""
    c, V = eigh((Q0 + Q0.T) / 2)
    keep = c > tol * c.sum()
    return V[:, keep], c[keep] / c[keep].sum()


def lag_matrix(tapers, h):
    """Matrix of 'taper(s), then smooth with the lag window h'."""
    N = tapers.shape[0]; t = np.arange(N)
    Hm = h[np.abs(t[:, None] - t[None, :])]
    return sum(tapers[:, k][:, None] * Hm * tapers[:, k][None, :] for k in range(tapers.shape[1])) / tapers.shape[1]


def parabolic_lag_window(N, W):
    """Lag window of the unit-area parabolic kernel (3 / 4W)(1 - f^2 / W^2) on |f| <= W."""
    a = 2 * np.pi * W * np.arange(N, dtype=float); h = np.ones(N)
    nz = a > 1e-6
    h[nz] = 3 * (np.sin(a[nz]) - a[nz] * np.cos(a[nz])) / a[nz] ** 3
    return h


def screen(m, v, S):
    """RMS dB error over the band from the exact moments, by the chi-square approximation."""
    nu = 2 * m ** 2 / v
    bias = DB * np.log(m / S) + DB * (digamma(nu / 2) - np.log(nu / 2))
    return float(np.sqrt((bias ** 2 + DB ** 2 * polygamma(1, nu / 2)).mean()))


def mc_estimates(V, c, X, nfft, idx):
    """The quadratic estimate with bank (V, c) on every record (rows of X), at the grid indices idx."""
    out = np.zeros((X.shape[0], len(idx)))
    for k in range(V.shape[1]):
        out += c[k] * np.abs(np.fft.fft(X * V[:, k][None, :], nfft, axis=1)[:, idx]) ** 2
    return out


def mc_score(Sh, S, masks):
    E = DB * np.log(Sh / S)
    return [float(np.sqrt((E[:, k] ** 2).mean())) for k in masks] + [float(np.sqrt(((Sh / S - 1) ** 2).mean()))]


def run(N, proc="ar4", M=1000, seed=5, shortlist=5):
    t0 = time.time(); P = PROCESSES[proc]
    nfft = 4 * N; lo, hi = P["band"]; step = max(1, int(round((hi - lo) * nfft / 200)))
    idx = np.arange(int(round(lo * nfft)), int(round(hi * nfft)), step); freqs = idx / nfft
    S = P["psd"](freqs)
    masks = [np.ones(len(freqs), bool), (freqs >= P["peaks"][0]) & (freqs <= P["peaks"][1]), (freqs >= P["low"][0]) & (freqs <= P["low"][1])]
    r = acf(P["psd"], N); T = Toeplitz(r); white = Toeplitz(np.r_[1.0, np.zeros(N - 1)])
    rng = np.random.default_rng(seed); X = (cholesky(toeplitz(r), lower=True) @ rng.standard_normal((N, M))).T
    fams = {}

    def add(fam, label, V, c):
        fams.setdefault(fam, []).append((screen(*moments(V, c, T, freqs), S), label, V, c))

    widths = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20]
    for NW in [1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 10, 12]:
        V, lam = ss.dpss(N, NW, int(2 * NW))
        for K in range(1, int(2 * NW) + 1):
            add("Multitaper, Slepian tapers, equal weights", f"NW={NW:g}, K={K}", V[:, :K], np.full(K, 1 / K))
    for K in range(1, 41):
        add("Multitaper, sine tapers, equal weights", f"K={K}", ss.sine_tapers(N, K), np.full(K, 1 / K))
    singles = {"no taper": np.ones(N) / np.sqrt(N), "Hann": ss.unit_taper("hann", N)}
    for a in (0.1, 0.25, 0.5):
        w = windows.tukey(N, a); singles[f"Tukey {a:g}"] = w / np.linalg.norm(w)
    for NW0 in (1, 1.5, 2, 2.5, 3):
        singles[f"Slepian NW={NW0:g}"] = ss.dpss(N, NW0, 1)[0][:, 0]
    for tn, w in singles.items():
        for Wn in widths:
            for kn, h in (("box", ss.box_lag_window(N, Wn / N)), ("parabola", parabolic_lag_window(N, Wn / N))):
                V, c = bank(lag_matrix(w[:, None], h))
                add("No taper, then smooth" if tn == "no taper" else "One taper, then smooth", f"{tn}, {kn} of half-width {Wn:g}/N", V, c)
    for NW0 in (1.5, 2, 2.5, 3):
        for Kh in range(2, int(2 * NW0)):
            Vh = ss.dpss(N, NW0, Kh)[0]
            for Wn in widths:
                V, c = bank(lag_matrix(Vh, ss.box_lag_window(N, Wn / N)))
                add("A few Slepian tapers, then smooth", f"{Kh} tapers of NW={NW0:g}, box of half-width {Wn:g}/N", V, c)
    for frac in (48, 32, 24, 16, 12, 10, 8, 6, 5, 4, 3, 2):
        L = 2 * (N // frac // 2)
        if L < 8:
            continue
        w = ss.unit_taper("hann", L)
        for stp, sl in ((L // 2, "half overlap"), (max(1, L // 8), "7/8 overlap")):
            U = np.stack([np.r_[np.zeros(s), w, np.zeros(N - L - s)] for s in range(0, N - L + 1, stp)], axis=1)
            add("Welch, Hann segments", f"segments of {L}, {sl}, {U.shape[1]} segments", U, np.full(U.shape[1], 1 / (U ** 2).sum()))
    fixed = {"Paper setting: multitaper, NW=4, K=7, equal weights": (ss.dpss(N, 4, 7)[0], np.full(7, 1 / 7)),
             "Paper setting: multitaper, NW=4, K=6, equal weights": (ss.dpss(N, 4, 6)[0], np.full(6, 1 / 6)),
             "Paper setting: Hann, then box of half-width 4/N": bank(lag_matrix(ss.unit_taper("hann", N)[:, None], ss.box_lag_window(N, 4 / N)))}

    print(f"\n{P['name']}, N = {N}; {M} records; RMS dB error at {len(freqs)} frequencies (band / peaks / low spectrum); relative RMS error; nu and half-power width (1/N) for white noise")
    print(f"{'family: best setting':98s} {'band':>6s} {'peaks':>6s} {'low':>6s} {'relRMS':>7s} {'nu':>6s} {'width':>6s}")
    rows = []

    def report(name, V, c):
        sc = mc_score(mc_estimates(V, c, X, nfft, idx), S, masks)
        m, v = moments(V, c, white, [0.25]); bw = ss.kernel_stats(ss.kernel_multitaper(V, 8 * N, weights=c))["bw3"] * N
        rows.append((sc, name, f"{2 * m[0] ** 2 / v[0]:6.1f} {bw:6.1f}"))

    for fam, cand in fams.items():
        best = min(((mc_score(mc_estimates(V, c, X, nfft, idx), S, masks)[0], label, V, c) for _, label, V, c in sorted(cand, key=lambda q: q[0])[:shortlist]), key=lambda q: q[0])
        report(fam + ": " + best[1], best[2], best[3])
    ad = []
    for NW in (1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 10):
        for K in (int(2 * NW) - 2, int(2 * NW) - 1, int(2 * NW)):
            V, lam = ss.dpss(N, NW, K)
            Sh = np.array([ss.multitaper_adaptive(x, V, lam, nfft)[0][idx] for x in X[:M // 2]])
            ad.append((mc_score(Sh, S, masks), f"Multitaper, Slepian tapers, adaptive weights ({M // 2} records): NW={NW:g}, K={K}", f"{'-':>6s} {'-':>6s}"))
            if NW == 4 and K == 7:
                fixed_ad = (ad[-1][0], "Paper setting: multitaper, NW=4, K=7, adaptive weights", ad[-1][2])
    rows.append(min(ad, key=lambda q: q[0][0]))
    rows.sort(key=lambda q: q[0][0])
    n_tuned = len(rows)
    for name, (V, c) in fixed.items():
        report(name, V, c)
    rows.append(fixed_ad)
    for i, (sc, name, tail) in enumerate(rows):
        if i == n_tuned:
            print("  untuned, for reference:")
        print(f"{name[:98]:98s} {sc[0]:6.2f} {sc[1]:6.2f} {sc[2]:6.2f} {sc[3]:7.3f} {tail}")
    print(f"[{time.time() - t0:.0f} s]")


if __name__ == "__main__":
    args = sys.argv[1:]
    procs = [a for a in args if a in PROCESSES] or list(PROCESSES)
    sizes = [int(a) for a in args if a.isdigit()]
    for proc in procs:
        for N in (sizes or PROCESSES[proc]["sizes"]):
            run(N, proc)

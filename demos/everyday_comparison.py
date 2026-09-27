"""The everyday estimators compared over an ensemble, with two metrics, adaptive weights, the hybrid, and matched nu.

Run: python demos/everyday_comparison.py  (writes demos/outputs/everyday_comparison.txt when redirected).
For the AR(4) process (65 dB dynamic range): median |dB error| over the whole band and RMS dB error over the peak
region (0.09-0.16 cycles/sample), over M realizations; and the equivalent dof of each estimator.
"""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import specsmooth as ss


def run(N, NW, M, seed, seg_len, nfft=None, W_matched=None):
    rng = np.random.default_rng(seed)
    nfft = nfft or 4 * N
    W = NW / N; K = int(2 * NW - 1)
    f = np.arange(nfft // 2) / nfft; truth = ss.ar_psd(ss.AR4, f)
    peak = (f >= 0.09) & (f <= 0.16)
    V, lam = ss.dpss(N, NW)
    Vh, _ = ss.dpss(N, NW / 2, max(1, int(NW) - 2))               # hybrid: the K' = 2(NW/2) - 2 well-concentrated tapers at half the bandwidth ...
    hb = ss.box_lag_window(N, W); hb2 = hb                           # ... then the same box of half-width W, so its half-power width matches the others
    hann = ss.unit_taper("hann", N); rect = np.ones(N) / np.sqrt(N)
    hseg = ss.unit_taper("hann", seg_len); step = max(1, (N - seg_len) // max(1, int((N - seg_len) / (seg_len / 2))))   # as in Figs. 6-7: 9 segments at N = 1024
    nu_q = lambda m, **q: (ss.dof_quadratic(ss.quadratic_matrix(N, 0.25, m, **q)), ss.kernel_stats(ss.kernel_quadratic(ss.quadratic_matrix(N, 0.0, m, **q).real, 8 * N)))
    ests = {
        "untapered periodogram, then box": (lambda x: ss.lag_window_estimate(x, hb, nfft)[0], nu_q("lagwindow", h=hb, taper=rect)),
        f"multitaper, K={K}, equal weights": (lambda x: ss.multitaper(x, V, nfft)[0], nu_q("multitaper", tapers=V)),
        f"multitaper, K={K - 1}, equal weights": (lambda x: ss.multitaper(x, V[:, :-1], nfft)[0], nu_q("multitaper", tapers=V[:, :-1])),
        f"multitaper, K={K}, adaptive weights": (None, None),
        f"hybrid: {Vh.shape[1]} Slepians (NW={NW / 2:g}), then box W": (lambda x: ss.hybrid_estimate(x, Vh, hb2, nfft)[0], nu_q("hybrid", tapers=Vh, h=hb2)),
        "Hann periodogram, then box": (lambda x: ss.lag_window_estimate(x, hb, nfft, taper=hann)[0], nu_q("lagwindow", h=hb, taper=hann)),
        f"Welch, Hann segments of {seg_len}, step {step}": (lambda x: ss.welch_sliding(x, hseg, nfft, step=step, overhang=False)[0], nu_q("welch", taper=hseg, step=step)),
    }
    if W_matched:
        hm = ss.box_lag_window(N, W_matched / N)
        ests[f"Hann periodogram, then box W={W_matched:g}/N (matched nu)"] = (lambda x: ss.lag_window_estimate(x, hm, nfft, taper=hann)[0], nu_q("lagwindow", h=hm, taper=hann))
    X = [ss.ar_process(ss.AR4, N, rng) for _ in range(M)]
    print(f"\nAR(4), N={N}, NW={NW} (W={NW}/N), {M} realizations, nfft={nfft}")
    print(f"{'estimator':58s} {'nu':>6s} {'width':>6s} {'2-tone':>7s} {'median |err| band':>18s} {'RMS err peaks':>14s} {'median |err| f>0.3':>19s}   (width = half-power width, 2-tone = smallest separation of two equal tones with a 3 dB dip, both in units of 1/N)")
    hi = f > 0.3
    for name, (fn, nu) in ests.items():
        errs = []; nus = []
        for x in X:
            if fn is None:
                S, _, nuf = ss.multitaper_adaptive(x, V, lam, nfft); nus.append(nuf[:nfft // 2])
            else:
                S = fn(x)
            errs.append(10 * np.log10(S[:nfft // 2] / truth))
        E = np.array(errs)
        if fn is None:
            nuf = np.array(nus); nu_s = f"{np.median(nuf):.1f}*"; wd = tt = "   -"; extra = f"   (*median over f; {np.median(nuf[:, peak]):.1f} at the peaks, {np.median(nuf[:, hi]):.1f} for f>0.3)"
        else:
            nu_s = f"{nu[0]:.1f}"; wd = f"{nu[1]['bw3'] * N:.1f}"; tt = f"{nu[1]['resolution'] * N:.1f}"; extra = ""
        print(f"{name:58s} {nu_s:>6s} {wd:>6s} {tt:>7s} {np.median(np.abs(E)):18.2f} {np.sqrt(np.mean(E[:, peak] ** 2)):14.2f} {np.median(np.abs(E[:, hi])):19.2f}{extra}")


if __name__ == "__main__":
    run(N=1024, NW=4, M=300, seed=21, seg_len=192, W_matched=6.5)
    run(N=256, NW=4, M=300, seed=22, seg_len=48, W_matched=6.5)
    run(N=128, NW=3, M=300, seed=23, seg_len=30, W_matched=5.0)

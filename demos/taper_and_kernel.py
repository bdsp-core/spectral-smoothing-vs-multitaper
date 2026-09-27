"""Which single taper and which smoothing kernel? The best width for each pair, scored on simulated records.

For each test case (the AR(4) process at N = 128, 256 and 1024; the EEG-like spectrum at N = 400 and 1024), each taper
(none, cosine tapers of 10, 25 and 50 percent, Hann, one Slepian taper with NW = 1 or 2) and each kernel (box, parabola),
the half-width with the smallest RMS dB error over the band is found on 1000 simulated records, the same records for
every pair. Prints the half-width and the error over the band, at the peaks and where the spectrum is low.
Also prints, for each taper, the factor N sum w^4 / (sum w^2)^2 by which it divides the degrees of freedom after smoothing.
Run: python demos/taper_and_kernel.py
"""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "demos"))
import numpy as np
from scipy.linalg import toeplitz, cholesky
import specsmooth as ss
import tuned_comparison as tc

WIDTHS = [1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 10, 12, 14, 16]


def smooth_est(X, w, h, nfft, idx):
    """The tapered, smoothed periodogram of every record (rows of X), at the grid indices idx."""
    N = X.shape[1]
    r = np.fft.ifft(np.abs(np.fft.fft(X * w[None, :], nfft, axis=1)) ** 2, axis=1).real
    lw = np.zeros(nfft); lw[:N] = h; lw[nfft - N + 1:] = h[1:][::-1]
    return np.fft.fft(r * lw[None, :], axis=1).real[:, idx]


def tapers_for(N):
    d = {"none": np.ones(N) / np.sqrt(N), "cosine 10%": ss.unit_taper("tukey", N, alpha=0.1), "cosine 25%": ss.unit_taper("tukey", N, alpha=0.25),
         "cosine 50%": ss.unit_taper("tukey", N, alpha=0.5), "Hann": ss.unit_taper("hann", N),
         "Slepian NW=1": ss.dpss(N, 1, 1)[0][:, 0], "Slepian NW=2": ss.dpss(N, 2, 1)[0][:, 0]}
    return d


def case(proc, N, M=1000, seed=5):
    P = tc.PROCESSES[proc]; nfft = 4 * N; lo, hi = P["band"]; step = max(1, int(round((hi - lo) * nfft / 200)))
    idx = np.arange(int(round(lo * nfft)), int(round(hi * nfft)), step); freqs = idx / nfft; S = P["psd"](freqs)
    masks = [np.ones(len(freqs), bool), (freqs >= P["peaks"][0]) & (freqs <= P["peaks"][1]), (freqs >= P["low"][0]) & (freqs <= P["low"][1])]
    rng = np.random.default_rng(seed); X = (cholesky(toeplitz(tc.acf(P["psd"], N)), lower=True) @ rng.standard_normal((N, M))).T
    return X, nfft, idx, S, masks


if __name__ == "__main__":
    print("Factor by which a taper divides the degrees of freedom after smoothing, N sum w^4 / (sum w^2)^2:")
    for tn, w in tapers_for(4096).items():
        print(f"  {tn:14s} {4096 * (w ** 4).sum() / (w ** 2).sum() ** 2:.3f}")
    for proc, N in (("ar4", 128), ("ar4", 256), ("ar4", 1024), ("eeg", 400), ("eeg", 1024)):
        X, nfft, idx, S, masks = case(proc, N)
        print(f"\n{tc.PROCESSES[proc]['name']}, N = {N}: best half-width, and RMS dB error over the band / at the peaks / where the spectrum is low")
        for tn, w in tapers_for(N).items():
            out = []
            for kn, lagw in (("box", ss.box_lag_window), ("parabola", ss.parabolic_lag_window)):
                best = min(((tc.mc_score(smooth_est(X, w, lagw(N, Wn / N), nfft, idx), S, masks), Wn) for Wn in WIDTHS), key=lambda q: q[0][0])
                out.append(f"{kn}: {best[1]:>4g}/N  {best[0][0]:5.2f} / {best[0][1]:5.2f} / {best[0][2]:5.2f}")
            print(f"  {tn:14s} " + "     ".join(out))

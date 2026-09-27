"""How far from a strong spectral line (e.g. mains interference) is each practical estimator biased?

A line of power P at f0 on a flat background S adds P H(f - f0) to the expected estimate, where H is the estimator's kernel of
unit area, and leaves the background unchanged. Let D = 10 log10(P H(0) / S) be the height of the line above the background in
the expected estimate, which is what a user sees. The expected bias at offset d from the line is then exactly
    10 log10(1 + 10^(D/10) H(d) / H(0))  dB,
so it depends only on D and on the shape of the kernel. The table gives the largest offset, in units of the half-bandwidth W, at
which the bias is 1 dB or more. At fixed 2NW the offsets are the same for every N.
Run: python demos/line_leakage.py
"""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np, specsmooth as ss


def kernels(N, NW, nfft):
    W = NW / N; K = int(2 * NW - 1)
    tuk, hann = ss.unit_taper("tukey", N, alpha=0.25), ss.unit_taper("hann", N)
    return {"recipe (b'): 25% cosine, parabola": ss.kernel_smoothed(tuk, ss.parabolic_lag_window(N, np.sqrt(2) * W), nfft),
            "Hann, then box": ss.kernel_smoothed(hann, ss.box_lag_window(N, W), nfft),
            f"multitaper, K = 2NW - 1": ss.kernel_multitaper(ss.dpss(N, NW, K)[0], nfft)}


def reach(H, f, W, D, thr=1.0):
    """Largest offset |d| / W at which a line D dB above the background biases the expected estimate by thr dB or more."""
    bias = 10 * np.log10(1 + 10 ** (D / 10) * H / H.max())
    hit = np.abs(f)[bias >= thr]
    return hit.max() / W


if __name__ == "__main__":
    heights = (20, 30, 40)
    for N in (256, 400, 1024):
        nfft = 64 * N; f = ss.signed_freq(nfft)
        print(f"N = {N}: largest offset from the line, in units of W, at which the expected bias is >= 1 dB")
        print(f"  {'2NW':>4}  {'estimator':36s}" + "".join(f"  D = {D} dB" for D in heights))
        for NW in (2, 3, 4):
            for name, H in kernels(N, NW, nfft).items():
                print(f"  {2 * NW:4d}  {name:36s}" + "".join(f"  {reach(H, f, NW / N, D):8.1f}" for D in heights))
        print()

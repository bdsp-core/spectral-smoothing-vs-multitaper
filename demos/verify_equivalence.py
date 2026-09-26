"""Numerical check of the central claim (run: python demos/verify_equivalence.py).

Claim 1 (exact):  with ALL N Slepian tapers weighted by their eigenvalues,
    sum_k lambda_k |J_k(f)|^2  =  int_{f-W}^{f+W} I(f') df'        (I = |X(f)|^2, the raw periodogram)
  because sum_k lambda_k v_k v_k^T is the Toeplitz matrix sin(2 pi W tau)/(pi tau), whose lag
  sequence is the inverse Fourier transform of a box of half-width W.
Claim 2 (approximate): the usual K = 2NW-1 equal-weight multitaper estimate is that box-smoothed
  periodogram minus the leakage carried by the low-concentration tapers it drops.
Also printed: kernel resolution/leakage and exact equivalent dof for MT vs "taper, then box-smooth".
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import specsmooth as ss

N, NW = 256, 4
W, K, nfft = NW / N, 2 * NW - 1, 16 * 256
rng = np.random.default_rng(1)
V, lam = ss.dpss_all(N, W)
print(f"N={N} NW={NW} K={K}  sum(lambda)={lam.sum():.4f} (2NW={2*NW})")
print("lambda[:K+2] =", np.array2string(lam[:K + 2], precision=4))

t = np.arange(N)
x = ss.ar_process(ss.AR4, N, rng)
x = x / x.std() + 0.5 * np.sin(2 * np.pi * 0.11 * t)
Sk = np.abs(np.fft.fft(V * x[:, None], nfft, axis=0)) ** 2
S_full = Sk @ lam
S_box, f = ss.lag_window_estimate(x, ss.box_lag_window(N, W), nfft)      # (1/2W) int I/N
rhs = 2 * W * N * S_box
print(f"\n[Claim 1] max|sum_k lam_k|J_k|^2 - int_(f-W)^(f+W) I| / max = {np.max(np.abs(S_full - rhs)) / rhs.max():.2e}")
S_mt = Sk[:, :K].mean(axis=1)
d = np.sort(np.abs(10 * np.log10(S_mt / S_box)))
print(f"[Claim 2] K-taper MT vs box-smoothed raw periodogram, |dB| over f: median {d[len(d)//2]:.3f}, "
      f"95% {d[int(0.95*len(d))]:.3f}, max {d[-1]:.3f}")

Vk, _ = ss.dpss(N, NW)
rect, hann, boh = np.ones(N) / np.sqrt(N), ss.unit_taper("hann", N), ss.unit_taper("bohman", N)
hb = ss.box_lag_window(N, W)
kernels = [("ideal box", ss.kernel_box(nfft, W)),
           ("multitaper K=2NW-1", ss.kernel_multitaper(Vk, nfft)),
           ("raw periodogram + box", ss.kernel_smoothed(rect, hb, nfft)),
           ("Hann periodogram + box", ss.kernel_smoothed(hann, hb, nfft)),
           ("Bohman periodogram + box", ss.kernel_smoothed(boh, hb, nfft))]
H_full = ss.kernel_multitaper(V, nfft, weights=lam)
print(f"[check] eigen-weighted MT kernel vs box*Fejer computed independently: {np.max(np.abs(H_full - kernels[2][1])) / H_full.max():.2e}")
print("\nkernel                      bw3(1/N)  mass>W   mass>2W  sidelobe(dB)  2-tone res(1/N)")
for name, H in kernels:
    s = ss.kernel_stats(H, W)
    print(f"{name:27s} {s['bw3']*N:7.2f}  {s['mass_out']:.4f}  {s['mass_out_2W']:.4f}  {s['peak_sidelobe_db']:8.1f}       {s['resolution']*N:.2f}")

f0 = 0.25
dofs = [("multitaper K=2NW-1", ss.quadratic_matrix(N, f0, "multitaper", tapers=Vk)),
        ("eigen-weighted, all N tapers", ss.quadratic_matrix(N, f0, "multitaper", tapers=V, weights=lam)),
        ("raw periodogram + box", ss.quadratic_matrix(N, f0, "lagwindow", h=hb)),
        ("Hann periodogram + box", ss.quadratic_matrix(N, f0, "lagwindow", h=hb, taper=hann)),
        ("Bohman periodogram + box", ss.quadratic_matrix(N, f0, "lagwindow", h=hb, taper=boh))]
print("\nexact equivalent dof at f0=0.25 (white noise); 2K =", 2 * K)
for name, Q in dofs:
    print(f"  {name:30s} {ss.dof_quadratic(Q):6.1f}")

w, s, err, H_fit = ss.fit_box_gaussian(kernels[1][1], nfft, np.linspace(0.5 * W, 1.2 * W, 36), np.linspace(0.02 * W, 0.6 * W, 30))
print(f"\nMT kernel ~ box(half-width {w*N:.2f}/N) * Gaussian(sigma {s*N:.2f}/N): relative squared error {err:.2e}")

fs = ss.signed_freq(nfft); o = np.argsort(fs)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for name, H in kernels + [("box*Gaussian fit to MT", H_fit)]:
    ax[0].plot(fs[o] * N, 10 * np.log10(H[o] + 1e-15), lw=1, label=name)
ax[0].set(xlim=(-4 * NW, 4 * NW), ylim=(-90, 10), xlabel="frequency (1/N)", ylabel="kernel (dB)", title=f"Expected-value smoothing kernels, NW={NW}")
ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)
h = slice(0, nfft // 2)
S_hb, _ = ss.lag_window_estimate(x, hb, nfft, taper=hann)
for S, lab in [(S_box, "raw periodogram + box"), (S_mt, f"multitaper K={K}"), (S_hb, "Hann periodogram + box")]:
    ax[1].plot(f[h], 10 * np.log10(S[h]), lw=.8, label=lab)
ax[1].set(xlabel="frequency (cycles/sample)", ylabel="dB", title="One realization: AR(4) + sinusoid"); ax[1].legend(fontsize=7); ax[1].grid(alpha=.3)
out = pathlib.Path(__file__).resolve().parents[1] / "figures" / "verify_equivalence.png"
fig.tight_layout(); fig.savefig(out, dpi=120); print("\nfigure:", out)

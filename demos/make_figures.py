"""Paper figures (run: python demos/make_figures.py). Writes figures/fig*.png.

fig1  kernels: multitaper vs 'taper, then smooth' at equal design bandwidth W
fig2  AR(4) stress test (70 dB dynamic range): who leaks
fig3  the 2015 whiteboard plan: variance (dof) and leakage versus measured resolution, per method
fig4  real EEG (InterestingSignal.mat, 128 Hz): multitaper spectrogram vs Hann-periodogram-then-box
"""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import specsmooth as ss

FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)


def fig1_kernels(N=256, NW=4, nfft=8192):
    W = NW / N
    Vk, _ = ss.dpss(N, NW)
    hb = ss.box_lag_window(N, W)
    rect, hann, boh = np.ones(N) / np.sqrt(N), ss.unit_taper("hann", N), ss.unit_taper("bohman", N)
    ks = [("ideal box, half-width W", ss.kernel_box(nfft, W), "k:"),
          (f"multitaper, K=2NW-1={2*NW-1}", ss.kernel_multitaper(Vk, nfft), "C3-"),
          ("raw periodogram, then box", ss.kernel_smoothed(rect, hb, nfft), "C0-"),
          ("Hann periodogram, then box", ss.kernel_smoothed(hann, hb, nfft), "C2-"),
          ("Bohman periodogram, then box", ss.kernel_smoothed(boh, hb, nfft), "C1-")]
    fs = ss.signed_freq(nfft); o = np.argsort(fs)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for name, H, st in ks:
        ax[0].plot(fs[o] * N, H[o] / N, st, lw=1.2, label=name)
        ax[1].plot(fs[o] * N, 10 * np.log10(np.maximum(H[o], 1e-15)), st, lw=1.2, label=name)
    ax[0].set(xlim=(-2.5 * NW, 2.5 * NW), xlabel="frequency (1/N)", ylabel="kernel (linear, x N)", title="main lobe")
    ax[1].set(xlim=(-6 * NW, 6 * NW), ylim=(-100, 5), xlabel="frequency (1/N)", ylabel="kernel (dB)", title="sidelobes = leakage")
    ax[0].legend(fontsize=7); [a.grid(alpha=.3) for a in ax]
    fig.suptitle(f"Expected-value smoothing kernels at equal design bandwidth (N={N}, NW={NW})")
    fig.tight_layout(); fig.savefig(FIG / "fig1_kernels.png", dpi=130); plt.close(fig)


def fig2_ar4(N=1024, NW=4, nfft=8192, seed=3):
    rng = np.random.default_rng(seed)
    W = NW / N
    x = ss.ar_process(ss.AR4, N, rng)
    f = np.arange(nfft // 2) / nfft
    truth = ss.ar_psd(ss.AR4, f)
    Vk, _ = ss.dpss(N, NW)
    hb = ss.box_lag_window(N, W)
    hann = ss.unit_taper("hann", N)
    ests = [("periodogram", ss.periodogram(x, nfft)[0], "0.6", .5),
            ("raw periodogram, then box", ss.lag_window_estimate(x, hb, nfft)[0], "C0", 1),
            ("Hann periodogram, then box", ss.lag_window_estimate(x, hb, nfft, taper=hann)[0], "C2", 1),
            (f"multitaper K={2*NW-1}", ss.multitaper(x, Vk, nfft)[0], "C3", 1),
            (f"multitaper K={2*NW-2} (drop the leakiest taper)", ss.multitaper(x, Vk[:, :-1], nfft)[0], "C1", 1)]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(f, 10 * np.log10(truth), "k", lw=2, label="true PSD (AR4)")
    for name, S, c, lw in ests:
        ax.plot(f, 10 * np.log10(S[:nfft // 2]), color=c, lw=lw, label=name)
    ax.set(xlabel="frequency (cycles/sample)", ylabel="dB", title=f"AR(4) process, N={N}, design bandwidth W={NW}/N")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(FIG / "fig2_ar4_leakage.png", dpi=130); plt.close(fig)


def fig3_tradeoff(N=256, nfft=8192, f0=0.25):
    """Sweep each method's own bandwidth parameter; plot dof and leakage against MEASURED half-power bandwidth."""
    hann, boh, rect = ss.unit_taper("hann", N), ss.unit_taper("bohman", N), np.ones(N) / np.sqrt(N)
    NWs = [1.5, 2, 2.5, 3, 4, 5, 6, 8]
    curves = {}

    def add(name, H, Q):
        s = ss.kernel_stats(H)
        curves.setdefault(name, []).append((s["bw3"] * N, ss.dof_quadratic(Q), s["mass_out_2bw"], s["peak_sidelobe_db"]))

    for NW in NWs:
        W = NW / N; K = int(2 * NW - 1)
        Vk, _ = ss.dpss(N, NW, K)
        add("multitaper (DPSS, K=2NW-1)", ss.kernel_multitaper(Vk, nfft), ss.quadratic_matrix(N, f0, "multitaper", tapers=Vk))
        V2 = Vk[:, :-1] if K > 1 else Vk
        add("multitaper (DPSS, K=2NW-2)", ss.kernel_multitaper(V2, nfft), ss.quadratic_matrix(N, f0, "multitaper", tapers=V2))
        St = ss.sine_tapers(N, K)
        add("multitaper (sine tapers)", ss.kernel_multitaper(St, nfft), ss.quadratic_matrix(N, f0, "multitaper", tapers=St))
        hb = ss.box_lag_window(N, W)
        for tname, tp in [("raw", rect), ("Hann", hann), ("Bohman", boh)]:
            add(f"{tname} periodogram, then box", ss.kernel_smoothed(tp, hb, nfft), ss.quadratic_matrix(N, f0, "lagwindow", h=hb, taper=tp))
        hg = ss.gaussian_lag_window(N, W / 2)
        add("Hann periodogram, then Gaussian", ss.kernel_smoothed(hann, hg, nfft), ss.quadratic_matrix(N, f0, "lagwindow", h=hg, taper=hann))
    for L in [16, 24, 32, 48, 64, 96, 128]:
        wl = ss.unit_taper("hann", L)
        add("Welch (Hann, 50% overlap)", ss.kernel_wosa(N, L, .5, nfft, wl), ss.quadratic_matrix(N, f0, "welch", seg_len=L, overlap=.5, taper=wl))

    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    for name, pts in curves.items():
        p = np.array(sorted(pts))
        ax[0].plot(p[:, 0], p[:, 1], "o-", ms=3, lw=1, label=name)
        ax[1].plot(p[:, 0], p[:, 2], "o-", ms=3, lw=1, label=name)
        ax[2].plot(p[:, 0], p[:, 3], "o-", ms=3, lw=1, label=name)
    ax[0].set(xlabel="half-power bandwidth (1/N)", ylabel="equivalent dof (2K-like)", title="variance: higher is better")
    ax[1].set(xlabel="half-power bandwidth (1/N)", ylabel="kernel mass beyond 2x half-power half-width", title="broadband leakage: lower is better", yscale="log")
    ax[2].set(xlabel="half-power bandwidth (1/N)", ylabel="peak sidelobe (dB)", title="peak sidelobe: lower is better")
    ax[0].legend(fontsize=7); [a.grid(alpha=.3) for a in ax]
    fig.suptitle(f"Resolution / variance / leakage per method, N={N} (the 2015 whiteboard plan)")
    fig.tight_layout(); fig.savefig(FIG / "fig3_tradeoff.png", dpi=130); plt.close(fig)


def fig4_eeg(NW=3, win_s=4.0, fmax=30.0):
    import scipy.io as sio
    p = ROOT / "legacy_matlab" / "InterestingSignal.mat"
    if not p.exists():
        print("fig4 skipped: legacy_matlab/InterestingSignal.mat not found"); return
    d = sio.loadmat(p); s = d["s"].ravel().astype(float); Fs = float(d["Fs"].ravel()[0])
    N = int(win_s * Fs); step = N // 2; nfft = 4 * N
    W = NW / N
    Vk, _ = ss.dpss(N, NW)
    hann = ss.unit_taper("hann", N); hb = ss.box_lag_window(N, W)
    f = np.arange(nfft // 2) / nfft * Fs; keep = f <= fmax
    starts = range(0, len(s) - N + 1, step)
    mt, sb, pg = [], [], []
    for st in starts:
        seg = s[st:st + N]; seg = seg - seg.mean()
        mt.append(ss.multitaper(seg, Vk, nfft)[0][:nfft // 2][keep])
        sb.append(ss.lag_window_estimate(seg, hb, nfft, taper=hann)[0][:nfft // 2][keep])
        pg.append(ss.periodogram(seg, nfft, taper=hann)[0][:nfft // 2][keep])
    mt, sb, pg = (10 * np.log10(np.array(a).T + 1e-12) for a in (mt, sb, pg))
    tt = (np.array(list(starts)) + N / 2) / Fs
    vmin, vmax = np.percentile(mt, [2, 99])
    fig, ax = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    for a, img, title in zip(ax[:3], (pg, mt, sb), ("Hann periodogram (unsmoothed)", f"multitaper NW={NW}, K={2*NW-1}",
                                                    f"Hann periodogram, then box of half-width W={W*Fs:.2f} Hz")):
        a.imshow(img, aspect="auto", origin="lower", extent=[tt[0], tt[-1], f[keep][0], f[keep][-1]], vmin=vmin, vmax=vmax, cmap="jet")
        a.set(ylabel="Hz", title=title)
    dd = mt - sb
    im = ax[3].imshow(dd, aspect="auto", origin="lower", extent=[tt[0], tt[-1], f[keep][0], f[keep][-1]], vmin=-3, vmax=3, cmap="RdBu_r")
    ax[3].set(ylabel="Hz", xlabel="time (s)", title=f"difference multitaper - smoothed (dB): median |diff| = {np.median(np.abs(dd)):.2f} dB, 95% = {np.percentile(np.abs(dd), 95):.2f} dB")
    fig.colorbar(im, ax=ax[3], fraction=.02)
    fig.tight_layout(); fig.savefig(FIG / "fig4_eeg_spectrograms.png", dpi=110); plt.close(fig)
    print(f"fig4: {len(tt)} epochs of {win_s:.0f} s; MT vs smoothed |diff| median {np.median(np.abs(dd)):.2f} dB, 95% {np.percentile(np.abs(dd), 95):.2f} dB")


if __name__ == "__main__":
    fig1_kernels(); print("fig1 done")
    fig2_ar4(); print("fig2 done")
    fig3_tradeoff(); print("fig3 done")
    fig4_eeg()
    print("figures in", FIG)

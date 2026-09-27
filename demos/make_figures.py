"""Paper figures (run: python demos/make_figures.py). Writes figures/*.png at 300 dpi, authored at final column width.

Main text (paper order):  fig9 graphical abstract (three routes, one estimate) | fig0b the motif: four estimators over their kernels |
  fig0a the record and the bias-variance sweep | fig5 the Slepian windows tile the box | fig6 three ways to build a taper bank |
  fig8 the everyday estimators (motif) | fig7 the everyday trio on EEG | fig10, fig11 seizure clips.
Supplement: fig1 kernels at equal bandwidth | fig2 AR(4): dropping the leakiest taper | fig3 resolution/variance/leakage sweep |
  fig6_full the taper banks with eigen-tapers and kernels.  fig4 (7-min spectrograms) is kept but no longer in the paper.

Presentation conventions (one style for every figure): bold lowercase panel letters outside each frame; one colour per
estimator (COLOR); reference truth heavy black; single-realization periodograms thin light grey; no suptitles; the
spectrograms share one perceptually uniform colour map and one dB range (SPEC_VLIM).
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

# ---------------------------------------------------------------- shared style
W2, W1 = 7.16, 3.5                     # IEEE double- and single-column widths, inches
plt.rcParams.update({
    "font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6.2, "legend.frameon": False, "lines.linewidth": 1.0, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "mathtext.fontset": "dejavusans",
    "savefig.dpi": 300, "figure.dpi": 100,
})
COLOR = {"truth": "black", "periodogram": "0.65", "multitaper": "#0072B2", "smooth": "#E69F00", "hann_box": "#999933", "raw_box": "#009E73",
         "welch": "#CC79A7", "bohman_box": "#56B4E9", "hann_gauss": "#D55E00", "box": "black", "band": "#0072B2"}
LW = {"truth": 1.7, "est": 1.0, "thin": 0.55, "heavy": 2.6}
SPEC_CMAP, SPEC_VLIM = "viridis", (10.0, 45.0)
TAPER_ALPHA = 0.25                     # recipe (b'): half a cosine on the first and last eighth of the record (a 25% Tukey taper)


def _recipe(N, Wn):
    """Recipe (b') at design half-bandwidth W = Wn/N: the 25% cosine taper and the lag window of a parabola whose
    half-power width is 2W (half-width sqrt(2) W). Wn = 0 gives the taper and no smoothing."""
    w = ss.unit_taper("tukey", N, alpha=TAPER_ALPHA)
    return w, (np.ones(N) if Wn == 0 else ss.parabolic_lag_window(N, np.sqrt(2) * Wn / N))


def _letters(axes, dx=-0.13, dy=1.03, size=8.5):
    """Bold lowercase panel letters at the top-left, outside the frame, in reading order."""
    for k, a in enumerate(np.ravel(axes)):
        a.text(dx, dy, f"({chr(97 + k)})", transform=a.transAxes, fontsize=size, fontweight="bold", va="bottom", ha="left")


def _save(fig, name):
    fig.savefig(FIG / name, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# ---------------------------------------------------------------- supplement: fig1, fig2, fig3, fig4
def fig1_kernels(N=256, NW=4, nfft=8192):
    W = NW / N
    Vk, _ = ss.dpss(N, NW)
    hb = ss.box_lag_window(N, W)
    rect, hann, boh = np.ones(N) / np.sqrt(N), ss.unit_taper("hann", N), ss.unit_taper("bohman", N)
    ks = [("ideal box, half-width W", ss.kernel_box(nfft, W), COLOR["box"], ":", 1.0),
          (f"multitaper, K = 2NW$-$1 = {2 * NW - 1}", ss.kernel_multitaper(Vk, nfft), COLOR["multitaper"], "-", 1.0),
          ("untapered periodogram, then box", ss.kernel_smoothed(rect, hb, nfft), COLOR["raw_box"], "-", 1.0),
          ("cosine-tapered periodogram, then parabola", ss.kernel_smoothed(*_recipe(N, NW), nfft), COLOR["smooth"], "-", 1.3),
          ("Hann periodogram, then box", ss.kernel_smoothed(hann, hb, nfft), COLOR["hann_box"], "-", 1.0),
          ("Bohman periodogram, then box", ss.kernel_smoothed(boh, hb, nfft), COLOR["bohman_box"], "-", 1.0)]
    fs = ss.signed_freq(nfft); o = np.argsort(fs)
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.5))
    for name, H, c, st, lw in ks:
        ax[0].plot(fs[o] * N, H[o] / N, color=c, ls=st, lw=lw, label=name)
        ax[1].plot(fs[o] * N, 10 * np.log10(np.maximum(H[o] / H.max(), 1e-15)), color=c, ls=st, lw=lw, label=name)
    ax[0].set(xlim=(-2.5 * NW, 2.5 * NW), xlabel="frequency offset (units of 1/N)", ylabel="kernel, linear (units of N)")
    ax[1].set(xlim=(-6 * NW, 6 * NW), ylim=(-100, 5), xlabel="frequency offset (units of 1/N)", ylabel="kernel (dB re peak)")
    h, l = ax[1].get_legend_handles_labels()
    _letters(ax); fig.tight_layout(rect=[0, 0.17, 1, 1]); fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0))
    _save(fig, "fig1_kernels.png")


def fig2_ar4(N=1024, NW=4, nfft=8192, seed=3):
    rng = np.random.default_rng(seed)
    W = NW / N
    x = ss.ar_process(ss.AR4, N, rng)
    f = np.arange(nfft // 2) / nfft
    truth = ss.ar_psd(ss.AR4, f)
    Vk, lam = ss.dpss(N, NW)
    hb = ss.box_lag_window(N, W)
    hann = ss.unit_taper("hann", N)
    ests = [("periodogram", ss.periodogram(x, nfft)[0], COLOR["periodogram"], LW["thin"], "-"),
            ("untapered periodogram, then box", ss.lag_window_estimate(x, hb, nfft)[0], COLOR["raw_box"], LW["est"], "-"),
            (f"multitaper, K = {2 * NW - 1}", ss.multitaper(x, Vk, nfft)[0], COLOR["multitaper"], LW["est"], "-"),
            (f"multitaper, K = {2 * NW - 2} (last taper dropped)", ss.multitaper(x, Vk[:, :-1], nfft)[0], COLOR["multitaper"], LW["est"], "--"),
            (f"multitaper, K = {2 * NW - 1}, adaptive weights", ss.multitaper_adaptive(x, Vk, lam, nfft)[0], COLOR["multitaper"], LW["est"], ":"),
            ("cosine-tapered periodogram, then parabola", ss.lag_window_estimate(x, _recipe(N, NW)[1], nfft, taper=_recipe(N, NW)[0])[0], COLOR["smooth"], LW["est"], "-")]
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.7), gridspec_kw={"width_ratios": [1.4, 1]})
    for name, S, c, lw, st in ests:
        ax[0].plot(f, 10 * np.log10(S[:nfft // 2]), color=c, lw=lw, ls=st, label=name)
        if name != "periodogram":
            ax[1].plot(f, 10 * np.log10(S[:nfft // 2]) - 10 * np.log10(truth), color=c, lw=lw, ls=st)
    ax[0].plot(f, 10 * np.log10(truth), color=COLOR["truth"], lw=LW["truth"], label="true PSD", zorder=10)
    ax[0].set(xlabel="frequency (cycles per sample)", ylabel="power (dB)", ylim=(-35, 55), xlim=(0, 0.5))
    ax[1].axhline(0, color=COLOR["truth"], lw=0.6); ax[1].set(xlabel="frequency (cycles per sample)", ylabel="estimate minus true PSD (dB)", ylim=(-10, 30), xlim=(0, 0.5))
    ax[0].legend(loc="upper center", bbox_to_anchor=(0.85, -0.25), ncol=3)
    _letters(ax); fig.tight_layout(); _save(fig, "fig2_ar4_leakage.png")


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
        add("multitaper, K = 2NW$-$1", ss.kernel_multitaper(Vk, nfft), ss.quadratic_matrix(N, f0, "multitaper", tapers=Vk))
        V2 = Vk[:, :-1] if K > 1 else Vk
        add("multitaper, K = 2NW$-$2", ss.kernel_multitaper(V2, nfft), ss.quadratic_matrix(N, f0, "multitaper", tapers=V2))
        St = ss.sine_tapers(N, K)
        add("multitaper, sine tapers", ss.kernel_multitaper(St, nfft), ss.quadratic_matrix(N, f0, "multitaper", tapers=St))
        hb = ss.box_lag_window(N, W)
        for tname, tp in [("untapered", rect), ("Hann", hann), ("Bohman", boh)]:
            add(f"{tname} periodogram, then box", ss.kernel_smoothed(tp, hb, nfft), ss.quadratic_matrix(N, f0, "lagwindow", h=hb, taper=tp))
        tk, hp = _recipe(N, NW)
        add("cosine taper, then parabola", ss.kernel_smoothed(tk, hp, nfft), ss.quadratic_matrix(N, f0, "lagwindow", h=hp, taper=tk))
        hg = ss.gaussian_lag_window(N, W / 2)
        add("Hann periodogram, then Gaussian", ss.kernel_smoothed(hann, hg, nfft), ss.quadratic_matrix(N, f0, "lagwindow", h=hg, taper=hann))
    for L in [16, 24, 32, 48, 64, 96, 128]:
        wl = ss.unit_taper("hann", L)
        add("Welch, Hann, 50% overlap", ss.kernel_wosa(N, L, .5, nfft, wl), ss.quadratic_matrix(N, f0, "welch", seg_len=L, overlap=.5, taper=wl))
    style = {"multitaper, K = 2NW$-$1": (COLOR["multitaper"], "o", "-"), "multitaper, K = 2NW$-$2": (COLOR["multitaper"], "s", "--"),
             "multitaper, sine tapers": (COLOR["multitaper"], "^", ":"), "untapered periodogram, then box": (COLOR["raw_box"], "o", "-"),
             "cosine taper, then parabola": (COLOR["smooth"], "o", "-"),
             "Hann periodogram, then box": (COLOR["hann_box"], "o", "-"), "Bohman periodogram, then box": (COLOR["bohman_box"], "s", "-"),
             "Hann periodogram, then Gaussian": (COLOR["hann_gauss"], "^", "-"), "Welch, Hann, 50% overlap": (COLOR["welch"], "D", "-")}
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.6))
    for name, pts in curves.items():
        p = np.array(sorted(pts)); c, m, st = style[name]
        for k, a in enumerate(ax):
            a.plot(p[:, 0], p[:, k + 1], color=c, marker=m, ls=st, ms=2.5, lw=0.9, label=name)
    for a in ax:
        a.axvline(8, color="0.7", lw=0.6, ls=":")
    ax[0].set(xlabel="half-power width (units of 1/N)", ylabel="equivalent degrees of freedom $\\nu$")
    ax[1].set(xlabel="half-power width (units of 1/N)", ylabel="kernel power beyond\n2$\\times$ half-power half-width", yscale="log")
    ax[2].set(xlabel="half-power width (units of 1/N)", ylabel="peak beyond 1.5$\\times$\nhalf-power half-width (dB)")
    h, l = ax[0].get_legend_handles_labels()
    _letters(ax, dx=-0.2); fig.tight_layout(rect=[0, 0.2, 1, 1], w_pad=1.2); fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0))
    _save(fig, "fig3_tradeoff.png")


def fig4_eeg(NW=3, win_s=4.0, fmax=30.0):
    """Seven minutes of the archive EEG: multitaper vs Hann-then-box spectrograms (no longer in the paper)."""
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
    fig, ax = plt.subplots(4, 1, figsize=(W2, 6.5), sharex=True)
    for a, img, title in zip(ax[:3], (pg, mt, sb), ("Hann periodogram (unsmoothed)", f"multitaper, NW = {NW}, K = {2 * NW - 1}",
                                                    f"Hann periodogram, then box of half-width {W * Fs:.2f} Hz")):
        im = a.imshow(img, aspect="auto", origin="lower", extent=[tt[0], tt[-1], f[keep][0], f[keep][-1]], vmin=vmin, vmax=vmax, cmap=SPEC_CMAP)
        a.set(ylabel="frequency (Hz)", title=title)
    dd = mt - sb
    im2 = ax[3].imshow(dd, aspect="auto", origin="lower", extent=[tt[0], tt[-1], f[keep][0], f[keep][-1]], vmin=-3, vmax=3, cmap="RdBu_r")
    ax[3].set(ylabel="frequency (Hz)", xlabel="time (s)", title="multitaper minus smoothed (dB)")
    fig.colorbar(im, ax=list(ax[:3]), fraction=.02, pad=.01, label="power (dB)"); fig.colorbar(im2, ax=ax[3], fraction=.02, pad=.01, label="dB")
    _letters(ax); _save(fig, "fig4_eeg_spectrograms.png")
    print(f"fig4: {len(tt)} epochs of {win_s:.0f} s; MT vs smoothed |diff| median {np.median(np.abs(dd)):.2f} dB, 95% {np.percentile(np.abs(dd), 95):.2f} dB")


# ---------------------------------------------------------------- fig5: the Slepian windows tile the box
def fig5_slepian_fill(N=256, NW=4, nfft=8192):
    """Running eigenvalue-weighted sum of |U_k|^2 converges to the box (Thomson 1982 eq. 8.3)."""
    W = NW / N
    V, lam = ss.dpss_all(N, W)
    fs = ss.signed_freq(nfft); o = np.argsort(fs)
    Uk = np.abs(np.fft.fft(V, nfft, axis=0)) ** 2 / N
    fig, ax = plt.subplots(2, 2, figsize=(W2, 4.4))
    t = np.arange(N)
    cols = plt.cm.viridis(np.linspace(0, 0.9, 8))
    for k in range(6):
        ax[0, 0].plot(t, V[:, k] + 0.25 * k, lw=0.9, color=cols[k], label=f"k = {k}")
    ax[0, 0].set(title="Slepian tapers, offset for display", xlabel="sample t", yticks=[], ylabel="taper (offset)")
    ax[0, 0].legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.25))
    for k in range(8):
        ax[0, 1].plot(fs[o] * N, Uk[o, k], lw=0.9, color=cols[k], label=f"k = {k}")
    ax[0, 1].axvspan(-NW, NW, color="0.92", lw=0)
    ax[0, 1].set(xlim=(-2 * NW, 2 * NW), title="spectral windows $|U_k(f)|^2$ (shading: $|f| \\leq W$)", xlabel="frequency offset (units of 1/N)", ylabel="$|U_k(f)|^2$ (unit-norm tapers)")
    ax[0, 1].legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.25))
    cum = np.cumsum(Uk * lam, axis=1)
    for K, c in [(1, cols[0]), (3, cols[2]), (5, cols[4]), (7, cols[6]), (9, cols[7]), (N, "black")]:
        ax[1, 0].plot(fs[o] * N, cum[o, K - 1] / (2 * NW), color=c, lw=1.0 if K < N else 1.6, label=f"first {K} tapers" if K < N else f"all N tapers (= the box)")
    ax[1, 0].plot(fs[o] * N, (np.abs(fs[o]) <= W) / (2 * W) / N, color="black", ls="--", lw=0.7, label="ideal box")
    ax[1, 0].set(xlim=(-2 * NW, 2 * NW), title="running sum $\\sum_{k<K} \\lambda_k |U_k(f)|^2 / 2NW$", xlabel="frequency offset (units of 1/N)", ylabel="kernel (unit area)")
    ax[1, 0].legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.32))
    ax[1, 1].semilogy(np.arange(N), np.maximum(1 - lam, 1e-16), "o-", ms=2, lw=0.7, color=COLOR["multitaper"])
    ax[1, 1].axvline(2 * NW, color=COLOR["hann_gauss"], ls="--", lw=0.8, label="2NW"); ax[1, 1].axvline(2 * NW - 1, color="0.5", ls=":", lw=0.8, label="K = 2NW$-$1")
    ax[1, 1].set(xlim=(-0.5, 2 * NW + 8), ylim=(1e-16, 2), xlabel="taper index k", ylabel="$1 - \\lambda_k$", title="energy outside the band, $1-\\lambda_k$")
    ax[1, 1].legend(loc="lower right")
    _letters(ax, dx=-0.2); fig.tight_layout(h_pad=2.6); _save(fig, "fig5_slepian_fill.png")


# ---------------------------------------------------------------- fig0: the motif and the sweep
def _peak_geometry(S, f):
    from scipy.signal import find_peaks
    pk, _ = find_peaks(S)
    out = []
    for p in pk:
        h = S[p] / 2
        lo = p
        while lo > 0 and S[lo] > h:
            lo -= 1
        hi = p
        while hi < len(S) - 1 and S[hi] > h:
            hi += 1
        out.append((f[p], S[p], f[hi] - f[lo]))
    return out


def _sweep_width(N, nfft, Ws, mc, rng, band):
    """Monte Carlo: RMS error (dB) of recipe (b') against the true AR(4) PSD, per design half-bandwidth W."""
    f = np.arange(nfft // 2) / nfft
    Sfull = ss.ar_psd(ss.AR4, np.arange(nfft) / nfft)
    S = Sfull[:nfft // 2]
    X = np.stack([ss.ar_process(ss.AR4, N, rng) for _ in range(mc)])
    sel = (f >= band[0]) & (f <= band[1])
    res = {k: [] for k in ("nu", "blur", "noise", "total", "total_band")}
    for Wn in Ws:
        tp, h = _recipe(N, Wn)
        est = np.stack([ss.lag_window_estimate(x, h, nfft, taper=tp)[0][:nfft // 2] for x in X])
        err = 10 * np.log10(est) - 10 * np.log10(S)
        Hk = ss.kernel_smoothed(tp, h, nfft)
        HS = np.fft.ifft(np.fft.fft(Sfull) * np.fft.fft(Hk)).real / nfft
        blur = 10 * np.log10(HS[:nfft // 2] / S)
        res["nu"].append(ss.dof_quadratic(ss.quadratic_matrix(N, 0.25, "lagwindow", h=h, taper=tp)))
        res["blur"].append(np.sqrt(np.mean(blur ** 2)))
        res["noise"].append(np.sqrt(np.mean(err.var(axis=0))))
        res["total"].append(np.sqrt(np.mean(err ** 2)))
        res["total_band"].append(np.sqrt(np.mean(err[:, sel] ** 2)))
    return {k: np.array(v) for k, v in res.items()}


def _motif_rows(ax_top, ax_bot, f, truth, ests, kers, titles, nus, bws, masses, N, fs, o, f1, S1, W_shade, colors, geometry=None):
    """The repeated motif: estimate over the truth (top), the estimator's kernel over the true peaks (bottom)."""
    d = (f - f1) * N; keep = np.abs(d) <= 40
    prof = 10 * np.log10(truth[keep] / S1)
    fc = f1 + 0.0147
    for j, (a, S, title, nu, bw, c) in enumerate(zip(ax_top, ests, titles, nus, bws, colors)):
        a.plot(f, 10 * np.log10(truth), color=COLOR["truth"], lw=LW["truth"], label="true PSD", zorder=5)
        a.plot(f, 10 * np.log10(S), color=c, lw=LW["thin"] if nu < 3 else LW["est"], label="estimate (one realization)")
        a.plot([fc - bw / 2 / N, fc + bw / 2 / N], [50, 50], color=COLOR["band"], lw=2.2, solid_capstyle="butt")
        if j == 0 and geometry is not None:
            pw, sep = geometry; x0 = 0.30
            a.plot([x0, x0 + pw / N], [-30, -30], color="0.3", lw=1.4, solid_capstyle="butt"); a.text(x0 + sep / N + 0.01, -30, f"peak width {pw:.0f}/N", fontsize=5.5, va="center", color="0.3")
            a.plot([x0, x0 + sep / N], [-39, -39], color="0.3", lw=1.4, solid_capstyle="butt"); a.text(x0 + sep / N + 0.01, -39, f"separation {sep:.0f}/N", fontsize=5.5, va="center", color="0.3")
        a.set(title=(f"{title}, $\\nu$ = {nu:.1f}" if "\n" in title else f"{title}\n$\\nu$ = {nu:.1f}"), xlabel="frequency (cycles per sample)", ylabel="power (dB)" if j == 0 else "", ylim=(-45, 57), xlim=(0, 0.5))
        if j == 0:
            a.legend(loc="upper right", fontsize=5.5)
    for j, (a, H, Wn, m, c) in enumerate(zip(ax_bot, kers, W_shade, masses, colors)):
        a.fill_between(d[keep], -80, prof, color="0.86", label="true spectrum (shape only)")
        if Wn > 0:
            a.axvspan(-Wn, Wn, color=COLOR["band"], alpha=0.12, lw=0, label="$|f| \\leq W$")
        a.plot(fs[o] * N, 10 * np.log10(np.maximum(H[o] / H.max(), 1e-15)), color=c, lw=0.9, label="kernel")
        ms = "< 0.001%" if m < 1e-5 else f"{100 * m:.2g}%"
        a.set(title=f"kernel: {ms} beyond 2W", xlim=(-40, 40), ylim=(-80, 5), xlabel="frequency offset (units of 1/N)", ylabel="kernel (dB re peak)" if j == 0 else "")
        if j == 0:
            a.legend(loc="upper right", fontsize=5)


def fig0_pedagogy(N=1024, nfft=4096, seed=11, W_right=4, W_wide=24, mc=300):
    """fig0b: four estimators (leaky+noisy, noisy, just right, too smooth) over their kernels; fig0a: the record and the sweep."""
    rng = np.random.default_rng(seed)
    x = ss.ar_process(ss.AR4, N, rng)
    f = np.arange(nfft // 2) / nfft
    truth = ss.ar_psd(ss.AR4, f)
    cosw, rect = ss.unit_taper("tukey", N, alpha=TAPER_ALPHA), np.ones(N) / np.sqrt(N)
    fs = ss.signed_freq(nfft); o = np.argsort(fs)
    peaks = _peak_geometry(truth, f)
    (f1, S1, w1), (f2, S2, w2) = peaks[:2]
    peak_width = 0.5 * (w1 + w2) * N; peak_sep = (f2 - f1) * N
    print(f"fig0: AR(4) peaks at f={f1:.4f}, {f2:.4f}; half-power width {peak_width:.1f}/N; separation {peak_sep:.1f}/N")
    W = W_right / N
    cols = [(rect, 0, "no taper\nleaky and noisy", COLOR["periodogram"]), (cosw, 0, "cosine taper\nstill noisy", COLOR["smooth"]),
            (cosw, W_right, f"cosine taper, parabola W = {W_right}/N\njust right", COLOR["smooth"]),
            (cosw, W_wide, f"cosine taper, parabola W = {W_wide}/N\ntoo smooth", COLOR["smooth"])]
    ests, kers, nus, bws, masses = [], [], [], [], []
    for tp, Wn, _, _ in cols:
        h = _recipe(N, Wn)[1]
        ests.append(ss.lag_window_estimate(x, h, nfft, taper=tp)[0][:nfft // 2])
        H = ss.kernel_smoothed(tp, h, nfft); kers.append(H)
        nus.append(ss.dof_quadratic(ss.quadratic_matrix(N, 0.25, "lagwindow", h=h, taper=tp)))
        st = ss.kernel_stats(H, W); bws.append(st["bw3"] * N); masses.append(st["mass_out_2W"])
    Ws = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 28, 32])
    band = (f1 - 0.02, f2 + 0.02)
    sw = _sweep_width(N, nfft, Ws, mc, np.random.default_rng(seed + 1), band)

    # fig0b: the motif
    fig, ax = plt.subplots(2, 4, figsize=(W2, 4.0))
    _motif_rows(ax[0], ax[1], f, truth, ests, kers, [c[2] for c in cols], nus, bws, masses, N, fs, o, f1, S1, [c[1] for c in cols], [c[3] for c in cols], geometry=(peak_width, peak_sep))
    _letters(ax); fig.tight_layout(w_pad=0.6, h_pad=1.5); _save(fig, "fig0b_estimates_kernels.png")

    # fig0a: the record and the sweep
    fig, (ax_ts, ax_err) = plt.subplots(1, 2, figsize=(W2, 2.6), gridspec_kw={"width_ratios": [1.1, 1]})
    t = np.arange(N); env = cosw / cosw.max(); amp = np.abs(x).max()
    ax_ts.plot(t, x, color="0.6", lw=0.45, label="record $x_t$")
    ax_ts.plot(t, env * x, color=COLOR["smooth"], lw=0.5, label="tapered record")
    ax_ts.plot(t, amp * env, color="black", ls="--", lw=0.8, label="cosine taper (25%), scaled to the record's peak amplitude"); ax_ts.plot(t, -amp * env, color="black", ls="--", lw=0.8)
    ax_ts.set(xlim=(0, N - 1), xlabel="sample t", ylabel="$x_t$", title=f"the record: N = {N} samples of the AR(4) process")
    ax_ts.legend(loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=1)
    ax_err.plot(Ws, sw["total"], color="black", marker="o", ms=2.2, lw=1.4, label="total RMS error, whole band")
    ax_err.plot(Ws, sw["total_band"], color="black", marker="o", ms=2.2, lw=1.0, ls="--", label=f"total RMS error, peak region {band[0]:.2f}$-${band[1]:.2f}")
    ax_err.plot(Ws, sw["noise"], color=COLOR["multitaper"], lw=1.2, label="noise: standard deviation of the estimate")
    ax_err.plot(Ws, sw["blur"], color=COLOR["hann_gauss"], lw=1.2, label="bias: blur by the kernel")
    for Wn, lab, dx, dy in [(0, "too noisy", 1.0, -1.7), (W_right, "just right", 0.4, 1.9), (W_wide, "too smooth", 1.0, -1.6)]:
        i = np.argmin(np.abs(Ws - Wn)); ax_err.plot(Wn, sw["total"][i], "o", ms=7, mfc="none", mec=COLOR["hann_gauss"], mew=1.4)
        ax_err.annotate(f"{lab}\n$\\nu$ = {sw['nu'][i]:.0f}", (Wn, sw["total"][i]), (Wn + dx, sw["total"][i] + dy), fontsize=6.5, color=COLOR["hann_gauss"])
    for xv, lab in [(peak_width / 2, "2W = peak width"), (peak_sep / 2, "2W = peak separation")]:
        ax_err.axvline(xv, color="0.6", ls=":", lw=0.8); ax_err.text(xv + 0.3, 7.05, lab, fontsize=6, color="0.4", va="top")
    ax_err.set(xlim=(-0.5, Ws[-1] + 0.5), ylim=(0, 7.2), xlabel="half-bandwidth W (units of 1/N)", ylabel="RMS error vs true PSD (dB)",
               title=f"cosine taper, then parabola: error vs width ({mc} realizations)")
    ax_err.legend(loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=1)
    _letters([ax_ts, ax_err], dx=-0.16); fig.tight_layout(); _save(fig, "fig0a_record_sweep.png")
    i_r, i_w = np.argmin(np.abs(Ws - W_right)), np.argmin(np.abs(Ws - W_wide))
    print(f"fig0: nu = {[round(float(v), 1) for v in nus]}; bws = {[round(float(v), 1) for v in bws]}; RMS dB error whole band / peak region: "
          f"W=0 {sw['total'][0]:.2f}/{sw['total_band'][0]:.2f}, W={W_right} {sw['total'][i_r]:.2f}/{sw['total_band'][i_r]:.2f}, "
          f"W={W_wide} {sw['total'][i_w]:.2f}/{sw['total_band'][i_w]:.2f}; whole-band minimum at W={Ws[np.argmin(sw['total'])]}/N, "
          f"peak-region minimum at W={Ws[np.argmin(sw['total_band'])]}/N")
    return sw, Ws


def fig8_everyday_motif(N=1024, NW=4, nfft=4096, seed=11, seg_len=192):
    """The same motif for the everyday estimators on the fig0 record: raw+box, Thomson K, Hann+box, Welch."""
    rng = np.random.default_rng(seed)
    x = ss.ar_process(ss.AR4, N, rng)
    f = np.arange(nfft // 2) / nfft; truth = ss.ar_psd(ss.AR4, f)
    fs = ss.signed_freq(nfft); o = np.argsort(fs)
    (f1, S1, _), _ = _peak_geometry(truth, f)[:2]
    W = NW / N; K = 2 * NW - 1
    Vk, _ = ss.dpss(N, NW)
    hb = ss.box_lag_window(N, W); rect = np.ones(N) / np.sqrt(N); cosw, hp = _recipe(N, NW)
    step = (N - seg_len) // 8; hann_seg = ss.unit_taper("hann", seg_len); nseg = len(range(0, N - seg_len + 1, step))
    cols = [(f"no taper, then box W = {NW}/N", ss.lag_window_estimate(x, hb, nfft)[0], dict(method="lagwindow", h=hb, taper=rect), COLOR["raw_box"]),
            (f"multitaper, K = {K}, NW = {NW}", ss.multitaper(x, Vk, nfft)[0], dict(method="multitaper", tapers=Vk), COLOR["multitaper"]),
            (f"cosine taper, then parabola W = {NW}/N", ss.lag_window_estimate(x, hp, nfft, taper=cosw)[0], dict(method="lagwindow", h=hp, taper=cosw), COLOR["smooth"]),
            (f"Welch, {nseg} Hann segments", ss.welch_sliding(x, hann_seg, nfft, step=step, overhang=False)[0], dict(method="welch", taper=hann_seg, step=step), COLOR["welch"])]
    ests, kers, nus, bws, masses = [], [], [], [], []
    for _, S, q, _ in cols:
        m = q.pop("method")
        Q0 = ss.quadratic_matrix(N, 0.0, m, **q); Qf = ss.quadratic_matrix(N, 0.25, m, **q)
        H = ss.kernel_quadratic(Q0, nfft); st = ss.kernel_stats(H, W)
        ests.append(S[:nfft // 2]); kers.append(H); nus.append(ss.dof_quadratic(Qf)); bws.append(st["bw3"] * N); masses.append(st["mass_out_2W"])
    fig, ax = plt.subplots(2, 4, figsize=(W2, 4.0))
    _motif_rows(ax[0], ax[1], f, truth, ests, kers, [c[0] for c in cols], nus, bws, masses, N, fs, o, f1, S1, [NW, NW, NW, 0], [c[3] for c in cols])
    _letters(ax); fig.tight_layout(w_pad=0.6, h_pad=1.5); _save(fig, "fig8_everyday_motif.png")
    err = [np.median(np.abs(10 * np.log10(S) - 10 * np.log10(truth))) for S in ests]
    print(f"fig8: nu = {[round(float(v), 1) for v in nus]}; bw = {[round(float(v), 1) for v in bws]}; mass beyond 2W = {[round(float(v), 4) for v in masses]}; median |dB error| vs truth = {[round(float(v), 1) for v in err]}; nseg {nseg} step {step}")


# ---------------------------------------------------------------- fig6: three ways to build a taper bank
def fig6_taper_banks(N=256, NW=4, nfft=8192, seg_len=64, full=False):
    """Compact (2 rows: the bank, its eigen-weight ladder) or full (4 rows: bank, eigen-tapers, ladder, kernel)."""
    W = NW / N; K = 2 * NW - 1
    t = np.arange(N)
    hann_seg = ss.unit_taper("hann", seg_len); cosw, hp = _recipe(N, NW)
    V, lam = ss.dpss_all(N, W)
    step = seg_len // 2
    Qs = [("average segments\n(Welch: Hann, L = N/4, 50%)", ss.quadratic_matrix(N, 0.0, "welch", taper=hann_seg, step=step).real, COLOR["welch"]),
          ("smooth one periodogram\n(cosine taper, parabola)", ss.quadratic_matrix(N, 0.0, "lagwindow", h=hp, taper=cosw).real, COLOR["smooth"]),
          ("slide a sinc window\n(= untapered, box)", ss.quadratic_matrix(N, 0.0, "lagwindow", h=ss.box_lag_window(N, W)).real, COLOR["raw_box"]),
          (f"Thomson multitaper\n(K = {K} Slepians, equal weights)", ss.quadratic_matrix(N, 0.0, "multitaper", tapers=V[:, :K]).real, COLOR["multitaper"])]
    fs = ss.signed_freq(nfft); o = np.argsort(fs)
    nrows = 4 if full else 2
    fig, ax = plt.subplots(nrows, 4, figsize=(W2, 1.75 * nrows + 0.5))
    kcols = plt.cm.viridis(np.linspace(0, 0.85, 4))
    for j, (title, Q, col) in enumerate(Qs):
        a = ax[0, j]
        if j == 0:
            for i, s in enumerate(range(0, N - seg_len + 1, step)):
                u = np.zeros(N); u[s:s + seg_len] = hann_seg; a.plot(t, u, lw=0.8, color=plt.cm.viridis(i / 7))
            a.set(title=title + "\nbank: one window at 7 offsets")
        elif j == 1:
            for i, fk in enumerate([0, 1, 2, 3]):
                a.plot(t, cosw * np.cos(2 * np.pi * fk / N * t) + 0.22 * (3 - i), lw=0.8, color=kcols[i], label=f"f' = {fk}/N")
            a.set(title=title + "\nbank: one taper modulated to f'"); a.legend(loc="upper right", fontsize=5, ncol=2)
        elif j == 2:
            w = ss.sinc_window(8 * N, W)
            for i, s in enumerate([-N // 2, 0, N // 2, N]):
                u = np.zeros(N); lo, hi = max(0, s - 4 * N), min(N, s + 4 * N)
                u[lo:hi] = w[lo - (s - 4 * N):hi - (s - 4 * N)]
                a.plot(t, u / w.max() + 0.6 * (3 - i), lw=0.8, color=kcols[i])
            a.set(title=title + "\nbank: a sinc at every offset")
        else:
            for i in range(4):
                a.plot(t, V[:, i] + 0.25 * (3 - i), lw=0.8, color=kcols[i], label=f"k = {i}")
            a.set(title=title + "\nbank: Slepian tapers k = 0..3"); a.legend(loc="upper right", fontsize=5, ncol=2)
        a.set(xlabel="sample t", yticks=[], xlim=(0, N - 1), ylabel="amplitude (curves offset)" if j == 0 else "")
        c, U = ss.eigen_tapers(Q)
        if j == 3:
            U = V
        r = 1
        if full:
            a = ax[1, j]
            for i in range(4):
                a.plot(t, U[:, i] * np.sign(U[N // 2 + i, i] if abs(U[N // 2 + i, i]) > 1e-6 else 1) + 0.25 * (3 - i), lw=0.8, color=kcols[i], label=f"k = {i}")
            a.set(title="eigen-tapers of Q, k = 0..3" if j != 3 else "eigen-tapers: the Slepians", xlabel="sample t", yticks=[], xlim=(0, N - 1), ylabel="amplitude (curves offset)" if j == 0 else "")
            if j == 0:
                a.legend(loc="upper right", fontsize=5, ncol=2)
            r = 2
        a = ax[r, j]
        cc = c / c.sum(); kmax = 12
        a.bar(np.arange(kmax), cc[:kmax], color=col)
        a.set(title=f"eigen-weight ladder: $\\nu$ = {ss.dof_from_weights(c):.1f}", xlabel="taper index k", ylabel="weight $c_k$ (normalized)" if j == 0 else "", ylim=(0, 0.45))
        H = ss.kernel_quadratic(Q, nfft); st = ss.kernel_stats(H, W)
        if full:
            a = ax[3, j]
            a.plot(fs[o] * N, 10 * np.log10(np.maximum(H[o] / H.max(), 1e-15)), color=col, lw=0.9)
            a.axvspan(-NW, NW, color=COLOR["band"], alpha=0.12, lw=0)
            a.set(title=f"kernel: width {st['bw3'] * N:.1f}/N,\n{100 * st['mass_out_2W']:.1f}% beyond 2W", xlim=(-24, 24), ylim=(-80, 5), xlabel="frequency offset (units of 1/N)", ylabel="kernel (dB re peak)" if j == 0 else "")
        print(f"fig6 {title.splitlines()[0]}: dof {ss.dof_from_weights(c):.1f}; bw {st['bw3']*N:.2f}/N; mass beyond 2W {100*st['mass_out_2W']:.2f}%; top weights {np.round(cc[:8], 3)}")
    _letters(ax); fig.tight_layout(w_pad=0.6, h_pad=1.6); _save(fig, "fig6_taper_banks_full.png" if full else "fig6_taper_banks.png")


# ---------------------------------------------------------------- fig9 / fig7: three routes on one EEG epoch
def fig7_three_routes(NW=4, win_s=8.0, start_s=200.0, fmax=40.0, L_sinc_mult=16, seg_len=192):
    """fig9 (graphical abstract): the exact trio, their differences on a log scale, and convergence in the window length.
    fig7: the everyday trio and their differences."""
    import scipy.io as sio
    p = ROOT / "legacy_matlab" / "InterestingSignal.mat"
    if not p.exists():
        print("fig7/fig9 skipped: legacy_matlab/InterestingSignal.mat not found"); return
    d = sio.loadmat(p); sig = d["s"].ravel().astype(float); Fs = float(d["Fs"].ravel()[0])
    N = int(win_s * Fs); nfft = 4 * N; W = NW / N; K = 2 * NW - 1
    st0 = int(start_s * Fs); x = sig[st0:st0 + N]; x = x - x.mean()
    f = np.arange(nfft // 2) / nfft * Fs; keep = f <= fmax
    V, lam = ss.dpss_all(N, W)
    hb = ss.box_lag_window(N, W); cosw, hp = _recipe(N, NW)
    S_mt_all = ss.multitaper(x, V, nfft, weights=lam)[0][:nfft // 2][keep]
    S_box = ss.lag_window_estimate(x, hb, nfft)[0][:nfft // 2][keep]
    Ls = [N, 2 * N, 4 * N, 8 * N, 16 * N]
    S_sinc = {L: ss.welch_sliding(x, ss.sinc_window(L, W), nfft)[0][:nfft // 2][keep] for L in Ls}
    dB = lambda S: 10 * np.log10(S)
    d_box = np.abs(dB(S_box) - dB(S_mt_all)); d_sinc = {L: np.abs(dB(S_sinc[L]) - dB(S_mt_all)) for L in Ls}
    # fig9
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.5), gridspec_kw={"width_ratios": [1.6, 1, 0.75]})
    a = ax[0]
    a.plot(f[keep], dB(S_mt_all), color="0.75", lw=LW["heavy"], label="multitaper: all N Slepian tapers, weights $\\lambda_k$")
    a.plot(f[keep], dB(S_box), color=COLOR["raw_box"], lw=1.0, label="smooth: untapered periodogram, box of half-width W")
    a.plot(f[keep], dB(S_sinc[16 * N]), color=COLOR["welch"], lw=0.8, ls="--", label=f"average: sinc window of length 16N slid one sample at a time")
    a.set(xlabel="frequency (Hz)", ylabel="power (dB)", xlim=(0, fmax)); a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=1)
    a = ax[1]
    a.semilogy(f[keep], np.maximum(d_box, 1e-16), color=COLOR["raw_box"], lw=0.8, label="smooth minus multitaper")
    a.semilogy(f[keep], np.maximum(d_sinc[16 * N], 1e-16), color=COLOR["welch"], lw=0.8, label="average (16N) minus multitaper")
    a.set(xlabel="frequency (Hz)", ylabel="|difference| (dB)", xlim=(0, fmax), ylim=(1e-15, 1)); a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=1)
    a = ax[2]
    mx = [d_sinc[L].max() for L in Ls]
    a.loglog(np.array(Ls) / N, mx, "o-", color=COLOR["welch"], ms=3, lw=1.0, label="max |difference|")
    a.loglog(np.array(Ls) / N, mx[1] * (Ls[1] / np.array(Ls)), color="0.5", ls=":", lw=0.8, label="1/L")
    a.set(xlabel="window length L / N", ylabel="max |average $-$ multitaper| (dB)", xticks=[1, 2, 4, 8, 16], xticklabels=["1", "2", "4", "8", "16"]); a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=1)
    _letters(ax); fig.tight_layout(w_pad=0.8); _save(fig, "fig9_graphical_abstract.png")
    print("fig9: max|box-mt| = %.2g dB; max|sinc-mt| by L/N: %s" % (d_box.max(), {L // N: round(float(d_sinc[L].max()), 4) for L in Ls}))
    # fig7: the everyday trio
    step = (N - seg_len) // 8; hann_seg = ss.unit_taper("hann", seg_len); nseg = len(range(0, N - seg_len + 1, step))
    every = [(f"multitaper: K = {K} Slepian tapers, equal weights", ss.multitaper(x, V[:, :K], nfft)[0], COLOR["multitaper"], LW["est"]),
             ("smooth: cosine-tapered periodogram, parabola 2W wide at half power", ss.lag_window_estimate(x, hp, nfft, taper=cosw)[0], COLOR["smooth"], LW["est"]),
             (f"average: Welch, {nseg} Hann segments of {seg_len / Fs:.1f} s, step {step / Fs:.2f} s", ss.welch_sliding(x, hann_seg, nfft, step=step, overhang=False)[0], COLOR["welch"], LW["est"])]
    dofs = [ss.dof_quadratic(ss.quadratic_matrix(N, 0.25, "multitaper", tapers=V[:, :K])),
            ss.dof_quadratic(ss.quadratic_matrix(N, 0.25, "lagwindow", h=hp, taper=cosw)),
            ss.dof_quadratic(ss.quadratic_matrix(N, 0.25, "welch", taper=hann_seg, step=step))]
    bws = [ss.kernel_stats(ss.kernel_multitaper(V[:, :K], nfft))["bw3"] * Fs, ss.kernel_stats(ss.kernel_smoothed(cosw, hp, nfft))["bw3"] * Fs,
           ss.kernel_stats(ss.kernel_wosa(N, seg_len, 0.5, nfft, hann_seg))["bw3"] * Fs]
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.5), gridspec_kw={"width_ratios": [1.5, 1]})
    a = ax[0]
    for (name, S, c, lw), nu in zip(every, dofs):
        a.plot(f[keep], dB(S[:nfft // 2][keep]), color=c, lw=lw, label=f"{name} ($\\nu$ = {nu:.1f})")
    a.set(xlabel="frequency (Hz)", ylabel="power (dB)", xlim=(0, fmax)); a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=1)
    a = ax[1]; ref = dB(every[0][1][:nfft // 2][keep]); out = {}
    for name, S, c, lw in every[1:]:
        dd = dB(S[:nfft // 2][keep]) - ref
        out[name.split(":")[0]] = (np.median(np.abs(dd)), np.percentile(np.abs(dd), 95))
        a.plot(f[keep], dd, color=c, lw=0.8, label=f"{name.split(':')[0]} minus multitaper: median |diff| {np.median(np.abs(dd)):.1f} dB")
    a.axhline(0, color="black", lw=0.5)
    a.set(xlabel="frequency (Hz)", ylabel="difference from\nmultitaper (dB)", xlim=(0, fmax), ylim=(-15, 15)); a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=1)
    _letters(ax); fig.tight_layout(w_pad=0.8); _save(fig, "fig7_everyday_eeg.png")
    print("fig7:", out, "dofs", np.round(dofs, 1), "bws(Hz)", np.round(bws, 2), "nseg", nseg, "step", step, "expected log-noise std (dB):", np.round(4.34 * np.sqrt(2 / np.array(dofs)), 2))


# ---------------------------------------------------------------- fig10 / fig11: seizure clips (ELROND; de-identified BDSP recordings, not in this repo)
ELROND_SEG = pathlib.Path.home() / "ELROND" / "Data" / "segments"
EEG_CASES = {  # tag: (file, [(seizure onset s, offset s), ...]) from labels/mbw_mbw/seizure_annotations.json (annotator mbw)
    "A": ("sub-I0003175292344_20170823183337.mat", [(292.0, 366.0)]),
    "C": ("sub-I0003175080331_ses-84_445.mat", [(49.0, 183.0), (298.0, 441.0)]),
}


def ictal_channel(X, Fs, t_on, t_off, lo=2.0, hi=20.0):
    """Index of the channel with the largest ictal-over-pre-ictal power rise in [lo, hi] Hz, and the rises in dB."""
    def bp(xc, t0, t1):
        s0, s1 = int(t0 * Fs), int(t1 * Fs); sg = xc[s0:s1] - xc[s0:s1].mean()
        P = np.abs(np.fft.fft(sg * ss.unit_taper("hann", len(sg)), 8 * len(sg))) ** 2; ff = np.arange(len(P)) / len(P) * Fs
        return P[(ff >= lo) & (ff <= hi)].mean()
    rise = np.array([10 * np.log10(bp(X[i], t_on + 5, min(t_off, t_on + 40)) / bp(X[i], max(0, t_on - 60), t_on - 5)) for i in range(X.shape[0])])
    return int(np.argmax(rise)), rise


def _load_elrond(path):
    import scipy.io as sio
    try:
        d = sio.loadmat(str(path), squeeze_me=False)
        X = np.asarray(d["data"], float); fs = float(np.asarray(d["Fs"]).flatten()[0])
        ch = [str(np.asarray(c).flatten()[0]).strip().upper() for c in np.asarray(d["channels"]).flatten()]
    except NotImplementedError:
        import h5py
        with h5py.File(path, "r") as h:
            X = np.ascontiguousarray(np.asarray(h["data"], dtype=np.float64).T)
            ch = ["".join(chr(int(v)) for v in np.asarray(h[ref][()]).flatten()).strip().upper() for ref in np.asarray(h["channels"]).flatten()]
            fs = float(np.asarray(h["Fs"]).flatten()[0])
    if X.shape[0] > X.shape[1]:
        X = X.T
    return X, fs, ch


def _pretty_channel(name):
    return name[0] + name[1:].lower() if len(name) > 1 and name[1:].isalpha() else name.capitalize()


def fig10_eeg_seizure(case="A", win_s=2.0, step_s=1.0, NW=2, fmax=30.0, t_pre=None, t_ictal=None, out="fig10_eeg_seizure.png", compact=False):
    """A 10-min scalp EEG clip containing seizures: the raw trace of the most involved channel, spectrograms by the
    cosine-tapered periodogram, multitaper (NW, K = 2NW-1) and recipe (b') at the same W; spectra at a pre-ictal and an
    ictal instant by all three plus Welch. compact=True: trace and three spectrograms only."""
    fname, ivals = EEG_CASES[case]; t_on, t_off = ivals[0]
    p = ELROND_SEG / fname
    if not p.exists():
        print(f"fig10 skipped: {p} not found"); return
    X, Fs, ch = _load_elrond(p)
    X = X - X.mean(axis=0, keepdims=True)
    N = int(win_s * Fs); step = int(step_s * Fs); nfft = 4 * N; W = NW / N; K = 2 * NW - 1
    f = np.arange(nfft // 2) / nfft * Fs; keep = f <= fmax
    Vk, _ = ss.dpss(N, NW); cosw, hp = _recipe(N, NW)
    L = int(round(1.4 / (2 * W))); hann_seg = ss.unit_taper("hann", L); wstep = L // 2
    starts = list(range(0, X.shape[1] - N + 1, step)); tt = (np.array(starts) + N / 2) / Fs
    ic, rise = ictal_channel(X, Fs, t_on, t_off); x = X[ic]; chn = _pretty_channel(ch[ic])
    print(f"fig10 case {case}: channel {ch[ic]} (ictal rise {rise[ic]:.1f} dB), Fs={Fs:.0f}, N={N}, W={W * Fs:.2f} Hz, K={K}, Welch L={L} ({L / Fs:.2f} s), {len(starts)} windows")
    pg, mt, sb, wl = [], [], [], []
    for s0 in starts:
        seg = x[s0:s0 + N]; seg = seg - seg.mean()
        pg.append(ss.periodogram(seg, nfft, taper=cosw)[0][:nfft // 2][keep])
        mt.append(ss.multitaper(seg, Vk, nfft)[0][:nfft // 2][keep])
        sb.append(ss.lag_window_estimate(seg, hp, nfft, taper=cosw)[0][:nfft // 2][keep])
        wl.append(ss.welch_sliding(seg, hann_seg, nfft, step=wstep, overhang=False)[0][:nfft // 2][keep])
    pg, mt, sb, wl = (10 * np.log10(np.array(a).T + 1e-9) for a in (pg, mt, sb, wl))
    nus = [2.0, ss.dof_quadratic(ss.quadratic_matrix(N, 0.25, "multitaper", tapers=Vk)),
           ss.dof_quadratic(ss.quadratic_matrix(N, 0.25, "lagwindow", h=hp, taper=cosw)),
           ss.dof_quadratic(ss.quadratic_matrix(N, 0.25, "welch", taper=hann_seg, step=wstep))]
    t_pre = t_pre if t_pre is not None else t_on - 45
    t_ictal = t_ictal if t_ictal is not None else t_on + 0.35 * (t_off - t_on)
    vmin, vmax = SPEC_VLIM
    t = np.arange(len(x)) / Fs
    bws = [ss.kernel_stats(H)["bw3"] * Fs for H in (np.abs(np.fft.fft(cosw, 16 * N)) ** 2, ss.kernel_multitaper(Vk, 16 * N), ss.kernel_smoothed(cosw, hp, 16 * N))]
    labels = [(pg, f"cosine-tapered periodogram: {win_s:.0f} s by {bws[0]:.1f} Hz, $\\nu$ = 2"), (mt, f"multitaper, K = {K}: {win_s:.0f} s by {bws[1]:.1f} Hz, $\\nu$ = {nus[1]:.0f}"),
              (sb, f"cosine taper, then parabola: {win_s:.0f} s by {bws[2]:.1f} Hz, $\\nu$ = {nus[2]:.1f}")]
    dd = mt - sb
    if compact:
        fig, axs = plt.subplots(4, 1, figsize=(W2, 5.8), gridspec_kw={"height_ratios": [0.7, 1, 1, 1], "hspace": 0.45}, sharex=True)
        axs[0].plot(t, x, color="black", lw=0.25); axs[0].set_ylim(-500, 500)
        for on, off in ivals:
            axs[0].axvspan(on, off, color=COLOR["hann_gauss"], alpha=0.15, lw=0)
        axs[0].set(ylabel=f"{chn} ($\\mu$V)", title=f"scalp EEG, channel {chn}, common-average reference; {len(ivals)} seizures shaded")
        for a, (img, title) in zip(axs[1:], labels):
            im = a.imshow(img, aspect="auto", origin="lower", extent=[tt[0], tt[-1], f[keep][0], f[keep][-1]], vmin=vmin, vmax=vmax, cmap=SPEC_CMAP)
            a.set(ylabel="frequency (Hz)", title=title)
        axs[-1].set(xlabel="time (s)", xlim=(0, t[-1]))
        cb = fig.colorbar(im, ax=list(axs[1:]), fraction=0.02, pad=0.015); cb.set_label("power (dB)")
        _letters(axs, dx=-0.07); _save(fig, out)
        print(f"fig10 case {case} (compact): channel {ch[ic]}; nus {np.round(nus, 1)}; multitaper vs recipe (b') median |diff| {np.median(np.abs(dd)):.2f} dB, 95% {np.percentile(np.abs(dd), 95):.2f} dB")
        return
    fig = plt.figure(figsize=(W2, 6.6))
    gs = fig.add_gridspec(4, 2, width_ratios=[3, 1.25], height_ratios=[0.75, 1, 1, 1], hspace=0.62, wspace=0.42)
    a = fig.add_subplot(gs[0, 0]); axes_all = [a]
    a.plot(t, x, color="black", lw=0.25); a.set_ylim(-500, 500)
    for on, off in ivals:
        a.axvspan(on, off, color=COLOR["hann_gauss"], alpha=0.15, lw=0)
    for tq, c in [(t_pre, COLOR["multitaper"]), (t_ictal, COLOR["hann_gauss"])]:
        a.axvline(tq, color=c, ls="--", lw=0.8)
    a.set(xlim=(0, t[-1]), ylabel=f"{chn} ($\\mu$V)", title=f"scalp EEG, channel {chn}, common-average reference; seizure shaded", xlabel="time (s)")
    a = fig.add_subplot(gs[0, 1]); axes_all.append(a); z0 = int(t_ictal * Fs); zl = int(6 * Fs)
    a.plot(np.arange(zl) / Fs, x[z0:z0 + zl], color=COLOR["hann_gauss"], lw=0.5)
    a.set(title="ictal rhythm, 6 s", xlabel="time (s)", ylabel="$\\mu$V")
    sp_axes = []
    for r, (img, title) in enumerate(labels):
        a = fig.add_subplot(gs[r + 1, 0]); sp_axes.append(a)
        im = a.imshow(img, aspect="auto", origin="lower", extent=[tt[0], tt[-1], f[keep][0], f[keep][-1]], vmin=vmin, vmax=vmax, cmap=SPEC_CMAP)
        for tq in (t_pre, t_ictal):
            a.axvline(tq, color="white", ls="--", lw=0.7)
        a.set(ylabel="frequency (Hz)", title=title, xlim=(tt[0], tt[-1]))
        if r == 2:
            a.set(xlabel="time (s)")
    side = []
    for r, (tq, name) in enumerate([(t_pre, "pre-ictal"), (t_ictal, "ictal")]):
        a = fig.add_subplot(gs[r + 1, 1]); side.append(a); j = int(np.argmin(np.abs(tt - tq)))
        a.plot(f[keep], pg[:, j], color=COLOR["periodogram"], lw=0.5, label="cosine-tapered periodogram ($\\nu$ = 2)")
        a.plot(f[keep], mt[:, j], color=COLOR["multitaper"], lw=1.3, label=f"multitaper ($\\nu$ = {nus[1]:.0f})")
        a.plot(f[keep], sb[:, j], color=COLOR["smooth"], lw=1.0, label=f"cosine taper, then parabola ($\\nu$ = {nus[2]:.1f})")
        a.plot(f[keep], wl[:, j], color=COLOR["welch"], lw=0.8, label=f"Welch, {L / Fs:.2f}-s segments ($\\nu$ = {nus[3]:.1f})")
        a.set(title=f"{name}, t = {tq:.0f} s", xlabel="frequency (Hz)", ylabel="power (dB)", xlim=(0, fmax))
        if r == 1:
            lo_, hi_ = a.get_ylim(); a.set_ylim(lo_, hi_ + 0.55 * (hi_ - lo_)); a.legend(loc="upper right", fontsize=5)
    a = fig.add_subplot(gs[3, 1]); side.append(a)
    a.hist(dd.ravel(), bins=np.linspace(-8, 8, 81), color="0.5")
    a.set(title=f"median |diff| {np.median(np.abs(dd)):.1f} dB", xlabel="multitaper minus smoothed (dB)", ylabel="pixels")
    cb = fig.colorbar(im, ax=sp_axes, fraction=0.02, pad=0.015); cb.set_label("power (dB)")
    order = [axes_all[0], axes_all[1], sp_axes[0], side[0], sp_axes[1], side[1], sp_axes[2], side[2]]
    _letters(order, dx=-0.09)
    _save(fig, out)
    print(f"fig10 case {case}: nus {np.round(nus, 1)}; multitaper vs recipe (b') median |diff| {np.median(np.abs(dd)):.2f} dB, 95% {np.percentile(np.abs(dd), 95):.2f} dB; pre {t_pre:.0f} s, ictal {t_ictal:.0f} s")
    # the ictal rhythm in the ictal window: harmonics located on the smoothed estimate, widths measured on the unsmoothed periodogram
    from scipy.signal import find_peaks
    j = int(np.argmin(np.abs(tt - t_ictal))); seg = x[starts[j]:starts[j] + N]; seg = seg - seg.mean(); nf = 16 * N
    ff = np.arange(nf // 2) / nf * Fs; sel = (ff >= 3) & (ff <= 30)
    Ssm = ss.lag_window_estimate(seg, hp, nf, taper=cosw)[0][:nf // 2]; P = ss.periodogram(seg, nf, taper=cosw)[0][:nf // 2]
    pk, _ = find_peaks(10 * np.log10(Ssm[sel]), prominence=4); fp = ff[sel][pk]; widths = []
    for fq in fp:
        near = np.flatnonzero(np.abs(ff - fq) <= 1.0); i = near[np.argmax(P[near])]; lo = hi = i
        while lo > 0 and P[lo] > P[i] / 2:
            lo -= 1
        while hi < len(P) - 1 and P[hi] > P[i] / 2:
            hi += 1
        widths.append((ff[i], ff[hi] - ff[lo]))
    print(f"fig10 case {case}: ictal window at {tt[j]:.0f} s: harmonics on the smoothed estimate at {np.round(fp, 1)} Hz, spacings {np.round(np.diff(fp), 1)} Hz; "
          f"on the unsmoothed periodogram the nearest peaks are at {np.round([w[0] for w in widths], 1)} Hz with half-power widths {np.round([w[1] for w in widths], 2)} Hz "
          f"(resolution of the tapered {win_s:.0f}-s window: {ss.kernel_stats(np.abs(np.fft.fft(cosw, nf)) ** 2)['bw3'] * Fs:.2f} Hz)")


# ---------------------------------------------------------------- fig12: sleep spindles, time resolution against frequency resolution
SLEEP_DIR = pathlib.Path.home() / "GithubRepos" / "sleep-yoda" / "dev" / "standardization" / "output"   # de-identified BDSP sleep recordings, not in this repo
SLEEP_SETTINGS = [(1.0, 2.0), (2.0, 2.0), (4.0, 1.0)]                                                  # (window length in s, full bandwidth 2W in Hz)


def _spindles(x, fs, band=(11.0, 16.0), factor=2.0, min_s=0.5, max_s=3.0):
    """Bursts of the 11-16 Hz envelope above `factor` times its median, lasting 0.5 to 3 s: (onsets, offsets) in samples."""
    from scipy.signal import butter, sosfiltfilt, hilbert
    sg = sosfiltfilt(butter(4, band, btype="band", fs=fs, output="sos"), x)
    env = np.abs(hilbert(sg)); k = int(0.2 * fs); env = np.convolve(env, np.ones(k) / k, "same")
    above = env > factor * np.median(env)
    d = np.diff(np.r_[0, above.astype(int), 0]); on = np.flatnonzero(d == 1); off = np.flatnonzero(d == -1)
    keep = ((off - on) >= min_s * fs) & ((off - on) <= max_s * fs)
    return on[keep], off[keep], sg


def fig12_sleep_spindles(channel="c4-m1", t0=2788.0, dur=40.0, fmax=25.0, step_s=0.1, out="fig12_sleep_spindles.png"):
    """Forty seconds of stage N2 sleep with spindles, as spectrograms at three choices of window length and bandwidth,
    each by the multitaper estimate (K = 2NW - 1, at least 1) and by recipe (b'). A white box in each panel is the
    resolution of that panel: the window length by the half-power width of the kernel."""
    import glob, h5py
    from scipy.signal import butter, sosfiltfilt
    files = sorted(glob.glob(str(SLEEP_DIR / "sub-S*_std.h5")))
    if not files:
        print(f"fig12 skipped: no sleep recording in {SLEEP_DIR}"); return
    pad = 5.0
    with h5py.File(files[0], "r") as f:
        fs = int(f.attrs["sampling_rate"]); a = int((t0 - pad) * fs); b = int((t0 + dur + pad) * fs)
        x = f["signals/" + channel][a:b, 0].astype(float) * 1e6
        st = f["annotations_source/original/stage"][a:b, 0]
    x = x - x.mean(); t = np.arange(len(x)) / fs - pad
    on, off, sg = _spindles(x, fs)
    inside = (on / fs - pad > 0) & (off / fs - pad < dur); on, off = on[inside], off[inside]
    xb = sosfiltfilt(butter(4, [0.3, 35.0], btype="band", fs=fs, output="sos"), x)
    chn = _pretty_channel(channel.split("-")[0]) + "-" + channel.split("-")[1].upper()
    print(f"fig12: channel {channel}, {dur:.0f} s of stage {sorted(set(np.unique(st).astype(int)))} (2 = N2), Fs = {fs}; {len(on)} spindles at "
          f"{np.round(on / fs - pad, 1)} s lasting {np.round((off - on) / fs, 2)} s")
    step = int(step_s * fs); rows = []
    for T, B in SLEEP_SETTINGS:
        N = int(T * fs); nfft = 1 << int(np.ceil(np.log2(8 * N))); W = (B / 2) / fs; NW = N * W; K = max(1, int(round(2 * NW)) - 1)
        V = ss.dpss(N, NW, K)[0]; cw, hp = _recipe(N, NW)
        starts = np.arange(0, len(x) - N + 1, step); tc = (starts + N / 2) / fs - pad
        ff = np.arange(nfft // 2) / nfft * fs; keep = ff <= fmax
        mt = np.empty((keep.sum(), len(starts))); sm = np.empty_like(mt)
        for i, s0 in enumerate(starts):
            seg = x[s0:s0 + N] - x[s0:s0 + N].mean()
            mt[:, i] = ss.multitaper(seg, V, nfft)[0][:nfft // 2][keep] / fs
            sm[:, i] = ss.lag_window_estimate(seg, hp, nfft, taper=cw)[0][:nfft // 2][keep] / fs
        nu_mt = ss.dof_quadratic(ss.quadratic_matrix(N, 0.25, "multitaper", tapers=V))
        nu_sm = ss.dof_quadratic(ss.quadratic_matrix(N, 0.25, "lagwindow", h=hp, taper=cw))
        bw_mt = ss.kernel_stats(ss.kernel_multitaper(V, 16 * N))["bw3"] * fs; bw_sm = ss.kernel_stats(ss.kernel_smoothed(cw, hp, 16 * N))["bw3"] * fs
        sel = (tc >= 0) & (tc <= dur); band = (ff[keep] >= 1) & (ff[keep] <= fmax)
        dd = np.abs(10 * np.log10(mt[band][:, sel]) - 10 * np.log10(sm[band][:, sel]))
        # how long and how wide a spindle appears: full widths at half maximum of the 11-16 Hz burst, in time and in frequency
        sig = (ff[keep] >= 11) & (ff[keep] <= 16); wt, wf = {"mt": [], "sm": []}, {"mt": [], "sm": []}
        for name, P in (("mt", mt), ("sm", sm)):
            p_t = P[sig].mean(axis=0)
            for o1, o2 in zip(on, off):
                c = (o1 + o2) / 2 / fs - pad; j = int(np.argmin(np.abs(tc - c))); j = j - 10 + int(np.argmax(p_t[max(0, j - 10):j + 11])) if j >= 10 else j
                base = np.median(p_t[sel]); h = base + (p_t[j] - base) / 2; lo = hi = j
                while lo > 0 and p_t[lo] > h:
                    lo -= 1
                while hi < len(p_t) - 1 and p_t[hi] > h:
                    hi += 1
                wt[name].append(tc[hi] - tc[lo])
                col = P[:, j]; i0 = int(np.argmax(np.where(sig, col, 0))); lo = hi = i0
                while lo > 0 and col[lo] > col[i0] / 2:
                    lo -= 1
                while hi < len(col) - 1 and col[hi] > col[i0] / 2:
                    hi += 1
                wf[name].append(ff[keep][hi] - ff[keep][lo])
        rows.append(dict(T=T, B=B, NW=NW, K=K, mt=mt, sm=sm, tc=tc, f=ff[keep], nu=(nu_mt, nu_sm), bw=(bw_mt, bw_sm)))
        print(f"fig12: window {T:g} s, 2W = {B:g} Hz (NW = {NW:g}, K = {K}): nu = {nu_mt:.1f} multitaper, {nu_sm:.1f} smoothed; kernel half-power width {bw_mt:.2f} / {bw_sm:.2f} Hz; "
              f"2 x window x width = {2 * T * bw_mt:.1f} / {2 * T * bw_sm:.1f}; "
              f"median |difference| {np.median(dd):.2f} dB, 95th percentile {np.percentile(dd, 95):.2f} dB; a spindle appears {np.median(wt['mt']):.1f} / {np.median(wt['sm']):.1f} s long "
              f"and {np.median(wf['mt']):.1f} / {np.median(wf['sm']):.1f} Hz wide (multitaper / smoothed; median of {len(on)} spindles)")
    allv = np.concatenate([10 * np.log10(r["sm"][r["f"] >= 4]).ravel() for r in rows]); vmin, vmax = np.percentile(allv, [20, 99.7])
    fig = plt.figure(figsize=(W2, 7.0))
    gs = fig.add_gridspec(4, 2, height_ratios=[0.8, 1, 1, 1], hspace=0.62, wspace=0.12)
    a0 = fig.add_subplot(gs[0, :]); axes = [a0]
    a0.plot(t, xb, color="black", lw=0.35, label=f"EEG, {chn}, 0.3$-$35 Hz")
    a0.plot(t, sg - 95, color=COLOR["hann_gauss"], lw=0.4, label="the same, 11$-$16 Hz (offset)")
    for o1, o2 in zip(on, off):
        a0.axvspan(o1 / fs - pad, o2 / fs - pad, color=COLOR["hann_gauss"], alpha=0.15, lw=0)
    a0.set(xlim=(0, dur), ylim=(-125, 150), yticks=[-100, 0, 100], xlabel="time (s)", ylabel="$\\mu$V", title=f"stage N2 sleep, channel {chn}; {len(on)} spindles shaded")
    a0.legend(loc="upper right", ncol=2, fontsize=5.5)
    for r, row in enumerate(rows):
        for c, (key, name) in enumerate((("mt", f"multitaper, K = {row['K']}"), ("sm", "cosine taper, then parabola"))):
            a = fig.add_subplot(gs[r + 1, c]); axes.append(a)
            im = a.imshow(10 * np.log10(row[key]), aspect="auto", origin="lower", extent=[row["tc"][0], row["tc"][-1], row["f"][0], row["f"][-1]], vmin=vmin, vmax=vmax, cmap=SPEC_CMAP)
            a.add_patch(plt.Rectangle((1.0, fmax - 2.0 - row["bw"][c]), row["T"], row["bw"][c], fill=False, ec="white", lw=1.0))
            a.set(xlim=(0, dur), ylim=(0, fmax), title=f"{name}: {row['T']:g} s by {row['bw'][c]:.1f} Hz, $\\nu$ = {row['nu'][c]:.1f}", ylabel="frequency (Hz)" if c == 0 else "")
            if c == 1:
                a.set_yticklabels([])
            if r == len(rows) - 1:
                a.set(xlabel="time (s)")
    cb = fig.colorbar(im, ax=axes[1:], fraction=0.02, pad=0.015); cb.set_label("power (dB re 1 $\\mu$V$^2$/Hz)")
    _letters(axes, dx=-0.06); _save(fig, out)


if __name__ == "__main__":
    fig0_pedagogy(); print("fig0a/fig0b done")
    fig5_slepian_fill(); print("fig5 done")
    fig6_taper_banks(); fig6_taper_banks(full=True); print("fig6 done (compact + full)")
    fig7_three_routes(); print("fig7/fig9 done")
    fig8_everyday_motif(); print("fig8 done")
    fig1_kernels(); print("fig1 done (supplement)")
    fig2_ar4(); print("fig2 done (supplement)")
    fig3_tradeoff(); print("fig3 done (supplement)")
    fig4_eeg()
    fig10_eeg_seizure(); fig10_eeg_seizure("C", out="fig11_eeg_two_seizures.png", compact=True); print("fig10/fig11 done")
    fig12_sleep_spindles(); print("fig12 done")
    print("figures in", FIG)

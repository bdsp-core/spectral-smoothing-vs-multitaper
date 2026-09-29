"""Paper figures (run: python demos/make_figures.py). Writes figures/*.png at 300 dpi, authored at final column width.

Main text (paper order): fig9 three routes, one estimate (Fig. 1) | fig0b four estimators and their kernels (Fig. 2) |
  fig0a the record and the choice of width (Fig. 3) | fig5 the Slepian tapers tile the box (Fig. 4) | fig6 three ways to build a
  bank of tapers (Fig. 5) | fig8 the estimators used in practice (Fig. 6) | fig7 the same on EEG (Fig. 7) | fig10 a seizure
  (Fig. 8) | fig12 sleep spindles (Fig. 9).
Supplement: fig1 kernels at equal resolution | fig2 AR(4): dropping the leakiest taper | fig3 resolution, variance and leakage |
  fig6_full the taper banks with their eigen-tapers and kernels. fig4 and fig11 are kept but are not in the paper.

Style. Two looks share one code path, chosen by the environment variable FIG_STYLE.
  "tufte" (the default) follows Tufte's principles: ink is spent on data. Axes have only a left and a bottom spine, ticks point
  outward, grids are faint hairlines, legends have no frame and curves are labelled directly where there is room, and each panel
  carries a short note saying what it shows. Colours are the Okabe-Ito set, which is safe for colour-blind readers: the true PSD is
  black, multitaper blue, the smoothing recipe orange, the untapered box green, Welch purple. A family of curves ordered by an
  index is drawn in shades of one hue. A kernel whose side lobes are too dense to draw is shown by its envelope over a light fill.
  "babadi_brown" is the MATLAB look of Babadi and Brown, IEEE Trans. Biomed. Eng. 61(5):1555-1564, 2014: boxed axes with inward
  ticks on all four sides, dotted grids, boxed legends, MATLAB's classic colours, true PSD blue and dashed, multitaper black.
In both: Times text with the symbols of the paper as axis labels; panel letters (a), (b), ... outside the top-left corner; kernels
in dB relative to their peak, centred at f Delta = 0.4 over [0, 1/2], with the main lobe zoomed in at the top left. The EEG figures
read de-identified one-channel excerpts from data/. Spectrograms use the jet colour map with a colour bar. Each route keeps one
colour in every figure (ROUTE).
"""
import os, sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, ConnectionPatch
import specsmooth as ss

FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

# ---------------------------------------------------------------- shared style
W2, W1 = 7.16, 3.5                     # IEEE double- and single-column widths, inches
STYLE = os.environ.get("FIG_STYLE", "tufte")
TUFTE = STYLE != "babadi_brown"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 9, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
    "axes.labelpad": 2, "xtick.major.size": 3, "ytick.major.size": 3, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.minor.size": 1.5, "ytick.minor.size": 1.5, "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "legend.fancybox": False, "legend.borderpad": 0.35, "legend.handlelength": 2.0, "legend.labelspacing": 0.2,
    "legend.borderaxespad": 0.5, "legend.handletextpad": 0.5, "patch.linewidth": 0.6, "lines.linewidth": 0.8,
    "savefig.dpi": 300, "figure.dpi": 100,
})
if TUFTE:
    # Okabe-Ito colours under MATLAB's letters, so that a family of curves keeps its order of hues in both styles
    MATLAB = {"b": "#0072B2", "g": "#009E73", "r": "#D55E00", "c": "#56B4E9", "m": "#CC79A7", "y": "#E69F00", "k": "0.35"}
    plt.rcParams.update({
        "axes.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False, "xtick.direction": "out", "ytick.direction": "out",
        "xtick.top": False, "ytick.right": False, "grid.linestyle": "-", "grid.linewidth": 0.35, "grid.color": "0.88",
        "legend.frameon": False, "axes.prop_cycle": plt.cycler(color=[MATLAB[c] for c in "bygrcmk"]),
    })
    TRUTH = dict(color="black", ls="-", lw=1.1, zorder=5)               # the true PSD: black, drawn over the estimate
    ROUTE = {"multitaper": "#0072B2", "smooth": "#E69F00", "raw_box": "#009E73", "sinc": "#005C43", "welch": "#CC79A7", "hann_box": "#7A4A1E",
             "bohman_box": "#56B4E9", "hann_gauss": "#7A4A1E", "periodogram": "0.6"}
    KERNEL, BAR, SHADE, INK = "0.15", "0.55", "0.91", "0.45"
else:
    MATLAB = {"b": (0, 0, 1), "g": (0, 0.5, 0), "r": (1, 0, 0), "c": (0, 0.75, 0.75), "m": (0.75, 0, 0.75), "y": (0.75, 0.75, 0),
              "k": (0.25, 0.25, 0.25)}     # MATLAB's classic ColorOrder, used for families of curves
    plt.rcParams.update({
        "axes.linewidth": 0.6, "axes.spines.top": True, "axes.spines.right": True, "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True, "grid.linestyle": ":", "grid.linewidth": 0.5, "grid.color": "0.2",
        "legend.frameon": True, "legend.edgecolor": "black", "legend.framealpha": 1.0,
        "axes.prop_cycle": plt.cycler(color=[MATLAB[c] for c in "bgrcmyk"]),
    })
    TRUTH = dict(color=MATLAB["b"], ls="--", lw=1.2)                       # the true PSD, as in Babadi and Brown
    ROUTE = {"multitaper": "black", "smooth": MATLAB["r"], "raw_box": MATLAB["g"], "sinc": MATLAB["m"], "welch": MATLAB["m"], "hann_box": MATLAB["c"],
             "bohman_box": MATLAB["y"], "hann_gauss": MATLAB["k"], "periodogram": "0.6"}
    KERNEL = (0.11, 0.30, 0.21)            # the dark green of their kernels
    BAR = (0, 0, 0.5625)                   # MATLAB's default bar colour
    SHADE, INK = "0.85", "black"           # shading of seizures and spindles; guide lines
# One name for each method, used word for word in every figure; parameters follow the name after a comma.
NAME = {"multitaper": "Multitaper", "smooth": "Cosine taper, then parabola", "raw_box": "Untapered, then box", "welch": "Welch",
        "periodogram": "Periodogram", "sinc": "Sliding sinc window", "hann_box": "Hann, then box", "bohman_box": "Bohman, then box",
        "hann_gauss": "Hann, then Gaussian"}
SPEC_CMAP, SPEC_VLIM = "jet", (10.0, 45.0)      # jet is the convention for EEG spectrograms (M.B.W.)
TAPER_ALPHA = 0.25                     # recipe (b'): half a cosine on the first and last eighth of the record (a 25% Tukey taper)
# Labels follow the notation of the paper (Babadi and Brown 2014): sample k, taper index i = 1, 2, ..., L tapers, time-bandwidth
# product alpha, resolution R = 2 alpha / (N Delta) (the full width; the code's half-bandwidth W = NW / N is R / 2 with Delta = 1).
LBL = {"S": r"PSD (dB re $\sigma^2\Delta$)" if TUFTE else r"$\hat{S}(f)$ (dB)", "fD": r"$f\Delta$", "fND": r"$fN\Delta$", "K": r"$\mathcal{K}(f)$ (dB re peak)" if TUFTE else r"$\mathcal{K}(f)$ (dB)", "Hz": "Frequency (Hz)",
       "s": "Time (s)"}


def _recipe(N, Wn):
    """Recipe (b') at design half-bandwidth W = Wn/N: the 25% cosine taper and the lag window of a parabola whose
    half-power width is 2W (half-width sqrt(2) W). Wn = 0 gives the taper and no smoothing."""
    w = ss.unit_taper("tukey", N, alpha=TAPER_ALPHA)
    return w, (np.ones(N) if Wn == 0 else ss.parabolic_lag_window(N, np.sqrt(2) * Wn / N))


def _letters(axes, dx=20, size=9):
    """Panel letters (a), (b), ... in reading order, just outside the top-left corner (dx points to its left; one value, or one
    per panel)."""
    axes = list(np.ravel(axes)); dxs = dx if np.ndim(dx) else [dx] * len(axes)
    for k, (a, d) in enumerate(zip(axes, dxs)):
        a.annotate(f"({chr(97 + k)})", xy=(0, 1), xycoords="axes fraction", xytext=(-d, 0), textcoords="offset points",
                   ha="right", va="center", fontsize=size)


def _grid(*axes):
    """Babadi and Brown: a dotted grid. Tufte: faint horizontal hairlines, enough to read a level and no more."""
    for a in axes:
        a.grid(True, axis="y" if TUFTE else "both"); a.set_axisbelow(True)


def _note(a, text, loc="upper right", size=7, color="0.1"):
    """A short note inside a panel saying what it shows (Tufte style only): words belong on the graphic."""
    if not TUFTE:
        return
    x, ha = (0.97, "right") if "right" in loc else (0.03, "left")
    y, va = (0.96, "top") if "upper" in loc else (0.04, "bottom")
    a.text(x, y, text, transform=a.transAxes, ha=ha, va=va, fontsize=size, color=color, linespacing=1.15)


def _head(a, text, size=7):
    """A short label above the left end of a panel (Tufte style only), for panels such as spectrograms that have no room inside."""
    if TUFTE:
        a.set_title(text, loc="left", fontsize=size, pad=2.5)


def _label(a, x, y, text, color, ha="left", va="center", size=7):
    """A curve labelled directly, in its own colour."""
    a.text(x, y, text, color=color, ha=ha, va=va, fontsize=size)


def _ramp(color, n, darkest=0.35, lightest=0.55):
    """n shades of one hue, from a shade darker than the hue to a light tint: a family of curves ordered by an index."""
    from matplotlib.colors import to_rgb
    c = np.array(to_rgb(color)); out = []
    for k in range(n):
        u = -darkest + (darkest + lightest) * k / max(1, n - 1)
        out.append(tuple(c * (1 + u)) if u < 0 else tuple(c + (1 - c) * u))
    return out


def _save(fig, name):
    """The figure as PNG (for the README and the figure review) and as PDF (for the paper: the lettering stays sharp in print)."""
    fig.savefig(FIG / name, bbox_inches="tight", pad_inches=0.02)
    fig.savefig((FIG / name).with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02, dpi=300, metadata={"CreationDate": None})
    plt.close(fig)


def _db(H):
    return 10 * np.log10(np.maximum(H / H.max(), 1e-16))


def _kernel_panel(fig, a, f_off, H, f0=0.4, zoom=0.01, ylim=(-150, 40), zylim=(-50, 5), inset=(0.2, 0.6, 0.4, 0.36)):
    """The kernel in dB relative to its peak, centred at f Delta = f0 over [0, 1/2], with its main lobe zoomed in at the top left
    (Babadi and Brown, Figs. 2(b), 3(b) and 6(b)). f_off is the signed frequency grid of H, in cycles per sample."""
    o = np.argsort(f_off); x = f0 + f_off[o]; y = _db(H[o]); keep = (x >= 0) & (x <= 0.5)
    from scipy.signal import find_peaks
    from scipy.ndimage import maximum_filter1d
    lobes, _ = find_peaks(y[keep])
    if TUFTE and len(lobes) > 60:          # side lobes too dense to draw: the envelope of their peaks; the inset shows the lobes
        gap = int(np.ceil(1.5 * np.median(np.diff(lobes))))
        env = maximum_filter1d(y, size=2 * gap + 1, mode="nearest")
        a.plot(x[keep], env[keep], color=KERNEL, lw=0.7)
    else:
        a.plot(x[keep], y[keep], color=KERNEL, lw=0.6)
    a.set(xlim=(0, 0.5), ylim=ylim, yticks=[-150, -100, -50, 0], xticks=[0, 0.1, 0.2, 0.3, 0.4, 0.5],
          xticklabels=["0", "0.1", "0.2", "0.3", "0.4", "0.5"])
    ai = a.inset_axes(inset); z = (x >= f0 - zoom) & (x <= f0 + zoom)
    ai.plot(x[z], y[z], color=KERNEL, lw=0.9)
    ai.set(xlim=(f0 - zoom, f0 + zoom), ylim=zylim, xticks=[f0 - zoom, f0, f0 + zoom], yticks=[-50, -25, 0])
    ai.set_xticklabels([f"{f0 - zoom:g}", f"{f0:g}", f"{f0 + zoom:g}"])
    ai.tick_params(labelsize=6.5 if TUFTE else 5.5, length=2, pad=1)
    a.add_patch(Rectangle((f0 - zoom, zylim[0]), 2 * zoom, zylim[1] - zylim[0], fill=False, ec=INK, lw=0.5 if TUFTE else 0.8, zorder=5))
    for yA, yB in ((zylim[1], 1), (zylim[0], 0)):
        fig.add_artist(ConnectionPatch(xyA=(f0 - zoom, yA), coordsA=a.transData, xyB=(1, yB), coordsB=ai.transAxes, ls="-" if TUFTE else "--",
                                       lw=0.4 if TUFTE else 0.6, color=INK))
    if TUFTE:
        for sp in ai.spines.values():
            sp.set_visible(True); sp.set_linewidth(0.4); sp.set_color(INK)
        ai.tick_params(color=INK)
    return ai


def _signed(v):
    """A taper with the usual sign convention (Percival and Walden): a symmetric taper has a positive sum, and an antisymmetric
    one starts with a positive lobe. For display only."""
    s = v.sum()
    if abs(s) > 1e-8 * np.abs(v).sum():
        return v * np.sign(s)
    return v * np.sign(v[np.argmax(np.abs(v) > 0.01 * np.abs(v).max())])


def _freq_axis(a, label=True):
    a.set(xlim=(0, 0.5), xticks=[0, 0.1, 0.2, 0.3, 0.4, 0.5], xticklabels=["0", "0.1", "0.2", "0.3", "0.4", "0.5"])
    if label:
        a.set_xlabel(LBL["fD"])


# ---------------------------------------------------------------- supplement: fig1, fig2, fig3, fig4
def fig1_kernels(N=256, NW=4, nfft=8192):
    W = NW / N
    Vk, _ = ss.dpss(N, NW)
    hb = ss.box_lag_window(N, W)
    rect, hann, boh = np.ones(N) / np.sqrt(N), ss.unit_taper("hann", N), ss.unit_taper("bohman", N)
    ks = [("Ideal box of width $R$", ss.kernel_box(nfft, W), "black", ":", 1.0),
          (f"Multitaper, $L = {2 * NW - 1}$", ss.kernel_multitaper(Vk, nfft), ROUTE["multitaper"], "-", 1.0),
          ("Untapered, then box", ss.kernel_smoothed(rect, hb, nfft), ROUTE["raw_box"], "-", 0.9),
          (NAME["smooth"], ss.kernel_smoothed(*_recipe(N, NW), nfft), ROUTE["smooth"], "-", 1.1),
          ("Hann, then box", ss.kernel_smoothed(hann, hb, nfft), ROUTE["hann_box"], "-", 0.9),
          ("Bohman, then box", ss.kernel_smoothed(boh, hb, nfft), ROUTE["bohman_box"], "--", 0.9)]
    fs = ss.signed_freq(nfft); o = np.argsort(fs)
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.5))
    for name, H, c, st, lw in ks:
        ax[0].plot(fs[o] * N, H[o] / N, color=c, ls=st, lw=lw, label=name)
        ax[1].plot(fs[o] * N, _db(H[o]), color=c, ls=st, lw=lw, label=name)
    ax[0].set(xlim=(-2.5 * NW, 2.5 * NW), ylim=(0, 0.27 if TUFTE else 0.22), xticks=np.arange(-10, 11, 2), xlabel=LBL["fND"], ylabel=r"$\mathcal{K}(f)/N\Delta$")
    ax[1].set(xlim=(-6 * NW, 6 * NW), ylim=(-100, 5), xlabel=LBL["fND"], ylabel=LBL["K"])
    ax[0].legend(loc="upper left", ncol=1, fontsize=6.3, borderaxespad=0.2) if TUFTE else ax[0].legend(loc="upper center", ncol=2, fontsize=6.5)
    _grid(*ax); _letters(ax); fig.tight_layout(w_pad=2.0, rect=(0, 0, 0.83, 1) if TUFTE else None)
    if TUFTE:                              # each curve of (b) named at its right-hand end, moved apart where two ends are close
        edge = (fs[o] * N >= 5 * NW) & (fs[o] * N <= 6 * NW)
        ends = sorted([[max(float(_db(H[o])[edge].max()), -97.0), name, c] for name, H, c, st, lw in ks[1:]], reverse=True)
        for k in range(1, len(ends)):
            ends[k][0] = min(ends[k][0], ends[k - 1][0] - 8.0)
        for y, name, c in ends:
            ax[1].text(6 * NW + 0.6, y, name, color=c, fontsize=6.5, va="center", ha="left", clip_on=False)
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
    ests = [("Periodogram", ss.periodogram(x, nfft)[0], ROUTE["periodogram"], 0.5, "-"),
            ("Untapered, then box", ss.lag_window_estimate(x, hb, nfft)[0], ROUTE["raw_box"], 0.9, "-"),
            (f"Multitaper, $L = {2 * NW - 1}$", ss.multitaper(x, Vk, nfft)[0], ROUTE["multitaper"], 0.9, "-"),
            (f"Multitaper, $L = {2 * NW - 2}$", ss.multitaper(x, Vk[:, :-1], nfft)[0], ROUTE["multitaper"] if TUFTE else MATLAB["c"], 0.9, "--" if TUFTE else "-"),
            (f"Multitaper, $L = {2 * NW - 1}$, adaptive", ss.multitaper_adaptive(x, Vk, lam, nfft)[0], ROUTE["multitaper"] if TUFTE else MATLAB["y"], 0.9, ":" if TUFTE else "-"),
            (NAME["smooth"], ss.lag_window_estimate(x, _recipe(N, NW)[1], nfft, taper=_recipe(N, NW)[0])[0], ROUTE["smooth"], 0.9, "-")]
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.7), gridspec_kw={"width_ratios": [1.3, 1]})
    from scipy.ndimage import median_filter
    span = 0.02                            # Tufte: panel (b) shows the running median of each error over this span of f Delta
    for name, S, c, lw, st in ests:
        err = 10 * np.log10(S[:nfft // 2]) - 10 * np.log10(truth)
        if not TUFTE:
            ax[0].plot(f, 10 * np.log10(S[:nfft // 2]), color=c, lw=lw, ls=st, label=name)
            if name != "Periodogram":
                ax[1].plot(f, err, color=c, lw=lw, ls=st)
            continue
        if st == "-":                      # the two variants of the multitaper estimate cannot be told apart in (a) and are left to (b)
            ax[0].plot(f, 10 * np.log10(S[:nfft // 2]), color=c, lw=lw, ls=st, label=name)
        if name != "Periodogram":
            ax[1].plot(f, median_filter(err, size=2 * int(span * nfft / 2) + 1, mode="mirror"), color=c, lw=1.0, ls=st)
    ax[0].plot(f, 10 * np.log10(truth), label="True PSD", **TRUTH)
    ax[0].set(ylabel=LBL["S"], ylim=(-35, 60)); _freq_axis(ax[0])
    ax[1].axhline(0, color="black", lw=0.6); ax[1].set(ylabel="Estimate $-$ true PSD,\nrunning median (dB)" if TUFTE else r"$\hat{S}(f)-S(f)$ (dB)", ylim=(-5, 25) if TUFTE else (-10, 30)); _freq_axis(ax[1])
    ax[0].legend(loc="upper right", fontsize=6.3)
    _grid(*ax); _letters(ax); fig.tight_layout(w_pad=2.0, rect=(0, 0, 0.84, 1) if TUFTE else None)
    floors = {name: float(np.median((10 * np.log10(S[:nfft // 2]) - 10 * np.log10(truth))[f >= 0.4])) for name, S, c, lw, st in ests[1:]}
    if TUFTE:                              # each curve of (b) named at its right-hand end, moved apart where two ends are close
        ends = sorted([[floors[name], name, c] for name, S, c, lw, st in ests[1:]], reverse=True)
        for k in range(1, len(ends)):
            ends[k][0] = min(ends[k][0], ends[k - 1][0] - 2.0)
        for y, name, c in ends:
            ax[1].text(0.508, y, name, color=c, fontsize=6.5, va="center", ha="left", clip_on=False)
    _save(fig, "fig2_ar4_leakage.png")
    print(f"fig2: median level above the truth for f Delta >= 0.4, in dB: { {k: round(v, 1) for k, v in floors.items()} }")


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
        add("Multitaper, $L = 2\\alpha-1$", ss.kernel_multitaper(Vk, nfft), ss.quadratic_matrix(N, f0, "multitaper", tapers=Vk))
        V2 = Vk[:, :-1] if K > 1 else Vk
        add("Multitaper, $L = 2\\alpha-2$", ss.kernel_multitaper(V2, nfft), ss.quadratic_matrix(N, f0, "multitaper", tapers=V2))
        St = ss.sine_tapers(N, K)
        add("Multitaper, sine tapers", ss.kernel_multitaper(St, nfft), ss.quadratic_matrix(N, f0, "multitaper", tapers=St))
        hb = ss.box_lag_window(N, W)
        for tname, tp in [("Untapered", rect), ("Hann", hann), ("Bohman", boh)]:
            add(f"{tname}, then box", ss.kernel_smoothed(tp, hb, nfft), ss.quadratic_matrix(N, f0, "lagwindow", h=hb, taper=tp))
        tk, hp = _recipe(N, NW)
        add(NAME["smooth"], ss.kernel_smoothed(tk, hp, nfft), ss.quadratic_matrix(N, f0, "lagwindow", h=hp, taper=tk))
        hg = ss.gaussian_lag_window(N, W / 2)
        add("Hann, then Gaussian", ss.kernel_smoothed(hann, hg, nfft), ss.quadratic_matrix(N, f0, "lagwindow", h=hg, taper=hann))
    for L in [16, 24, 32, 48, 64, 96, 128]:
        wl = ss.unit_taper("hann", L)
        add("Welch, Hann segments, 50% overlap", ss.kernel_wosa(N, L, .5, nfft, wl), ss.quadratic_matrix(N, f0, "welch", seg_len=L, overlap=.5, taper=wl))
    style = {"Multitaper, $L = 2\\alpha-1$": (ROUTE["multitaper"], "o", "-"), "Multitaper, $L = 2\\alpha-2$": (ROUTE["multitaper"], "s", "--"),
             "Multitaper, sine tapers": (ROUTE["multitaper"], "^", ":"), "Untapered, then box": (ROUTE["raw_box"], "o", "-"),
             NAME["smooth"]: (ROUTE["smooth"], "o", "-"),
             "Hann, then box": (ROUTE["hann_box"], "o", "-"), "Bohman, then box": (ROUTE["bohman_box"], "s", "-"),
             "Hann, then Gaussian": (ROUTE["hann_gauss"], "^", "--" if TUFTE else "-"), "Welch, Hann segments, 50% overlap": (ROUTE["welch"], "D", "-")}
    skirt = (NAME["smooth"], "Hann, then Gaussian", "Welch, Hann segments, 50% overlap")     # no flat top: beyond 1.5 half-widths lies the skirt of the main lobe
    fig, axs = plt.subplots(2, 2, figsize=(W2, 4.6))
    ax = [axs[0, 0], axs[0, 1], axs[1, 0]]
    for name, pts in curves.items():
        p = np.array(sorted(pts)); c, m, st = style[name]
        for k, a in enumerate(ax):
            a.plot(p[:, 0], p[:, k + 1], color=c, marker=m, ls=st, ms=2.5, lw=0.8, mfc=c if TUFTE and k == 2 and name in skirt else "none", label=name)
    for a in ax:
        a.axvline(8, color="black", lw=0.7, ls=":")
        a.set_xlabel("Half-power width $\\times\\, N\\Delta$")
    if TUFTE:
        ax[0].text(8.5, 0.97, "$8/(N\\Delta)$, the width in Table 2", transform=ax[0].get_xaxis_transform(), fontsize=6.5, va="top", color="0.25")
    if TUFTE:
        ax[2].text(0.97, 0.9, "filled markers: no flat top,\nso this is the skirt of the main lobe", transform=ax[2].transAxes, fontsize=6.5, ha="right", va="top", color="0.25")
    ax[0].set(ylabel="Equivalent degrees of freedom $\\nu$" if TUFTE else "$\\nu$"); ax[1].set(ylabel="Kernel power beyond twice\nthe half-power half-width", yscale="log")
    ax[2].set(ylabel="Peak beyond 1.5 times the\nhalf-power half-width (dB)")
    _grid(*ax)
    axs[1, 1].axis("off"); h, l = ax[0].get_legend_handles_labels(); axs[1, 1].legend(h, l, loc="center", fontsize=7)
    _letters(ax, dx=24); fig.tight_layout(w_pad=2.0, h_pad=1.5)
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
    for a, img in zip(ax[:3], (pg, mt, sb)):
        im = a.imshow(img, aspect="auto", origin="lower", extent=[tt[0], tt[-1], f[keep][0], f[keep][-1]], vmin=vmin, vmax=vmax, cmap=SPEC_CMAP)
        a.set(ylabel=LBL["Hz"])
    dd = mt - sb
    im2 = ax[3].imshow(dd, aspect="auto", origin="lower", extent=[tt[0], tt[-1], f[keep][0], f[keep][-1]], vmin=-3, vmax=3, cmap="RdBu_r")
    ax[3].set(ylabel=LBL["Hz"], xlabel=LBL["s"])
    fig.colorbar(im, ax=list(ax[:3]), fraction=.02, pad=.01, label="PSD (dB)"); fig.colorbar(im2, ax=ax[3], fraction=.02, pad=.01, label="dB")
    _letters(ax); _save(fig, "fig4_eeg_spectrograms.png")
    print(f"fig4: {len(tt)} epochs of {win_s:.0f} s; MT vs smoothed |diff| median {np.median(np.abs(dd)):.2f} dB, 95% {np.percentile(np.abs(dd), 95):.2f} dB")


# ---------------------------------------------------------------- fig5: the Slepian tapers tile the box
def fig5_slepian_fill(N=256, NW=4, nfft=8192):
    """Running eigenvalue-weighted sum of |H^(i)|^2 converges to the box (Thomson 1982 eq. 8.3)."""
    W = NW / N
    V, lam = ss.dpss_all(N, W)
    fs = ss.signed_freq(nfft); o = np.argsort(fs)
    Uk = np.abs(np.fft.fft(V, nfft, axis=0)) ** 2 / N
    fig, ax = plt.subplots(2, 2, figsize=(W2, 4.6))
    t = np.arange(N)
    a = ax[0, 0]                                                   # Babadi and Brown, Fig. 5(a): the first, second and seventh tapers
    three = _ramp(ROUTE["multitaper"], 3) if TUFTE else None
    for j, (i, c) in enumerate(((0, "black"), (1, MATLAB["b"]), (6, MATLAB["r"]))):
        a.plot(t, _signed(V[:, i]), color=three[j] if TUFTE else c, lw=0.9, label=f"$h^{{({i + 1})}}_k$")
    a.set(xlim=(0, N), xticks=[0, N // 4, N // 2, 3 * N // 4, N], ylim=(-0.2, 0.2), xlabel="$k$", ylabel="$h^{(i)}_k$")
    a.legend(loc="lower left", fontsize=7)
    a = ax[0, 1]
    for j, (i, c) in enumerate(((0, "black"), (1, MATLAB["b"]), (2, MATLAB["g"]))):
        a.plot(fs[o] * N, Uk[o, i], color=three[j] if TUFTE else c, lw=0.9, label=f"$i = {i + 1}$")
    for s_ in (-NW, NW):
        a.axvline(s_, color="black", ls=":", lw=0.8)
    a.set(xlim=(-2 * NW, 2 * NW), ylim=(0, 0.65), xlabel=LBL["fND"], ylabel="$|H^{(i)}(f)|^2/N\\Delta^2$")
    if TUFTE:
        a.text(-NW + 0.2, 0.62, "$-R/2$", fontsize=6.5, ha="left", va="center"); a.text(NW - 0.2, 0.62, "$R/2$", fontsize=6.5, ha="right", va="center")
    a.legend(loc="upper right", fontsize=7)
    a = ax[1, 0]
    cum = np.cumsum(Uk * lam, axis=1)
    shades = _ramp(ROUTE["multitaper"], 5)[::-1] if TUFTE else [MATLAB[c] for c in "bgrcm"]
    for K, c in list(zip((1, 3, 5, 7, 9), shades)) + [(N, "black")]:
        a.plot(fs[o] * N, cum[o, K - 1] / (2 * NW), color=c, lw=0.9 if K < N else 1.6, label=f"$L = {K}$" if K < N else "$L = N$")
    a.plot(fs[o] * N, (np.abs(fs[o]) <= W) / (2 * W) / N, color="black", ls="--", lw=0.7, label="Box")
    a.set(xlim=(-2 * NW, 2 * NW), ylim=(0, 0.2), yticks=[0, 0.05, 0.1, 0.15, 0.2], yticklabels=["0.00", "0.05", "0.10", "0.15", "0.20"],
          xlabel=LBL["fND"], ylabel="$\\mathcal{K}(f)/N\\Delta$")
    a.legend(loc="upper center", ncol=4, fontsize=6.5, handlelength=1.5, columnspacing=0.8)
    a = ax[1, 1]
    a.semilogy(np.arange(1, N + 1), np.maximum(1 - lam, 1e-16), "o-", ms=2.5, lw=0.7, mfc="none", color="black")
    a.axvline(2 * NW - 1, color=INK, ls=":", lw=0.8, label="$L = 2\\alpha-1$"); a.axvline(2 * NW, color="black" if TUFTE else MATLAB["r"], ls="--", lw=0.8, label="$2\\alpha$")
    a.set(xlim=(0.5, 2 * NW + 9), ylim=(1e-16, 2), xlabel="$i$", ylabel="$1 - \\lambda_i$")
    a.legend(loc="lower right", fontsize=7)
    _grid(*ax.ravel()) if TUFTE else _grid(ax[0, 1], ax[1, 0], ax[1, 1]); _letters(ax, dx=22)
    fig.tight_layout(w_pad=2.0, h_pad=1.2); _save(fig, "fig5_slepian_fill.png")


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


def _motif_rows(fig, ax_top, ax_bot, f, truth, ests, kers, fs, colors, lws, zooms, notes=None):
    """The repeated motif: the estimate over the true PSD (top), and the estimator's kernel (bottom)."""
    for j, (a, S, c, lw) in enumerate(zip(ax_top, ests, colors, lws)):
        a.plot(f, 10 * np.log10(S), color=c, lw=lw, label="Estimate")
        a.plot(f, 10 * np.log10(truth), label="True PSD", **TRUTH)
        a.set(ylim=(-45, 60)); _freq_axis(a, label=False)
        if j == 0:
            a.set_ylabel(LBL["S"])
            if TUFTE:
                _label(a, 0.015, -37, "True PSD", "black")
            else:
                a.legend(loc="upper right", fontsize=6.5)
        if notes:                          # the note names the estimate and is written in its colour
            _note(a, notes[j], color="0.35" if c in ("black", ROUTE["periodogram"]) else c)
    for j, (a, H, z) in enumerate(zip(ax_bot, kers, zooms)):
        _kernel_panel(fig, a, fs, H, zoom=z)
        a.set_xlabel(LBL["fD"])
        if j == 0:
            a.set_ylabel(LBL["K"])
    _grid(*ax_top, *ax_bot)


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
    cols = [(rect, 0), (cosw, 0), (cosw, W_right), (cosw, W_wide)]
    ests, kers, nus, bws, masses = [], [], [], [], []
    for tp, Wn in cols:
        h = _recipe(N, Wn)[1]
        ests.append(ss.lag_window_estimate(x, h, nfft, taper=tp)[0][:nfft // 2])
        H = ss.kernel_smoothed(tp, h, nfft); kers.append(H)
        nus.append(ss.dof_quadratic(ss.quadratic_matrix(N, 0.25, "lagwindow", h=h, taper=tp)))
        st = ss.kernel_stats(H, W); bws.append(st["bw3"] * N); masses.append(st["mass_out_2W"])
    Ws = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 28, 32])
    band = (f1 - 0.02, f2 + 0.02)
    sw = _sweep_width(N, nfft, Ws, mc, np.random.default_rng(seed + 1), band)

    # fig0b: the motif
    fig, ax = plt.subplots(2, 4, figsize=(W2, 3.9))
    cols4 = [ROUTE["periodogram"]] + [ROUTE["smooth"]] * 3 if TUFTE else ["black"] * 4
    notes = [f"{NAME['periodogram']}, untapered", f"{NAME['periodogram']}, cosine taper", f"{NAME['smooth']},\n$R = {2 * W_right}/(N\\Delta)$",
             f"{NAME['smooth']},\n$R = {2 * W_wide}/(N\\Delta)$"]
    notes = [f"{n}\n$\\nu = {v:.1f}$" for n, v in zip(notes, nus)]
    _motif_rows(fig, ax[0], ax[1], f, truth, ests, kers, fs, cols4, [0.45, 0.45, 0.8, 0.8], [0.01, 0.01, 0.01, 0.04], notes)
    _letters(ax, dx=18); fig.tight_layout(w_pad=0.5, h_pad=0.8); _save(fig, "fig0b_estimates_kernels.png")

    # fig0a: the record and the sweep
    fig, (ax_ts, ax_err) = plt.subplots(2, 1, figsize=(W1, 4.4), gridspec_kw={"height_ratios": [1, 1.3]})
    t = np.arange(N); env = cosw / cosw.max(); amp = np.abs(x).max()
    ax_ts.plot(t, x, color="black", lw=0.35)
    ax_ts.plot(t, env * x, color=ROUTE["smooth"], lw=0.35)
    ax_ts.plot(t, amp * env, color="black", ls="--", lw=0.8); ax_ts.plot(t, -amp * env, color="black", ls="--", lw=0.8)
    if TUFTE:
        ax_ts.set_ylim(-1.15 * amp, 1.45 * amp)
        _label(ax_ts, 8, 1.36 * amp, "Record", "black"); _label(ax_ts, 300, 1.36 * amp, "Tapered record", ROUTE["smooth"])
        _label(ax_ts, 700, 1.36 * amp, "Taper (dashed)", "black")
    ax_ts.set(xlim=(0, N - 1), xticks=[0, 255, 511, 767, 1023], ylim=(-1.15 * amp, (1.45 if TUFTE else 1.15) * amp), xlabel="$k$", ylabel="$x_k$")
    R = 2 * Ws
    ax_err.plot(R, sw["total"], color="black", marker="o", ms=2.2, lw=1.1, label="Total, whole band")
    ax_err.plot(R, sw["total_band"], color="black", marker="s", ms=2.2, mfc="none", lw=0.9, ls="--", label="Total, peak region")
    part = dict(color="black", lw=0.55, zorder=1.5) if TUFTE else None      # Tufte: the two parts of the error as thin lines under their total
    ax_err.plot(R, sw["noise"], label="Noise", **(part or dict(color=MATLAB["b"], lw=1.0)))
    ax_err.plot(R, sw["blur"], label="Bias", **(part or dict(color=MATLAB["r"], lw=1.0)))
    # the three cases of Fig. 2, each named beside its circle: (where the text starts, its alignment)
    cases = ((0, "too noisy", (9.0, 6.6), "left"), (W_right, "just right", (10.0, 5.0), "left"), (W_wide, "too smooth", (2 * W_wide + 1.5, 2.9), "left"))
    for Wn, word, (tx, ty), ha in cases:
        i = np.argmin(np.abs(Ws - Wn)); ax_err.plot(2 * Wn, sw["total"][i], "o", ms=7, mfc="none", mec="black", mew=0.9)
        if TUFTE:
            ax_err.annotate(f"{word}, $\\nu = {sw['nu'][i]:.1f}$", xy=(2 * Wn, sw["total"][i]), xytext=(tx, ty), fontsize=6.5, ha=ha, va="center",
                            arrowprops=dict(arrowstyle="-", lw=0.4, color=INK, shrinkA=1.5, shrinkB=4.5))
    top = 9.3 if TUFTE else 8.5
    for xv, word in ((peak_width, "peak width"), (peak_sep, "peak separation")):
        ax_err.axvline(xv, color=INK, ls=":", lw=0.7 if TUFTE else 0.9)
        if TUFTE:
            ax_err.text(xv + 0.7, top - 0.12, word, fontsize=6.5, color="0.25", va="top")
    ax_err.set(xlim=(-1, 2 * Ws[-1] + (5 if TUFTE else 1)), ylim=(0, top), yticks=[0, 2, 4, 6, 8], xlabel="$RN\\Delta$", ylabel="RMS error (dB)")
    if TUFTE:
        top_ax = ax_err.secondary_xaxis("top"); at = [i for i, w in enumerate(Ws) if w in (0, 4, 8, 12, 16, 24, 32)]
        top_ax.set_xticks([R[i] for i in at]); top_ax.set_xticklabels([f"{sw['nu'][i]:.0f}" for i in at]); top_ax.set_xlabel("$\\nu$", labelpad=3)
        top_ax.tick_params(direction="out", length=3, width=0.5, labelsize=7.5); top_ax.spines["top"].set_linewidth(0.5)
        e = len(R) - 1
        _label(ax_err, R[e], sw["total_band"][e] + 0.3, "Total, peak region", "black", ha="right", va="bottom")
        _label(ax_err, R[e] + 1.2, sw["total"][e], "Total,\nwhole\nband", "black", ha="left", va="center")
        _label(ax_err, R[e], sw["noise"][e] + 0.25, "Noise", "black", ha="right", va="bottom")
        _label(ax_err, R[e - 6], sw["blur"][e - 6] - 0.4, "Bias", "black", ha="left", va="top")
    else:
        ax_err.legend(loc="upper right", fontsize=6.8)
    _grid(ax_err); _letters([ax_ts, ax_err], dx=22); fig.tight_layout(h_pad=1.0); _save(fig, "fig0a_record_sweep.png")
    i_r, i_w = np.argmin(np.abs(Ws - W_right)), np.argmin(np.abs(Ws - W_wide))
    print(f"fig0: nu = {[round(float(v), 1) for v in nus]}; bws = {[round(float(v), 1) for v in bws]}; mass beyond 2W = {[round(float(v), 4) for v in masses]}; RMS dB error whole band / peak region: "
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
    cols = [(ss.lag_window_estimate(x, hb, nfft)[0], dict(method="lagwindow", h=hb, taper=rect), ROUTE["raw_box"]),
            (ss.multitaper(x, Vk, nfft)[0], dict(method="multitaper", tapers=Vk), ROUTE["multitaper"]),
            (ss.lag_window_estimate(x, hp, nfft, taper=cosw)[0], dict(method="lagwindow", h=hp, taper=cosw), ROUTE["smooth"]),
            (ss.welch_sliding(x, hann_seg, nfft, step=step, overhang=False)[0], dict(method="welch", taper=hann_seg, step=step), ROUTE["welch"])]
    ests, kers, nus, bws, masses = [], [], [], [], []
    for S, q, _ in cols:
        m = q.pop("method")
        Q0 = ss.quadratic_matrix(N, 0.0, m, **q); Qf = ss.quadratic_matrix(N, 0.25, m, **q)
        H = ss.kernel_quadratic(Q0, nfft); st = ss.kernel_stats(H, W)
        ests.append(S[:nfft // 2]); kers.append(H); nus.append(ss.dof_quadratic(Qf)); bws.append(st["bw3"] * N); masses.append(st["mass_out_2W"])
    fig, ax = plt.subplots(2, 4, figsize=(W2, 3.9))
    notes = [f"{n}\n$\\nu = {v:.1f}$" for n, v in zip((NAME["raw_box"], f"{NAME['multitaper']}, $L = {K}$", NAME["smooth"], f"{NAME['welch']}, {nseg} Hann segments"), nus)]
    _motif_rows(fig, ax[0], ax[1], f, truth, ests, kers, fs, [c[2] for c in cols], [0.8] * 4, [0.01] * 4, notes)
    floors = [float(np.median(10 * np.log10(S[f >= 0.4] / truth[f >= 0.4]))) for S in ests]
    if TUFTE:                              # how far above the truth the two leaky estimates level off
        for a, gap in zip(ax[0][:2], floors[:2]):
            x0 = 0.45; y0 = 10 * np.log10(truth[np.argmin(np.abs(f - x0))])
            a.annotate("", xy=(x0, y0 + gap), xytext=(x0, y0), arrowprops=dict(arrowstyle="<->", lw=0.5, color="0.15", shrinkA=0, shrinkB=0, mutation_scale=5), zorder=6)
            a.text(x0, y0 - 4, f"{gap:.0f} dB", ha="center", va="top", fontsize=6.5, zorder=6)
    _letters(ax, dx=18); fig.tight_layout(w_pad=0.5, h_pad=0.8); _save(fig, "fig8_everyday_motif.png")
    print(f"fig8: median level above the truth for f Delta >= 0.4, in dB = {[round(v, 1) for v in floors]}")
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
    Qs = [("average segments", ss.quadratic_matrix(N, 0.0, "welch", taper=hann_seg, step=step).real),
          ("smooth one periodogram", ss.quadratic_matrix(N, 0.0, "lagwindow", h=hp, taper=cosw).real),
          ("slide a sinc window", ss.quadratic_matrix(N, 0.0, "lagwindow", h=ss.box_lag_window(N, W)).real),
          ("Thomson multitaper", ss.quadratic_matrix(N, 0.0, "multitaper", tapers=V[:, :K]).real)]
    fs = ss.signed_freq(nfft); o = np.argsort(fs)
    nrows = 4 if full else 2
    fig, ax = plt.subplots(nrows, 4, figsize=(W2, 1.6 * nrows + 0.3))
    cyc = [MATLAB[c] for c in "bgrcmyk"]
    hue = [ROUTE[k] for k in ("welch", "smooth", "raw_box", "multitaper")]
    heads = (f"{NAME['welch']}, {len(range(0, N - seg_len + 1, step))} Hann segments", NAME["smooth"], f"{NAME['sinc']}\n(= {NAME['raw_box'].lower()})",
             f"{NAME['multitaper']}, $L = {K}$")
    for j, (title, Q) in enumerate(Qs):
        a = ax[0, j]; _head(a, heads[j], size=6.5)
        four = _ramp(hue[j], 4) if TUFTE else None
        if j == 0:
            for i, s in enumerate(range(0, N - seg_len + 1, step)):
                u = np.zeros(N); u[s:s + seg_len] = hann_seg; a.plot(t, u, lw=0.8, color=hue[j] if TUFTE else cyc[i % 7])
            a.set_ylabel("$h_{k-n}$")
            if TUFTE:
                a.set_ylim(-0.005, 0.27); _note(a, f"shifts $n = 0, {step}, \\ldots, {N - seg_len}$", size=6.5)
        elif j == 1:
            for i, fk in enumerate([0, 1, 2, 3]):
                a.plot(t, cosw * np.cos(2 * np.pi * fk / N * t), lw=0.8, color=four[i] if TUFTE else cyc[i], label=f"$f'N\\Delta = {fk}$")
            a.set(ylabel="$h_k\\cos(2\\pi f'k\\Delta)$", ylim=(-0.08, 0.13)); a.legend(loc="upper center", fontsize=6.5 if TUFTE else 5.5, ncol=2, handlelength=1.2, columnspacing=0.6)
        elif j == 2:
            w = ss.sinc_window(8 * N, W)
            for i, s in enumerate([-N // 2, 0, N // 2, N]):
                u = np.zeros(N); lo, hi = max(0, s - 4 * N), min(N, s + 4 * N)
                u[lo:hi] = w[lo - (s - 4 * N):hi - (s - 4 * N)]
                a.plot(t, u / w.max(), lw=0.8, color=four[i] if TUFTE else cyc[i])
            a.set_ylabel("$h_{k-n}/\\max h$")
            if TUFTE:
                a.set_ylim(-0.28, 1.4); _note(a, f"at $k = {-N // 2}, 0, {N // 2}, {N}$", size=6.5)
        else:
            for i in range(4):
                a.plot(t, _signed(V[:, i]), lw=0.8, color=four[i] if TUFTE else ["black", MATLAB["b"], MATLAB["g"], MATLAB["r"]][i], label=f"$i = {i + 1}$")
            a.set(ylabel="$h^{(i)}_k$", ylim=(-0.13, 0.2)); a.legend(loc="upper center", fontsize=6.5 if TUFTE else 5.5, ncol=2, handlelength=1.2, columnspacing=0.6)
        a.set(xlabel="$k$", xlim=(0, N - 1), xticks=[0, N // 2, N])
        c, U = ss.eigen_tapers(Q)
        if j == 3:
            U = V
        r = 1
        if full:
            a = ax[1, j]
            for i in range(4):
                a.plot(t, _signed(U[:, i]), lw=0.8, color=four[i] if TUFTE else ["black", MATLAB["b"], MATLAB["g"], MATLAB["r"]][i], label=f"$i = {i + 1}$")
            a.set(xlabel="$k$", xlim=(0, N - 1), xticks=[0, N // 2, N], ylabel=("Eigen-taper $h^{(i)}_k$" if TUFTE else "$h^{(i)}_k$") if j == 0 else "")
            if j == 3:
                _note(a, "the bank itself, as in (d)", size=6.5)
            a.set_ylim(a.get_ylim()[0], a.get_ylim()[1] + 0.45 * (a.get_ylim()[1] - a.get_ylim()[0]))
            if j == 0:
                a.legend(loc="upper center", fontsize=6.5 if TUFTE else 5.5, ncol=2, handlelength=1.2, columnspacing=0.6)
            r = 2
        a = ax[r, j]
        cc = c / c.sum(); kmax = 12
        if TUFTE:
            a.bar(np.arange(1, kmax + 1), cc[:kmax], color=hue[j], lw=0, width=0.8); _note(a, f"$\\nu = {ss.dof_from_weights(c):.1f}$")
        else:
            a.bar(np.arange(1, kmax + 1), cc[:kmax], color=BAR, edgecolor="black", lw=0.4, width=0.8)
        a.set(xlabel="$i$", ylabel=("Weight $c_i/\\sum c_i$" if TUFTE else "$c_i/\\sum c_i$") if j == 0 else "", ylim=(0, 0.2), xlim=(0.3, kmax + 0.7))
        H = ss.kernel_quadratic(Q, nfft); st = ss.kernel_stats(H, W)
        if full:
            a = ax[3, j]
            a.plot(fs[o] * N, _db(H[o]), color=hue[j] if TUFTE else KERNEL, lw=0.8)
            for s_ in (-NW, NW):
                a.axvline(s_, color=INK, ls=":", lw=0.7)
            a.set(xlim=(-24, 24), ylim=(-100 if TUFTE else -80, 5), xlabel=LBL["fND"], ylabel=LBL["K"] if j == 0 else "")
            _grid(a)
        print(f"fig6 {title}: dof {ss.dof_from_weights(c):.1f}; bw {st['bw3']*N:.2f}/N; mass beyond 2W {100*st['mass_out_2W']:.2f}%; top weights {np.round(cc[:8], 3)}")
    if TUFTE:
        _grid(*ax.ravel())
    _letters(ax, dx=26); fig.tight_layout(w_pad=0.4, h_pad=1.0); _save(fig, "fig6_taper_banks_full.png" if full else "fig6_taper_banks.png")


# ---------------------------------------------------------------- fig9 / fig7: three routes on one EEG epoch
def fig7_three_routes(NW=4, win_s=8.0, start_s=200.0, fmax=40.0, L_sinc_mult=16, seg_len=192):
    """fig9 (Fig. 1): the exact trio, their differences on a log scale, and convergence in the window length.
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
    dB = lambda S: 10 * np.log10(S / Fs)                              # per hertz: dB re 1 microvolt squared per hertz
    d_box = np.abs(dB(S_box) - dB(S_mt_all)); d_sinc = {L: np.abs(dB(S_sinc[L]) - dB(S_mt_all)) for L in Ls}
    # fig9 (Fig. 1), laid out as Babadi and Brown's Fig. 1: (a) across the top, (b) and (c) below
    fig = plt.figure(figsize=(W2, 4.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1], width_ratios=[1.6, 1], hspace=0.42, wspace=0.28)
    a = fig.add_subplot(gs[0, :]); axes = [a]
    a.plot(f[keep], dB(S_mt_all), color=ROUTE["multitaper"], lw=2.4, label="Multitaper, all $N$ tapers, weights $\\lambda_i$")
    a.plot(f[keep], dB(S_box), color=ROUTE["raw_box"], lw=1.0, ls="--", label=f"{NAME['raw_box']} of width $R$")
    a.plot(f[keep], dB(S_sinc[16 * N]), color=ROUTE["sinc"], lw=1.0, ls=":", label=f"{NAME['sinc']}, $N_s = 16N$")
    a.set(xlabel=LBL["Hz"], ylabel=PSD_UNIT if TUFTE else LBL["S"], xlim=(0, fmax))
    a.legend(loc="upper right", **({"title": "Three estimates that coincide", "title_fontsize": 7, "alignment": "left"} if TUFTE else {}))
    a = fig.add_subplot(gs[1, 0]); axes.append(a)
    a.semilogy(f[keep], np.maximum(d_box, 1e-16), color=ROUTE["raw_box"], lw=0.7, label=f"{NAME['raw_box']} $-$ multitaper")
    a.semilogy(f[keep], np.maximum(d_sinc[16 * N], 1e-16), color=ROUTE["sinc"], lw=0.7, label=f"{NAME['sinc']} $-$ multitaper")
    a.set(xlabel=LBL["Hz"], ylabel="|Difference| (dB)", xlim=(0, fmax), ylim=(1e-15, 100 if TUFTE else 10), yticks=[1e-15, 1e-10, 1e-5, 1])
    if TUFTE:
        _label(a, 0.98 * fmax, 8, f"{NAME['sinc']} $-$ multitaper", ROUTE["sinc"], ha="right")
        _label(a, 0.98 * fmax, 1e-11, f"{NAME['raw_box']} $-$ multitaper: rounding error", ROUTE["raw_box"], ha="right")
    else:
        a.legend(loc="center right", fontsize=6.5)
    a = fig.add_subplot(gs[1, 1]); axes.append(a)
    mx = [d_sinc[L].max() for L in Ls]
    a.loglog(np.array(Ls) / N, mx, "o-", color=ROUTE["sinc"], ms=3.5, mfc="none", lw=0.9, label="Maximum |difference|")
    a.loglog(np.array(Ls) / N, mx[1] * (Ls[1] / np.array(Ls)), color="black", ls=":", lw=0.9, label="$1/N_s$")
    a.set(xlabel="$N_s/N$", ylabel="Max. |difference| (dB)", xticks=[1, 2, 4, 8, 16], xticklabels=["1", "2", "4", "8", "16"])
    a.minorticks_off()
    if TUFTE:
        _label(a, 6.0, mx[1] * (Ls[1] / (6.0 * N)) * 1.9, "$1/N_s$", "black")            # the one curve is named by the axis
    else:
        a.legend(loc="upper right", fontsize=6.5)
    _grid(*axes); _letters(axes, dx=[22, 22, 30]); _save(fig, "fig9_graphical_abstract.png")
    print("fig9: max|box-mt| = %.2g dB; max|sinc-mt| by L/N: %s" % (d_box.max(), {L // N: round(float(d_sinc[L].max()), 4) for L in Ls}))
    # fig7: the everyday trio, stacked in one column as Babadi and Brown's Fig. 2
    step = (N - seg_len) // 8; hann_seg = ss.unit_taper("hann", seg_len); nseg = len(range(0, N - seg_len + 1, step))
    every = [("multitaper", f"{NAME['multitaper']}, $L = {K}$", ss.multitaper(x, V[:, :K], nfft)[0], ROUTE["multitaper"], 1.0),
             ("smooth", NAME["smooth"], ss.lag_window_estimate(x, hp, nfft, taper=cosw)[0], ROUTE["smooth"], 0.9),
             ("average", f"{NAME['welch']}, {nseg} Hann segments", ss.welch_sliding(x, hann_seg, nfft, step=step, overhang=False)[0], ROUTE["welch"], 0.9)]
    dofs = [ss.dof_quadratic(ss.quadratic_matrix(N, 0.25, "multitaper", tapers=V[:, :K])),
            ss.dof_quadratic(ss.quadratic_matrix(N, 0.25, "lagwindow", h=hp, taper=cosw)),
            ss.dof_quadratic(ss.quadratic_matrix(N, 0.25, "welch", taper=hann_seg, step=step))]
    bws = [ss.kernel_stats(ss.kernel_multitaper(V[:, :K], nfft))["bw3"] * Fs, ss.kernel_stats(ss.kernel_smoothed(cosw, hp, nfft))["bw3"] * Fs,
           ss.kernel_stats(ss.kernel_wosa(N, seg_len, 0.5, nfft, hann_seg))["bw3"] * Fs]
    fig, ax = plt.subplots(2, 1, figsize=(W1, 4.3), sharex=True, gridspec_kw={"height_ratios": [1.3, 1]})
    a = ax[0]
    for key, name, S, c, lw in every:
        a.plot(f[keep], dB(S[:nfft // 2][keep]), color=c, lw=lw, label=name)
    a.set(ylabel=PSD_UNIT if TUFTE else LBL["S"], xlim=(0, fmax)); a.legend(loc="upper right", fontsize=6.8)
    a = ax[1]; ref = dB(every[0][2][:nfft // 2][keep]); out = {}
    for z, (key, name, S, c, lw) in enumerate(every[1:]):
        dd = dB(S[:nfft // 2][keep]) - ref
        out[key] = (np.median(np.abs(dd)), np.percentile(np.abs(dd), 95))
        a.plot(f[keep], dd, color=c, lw=0.7, label=f"{name} $-$ multitaper", zorder=3 - z)
    a.axhline(0, color="black", lw=0.5)
    sd = 4.34 * np.sqrt(2 / dofs[0])
    if TUFTE:
        a.axhspan(-sd, sd, color="0.9", lw=0, zorder=0); _note(a, f"Grey band: $\\pm${sd:.1f} dB, the standard deviation\nof the multitaper estimate", size=6.5)
    a.set(xlabel=LBL["Hz"], ylabel="Difference (dB)", xlim=(0, fmax), ylim=(-15, 15)); a.legend(loc="lower right", fontsize=6.3)
    _grid(*ax); _letters(ax, dx=22); fig.tight_layout(h_pad=0.6); _save(fig, "fig7_everyday_eeg.png")
    print("fig7:", out, "dofs", np.round(dofs, 1), "bws(Hz)", np.round(bws, 2), "nseg", nseg, "step", step, "expected log-noise std (dB):", np.round(4.34 * np.sqrt(2 / np.array(dofs)), 2))


# ---------------------------------------------------------------- fig10 / fig11: seizure clips (de-identified excerpts in data/)
DATA = ROOT / "data"                   # one channel of each recording, built by demos/make_data_excerpts.py
EEG_CASES = {"A": "eeg_seizure_1.npz", "C": "eeg_seizure_2.npz"}


def load_excerpt(name):
    """An excerpt from data/ as a dict: x (microvolts), fs, channel, and for the seizure clips seizures_s and ictal_rise_db."""
    p = DATA / name
    if not p.exists():
        return None
    with np.load(p, allow_pickle=False) as d:
        return {k: (d[k].astype(float) if d[k].dtype.kind == "f" and d[k].ndim else d[k][()] if d[k].ndim == 0 else d[k]) for k in d.files}


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


PSD_UNIT = "PSD (dB re 1 $\\mu$V$^2$/Hz)"


def _spectrogram(fig, a, img, tt, f, vmin, vmax, bar=True):
    """A spectrogram with its own colour bar at the right, as in Babadi and Brown's Figs. 7 and 8 (bar=False: the image alone,
    for panels that share one bar)."""
    im = a.imshow(img, aspect="auto", origin="lower", extent=[tt[0], tt[-1], f[0], f[-1]], vmin=vmin, vmax=vmax, cmap=SPEC_CMAP)
    _frame(a)
    if not bar:
        return im
    cb = fig.colorbar(im, ax=a, fraction=0.035, pad=0.015); cb.ax.tick_params(labelsize=6.5, direction="out" if TUFTE else "in")
    cb.outline.set_linewidth(0.4 if TUFTE else 0.6)
    return cb


def _frame(a):
    """An image has an edge: in the Tufte style, where line panels have two spines, an image keeps a thin frame on all four sides."""
    if TUFTE:
        for sp in a.spines.values():
            sp.set_visible(True); sp.set_linewidth(0.4)


def fig10_eeg_seizure(case="A", win_s=2.0, step_s=1.0, NW=2, fmax=30.0, t_pre=None, t_ictal=None, out="fig10_eeg_seizure.png", compact=False):
    """A 10-min scalp EEG clip containing seizures, laid out as Babadi and Brown's Fig. 7: the raw trace of the most involved
    channel and six seconds of the ictal rhythm; spectrograms by the cosine-tapered periodogram, multitaper (NW, K = 2NW-1) and
    recipe (b') at the same W, each with its colour bar; to their right, spectra at a pre-ictal and an ictal instant by all three
    plus Welch, and the distribution of the multitaper-minus-smoothed differences. compact=True: trace and three spectrograms only."""
    d = load_excerpt(EEG_CASES[case])
    if d is None:
        print(f"fig10 skipped: data/{EEG_CASES[case]} not found"); return
    x = d["x"]; Fs = float(d["fs"]); chn = str(d["channel"]); ivals = [tuple(v) for v in d["seizures_s"]]; t_on, t_off = ivals[0]
    N = int(win_s * Fs); step = int(step_s * Fs); nfft = 4 * N; W = NW / N; K = 2 * NW - 1
    f = np.arange(nfft // 2) / nfft * Fs; keep = f <= fmax
    Vk, _ = ss.dpss(N, NW); cosw, hp = _recipe(N, NW)
    L = int(round(1.4 / (2 * W))); hann_seg = ss.unit_taper("hann", L); wstep = L // 2
    starts = list(range(0, len(x) - N + 1, step)); tt = (np.array(starts) + N / 2) / Fs
    print(f"fig10 case {case}: channel {chn} (ictal rise {float(d['ictal_rise_db']):.1f} dB), Fs={Fs:.0f}, N={N}, W={W * Fs:.2f} Hz, K={K}, Welch L={L} ({L / Fs:.2f} s), {len(starts)} windows")
    pg, mt, sb, wl = [], [], [], []
    for s0 in starts:
        seg = x[s0:s0 + N]; seg = seg - seg.mean()
        pg.append(ss.periodogram(seg, nfft, taper=cosw)[0][:nfft // 2][keep])
        mt.append(ss.multitaper(seg, Vk, nfft)[0][:nfft // 2][keep])
        sb.append(ss.lag_window_estimate(seg, hp, nfft, taper=cosw)[0][:nfft // 2][keep])
        wl.append(ss.welch_sliding(seg, hann_seg, nfft, step=wstep, overhang=False)[0][:nfft // 2][keep])
    off = 10 * np.log10(Fs)                                          # per hertz: dB re 1 microvolt squared per hertz
    pg, mt, sb, wl = (10 * np.log10(np.array(a).T + 1e-9) - off for a in (pg, mt, sb, wl))
    nus = [2.0, ss.dof_quadratic(ss.quadratic_matrix(N, 0.25, "multitaper", tapers=Vk)),
           ss.dof_quadratic(ss.quadratic_matrix(N, 0.25, "lagwindow", h=hp, taper=cosw)),
           ss.dof_quadratic(ss.quadratic_matrix(N, 0.25, "welch", taper=hann_seg, step=wstep))]
    t_pre = t_pre if t_pre is not None else t_on - 45
    t_ictal = t_ictal if t_ictal is not None else t_on + 0.35 * (t_off - t_on)
    vmin, vmax = (float(np.round(v - off)) for v in SPEC_VLIM)
    t = np.arange(len(x)) / Fs
    imgs = (pg, mt, sb)
    bws = [ss.kernel_stats(H)["bw3"] * Fs for H in (np.abs(np.fft.fft(cosw, 16 * N)) ** 2, ss.kernel_multitaper(Vk, 16 * N), ss.kernel_smoothed(cosw, hp, 16 * N))]
    heads = [f"{n}: {win_s:g} s by {b:.1f} Hz, $\\nu = {v:.1f}$" for n, b, v in zip((NAME["periodogram"], f"{NAME['multitaper']}, $L = {K}$", NAME["smooth"]), bws, nus)]
    dd = mt - sb
    if compact:
        fig, axs = plt.subplots(4, 1, figsize=(W2, 5.8), gridspec_kw={"height_ratios": [0.7, 1, 1, 1], "hspace": 0.3}, sharex=True)
        axs[0].plot(t, x, color="black", lw=0.25); axs[0].set_ylim(-500, 500)
        for on, off in ivals:
            axs[0].axvspan(on, off, color=SHADE, lw=0)
        axs[0].set(ylabel="EEG ($\\mu$V)")
        for a, img, hd in zip(axs[1:], imgs, heads):
            _spectrogram(fig, a, img, tt, f[keep], vmin, vmax); a.set(ylabel=LBL["Hz"]); _head(a, hd)
        fig.colorbar(plt.cm.ScalarMappable(cmap=SPEC_CMAP), ax=axs[0], fraction=0.035, pad=0.015).ax.set_visible(False)
        axs[-1].set(xlabel=LBL["s"], xlim=(0, t[-1]))
        _letters(axs, dx=24); _save(fig, out)
        print(f"fig10 case {case} (compact): channel {chn}; nus {np.round(nus, 1)}; multitaper vs recipe (b') median |diff| {np.median(np.abs(dd)):.2f} dB, 95% {np.percentile(np.abs(dd), 95):.2f} dB")
        return
    fig = plt.figure(figsize=(W2, 7.0 if TUFTE else 6.6))                # Tufte: room for a line of words above each panel
    gs = fig.add_gridspec(4, 2, width_ratios=[3, 1.15], height_ratios=[0.8, 1, 1, 1], hspace=0.58 if TUFTE else 0.42, wspace=0.34)
    a = fig.add_subplot(gs[0, 0]); axes_all = [a]
    a.plot(t, x, color="black", lw=0.25, rasterized=True); a.set_ylim(-500, 500)      # 120,000 points: a picture, not a path, in the PDF
    for on, off in ivals:
        a.axvspan(on, off, color=SHADE, lw=0)
    for tq, c, word in [(t_pre, MATLAB["b"], "(d)"), (t_ictal, MATLAB["r"], "(f)")]:
        if TUFTE:                          # colour is kept for the routes: a mark above the trace, named by the panel that uses it
            a.plot(tq, 1.04, marker="v", ms=3.5, color="black", transform=a.get_xaxis_transform(), clip_on=False)
            a.annotate(word, xy=(tq, 1.04), xycoords=a.get_xaxis_transform(), xytext=(4, -1), textcoords="offset points", fontsize=6.5, va="center")
        else:
            a.axvline(tq, color=c, ls="--", lw=1.0)
    a.set(xlim=(0, t[-1]), ylabel="EEG ($\\mu$V)", xlabel=LBL["s"])
    fig.colorbar(plt.cm.ScalarMappable(cmap=SPEC_CMAP), ax=a, fraction=0.05 if TUFTE else 0.035, pad=0.015).ax.set_visible(False)   # aligns the trace with the spectrograms
    a = fig.add_subplot(gs[0, 1]); axes_all.append(a); z0 = int(t_ictal * Fs); zl = int(6 * Fs)
    a.plot(np.arange(zl) / Fs, x[z0:z0 + zl], color="black", lw=0.5)
    a.set(xlabel=LBL["s"], ylabel="EEG ($\\mu$V)", xlim=(0, 6))
    sp_axes = []
    for r, img in enumerate(imgs):
        a = fig.add_subplot(gs[r + 1, 0]); sp_axes.append(a)
        im = _spectrogram(fig, a, img, tt, f[keep], vmin, vmax, bar=not TUFTE); _head(a, heads[r])
        for tq in (t_pre, t_ictal):
            a.axvline(tq, color="white", ls="--", lw=0.8)
        a.set(ylabel=LBL["Hz"], xlim=(tt[0], tt[-1]))
        if r == 2:
            a.set(xlabel=LBL["s"])
    if TUFTE:                              # one scale, so one bar, with its unit
        cb = fig.colorbar(im, ax=sp_axes, fraction=0.05, pad=0.015, aspect=60); cb.set_label(PSD_UNIT, fontsize=8)
        cb.ax.tick_params(labelsize=6.5, direction="out"); cb.outline.set_linewidth(0.4)
    side = []
    for r, tq in enumerate((t_pre, t_ictal)):
        a = fig.add_subplot(gs[r + 1, 1]); side.append(a); j = int(np.argmin(np.abs(tt - tq)))
        a.plot(f[keep], pg[:, j], color=ROUTE["periodogram"], lw=0.5, label=NAME["periodogram"])
        a.plot(f[keep], mt[:, j], color=ROUTE["multitaper"], lw=1.1, label=f"{NAME['multitaper']}, $L = {K}$")
        a.plot(f[keep], sb[:, j], color=ROUTE["smooth"], lw=0.9, label=NAME["smooth"])
        a.plot(f[keep], wl[:, j], color=ROUTE["welch"], lw=0.8, label=f"{NAME['welch']}, {L / Fs:.1f}-s Hann segments")
        a.set(xlabel=LBL["Hz"], xlim=(0, fmax)); a.set_ylabel(PSD_UNIT.replace(" (", "\n(", 1), fontsize=7.5, linespacing=1.0) if TUFTE else a.set_ylabel("PSD (dB)")
        _grid(a)
        if r == 1:
            lo_, hi_ = a.get_ylim(); a.set_ylim(lo_, hi_ + (0.95 if TUFTE else 0.5) * (hi_ - lo_)); a.legend(loc="upper right", fontsize=6.5 if TUFTE else 5.8, handlelength=1.4, borderaxespad=0.2)
        _head(a, f"{('Before the seizure', 'In the seizure')[r]}, $t = {tq:.0f}$ s")
    a = fig.add_subplot(gs[3, 1]); side.append(a)
    if TUFTE:
        cnt, edges = np.histogram(dd.ravel(), bins=np.linspace(-8, 8, 81))
        a.bar(edges[:-1], cnt / 1e3, width=np.diff(edges), align="edge", color=BAR, lw=0); a.axvline(0, color="black", lw=0.5)
        a.set(xlabel="Difference (dB)", ylabel="Pixels (thousands)", xlim=(-8, 8)); _head(a, "Panel (e) minus panel (g)")
        _note(a, f"median\n|difference|\n{np.median(np.abs(dd)):.1f} dB", size=6.5)
    else:
        a.hist(dd.ravel(), bins=np.linspace(-8, 8, 81), color=BAR, edgecolor="black", lw=0.2)
        a.set(xlabel="Multitaper $-$ smoothed (dB)", ylabel="Pixels", xlim=(-8, 8))
    order = [axes_all[0], axes_all[1], sp_axes[0], side[0], sp_axes[1], side[1], sp_axes[2], side[2]]
    _letters(order, dx=24)
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
SLEEP_EXCERPT = "eeg_sleep_n2.npz"                                                                   # 40 s of stage N2 with 5 s on each side, in data/
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


def fig12_sleep_spindles(fmax=25.0, step_s=0.1, out="fig12_sleep_spindles.png"):
    """Forty seconds of stage N2 sleep with spindles, as spectrograms at three choices of window length and bandwidth,
    each by the multitaper estimate (K = 2NW - 1, at least 1) and by recipe (b'). A white box in each panel is the
    resolution of that panel: the window length by the half-power width of the kernel."""
    from scipy.signal import butter, sosfiltfilt
    d = load_excerpt(SLEEP_EXCERPT)
    if d is None:
        print(f"fig12 skipped: data/{SLEEP_EXCERPT} not found"); return
    x = d["x"]; fs = int(d["fs"]); st = d["stage"]; pad = float(d["pad_s"]); dur = float(d["duration_s"]); chn = str(d["channel"])
    x = x - x.mean(); t = np.arange(len(x)) / fs - pad
    on, off, sg = _spindles(x, fs)
    inside = (on / fs - pad > 0) & (off / fs - pad < dur); on, off = on[inside], off[inside]
    xb = sosfiltfilt(butter(4, [0.3, 35.0], btype="band", fs=fs, output="sos"), x)
    print(f"fig12: channel {chn}, {dur:.0f} s of stage {sorted(set(np.unique(st).astype(int)))} (2 = N2), Fs = {fs}; {len(on)} spindles at "
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
    fig = plt.figure(figsize=(W2, 8.5))
    gs = fig.add_gridspec(5, 2, height_ratios=[0.8, 1, 1, 1, 1.1], hspace=0.5 if TUFTE else 0.38, wspace=0.16)
    a0 = fig.add_subplot(gs[0, :]); axes = [a0]
    a0.plot(t, xb, color="black", lw=0.35, label=f"EEG, {chn}, 0.3$-$35 Hz")
    a0.plot(t, sg - 95, color="0.4" if TUFTE else MATLAB["r"], lw=0.4, label="11$-$16 Hz, offset")
    for o1, o2 in zip(on, off):
        a0.axvspan(o1 / fs - pad, o2 / fs - pad, color=SHADE, lw=0)
    a0.set(xlim=(0, dur), ylim=(-125, 150), yticks=[-100, 0, 100], xlabel=LBL["s"], ylabel="EEG ($\\mu$V)")
    if TUFTE:
        _label(a0, 0.4, 135, f"EEG, {chn}, 0.3$-$35 Hz", "black"); _label(a0, 0.4, -66, "11$-$16 Hz, offset", "0.3")
    else:
        a0.legend(loc="upper right", ncol=2, fontsize=6.5)
    for r, row in enumerate(rows):
        for c, key in enumerate(("mt", "sm")):
            a = fig.add_subplot(gs[r + 1, c]); axes.append(a); _frame(a)
            _head(a, f"{(f'Multitaper, $L = {row[chr(75)]}$', NAME['smooth'])[c]}: {row['T']:g} s by {row['bw'][c]:.1f} Hz, $\\nu = {row['nu'][c]:.1f}$", size=6.5)
            im = a.imshow(10 * np.log10(row[key]), aspect="auto", origin="lower", extent=[row["tc"][0], row["tc"][-1], row["f"][0], row["f"][-1]], vmin=vmin, vmax=vmax, cmap=SPEC_CMAP)
            a.add_patch(plt.Rectangle((1.0, fmax - 2.0 - row["bw"][c]), row["T"], row["bw"][c], fill=False, ec="white", lw=1.0))
            if TUFTE and r == 0 and c == 0:
                a.text(1.0 + row["T"] + 0.7, fmax - 2.0 - row["bw"][c] / 2, "resolution of the panel", color="white", fontsize=6.5, va="center")
            a.set(xlim=(0, dur), ylim=(0, fmax), ylabel=LBL["Hz"] if c == 0 else "")
            if c == 1:
                a.set_yticklabels([])
            if r == len(rows) - 1:
                a.set(xlabel=LBL["s"])
    cb = fig.colorbar(im, ax=axes[1:], fraction=0.02, pad=0.015); cb.set_label(PSD_UNIT)
    cb.ax.tick_params(direction="out" if TUFTE else "in"); cb.outline.set_linewidth(0.4 if TUFTE else 0.6)
    ah = fig.add_subplot(gs[4, :]); axes.append(ah)
    widths = [bw for row in rows for bw in row["bw"]]
    span = max(widths) - min(widths)
    ah.set(xlim=(max(0, min(widths) - 0.35 * span), max(widths) + 0.35 * span),
           ylim=(0.1 if TUFTE else 0.45, len(rows) + 0.8), xlabel="Achieved half-power width (Hz)",
           yticks=list(range(len(rows), 0, -1)),
           yticklabels=[f"{row['T']:g} s, $R={row['B']:g}$ Hz" for row in rows])
    ah.grid(axis="x")
    for i, row in enumerate(rows):
        y = len(rows) - i
        ah.plot(row["bw"], [y, y], color=INK, lw=0.55, zorder=2)
        for c, (bw, nu) in enumerate(zip(row["bw"], row["nu"])):
            ah.plot(bw, y, marker=("o", "s")[c], ms=4, color=ROUTE[("multitaper", "smooth")[c]],
                    ls="none", label=("Multitaper", "Smoothing")[c] if i == 0 else None, zorder=3)
            ah.text(bw, y + 0.12, f"{bw:.1f} Hz\n" + rf"$\nu={nu:.1f}$", ha="center", va="bottom", fontsize=7.4)
        if i == 0:
            ah.text(np.mean(row["bw"]), y + 0.15,
                    rf"$\alpha={row['NW']:g},\ L={row['K']}$: small-$\alpha$ regime",
                    ha="center", va="bottom", fontsize=6.4)
    if TUFTE:
        r0 = rows[-1]; yb = 1 - 0.28
        _label(ah, r0["bw"][0], yb, "Multitaper", ROUTE["multitaper"], ha="center", va="top"); _label(ah, r0["bw"][1] - 0.04, yb, NAME["smooth"], ROUTE["smooth"], ha="left", va="top")
    else:
        ah.legend(loc="lower right", ncol=2, fontsize=6.5)
    _letters(axes, dx=[22] + [22, 6] * len(rows) + [22]); _save(fig, out)


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

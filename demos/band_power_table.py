"""Band powers of the seizure clip by the everyday estimators (run: python demos/band_power_table.py).

For every 2-s window of the clip and each of delta/theta/alpha/beta, the band power (dB) from the Hann periodogram,
the multitaper estimate, the Hann-then-box estimate and Welch's method. Reports (i) the ictal-minus-preictal change per
band by each estimator and (ii) the per-window disagreement of each estimator with the multitaper estimate.
Prints a LaTeX table (paper/bandpower_table.tex) and a summary.
"""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "demos"))
import numpy as np
import specsmooth as ss
from make_figures import _load_elrond, ictal_channel, ELROND_SEG, EEG_CASES

BANDS = [("delta", 1, 4), ("theta", 4, 8), ("alpha", 8, 13), ("beta", 13, 30)]


def band_powers(case="A", win_s=2.0, step_s=1.0, NW=2):
    fname, ivals = EEG_CASES[case]
    X, Fs, ch = _load_elrond(ELROND_SEG / fname); X = X - X.mean(axis=0, keepdims=True)
    N = int(win_s * Fs); step = int(step_s * Fs); nfft = 4 * N; W = NW / N
    f = np.arange(nfft // 2) / nfft * Fs
    Vk, _ = ss.dpss(N, NW); hann = ss.unit_taper("hann", N); hb = ss.box_lag_window(N, W)
    L = int(round(1.4 / (2 * W))); hann_seg = ss.unit_taper("hann", L); wstep = L // 2
    t_on, t_off = ivals[0]
    ic, _ = ictal_channel(X, Fs, t_on, t_off); x = X[ic]
    starts = np.arange(0, len(x) - N + 1, step); tt = (starts + N / 2) / Fs
    ests = {"periodogram": [], "multitaper": [], "Hann+box": [], "Welch": []}
    for s0 in starts:
        sg = x[s0:s0 + N] - x[s0:s0 + N].mean()
        ests["periodogram"].append(ss.periodogram(sg, nfft, taper=hann)[0][:nfft // 2])
        ests["multitaper"].append(ss.multitaper(sg, Vk, nfft)[0][:nfft // 2])
        ests["Hann+box"].append(ss.lag_window_estimate(sg, hb, nfft, taper=hann)[0][:nfft // 2])
        ests["Welch"].append(ss.welch_sliding(sg, hann_seg, nfft, step=wstep, overhang=False)[0][:nfft // 2])
    df = Fs / nfft
    out = {}
    for name, S in ests.items():
        S = np.array(S)
        out[name] = {b: 10 * np.log10(S[:, (f >= lo) & (f < hi)].sum(axis=1) * df) for b, lo, hi in BANDS}
    ictal = np.zeros(len(tt), bool)
    for on, off in ivals:
        ictal |= (tt >= on) & (tt <= off)
    pre = (tt < ivals[0][0] - 5) & (tt >= ivals[0][0] - 125)
    return out, tt, ictal, pre, ch[ic]


def main(cases=("A", "C")):
    lines = [r"\begin{tabular}{llcccc|ccc}", r"\toprule",
             r"clip & band & periodogram & multitaper & Hann+box & Welch & periodogram & Hann+box & Welch\\",
             r" & & \multicolumn{4}{c|}{ictal minus pre-ictal band power (dB)} & \multicolumn{3}{c}{median $|$difference from multitaper$|$ per window (dB)}\\",
             r"\midrule"]
    names = ["periodogram", "multitaper", "Hann+box", "Welch"]
    for case in cases:
        out, tt, ictal, pre, chan = band_powers(case)
        for k, (b, lo, hi) in enumerate(BANDS):
            chg = {n: out[n][b][ictal].mean() - out[n][b][pre].mean() for n in names}
            dev = {n: np.median(np.abs(out[n][b] - out["multitaper"][b])) for n in names if n != "multitaper"}
            label = f"Fig.~\\ref{{fig:eeg}} ({chan.title()})" if case == "A" else f"Fig.~\\ref{{fig:eeg2}} ({chan.title()})"
            lines.append((label if k == 0 else "") + f" & {b} ({lo}--{hi} Hz) & " + " & ".join(f"{chg[n]:+.1f}" for n in names) + " & " + " & ".join(f"{dev[n]:.2f}" for n in names if n != "multitaper") + r"\\")
        if case != cases[-1]:
            lines.append(r"\midrule")
        print(f"case {case}: channel {chan}; windows: {ictal.sum()} ictal, {pre.sum()} pre-ictal, {len(tt)} total")
    lines += [r"\bottomrule", r"\end{tabular}"]
    tex = "\n".join(lines)
    (ROOT / "paper" / "bandpower_table.tex").write_text(tex + "\n")
    print(tex)


if __name__ == "__main__":
    main()

"""The best width of the parabolic kernel for the alpha peak of the EEG-like spectrum (Section II-D3), and the asymptotic
efficiencies of the box and the Gaussian relative to the parabola.

1. The formula b_opt = (15 C_h / (N Delta))^(1/5) |S / S''|^(2/5) at 10 Hz for N = 400 samples at 200 Hz (a 2-s record), for
   the untapered periodogram (C_h = 1) and the 25% cosine taper of recipe (b'). The parabola's half-power width is sqrt(2) b.
2. The half-power width that minimizes the exact error at 10 Hz, from the exact mean m and variance v of the smoothed
   periodogram of the Gaussian process (demos/tuned_comparison.moments), for two criteria:
     relative mean square error of the estimate, ((m - S)^2 + v) / S^2;
     mean square error in dB, from the chi-square approximation with nu = 2 m^2 / v (tuned_comparison.screen).
3. The asymptotic RMS-error ratios, each kernel at its own best width: (mu2^(1/2) int G^2)^(2/5) relative to the parabola.
Run: python demos/optimal_kernel.py
"""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "demos"))
import numpy as np
import specsmooth as ss
import tuned_comparison as tc
from scipy.optimize import minimize_scalar

FS, N, F0 = 200.0, 400, 10.0                      # sampling rate (Hz), record length, frequency of the alpha peak (Hz)
DELTA = 1 / FS


def psd_hz(hz):
    return tc.eeg_psd(np.asarray(hz) / FS)        # the EEG-like PSD, in units per cycle/sample, at frequencies in Hz


def errors(taper, T, b):
    """Relative MSE and dB MSE at F0 of the periodogram tapered with `taper` and smoothed with a parabola of half-width b Hz."""
    S = psd_hz(F0)
    V, c = tc.bank(tc.lag_matrix(taper[:, None], ss.parabolic_lag_window(N, b * DELTA)))
    m, v = tc.moments(V, c, T, [F0 * DELTA])
    return ((m[0] - S) ** 2 + v[0]) / S ** 2, tc.screen(m, v, np.array([S])) ** 2


def exact_optimum(taper, T):
    """Half-power widths sqrt(2) b (Hz) that minimize the relative MSE and the dB MSE at F0 (bounded scalar minimization in b;
    on a 0.01-Hz grid over [0.6, 2.2] Hz each criterion, with either taper, falls and then rises, with one change of slope)."""
    return tuple(np.sqrt(2) * minimize_scalar(lambda b: errors(taper, T, b)[j], bounds=(0.6, 2.2), method="bounded",
                                              options={"xatol": 1e-5}).x for j in (0, 1))


if __name__ == "__main__":
    S = float(psd_hz(F0)); h = 1e-3
    S2 = float((psd_hz(F0 + h) - 2 * psd_hz(F0) + psd_hz(F0 - h)) / h ** 2)       # per Hz^2
    print(f"EEG-like spectrum at {F0:g} Hz: S = {S:.3f}, S'' = {S2:.2f} per Hz^2; N = {N}, N Delta = {N * DELTA:g} s")
    T = tc.Toeplitz(tc.acf(tc.eeg_psd, N))
    for name, taper in (("no taper", np.ones(N) / np.sqrt(N)), ("25% cosine taper", ss.unit_taper("tukey", N, alpha=0.25))):
        Ch = N * np.sum(taper ** 4)
        b_opt = (15 * Ch / (N * DELTA)) ** 0.2 * abs(S / S2) ** 0.4
        rel_w, db_w = exact_optimum(taper, T)
        print(f"  {name:17s} C_h = {Ch:.3f}: formula b_opt = {b_opt:.3f} Hz, half-power width {np.sqrt(2) * b_opt:.2f} Hz; "
              f"exact optimum of the half-power width: relative MSE {rel_w:.3f} Hz, dB MSE {db_w:.3f} Hz")
    u = np.linspace(-3, 3, 600001); du = u[1] - u[0]
    b = 1 / np.sqrt(2); par = np.where(np.abs(u) <= b, 0.75 / b * (1 - (u / b) ** 2), 0.0)
    box = np.where(np.abs(u) <= 0.5, 1.0, 0.0)
    sg = 1 / (2 * np.sqrt(2 * np.log(2))); gau = np.exp(-0.5 * (u / sg) ** 2) / (sg * np.sqrt(2 * np.pi))
    eff = lambda G: (np.sqrt(np.sum(u ** 2 * G) * du) * np.sum(G ** 2) * du) ** 0.4           # RMS error at the best width, up to a constant
    print("Asymptotic RMS error at the best width, relative to the parabola: "
          + ", ".join(f"{n} {eff(G) / eff(par):.4f}" for n, G in (("box", box), ("Gaussian", gau))))

"""Expected-value smoothing kernels: E[estimate](f) = (H * S)(f) for every quadratic estimator here."""
import numpy as np
from .estimators import acs, _place_lags


def signed_freq(nfft):
    f = np.arange(nfft) / nfft
    return np.where(f >= 0.5, f - 1.0, f)


def normalize_area(H):
    """Scale so that the kernel integrates to 1 over one cycle (grid spacing 1/nfft)."""
    H = np.asarray(H, float)
    return H / (H.sum() / len(H))


def kernel_from_lag(h, nfft):
    """Kernel on the grid from a symmetric lag sequence h (tau = 0..len-1)."""
    h = np.asarray(h, float)
    N = len(h)
    hfull = np.concatenate([h[:0:-1], h])
    return normalize_area(np.fft.fft(_place_lags(hfull, N, nfft)).real)


def kernel_multitaper(tapers, nfft, weights=None):
    """sum_k w_k |V_k(f)|^2 (unit area): the kernel of a (weighted) multitaper estimate."""
    Vk = np.abs(np.fft.fft(tapers, nfft, axis=0)) ** 2
    H = Vk.mean(axis=1) if weights is None else Vk @ np.asarray(weights, float)
    return normalize_area(H)


def kernel_smoothed(taper, h, nfft):
    """Kernel of 'taper, then smooth with lag window h': |Taper(f)|^2 convolved with H(f)."""
    taper = np.asarray(taper, float)
    N = len(taper)
    h = np.asarray(h, float)[:N]
    hfull = np.concatenate([h[:0:-1], h])
    return normalize_area(np.fft.fft(_place_lags(acs(taper) * hfull, N, nfft)).real)


def kernel_box(nfft, W):
    return normalize_area((np.abs(signed_freq(nfft)) <= W).astype(float))


def kernel_gaussian(nfft, sigma_f):
    return normalize_area(np.exp(-0.5 * (signed_freq(nfft) / sigma_f) ** 2))


def kernel_wosa(N, seg_len, overlap, nfft, taper=None):
    """Kernel of Welch/WOSA: the segment taper's |W(f)|^2 (segments share one kernel)."""
    w = np.ones(seg_len) / np.sqrt(seg_len) if taper is None else np.asarray(taper, float)
    return normalize_area(np.abs(np.fft.fft(w, nfft)) ** 2)


def kernel_stats(H, W=None):
    """Resolution / leakage summaries of a unit-area kernel on the nfft grid.

    Returns dict: bw3 (half-power full width, cycles/sample), mass_out (fraction outside |f|<=W, if W given),
    mass_out_2bw (fraction outside twice the half-power half-width), peak_sidelobe_db (beyond 1.5 x half-power half-width),
    resolution (smallest tone separation at which two equal tones show a dip, Rayleigh-style).
    """
    H = np.asarray(H, float)
    nfft = len(H)
    fs = signed_freq(nfft)
    df = 1.0 / nfft
    peak = H.max()
    main = H >= peak / 2
    bw3 = main.sum() * df
    hw = bw3 / 2
    out = {"bw3": bw3}
    if W is not None:
        out["mass_out"] = H[np.abs(fs) > W].sum() / H.sum()
        out["mass_out_2W"] = H[np.abs(fs) > 2 * W].sum() / H.sum()
    out["mass_out_2bw"] = H[np.abs(fs) > 2 * hw].sum() / H.sum()
    far = H[np.abs(fs) > 1.5 * hw]
    out["peak_sidelobe_db"] = 10 * np.log10(far.max() / peak) if far.size and far.max() > 0 else -np.inf
    # two-tone resolution: smallest separation d at which the response to equal tones at +-d/2,
    # R(f) = H(f-d/2) + H(f+d/2), dips by at least 3 dB between the tones (Rayleigh-style, ripple-proof)
    Hs = np.fft.fftshift(H)
    c = nfft // 2
    res = np.nan
    for k in range(1, nfft // 4):
        R = np.roll(Hs, k) + np.roll(Hs, -k)
        if R[c - k:c + k + 1].min() < 0.5 * R.max():
            res = 2 * k * df
            break
    out["resolution"] = res
    return out


def fit_box_gaussian(H_target, nfft, widths, sigmas):
    """Least-squares fit of a box (half-width w) convolved with a Gaussian (std s) to a kernel; returns (w, s, err, H_fit).

    This is the idea behind the 2016 MATLAB script a_Step8_ApproximateMTSA_Kernel_WithBoxPlusGaussian.m.
    """
    tau = np.arange(nfft // 2)
    best = (None, None, np.inf, None)
    Ht = normalize_area(H_target)
    for w in widths:
        hb = np.sinc(2 * w * tau)
        for s in sigmas:
            h = hb * np.exp(-2 * (np.pi * s * tau) ** 2)
            Hf = kernel_from_lag(h, nfft)
            err = np.sum((Hf - Ht) ** 2) / np.sum(Ht ** 2)
            if err < best[2]:
                best = (w, s, err, Hf)
    return best

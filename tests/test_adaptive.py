"""Thomson's adaptive weights and the 'few tapers, then smooth' hybrid."""
import numpy as np
import specsmooth as ss


def test_adaptive_weights_reduce_to_equal_weights_on_white_noise():
    rng = np.random.default_rng(0)
    N, NW = 256, 4
    V, lam = ss.dpss(N, NW)
    x = rng.standard_normal(N)
    Sa, _, nu = ss.multitaper_adaptive(x, V, lam, 1024)
    Se = ss.multitaper(x, V, 1024)[0]
    assert np.median(np.abs(10 * np.log10(Sa / Se))) < 0.2
    assert np.median(nu) > 13.0                       # close to 2K = 14


def test_adaptive_weights_remove_the_leakage_floor():
    rng = np.random.default_rng(3)
    N, NW, nfft = 1024, 4, 4096
    V, lam = ss.dpss(N, NW)
    f = np.arange(nfft // 2) / nfft
    truth = ss.ar_psd(ss.AR4, f)
    errs_eq, errs_ad, nus = [], [], []
    for _ in range(20):
        x = ss.ar_process(ss.AR4, N, rng)
        Sa, _, nu = ss.multitaper_adaptive(x, V, lam, nfft)
        errs_ad.append(np.median(np.abs(10 * np.log10(Sa[:nfft // 2] / truth))))
        errs_eq.append(np.median(np.abs(10 * np.log10(ss.multitaper(x, V, nfft)[0][:nfft // 2] / truth))))
        nus.append(np.median(nu[:nfft // 2]))
    assert np.mean(errs_ad) < 0.8 * np.mean(errs_eq)
    assert np.mean(nus) < 13.0                        # the price: fewer effective tapers where the spectrum is low


def test_hybrid_quadratic_form_matches_estimator():
    rng = np.random.default_rng(1)
    N = 128
    V, _ = ss.dpss(N, 2)
    h = ss.box_lag_window(N, 2 / N)
    x = rng.standard_normal(N)
    nfft = 512; j = 100
    S = ss.hybrid_estimate(x, V, h, nfft)[0][j]
    Q = ss.quadratic_matrix(N, j / nfft, "hybrid", tapers=V, h=h)
    xf = x.astype(complex)
    assert abs((xf.conj() @ Q @ xf).real / S - 1) < 1e-9

# Legacy MATLAB (2013–2016), kept for reference only

Imported from Box `Brandon - PHI/!@@@-Work/Papers_InProgress/SpectralEstimation/` on 2026-09-26.
Nothing here is maintained; the Python package `specsmooth/` replaces it. Left out on purpose:
`csa/` (an unrelated 2014 Neurology paper misfiled here) and `codegen/` (MATLAB Coder junk).

## What the scripts did, and where that lives now

| MATLAB | Purpose | Python |
|---|---|---|
| `a_Step1_WOSA*.m` | Welch / WOSA baseline (Bronez 1992 comparison) | `specsmooth.welch`, `kernel_wosa` |
| `a_Step2_UnpackMTSA*.m`, `fcnGetStuff.m`, `dpsschk.m` | pull tapers out of Chronux `mtspectrumc`, build the effective MT kernel = mean of squared taper spectra | `specsmooth.dpss`, `dpss_all`, `kernel_multitaper` |
| `a_Step2B_SpectralSmoothing.m`, `fcnSmSpect*.m`, `fcnSmSpectGauss.m` | Gaussian-windowed periodogram convolved with the MT kernel; two-peak resolution check | `lag_window_estimate`, `smooth_on_grid`, `kernel_smoothed`, `kernel_stats` |
| `a_Step3_*`, `fcnFind*Res*.m`, `fcnFindLeakage*.m`, `fcnFindVariance*.m`, `a_Step6_ResLeakVar_v2.m`, `a_Step7_GetRV.m`, `fcnGetRV_ConvEnv.m` | resolution / leakage / variance of MT vs smoothed spectra, Monte Carlo on Gaussian noise | `kernel_stats`, `dof_quadratic`, `mc_white_noise`, `demos/make_figures.py::fig3_tradeoff` |
| `a_Step4_GenerateGaussianNoise.m`, `a_Step5_SpectralEstimates_GN.m`, `fcnGenSignalFromPSD.m`, `fcnGetSignal*.m`, `TimeseriesFromPSD.m` | test signals with known PSD | `specsmooth.signals` |
| `a_Step8_ApproximateMTSA_Kernel_WithBoxPlusGaussian.m`, `papouliswin.m` | fit the MT kernel with box ⊛ Papoulis/Gaussian | `fit_box_gaussian` (and the exact result it was groping toward: `tests/test_identity.py`) |
| `a_AR4_MT_Mimic*.m`, `a_MT_Mimic.m`, `fcn_Get_MT_Mine.m`, `fcn_Get_SP.m`, `a_MyMT_SpecImplementation.m` | 2013 first attempts: Gaussian smoothing of a periodogram to mimic MT on AR(4) and on `DATA_Spike.mat` | `demos/verify_equivalence.py`, `demos/make_figures.py::fig2_ar4` |
| `a_DPSS_play.m`, `a_Fig1_VaryLengths.m`, `a_Fig2TruncationFreq.m`, `a_FourierSeries*.m`, `a_CyclicConvolution.m`, `a_TestTimeFreqConv.m`, `a_conv_mult_play.m`, `Example1D.m`, `Example2D.m`, `FourierT.m`, `IFourierT.m`, `figures.m` | 2013 tutorial/figure experiments for the leakage section | not ported (figure ideas noted in STATUS.md) |
| `mtspecgram_mbw.m`, `mtspectrumc.m`, `mtfftc.m`, `getparams.m`, `getfgrid.m`, `change_row_to_column.m`, `welch_mbw.m`, `interpMatrix.m`, `jbfill.m`, `license.txt` | Chronux / File Exchange helpers | scipy |
| `simulatedEEG.m`, `phasereset.m`, `peak.m`, `noise.m`, `Makinen*.m`, `dipole.mat`, `meanpower.mat`, `nickloc31.locs`, `read.me` | Rafal Bogacz's EEGLAB phase-reset simulator (third party) | not ported |
| `third_party/stoica_moses/` | companion code to Stoica & Moses, *Spectral Analysis of Signals* (`daniellse.m` is the box-smoothed periodogram) | not ported |

Data: `DATA_Spike.mat` (19 ch × 1281, 128 Hz), `DATA_Sedation.mat` (19 ch × 2001), `InterestingSignal.mat` (1 ch, 7 min, 128 Hz; used by `demos/make_figures.py::fig4_eeg`), `TAPERS.mat` (2000 × 19 DPSS).

"""specsmooth: multitaper spectral estimation viewed as kernel smoothing of a periodogram.

Python replacement for the 2013-2016 MATLAB scripts (see legacy_matlab/).
"""
from .tapers import sinc_toeplitz, dpss_all, dpss, unit_taper, sine_tapers, sinc_window
from .estimators import (periodogram, multitaper, multitaper_adaptive, hybrid_estimate, acs, lag_window_estimate, box_lag_window,
                         gaussian_lag_window, smooth_on_grid, welch, welch_sliding, quadratic_matrix)
from .kernels import (signed_freq, normalize_area, kernel_from_lag, kernel_multitaper, kernel_smoothed,
                      kernel_box, kernel_gaussian, kernel_wosa, kernel_stats, fit_box_gaussian,
                      kernel_quadratic, eigen_tapers, dof_from_weights)
from .metrics import equivalent_dof, dof_quadratic, mc_white_noise
from .signals import AR4, ar_process, ar_psd, signal_from_psd, two_tones

__all__ = [n for n in dir() if not n.startswith("_")]

# Three equivalent routes to estimating power spectra: averaging, multitaper, and smoothing

Working repo for a short methods paper (with code): the multitaper (Slepian / Thomson) spectral
estimate is, to a precise degree, a periodogram smoothed with a designed kernel. Smoothing is easier
to explain and to control (one taper, one kernel width) and, done right, gives nearly the same
estimate. The question the paper answers is what "done right" and "nearly" mean, with numbers.

**Start with [STATUS.md](STATUS.md)** — what exists, whether the proof is complete and correct, which
figures and demos the paper needs, and the plan.

## Layout

| path | what |
|---|---|
| `specsmooth/` | the Python package (numpy/scipy): tapers, estimators, expected-value kernels, variance/resolution/leakage metrics, test signals |
| `demos/verify_equivalence.py` | numerical proof of the central identity plus the kernel / dof tables quoted in STATUS.md |
| `demos/make_figures.py` | the paper figures (`figures/fig0..fig5`) |
| `demos/everyday_comparison.py` | Table I: the estimators at one bandwidth over 300 records of the AR(4) process |
| `demos/tuned_comparison.py` | Table III: every family tuned to its best setting, on the AR(4) process and an EEG-like spectrum |
| `demos/taper_and_kernel.py` | which single taper and smoothing kernel work best (the basis of recipe b') |
| `demos/frequency_domain_multitaper.py` | multitaper from one FFT, and its split into a smoothed periodogram plus a cross term |
| `demos/rounded_kernel.py` | smoothing with the rounded-corner kernel built from the K tapers, with and without a taper built from them |
| `demos/local_width.py` | how much a width chosen separately at each frequency would gain |
| `demos/kernel_matched.py` | one taper plus a smoothing kernel matched to the multitaper kernel: same bias, different estimates |
| `demos/band_power_table.py` | Table IV: band powers on the two seizure clips |
| `demos/fit_single_window.py` | least-squares search for the single window closest to the multitaper estimator |
| `tests/` | the claims as pytest tests |
| `paper/` | the manuscript in IEEE TBME format (`main.tex`, built with the journal's `ieeecolor2.cls` and `generic.sty`), its supplement (`supplement.tex`), the frozen single-column draft v6 (`main_onecolumn.tex`), the review map (`ARGUMENT.md`), the 2015 draft (`main_2015.tex`) and older notes |
| `notes/` | the 2001 typed manuscript "Periodogram Averaging with a Sliding Window" (scanned, with annotations) and lecture notes |
| `legacy_matlab/` | the 2013–2016 MATLAB, unmaintained; `legacy_matlab/README.md` maps each script to the Python that replaced it |
| `references/` | third-party PDFs (git-ignored; a local copy only) |

## Quickstart

```bash
pip install -r requirements.txt        # numpy, scipy, matplotlib, pytest
python -m pytest -q                    # the identity, kernel, and variance claims as tests
python demos/verify_equivalence.py     # prints the numbers, writes figures/verify_equivalence.png
python demos/make_figures.py           # writes figures/fig0_pedagogy.png ... fig5_slepian_fill.png
```

## The result in one paragraph

Let `v_k` be the N Slepian sequences for bandwidth W and `λ_k` their concentrations, and
`J_k(f) = Σ_t v_k[t] x[t] e^{-i2πft}`. Then, exactly,

    Σ_{k=0}^{N-1} λ_k |J_k(f)|²  =  ∫_{f-W}^{f+W} |X(f')|² df'

so the eigenvalue-weighted multitaper estimate over *all* tapers **is** the raw periodogram averaged
over a box of half-width W, and (Welch/Nuttall–Carter) it is also the average of periodograms of a sinc
window of half-bandwidth W slid across the record one sample at a time; the Slepians are that sliding
window's principal components (`tests/test_banks.py`). The usual estimator keeps only the K = 2NW−1 tapers with λ_k ≈ 1; it is
the box-smoothed periodogram minus the broadband leakage that the discarded tapers carry. That is
why "taper, then smooth" reproduces multitaper: the taper removes the leakage, the kernel sets the
bandwidth. The price is variance (a single taper wastes data at the segment edges):
at NW = 4 the equivalent dof are 17.2 (raw periodogram + box, identical to the eigen-weighted
multitaper), 14.0 (K = 7 multitaper), 8.9 (Hann + box), 7.3 (Bohman + box).

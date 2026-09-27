# The paper's presentation and arguments (draft v5, 2026-09-26)

**v5 (open issues from the review, no further review runs):** adaptive-weight multitaper and the hybrid implemented;
a new ensemble table (300 realizations, three regions of the spectrum) replaces single-realization numbers, and the
three facts are restated on it: *warning* (untapered smoothing errs by 13.7 dB where the spectrum is low, Hann by 1.5),
*price* (ν = 8.9 vs 14.0, 2.3 vs 1.8 dB at the peaks, or a kernel 13/N wide to match ν), *surprise* (default MT errs by
6.3 dB off-peak; K = 6 gives 2.1, adaptive weights 1.7 at ν = 6.5; level with Hann + box over the band). Welch has the
lowest errors here and pays in two-tone resolution; two Slepians + box is the best smoother. Related work is now
"what is known" / "what we add"; §3.4, §4, §5.4 and the EEG section are split into single-purpose paragraphs.


**v4 (after one review → revise → review iteration with the bdsp-core agents; reports in `review/reports/`):**
abstract in two paragraphs with the numbers and earned verbs; introduction gains a News paragraph, a five-item
contributions list and the cost-of-not-knowing; §5.4 reordered as three facts (warning, price, surprise) plus a
Welch remark, with the regime of the "can beat" claim stated; eq. (7) carries its K/2NW factor; a Limitations
section; honest data-availability statement (seizure clips via BDSP); figures re-authored under one style.


**v3 changes (after the comparison with Babadi & Brown 2014):** figures now sit beside the text that cites them;
a graphical abstract (three routes, one estimate) opens the paper; the method-by-method figures use one repeated
motif (estimate over truth on top, kernel over the true peaks below); §5 and §6 merged into "The three methods are
one" with four short paragraphs on what the everyday versions trade; the two recipes are boxed; Figs 1–3 of v2 and
the full taper-bank figure moved to a supplement (S1–S4); the EEG demonstration is now a ten-minute ELROND clip
with a focal seizure (chirp on Pz), in the Babadi–Brown layout; ~7,800 words (from ~10,000), 20 pages incl. refs.


A high-level map of `main.tex` for review. Each claim is tagged **exact**, **approximate** (with numbers),
or **numerical** (verified in `tests/` or a figure). The last section lists the weak points and the
decisions that need your call.

## Thesis in three sentences

1. Three ways of estimating a spectrum are in everyday use: average periodograms of segments
   (Bartlett, Welch; Kay's textbook route), smooth one periodogram with a kernel (Daniell,
   Blackman–Tukey; what pipelines do to spectrograms), and Thomson's multitaper method (the
   recommended, "optimal", opaque one).
2. They are the same estimator: every one is a quadratic form in the data, i.e. a bank of tapers with
   weights, and the three methods are three ways of building the bank from one window (shift it in
   time, shift it in frequency, or take orthogonal tapers). One correspondence is exact (Thomson 1982,
   eq. 8.3): multitaper with all N Slepians weighted by their eigenvalues = the periodogram averaged
   over a box of half-width W = the average of periodograms of a sinc window slid across the record.
   The Slepians are the principal components of that sliding window.
3. The everyday multitaper estimator (K = 2NW−1 tapers, equal weights) is the same estimate with its
   leaky components removed. Neither intuitive route can do that subtraction, but both can prevent the
   leakage by tapering first, at a quantified price: a single taper costs about a third of the degrees
   of freedom (its eigen-weight ladder is sloped); short Welch segments cost kernel shape instead.

## Order of presentation

| § | what it does | figure |
|---|---|---|
| 1 Introduction | the three methods as most people meet them; the claim that they are one; Thomson (8.3) as the unnoticed bridge; roadmap | — |
| 2 The periodogram and the one picture | periodogram; noisy (ν = 2) and leaky (Fejér kernel); *every* estimator = truth convolved with an expected-value kernel H, plus noise; three numbers per estimator: resolution, leakage, ν | Fig 0 (c, g) |
| 3.1 Average periodograms of segments | Bartlett/Welch; kernel = segment window's spectral window; ν ≈ 2 × segments; segment length = bandwidth; Kay's account | — |
| 3.2 Smooth one periodogram | Daniell/Blackman–Tukey; kernel = box ⊛ Fejér; ν ≈ 2·2NW; leakage unchanged; averaging of a sliding window is a smoothing (classical; App. C) | — |
| 3.3 Taper first | shapes the skirts; needed by both routes; does nothing for variance | Fig 0 (d, h), Fig 1 |
| 3.4 How wide? Blur against noise | the bandwidth choice as bias–variance: noise falls as 4.3·√(2/ν) dB, blur grows once 2W approaches the narrowest feature; rule: set 2W to the width of the narrowest feature that must survive | Fig 0 (b, e, f) |
| 4 Multitaper as usually presented | Slepians, concentrations, eigenspectra, K = 2NW−1; the kernel is the sum of the K spectral windows, nearly a box; Thomson's optimality (max concentration among orthogonal banks) | Fig 5 |
| 5.1 Every estimator is a bank of tapers | Q = Σ c_k v_k v_kᵀ; time-shifted / frequency-shifted / orthogonal banks; kernel and ν from the bank; ν = 2(Σc)²/Σc² ("eigen-weight ladder"); flat ladder = Thomson, sloped = single-taper smoother, nearly flat = Welch 50 % Hann | Fig 6 |
| 5.2 The identity | Thomson (8.3), boxed; holds per record to 1e-13 | Fig 7 top |
| 5.3 Why it is true | A = sinc Toeplitz = box kernel as a matrix; Slepians = its eigenvectors; "the Slepians are the box kernel, factored" | App. A |
| 5.4 Averaging is smoothing; Slepians = principal components of a sliding sinc | unit-step Welch with overhang has Toeplitz Q with lags r_w(τ); sinc window ⇒ Q → A with error ∝ 1/L (12 % at L = N, 1.5 % at 4N, 0.17 % at 32N) | Fig 7 top, App. C |
| 5.5 The everyday estimator | MT_K ≈ box-smoothed periodogram − leaky remainder (the dropped eigenspectra); no single window in either family can subtract it (LS fits return the sliding sinc / rect + box); tapering prevents it instead; everyday trio agrees on EEG to within its own noise | Fig 7 bottom |
| 6.1 Smoothing cannot undo leakage | AR(4), 65 dB range: raw + box floors 20 dB high; Hann + box tracks to ~2 dB | Fig 2 |
| 6.2 A single taper costs variance; short segments cost kernel shape | Table 1 at N = 256, W = 4/N: ν = 17.2 (raw + box = all-taper MT), 14.0 (MT K = 7), 8.9 (Hann + box), 7.3 (Bohman + box), 11.1 (Hann + Gaussian), 13.4 (Welch 7 Hann segments). Welch has the *most* ν at matched half-power width but a kernel with no flat top and broadband leakage equal to the untapered box; Bronez's MT advantage appears at matched leakage | Fig 3, Table 1 |
| 6.3 The last Slepian taper leaks | λ₆ = 0.94 ⇒ −22 dB skirt; MT K = 7 floors 10 dB high on AR(4); worse at N = 128; adaptive weights / K = 2NW−O(log NW) fix it; Hann + box can beat default MT | Fig 2 |
| 6.4 Hybrid | Riedel–Sidorenko–Thomson 1994: a few tapers, then smooth | — |
| 7 Recipe: three routes to one estimate | exact trio (a: all Slepians, weights λ/2NW; c: raw periodogram ⊛ box; b: sliding sinc window, unit step, overhang, ÷2NW), windows *derived* from equality with (a); everyday trio (a′: K tapers / adaptive; c′: Hann + box; b′: Hann segments of length ≈ 1.4/(2W), 50 % overlap) with expected ν, resolution, leakage; when to prefer which | Fig 7 |
| 7 EEG: seizures | two ten-minute ELROND clips: one focal seizure (harmonic chirp, C4) and a clip with two seizures (harmonic stacks 20 → 3 Hz, Fp1); bandwidth set by the rule (harmonic peak ≈ 1.5 Hz wide ⇒ 2W = 2 Hz); Hann periodogram / MT (NW = 2, K = 3) / Hann + box spectrograms; pre-ictal and ictal spectra by all three + Welch; MT vs Hann + box median 1.5 and 1.2 dB per pixel; Table 2: band powers (ictal rise 17–18 dB by every estimator within 0.6 dB; per-window |diff| from MT ≈ 1 dB Hann+box, 0.5 dB Welch) | Figs 8–9, Table 2 |
| 9 Prior work | Thomson (8.3); quadratic-estimator view (Percival–Walden, Riedel–Sidorenko, Walden); Welch/Nuttall–Carter for averaging = smoothing; Hansson-Sandsten's Welch approximation of MT (the sinc window makes it exact); Abreu–Romero, Karnik et al.; Bronez; Riedel–Sidorenko–Thomson 1994; the neuroscience tutorials and Kay do not mention any of it | — |
| 10 Conclusion | one estimator, three constructions; MT parameters mapped onto taper + width; the bandwidth rule; what the hand-rolled routes gain and lose | — |
| App. A | five-line proof of (8.3) | |
| App. B | quadratic forms: bank, kernel, ν (trace formula and ladder formula, agree to 3 %), resolution/leakage metrics, grid smoothing = lag window | |
| App. C | C.1 continuous time: Welch shifts as a Riemann sum, Δ → 0, the delta-function step ⇒ Σ over all shifts = raw periodogram ⊛ \|W\|² (the 2001 note, Gaussian window ⇒ Gaussian kernel; the shift integral must include overhanging positions). C.2 discrete finite record: unit-step with overhang ⇒ Toeplitz Q ⇒ lag-window estimate, exact, no limit; tapered-then-smoothed is *not* a sliding-window average. C.3 sinc ⇒ A, Slepians = principal components; Welch 50 % as approximation; the Fig 3 caveat | |

## The claims, by status

**Exact (proved, tested to machine precision)**
- (8.3): Σ_k λ_k |J_k(f)|² = ∫_{f−W}^{f+W} I. `tests/test_identity.py`.
- Sliding a window with unit step and overhang = lag-window estimate with lag window r_w (any window,
  finite record). `tests/test_banks.py::test_sliding_window_is_lag_window_estimate`.
- Eigen-tapers of the box-smoothed periodogram = Slepians, weights λ_k/2NW.
- Every quadratic estimator is a weighted multitaper estimator (eigendecomposition); kernel = Σ c_k|V_k|².

**Exact in a limit / approximate with a rate**
- Sliding sinc → A as 1/L (numbers above). Fig 7 uses L = 16N: 0.06 dB max difference.
- ν = 2(Σc)²/Σc² at interior frequencies: within 3 % of the exact trace formula for all estimators used.
- Log-noise std ≈ 4.3·√(2/ν) dB once ν is more than a few (at ν = 2 the true value is 5.6 dB, not 4.3).

**Numerical findings (figures)**
- Fig 0: bias–variance sweep; whole-band optimum W ≈ 8/N, peak-region optimum W ≈ 4–5/N, rule 2W ≈
  narrowest feature. N = 1024, AR(4), 300 realizations.
- Fig 2 / Table 1 / Fig 3: leakage, ν, resolution per method (N = 256 for the table, 1024 for Fig 2).
- Fig 7: exact trio identical; everyday trio within noise on EEG (largest differences in the valleys
  between harmonics, where MT's 7th taper fills in).
- LS fits (not in the paper, STATUS §2e): the Welch window / (taper, kernel) pair closest to MT_K in
  Frobenius norm is essentially the sliding sinc / rect + box; distance 0.27–0.29 vs 0.86 for Hann + box.

## What is claimed as new (and what is not)

Not new, and cited: (8.3) itself; the quadratic-estimator unification; averaging = smoothing
(Welch, Nuttall–Carter); MT vs Welch (Bronez); MT vs smoothed periodogram (Riedel–Sidorenko–Thomson).

Presented as new *exposition and bookkeeping*: the three-constructions-of-one-bank picture with the
eigen-weight ladder as the explanation of variance cost; the sliding-sinc form of (8.3) and "Slepians
= principal components of the sliding sinc" (an immediate consequence of two classical results; we
say so); exact ν/leakage/resolution accounting per route; the observation that Hann + box beats
default equal-weight MT on high-dynamic-range spectra; the bandwidth rule with the bias–variance
figure; the three-route recipe; the EEG validation.

## Weak points and decisions for you

1. **Title.** "Three Equivalent Routes to Spectral Estimation: Averaging, Smoothing, and Multitaper" (chosen by
   M.B.W., 2026-09-26); alternatives considered: "Three Routes to One Spectrum", "Multitaper Spectral Analysis Is Smoothing", "Averaging, Smoothing, Multitaper: One
   Spectral Estimator", "The Unity of Spectral Estimators".
2. **The exact trio is the leaky estimator.** Everything that is exactly equal is the untapered,
   λ-weighted, all-taper estimate, which nobody should use on a coloured spectrum. The paper says this
   plainly (§5.5, §7) and treats the everyday trio as "close, not identical". Is that framing acceptable,
   or do you want the exact trio de-emphasised?
3. **"Optimal window" for the everyday routes.** For the K-taper estimator there is no exact
   single-window equivalent (it is a projection; the LS fits confirm). The paper's answer is: derive
   the exact windows (sinc; rect + box), then remove leakage by tapering rather than by subtraction,
   and quantify the cost. If you wanted a derived "best Hann-like window", that is a different
   (approximate) optimisation and is not in the paper.
4. **Welch.** At matched half-power width Welch has *more* ν than MT (Fig 3, Table 1), contrary to the
   folk statement (and to v1's Appendix C, now corrected). The paper resolves it via kernel shape
   (no flat top, base 2× the half-power width, broadband leakage like the untapered box) and Bronez's
   matched-leakage comparison. Worth a check that this reading of Bronez is fair.
5. ~~Adaptive weights~~ implemented and compared (Table 3, Fig S2). The EEG figures still use equal weights.
6. ~~Band-power table and second seizure case~~ done (Table 2, Fig 9); data use confirmed by the user.
7. **Fig 0's N changed** from 256 to 1024 so that W = 4/N is "just right" (at N = 256 the AR(4) peaks
   are only 1.7/N wide, narrower than the Hann main lobe, and W = 4/N merged them). Fig 1/Table 1 remain
   at N = 256 and Fig 2 at N = 1024.
8. **Length.** Sixteen pages with seven figures and an appendix; IEEE SPL (4 pp) is no longer realistic.
   J. Neurosci. Methods, NeuroImage methods, or IEEE TBME (the Babadi–Brown venue) fit the audience.
9. **The 2001 note** is now App. C.1 (continuous time, limit of all overlaps), credited as "an unpublished
   2001 note by one of us". Cite it as a preprint instead?

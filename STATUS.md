# Project status — 2026-09-26

Reviewed: the Box material (2001 manuscript, 2013–2016 MATLAB, 2015 draft and whiteboard), then
re-derived and numerically verified the central claim in Python (`specsmooth/`, `tests/`, `demos/`),
cross-checked once in MATLAB R2025b. Numbers below are from `demos/verify_equivalence.py`
(N = 256, NW = 4, K = 7) unless stated.

## 1. What exists

| item | date | state |
|---|---|---|
| `notes/SlidingWindow_NoteAndDraft.pdf` — "Periodogram Averaging with a Sliding Window: Equivalence of the Blackman-Tukey and Bartlett Methods", two typed versions (Aug 17 and Aug 19, 2001) + two sets of typed notes on nonparametric spectral estimation (Jan 2001), all with handwritten annotations; 39 scanned pages, no text layer | 2001 | complete 5-page derivation; never submitted |
| `paper/main.tex`, `paper/SpectralEstimationReview.pdf` — "Spectral Estimation Review Paper", Biswal & Westover | May 2015 (compiled Jun 2016) | 130 lines: intro/convolution-theorem/leakage prose, spectral representation theorem, and the bias identity E|J(f)|² = |W|² ∗ S. Stops there. |
| `paper/WhiteBoardOutline.JPG` | Aug 2015 | the experiment plan: MT(W,T,K) vs WOSA(T,W′) vs smoothing(T,W″); hold resolution R fixed, compare variance V and leakage L |
| `legacy_matlab/` — 60 own scripts + Chronux/Stoica/Bogacz helpers, EEG snippets (`DATA_Spike.mat`, `InterestingSignal.mat`) | 2013–2016 | build the effective MT kernel, smooth a Gaussian-tapered periodogram with it, measure resolution/leakage/variance; last script (Jun 2016) fits the MT kernel with box ⊛ Papoulis |
| `~/Downloads/Project Advertisement.docx` — student project pitch with Sunil Nagaraj (U Twente) | undated (≈2018–19) | nothing in Box followed from it |

### 2e. The third thread (2026-09-26): averaging periodograms
Kay's textbook route (average periodograms of segments) joins the family exactly: with unit step and the window
allowed to overhang the zero-padded record, Q_{tt'} = r_w(t−t') (the window's autocorrelation), i.e. the raw
periodogram smoothed with |W(f)|² (Welch 1967 / Nuttall–Carter 1982; the 2001 manuscript). With a sinc window
w_t = 2W sinc(2Wt) this is the sinc Toeplitz matrix A, so **Welch with a sliding sinc window = box-smoothed
periodogram = Thomson all tapers λ-weighted**, with error ∝ 1/L in the window length (12 % at L = N, 1.5 % at 4N,
0.17 % at 32N; `tests/test_banks.py`). The Slepians are the principal components of the sliding sinc window.
The K-taper estimator has no exact single-window equivalent in either family: least-squares fits of a segment
window or of a (taper, kernel) pair to Q_K return essentially the sliding sinc / rect + box (Frobenius distance
0.27–0.29 vs 0.86 for Hann + box), i.e. no single window can subtract the leaky remainder. Fig 3 caveat: at
matched *half-power* width Welch (Hann, 50 %) has the most dof of all methods but a kernel with no flat top and
broadband leakage equal to the untapered box; Bronez's advantage for MT appears only at matched leakage.
`specsmooth`: `sinc_window`, `welch_sliding`, `quadratic_matrix('welch', step=, overhang=)`, `eigen_tapers`,
`dof_from_weights`, `kernel_quadratic`.

## 2. Is the proof complete, and is it correct?

**Short answer: the equivalence with multitaper was never written down; what was written is correct
but is not that theorem. The missing piece is a five-line argument, now proven and tested.**

### 2a. The 2001 manuscript (Blackman-Tukey ≡ sliding-window Bartlett)
Claims: averaging Gaussian-windowed periodograms over a window that slides with unbounded overlap
equals convolving the raw periodogram |x_T(ω)|² with the kernel |w(ω)|². The chain of equalities
(eqs. 6–8) is right. Two things to tighten if it is reused: the record is silently treated as
infinite/periodic (windows run off the ends; for a finite record the segment count and edge
weighting need a sentence), and the limit N → ∞ of the sum is really a Riemann sum, fine. This is
the WOSA ↔ lag-window equivalence known since Welch (1967) and Nuttall & Carter (1982); it is a
good *section*, not a paper.

### 2b. The 2015 draft (bias of a tapered periodogram)
Derives E[|J(f)|²] = |W(f)|² ∗ S(f) from the spectral representation theorem. Correct and standard
(Percival & Walden 1993, §6.3). Notational slips to fix: the Fourier sign convention differs between
eq. (1) and eq. (2); limits mix ±1/2 and ±F_s/2 while Δ factors are inconsistent; the covariance of
dZ needs a Dirac delta, not "Δ(f′−f)"; the ACS derivation has a stray E and an f/f′ mix-up in the
E[dZ*dZ] line. Everything after "results for LTI systems / results for windowing" is empty. There is
no multitaper section at all.

### 2c. The claim itself: multitaper = smoothed periodogram
Never stated in either draft. The 2016 script `a_Step8_ApproximateMTSA_Kernel_WithBoxPlusGaussian.m`
found *numerically* that the K-taper kernel ≈ box ⊛ smooth bump. The exact statement is:

> Let A be the N×N Toeplitz matrix A[t,t′] = sin(2πW(t−t′))/(π(t−t′)), A[t,t] = 2W. Its eigenvectors
> are the Slepian sequences v_k (unit norm) with eigenvalues λ_k ∈ (0,1), so A = Σ_k λ_k v_k v_kᵀ.
> With J_k(f) = Σ_t v_k[t] x[t] e^{−i2πft},
>
>   Σ_{k=0}^{N−1} λ_k |J_k(f)|² = Σ_{t,t′} A[t,t′] x[t] x[t′] e^{−i2πf(t−t′)} = Σ_τ h_τ ŝ_τ e^{−i2πfτ},
>
> with h_τ = sin(2πWτ)/(πτ) and ŝ_τ = Σ_t x[t] x[t+τ]. Since h_τ is the inverse transform of the
> box 1{|f| ≤ W} and ŝ_τ that of the periodogram I(f) = |X(f)|², the right side is ∫_{f−W}^{f+W} I(f′) df′.
> Corollary: Σ_k λ_k ≈ 2NW with λ_k ≈ 1 for k < 2NW−1 and ≈ 0 after, so the standard estimator
> (1/K) Σ_{k<K} |J_k|² equals the box-averaged periodogram (1/2W)∫_{f−W}^{f+W} I/N minus the terms
> with small λ_k, which are exactly the ones carrying broadband leakage.

Verified (`tests/test_identity.py`, `demos/verify_equivalence.py`; MATLAB cross-check 1.95e−15):
identity holds to 2e−15; the K-taper kernel and box ⊛ Fejér kernel differ only in the sidelobes.

**Literature pass done (see LITERATURE.md).** The identity is Thomson's own equation (8.3), Proc.
IEEE 1982, §VIII, p. 1070, including his remark that the smoothed periodogram and the eigenspectra
agree "only for white spectra" because the higher-order eigenspectra carry the bias. The general
quadratic-estimator equivalence is in Percival & Walden (1993, ch. 7), Riedel & Sidorenko (1995,
§5) and Walden (2000); the empirical multitaper-vs-smoothed-periodogram comparison was done by
Riedel, Sidorenko & Thomson (1994, Phys. Plasmas). No practitioner-facing account exists (Babadi &
Brown 2014, Prerau 2017 and Wikipedia do not mention it), which is consistent with Kay not knowing
it. So: cite (8.3) up front, and make the paper the exposition, the quantified price of the
single-taper route, and the EEG recipe.

### 2d. What "nearly the same" turns out to mean (the honest part)
1. **Raw periodogram + box has the same variance as multitaper but keeps the periodogram's leakage.**
   dof 17.2 vs 14.0; kernel mass beyond 2W: 1.4 % vs 0.2 %; peak sidelobe −17 dB vs −22 dB. On white
   noise + tone the two estimates agree to 0.4 dB (median); on the AR(4) process (70 dB dynamic
   range) they disagree by 6–7 dB. Smoothing in frequency cannot undo leakage; only tapering (or
   prewhitening) can. `fig2_ar4_leakage.png`.
2. **Taper, then smooth, matches multitaper's leakage control at a variance cost.** Hann + box: dof
   8.9 (−37 % vs K = 7), sidelobe −36 dB, mass beyond 2W ≈ 0. Bohman (Papoulis) + box: dof 7.3, −31 dB.
   On 7 min of real EEG (`fig4`) the multitaper and Hann + box spectrograms differ by 1.2 dB median,
   4.4 dB at the 95th percentile, i.e. at the level of the estimators' own noise (std of a dof-9 vs
   dof-14 log-estimate is ≈ 2.0 vs 1.6 dB).
3. **The equal-weight K = 2NW−1 multitaper is itself leaky.** Its last taper has λ ≈ 0.94 and puts
   the kernel sidelobe at −22 dB; on AR(4) the K = 7 estimate floors 10 dB above the truth where a
   Hann + box estimate tracks it. Short records make it worse: at N = 128, NW = 3 the median |dB|
   error against the true AR(4) spectrum over 200 realizations is 13.4 dB (raw + box), 7.1 dB (MT, K = 5), 3.5 dB (MT, K = 4),
   2.1 dB (Hann + box) — `demos/outputs/short_record_leakage.txt` (the earlier 20/12/6/2 were one realization). Dropping the last taper helps; Thomson's adaptive
   weights are the real fix. The paper must compare against the adaptive-weight estimator, not only
   equal weights, or the smoothing method will look better than it should.
4. **Sine (Riedel-Sidorenko) tapers and the DPSS behave alike in this comparison** (`fig3`).

## 3. Figures and demos the paper needs

Done (`demos/make_figures.py`, `figures/`):
- **Fig 1** kernels at equal design bandwidth, linear and dB: box, MT, raw+box, Hann+box, Bohman+box.
- **Fig 2** AR(4) stress test: who leaks.
- **Fig 3** the whiteboard plan: dof, broadband leakage, and peak sidelobe vs *measured* half-power
  bandwidth for MT (K = 2NW−1, 2NW−2), sine-taper MT, raw/Hann/Bohman + box, Hann + Gaussian, Welch.
- **Fig 4** real EEG spectrograms (128 Hz, 4 s epochs): Hann periodogram, MT, Hann + box, difference.
- `verify_equivalence.png` and the printed tables (identity error, kernel stats, dof).
- **Fig 0 (pedagogy)**, redone 2026-09-26: (a) the record (N = 1024, AR(4)) with the Hann taper; (b) Monte-Carlo
  RMS dB error vs box half-width W, split into blur bias and noise, whole band and peak region; (c–f) periodogram,
  Hann periodogram, Hann + box W = 4/N ("just right": 2W ≈ the 7/N peak width, ν = 8.9), Hann + box W = 24/N
  ("too smooth": 2W > the 30/N peak separation, ν = 50); (g–j) each kernel drawn over the true peaks for scale.
  The rule the paper states (§3.3 of main.tex): set 2W to the width of the narrowest feature that must survive;
  whole-band MSE prefers W ≈ 8/N, the peak region W ≈ 4–5/N.
- **Fig 5** the Slepian windows tile the band (Thomson eq. 8.3 as a running sum).
- **Fig 6 (taper banks)**, 2026-09-26: every quadratic estimator is a bank of tapers with weights (eigen-decomposition
  of Q). Columns: Welch (time-shifted bank), Hann + Gaussian (frequency-shifted bank), sliding sinc = raw + box =
  Thomson all tapers λ-weighted (eigen-tapers are the Slepians), Thomson K = 7. Rows: the bank, its eigen-tapers, the
  eigen-weight ladder with ν = 2(Σc)²/Σc², the kernel. Flat ladder = Thomson; sloped ladder = single-taper smoother
  (this is *why* it costs dof); Welch 50% Hann is nearly flat (ν 13.4 for 7 segments) but its kernel has no flat top.
- **Fig 7 (three routes, one answer)**, 2026-09-26, on an 8-s EEG epoch: exact trio (Thomson all-λ / raw + box /
  sinc window of length 16N slid one sample at a time) agree to 3e-13 dB and 0.06 dB; everyday trio (Thomson K = 7 /
  Hann + box / Welch 9 Hann segments) differ by a median 1.6 and 0.9 dB, i.e. their own noise.

- **Draft v3 figure set (2026-09-26, after comparing with Babadi & Brown 2014):** Fig 1 `fig9_graphical_abstract`
  (exact trio on EEG), Fig 2 `fig0b_estimates_kernels` (estimate-over-kernel motif, 4 estimators), Fig 3
  `fig0a_record_sweep` (record + bias–variance sweep), Fig 4 `fig5_slepian_fill`, Fig 5 `fig6_taper_banks` (compact),
  Fig 6 `fig8_everyday_motif` (raw+box / Thomson K / Hann+box / Welch on the AR(4) record), Fig 7 `fig7_everyday_eeg`,
  Fig 8 `fig10_eeg_seizure` (ELROND clip, case A: `~/ELROND/Data/segments/sub-I0003175292344_20170823183337.mat`,
  seizure 292–366 s, channel C4 = largest 2–20 Hz ictal rise, NW = 2 in 2-s windows), Fig 9 `fig11_eeg_two_seizures`
  (case C: `sub-I0003175080331_ses-84_445.mat`, seizures 49–183 and 298–441 s, Fp1), Table 2 = `paper/bandpower_table.tex`
  from `demos/band_power_table.py` (ictal-minus-pre-ictal band power by estimator; per-window |diff| from MT ≈ 1 dB
  Hann+box, 0.5 dB Welch). The user confirmed the de-identified BDSP clips may be shown. Supplement: S1 `fig1_kernels`, S2 `fig2_ar4_leakage`,
  S3 `fig3_tradeoff`, S4 `fig6_taper_banks_full`. `fig4_eeg_spectrograms` (7-min InterestingSignal) is no longer in
  the paper. Floats fixed with placeins + [!htb]; recipes boxed (framed); §5 and §6 merged; ~7,800 words.

- **Review iteration (2026-09-26, bdsp-core paper-agents via Bedrock, see `review/`):** baseline v3 scored 70.6/102
  (truthfulness agent timed out; Scholar agents skipped at the user's request). Applied: figures re-authored under one
  style (`COLOR`, panel letters, 300 dpi, viridis with shared 10–45 dB scale, convergence inset in Fig 1); consistency
  fixes (Welch leakage/base, width rule, Bohman, 0.7 dB, sliding-sinc exactness, eq. 7 normalization K/2NW); new
  `demos/fit_single_window.py`; tests for 2N/32N convergence and Monte-Carlo dof; abstract/intro rewritten with
  numbers and a contributions list; Limitations section; data-availability statement. v4 review running.

- **Open issues from the review addressed (2026-09-26, draft v5, no further review runs):** Thomson's adaptive weights
  (`ss.multitaper_adaptive`) and the few-tapers-then-smooth hybrid (`ss.hybrid_estimate`, `quadratic_matrix('hybrid')`)
  implemented and tested; `demos/everyday_comparison.py` scores eight estimators over 300 realizations at N = 1024, 256,
  128 in three regions (whole band / peaks / low spectrum). Findings now in the paper (Table 3): the single realization
  shown in Fig 6 was unusually leaky (its 11.6 / 4.6 / 1.3 / 0.8 dB are 5.3 / 2.3 / 1.4 / 1.0 dB over the ensemble);
  adaptive weights bring MT level with Hann + box over the band (1.4 dB) at ν = 6.5 where the spectrum is low; at the
  peaks MT wins (1.8 vs 2.3 dB); matched-ν Hann + box needs a 13/N kernel vs 7.4/N; two Slepians (NW = 2) + box give
  ν = 13.3 with a −34 dB side lobe (the Riedel–Sidorenko–Thomson hybrid, best of the smoothing family); Welch has the
  lowest errors and ν = 17.6 but two-tone resolution 10.8/N vs 9.0/N. Ictal harmonics measured (0.9 Hz wide in 2-s
  windows, 7.4 Hz apart). Related work rewritten as known / added; paragraphs split; recipes relabelled (a, b, c) =
  (multitaper, smooth, average). New references were added from memory (Scholar agents were skipped); all were checked on 2026-09-26, see
  `review/reports/reference_verification.md`:
  Thomson & Chave 1991, Bruns 2004, Wahba 1980, Hurvich 1985, Haley & Anitescu 2017, Harris 1978, Mitra & Pesaran 1999,
  Bokil et al. 2010, Satterthwaite 1946.

- **Prose line edit (2026-09-26):** the manuscript text was rewritten against the blader/humanizer checklist (25
  patterns from Wikipedia's "Signs of AI writing"): staged contrasts, cleft openers, aphorisms, idioms, the packaged
  "warning / price / surprise" triad, rhetorical questions and colon- or semicolon-joined clauses were replaced with plain
  declarative sentences; section headings made plain. Numbers and citations verified unchanged against the backup
  (`review/backups/*_pre_humanize.tex`).

- **TBME format (2026-09-26):** `paper/main.tex` now uses the official TBME template (December 2025; `ieeecolor2.cls`,
  `generic.sty`): structured abstract (249 words), index terms, IMRaD sections (Methods II, Results III, Discussion IV),
  figure*/table* floats, references renumbered in order of first citation, appendices A–C, AI-use statement in the
  Acknowledgment. Figs. S1–S4 moved to `paper/supplement.tex`. Length 15 pages as of the prior-work paragraph on "hopelessly obsolete" (three references spill onto page 15); TBME's standard is 8 and the maximum 12
  (with the Editor-in-Chief's permission; overlength charges apply beyond 8). `paper/main_onecolumn.tex` is draft v6 frozen.

- **Submission details (2026-09-26):** repository link in Data and Code Availability (the repository is public); NIH
  funding (R01HL161253, R01NS126282) in the first-page footnote; IRB statement in Methods (Stanford #83833, BIDMC
  #2016P000058, MGH #2013P001024, waiver of consent). All 27 references checked against Crossref, publisher pages and
  source texts; Thomson 1990 (quadratic-inverse) added as reference 28. Still open in `paper/main.tex`: submission date,
  corresponding e-mail, and the wording of the AI-use sentence. The repository has no LICENSE file.

- **Tuned comparison (2026-09-26, not yet in the manuscript):** `demos/tuned_comparison.py` searches each family's
  parameters and scores the best setting on 1000 simulated records (output in `demos/outputs/tuned_comparison.txt`).
  RMS dB error over the band, best setting of each family: AR(4) N=256: one taper then smooth 2.42, Welch 2.48, Slepian
  multitaper 2.67 (equal weights) and 2.89 (adaptive); AR(4) N=1024: 1.37, 1.39, 1.46, 1.52; EEG-like N=400: 1.89, 1.90,
  1.97, 2.00; EEG-like N=1024: 1.36, 1.36, 1.40, 1.41. The routes tie to within 2-8% once each is tuned, and the smoothed
  periodogram is never behind. The best single taper is a light Tukey taper (10-25%) with a parabolic kernel, not Hann
  with a box; at the paper's fixed W = 4/N, Hann-then-box trails multitaper on the EEG-like spectrum (2.2 vs 1.8 dB).

Still needed:
- ~~Adaptive-weight multitaper~~ done (in Fig S2 and Table 3; not in the Fig S3 sweep).
- **Matched-resolution comparison** as a table: for each method pick the parameter giving the same
  half-power bandwidth, report dof and leakage (Fig 3 read at fixed x).
- **Time-frequency version of Fig 3** on EEG: bias/variance of band powers (delta/theta/alpha/beta)
  from MT vs taper+smooth against a long-window reference, so a neurophysiologist sees the practical
  consequence.
- **A recipe figure/box**: "for multitaper (N, NW): use Hann (or Bohman) taper, box kernel of half-width
  NW/N, expect dof ≈ 0.63·2K" with the constant read off Fig 3.
- Optional: the 2001 sliding-window equivalence as one figure (overlap → dof, converging to the
  lag-window value), tying the old manuscript in.

## 4. Plan for the short paper

Target: IEEE Signal Processing Letters (4 pp) or a J. Neurosci. Methods short communication; code
as this repo. Outline:
1. Motivation: MT is the standard but opaque; practitioners already smooth spectrograms.
2. Setup and the bias identity (from the 2015 draft, corrected).
3. Theorem + corollary (§2c) and its reading: MT = box-smoothed periodogram with leakage removed.
4. What the corollary does *not* give: leakage (Fig 2), variance cost of a single taper (Fig 3).
5. Recipe and EEG demonstration (Fig 4, band-power table).
6. Relation to prior work (Thomson 1982; Walden/McCoy/Percival 1995; Riedel-Sidorenko 1995;
   Bronez 1992's performance comparison; Welch/Nuttall-Carter for §2a).

## 5. To-do, in order
1. ~~Literature pass on §2c~~ done, see LITERATURE.md: build on Thomson (8.3); add the Riedel-Sidorenko-Thomson 1994 hybrid (few tapers, then smooth) to Fig. 3.
2. ~~Adaptive-weight MT in `specsmooth`~~ done 2026-09-26.
3. ~~Fig 0~~ done; ~~band-power EEG table~~ done (Table 2, two seizure clips).
4. ~~Rewrite `paper/main.tex` around the outline in §4~~ done twice: v1 (identity-centred), v2 on 2026-09-26
   (three routes: averaging / smoothing / multitaper are one estimator; Fig 0 bias–variance; Figs 6–7). Still to
   fix: the notational slips in §2b are moot (the 2015 text is no longer used); the band-power EEG table; adaptive
   weights; the title is now "Three Equivalent Routes to Estimating Power Spectra: Averaging, Multitaper, and Smoothing" (M.B.W., 2026-09-26).
5. ~~Push the repo to GitHub~~ pushed to `origin/main` (bdsp-core/spectral-smoothing-vs-multitaper, private) on 2026-09-26. Still to do: share with co-authors.

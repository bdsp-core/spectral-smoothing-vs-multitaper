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

**This is almost certainly not new.** It is the "ideal band-limited estimator" argument in Thomson
(1982) and Percival & Walden §7.1, and the multitaper-as-quadratic-window view of Walden, McCoy &
Percival (1995), Riedel & Sidorenko (1995), Hansson & Salomonsson (1997), Prieto et al. (2007). Do
a literature pass before claiming a theorem; the paper's contribution is the exposition and the
practical recipe with quantified trade-offs, which is what the advertisement promised anyway.

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
   error against the true AR(4) spectrum is ≈ 20 dB (raw + box), 12 dB (MT, K = 5), 6 dB (MT, K = 4),
   2 dB (Hann + box) — `tests/test_identity.py`. Dropping the last taper helps; Thomson's adaptive
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

Still needed:
- **Fig 0 (pedagogy)** one panel each: truncation → sinc leakage; taper → less leakage, wider lobe;
  a periodogram and its box-smoothed version; the MT kernel drawn as the sum of the K taper spectra
  (the 2015 draft's "Figure XX" placeholders; ideas in `legacy_matlab/a_Fig1_VaryLengths.m`, `a_Fig2TruncationFreq.m`).
- **Adaptive-weight multitaper** in Figs 2–3 (implement Thomson's iterative weights in `specsmooth.estimators`).
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
1. Literature pass on §2c (half a day). Decide the framing accordingly.
2. Adaptive-weight MT in `specsmooth`; regenerate Figs 2–3.
3. Fig 0 and the band-power EEG table.
4. Rewrite `paper/main.tex` around the outline in §4; fix the notational slips in §2b.
5. Push the repo to GitHub (not done; owner/visibility is your call) and share with a co-author.

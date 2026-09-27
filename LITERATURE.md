# Literature pass: is "multitaper = smoothed periodogram" known? (2026-09-26)

Question: is the identity in STATUS.md §2c (eigenvalue-weighted multitaper over all N Slepian
tapers = periodogram box-averaged over [f−W, f+W]) and the practical claim built on it already in
the literature? Sources checked: the PDFs in `references/` (Thomson 1982, Bronez 1992, Riedel &
Sidorenko 1995, Prieto et al. 2007, Babadi & Brown 2014, Percival's chapter-7 course notes), the
arXiv texts of Riedel-Sidorenko-Thomson 1994, Abreu & Romero 2017, Karnik-Romberg-Davenport 2022,
Haley & Anitescu 2017, Astfalck et al. 2024, Hansson-Sandsten 2012, the abstract of Walden 2000,
and the Prerau 2017 tutorial and Wikipedia as proxies for what practitioners see.

## Verdict

**The identity is Thomson's own, equation (8.3) of the 1982 paper.** The general fact that any
kernel-smoothed quadratic estimator is another quadratic estimator, hence a multitaper estimator
with eigenvector tapers and eigenvalue weights, is in Percival & Walden (1993, ch. 7), Riedel &
Sidorenko (1995, §5) and Walden (2000). The empirical comparison the 2015 whiteboard planned was
done in 1994 on tokamak data by Riedel, Sidorenko and Thomson. **What does not exist, as far as
this pass found, is a practitioner-facing account that states (8.3) as the bridge, quantifies the
price of the single-taper route, and turns it into a recipe for EEG-type data.** Kay not knowing
the result is consistent with where it lives: a half-page in Thomson's Section VIII, and a
Physics of Plasmas paper.

## What each source says

**Thomson (1982), Proc. IEEE 70:1055, §VIII "Relations between eigenestimates and the
periodogram", p. 1070.** Expands the Dirichlet kernel in the DPSWFs, obtains the periodogram for
|f − f₀| < W as I(f) = (1/N)|Σ_{k=0}^{N−1} U_k(N,W; f−f₀) y_k(f₀)|² (eq. 8.2), and then: "If one now
uses a uniform smoother of width 2W the result is Ī_W(f₀) = (1/2NW) Σ_{k=0}^{N−1} λ_k(N,W) Ŝ_k(f₀)
(8.3). This shows the smoothed periodogram to be equivalent to a weighted average of the
eigenspectra with weights independent of frequency. In addition the average is over *all* N
eigenspectra, not just the 2NW whose large eigenvalues imply that the information contained in
these coefficients is of local origin." He then compares λ_k with the least-squares weights and
concludes "the two are equal *only* for white spectra S(f) ≡ σ², and that otherwise the higher
order eigenspectra contribute significant bias to the periodogram." That is exactly Claim 1 and
the caveat on Claim 2 in STATUS.md. Section IV also contains the observation that the K-taper
spectral window approximates a box on [−W, W].

**Percival & Walden (1993), ch. 7 (course notes in `references/chpt-07_...pdf`).** "Quadratic
spectral estimators": lag-window, direct and WOSA estimators are all x^H Q x; any positive
semidefinite Q factors as Q = A Aᵀ, so every such estimator is a multitaper estimator with the
columns of A as tapers. The reverse direction of our identity, in textbook form.

**Riedel & Sidorenko (1995), IEEE TSP 43:188, §5 "Kernel smoothers".** Smoothing a quadratic
estimator with matrix [a_nm] by a kernel G of half-width w gives the quadratic estimator with
b_nm = a_nm ∫_{−w}^{w} G(g) e^{i2π(m−n)g} dg; for the box kernel they write b_nm = a_nm
sin(2π(m−n)w)/(2π(m−n)w) explicitly. Theorem 5.1: the periodogram smoothed over the whole band by
the Epanechnikov kernel decomposes *exactly* into the discrete minimum-bias tapers. Their example
(Tukey-tapered periodogram + box, w = 0.01) has eigenvectors close to sinusoids. They do not spell
out the untapered-periodogram + box ↔ Slepian case, but it is a_nm = 1 in their formula.

**Riedel, Sidorenko & Thomson (1994), Phys. Plasmas 1:485 (arXiv 1804.00003).** Compares smoothed
tapered periodogram, adaptive multitaper and hybrids on TFTR microwave-scattering data spanning
four decades. "The adaptive multitaper estimate and the smoothed tapered periodogram are virtually
identical, showing that the difference in the estimates tends to zero as N increases." For 300-point
segments "even an optimized smoothed tapered periodogram has a 24 % larger relative RMSE than the
hybrid method" (four tapers, then kernel-smoothed), which beats both pure methods. Explanation:
"multitaper analysis uses all of the possible degrees of freedom in the bandwidth [f−W, f+W], while
the smoothed tapered periodogram has its effective kernel support broadened by at least 1/(2NΔt)
due to the spectral window support ... the advantage of multitaper decreases as N increases." This
is our Fig. 3 finding (single taper wastes dof), with the same mechanism.

**Walden (2000), Biometrika 87:767 (abstract only; full text paywalled).** "Cross-spectral
estimators are represented by a weighted average of orthogonally-tapered cross-periodograms, with
the weights corresponding to a set of rescaled eigenvalues. Such a structure not only encompasses
the Thomson estimators, using Slepian and sine tapers, but also Welch's weighted overlapped segment
averaging estimator and lag window estimators including frequency-averaged cross-periodograms."

**Abreu & Romero (2017), IEEE Trans. Inf. Theory (arXiv 1703.08190).** Theorem 2.1: the L¹ distance
between the K-taper spectral window (1/K)Σ|U_k|² and the ideal box 1_{[−W,W]}/(2W) is ≲ log N / K,
"a fact discovered by Thomson". Eq. (2.13) names Thomson's "modified method, where all Slepian
sequences are used as tapers ... weighted with the eigenvalues."

**Karnik, Romberg & Davenport (2022), IEEE Trans. Inf. Theory (arXiv 2103.11586).** Lemma 8
(Appendix B) is the identity in matrix form: Ψ(f) = Σ_{k=0}^{N−1} λ_k Ŝ_k(f) = x* E_f B E_f* x with B the
sinc Toeplitz (prolate) matrix, evaluated on a grid by FFT through a circulant extension of the
sinc sequence. They use it purely as a fast algorithm for approximating the K-taper estimate, not
as an interpretation. Their non-asymptotic bounds also say K = 2NW − O(log NW) tapers (fewer than
the customary 2NW−1 or 2NW−2) give better leakage protection "especially in scenarios where the
spectrum has a large dynamic range", which matches our AR(4) result that the last taper leaks.

**Hansson-Sandsten (2012), EUSIPCO.** Approximates the Thomson estimator by a Welch structure by
matching covariance matrices. Relevant to the 2001 manuscript in `notes/` (WOSA ↔ smoothing).

**Bronez (1992), IEEE TSP 40:2941.** Multitaper vs WOSA only, at matched leakage/resolution/variance.
No smoothed periodogram.

**Haley & Anitescu (2017)**, "Optimal bandwidth for multitaper spectrum estimation"; **Astfalck,
Sykulski & Cripps (2024)**, "Bias correction of quadratic spectral estimators" (arXiv 2410.12386):
both work inside the quadratic-estimator view that unifies lag-window, multitaper and Welch; the
view is current, not just historical.

**Practitioner-facing sources.** Babadi & Brown (2014, IEEE TBME review; local copy
`references/Babadi_Brown_2014_ReviewOfMultitaperSpectralAnalysis_IEEE-TBME.pdf`), Prerau et al. (2017,
Physiology tutorial) and the Wikipedia article describe multitaper as averaging orthogonally
tapered periodograms and compare it with the periodogram and a Hann estimate; none relates it to a
smoothed periodogram or a lag-window estimator.

## Not checked

Full text of Walden (2000); the 1993 and 2020 Percival & Walden books themselves (only the chapter
7 course notes); Thomson (1990) "Quadratic-inverse spectrum estimates"; Nuttall & Carter (1982) on
WOSA vs lag-window. Worth a library afternoon before submission; the 2020 P&W edition may have
(8.3) as an exercise.

## Consequence for the paper

Cite Thomson (8.3) in the first paragraph and build on it, rather than presenting the identity as
new. The contribution is the exposition plus what this repo adds: exact dof accounting per method
(STATUS.md §2d), the kernel comparison at matched resolution (Fig. 3), the EEG demonstration
(Fig. 4), the observation that a tapered periodogram + box beats the equal-weight K = 2NW−1
estimator on high-dynamic-range spectra (anticipated by Thomson's "only for white spectra" and
quantified by Karnik et al.), and a one-box recipe. Riedel-Sidorenko-Thomson 1994 should be cited
as the prior quantitative comparison, and the hybrid "few tapers, then smooth" they recommend is a
natural addition to Fig. 3.

## Added 2026-09-26: the averaging thread

Kay (1988, *Modern Spectral Estimation*) presents nonparametric estimation as averaging periodograms
(Bartlett 1948; Welch 1967). Nuttall & Carter (1982, Proc. IEEE 70:1115) give the general "combined time and
lag weighting" quadratic estimator that contains both Welch and Blackman–Tukey; Hansson-Sandsten (EUSIPCO 2012)
approximates Thomson's estimator by a Welch structure. The paper's new statement (§5.4, App. C): with a sinc
window slid one sample at a time over the zero-padded record, Welch's method *is* Thomson's λ-weighted
all-taper estimate, and the Slepians are the principal components of the sliding window. Not found stated in
this form in the sources checked; it is an immediate consequence of Welch/Nuttall–Carter plus Thomson (8.3), so
the paper presents it as such and cites both. Worth checking Hansson-Sandsten 2012 in full before submission.

# Reference verification, 2026-09-26

All 28 entries in `paper/main.tex` were checked. Each entry was matched to a record in Crossref, to the publisher or
repository page, or to the source text itself. The sentence that cites each entry was then compared with the abstract or
text of the source. The nine entries that had been added from memory are marked with an asterisk.

## Bibliographic details

| # | Entry | Result | Source of the check |
|---|---|---|---|
| 1 | Bartlett 1948, Nature 161 | correct; issue no. 4096 added | doi:10.1038/161686a0 |
| 2 | Welch 1967, IEEE Trans. Audio Electroacoust. 15(2) | correct | doi:10.1109/TAU.1967.1161901 |
| 3 | Kay 1988, Modern Spectral Estimation (book) | correct | Open Library record, Prentice-Hall, Englewood Cliffs, 1988 |
| 4 | Daniell 1946, discussion, Suppl. J. R. Stat. Soc. 8 | correct; issue no. 1 added, journal name put in the usual order | Symposium issue 8(1), pp. 27-97 (Bartlett's paper is doi:10.2307/2983611); reference list of Percival and Walden (2020) |
| 5 | Blackman and Tukey 1958 (Dover book) | correct; Dover printings are catalogued as 1958 and 1959 | Open Library; the same text is Bell Syst. Tech. J. 37(1) 185-282 and 37(2) 485-569 |
| 6 | Thomson 1982, Proc. IEEE 70(9) | correct; eq. (8.3) and the quoted phrase "only for white spectra" are in Sec. VIII | doi:10.1109/PROC.1982.12433; local copy of the paper |
| 7 | Babadi and Brown 2014, IEEE TBME 61(5) | correct | doi:10.1109/TBME.2014.2311996 |
| 8 | Prerau et al. 2017, Physiology 32(1) | correct | doi:10.1152/physiol.00062.2015 |
| 9 | Slepian 1978, Bell Syst. Tech. J. 57 | correct; issue no. 5 added | doi:10.1002/j.1538-7305.1978.tb02104.x |
| 10 | Nuttall and Carter 1982, Proc. IEEE 70(9) | correct | doi:10.1109/PROC.1982.12435 |
| 11 | Percival and Walden 1993; 2nd ed. 2020 | correct | Open Library (1993); doi:10.1017/9781139235723 (2020) |
| 12 | Wahba 1980, JASA 75(369) * | correct | doi:10.1080/01621459.1980.10477441 |
| 13 | Hurvich 1985, JASA 80(392) * | correct | doi:10.1080/01621459.1985.10478207 |
| 14 | Haley and Anitescu 2017, IEEE SPL 24(11) * | correct | doi:10.1109/LSP.2017.2719943 |
| 15 | Riedel and Sidorenko 1995, IEEE TSP 43(1) | correct | doi:10.1109/78.365298 |
| 16 | Satterthwaite 1946, Biometrics Bull. 2(6) * | correct, pp. 110-114 | doi:10.2307/3002019 |
| 17 | Hansson-Sandsten 2012, EUSIPCO | correct, pp. 440-444; conference name written out | the paper itself (EURASIP proceedings); Lund University record |
| 18 | Riedel, Sidorenko and Thomson 1994, Phys. Plasmas 1(3) | correct | doi:10.1063/1.870794; arXiv:1804.00003 |
| 19 | Thomson and Chave 1991, book chapter * | correct: Advances in Spectrum Analysis and Array Processing, vol. 1, ch. 2, pp. 58-113 | scanned copy of the chapter; cited identically by Haley and Anitescu (2017) |
| 20 | Karnik, Romberg and Davenport 2022, IEEE TIT 68(7) | correct | doi:10.1109/TIT.2022.3151415; arXiv:2103.11586 |
| 21 | Bronez 1992, IEEE TSP 40(12) | correct | doi:10.1109/78.175738; local copy of the paper |
| 22 | Walden 2000, Biometrika 87(4) | correct | doi:10.1093/biomet/87.4.767 |
| 23 | Abreu and Romero 2017, IEEE TIT 63(12) | correct | doi:10.1109/TIT.2017.2718963 |
| 24 | Harris 1978, Proc. IEEE 66(1) * | correct | doi:10.1109/PROC.1978.10837 |
| 25 | Mitra and Pesaran 1999, Biophys. J. 76(2) * | correct | doi:10.1016/S0006-3495(99)77236-X |
| 26 | Bokil et al. 2010, J. Neurosci. Methods 192(1) * | correct | doi:10.1016/j.jneumeth.2010.06.020 |
| 27 | Bruns 2004, J. Neurosci. Methods 137(2) * | correct; an erratum exists, J. Neurosci. Methods 143(2):237, 2005 | doi:10.1016/j.jneumeth.2004.03.002 |
| 28 | Thomson 1990, Philos. Trans. R. Soc. Lond. A 332(1627) | new entry, pp. 539-597 | doi:10.1098/rsta.1990.0130 |

No entry is retracted according to OpenAlex.

## Added 2026-09-27

The list now has 30 entries, renumbered in order of first citation, so the numbers in the table above are those of 2026-09-26.

| Entry | Result | Source of the check |
|---|---|---|
| Priestley 1962, Technometrics 4(4), pp. 551-564 | correct | doi:10.1080/00401706.1962.10490039 |
| Epanechnikov 1969, Theory Probab. Appl. 14(1), pp. 153-158 | correct | doi:10.1137/1114019 |

Both are cited for the parabolic kernel as the shape with the least mean square error. Priestley's abstract says the paper discusses
the construction of optimum estimates; the attribution of the quadratic window to it is the standard one (the Bartlett-Priestley
window) and was not checked against the full text. The efficiencies quoted in the manuscript (box 3%, Gaussian 2% worse in RMS
error) were recomputed numerically.

## Literature check for the split into a smoothed periodogram and a cross term, 2026-09-27

Question: has the exact split of the K-taper multitaper estimate into a smoothed periodogram plus a cross term, with the share of the
variance in the cross term, been published? Searched: the local reference PDFs, Crossref, OpenAlex and the web.

| Source | What it contains | Read in full? |
|---|---|---|
| McCloud, Scharf and Mullis 1999, IEEE Trans. Signal Process. 47(3), 839-843, doi:10.1109/78.747788 | lag-windowed estimators have multiple-window implementations, by an approximate low-rank factorization of the band-limiting Toeplitz matrix; "roughly equivalent" | abstract only (paywalled) |
| Stoica and Moses 2005, Spectral Analysis of Signals, Sec. 5.3.3 and Complement 5.6.2 | the Slepian estimator as the smoothed periodogram with the matrix replaced by its K leading eigenvectors (approximate); the Daniell and Blackman-Tukey estimators as exact multiwindow estimators | yes (local copy) |
| Riedel and Sidorenko 1995 | kernel smoothers have equivalent tapers close to sine tapers; sine-taper estimate as differences of FFT values; cancellation of side lobes between neighbouring Fourier coefficients; parabolic weighting; a split-cosine taper with a box is nearly a 4-taper estimate | yes (arXiv:1803.04078) |
| Walden, McCoy and Percival 1994, 1995 | exact variance of multitaper estimates for real processes; effective bandwidth | abstracts only |

Result: the approximate equivalence is established and is now credited. The exact split with a cross term, and the formula for its share
of the variance, were not found. This is not proof of novelty: the full text of McCloud et al. and the Percival and Walden
textbook were not available. An author with library access should read both before the manuscript claims the split as new.

## Sentences changed because the source did not support them as written

| Where | Before | After | Reason |
|---|---|---|---|
| Limitations | jackknife and quadratic-inverse refinements, both cited to Thomson and Chave 1991 | jackknife cited to Thomson and Chave 1991, quadratic-inverse to Thomson 1990 | the quadratic-inverse estimate is from Thomson 1990 |
| Relation to prior work | Bruns showed that Fourier, Hilbert and wavelet analyses "are one method" | "are formally equivalent" | the wording of the paper's abstract |
| Relation to prior work | the texts "present the three methods separately" | the tutorials, software and textbooks "do not relate the three methods to one another" | none of the five sources covers all three methods; Chronux is software |
| Recipes | "an orthogonal bank has advantages there that a single taper lacks", cited to Walden 2000 | "the statistical properties of multitaper cross-spectral estimates are well established" | Walden's framework also covers Welch and lag-window estimators, so it does not support an advantage of orthogonal tapers |
| Relation to prior work | Hansson-Sandsten: "using least squares" | "in the mean-square sense" | the paper's own term |

## Sentence added

Relation to prior work now states that Thomson and Chave remarked that section averaging and multiple-window estimates
are mathematically similar. The chapter says so at the start of its Sec. 2.3.

## Claims checked and left unchanged

- Riedel et al. 1994: a hybrid of a few tapers followed by a kernel smoother performed best; the advantage of multitaper over a
  smoothed tapered periodogram decreases as N increases, because the extra broadening scales with the Rayleigh resolution.
- Bronez 1992: multitaper always outperformed Welch's method when two of leakage, variance and resolution were held equal.
- Karnik et al. 2022: recommends K = 2NW - O(log NW) tapers; gives a fast approximation of the multitaper estimate.
- Abreu and Romero 2017: the average of the first K squared Slepian functions approaches the ideal band-pass kernel; rate in the L1 norm.
- Thomson and Chave 1991: the chapter gives the adaptive-weight equation, its eq. (2.20).
- Wahba 1980, Hurvich 1985, Haley and Anitescu 2017: data-driven choice of smoothing bandwidth, as cited.
- Nuttall and Carter 1982: Blackman-Tukey and Welch estimators are special cases of combined time and lag weighting.

## For the authors to note

Thomson and Chave (1991, Sec. 2.3) call periodograms and smoothed periodograms "hopelessly obsolete". The manuscript
recommends a tapered and smoothed periodogram as one of three adequate routes. Relation to Prior Work now answers this in
the paragraph "The view that smoothing is obsolete": the remark holds for the untapered periodogram and not for a tapered one.

Not checked against the source text, because no copy was available: Kay 1988, Blackman and Tukey 1958, Daniell 1946,
Bartlett 1948, Welch 1967, Satterthwaite 1946, Mitra and Pesaran 1999, Bokil et al. 2010, Harris 1978 (abstract only).
Their bibliographic details are confirmed; the statements cited to them are standard attributions.

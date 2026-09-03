# PREREG: does word-learning CAPACITY obey the anchor law?

Registered before running. The anchor law now holds in two places: the
assembly-formation ceiling (`CAP-ANCHOR-RATIO`: anchor 2x -> M* 2.86x; p and
beta excursions undone by a computed anchor) and alignment ACCURACY under load
(`PREREG_alignment_load.md` L2: accuracy tracks the referent's drive share).
The question that decides how to SIZE a lexical area is whether the number of
word types one area can hold follows the same law as the number of
assemblies: a ratio in n/k, moved by the anchor.

## The learner, unchanged

`unaligned_scenes.Aligner` exactly as it passed U1-U3: feedforward LEX and
FEAT, one LEX -> FEAT fiber, column scaling on FEAT, reconstruction readout.
Nothing in the model changes between cells; only n, k, the phon stimulus size
s, and the vocabulary size V.

## The corpus, synthetic on purpose

V referents, each a bundle (IDENT_i, CAT_{i mod 4}) -- the lesion corpus's
shape, [DOG, ANIMAL]. Word w_i names referent i. A scene is 3 referents drawn
uniformly; its sentence is their three words. Each referent appears in about
E = 6 scenes. This is not a claim about a lexicon; it is the substrate's
capacity question asked with the mechanism that passed on real groundings.

## The ceiling

Per-type alignment against the whole inventory (chance 1/V), the stringent
metric. V* is the vocabulary at which the per-type accuracy curve crosses
0.90, interpolated in log2(V) between bracketing grid points -- the same
reading `ceiling_from_curve` gives M*; a curve that never crosses inside the
grid is CENSORED and reported as a bound, never a value.

Grid: V in (16, 32, 64, 128, 256, 512). Seeds (42, 1, 2, 3, 4). Cells:

    A  n=1000  k=50   s=k     n/k = 20
    B  n=2000  k=50   s=k     n/k = 40
    C  n=4000  k=50   s=k     n/k = 80
    D  n=4000  k=100  s=k     n/k = 40   <- the ratio-law cell, pairs with B
    E  n=2000  k=50   s=2k               <- the anchor cell, pairs with B

## Bars

    W1  DIRECTION.  V* rises with n/k at fixed k: V*(C) > V*(B) > V*(A),
        each step larger than the pooled seed CI of the two cells.
        FAIL: any inversion beyond seed noise.

    W2  RATIO LAW.  V*(D) within +/-25% of V*(B) -- same n/k, n and k
        both doubled. This is CS1 asked of word learning.
        FAIL: outside +/-40%.  Between: inconclusive.

    W3  ANCHOR.  V*(E) / V*(B) >= 1.3 (F2 gave 2.86 for assemblies; a weaker
        but same-sign effect is the bar, since alignment is a conjunction
        and the anchor enters through the LEX side only).
        FAIL: <= 1.1.

    Every cell must be UNCENSORED for its bar to be judged; a censored cell
    voids the bars it takes part in and is reported as a bound.

## What is NOT claimed

* Nothing about word ORDER or roles; V2-style induction is not re-run.
* V* here is an alignment ceiling with FEAT fixed at n=1000, k=50, so a
  FEAT-side limit could bind first at large V. If it does, the curve's shape
  will say so (accuracy falling while LEX is far from full) and W1 is void.
* Synthetic bundles have exactly one distinctive feature; real referents
  share more. The number is an upper bound on a real lexicon's V*.

## Interpretation, stated now

* W1, W2, W3 pass -> one law sizes both assemblies and words: neurons per
  word is a function of n/k moved by the anchor, and "large" becomes a
  computed quantity. CHILDES-scale sizing follows from it.
* W2 fails -> word capacity is NOT a ratio law; the lexical fiber has its own
  scaling and the anchor law's scope stops at assembly formation. Report the
  measured exponent as a fit, adopt nothing.
* W3 fails -> the anchor moves accuracy (L2) but not capacity; the two are
  different limits and the registration's premise was wrong.

---

## Amendment 1 (2026-09-03, before any bar was read): FEAT was following n; exposures 6 -> 12

The first build let FEAT scale with n although the registration fixes it at
1000 x 50. The sweep was stopped after cells A and B had been run at V <= 128
and no bar had been judged. What those partial curves showed is worth keeping:
with FEAT following n, cell B (n=2000) read LOWER than cell A (n=1000) at
V=16 (0.56-0.81 against 0.80-1.00), and an exposure ladder at V=16 gave

    FEAT follows n:   E=6   A 1.000  B 0.812  C 0.875
                      E=48  A 1.000  B 0.875  C 0.625
    FEAT fixed 1000:  E=6   A 1.000  B 1.000  C 1.000
                      E=48  A 1.000  B 1.000  C 1.000

so the degradation with n was a READOUT FLOOR -- more FEAT columns competing
in the reconstruction's k-WTA -- and not capacity. That is itself a finding
about the readout (reconstruction accuracy at fixed evidence falls with the
size of the area reconstructed into), recorded here and not judged. FEAT is
now fixed as registered. Exposures are raised from 6 to 12 so the curves
start above threshold and the ceiling is approached from above; at E=6 even
the fixed-FEAT build sat within noise of 0.90 at V=16. Bars unchanged.

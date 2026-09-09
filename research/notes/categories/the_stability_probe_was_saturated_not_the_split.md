# #91 toy probe: fixed a real saturation bug, then found a genuine null at toy scale

**Script:** `research/experiments/task91_emergent_noun_verb.py`
**Artifact:** `task91_emergent_noun_verb.log` (superseded by this note's numbers;
the committed log predates the probe fix and is stale)

## What this is

#91 asks whether noun/verb category can emerge from an architectural
asymmetry alone (PHON->LEX1/LEX2 with LEX1 privileging VISUAL and LEX2
privileging MOTOR, per Mitropolsky & Papadimitriou 2025 sec 2.1-2.2),
replacing `grounding.py`'s hand-authored `dominant_modality` routing.
This session picked it up, found the standing toy harness's own
committed log stale and cut off mid-run, and re-ran it.

## The probe was saturated -- fixed before trusting any conclusion

First re-run (current source, before any edit): stability read ~0.97 in
BOTH the asymmetric arm AND the flat control -- the saturation
signature (a metric with no room to move can't discriminate anything,
per the #28 lesson explicitly cited in the script's own diagnostic
text). Root cause: `stability()` kept firing `{PHON: [lex], lex: [lex]}`
through every recurrent round, and PHON->lex is one of the four
deliberately STRENGTHENED fibers -- so PHON alone was pinning the
winners regardless of what the area itself held. Fixed: drop PHON
(inhibit + unfix, matching the pattern `recall()` already used for its
own return leg) after the first round, so the recurrent phase tests
genuine self-sustenance, not a re-selection PHON forces every step.

## Corrected result: the null is real, not an artifact

| arm            | noun gap | verb gap | accuracy | stability |
|----------------|----------|----------|----------|-----------|
| asymmetric     | +0.009   | −0.013   | 0.500    | 0.477     |
| CONTROL flat   | +0.010   | −0.020   | 0.528    | 0.278     |

- **Stability is no longer saturated** (0.477 vs 0.278) -- and the
  ~0.20 gap is in the sensible direction: the strengthened fibers DO
  raise aggregate assembly self-sustenance over the flat control. The
  mechanism is doing SOMETHING.
- **E1 fails** (verb gap wrong sign), **E2 fails** (accuracy at exact
  chance, 0.500), **E3 fails** (non-monotonic: 0.034 -> −0.039 -> 0.009
  across complement counts 1/2/4). None of the pre-registered
  per-word-category predictions hold.
- **A second, independently-built probe agrees.** `recall()` (the
  paper's own round-trip test, PHON->lex->PHON) was inspected directly
  (not just its `classify()` aggregate): raw values sit in a low, noisy
  band (0.02-0.20) with no noun/verb pattern (`fish` and `dog` both
  land LEX1; `cat` and `girl`, both nouns, land LEX2). Two probes built
  on different dynamics reaching the same "no category signal" verdict
  is the #28-shaped convergence that rules out "wrong instrument."

## Reading: exposure, not mechanism, is the leading suspect

The toy protocol trains on 120 sentences total (~10/word) at n=10^4 --
a large deviation from the paper's own setting and far short of this
project's OWN standing lesson, repeated across #52/#121/#151 this
session: toy-scale substrates routinely show nothing that the
production recipe (n=10^5, full exposure) shows cleanly. The paper's
own cited claim (Fig 3d) is about a GROWING gap under co-occurrence
diversity, which this toy's ~10-sentence/word budget may simply not
reach. `numpy_exact`'s O(M^2) cost ([[numpy-exact-95x-slower]]) makes a
naive scale-up expensive; the cheap next probe is exposure alone
(more sentences/word, same tiny vocabulary) before touching n or k.

## Sequential extension: does exposure rescue the null? NO -- on the
## two trustworthy levels, and the third is numerically corrupted

**Script:** `research/experiments/task91_exposure_sweep.py`
**Artifact:** `task91_exposure_sweep.log`

Registered before running: hold n=10^4, k=50, and the 12-word
vocabulary fixed; vary ONLY sentences/word (10x/40x/160x); run both
arms at every level so a rising control voids any reading; decision
rule pre-stated (production-scale testing warranted only if accuracy
clears baseline AND the control stays flat).

| sentences/word | arm        | noun gap | verb gap | accuracy | stability |
|-----------------|------------|----------|----------|----------|-----------|
| 10x (baseline)  | asymmetric | +0.009   | −0.013   | 0.500    | 0.477     |
| 10x             | control    | +0.010   | −0.020   | 0.528    | 0.278     |
| 40x             | asymmetric | −0.044   | −0.153   | 0.444    | 0.360     |
| 40x             | control    | −0.010   | +0.040   | 0.389    | 0.358     |
| 160x            | asymmetric | −0.063   | −0.085   | **0.389**| 0.324     |
| 160x            | control    | +0.034   | +0.015   | 0.556    | 0.216     |

**The 160x row is NUMERICALLY CORRUPTED, not evidence.** The run threw
`RuntimeWarning: overflow encountered in multiply` /
`invalid value encountered in multiply` at
`numpy_engine/_exact.py:339`. Root-caused precisely: `apply_to`'s
per-event potentiation factor `(1.0 + beta) ** mult` is computed
UNCLAMPED: with 1920 sentences on a 12-word vocabulary, some
self-recurrent fiber's co-firing count (`mult`) grew large enough for
`factor` to overflow to `inf` in float64 BEFORE the existing post-
multiply `w_max` clamp ever saw it, and `0 * inf = nan` poisoned cells
no downstream clamp can recover (NaN propagates through
`np.minimum`). This is a real engine gap distinct from the overflow
class already fixed at that same call site (documented in its own
docstring) -- flagged as its own out-of-scope unit
(`task_a0ad124c`), not fixed here.

**On the two CLEAN levels (10x, 40x), there is no rescue signal --
if anything the opposite.** Accuracy in the asymmetric arm FALLS with
exposure (0.500 -> 0.444), and the control falls too (0.528 -> 0.389).
Both arms decline together, which does not read as "the asymmetry's
effect strengthens with training" in either direction -- it reads as
a 12-word, k=50 area becoming progressively MORE crowded/confusable as
more sentences pile weight onto the same tiny set of assemblies,
unrelated to the architectural asymmetry. P-RESCUE fails on the
trustworthy data; the decision rule's bar (accuracy clears baseline
AND control stays flat) is not met even setting the corrupted level
aside.

## Status

#91 closes on this design. The stability-probe defect is fixed and
committed. The corrected measurement is honest on both readouts
(stability, recall) and both exposure levels that produced valid
numbers, and the verdict at every trustworthy point is a genuine null:
no per-word category signal from the architectural asymmetry alone, at
n=10^4/k=50/12-word vocabulary, at 10x-40x exposure. Production
integration (replacing `grounding.py`'s modality map) is NOT warranted
-- doing so on an unreplicated toy mechanism would be exactly the
"promote on the strength of a number that should be doubted" pattern
this repo's process discipline exists to prevent. Per the paper-
fidelity deviations already logged in the parent script (n=10^4 not
10^5, no C_i context areas, m=0), those -- not exposure -- are the
next candidates if this mechanism is revisited.

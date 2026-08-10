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

## Status

Diagnostic unit closed: the probe defect is fixed and committed, the
corrected measurement is honest (neither saturated nor accidentally
degenerate), and the verdict is a genuine null at this scale on TWO
independent readouts. Production integration (replacing
`grounding.py`'s modality map) is NOT warranted on this evidence --
doing so on an unreplicated toy mechanism would be exactly the
"promote on the strength of a number that should be doubted" pattern
this repo's process discipline exists to prevent. The exposure-scaling
follow-on (more sentences/word, same n/k) is the next registered step,
not yet run.

# What the P600 result can and cannot support

A standing limits document for the ERP line, written after three successive
confound removals and one retracted fourth. Its purpose is to make the claim
quotable without a footnote hunt, and to make the overclaims unavailable.

## The number

**p600 AUC = 0.7167 ± 0.0805**, cold, 10 seeds, `disk_hits=0 trained_fresh=10`,
above chance on every seed. Probe target area matched via `expected_slot`
(`f79c4f5`).

History, because the number has moved and will be quoted from old text:

| | AUC | what was matched |
|---|---|---|
| originally reported | 0.9056 | nothing |
| `f79c4f5` | **0.7167** | probe TARGET area |
| `0d70cc8`, **retracted** `a10a46a` | ~~0.6056~~ | also the source core — which removed the mechanism, not a confound |

## What it supports

**A rank claim, and only a rank claim.** Violation items rank above grammatical
items more often than chance, reproducibly across seeds and across three
metric changes.

## What it does not support

**1. Any magnitude.** The metric is saturated: every probe reads p600 ≈
0.989–0.994, and the total between-condition span is **0.0064**. The graded
detector's own threshold is 0.0760 — 11.9× more than anything observed.

- **Cohen's d is invalid here.** `p600_excess` is clipped against the
  grammatical median, so its standard deviation is not the standard deviation
  of anything. `CalibrationReport.summary()` already demotes it behind the AUC;
  `test_wobbly_stress` was asserting on it and inverted on negative values.
- **Excess margins are invalid**, for the same clipping reason.

The saturation has a mechanical cause, and it is not a tuning problem. The
metric is `1 − drive/w` where `w` is the **materialised** neuron count, so as
training grows `w`, the metric is pushed toward 1.0 regardless of what the
parser learned. Every scale-free alternative lands identically
(`erp_denominator_invariance.log`), because the problem is the **pool**:
`pool/k ≈ 1.4`, so the assembly *is* ~71% of the candidate population and has
nothing to stand out from. That is #104.

**2. That the items test what their labels say.** On the parser every ERP
number is measured on, **9 of the 11 default frame words have no core lexicon
entry**, and `chases` — the main verb of five of nine items — classifies NOUN.
Those five items contain no verb, so their "category violation" violates
nothing the parser parsed. See `the_calibration_frames_are_untrained.md`.

Something separates the two stimulus sets. "Grammatical sentence vs category
violation" is not a description of what was contrasted.

**3. Anything about the novel / N400 arm.** On the default frames, **12 of 18
novel probes have no defined stability reading at all**, so 0.55 of that arm's
p600 is a substituted constant — shipped 0.9928 against 0.6252 honest. See
`the_substituted_value_carries_the_novel_arm.md`. #28 is blocked on this, not
merely slowed by it.

**4. That this is a P600 in the neurolinguistic sense, or that it reflects
learning at all.** Mechanically the quantity is

```
1 − normalised pre-kWTA drive (core_area → role_area)
```

which reads like "has this (category, role) pathway been trained?" **It is not.**
Measured with the source area held fixed and only the pathway varied
(`the_p600_is_area_identity_not_pathway_learning.md`):

| contrast | isolates | AUC |
|---|---|---|
| agent-only vs patient-trained nouns | pathway learning alone | **0.5150 ±0.0725** |
| verb vs patient-trained noun | the shipped effect | 0.8870 ±0.0275 |
| verb vs agent-only noun | **source area alone** | **0.9800 ±0.0400** |

A noun never bound into `ROLE_PATIENT` reads the same as one that was. Swapping
the source area alone reproduces the whole effect. So the claim is not "the
parser detects a selectional violation", nor even "it learned which types fill
which slots", but **"nouns and verbs live in different areas, and those areas
differ in their drive into ROLE_PATIENT."**

## The granularity caveat, which is separate from the saturation

3 grammatical × 3 violation = 9 pairs, so the AUC moves in steps of 1/9 and
8 of 10 seeds read exactly 6/9. **Differences finer than 0.111 are not
resolvable by this item set**, and a ceiling is cheap to hit — which is how the
repaired frames' 1.0000 initially read as an improvement. Widen the item set
before reading any fine difference.

## Why confounds keep dominating, stated as a prediction

Each confound found so far was worth ≈ 0.005, against a total signal span of
0.0064. **That is not coincidence and it is not over.** At this dynamic range
any systematic asymmetry is competitive with the effect, so the expected number
of remaining confounds is not zero.

The structural reason is that the substrate is **totalizing**: k-WTA always
returns k winners, an untrained word still receives a category, a zero-synapse
fiber still yields an assembly. Nothing can refuse. **In a system with no ⊥,
"the number came from the apparatus" is the null hypothesis** and must be
actively excluded rather than assumed away. Pre-registered degenerate arms are
therefore not hygiene here; they are what separates a result from an artefact.

The one failure mode a harness cannot catch is a manipulation described
wrongly — `erp_source_core_identity_control.py` claimed to change "only the
category LABEL" while `forced_category` is a projection target. Reading what
the knob touches is the only defence.

## Two ways forward, and the cost of each

1. **Fix the pool (#104)** so the metric has headroom. This is the price of ever
   quoting an ERP *magnitude* again; without it the line can produce rank
   statistics indefinitely and nothing else.
2. **Downgrade and redirect.** State the P600 as a rank effect with these
   caveats attached and move effort to the composition line — VP composition
   graded by shared parents, never-seen combinations receiving the same
   gradient, that gradient reproducing on abstract symbols. Those bear on the
   actual research question; the ERP line is adjacent validation.

## Open, and named

- `erp_pathway_vs_area_control.py` decomposes the effect into pathway learning
  vs source-area identity, which no shipped contrast separates.
- `TRAINED_AREA_MATCHED_CALIBRATION_FRAMES` scores 1.0000 and still needs a
  degenerate control that changes no typing — the same frames on an untrained
  or shallow parser.
- The `defined_values` aggregation is measured but unflipped
  (`erp_stability_aggregation_study.py`), and is nearly free once the items are
  repaired.

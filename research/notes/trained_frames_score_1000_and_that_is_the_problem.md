# The repaired frames score a perfect 1.0000, and that is why they fail

Two pre-registered cold studies, 10 seeds, `disk_hits=0 trained_fresh=10`,
`research/experiments/erp_trained_frames_study.py`. Three frame sets, paired so
each study isolates one lever and `area_matched` is shared between them.

## Study A — item POSITIONS alone

| metric | default | area_matched | delta |
|---|---|---|---|
| `p600_auc_of_raw` | 0.7167 ±0.0805 | 0.7556 ±0.1046 | +0.0389 ±0.0992 |
| `p600_span_of_raw` | 0.0064 ±0.0008 | 0.0063 ±0.0008 | −0.0000 ±0.0003 |
| `untrained_items` | 9/9 | 9/9 | 0 |

**No detectable change**, and that is the correct answer rather than a null
result: the two sets differ ONLY in their novel-arm items, and `p600_auc_of_raw`
is a grammatical-vs-violation statistic. A delta of zero here is an internal
consistency check that passes.

So this settles the question #114 was opened for. **Promoting
`AREA_MATCHED_CALIBRATION_FRAMES` cannot be justified on the P600 AUC** — it
does not move it and was never going to. Its value is entirely in the novel arm,
where it removes the 12-of-18 probes whose stability term was a substituted
constant (`the_substituted_value_carries_the_novel_arm.md`, −0.3676). That is a
definedness argument, not an AUC argument, and it should be made as one.

## Study B — VOCABULARY alone

| metric | area_matched | trained_area_matched | delta |
|---|---|---|---|
| `p600_auc_of_raw` | 0.7556 ±0.1046 | **1.0000 ±0.0000** | +0.2444 ±0.1046 |
| `p600_span_of_raw` | 0.0063 ±0.0008 | 0.0077 ±0.0007 | +0.0013 ±0.0010 |
| `untrained_items` | 9/9 | **0/9** | −9 |

`VERDICT: FAIL`, on this criterion:

> CONSTANT across all seeds (1.0000) — zero variance is the signature of a
> structural artefact, not an effect.

## Why a perfect score is a failure

This is the `afferent_energy` signature inverted. That candidate was rejected on
**AUC 0.000 with zero variance across seeds**, and the diagnosis was that zero
variance means a CONSTANT — a probe reading something that does not depend on
the condition. 1.000 with zero variance is the same fact with the sign flipped.

The `must_vary` criterion was written into the harness precisely so this could
not be read as good news, and it did its job. Worth stating plainly because the
temptation is real: the items ARE better, the untrained count DID go 9 → 0, and
the number went UP. Every surface signal says adopt.

## The leading hypothesis: the confound one level over

`f79c4f5` area-matched the probe's TARGET area. But `measure_live_integration`
still derives the SOURCE core from the OBSERVED category:

```python
core = CATEGORY_TO_CORE.get(category)      # observed, not expected
...
anchored_p600_live(parser, core, role_area, ...)
```

So with `expected_slot` on, both arms probe ROLE_PATIENT — but the violation arm
reads **VERB_CORE → ROLE_PATIENT** and its grammatical control reads
**NOUN_CORE → ROLE_PATIENT**. Different source areas, by construction, in every
frame set, exactly as `structural_role_area` was for the target.

With the old items that difference was buried: five of nine had no main verb, so
the arms were noisy in other ways. Clean items may have simply exposed it. If so,
**the P600 here is measuring whether the critical word is a noun or a verb**,
not how hard it was to integrate.

## What would settle it

The same degenerate control that settled #108: hold the CONDITION constant and
vary only the critical word's category, and check whether the AUC survives. If a
condition-constant contrast also reads 1.000, the metric is reading core
identity. `p600_is_confounded_with_area_identity.md` is the template.

Until then the frames are **a better item design and an unusable measurement**,
and they stay out of the default. The 1.0000 is the reason to doubt them, not
the reason to ship them.

Related: [[fake-perfect-probe-signatures]], [[ensemble-not-realization]],
`the_calibration_frames_are_untrained.md`.

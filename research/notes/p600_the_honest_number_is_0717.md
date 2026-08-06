# The P600 is real and was inflated by ~0.19 AUC

**Resolution of #108.** The headline `p600_auc 0.9056` was measured with the two
arms probing DIFFERENT BRAIN AREAS. Area-matched, the effect survives at
**0.7167**, above chance on every seed.

## The confound

`structural_role_area` dispatches the probe on the word's **observed** category:

```python
if category == "VERB":
    return VP
if category in ("NOUN", "PRON"):
    return ROLE_PATIENT if verb_seen else ROLE_AGENT
```

A category violation *is* a word whose observed category differs from the
expected one. So in `the dog chases finds`, the critical word `finds` is
observed as a VERB and probes **VP**, while its grammatical control `cat` probes
**ROLE_PATIENT**. The arms differ by area in every frame set, at every seed,
under every definition of energy — and area identity alone reproduces the
headline AUC with the condition held constant.

**Area-matched FRAMES cannot fix this**, which is the part that took longest to
see. Both `DEFAULT_CALIBRATION_FRAMES` and `AREA_MATCHED_CALIBRATION_FRAMES`
already put the critical word in object position; the dispatch splits them
anyway, because it reads the word, not the position. Measured on both sets:
grammatical → ROLE_PATIENT ×3, category_violation → VP ×3. The frames fix the
*novel_noun* arm (8 of 12 of its items expected ROLE_AGENT); only the dispatch
fixes the *violation* arm.

## The measurement

`expected_role_area` claims the post-verb object slot and nothing else: after a
verb that licenses an object, the next content word is expected in ROLE_PATIENT
whether it turns out to be `cat` or `finds`. Both arms then read the same area
and the contrast becomes "did the word deliver drive into the slot the grammar
predicted?" — which is what a P600 is.

Cold, through `research/harness.py`, `ASSEMBLIES_BACKBONE_CACHE=0`:

```
study over 10 seeds [11, 12, 13, 42, 7, 19, 23, 31, 37, 101] (counterbalanced)
  substrate: disk_hits=0 trained_fresh=10 backbone_cache=OFF
  p600_auc_of_raw    0.9056+/-0.0268  ->  0.7167+/-0.0805   delta -0.1889  [PASS]
  p600_span_of_raw   0.0080+/-0.0006  ->  0.0064+/-0.0008   delta -0.0017  [PASS]
  VERDICT: PASS
```

Pre-registered bar, all three met: the arms area-match; every seed is above
chance (42 included — the only seed that has ever caught a structural change
here); the violation arm is not constant.

**The DROP is the result.** Removing a confound should shrink an inflated
effect, not reverse it. 0.9056 was the confound; 0.7167 is the effect.

## What I got wrong, and it blocked this for a day

The previous docstring said, under a heading reading **RESOLVED**:

> IT IS THE BACKBONE CACHE, AND THE DISPATCH REALLY DOES INVERT ON A
> FRESHLY-TRAINED PARSER.

with this table:

| | result | runtime |
|---|---|---|
| warm cache, default path | 62 passed | 75–128s |
| warm cache, `ERP_EXPECTED_SLOT=1` | 62 passed | |
| COLD cache, default path | 62 passed | 430s |
| COLD cache, `ERP_EXPECTED_SLOT=1` | **4 FAILED** | 339s |

The table is real. The inference was not. The cause was `test_acquisition.py`
setting `EMERGENT_DEV_CURRICULUM=1` **at module level**, which pytest executes
at collection and which therefore reconfigured training for every later test in
the process. Warm runs were immune because they deserialize a parser instead of
training one — which is exactly what made the cache look causal. Fixed in
`7e8c61b`; the same selection now reads 87 passed, and 86 passed with the
dispatch on.

Note what the wrong diagnosis had going for it: a clean 2×2, a plausible
mechanism (pre-grown pathways wiring bootstrap neurons while later training
recruits different ones), and a correct observation that an A/B built on cached
parsers is evidence about cached parsers only. All of that, and still wrong.

> **Reproducible under condition X is not caused by X.**

## Caveat on granularity

3 grammatical × 3 violation = 9 pairs, so AUC moves in steps of 1/9 and 8 of 10
seeds read exactly 6/9. The 0.7167 is coarse by construction. Widen the frame
set before reading any finer difference off it — including before comparing
0.7167 against any other published AUC.

## Still open

`afferent_energy` was rejected on measurement: AUC exactly 0.000 with zero seed
variance. That rejection was itself measured while the arms probed different
areas — zero variance is the signature of a constant, which is what a dead probe
on the wrong area returns. It is worth re-measuring now that the arms match.

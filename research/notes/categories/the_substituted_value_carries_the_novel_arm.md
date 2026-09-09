# The novel arm's P600 is 0.9928, and 0.55 of it is a substituted value

`e8455cc` found that `area_health`'s margin went undefined exactly when
separation was PERFECT, so the filter that looked like hygiene averaged the
worse half. Every `.or_else(fallback)` on the ERP path makes the same bet —
that undefinedness is uncorrelated with outcome — and none had been measured.

Two censuses, `research/experiments/erp_or_else_census.py` and
`erp_or_else_impact.py`, seeds 11/12/42.

## Which fallbacks fire at all

| site | undefined / calls | verdict |
|---|---|---|
| `anchored_p600_live` | 0 / 90 | never fires — free |
| `measure_lexical_surprise` | 0 / 81 | never fires — free |
| `phrase_stability` | 84 / 135 | fires, one reason only |
| `_role_binding_margin` | 117 / 147 | fires almost always |

`phrase_stability`'s single reason is the #108 defect: **`VP` has no self-fiber**,
and `_phrase_areas_for_category` puts `VP` in the list for every category. So at
least one reading per probe is undefined, always, on every arm.

## The rate is not the impact

The aggregation at `erp/adapters.py:637` is

```python
stabilities = [r.or_else(0.0) for r in readings]
mean        = sum(stabilities)/len(stabilities) if stabilities else 1.0
instability = 1.0 - mean
```

The two aggregations differ in a way that is **not a level shift**:

* one undefined among several → substituting 0.0 drags the mean down, raising
  instability, raising p600 — a small bias;
* **all** readings undefined → substituting gives mean 0.0, instability 1.0, the
  MAXIMUM. Dropping them gives an empty list, which takes the `else 1.0` branch:
  instability 0.0, the MINIMUM. Same probe, opposite ends of the range.

## Measured

| arm | probes | areas/probe | undefined | ALL undefined | p600 shipped | p600 honest | delta |
|---|---|---|---|---|---|---|---|
| category_violation | 18 | 2.00 | 18/36 | 0/18 | 0.9935 | 0.9906 | −0.0029 |
| grammatical | 18 | 2.00 | 18/36 | 0/18 | 0.9892 | 0.9863 | −0.0029 |
| **novel_noun** | 18 | 1.33 | 18/24 | **12/18** | 0.9928 | **0.6252** | **−0.3676** |

**On 12 of 18 novel probes there is no defined reading at all**, so the entire
stability term — 0.55 of the p600 — is the substituted constant. The two probes
responsible:

```
novel_noun  word=bird   role=ROLE_AGENT  areas=[VP]   0.9922 -> 0.4422
novel_noun  word=small  role=VP          areas=[VP]   1.0000 -> 0.4500
```

Those are exactly `holdout noun subject` ("the bird sees the cat", novel word in
SUBJECT position) and `holdout adj attributive` ("the small dog runs"). The role
area they expect has no active assembly, so `areas_with_active_assembly` returns
`[VP]` alone, and `VP` is the one area that is never defined.

## What it means

The substitution is not a bias to correct — on the novel arm it **is** the
answer. Any comparison involving `novel_noun` p600 is reading a constant.

It also identifies the fix. These are the two frames
`AREA_MATCHED_CALIBRATION_FRAMES` replaces with novel-word-in-object items. Put
the novel word in object position and the expected role area has an active
assembly, so the readings are defined and there is nothing left to substitute.

**Fix the item design first and the aggregation change becomes nearly free** —
−0.0029 on the two arms that already have a defined reading. Doing it in the
other order changes every published magnitude to move a term that should not
have existed.

Related: [[undefinedness-correlates-with-outcome]], and
`the_calibration_frames_are_untrained.md`, which finds a second and independent
defect in the same items.

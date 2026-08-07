# The P600 is area identity. Pathway learning contributes nothing.

`anchored_p600_live` documents its mechanism in one sentence that contains two
separable claims:

> a category violation routes a **wrongly-typed core** through an **untrained
> pathway** and delivers LESS

- **(a) untrained pathway** — this word's assembly never strengthened synapses
  into the expected role area.
- **(b) wrongly-typed core** — the drive is read out of a *different area*
  (`VERB_CORE` not `NOUN_CORE`), which differs in size, training history and
  degree distribution.

**Every ERP contrast ever shipped varies (a) and (b) together**, because the
only way it builds a violation is to put a verb in a noun slot. So the two have
never been separated, and (b) alone would produce a separation containing no
learning at all.

## The design

The parser supplies the split for free. Among its 71 trained nouns:

| | n | patient pathway |
|---|---|---|
| in `role_lexicons[ROLE_PATIENT]` | 36 | **trained** |
| in `[ROLE_AGENT]` but not patient | 29 | **not trained** |

Both are nouns, so both project into `NOUN_CORE` and both are read as
`NOUN_CORE → ROLE_PATIENT`. Both have been bound into *some* role area, so "has
any role training" is held constant too. Only the specific pathway differs.

Three arms, one skeleton (`the dog want ___`), one probe position:

```
patient_trained   noun bound into ROLE_PATIENT
agent_only        noun bound into ROLE_AGENT only
verb_object       trained VERB              <- the standard violation
```

Every item has **zero occurrences** in the corpus that trained this depth, so
frequency is matched by construction, and every item is required to classify as
its intended category on the parser being measured. 10 items per arm gives 100
pairs — AUC granularity 0.01, against the shipped frame sets' 1/9.

## The result

5 seeds, `research/experiments/pathway_vs_area.log`:

| contrast | isolates | AUC |
|---|---|---|
| `agent_only` vs `patient_trained` | **(a) pathway only** | **0.5150 ±0.0725** |
| `verb_object` vs `patient_trained` | (a+b) the shipped effect | 0.8870 ±0.0275 |
| `verb_object` vs `agent_only` | **(b) area only** | **0.9800 ±0.0400** |

Per-seed for (a): 0.590, 0.585, 0.445, 0.465, 0.490 — straddling chance with no
consistent direction. Per-seed for (b): 0.990, 0.995, 0.995, 0.920, 1.000.

**A noun that was never bound into `ROLE_PATIENT` reads the same as one that
was** — same position, same sentence, same source area. The 36-vs-29 split in
the role lexicons is invisible to the metric. Meanwhile swapping the source area
alone reproduces the entire effect, and slightly exceeds it.

## What this means

**"Untrained pathway" is the wrong description of the P600.** The quantity does
not respond to word-level pathway learning at all. It responds to *which area
the drive is read out of*.

That is a substantially weaker claim than the one the ERP line has been making.
Not "the parser detects a selectional violation", not even "the parser learned
which types occupy which slots" — but **"nouns and verbs live in different
areas, and those areas differ in their drive into ROLE_PATIENT."** The
categoriser routes the word; the areas do the rest.

It also predicts the control still outstanding: **the effect should survive on a
parser with far less role training**, since role training is not what it reads.

## A mechanistic candidate, and it is testable

Why would word-level pathway learning be invisible? Because the core assemblies
may not be distinct enough to carry it. Role binding is already known to sit in
the **crowding regime** (#52, pairwise overlap 0.15–0.22 rather than 1.000). If
individual noun assemblies in `NOUN_CORE` heavily overlap, their drive into
`ROLE_PATIENT` is nearly identical regardless of which ones were bound there —
exactly the null this measured. The area-level difference survives because
`NOUN_CORE` and `VERB_CORE` do not overlap.

That is a substrate-capacity explanation, not a metric bug, and it connects this
result to the composition line rather than to the ERP plumbing.

## Found on the way: TWO_WORD has an empty ROLE_PATIENT

The depth control could not run. At `TWO_WORD`, `role_lexicons[ROLE_PATIENT]`
is **empty** and only 2–3 verbs are trained, so no matched pool exists and the
script refused rather than compare unequal arms.

Which means `assess_erp_readiness` opens the p600 gate on `ROLE_AGENT` bindings
alone, and `test_wobbly_stress::test_readiness_gates_track_curriculum_depth` is
comparing a depth with **no patient bindings at all** against one with 36. That
test's "depth buys separation" framing is measuring the presence of the arm, not
its quality.

`VOCABULARY_SPURT` would be the ideal zero-role control — its phases are
`["lexicon", "distributional"]`, no roles — but the readiness gate shuts with no
role binding stored, so every probe returns the early `0.0` and the contrast is
void. Getting a true zero-role reading requires reading `input_drive` directly
rather than going through the gated ERP path.

## Status of the P600 claim after this

The rank effect is still real and still reproducible. What has changed is what
it is evidence *of*. See `what_the_p600_can_and_cannot_support.md`; this note
supersedes the "type learning" reading offered there, downward.

Related: [[silent-no-op-dead-fibers]] (a mechanism that is wired and does no
work), `p600_is_confounded_with_area_identity.md` (the same fact for the probe's
target area), `the_source_core_is_still_the_observed_category.md` (the retracted
attempt to fix this by matching the source, which deletes the only thing the
metric responds to).

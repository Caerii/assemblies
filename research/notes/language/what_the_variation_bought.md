# What the corpus variation bought — and what was never wired to read it

**Task #129. Experiment: `research/experiments/what_variation_buys.py` (pre-registered; predictions P1–P4 in its docstring). Follows e909c61 ("the corpus now varies on every censused axis"), which argued a cue that never varies carries no information — but never showed the varied cues are USED.**

## The census half was decided by code reading, before any training

The repo already contained a full morphosyntax subsystem (`parser_mixins/morphosyntax.py`): `detect_*` Python teachers, `train_*` neural phases projecting `feature_stim + word → {TENSE, NUMBER, MOOD, POLARITY}`. Its state at the start of this unit:

1. **"tense" was live but one-class until e909c61.** The phase runs in SENTENCES (`schedule.py`), but the old corpus was all-present, so `train_tense` projected `tense_PRESENT` for every sentence — nothing discriminative to learn. The variation was the missing precondition, not the mechanism.
2. **"number" was doubly dead.** The schedule supports the phase (`schedule.py:152`) but **no stage listed it** — the same dormant-selector shape as `phrases` (a selector over an unfilled field is a silent no-op). And its teacher `detect_number` read grounding features (`"SG"`/`"PL"`) that **nothing in the pipeline ever sets**: plural surfaces inherit the lemma's `GroundingContext` verbatim (`_register_surface_forms`), so the teacher said SG for every token in every corpus. One-class twice over.
3. **Nothing read either area back.** `detect/train` pairs with no `recall`: the learned content was unmeasurable in principle. (`incremental.py` binds VERB_CORE→TENSE during a parse, but no consumer ever asked TENSE what it held.)

Fixes landed with this unit: `PAST_RATE`/`PLURAL_RATE` module knobs (the PASSIVE_EVERY off-switch lesson applied), "number" scheduled into SENTENCES/COMPLEX_GRAMMAR/CONVERSATION, `detect_number` resolves plurality from lexicon forms (mirroring `lookup_verb_form`; noun forms only — verb agreement is not decidable at form level), and `recall_tense`/`recall_number` — a reconstruction-style readout (argmax over stimulus images, ties refuse to answer) matching `parse_roles_by_reconstruction`'s RECALL step.

## What recall_* measures, stated honestly

`train_tense`/`train_number` co-fire the surface FORM's assembly with a feature stimulus. What is recallable is therefore a **lexical association** ("chased"→PAST, "dogs"→PL), not sentence tense: the aux+participle logic lives in the Python detector the substrate never sees. The substrate claim under test is capacity and separation: can one feature area hold a two-class contrast over ~100 forms and return it, form by form?

## The instrumented dry run caught the test set lying (three ways)

The first scored launch was killed after instrumentation showed the "plural" test items in the *no-variation control* were: `fish, answers, hopes, loves, fears, surprises`.

- **Noun–verb homographs.** `_build_lexicon_index` keeps the FIRST entry per surface (nouns before verbs), so "loves" — a 3sg verb in the corpus — resolves to the noun entry and gets classed as a noun plural. Same-name-two-meanings, in the lexicon this time. The same collision biases `detect_number`'s teacher and hides these forms' verbhood from `lookup_verb_form` (so `detect_tense` cannot see them as verbs either). Spun off as its own task (index should hold all entries per surface).
- **Zero-derivation pasts.** "put/let/cut/read" have past == lemma; lemmas register unconditionally and train as present. The PAST class rule needed the same `w != lemma` exclusion the PRESENT rule already had.
- **"fish"** is its own plural.

The fixed `test_sets` scans raw `NOUNS`/`VERBS` data (independent of the index-order bug) and excludes all form-level-ambiguous surfaces from every class.

## P4 was refuted before the scored run — passives leak past forms

Registered P4 said the no-variation corpus attests zero past/plural forms. Wrong for a mechanical reason found in the smoke run: **passives inject regular participles** ("is chased" — ppart == past form for regular verbs) at `PASSIVE_EVERY` regardless of `PAST_RATE`, and hand-authored curriculum sentences carry "was". The plural half of P4 held exactly (PL test set empty at `PLURAL_RATE=0`). Recorded as an addendum in the experiment before the scored run. Related detector artifact, noted not fixed: `detect_tense` labels a PRESENT passive ("the cat **is** chased") PAST, because the participle is the past form — at form level this is consistent ("chased" IS a past form), but it mislabels copula forms, so "is"/"was" carry mixed teacher signal.

## Scored results (5 seeds × 3 arms, n=3000, k=30, curriculum → SENTENCES)

| arm | tense balanced (5 seeds) | PAST acc | PRES acc | img sep | number balanced | PL acc | SG acc |
|---|---|---|---|---|---|---|---|
| DEFAULT | **0.652 ± 0.121** (95% CI) | 0.57–0.83 | 0.52–0.76 | 0.00–0.10 | **0.520 ± 0.056** | 0.0–0.2 | 0.9–1.0 |
| ABLATE | 0.0 (all ties) | 0.0 | 0.0 | **1.0** | 0.0 (all ties) | 0.0 | 0.0 | 
| NO_VARIATION | 0.53–0.66 | 0.2–0.4 (n=10) | 0.84–0.95 (n=38) | 0.07–0.20 | — (PL n=0) | — | — |

**Verdicts on the registered predictions:**

- **P1 REFUTED at its bar.** DEFAULT tense balanced is 0.652, not ≥0.75. The signal is REAL — every seed beats the ablated null, which reads literally nothing (identical images, sep=1.0, the readout refuses to answer), and image separation is healthy — but it is weak and seed-variable (0.54 on the worst seed is barely above a coin).
- **P2 CONFIRMED, in the strongest form.** The null is all-ties, not merely at-chance: DEFAULT's accuracy is attributable entirely to the learned association, not to registration or k-WTA artifacts.
- **P3 REFUTED decisively — and this is the finding.** Number balanced ≈ 0.52 = chance. Not because the area can't answer (image separation 0.03–0.17, and SG reads 0.9–1.0) but because the PL associations never formed: every sentence trains several singular noun tokens into NUMBER, while a given plural form rides ~30% of subjects only. Hebbian mass follows token frequency, and the frequent class swamps the rare one. This is the same shape as the Zipf result (frequency closes composition) and the starvation results (PASSIVE_EVERY, DITRANSITIVE_EVERY existing at all): **the substrate learns what the corpus hammers, and realistic corpora hammer unevenly.**
- **P4 REFUTED as written** (10 past forms attested per seed at PAST_RATE=0 — passive participles + "was", exactly the addendum's mechanism). The plural half held exactly (PL n=0).

Also visible: NO_VARIATION's PAST accuracy is 0.2–0.4 on the 10 participle-taught forms — passives alone teach a little past tense, which is why its balanced number floats above 0.5 despite the axis being "off".

## What this means

The corpus variation bought exactly one thing so far: it made the question POSABLE (the ablate/no-variation arms cannot even be scored, or score on majority only). The learned yield is a weak tense association and no number contrast, and the failure mode is not mechanism absence — every wire now exists and carries signal — but **frequency imbalance**, our best-documented systemic enemy. The forcing-rate pattern (PASSIVE_EVERY, DITRANSITIVE_EVERY) is the crutch that has papered over this three times; a per-phenomenon rate knob is exactly what does NOT scale to a real corpus. The principled question this opens: a frequency-normalizing mechanism at the learning rule (the norm_init lesson, applied to association mass rather than degree bias), measured against the same readout — NOT a PLURAL_EVERY knob.

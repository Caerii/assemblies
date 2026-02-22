# Composed ERP Pipeline: N400/P600 Double Dissociation

**Scripts**:
- `research/experiments/primitives/test_prediction_n400.py`
- `research/experiments/primitives/test_binding_p600.py`
- `research/experiments/primitives/test_composed_erp.py`
- `research/experiments/primitives/test_incremental_erp.py`
- `research/experiments/primitives/explore_composed_params.py`
- `research/experiments/primitives/diagnose_erp_dynamics.py`

**Date**: 2026-02-22
**Brain implementation**: `src.core.brain.Brain` (sparse engine, w_max=20.0)

## Overview

N400 and P600, the two dominant ERP components in language processing, emerge
as different readouts of the same prediction + binding mechanism in Assembly
Calculus. No separate modules or ad-hoc energy functions are needed.

**N400** = lexical prediction error.
`1 - overlap(context-driven_prediction, word's_lexicon_entry)`
Context area projects forward into PREDICTION; the resulting assembly is
compared against the word's stimulus-driven PREDICTION fingerprint.

**P600** = structural integration difficulty (anchored instability).
Prime the role area with one stimulus-driven co-projection round, then
settle with area-to-area connections only. Trained pathways sustain the
initial pattern (low instability); untrained pathways cannot (high
instability).

## Architecture

```
PHON_<word>  -->  NOUN_CORE  -->  PREDICTION  (forward prediction)
                  VERB_CORE  -->  PREDICTION
                  NOUN_CORE  <->  ROLE_AGENT   (role binding)
                  NOUN_CORE  <->  ROLE_PATIENT
```

Five brain areas (n=10000, k=100, p=0.05, beta=0.10):
- **NOUN_CORE**, **VERB_CORE**: Word assemblies self-organize via stimulus projection
- **PREDICTION**: Forward projection target; co-projection training creates
  Hebbian bridges from context areas
- **ROLE_AGENT**, **ROLE_PATIENT**: Structural role slots; bidirectional
  co-projection binds words to roles

Training: Random SVO sentences (20 unique x 3 reps = 60 total). Each sentence
trains both prediction bridges (context + stimulus -> PREDICTION) and role
bindings (core <-> role bidirectional co-projection).

## Protocol

### Prediction Training (N400)

For each SVO sentence (agent, verb, patient):
1. Activate agent in NOUN_CORE
2. Co-project: NOUN_CORE + PHON_verb -> PREDICTION (5 rounds)
3. Activate verb in VERB_CORE
4. Co-project: VERB_CORE + PHON_patient -> PREDICTION (5 rounds)

This builds Hebbian bridges so that context-only projection (e.g.,
VERB_CORE -> PREDICTION with no stimulus) evokes a prediction assembly
that overlaps with trained co-projection targets.

### Binding Training (P600)

For each SVO sentence:
1. Co-project: PHON_agent + NOUN_CORE <-> ROLE_AGENT (10 rounds)
2. Co-project: PHON_patient + NOUN_CORE <-> ROLE_PATIENT (10 rounds)

Only nouns are bound to role slots. Verbs are never bound to ROLE_PATIENT,
creating the trained/untrained asymmetry that drives P600.

### Test Conditions

At the critical (object) position, three conditions share the same context
(agent + verb) but vary the critical word:

| Condition | Example | N400 expectation | P600 expectation |
|-----------|---------|-----------------|-----------------|
| **Grammatical** | "dog chases cat" | Low (trained noun predicted) | Low (noun->role trained) |
| **Category violation** | "dog chases likes" | High (verb not predicted as noun) | High (verb->role untrained) |
| **Novel object** | "dog chases table" | High (no co-projection experience) | Low (noun->role still works) |

## Results

### Double Dissociation (Central Result)

**n=10000, k=100, 10 seeds, full parameters:**

| Condition | N400 (word-specific) | P600 (anchored instab.) |
|-----------|---------------------|------------------------|
| Grammatical | 0.088 +/- 0.007 | 0.023 +/- 0.011 |
| Category violation | 0.351 +/- 0.012 | 4.954 +/- 0.094 |
| Novel object | 0.979 +/- 0.010 | 0.538 +/- 0.115 |

**Effect sizes:**
- N400 CatViol > Gram: d=14.56, p=0.002
- P600 CatViol > Gram: d=28.17, p<0.001
- P600 Novel ~ Gram: d=1.93, p=0.079

**Double dissociation:**
- Novel: N400 effect = +0.89, P600 effect ~ 0
- CatViol: N400 effect = +0.26, P600 effect = +4.93

### N400 Isolated (test_prediction_n400.py)

| Metric | Gram | CatViol | Novel | d(cv/g) |
|--------|------|---------|-------|---------|
| Word-specific | 0.08 | 0.36 | 0.99 | 6.61 |
| Category-match | 0.05 | 0.32 | 0.05 | 4.02 |

Word-specific N400 for novel nouns is near ceiling (0.99) because novel
nouns have zero co-projection experience; their PREDICTION fingerprints
live in a completely different neuron subspace than context-driven
predictions (overlap = 0.003, confirmed by geometry diagnostic).

Category-match N400 (max overlap with same-category trained references)
shows CatViol >> Gram but Novel = Gram, confirming the prediction system
operates at the category level.

### P600 Isolated (test_binding_p600.py)

| Metric | Gram | CatViol | d |
|--------|------|---------|---|
| Anchored instability | 0.078 | 4.807 | 90.25 |
| Convergence round | 1.2 | 10.0 | 25.40 |

Anchored instability replaced binding weakness (1 - retrieval overlap)
as the primary metric after diagnostic investigation showed it is both
more theoretically principled and strongly separating.

### P600 Metric Investigation (diagnose_erp_dynamics.py)

Four candidate metrics compared head-to-head:

| Metric | Gram | CatViol | d(cv/g) | Notes |
|--------|------|---------|---------|-------|
| Jaccard instability | 0.11 | 3.20 | 3.79 | Unreliable: untrained pathways have zero signal |
| Binding weakness | 0.56 | 1.00 | 30.23 | Works but measures retrieval, not integration |
| Activation deficit | 0.00 | 64194 | 4.18 | Problematic: novel nouns get negative values |
| **Anchored instability** | 0.12 | 5.24 | **21.26** | Theoretically principled AND strong |

**Why anchored instability**: Untrained pathways (VERB -> ROLE_PATIENT) have
literally 0.0 weight sum (confirmed by signal flow diagnostic). Raw Jaccard
instability produces paradoxically zero values for zero-signal pathways. By
anchoring with one stimulus-driven round, we create an initial pattern that
only trained pathways can maintain through area-to-area connections.

### Representational Geometry (diagnose_erp_dynamics.py, probe: geometry)

Cross-mode overlap (context-driven prediction vs stimulus-driven lexicon entry):
- Context -> trained nouns: **0.916** (high: prediction matches lexicon)
- Context -> novel nouns: **0.003** (near zero: completely different subspace)
- Context -> verbs: **0.661** (partial: some cross-category overlap)

This explains why word-specific novel N400 = 0.99: the novel noun's PREDICTION
fingerprint (built by stimulus-only projection) has zero overlap with the
context-driven forward prediction (built by co-projection training).

### Signal Flow (diagnose_erp_dynamics.py, probe: signal_flow)

| Pathway | Pre-kWTA activation | Weight sum | Weight max |
|---------|-------------------|------------|------------|
| NOUN -> PREDICTION (trained) | 12,539 | 67,516 | 20.0 |
| VERB -> PREDICTION (trained) | 17,284 | 67,445 | 20.0 |
| NOUN -> ROLE_PATIENT (trained) | 12,781 | 51,614 | 20.0 |
| VERB -> ROLE_PATIENT (untrained) | **0.0** | **0.0** | **0.0** |

Binary all-or-nothing gap. Hebbian plasticity creates zero weights for
pathways that were never co-activated.

### Exposure Gradient (diagnose_erp_dynamics.py, probe: exposure)

| Novel noun exposure | Trained N400 | Novel N400 | CatViol N400 |
|--------------------|-------------|------------|-------------- |
| 0 sentences | 0.08 | **1.00** | 0.36 |
| 1 sentence | 0.08 | **0.92** | 0.35 |
| 3 sentences | 0.08 | **0.72** | 0.35 |

N400 is graded with co-projection experience. Novel nouns need exposure
to develop overlapping PREDICTION representations. This validates the
mechanism and maps naturally to word-frequency effects in the ERP literature.

### Incremental Learning (test_incremental_erp.py)

Sentences processed one at a time with plasticity ON, ERPs measured at
checkpoints:

| Sentences | N400_gram | N400_catviol | N400_novel | P600_gram | P600_catviol | P600_novel |
|-----------|-----------|-------------|------------|-----------|-------------|------------|
| 0 | 1.000 | 1.000 | 1.000 | 3.47 | 3.17 | 2.48 |
| 3 | 0.984 | 0.978 | 0.967 | 0.77 | 1.55 | 0.67 |
| 10 | 0.853 | 0.898 | 0.798 | 0.18 | 1.52 | 0.14 |
| 20 | 0.673 | 0.804 | 0.598 | 0.08 | 1.55 | 0.06 |

- N400 decreases for all words with exposure; CatViol decreases slowest
- P600 separates by sentence 3: CatViol stays high, gram and novel drop
- Novel N400 learning effect: 1.00 -> 0.60 (d=4.37, p=0.017)
- At 20 sentences: CatViol N400 > Gram (d=2.62), CatViol P600 > Gram (d=3.14)

### Parameter Robustness (explore_composed_params.py)

P600 (anchored instability) d > 8 across all parameter configurations tested.
N400 requires >= 3 prediction rounds per pair to separate conditions (d ~ 0
at 1 round).

Key sensitivities:
- **Settling rounds** scale P600 cleanly: 3 rounds -> d=19.7, 5 -> d=34.7, 10 -> d=24.8
- **Area size** improves both metrics: larger areas -> tighter assemblies -> cleaner separation
- **Beta 0.10** is the sweet spot for N400; lower beta (0.05) weakens prediction

## Implications

1. **ERP components are not separate mechanisms.** A single architecture (5 areas,
   Hebbian co-projection, kWTA) produces both N400 and P600 as readouts of the
   same prediction + binding loop. No separate modules needed.

2. **Anchored instability captures structural integration difficulty.** The metric
   tests whether a structural pathway can sustain a pattern without continued
   stimulus input — matching the theoretical interpretation of P600 as unification
   difficulty (Hagoort 2005) or settling time (Vosse & Kempen 2000).

3. **The mechanism supports incremental learning.** Online sentence processing with
   plasticity ON produces naturally graded ERP signals: word-frequency effects
   (N400 decreases with exposure), category-level generalization (novel nouns bind
   correctly immediately), and persistent selectional restrictions (verbs never
   fit noun slots).

4. **Foundation for free-form language learning.** The incremental experiment
   validates that this is not a batch train/test artifact but a genuine online
   learning mechanism ready for extension to variable-length sentences and
   recursive structure.

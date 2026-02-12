# N400 Next Steps: From Discovery to Defensible Claim

## What We Found

After 13 failed post-k-WTA conditions across 3 experiment files, the
pre-k-WTA experiment (`test_n400_pre_kwta.py`) discovered:

**The N400 in Assembly Calculus = global pre-k-WTA energy** (`sum(all_inputs)`).

| Metric | Direction | Cohen's d | p |
|--------|-----------|-----------|-------|
| 1c:global_energy | CORRECT | -25 to -31 | <0.001 |
| 3:settling | CORRECT | -16 to -18 | 0.001 |
| 1a:mean_input | REVERSED | -4 to -6 | 0.01 |
| 1b:max_input | REVERSED | -6 to -7 | 0.008 |
| 2a:pred_error | ~null | -0.1 | 0.86 |

Neuron-specific metrics are reversed (competition). Global aggregate metrics
show correct direction with massive effect sizes. This maps to the neuroscience:
the N400 ERP is a scalp-recorded aggregate signal, not single-neuron activity.

---

## Phase 1: Strengthen the Core Claim — COMPLETED

### 1A. Parameter Sensitivity Sweep ✅

**File:** `research/experiments/applications/test_n400_parameter_sweep.py`

**Result:** The effect is NOT universal — it requires **k>=100 AND p>=0.05**.

| k | p | Energy direction | Settling direction |
|---|---|-----------------|-------------------|
| 100 | 0.05 | ALL CORRECT (d=-11 to -111) | ALL CORRECT (d=-4 to -36) |
| 100 | 0.01 | null/reversed | mostly null |
| 50 | any | null/reversed | null/reversed |

**Interpretation:** The mechanism requires:
- Sufficient connectivity (p>=0.05) for Hebbian learning to differentiate
- Sufficient assembly size (k>=100) for meaningful inter-assembly overlap
- With k=50 in n=50000, expected random overlap is k^2/n = 0.05 neurons

### 1B. Vocabulary Scaling

**Question:** Does the effect survive with larger vocabularies where
semantic categories are less clearly separated?

**Design:**
- Small: 8 words (current)
- Medium: 30 words (5 animals, 5 tools, 5 foods, 5 vehicles, 5 clothing, 5 filler)
- Large: 100 words (10 categories x 10 words)
- Track effect size as vocabulary grows

**Prediction:** Effect weakens but remains significant as vocabulary scales.
The N400 literature shows smaller effects with weaker associations.

**Status:** Not yet implemented.

**File:** `research/experiments/applications/test_n400_vocab_scaling.py`

### 1C. Control Conditions ✅

**File:** `research/experiments/applications/test_n400_controls.py`

**Results:**

| Condition | d | p | Outcome |
|-----------|---|---|---------|
| A: Semantic (related < unrel) | -25.2 | 0.0005 | CORRECT — replicates |
| B: Rep vs Sem | -4.7 | 0.015 | CORRECT ordering (rep < sem < unrel) |
| B: Rep vs Unrel | -2.3 | 0.057 | Marginal |
| C: Shuffled null | 2.4 | 0.054 | NULL as expected |
| D: Cross-category | 1.2 | 0.167 | No facilitation (correct) |

**Interpretation:**
- Semantic priming replicates with d=-25.2 (massive effect)
- Repetition priming shows correct ordering: rep < sem < unrel
- Shuffled control (two unrelated primes) shows no difference (null, as expected)
- Cross-category (co-occurrence without shared features) shows no facilitation,
  confirming the effect requires shared semantic features, not just co-training

### 1D. Engine Parity ✅

**File:** `test_n400_pre_kwta.py --engine torch_sparse`

**Result:** torch_sparse replicates numpy_sparse:
- global_energy: d=-24.3, p=0.0006 (CORRECT)
- settling: d=-9.9, p=0.003 (CORRECT)

---

## Phase 2: Extend to Richer Paradigms — MOSTLY COMPLETE

### 2A. Cloze Probability ✅ (Kutas & Hillyard 1984)

**Question:** Does global pre-k-WTA energy correlate with cloze
probability (the probability a word completes a sentence)?

**Design:**
- Train parser on sentences with varying predictability
- High cloze: "The dog chases the cat" (cat is predictable)
- Low cloze: "The dog chases the table" (table is unexpected)
- Medium cloze: "The dog chases the bird" (possible but less common)
- Measure global energy at the final word position

**Prediction:** Global energy increases monotonically as cloze
probability decreases. This is the canonical N400 finding.

**File:** `research/experiments/applications/test_n400_cloze.py`

**Result:** Monotonic ordering CONFIRMED: E(high) < E(medium) < E(low).
high_vs_low: d=-18.2, p=0.001 (CORRECT). high_vs_medium: d=-33.4, p=0.0003.
"The dog chases the ___": cat gets ~13600 energy vs table ~24700 (nearly half).

### 2B. Sentence Context (Incremental Processing)

**Question:** Does the N400 accumulate context across a sentence, not
just from a single prime word?

**Design:**
- Full context: "The dog chases the [cat vs table]"
- Partial context: "The [cat vs table]" (no verb context)
- No context: "[cat vs table]" (bare word)
- Measure global energy at target, compare across conditions

**Prediction:** Full context > Partial > None for facilitation of
congruent targets (larger N400 difference with more context).

**File:** `research/experiments/applications/test_n400_sentence_context.py`

### 2C. Graded Relatedness ✅

**Question:** Does global energy scale with degree of semantic
relatedness, not just related vs unrelated?

**Design:** Use feature overlap as a continuous measure:
- Same category, many shared features (dog → cat: ANIMAL + co-training)
- Same category, fewer shared features (dog → fish: ANIMAL only)
- Different category, some association (dog → ball: co-training)
- Completely unrelated (dog → book: no shared features or co-training)

**Prediction:** Global energy decreases monotonically with relatedness.

**File:** `research/experiments/applications/test_n400_graded.py`

**Result:** Binary structure, not smooth gradient. Identity (d=-20.7, p=0.0008)
shows massive facilitation, but within-category primes barely differ from
unrelated (d=-0.13). Co-occurrence frequency doesn't create additional
differentiation when all words share the same category feature equally.

### 1B. Vocabulary Scaling ✅

**File:** `research/experiments/applications/test_n400_vocab_scaling.py`

**Result:** Effect persists across all scales: Small d=-57.7, Medium d=-13.8,
Large d=-26.7. All significant. Weakens from small to medium (as predicted
by the N400 literature) but remains robust.

---

## Phase 3: Theoretical Analysis — PARTIALLY COMPLETE

### 3A. Why Global Energy Works ✅

**Question:** Can we derive analytically WHY `sum(all_inputs)` shows
facilitation while `mean(all_inputs[target_neurons])` shows interference?

**Sketch:**
- Let W be the Hebbian weight matrix after training
- Related prime activates assembly R with overlap δ to target T
- `all_inputs = W[R,:].sum(axis=0)` (input from R's active neurons)
- Target-neuron input: `sum(all_inputs[T])` — dominated by competition
  between R and T for shared neurons
- Global input: `sum(all_inputs)` — includes ALL secondary activation
  from R's trained connections, not just T's neurons

The key: Hebbian learning creates diffuse connections from R to many
neurons (not just T), and these contribute to global energy. Related
primes have more diffuse connections (via shared features) → more
global activation.

**Deliverable:** See `research/plans/N400_MATHEMATICAL_ANALYSIS.md`.

### 3B. Mapping to ERP Literature

**Question:** How precisely does `sum(all_inputs)` map to scalp-recorded
N400 amplitude?

**Analysis:**
- N400 is measured as voltage deflection at central-parietal electrodes
- ERP = sum of post-synaptic potentials from large neuron populations
- `all_inputs` = total synaptic input across all neurons in a region
- The mapping is: `N400_amplitude ∝ -sum(all_inputs)` (negative because
  the N400 is a negative-going deflection, larger absolute value for
  unexpected words)
- This is exactly the "ease of access" interpretation (Kutas &
  Federmeier 2011)

**Deliverable:** Discussion section mapping AC quantities to ERP measurements.

---

## Phase 4: Beyond N400 — MOSTLY COMPLETE

### 4A. P600 (Syntactic Violations) ✅

**File:** `research/experiments/applications/test_p600_syntactic.py`

**Design:** Three conditions at the object position of SVO sentences:
- Grammatical: "the dog chases the cat" (trained animal)
- Semantic violation: "the dog chases the table" (untrained object)
- Category violation: "the dog chases the likes" (verb in noun position)

Three processing pipeline stages:
1. **Bootstrap**: Materialize random baseline weights for all core→structural pairs
2. **Consolidation**: Replay role/phrase training WITHOUT reset_area_connections(),
   creating persistent Hebbian connections for trained pathways
3. **Measurement**: N400 during word settling, P600 during structural integration

**Results:**

| Metric | Comparison | d | p | Direction |
|--------|-----------|---|---|-----------|
| N400 (core energy) | sem vs gram | 7.9 | 0.0001 | N400_EFFECT ✓ |
| Core instability | sem vs gram | 32.7 | <0.0001 | HIGHER_INSTAB ✓ |
| P600 instability | cat vs gram | 5.7 | 0.0002 | P600_EFFECT ✓ |
| P600 instability | sem vs gram | 0.11 | 0.814 | null (correct) |
| P600 instability | cat vs sem | 5.6 | 0.0002 | P600_EFFECT ✓ |

**Key findings:**

1. **N400 in sentence context CONFIRMED**: Semantic violations produce
   elevated core area energy (d=6.4, p=0.008). Extends the word-pair
   N400 finding to sentence-level processing.

2. **Core-area instability DISCOVERED**: Untrained nouns show 3x higher
   Jaccard instability during settling in NOUN_CORE (3.75 vs 1.33,
   d=32.7, p<0.0001). Without Hebbian self-recurrence, assemblies wobble.
   This captures lexical-semantic settling difficulty.

3. **P600 via structural instability CONFIRMED**: After consolidation,
   category violations show elevated Jaccard instability in structural
   areas (d=5.7, p=0.0002). Per-area: ROLE_AGENT d=39.1, ROLE_PATIENT
   d=26.2. Trained NOUN_CORE→ROLE converges in 1-2 rounds (instab ~0.1),
   random VERB_CORE→ROLE oscillates (instab ~1.5).

4. **P600 selective for structural violations**: Semantic violations
   show NULL P600 (d=0.11, p=0.81). This matches the literature:
   P600 is selective for syntactic/structural violations, not semantic
   (Friederici 2002, Kuperberg 2007).

5. **TRIPLE DISSOCIATION ACHIEVED**:
   - N400 (core energy): Selective for semantic anomalies
   - Core instability: Lexical-semantic settling difficulty
   - P600 (structural instability): Selective for structural anomalies

**Three innovations required:**
1. **Bootstrap connectivity**: 3-step projection trick materializes
   random baseline weights for empty core→structural weight matrices
2. **Consolidation pass**: Replays role/phrase training WITHOUT
   reset_area_connections(), creating persistent Hebbian connections
3. **Per-round Jaccard tracking**: Instability in both core and
   structural areas during settling rounds

**Theory:** See `research/plans/P600_REANALYSIS.md` for the settling
time interpretation (Vosse & Kempen 2000), integration update cost
(Brouwer & Crocker 2017), consolidation rationale, and per-area results.

### 4B. N400/P600 Triple Dissociation ✅

**ACHIEVED** via three distinct metrics:
- N400 (core energy): Selective for semantic violations
- Core instability: Lexical settling difficulty (sem >> gram)
- P600 (structural instability): Selective for category violations
- P600 null for semantic violations (correct: they're syntactically valid)

### 4C. Mismatch Negativity (MMN)

**Question:** Does the global-energy framework generalize to other
predictive ERP components beyond N400?

The MMN is an earlier component (~150ms) reflecting prediction error
for simple auditory features. If `sum(all_inputs)` works for both
N400 (semantic) and MMN (perceptual), the framework has genuine
generality.

**Status:** Not yet implemented.

---

## Priority and Dependencies

```
Phase 1 (strengthen core claim) — COMPLETED
├── 1A: Parameter sweep ──────── ✅ k≥100 AND p≥0.05 required
├── 1B: Vocabulary scaling ───── ✅ persists at 12-46 nouns
├── 1C: Control conditions ───── ✅ 4/4 controls pass
└── 1D: Engine parity ────────── ✅ torch_sparse matches numpy

Phase 2 (richer paradigms) — MOSTLY COMPLETE
├── 2A: Cloze probability ───── ✅ monotonic ordering confirmed
├── 2B: Sentence context ────── (subsumed by P600 experiment)
└── 2C: Graded relatedness ──── ✅ identity priming massive, within-category binary

Phase 3 (theory) — COMPLETE
├── 3A: Mathematical analysis ── ✅ derivation complete
└── 3B: ERP mapping ─────────── ✅ documented in claims

Phase 4 (beyond N400) — MOSTLY COMPLETE
├── 4A: P600 ──────────────────── ✅ N400 confirmed (d=7.9)
│                                    Core instability: sem>>gram (d=32.7)
│                                    P600 instability: cat>gram (d=5.7)
│                                    P600 null for sem (d=0.11, correct)
│                                    Bootstrap + consolidation + Jaccard
├── 4B: N400/P600 dissociation ── ✅ TRIPLE DISSOCIATION ACHIEVED
└── 4C: MMN ───────────────────── not yet implemented
```

## Remaining Work

1. **Agreement violations:** Test "the dogs *chases* the cat" where
   the noun category is correct but morphosyntactic features mismatch.
   This may produce a different P600 pattern.

2. **Sentence context (2B):** Systematic comparison of N400 effect
   with varying amounts of sentence context (bare word vs partial
   vs full context).

3. **MMN (4C):** Test whether global energy generalizes to perceptual
   prediction error, not just semantic.

4. **Larger vocabulary scaling:** Test with 100+ nouns to approach
   realistic vocabulary sizes.

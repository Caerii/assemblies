# P600 Reanalysis: Bootstrap Connectivity and Assembly Instability

## Background

The P600 is a positive-going ERP component peaking ~600ms post-stimulus,
associated with syntactic violations, structural reanalysis, and
integration difficulty. Unlike the N400 (which we model as global
pre-k-WTA energy reflecting semantic access difficulty), the P600
reflects a later processing stage where the parser attempts to integrate
a word into the sentence's syntactic structure.

## Theoretical Accounts of the P600

### 1. Reanalysis/Repair (Friederici 2002)

The P600 reflects Phase 3 of the neurocognitive model of sentence
processing: controlled, strategic reanalysis when initial parsing fails.
Syntactic violations trigger garden-path recovery, with P600 amplitude
reflecting the cost of structural repair.

### 2. Unification Difficulty (Hagoort 2005)

In the Memory, Unification, and Control (MUC) model, the P600 reflects
unification difficulty — the computational cost of binding a word's
syntactic features into the sentence's evolving structure. Violations
increase unification cost because the incoming features conflict with
expectations.

### 3. Settling Time / Competitive Inhibition (Vosse & Kempen 2000)

In the competitive inhibition model, syntactic parsing proceeds via
lateral inhibition between competing structural analyses. The P600
reflects settling time — how long it takes the network to converge on
a single parse. Violations increase settling time because the correct
parse receives less support.

### 4. Integration Update Cost (Brouwer & Crocker 2017)

In the Retrieval-Integration account, the P600 reflects the update
cost when integrating a word into the discourse representation. The
N400 reflects retrieval of lexical-semantic features, while the P600
reflects the cost of updating the mental model. Semantic anomalies
can produce both N400 and P600 (biphasic response) when they also
disrupt structural integration.

### 5. Cue-Based Retrieval Interference (Lewis & Vasishth 2005)

Syntactic processing relies on cue-based retrieval from working
memory. The P600 reflects retrieval interference when multiple items
match retrieval cues. Violations create cue conflicts, increasing
interference and processing cost.

### 6. Biphasic Model (Kuperberg 2007)

Semantic anomalies can produce BOTH N400 and P600 when they disrupt
both semantic access (N400) and thematic role assignment (P600).
Category violations produce primarily P600 (structural), while
semantic violations produce a biphasic N400+P600 response.

## Our Model: Cumulative Structural Energy

We model the P600 as **cumulative pre-k-WTA energy during structural
integration**. After a word settles in its core area (lexical
processing, ~N400 time window), the parser projects the core
assembly into structural areas (SUBJ, OBJ, ROLE_AGENT,
ROLE_PATIENT, VP) for role binding and phrase building. The total
energy across multiple settling rounds captures the computational
cost of structural integration.

### The Metric

```
P600 = Σ_{r=1}^{R} sum(all_inputs_r)
```

where `all_inputs_r` is the full pre-k-WTA input vector in the
structural area at settling round `r`.

- **Low cumulative energy** (small P600): The core assembly provides
  a compatible signal to the structural area. Lower total work.

- **High cumulative energy** (large P600): The core assembly provides
  an incompatible or anomalous signal. Higher total work.

### Why Cumulative Energy Works

The cumulative energy metric captures total structural processing
cost, which depends on the **core assembly's properties**:

1. **Grammatical continuations** (trained nouns in trained pathways):
   The core assembly was formed during sentence processing with
   Hebbian-trained self-recurrence weights. It provides a coherent,
   efficient signal to structural areas → lower total energy.

2. **Semantic violations** (untrained nouns): The core assembly
   was never reinforced through sentence-level training. It provides
   a less coherent signal → moderately higher total energy.

3. **Category violations** (verbs in noun position): The signal
   comes from a different core area entirely (VERB_CORE instead of
   NOUN_CORE), providing a maximally incompatible signal to the
   noun-expecting structural areas → highest total energy.

This maps to the **unification difficulty** interpretation
(Hagoort 2005): more total neural work when the incoming
representation doesn't match the structural template.

### Assembly Instability WITH Consolidation (Primary P600 Metric)

Assembly instability (Jaccard distance between successive winner sets)
works as the P600 metric, but ONLY after a consolidation pass that
creates persistent Hebbian connections for trained core→structural
pathways.

**The problem:** During `train_roles()` and `train_phrases()`, the
parser calls `reset_area_connections()` after each word/sentence,
wiping all Hebbian-trained core→structural weights. After training,
ALL core→structural pathways have only random baseline weights.
Without the trained/untrained asymmetry, instability is near ceiling
for all conditions (~2.3/4.0).

**The solution: Consolidation pass.** After bootstrap (which
materializes random baseline weights), replay the same role binding
and phrase structure projections WITHOUT the reset. This creates
Hebbian-strengthened connections for trained pathways:

- NOUN_CORE→ROLE_AGENT: Hebbian-strengthened (trained animal nouns)
- NOUN_CORE→ROLE_PATIENT: Hebbian-strengthened (trained animal nouns)
- NOUN_CORE→VP: Hebbian-strengthened (trained subjects/objects)
- VERB_CORE→VP: Hebbian-strengthened (trained verbs)
- VERB_CORE→ROLE_*: Random only (verbs don't get role annotations)
- *→SUBJ/OBJ: Random only (never explicitly trained)

This asymmetry makes instability differentiate:
- **Trained pathways** (NOUN_CORE→ROLE): Assembly converges in 1-2
  rounds (instability ~0.1-0.2)
- **Random pathways** (VERB_CORE→ROLE): Assembly oscillates across
  all rounds (instability ~1.5-1.6)

The per-area breakdown confirms this mechanism:

| Area | Gram (trained nouns) | Cat viol (verbs) | d | p |
|------|---------------------|------------------|---|---|
| ROLE_AGENT | 0.19 | 1.58 | 20.5 | 0.001 |
| ROLE_PATIENT | 0.10 | 1.55 | 58.8 | <0.001 |
| VP | 0.03 | 0.17 | 10.9 | 0.003 |
| SUBJ (random) | 2.42 | 2.29 | -2.5 | 0.048 |
| OBJ (random) | 2.39 | 2.30 | -1.5 | 0.126 |

Consolidated areas (ROLE_AGENT, ROLE_PATIENT, VP) show enormous
differentiation. Random-only areas (SUBJ, OBJ) remain near ceiling
with no meaningful differentiation.

### Why Cumulative Energy Reverses After Consolidation

Before consolidation, cumulative energy worked as a P600 metric
(cat > sem > gram) because all pathways had random weights and the
energy differences came from core assembly properties.

After consolidation, cumulative energy is REVERSED: trained pathways
produce MORE total energy (Hebbian-strengthened weights amplify the
signal), while random pathways produce less. This is the opposite
of the P600 pattern. Cumulative energy only works as P600 when all
pathway weights are uniformly random.

## Bootstrap Connectivity

### The Problem

In the sparse engine implementation, area-to-area weight matrices start
as empty (0×0) and are materialized on-demand during the first projection
through each pathway. After `parser.train()`, pathways that were never
exercised still have empty weight matrices:

- **VERB_CORE → SUBJ/OBJ**: Verbs are never subjects or objects in training
- **VERB_CORE → ROLE_PATIENT**: Verbs aren't assigned patient roles
- Other unexpercised core→structural combinations

When attempting to project through an empty pathway, the sparse engine's
zero-signal early return (`_sparse.py:167-173`) prevents `_expand_connectomes`
from running, creating a catch-22: we can't materialize weights because we
can't project, and we can't project because there are no weights.

### The Solution

We **bootstrap** connectivity by forcing a projection through each empty
pathway with plasticity OFF:

```python
brain.disable_plasticity = True
# Step 1: Ensure structural area has winners (w > 0)
brain.project({arbitrary_stimulus: [struct_area]}, {})
# Step 2: Ensure core area has winners
brain.project({arbitrary_stimulus: [core_area]}, {})
# Step 3: Project stimulus + core → struct
# Stimulus provides non-zero signal (avoids zero-signal early return)
# Core in from_areas triggers _expand_connectomes for core→struct
brain.project(
    {arbitrary_stimulus: [struct_area]},
    {core_area: [struct_area]},
)
brain.disable_plasticity = False
```

Step 3 is the key: the stimulus provides non-zero input (avoiding the
zero-signal early return), while the core area's presence in `from_areas`
triggers `_expand_connectomes` to materialize the `core → struct` weight
matrix with random binomial(p) baseline values.

### Biological Plausibility

This models the biological reality that anatomical fibers exist between
cortical areas before any learning occurs:

- The arcuate fasciculus connects Broca's and Wernicke's areas
- Thalamocortical projections provide baseline connectivity
- White matter tracts are established during development

Training strengthens specific pathways through Hebbian plasticity
(long-term potentiation), but all pathways start with some baseline
connectivity. The bootstrapped weights (binomial with p≈0.05) model
this anatomical baseline — much weaker than Hebbian-trained connections
but not zero.

## Relationship to N400 Settling Dynamics (Path 3)

The N400 settling dynamics experiment (`test_n400_pre_kwta.py`, Path 3)
measures cumulative pre-k-WTA energy across rounds as a word settles in
its core area. The P600 cumulative energy metric differs in two ways:

1. **Different target areas**: N400 settling is in core areas (NOUN_CORE);
   P600 settling is in structural areas (SUBJ, OBJ, etc.)

2. **Different mechanism**: N400 settling reflects lexical access
   difficulty (how hard to activate the word's assembly); P600 settling
   reflects structural integration difficulty (how hard to bind the
   word into syntactic structure)

The two metrics capture different stages of language processing:
N400 (~400ms) for lexical-semantic access, P600 (~600ms) for
syntactic-structural integration.

## Results (With Consolidation, 5 Seeds)

| Condition | N400 (core energy) | Core instability | P600 instability | P600 cum. energy |
|-----------|-------------------|-----------------|-----------------|-----------------|
| Grammatical ("cat") | 23537 (LOW) | 1.33 (LOW) | 1.14 (LOW) | 346295 (HIGH) |
| Semantic violation ("table") | 24872 (HIGH) | 3.75 (HIGH) | 1.15 (NULL) | 282778 (MED) |
| Category violation ("likes") | 15303 (VERB_CORE) | 2.03 (MED) | 1.59 (HIGH) | 264303 (LOW) |

### N400: Confirmed (d=7.9, p=0.0001)

Semantic violations produce elevated core area energy, replicating the
N400 finding from word-pair experiments in a sentence context.

### Core-Area Instability: Confirmed (d=32.7, p<0.0001)

Semantic violations produce dramatically higher Jaccard instability
within NOUN_CORE during word settling (3.75 vs 1.33). Untrained nouns
("table") have no Hebbian-trained self-recurrence in NOUN_CORE, so
their assembly wobbles across rounds. Trained nouns ("cat") converge
quickly via self-recurrence accumulated during training.

This is a lexical-semantic metric (measured in the core area during
word processing), complementing the N400 energy metric.

### P600 Structural Instability: Confirmed for Category Violations

| Comparison | d | p | Direction |
|-----------|---|---|-----------|
| cat vs gram | 5.7 | 0.0002 | P600_EFFECT ✓ |
| sem vs gram | 0.11 | 0.814 | null (correct) |
| cat vs sem | 5.6 | 0.0002 | P600_EFFECT ✓ |

The instability metric correctly distinguishes category violations
from grammatical completions (d=5.5, p=0.011) with correct grading
(cat > sem, d=3.5, p=0.027). Semantic violations do NOT show
elevated structural instability — they are syntactically valid
(a noun in a noun slot), just semantically unexpected.

This selectivity matches the neurolinguistic literature:
- **P600** is selective for structural/syntactic violations
  (Friederici 2002, Hagoort 2005)
- **Semantic violations produce N400 but not P600** in standard
  paradigms (Kutas & Federmeier 2011)
- **Category violations produce the largest P600** because they
  require maximal structural reanalysis (Kuperberg 2007)

### P600 Cumulative Energy: Reversed After Consolidation

After consolidation, cumulative energy shows the OPPOSITE pattern
(gram > sem > cat). Hebbian-strengthened connections amplify total
energy for trained pathways. This metric only works as P600 when
all pathways have uniformly random weights (pre-consolidation).

### Double Dissociation

The double dissociation is now CLEANER than before:

- **N400 (core energy)**: Selective for SEMANTIC anomalies
  (sem > gram, d=7.9, p=0.0001; cat vs gram irrelevant —
  different core area)
- **P600 (structural instability)**: Selective for STRUCTURAL
  anomalies (cat > gram, d=5.7, p=0.0002; sem vs gram null)
- **Core instability**: Captures lexical-semantic settling
  difficulty (sem > gram, d=32.7, p<0.0001)

This triple dissociation maps to three processing stages:
1. **Lexical access (~250ms)**: Core-area instability
2. **Semantic integration (~400ms)**: N400 (global energy)
3. **Structural integration (~600ms)**: P600 (structural instability)

## Falsification Criteria

1. **Consolidation doesn't create asymmetry**: If replaying training
   without reset doesn't strengthen specific pathways above baseline,
   the consolidation approach fails.
   **Status: PASSED** — ROLE_AGENT/ROLE_PATIENT instability drops from
   ~2.3 (random) to ~0.1-0.2 (consolidated) for trained nouns.

2. **Wrong direction**: If violations show LOWER instability than
   grammatical (trained pathways oscillate more), the model is wrong.
   **Status: PASSED** — cat (1.67) > gram (1.16), d=5.5, p=0.011.

3. **No selectivity for structural violations**: If semantic violations
   show the same P600 pattern as category violations, the metric doesn't
   distinguish syntactic from semantic processing.
   **Status: PASSED** — sem vs gram is null (d=0.25, p=0.70), while
   cat vs gram is significant (d=5.5, p=0.011). P600 is selective
   for structural violations, matching the literature.

4. **Instability doesn't correlate with P600 literature**: If the
   pattern doesn't match known P600 effects (larger for syntactic than
   semantic violations, largest for category violations), the metric
   lacks construct validity.
   **Status: PASSED** — cat > sem > gram ordering matches Friederici 2002
   and Kuperberg 2007.

## Implementation

- **Experiment**: `research/experiments/applications/test_p600_syntactic.py`
- **Bootstrap**: `infrastructure.bootstrap.bootstrap_structural_connectivity()` —
  materializes random baseline weights for all core→structural pairs (runs FIRST)
- **Consolidation**: `infrastructure.consolidation.consolidate_role_connections()`
  and `consolidate_vp_connections()` — replay role binding and phrase
  structure training WITHOUT `reset_area_connections()`, creating
  persistent Hebbian connections for trained pathways (runs SECOND)
- **P600 measurement**: `metrics.instability.measure_p600_settling()` —
  Jaccard-based instability across settling rounds in structural areas.
  Core assembly is fixed during measurement (stable lexical representation
  by ~600ms).
- **Core instability**: Per-round Jaccard tracking in core area during
  the 10 rounds of critical word settling. Trained words converge
  faster via self-recurrence.
- **Analysis**: Reports instability (primary P600), cumulative energy
  (reversed after consolidation), and core instability (lexical settling)

## Key References

- Brouwer, H., & Crocker, M. W. (2017). On the proper treatment of the
  N400 and P600 in language comprehension. *Frontiers in Psychology*, 8, 1327.
- Catani, M., & Mesulam, M. (2008). The arcuate fasciculus and the
  disconnection theme in language and aphasia. *Cortex*, 44(8), 953-961.
- Cheyette, S. J., & Plaut, D. C. (2017). Modeling the N400 ERP
  component as transient semantic over-activation within a neural
  network model of word comprehension. *Cognition*, 162, 153-166.
- Friederici, A. D. (2002). Towards a neural basis of auditory sentence
  processing. *Trends in Cognitive Sciences*, 6(2), 78-84.
- Hagoort, P. (2005). On Broca, brain, and binding: a new framework.
  *Trends in Cognitive Sciences*, 9(9), 416-423.
- Hagoort, P., Brown, C., & Groothusen, J. (1993). The syntactic positive
  shift (SPS) as an ERP measure of syntactic processing. *Language and
  Cognitive Processes*, 8(4), 439-483.
- Kuperberg, G. R. (2007). Neural mechanisms of language comprehension:
  Challenges to syntax. *Brain Research*, 1146, 23-49.
- Lewis, R. L., & Vasishth, S. (2005). An activation-based model of
  sentence processing as skilled memory retrieval. *Cognitive Science*,
  29(3), 375-419.
- Osterhout, L., & Holcomb, P. J. (1992). Event-related brain potentials
  elicited by syntactic anomaly. *Journal of Memory and Language*, 31(6),
  785-806.
- van Herten, M., Kolk, H. H. J., & Chwilla, D. J. (2005). An ERP study
  of P600 effects elicited by semantic anomalies. *Cognitive Brain
  Research*, 22(2), 241-255.
- Vosse, T., & Kempen, G. (2000). Syntactic structure assembly in human
  parsing: a computational model based on competitive inhibition and
  a lexicalist grammar. *Cognition*, 75(2), 105-143.

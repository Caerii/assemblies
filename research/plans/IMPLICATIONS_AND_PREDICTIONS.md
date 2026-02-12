# Implications and Predictions: N400/P600 Triple Dissociation in Assembly Calculus

## 1. Triple Dissociation: Significance

Three independent neural metrics emerge from a **single architecture** with
no modular assumptions — no separate "semantic module" or "syntactic parser":

| Stage | Metric | Brain correlate | Selective for | Effect size |
|-------|--------|----------------|---------------|-------------|
| Lexical access (~250ms) | Core-area instability | Early left anterior negativity? | Lexical settling difficulty | d=32.7 |
| Semantic integration (~400ms) | Global pre-k-WTA energy | N400 | Semantic anomaly | d=7.9 |
| Structural integration (~600ms) | Structural instability | P600 | Syntactic violation | d=5.7 |

### Why this matters

Most neurolinguistic models require **separate processing modules** to explain
the N400/P600 dissociation (Friederici 2002, Hagoort 2005). Our result shows
that a single substrate — recurrent Hebbian assemblies with k-WTA competition —
produces all three signatures through different aspects of the same computation:

- **N400 (energy):** How much total work the network does during lexical access.
  Measured as `sum(all_inputs)` before winner selection. Related primes reduce
  energy via feature overlap.

- **Core instability (Jaccard):** How much the winner set changes across rounds
  in the core area. Trained words have self-recurrence that stabilizes the
  assembly; untrained words oscillate.

- **P600 (structural instability):** How much the winner set changes in
  structural areas (ROLE_AGENT, ROLE_PATIENT, VP). Hebbian-consolidated
  core→structural pathways produce stable assemblies; unconsolidated pathways
  (VERB_CORE→ROLE_*) produce oscillation.

This maps directly to Friederici's (2002) three-phase model:
1. Phase 1 (~150ms): Initial phrase structure building — **core instability**
2. Phase 2 (~400ms): Lexical-semantic integration — **N400 energy**
3. Phase 3 (~600ms): Syntactic reanalysis/repair — **P600 instability**

The key theoretical contribution is that these phases are not separate
*modules* but different *measurement windows* on the same recurrent settling
process.


## 2. Consolidation as a Model of Learning

### The mechanism

Training in the Assembly Calculus parser uses `reset_area_connections()` after
each word/sentence, creating temporary connections analogous to working memory.
Consolidation replays the same projections WITHOUT the reset, creating
persistent Hebbian connections analogous to long-term memory.

This maps to the **complementary learning systems** framework
(McClelland et al. 1995, O'Reilly & Norman 2002):

| AC concept | Cognitive analogue | Neural correlate |
|------------|-------------------|-----------------|
| `reset_area_connections()` | Working memory clearance | Hippocampal pattern separation |
| Training with reset | Episodic encoding | Fast hippocampal learning |
| Consolidation (replay without reset) | Systems consolidation | Slow cortical learning |
| Hebbian-strengthened pathways | Procedural knowledge | Cortical LTP |

### Implications

1. **The parser's training cycle mirrors sleep consolidation.** Training
   encodes individual episodes (sentences) with interference prevention (reset).
   Consolidation replays these episodes to extract statistical regularities
   (which word categories go in which structural slots). This is structurally
   analogous to hippocampal replay during sleep.

2. **The reset is necessary for generalization.** Without reset, Hebbian
   learning would create item-specific connections (e.g., "cat" always goes
   to the same role neurons). The reset forces each training episode to
   create a fresh assembly in the role area, and consolidation then
   strengthens the PATHWAY (core→role) rather than the specific neurons.
   This produces category-level structural knowledge.


## 3. Testable Predictions

### 3.1 Agreement violations (morphosyntactic)

**Setup:** "The dogs *chases* the cat" — correct word category (verb) but
wrong morphological features (singular vs plural agreement).

**Prediction:** P600 should be INTERMEDIATE between grammatical and category
violation. The verb→VP pathway is Hebbian-consolidated (verbs were trained
in VP merge), so structural integration proceeds via strengthened connections.
But the agreement mismatch creates a secondary instability as the parser
attempts to bind the singular verb with the plural subject's assembly.

Expected: `P600(cat_viol) > P600(agree_viol) > P600(gram)`

**Implementation:** Extend vocabulary with plural markers (e.g., "dogs" with
feature ["DOG", "ANIMAL", "PLURAL"]) and singular/plural verb forms. The
agreement violation uses a verb that IS Hebbian-consolidated but triggers
feature mismatch during role binding.

### 3.2 Garden-path recovery

**Setup:** "The horse raced past the barn fell"

**Prediction:** Elevated P600 at "fell" because the parser must reanalyze
"raced" from main verb to reduced relative clause. The initial parse
(horse→AGENT, raced→VP) creates Hebbian-strengthened connections that must
be overridden — this should produce high structural instability in
ROLE_AGENT and VP as the system attempts to reassign "horse" from agent
to patient.

**Implementation:** Requires multi-clause sentence processing. The parser
would need to maintain active assemblies across clause boundaries. Current
architecture supports this (assemblies persist in brain areas unless
explicitly inhibited).

### 3.3 Relative clause complexity

**Setup:** Compare center-embedded ("The cat that the dog chased ran") vs
right-branching ("The dog chased the cat that ran").

**Prediction:** Center-embedded should produce higher structural instability
because:
- The subject ("cat") must be held active while processing the embedded
  clause ("the dog chased")
- At "ran", the parser must retrieve "cat" as agent despite "dog" being
  more recently active
- This creates retrieval interference in ROLE_AGENT (Lewis & Vasishth 2005)

Expected: `P600(center-embedded) > P600(right-branching)`

### 3.4 Cross-linguistic word order

**Setup:** Train separate parsers on SVO ("dog chases cat") and SOV
("dog cat chases") word orders.

**Prediction:** Each parser should show N400/P600 effects for violations
of ITS trained word order. An SVO-trained parser should show elevated P600
for SOV sentences (verb in wrong position), and vice versa. The specific
structural areas showing instability should differ:

- SVO parser: Violations at object position → ROLE_PATIENT instability
- SOV parser: Violations at verb position → VP instability

### 3.5 L2 processing (unconsolidated grammar)

**Setup:** Compare a fully consolidated parser (L1) vs a parser trained
but NOT consolidated (L2 beginner).

**Prediction:** The unconsolidated parser should show elevated P600 for
ALL conditions, including grammatical sentences. Without Hebbian-
strengthened core→structural connections, even correct structures require
settling through random pathways.

This maps to the L2 processing literature:
- Steinhauer et al. (2009): L2 learners show delayed/reduced P600
- Tanner et al. (2014): L2 P600 depends on proficiency
- Our model predicts: L2 P600 amplitude inversely correlates with
  consolidation degree (i.e., hours of immersion/practice)

Specific prediction: `P600(L2_gram) ≈ P600(L1_sem_viol)` — an L2 learner
processing grammatical sentences should show similar structural instability
to an L1 speaker encountering semantic violations, because both involve
unconsolidated pathways.

### 3.6 Developmental trajectory

**Setup:** Vary the number of consolidation rounds (1, 5, 10, 50) to
model developmental stages.

**Prediction:** P600 amplitude for grammatical sentences should DECREASE
with more consolidation rounds (stronger Hebbian connections → faster
convergence), while P600 for category violations remains HIGH (VERB_CORE→ROLE
is never consolidated regardless of rounds).

This maps to developmental ERP literature:
- Hahne et al. (2004): Children show delayed P600 relative to adults
- Friedrich & Friederici (2005): P600 matures with syntactic experience

The model predicts a specific relationship:
```
P600_gram(consolidation_rounds) ~ 1/log(rounds)
P600_cat(consolidation_rounds) ~ constant
```

### 3.7 Clinical predictions (SLI/DLD)

**Setup:** Model specific language impairment (SLI) / developmental language
disorder (DLD) as impaired consolidation (lower Hebbian learning rate during
consolidation, or fewer consolidation rounds).

**Prediction:** SLI/DLD patients should show:
- Elevated P600 for ALL conditions (weak consolidation → all pathways
  remain closer to random baseline)
- PRESERVED N400 (core-area energy depends on stimulus-recurrence
  overlap, not on core→structural connections)
- This pattern (impaired P600, preserved N400) is exactly what the
  SLI/DLD literature shows (Fonteneau & van der Lely 2008)

### 3.8 Adaptation / habituation to violations

**Setup:** Present repeated category violations in sequence (e.g., multiple
sentences with verbs in noun position).

**Prediction:** P600 should DECREASE across repetitions. Each violation
triggers a projection through VERB_CORE→ROLE with Hebbian learning ON,
gradually strengthening these "incorrect" pathways. After enough repetitions,
the instability should decrease, modeling syntactic adaptation.

This maps to:
- Coulson et al. (1998): P600 habituation with repeated violations
- Kaan (2007): Syntactic priming reduces P600

### 3.9 N400 × P600 interaction

**Setup:** "The dog chases the honestly" (adverb in noun position — both
semantic anomaly and category violation).

**Prediction:** Should produce BOTH elevated N400 (different core area →
no facilitation from context) AND elevated P600 (unconsolidated pathway
→ structural instability). This biphasic response is well-documented
(Kuperberg 2007) but has been difficult to model with single-mechanism
accounts.


## 4. Scaling Requirements

### Current state

- **Vocabulary:** 8-17 words (4-6 animals, 2-6 objects, 4 verbs, 1 function)
- **Categories:** 2-3 (animals, objects/furniture, vehicles)
- **Training:** 20-40 SVO sentences
- **Parameters:** n=50000, k=100, p=0.05, beta=0.05

### Medium-term targets (next experiments)

- **50+ words:** Multiple categories with graded feature overlap
  - Animals: 6 domestic + 6 wild (overlap on ANIMAL but differ on
    DOMESTIC/WILD)
  - Objects: furniture, tools, foods, clothing (4 categories × 6)
  - Verbs: transitive, intransitive, ditransitive (for argument structure)
  - Function words: determiners, prepositions, complementizers

- **Feature hierarchy:** Currently binary (ANIMAL, FURNITURE). Need
  hierarchical features:
  ```
  DOG → [ANIMAL, DOMESTIC, CANINE, MAMMAL]
  CAT → [ANIMAL, DOMESTIC, FELINE, MAMMAL]
  WOLF → [ANIMAL, WILD, CANINE, MAMMAL]
  SNAKE → [ANIMAL, WILD, REPTILE]
  ```
  This creates graded similarity (DOG-CAT share 3/4 features, DOG-WOLF
  share 3/4 but different ones, DOG-SNAKE share only 1/4).

- **Frequency distributions:** Currently uniform training. Need Zipfian
  frequency distributions matching natural language statistics.

### Long-term targets

- **1000+ words:** Requires computational scaling. Current n=50000 with
  p=0.05 gives ~2500 active connections per neuron. May need n=500000+
  for 1000 words to avoid assembly interference.

- **Hierarchical category structure:** Superordinate, basic, subordinate
  levels (e.g., LIVING_THING > ANIMAL > DOG > POODLE).

- **Distributional semantics integration:** Ground word features from
  co-occurrence statistics (word2vec, GloVe) rather than hand-crafted
  category labels. This would test whether the N400 mechanism works with
  continuous-valued feature overlap, not just categorical.

### What the vocab module needs to support

The `research/experiments/vocab/` module currently supports:
- Fixed vocabulary definitions (`standard.py`, `scaling.py`)
- Auto-generated training sentences (`training.py`, `scaling.py`)
- Auto-generated test pairs from category structure (`scaling.py`)

Future requirements:
- **Parametric vocabulary generation:** Generate vocabularies with
  controlled properties (n_categories, words_per_category, feature_depth,
  feature_overlap_ratio)
- **Hierarchical feature trees:** Define feature hierarchies that produce
  graded similarity automatically
- **Frequency control:** Specify target frequency distributions for
  training sentence generation
- **Cross-linguistic templates:** SVO, SOV, VSO, and other word order
  templates for the same vocabulary


## 5. Dynamics to Capture

### Round-by-round settling trajectories

Currently we report aggregate metrics (total instability, cumulative
energy). The round-by-round trajectory contains richer information:

```
Round 1: Assembly forms from stimulus + recurrence
Round 2: k-WTA competition selects winners
Round 3: Hebbian recurrence reinforces winners (if trained)
Round 4+: Assembly converges (trained) or oscillates (untrained)
```

The SHAPE of this trajectory matters:
- **Trained words:** Exponential convergence (instability drops rapidly)
- **Untrained words:** Oscillatory instability (bounces between states)
- **Category violations:** Flat high instability (no convergence)

These trajectory shapes map to ERP waveform morphology:
- The N400 has a characteristic onset (~250ms) and peak (~400ms)
- The P600 has a broader distribution (~500-800ms)
- Trajectory analysis could map AC rounds to ERP time course

### Individual differences as seed variance

Seed-to-seed variance in our experiments models individual differences.
Currently we average across seeds. But the VARIANCE is informative:

- High variance in P600 across seeds → parameter regime is unstable
- Low variance in N400 → robust, bottom-up mechanism
- Correlations between N400 and P600 across seeds → shared vs independent
  processing stages

### Priming × context interactions

Current experiments test either word-pair priming (N400) or sentence
context (P600) separately. The interaction is crucial:

- How does sentence context modulate single-word priming?
- Does a semantically constraining context ("the dog chases the ___")
  ENHANCE or REDUCE the N400 priming effect for related completions?
- This is the cloze probability × relatedness interaction, which is
  central to the predictive processing debate (Kuperberg & Jaeger 2016)


## 6. Connections to Broader Theory

### Predictive processing

The N400 maps naturally to prediction error in predictive coding
frameworks. Global energy = total deviation from predicted state.
Related primes reduce prediction error (lower energy). This connects
to Nour Eddine et al. (2024): N400 = lexico-semantic prediction error.

The P600 then maps to model UPDATE cost — the computational work
required to revise the structural model when predictions fail. This
aligns with Brouwer & Crocker (2017).

### Free energy principle

The Assembly Calculus k-WTA competition can be interpreted as variational
inference: the brain selects the k neurons that minimize total energy
(free energy). Global pre-k-WTA energy is literally the free energy
before the variational step. The N400 is thus a direct observable of
the brain's free energy at the lexical-semantic level.

### Binding problem

Assemblies are neural implementations of variable binding. The
consolidation mechanism shows how the brain can learn to bind specific
types (e.g., nouns) to specific roles (e.g., agent) while maintaining
generalization across instances. The P600 reflects the cost of binding
failure — when the system cannot efficiently bind an incoming
representation to an expected structural slot.


## References

- Brouwer, H., & Crocker, M. W. (2017). On the proper treatment of the
  N400 and P600 in language comprehension. *Frontiers in Psychology*, 8, 1327.
- Coulson, S., King, J. W., & Kutas, M. (1998). Expect the unexpected:
  Event-related brain response to morphosyntactic violations.
  *Language and Cognitive Processes*, 13(1), 21-58.
- Fonteneau, E., & van der Lely, H. K. J. (2008). Electrical brain
  responses in language-impaired children reveal grammar-specific deficits.
  *PLoS ONE*, 3(3), e1832.
- Friederici, A. D. (2002). Towards a neural basis of auditory sentence
  processing. *Trends in Cognitive Sciences*, 6(2), 78-84.
- Friedrich, M., & Friederici, A. D. (2005). Phonotactic knowledge and
  lexical-semantic processing in one-year-olds. *Brain Research*, 1047(1), 10-21.
- Hagoort, P. (2005). On Broca, brain, and binding: a new framework.
  *Trends in Cognitive Sciences*, 9(9), 416-423.
- Hahne, A., Eckstein, K., & Friederici, A. D. (2004). Brain signatures
  of syntactic and semantic processes during children's language development.
  *Journal of Cognitive Neuroscience*, 16(7), 1302-1318.
- Kaan, E. (2007). Event-related potentials and language processing.
  *Language and Linguistics Compass*, 1(6), 571-591.
- Kuperberg, G. R. (2007). Neural mechanisms of language comprehension:
  Challenges to syntax. *Brain Research*, 1146, 23-49.
- Kuperberg, G. R., & Jaeger, T. F. (2016). What do we mean by prediction
  in language comprehension? *Language, Cognition and Neuroscience*, 31(1), 32-59.
- Lewis, R. L., & Vasishth, S. (2005). An activation-based model of
  sentence processing as skilled memory retrieval. *Cognitive Science*,
  29(3), 375-419.
- McClelland, J. L., McNaughton, B. L., & O'Reilly, R. C. (1995). Why
  there are complementary learning systems in the hippocampus and neocortex.
  *Psychological Review*, 102(3), 419-457.
- Nour Eddine, S., Brothers, T., Wang, L., Spratling, M., & Kuperberg, G. R.
  (2024). A predictive coding model of the N400. *Cognition*, 246, 105755.
- O'Reilly, R. C., & Norman, K. A. (2002). Hippocampal and neocortical
  contributions to memory. *Trends in Cognitive Sciences*, 6(12), 505-510.
- Steinhauer, K., White, E. J., & Drury, J. E. (2009). Temporal dynamics
  of late second language acquisition. *Studies in Second Language
  Acquisition*, 31(1), 99-130.
- Tanner, D., Inoue, K., & Osterhout, L. (2014). Brain-based individual
  differences in online L2 grammatical comprehension. *Bilingualism:
  Language and Cognition*, 17(2), 277-293.

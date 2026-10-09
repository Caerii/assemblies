# Papers: what the substrate has earned, in what order, and what each still needs

Status: plan, written 2026-09-09. It cites register IDs from
`neural_assemblies/theory.py` (rendered at [../../docs/register.md](../../docs/register.md))
and the registrations under [../notes/](../notes/README.md). Nothing here is
evidence; every number below has a registration behind it, and the
registration is the source.

This document reconciles the earlier plans with what was measured since they
were written. Read it after
[ASSEMBLY_SYSTEMS_PAPER_VISION.md](ASSEMBLY_SYSTEMS_PAPER_VISION.md), whose
thesis it keeps, and before starting anything under
[../papers/](../papers/README.md).

**Reconciled 2026-10-06** with Amendments 17 to 26 of the memory line.
They moved three things. P1's open scaling question has an answer at the
optimum (the in-degree, Amendment 21), and its learning-rate law is a
fraction of the field's own convergence threshold (Amendment 17). P4 has a
measured phase structure of its own (Amendments 18, 23, 24). And a new
paper, P7, comes out of Amendments 25 and 26: when the write happens, and
how strongly the area adapts, decide whether one recurrent area stores
attractors or sequences -- which also gives a measured reason why the
calculus's sequence operation does not advance in this package. The
ladder, P1, P4, the new P7, the order and the debts below are updated; this
section is kept as written.

## 1. What the earlier plans asked for, and what happened

**The vision paper.** [ASSEMBLY_SYSTEMS_PAPER_VISION.md](ASSEMBLY_SYSTEMS_PAPER_VISION.md)
proposed one long systems paper: assemblies as measurable macrostates,
operations as dynamical protocols with regime contracts, and a regime-first
methodology. Its one-sentence center was that we characterize the regimes
under which assembly operations become reliable primitives. That center
survived and became the working method: every result now enters through a
pre-registered bar, and adopted results live in the register with an ID and
a status of PROVED, MEASURED, or EXTENSION. Its five priority experiments
fared unevenly.

| Vision priority | What happened |
|-----------------|---------------|
| Finite-size scaling of projection stability | Done for capacity instead: the ceiling is a function of n/k alone (`CAP-RATIO`), and the refracted ceiling follows a square law across seven (n, k) cells to 16,384 items (`REFRACTION-ANTI-MERGING`). |
| Pattern-completion basin maps | Not done as a registered line. The half-cue recall curves in the memory line are the closest thing. |
| Merge and association reliability over seeds | The Phase 3 audit of [SOUNDNESS_PROGRAM.md](SOUNDNESS_PROGRAM.md) found the standing evidence for reactivation and retrieval to be dead probes (winner sets fixed from round one, independent of beta). Nothing there is citable. |
| LRI recall phase diagrams | Replaced by the refracted-arc transition machine and its soft-transition census (`SEQ-EXACT-RECOVERY`). |
| Ablations of recurrence, plasticity, feedback, inhibition | Done as arms inside registrations: refraction strength, norm_init, the convergence gate, the state-blind and register-blind transducers. |

One long paper is now the wrong shape. The results split into claims with
different audiences and different readiness, and a single manuscript would
hold the weakest of them hostage to the strongest.

**The three throughlines.** [THEORETICAL_THROUGHLINES.md](THEORETICAL_THROUGHLINES.md)
asked for statistical mechanics, linear algebra, and network physics views,
and [PRIORITIES_AND_GAPS.md](PRIORITIES_AND_GAPS.md) asked for one derived
result. The linear-algebra view produced two proved identities
(`HEBB-OUTER-PRODUCT`, `DRIVE-SPLIT`) and the exact count-then-apply drive
the GPU substrate is built on. The sequence line produced the one derived
number that then matched a measurement: the presentation at which a synapse
potentiated once per presentation reaches the weight clip,
c* = ln(w_max · max(1, kp) / base) / ln(1 + beta), predicted 28 to 31 and
measured as the edge of the training window
([PREREG_s5_cliff_anatomy.md](../notes/sequence/PREREG_s5_cliff_anatomy.md),
Addendum 5). The square capacity law still lacks a derivation; it has the
Willshaw form and an empirical constant.

**The other gap items.** Autonomous recurrence (item 1) was a documentation
gap and is closed. The claims pipeline (item 3) was overtaken: the register
is the claim layer now, and [../claims/index.json](../claims/index.json)
holds one formalized claim and six evidence summaries from February 2026,
three of which the Phase 3 audit voided. A biological comparison (item 4)
has not started. Catastrophic forgetting (item 5) got its answer from the
memory line: a Hebbian recurrent area merges its items as it fills, and
refraction at half beta stops the merging, which is the whole
25× (`REFRACTION-ANTI-MERGING`). Falsifiability (item 6) is the
registration discipline itself.

**The foundation-model framing.** [BRIDGE_WEBSCALE_CURRICULUM.md](BRIDGE_WEBSCALE_CURRICULUM.md)
weighed titles of the form "assemblies are all you need". The rule this plan
adopts: no sufficiency title, and no "foundation model" in a title, until a
scaling curve exists against a deep-learning baseline on a corpus where
history is worth something. Today the best sequence result is a local-rule
temporal memory that beats a bigram by 0.148 mean reciprocal rank on a
synthetic chain corpus at twenty seeds
([PREREG_temporal_memory.md](../notes/sequence/PREREG_temporal_memory.md)).
That is a mechanism result. It is not a language model.

**The thesis.** Phase 5 of the soundness program, compositional
generalization against deep learning on real input, is still the
destination. Papers 1 to 4 below are what the substrate has earned on the
way there, and they are worth publishing on their own.

## 2. The ladder

Readiness means: results complete and adopted, figures from committed data,
and a draft could start this week. Each row names what stands between the
registration and a manuscript.

| # | Working title | The claim in one sentence | Register IDs | Readiness |
|---|---------------|---------------------------|--------------|-----------|
| P1 | Refraction makes a recurrent k-WTA area an associative memory, and the learning rate that serves it scales with fan-in | Each at its own best learning rate, a refracted area completes 14 to 57× as many distinct items as a Hebbian one; the best rate is 0.18 to 0.19 of the convergence threshold theta, and capacity there follows the in-degree n p, not n/k; written six to forty times more weakly, the same circuit is a recognition memory holding 15 to 86× as many items. | `REFRACTION-ANTI-MERGING`, `CAP-RATIO`, `CAP-ANCHOR-RATIO`, `CAP-CLIFF`, `REFRACTION-CANCELS-CONVERGENCE` | results complete (Amendments 1 to 24); a draft can start; the in-degree exponent is open |
| P2 | An exact assembly transition machine, and the anatomy of its errors | The refracted-arc machine runs 2000 random steps without error on 40 of 40 brains; its rare soft transitions are ties between a block's least-connected neuron and the best-connected outsider, and training just below the weight clip removes them. | `SEQ-EXACT-RECOVERY`, `SEQ-REGIME-CLIFF`, `ARC-CONJUNCT-EXPOSURE`, `REFRACTION-PROPORTIONAL`, `KWTA-TIE-FRAGILE`, `SEQ-ORGAN-EMBEDS` | results complete; two figures queued |
| P3 | A hash-regenerated GPU substrate for the assembly calculus, and what lazy sampling did to earlier results | Regenerating each brain's connectome from a hash and batching brains gives exact drives at sixty to seventy times the numpy throughput, and the parity gates it required exposed a sampler artifact behind every earlier derailment. | substrate DESIGN notes, [PREREG_sampler_audit.md](../notes/sequence/PREREG_sampler_audit.md) | results complete; timing table queued |
| P4 | Statistical mechanics of the substrate: control parameters, order parameters, measured phase structure | The memory's phase diagram in the write-load plane has four regions (no memory, recognition only, recall, lost) bounded by cliffs; the recall onset is a critical coupling at 0.163 to 0.170 theta for n = 2000 to 16000, opened only by load; capacity, the gain crossover and the regime floor are functions of a few ratios. | `AC-CAP`, `CAP-RATIO`, `CAP-ANCHOR-RATIO`, `CAP-CLIFF`, `SEQ-REGIME-CLIFF`, `HEBB-OUTER-PRODUCT`, `DRIVE-SPLIT`, `REFRACTION-ANTI-MERGING` (Amendments 18, 23, 24), plus [../theory/assembly_statmech.tex](../theory/assembly_statmech.tex) | phase diagram and onset measured; the derivation is a conjecture (a saddle-node of the overlap map) |
| P5 | Local-rule sequence prediction with assemblies (the temporal memory) | The state is the previous arc and predicted neurons win; this carries agreement across two distractors and predicts order-10 sequences exactly inside the training window. | none yet adopted | gated on Amendment 2 and the register study |
| P6 | Cross-situational word learning and word order without annotation | Alignment 0.99 and six of six word orders from scene-sentence pairs with a homeostatic lexicon; word capacity scales with the lexicon's n. | aligner line, [PREREG_word_capacity.md](../notes/aligner/PREREG_word_capacity.md) | measured; needs real input and the dead-probe audit |
| P7 | When plasticity acts decides what a recurrent area stores: attractors, trajectories, and the calculus's sequence operation | One area and one causal Hebbian rule store attractors when the write feeds the item's own rounds under refraction weaker than plasticity, and replayable sequences (207 to 950 chains of eight states) when refraction outpaces plasticity or the write comes after the item; a burst-timing rule stores neither; written one round per element, the calculus's own ordered_recall advances where it never did. | `WRITE-TIMING-DECIDES-ATTRACTOR`, `ADAPTATION-SWITCHES-MEMORY-TYPE`, `ORDERED-RECALL-BY-TRANSITIONS`, `REFRACTION-CANCELS-CONVERGENCE` | three registered results incl. the calculus operation; needs the boundary derivation, distinctness on fresh brains, the length limit |
| P8 | The budget of a sequence area: an interference load law, a hazard compounded over length, and a token code | One area replays a sequence whole up to a critical load rho = L k ln n / (n^2 p) (predicted and confirmed at unseen cells), safe at rho <= 0.09 for any split into sequences once the refraction recovers over the interval between a neuron's uses; past it short sequences fail one by one by a hazard compounded over length, and recurring words are coded as tokens carrying a type trace. | `SEQUENCE-LOAD-LAW`, `RECOVERY-SCALES-WITH-AREA`, `RECOVERY-PEAK-AT-SMALL-AREAS`, `SEQUENCE-BUDGET-ANY-SPLIT`, `RECURRING-WORDS-CODED-AS-TOKENS`, plus the budget section of [../theory/assembly_statmech.tex](../theory/assembly_statmech.tex) | five registered results (Amendments 37 to 42, one with three failed bars); needs Amendment 43 (hazard), the link law, grammar vs random reuse |

### P1. Refracted memory

Evidence: [PREREG_refraction_memory.md](../notes/memory/PREREG_refraction_memory.md),
bars R1 to R7, N1 to N3, Q1 to Q4, G1 to G5, Amendments 1 to 24; twenty
brains per cell; register entry `REFRACTION-ANTI-MERGING`. Figures:
`memory_recall_vs_M.png`, `memory_ceiling_vs_nk.png`, `memory_gate_U.png`;
new figures needed for Amendments 9, 11, 13 and 16 (below).

**Reframed 2026-10-01.** Amendments 9 to 16 change what the paper is about.
The original sentence -- 0.345 to 0.502 (n/k)^2 assemblies, 23 to 38 times
the Hebbian ceiling, n/k alone -- is a rank-1 IDENTIFICATION ceiling at one
fixed, over-strong learning rate. What now stands, each on registered bars:

1. **The multiplier on recall.** With each memory at its own best learning
   rate, the refracted area completes 14 to 57 times as many DISTINCT items
   (recovered >= 0.8 and nearest its own item) as the Hebbian area, at ten
   (n, k, p) cells (Amendment 14, W1).
2. **What the constant is.** Random patterns written into the same circuit
   at the model's own write strength fail near Willshaw's operating point
   (0.69 to 0.92 (n/k)^2, half the synapses potentiated); refraction raises
   the pattern efficiency from about 2% to about 50% (Amendment 9). The
   residual is synapse-pair reuse (a correlate, not shown to be a cause).
3. **Identification and completion want opposite writes** (Amendment 11);
   "n/k alone" belongs to the operating write. Every capacity is quoted with
   its criterion.
4. **The learning-rate law** -- the paper's new result: ln(1 + beta*) =
   0.285 sqrt(2 (1 - p)) / sqrt(k p / 2), confirmed on new brains, at unseen
   cells, and at p = 0.125 to 0.75 (Amendments 13, 16); the field's beta = 0.1
   is 1.3 to 2.7 times too strong. Amendment 17 found it is 0.18 to 0.19 of
   the field's own convergence-threshold form theta = sqrt((1 - p) ln n /
   (p k)) at k p = 30 to 80 (T1), though not below the regime floor (T2): a
   rule on the theory's own scale, with the constant still to derive.
5. **Depth as over-writing.** At each number of write rounds' own best rate
   capacity is the same (Amendment 15); "8 rounds beat 16" was over-writing.
6. **Capacity at the optimum follows the in-degree.** At equal in-degree
   d = n p, cells whose n/k differs two- to fourfold hold capacities within
   11% (Amendment 21, D2); the exponent in d is 1.5 to 1.67, and the
   registered prediction missed one new cell of eight (D1, D3). The onset --
   the weakest write that completes -- is 0.16 to 0.18 theta at ten cells
   (Amendment 18); P4 carries it.
7. **One circuit, two memories.** Each at its own best write, recognition
   holds 15 to 86 times as many items as recall at a write six to forty
   times weaker, and it scales with n/k where recall scales with d
   (Amendment 23; Amendments 19 and 20 failed X1 to X3 and R3 on the way).

Title candidate: *Refraction makes a recurrent k-WTA area an associative
memory, and the learning rate that serves it scales with fan-in.*

What it still needs:

- The in-degree law's exponent. Amendment 21 settled what the capacity at
  the optimum depends on (d, not n/k) but not the exponent (1.669 against
  the bar's [1.35, 1.65]); the paper states the law as measured over its
  cells with the exponent open.
- A theoretical account of the law: the sqrt(fan-in) scaling and the (1 - p)
  factor follow from a signal-against-connectivity-noise argument;
  Amendment 17 puts the constant at 0.18 to 0.20 of theta. The saddle-node
  conjecture in [../theory/assembly_statmech.tex](../theory/assembly_statmech.tex)
  is a candidate account of the onset, not yet of the optimum.
- The failed and voided bars in the text: Q1 out of regime, G3 no U-shape,
  G5 the budget cost, the overturned Amendment 1, PE-3/5/6/S (Amendment 9),
  Amendment 10 void by instrument, Amendment 12's merged-recall failure,
  D2/D3 (Amendment 15), T2 (Amendment 17), O3/O4 (Amendment 18), X1 to X3
  (Amendment 19), R3 (Amendment 20), D1/D3 (Amendment 21), Amendment 22 void
  by a stop rule chance could arm, Q2 (Amendment 23). They locate the regime
  floor, the criteria and the instruments.
- The caveats that bound the claim: graded stimuli; the bias-masked readout
  as a readout-mode primitive
  ([DESIGN_readout_mode.md](../notes/substrate/DESIGN_readout_mode.md)); one
  area and one fiber type; the lexicon's learner follows a different rule
  (word capacity, Amendment 4).
- Related work from the literature synthesis
  ([LITERATURE_SYNTHESIS.md](../notes/LITERATURE_SYNTHESIS.md)): refraction in
  the sequence and speech papers, E%-WTA and dendritic gating as competing
  anti-interference mechanisms, and the convergence thresholds of ITCS 2019
  and COLT 2022.

### P2. The transition machine

Evidence: [DESIGN_sequence_port.md](../notes/sequence/DESIGN_sequence_port.md)
(GATE-1 to GATE-3), [PREREG_s5_cliff_anatomy.md](../notes/sequence/PREREG_s5_cliff_anatomy.md)
(Addenda 3 to 8: 84,000 pairs, 500 organs), [PREREG_sampler_audit.md](../notes/sequence/PREREG_sampler_audit.md).
Dabagia, Papadimitriou and Vempala (2025) prove the construction
(`SEQ-FSM`); this paper measures it at width and names the failure mode.

Still needed:

- The strength-0.1 census rerun and the arc-clip dump, both on the GPU
  queue, for `organ_strength_pinned.png` and the relocation figure.
- A worked Binomial-tail probability for the soft rate, compared with the
  measured 0.039% (95% interval 0.028 to 0.055%) at 15 presentations, and
  the rate's growth with the state area's size.
- The word-problem groups introduced for an ML audience, with the null
  result that the group's Cayley graph does not matter (bar C3).
- The clip window as the paper's design rule: strength pinned at beta,
  presentations between 20 and 28, with the cliff at 30.

### P3. The substrate and the sampler audit

Evidence: [DESIGN_gpu_hashed_drive.md](../notes/substrate/DESIGN_gpu_hashed_drive.md),
[DESIGN_present_only.md](../notes/substrate/DESIGN_present_only.md),
[DESIGN_dense_floor.md](../notes/substrate/DESIGN_dense_floor.md),
[../../docs/gpu_scale_design.md](../../docs/gpu_scale_design.md), the sampler
audit, the parity gates (drive replay to a relative 5e-6, identity across
width), and the throughput figure (numpy 299 s against 4 to 5 s for twenty
brains over 2000 steps; the capacity grid to 16,384 items in five minutes).

This is where the vision paper's instrumentation contribution lives, with
the content the instrumentation actually produced: the gates caught our own
artifacts. The lazily drawn numpy connectome produced a false horizon, a
soft-transition rate seven and a half times too high, every derailment in
the earlier sequence results, and the load window's lower edge. The
standing rules of the soundness program (a perfect score falsifies the
measurement until shown otherwise; construct the true negative; report
distributions) are the method section.

Still needed: a clean timing table for the int8 build (queued), the
parity-error figure, the memory and launch-budget model, and a
reproducibility appendix listing every `--tag` run. The package itself
should go to a software venue (a JOSS-style paper) separately from the
audit, which is a methods result.

### P4. Statistical mechanics

[../theory/assembly_statmech.tex](../theory/assembly_statmech.tex) (dated
2026-07-29) already sets up microstates, order parameters, and control
parameters, and reports the gain crossover g_c = 1.715, 1.915, 2.176 at
n = 2000, 4000, 8000 with a registered failed prediction. It predates the
capacity entries in the register and must absorb them: the ceiling as a
ratio in n/k, the anchor ratio at formation, the cliff past the ceiling, the
regime floor k p ≥ 3 ln n as a cliff, and the tie-fragility of the k-WTA bar.

This is the paper the vision document described, written with laws instead
of a program. Since 2026-10-03 the tex holds measured phase structure of its
own: the memory's phase diagram in the write-load plane (Propositions
`prop:phases` and `prop:loadassist`, from Amendments 18 and 23), the onset
converging at 0.163 to 0.170 theta for n = 2000 to 16000 (Amendment 24), and
two open conjectures (`conj:onset`, `conj:saddle`). It needs the one
derivation the priorities document asked for, and the candidate is now
specific: a mean-field one-round overlap map for a clipped multiplicative
count matrix under k-WTA, whose saddle-node gives beta_c / theta near 0.17.
An exploratory probe found that map sigmoidal, with its unstable point at
the half cue; Amendment 24's settling instrument (exact winner sets) was
tie-fragile and could not test the predicted slowing, so the overlap-
relaxation test is owed. The clip window remains the derived-and-matched
example. Do not start the draft before P1 is written, since P4 quotes P1's
law.

### P5. The temporal memory

Gated. Before a draft: bars TM-7 to TM-10 on fresh seeds and at gap 3
(Amendment 2 of [PREREG_temporal_memory.md](../notes/sequence/PREREG_temporal_memory.md)),
FR-1 to FR-5 of [PREREG_feature_register.md](../notes/sequence/PREREG_feature_register.md),
a corpus with a larger oracle gap than the chain corpus's 0.21
([PREREG_agreement_corpus.md](../notes/sequence/PREREG_agreement_corpus.md)),
a baseline ladder (bigram, trigram, a small recurrent network and a small
transformer at matched step counts), and a scaling curve in n. The
successor-state negative result ([PREREG_successor_state.md](../notes/sequence/PREREG_successor_state.md))
and the literature review belong in the paper as the reason the design is
what it is: every local-rule sequence model in the literature converged on
the same temporal-memory shape, and merging contexts by their future needs
EM or gradients.

### P6. Language

The aligner line is measured and honest: cross-situational learning needs
homeostasis, alignment tracks the referent's drive share, word order comes
out without annotation. It extends Mitropolsky and Papadimitriou's simulated
acquisition, and the extension is only worth a paper on input closer to
child-directed speech. CHILDES transcripts are cite-only and never
committed; their download terms are respected. The ERP line
(the N400 as pre-k-WTA energy, formalized in
[../claims/N400_GLOBAL_ENERGY.md](../claims/N400_GLOBAL_ENERGY.md)) was
later found saturated upstream, and the P600 contrast was area identity;
those enter, if at all, as rank claims. The emergent parser's numbers must
be re-measured against the substrate rather than its Python routes
([../PRIMITIVES_AUDIT.md](../PRIMITIVES_AUDIT.md)) before any of them are
quoted. Not before P1 to P3.

### P7. Attractors or sequences

Evidence: [PREREG_refraction_memory.md](../notes/memory/PREREG_refraction_memory.md)
Amendments 25 and 26 (twenty brains per cell, new brains each), register
entries `WRITE-TIMING-DECIDES-ATTRACTOR` and `ADAPTATION-SWITCHES-MEMORY-TYPE`;
`REFRACTION-CANCELS-CONVERGENCE` for the churn transition near s = 0.8 beta.

What stands, each on registered bars:

1. **When the write happens.** The round write's own counts, written after
   the item's rounds instead of during them, store no attractor at any of 21
   rates over a 32-fold range; the online write stores 451 to 1442 items
   (Amendment 25, W1). Without the write's feedback refraction relocates the
   activity every round, and what is stored is the item's trajectory (W4).
   A burst-timing rule (Butts, Kanold and Shatz 2007) and a burst-gated online
   write store nothing (W2, W3).
2. **How strongly the area adapts.** With the write online, refraction
   weaker than plasticity (s = 0.5 beta) holds the item still and stores
   attractors; refraction stronger (s = 1.5 beta) moves it every round and
   stores sequences that the area replays by itself, all seven steps from
   half of the first state: 207 to 950 sequences of eight states at three
   cells (Amendment 26, S1, S2). No sequence is stored at 0.5 theta, where
   attractors already are: sequences need a stronger write.
3. **The calculus's sequence operation.** The construction of Dabagia,
   Papadimitriou and Vempala presents each element long enough to make it an
   attractor and writes the bridge to the next one briefly, so recall needs a
   veto (long-range inhibition) and is bounded by the bridge-to-self weight
   ratio. That is the attractor phase. `ordered_recall` advances zero steps
   in this package (`test_ordered_recall_advances.py`, strict xfail,
   [PREREG_ordered_recall_reproduction.md](../notes/sequence/PREREG_ordered_recall_reproduction.md));
   an exploratory probe of dwell-and-hop items found the same imbalance --
   states written every round, transitions once -- with free replay covering
   10 to 29% of the stored states. The moving phase is where a sequence is
   native and an attractor impossible.

Still needed:

- Done 2026-10-06: a chosen sequence written one round per element at a
  write near theta recalls 7 of 7 steps on 17 of 20 seeds through the
  calculus's own operation, where eight rounds per element advance 0
  (`ORDERED-RECALL-BY-TRANSITIONS`).
- Done 2026-10-06, the sequence-length limit (Amendment 27): without
  refraction one area replays a single sequence of 0.063 to 0.106 p (n/k)^2
  elements and loses it all at once (L1 passes; per brain the failure is
  all-or-none); refraction lengthens the limit 2 to 31 times at n/k <= 67
  and shortens it at n/k = 133, where the refracted chain breaks at step
  n/k on nearly every brain (post hoc).
- Done 2026-10-06, the tiling deadline (Amendment 28, D1 to D3 pass;
  `SEQUENCE-TILING-DEADLINE`): the refracted sequence tiles the area, replay
  breaks at multiples of n/k (49 of 51 breaks), and a bias that recovers
  replays 4 n/k elements whole on every brain. Next: a bias that DECAYS
  with a time constant (the biological form), and why replay breaks at the
  wrap when the one-step transition there is not weak.
- Done 2026-10-07 (Amendments 29 to 34): recovery sets the length limit
  (`RECOVERY-SETS-SEQUENCE-LENGTH`: 6890 elements at (8000, 60), the best
  limit following the in-degree); context by refraction (A30); balanced
  two-way recall steered by LRI (`BIDIRECTIONAL-RECALL-BY-LRI`); sequences
  of sequences across two areas (`SEQUENCES-OF-SEQUENCES-ACROSS-AREAS`);
  robustness (`SEQUENCE-MEMORY-ROBUSTNESS`). P7 now has a result section per
  capacity. Next: the circuit finding chunk boundaries itself (here they
  are given), noise during writing and on synapses, and the switch
  conjecture at new cells.
- The boundary in s / beta, derived: the identity
  net_{t+1} - net_t = (beta - s) raw_t puts cancellation at s = beta, the
  churn transition was measured near 0.8 beta, and how the boundary moves
  with T and load is open.
- Distinctness and forward-only replay on fresh brains (both exploratory so
  far: replay lands on its own state at 0.95 against 0.10 for the best of
  thousands of others, and runs backward at chance).
- The capacity law: S3 (n/k) and S3d (in-degree) both failed; three cells
  give C ~ d^1.05 (n/k)^0.75 exactly, which is a description, not a law.

Relation to P2 and P5: P2's transition machine is the attractor phase with
the veto engineered in (an arc area and refraction); P5's temporal memory is
a multi-area sequence model. P7 is the single-area mechanism under both.

Title candidate: *When plasticity acts decides what a recurrent area
stores: attractors, trajectories, and the assembly calculus's sequence
operation.*

### P8. The budget of a sequence area

Draft: [../papers/drafts/sequence_budget/](../papers/drafts/sequence_budget/README.md)
(manuscript, figures and intervals built from committed records); the
chronological record is the theory notebook.

Evidence: [PREREG_refraction_memory.md](../notes/memory/PREREG_refraction_memory.md)
Amendments 35 to 42 (twenty new brains each, every cell judged never run
before), register entries `SEQUENCE-LOAD-LAW`, `RECOVERY-SCALES-WITH-AREA`,
`RECOVERY-PEAK-AT-SMALL-AREAS`, `SEQUENCE-BUDGET-ANY-SPLIT` and
`RECURRING-WORDS-CODED-AS-TOKENS`; the theory is the section "The sequence
memory's budget" of [../theory/assembly_statmech.tex](../theory/assembly_statmech.tex),
every statement there marked PROVED, MEASURED or OPEN.

P7 says what a recurrent area stores; P8 says how much, how it fails, and
what a program built from areas must budget for -- the cost model of an
"assembly compiler" (programs of sequences, chunks and plans allocated to
areas with a predicted reliability).

What stands, each on registered bars:

1. **A load law.** One area replays one sequence whole on every brain up to a
   critical interference load rho = L k ln n / (n^2 p) -- a variable derived
   to first order -- and on none a factor 1.07 to 1.15 above it. The critical
   value was predicted from a survey and confirmed at two cells never run
   (0.141 and 0.115 against 0.142; Amendment 37).
2. **Its failure, recorded.** At n/k = 300 with tau = 64 the cliff fell to
   0.081 at three cells, below the law's band and its safe rule (Amendment
   38: three bars failed). The failure located the next variable.
3. **The recovery time is matched to reuse.** A neuron is used once every
   n/k elements; tau = n/k / 2 restores the safe rule at large areas
   (Amendment 39), tau = n/k buys 1.65 times the budget at n/k = 20, 1.24 at
   50, nothing at 100 and costs 9% at 200 (Amendment 41). Under n/k / 2 the
   critical load is nearly constant, 0.122 from n/k = 20 to 100.
4. **Any split is safe.** Up to rho = 0.09 every sequence is whole however
   the budget is split (16, 64 or thousands of elements); past it short
   sequences fail one by one, every brain losing the same share (Amendment 40).
5. **Recurring words are tokens with a type trace.** Two occurrences of a
   word share ~ 0.1 of their neurons, two different words ~ 0.001; up to five
   uses per word and 5% replay noise cost nothing inside the budget; noise
   and load multiply (Amendment 42).

Proved within stated models (the theory section): the cliff's location moves
with the logarithm of what is stored and its width with the hazard's slope
(a large-deviation consequence of a capture-and-hazard model); a type is
recoverable from m tokens with Spearman-Brown reliability, and from one pair
with d' ~ sqrt(ICC k).

Failed or refuted on the way, to be reported: the constant's transfer to
n/k = 300 at fixed tau (Amendment 38); an ln(n/k) rescaling of rho, which
halved the residuals after the fact and failed at new small-n/k cells
(exploratory); the conjecture that every knob acts through the dispersion of
interference, measured before registration (rho_50 D ranges 0.12 to 0.56).

Still needed:

- Amendment 43 (registered 2026-10-08): short sequences' capture and hazard
  predict 256-, 1024- and whole-load sequences' cliffs within 5%.
- The signal side of the tau peak: a successor's structural input from its
  predecessor against tau.
- The hazard over three decades, against the Kramers form
  ln h ~ -(rho_c - rho)^(3/2).
- Recurrence under grammar against shuffled controls (consistency, not
  recurrence, sets the cost: the theory section's conjecture).
- The type trace read out: exposures to a stable type (Spearman-Brown) and
  same-word detection growing as sqrt(k), registered.
- The link budget between areas: set by the source area alone in an
  exploratory sweep (doubling the target area changes it by under 2%,
  doubling the source multiplies it by 1.36 to 1.7); registered at new pairs.
- Universality: the law under threshold E/I dynamics in place of k-WTA.
- A compiled program of language scale run inside its predicted budget.

Relation to P4 and P7: P7 is the mechanism (what one area stores), P8 the
resource law on top of it, and P4 quotes P8's large-deviation and
variance-components results as its statistical-mechanics content for
sequences.

Title candidate: *The budget of a sequence area: an interference load law, a
hazard compounded over length, and a token code with a type trace.*

## 3. Rules for every draft

These come from the repository's own documents and are not optional.

- **Downstream of the register.** A paper cites register IDs. A number
  without a registered bar and at least three seeds (twenty on the hashed
  substrate) does not go in. `ensemble_from_values` refuses fewer.
- **Failed bars go in.** Four of the six predictions registered in the week
  of 2026-09-05 failed, and each failure identified the mechanism that
  replaced it. The paper reports them with their numbers.
- **Distributions, never bare means.** Per-seed values with an interval;
  the engine named; the sampler caveat on any number measured on the
  sampled numpy engine before the port.
- **A perfect score falsifies the measurement** until an invariance sweep
  shows the number can move. Every 1.000 in a draft gets that sweep.
- **Literature claims are cited to their papers.** The package is never the
  proof of a theorem; the Turing-completeness and FSM constructions belong to
  Dabagia, Papadimitriou and Vempala.
- **Writing.** The rules in [../../docs/documentation_style.md](../../docs/documentation_style.md):
  every term introduced before use, every figure with axis labels and a
  how-to-read caption, no sufficiency titles.

## 4. Order and dependencies

P1 first; it is complete and has the largest multiplier. P7 next on the
GPU: its remaining experiments (the chosen-sequence test, distinctness on
fresh brains) are short, and it shares P1's substrate and refraction
section. P2 third; it shares the refraction mechanism and is P7's attractor
phase with an engineered veto. P3 is written alongside them as the methods
companion and carries the audit. P4 after P1 and P7, because it quotes P1's
law and P7's boundary. P5 waits on the GPU queue (Amendment 2, then the
register study), and P6 on real input. P8 follows P7 on the same substrate and
feeds P4; it waits on Amendment 43 and the registered link and grammar
studies listed under it.

The next concrete steps: the P7 chosen-sequence registration (GPU), and a
draft directory `research/papers/drafts/refracted_memory/` with P1's figures
copied from committed data (CPU) -- the second does not contend for the GPU.

## 5. Organizing debts this plan exposes

- `research/claims/index.json` lists six evidence summaries from February
  2026. The Phase 3 audit voided three (reactivation and retrieval were
  dead probes; the 256-word lexicon result is void as a memory result).
  The index should say so, and new claims should go to the register.
- `research/open_questions.md` was last updated 2026-04-22 and still marks
  Q12 and Q20 completed on the voided protocol. It needs a status note and
  a pointer to the register.
- [IMPLICATIONS_AND_PREDICTIONS.md](IMPLICATIONS_AND_PREDICTIONS.md)
  presents the N400 and P600 triple dissociation; the language notes since
  found the N400 saturated and the P600 confounded with area identity. A
  header note now says so.
- [../theory/assembly_statmech.tex](../theory/assembly_statmech.tex) and
  the register describe the same substrate with different vocabularies. P4
  merges them; until then, the register is authoritative.
- `research/papers/drafts/` does not exist yet. It is created with P1.
- Closed: `Result` has an `engine` field, and
  `test_measured_results_state_engine_or_explicit_provenance_gap` requires
  every MEASURED entry to name its engine or state its provenance gap.
- Closed 2026-10-06: `ordered_recall` advances when each element is written
  in one stimulus-and-recurrence round at a write near theta
  (`ORDERED-RECALL-BY-TRANSITIONS`; PREREG_ordered_recall_reproduction.md
  Amendment 2), pinned by a passing test beside the strict xfail, which stays
  as the record of the eight-round construction. The route is the written
  bridge, not inhibition (OR-4 failed), so it is not a reproduction of the
  paper's mechanism.

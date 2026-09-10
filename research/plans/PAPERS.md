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
| P1 | Refraction makes a recurrent k-WTA area an associative memory with a square capacity law | An area refracted at half beta and read with its bias masked stores 0.35 to 0.50 (n/k)² assemblies, 23 to 38× the Hebbian ceiling, because refraction stops items merging while they are written. | `REFRACTION-ANTI-MERGING`, `CAP-RATIO`, `CAP-ANCHOR-RATIO`, `CAP-CLIFF`, `REFRACTION-CANCELS-CONVERGENCE` | results complete |
| P2 | An exact assembly transition machine, and the anatomy of its errors | The refracted-arc machine runs 2000 random steps without error on 40 of 40 brains; its rare soft transitions are ties between a block's least-connected neuron and the best-connected outsider, and training just below the weight clip removes them. | `SEQ-EXACT-RECOVERY`, `SEQ-REGIME-CLIFF`, `ARC-CONJUNCT-EXPOSURE`, `REFRACTION-PROPORTIONAL`, `KWTA-TIE-FRAGILE`, `SEQ-ORGAN-EMBEDS` | results complete; two figures queued |
| P3 | A hash-regenerated GPU substrate for the assembly calculus, and what lazy sampling did to earlier results | Regenerating each brain's connectome from a hash and batching brains gives exact drives at sixty to seventy times the numpy throughput, and the parity gates it required exposed a sampler artifact behind every earlier derailment. | substrate DESIGN notes, [PREREG_sampler_audit.md](../notes/sequence/PREREG_sampler_audit.md) | results complete; timing table queued |
| P4 | Statistical mechanics of the substrate: control parameters, order parameters, measured phase structure | Capacity, the gain crossover, and the regime floor are functions of a few ratios, and each failure is a cliff rather than a slope. | `AC-CAP`, `CAP-RATIO`, `CAP-ANCHOR-RATIO`, `CAP-CLIFF`, `SEQ-REGIME-CLIFF`, `HEBB-OUTER-PRODUCT`, `DRIVE-SPLIT`, plus [../theory/assembly_statmech.tex](../theory/assembly_statmech.tex) | tex draft exists; needs merging with the register and one derivation |
| P5 | Local-rule sequence prediction with assemblies (the temporal memory) | The state is the previous arc and predicted neurons win; this carries agreement across two distractors and predicts order-10 sequences exactly inside the training window. | none yet adopted | gated on Amendment 2 and the register study |
| P6 | Cross-situational word learning and word order without annotation | Alignment 0.99 and six of six word orders from scene-sentence pairs with a homeostatic lexicon; word capacity scales with the lexicon's n. | aligner line, [PREREG_word_capacity.md](../notes/aligner/PREREG_word_capacity.md) | measured; needs real input and the dead-probe audit |

### P1. Refracted memory

Evidence: [PREREG_refraction_memory.md](../notes/memory/PREREG_refraction_memory.md),
bars R1 to R7, N1 to N3, Q1 to Q4, G1 to G5, Amendments 1 to 6; twenty
brains per cell; grids to 16,384 items; seven (n, k) cells. Figures:
`memory_recall_vs_M.png`, `memory_ceiling_vs_nk.png`, `memory_gate_U.png`.

The paper is short. One mechanism, one law, one multiplier, one gate. What
it still needs:

- A capacity argument for the square form. The Willshaw and sparse Hopfield
  literature (Willshaw 1969; Tsodyks and Feigelman 1988; Amit, Gutfreund
  and Sompolinsky) gives (n/k)² scaling for sparse binary memories under
  the right readout; the paper must say which of those assumptions the
  refracted area meets and which it does not, and report the 0.40 constant
  as empirical. The exponent drifting from 2.1 to 1.8 across the grid is a
  finite-size statement and must be reported as such, not as a power law.
- The failed bars in the text: Q1 out of regime, G3 no U-shape, G5 the
  budget cost, and the overturned Amendment 1. They locate the regime
  floor and the gate's mechanism.
- The two caveats that bound the claim: graded stimuli are required (a
  zero-or-size stimulus rotates the refracted item), and the readout must be
  bias-masked, which the paper should present as a readout-mode primitive
  ([DESIGN_readout_mode.md](../notes/substrate/DESIGN_readout_mode.md)).
- A Hebbian control with its own optimum: the control's rounds window is
  attractor dominance, and the paper must show the control at its best
  setting, not at the refracted arm's.

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
of a program. It needs the one derivation the priorities document asked for.
The two candidates with data already in hand are the square capacity law and
the clip window; the clip window is derived and matched, so the capacity law
is the open one. Do not start the draft before P1 is written, since P4
quotes P1's law.

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

P1 first; it is complete and has the largest multiplier. P2 second; it
shares P1's substrate section and the refraction mechanism. P3 is written
alongside them as the methods companion and carries the audit. P4 after P1,
because it quotes P1's law. P5 waits on the GPU queue (Amendment 2, then the
register study), and P6 on real input.

The next concrete step is a draft directory `research/papers/drafts/refracted_memory/`
with P1's figures copied from committed data, started once the two queued
figure reruns for P2 have landed so the GPU is not contended.

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
- The register does not say, per entry, which engine measured it. Of the
  fifteen MEASURED entries only four name the engine in their source or
  evidence text; after the sampler audit that column is load-bearing (a
  number from the sampled numpy engine is void for sequence dynamics until
  re-run materialized or hashed). Owed: an `engine` field on `Result`,
  filled from each entry's registration, before any of P1 to P4 quotes it.

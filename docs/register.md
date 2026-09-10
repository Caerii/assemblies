# Register of results

Rendered from `neural_assemblies/theory.py` by `python -m neural_assemblies.theory --render`; do not edit by hand. Each entry is cited elsewhere by its ID in double brackets. Statuses: PROVED (in the cited source, inside its preconditions), MEASURED (in this repository, in the regime named), EXTENSION (relied on beyond either).

| ID | Status | Engine / substrate | Claim |
|----|--------|--------------------|-------|
| [`SEQ-TIME-IN-WEIGHTS`](#seq-time-in-weights) | PROVED | Not an empirical entry | Sequence/temporal structure is carried by DIRECTED inter-assembly weights, not by an accumulator or a decaying trace. |
| [`SEQ-REGIME`](#seq-regime) | PROVED | Not an empirical entry | The sequence theorems ASSUME that a target neuron receives kp >= 3 ln n synapses FROM THE DRIVING ASSEMBLY: with it, the expected drive separates the intended winners from the rest by a margin the concentration bounds can use. |
| [`SEQ-BETA-WINDOW`](#seq-beta-window) | PROVED | Not an empirical entry | Sequence learning needs beta in a WINDOW: large enough to write a transition in finite presentations, small enough that the assemblies formed on presentation 1 do not move. |
| [`SEQ-FSM`](#seq-fsm) | PROVED | Not an empirical entry | A finite-state machine is simulable by three areas: input, state, and a CONJUNCTION arc that fires for (state, symbol) and projects to the next state. |
| [`SEQ-TRANSDUCER`](#seq-transducer) | PROVED | Not an empirical entry | Prediction/output is an FSM with one more area, fired together with the state update during training -- a transducer. |
| [`SEQ-TM`](#seq-tm) | PROVED | Not an empirical entry | A Turing machine is simulable by an FSM plus three-area tape cycles, about ten areas in total. |
| [`SEQ-REGIME-CLIFF`](#seq-regime-cliff) | MEASURED | numpy_sparse; sampled arc in the original sweep; materialized reruns require their own artifact provenance | Crossing the kp >= 3 ln n floor is a CLIFF, not a slope: below it recovery is almost never exact and the machine fails; above it every seed runs correctly. |
| [`SEQ-EXACT-RECOVERY`](#seq-exact-recovery) | MEASURED | mixed: vendored nemo_numpy reference, numpy_sparse sampled/materialized, hashed ArcFSM and soft-census organs; see per-evidence caveats | The state area is a DISCRETE attractor: k-WTA maps a whole neighbourhood onto exactly one stored assembly in one step. |
| [`SEQ-TEMPORAL-CARRY`](#seq-temporal-carry) | MEASURED | hashed transducer / temporal organ (20 brains per cell) | A transducer whose STATE is its previous arc (state_mode='copy') and whose PREDICTED arc neurons win (the lateral ARC -> ARC fiber's top-k above half its maximum get (1 + g) x drive, g = 1) carries a feature across distractors by local rules alone. |
| [`ARC-CONJUNCT-EXPOSURE`](#arc-conjunct-exposure) | MEASURED | vendored reference/nemo_numpy (explicit NumPy matrices) | A conjunction area collapses onto whichever conjunct is exposed more often, unless an opposing force (refraction) is present. |
| [`REFRACTION-PROPORTIONAL`](#refraction-proportional) | MEASURED | vendored reference/nemo_numpy (explicit NumPy matrices) | Refraction must charge in proportion to the winner's raw drive. |
| [`REFRACTION-NEEDS-LOAD`](#refraction-needs-load) | MEASURED | numpy_sparse, sampled versus explicitly materialized arc; sampled load floor is retracted | A refracted conjunction area has a CEILING in load M*k/n: above ~1.3 its conjunctions do not fit (10/10 correct at load 1.26, 0/10 at 1.80). |
| [`REFRACTION-ANTI-MERGING`](#refraction-anti-merging) | MEASURED | hashed AssemblyMemory; materialized numpy_sparse mirror with summed stimulus parts (not an identical stimulus protocol) | A recurrent k-WTA area refracted at HALF beta and read with the refraction bias MASKED holds ~25x the Hebbian ceiling: at n/k = 67 M* ~ 1600-2200 stored assemblies against 64-89 for the control, x34-38 at n/k = 33, >= x13-16 at n/k = 133 (censored). |
| [`REFRACTION-CANCELS-CONVERGENCE`](#refraction-cancels-convergence) | MEASURED | hashed substrate (HashedArea with AreaFiber/StimulusFiber) | [RE-MEASURED 2026-09-04 with the selector fixed (1b475fc): the churn above ~0.75 beta stands; the intermediate-strength rows were a selector artefact -- at 0.5 beta the recurrent assembly converges, relocates once when the clip binds (~round 40-60, the registered P2 prediction) and holds; at 0.7 beta most brains no longer converge. |
| [`AC-CAP`](#ac-cap) | MEASURED | incompletely recorded: graded-similarity evidence compares explicit, materialized and sampled numpy_sparse; capacity-note run provenance remains unresolved | Assembly capacity is EXTENSIVE: about M_max ~ 1.15 n/k distinct assemblies per area. |
| [`RATE-HETEROGENEITY`](#rate-heterogeneity) | MEASURED | UNRECORDED: evidence contains numbers but no identifiable run or engine | Learning rate is settable PER FIBER and genuinely bites: two fibers into the same area, driven by the same projections, diverge by more than an order of magnitude in weight. |
| [`DUAL-RATE`](#dual-rate) | EXTENSION | Not an empirical entry | Running fast and slow pathways at once is FUNCTIONALLY useful: a high-beta fiber binds in one shot (episodic) while a low-beta fiber accumulates statistics (semantic), and a system with both does something neither does alone. |
| [`SEQ-ORGAN-EMBEDS`](#seq-organ-embeds) | MEASURED | numpy_sparse (original organ-density experiment; sampled-arc provenance limitation) | A sequence organ runs at its own regime INSIDE a brain whose ambient density is far lower, given per-fiber p. |
| [`SEQ-STATE-CODE-EMERGENT`](#seq-state-code-emergent) | EXTENSION | Not an empirical entry | The state alphabet can be INDUCED from data rather than assigned. |
| [`KWTA-TIE-FRAGILE`](#kwta-tie-fragile) | MEASURED | torch/CUDA selector prototype; not a Brain-engine conformance claim | The k-WTA bar is routinely TIED, so anything that perturbs the drive in its last bits -- a change of summation order, of arithmetic, or of tie-break policy -- can change WHICH neurons fire, not merely their order. |
| [`HEBB-OUTER-PRODUCT`](#hebb-outer-product) | PROVED | Not an empirical entry | The Hebbian co-firing count is a SUM OF RANK-1 OUTER PRODUCTS: count = SUM_t x_{t-1} x_t^T, with x_t the 0/1 winner indicator at round t. |
| [`DRIVE-SPLIT`](#drive-split) | PROVED | Not an empirical entry | With a Bernoulli 0/1 base B and G[i,j] = chain(1, count[i,j]), the drive splits as 1_S^T (B (.) G) = 1_S^T B + SUM_{i in S, j} B[i,j] D[i,j] where D = G - 1 is nonzero only on potentiated cells. |
| [`CAP-RATIO`](#cap-ratio) | MEASURED | hashed AssemblyMemory / exact count-then-apply path | The assembly-capacity ceiling M* is a function of n/k ALONE, not of n and k separately. |
| [`CAP-ANCHOR-RATIO`](#cap-anchor-ratio) | MEASURED | hashed AssemblyMemory / capacity-scaling protocol | The capacity ceiling is set at FORMATION by the ratio of the stimulus anchor to the trained recurrent pull. |
| [`CAP-CLIFF`](#cap-cliff) | MEASURED | hashed AssemblyMemory / exact count-then-apply path | Capacity failure is a CLIFF, not a slope: past the ceiling the assemblies shatter rather than degrading gracefully. |

## SEQ-TIME-IN-WEIGHTS

**Status.** PROVED. **Source.** Dabagia, Papadimitriou & Vempala, 'Computation with Sequences in a Model of the Brain' (arXiv:2306.03812), Thm 1.

**Claim.** Sequence/temporal structure is carried by DIRECTED inter-assembly weights, not by an accumulator or a decaying trace.

**Requires.**
- plasticity on the directed fiber
- assemblies stable enough to be re-presented

**Used by.** `neural_assemblies/programs/nemo_fsm.py`

**Caveat.** Read the contrapositive too: a recurrent buffer accumulating context is NOT how this model represents history, which is the architectural account of the CONTEXT-area collapse in #14.

## SEQ-REGIME

**Status.** PROVED. **Source.** Dabagia et al. (arXiv:2306.03812); assumed by every theorem in the paper, and satisfied by its own FSM demo at n=5000, k=70, p=0.4 (kp=28 per conjunct pair vs floor 25.6). Wording corrected 2026-09-09 after an external review noted the earlier 'reliable only when' promoted a sufficient condition to a necessary one.

**Claim.** The sequence theorems ASSUME that a target neuron receives kp >= 3 ln n synapses FROM THE DRIVING ASSEMBLY: with it, the expected drive separates the intended winners from the rest by a margin the concentration bounds can use. It is a SUFFICIENT condition inside the proofs, one hypothesis among several: the theorems also bound the sequence length and the overlap between stored assemblies, take beta inside a window ([[SEQ-BETA-WINDOW]]) and assume a normalization schedule on the weights. That crossing the floor FAILS in practice is this repository's measurement ([[SEQ-REGIME-CLIFF]]), not a theorem: necessity is measured, sufficiency is proved.

**Requires.**
- counted PER AREA, over the sources that co-fire
- k is the SOURCE assembly's size, not the target's

**Evidence.**
- research/experiments/seq_a1_exactness_sweep.py

**Used by.** `neural_assemblies.diagnostics.regime_audit`

**Caveat.** An organ is in-regime only when EVERY area in it is. Several of this repo's null results were recorded an order of magnitude below floor, where failure is predicted regardless of the mechanism under test -- those nulls are not evidence.

## SEQ-BETA-WINDOW

**Status.** PROVED. **Source.** Dabagia et al. (arXiv:2306.03812), sequence-memorization analysis.

**Claim.** Sequence learning needs beta in a WINDOW: large enough to write a transition in finite presentations, small enough that the assemblies formed on presentation 1 do not move.

**Requires.**
- per-fiber beta, so the window can differ across fibers

**Caveat.** Explains the non-monotone recall in #56 (better at 3 repetitions than at 8) as a violation from below rather than as noise.

## SEQ-FSM

**Status.** PROVED. **Source.** Dabagia et al. (arXiv:2306.03812), Thm 4; demo at n=5000, k=70, p=0.4, beta=0.1, 15 presentations.

**Claim.** A finite-state machine is simulable by three areas: input, state, and a CONJUNCTION arc that fires for (state, symbol) and projects to the next state.

**Requires.**
- [[SEQ-REGIME]] in every area
- each transition presented comparably often
- teacher-forced write onto the TARGET state assembly

**Evidence.**
- research/experiments/seq_a1_fsm_parity.py (10/10 seeds)
- neural_assemblies/reference/nemo_numpy/fsm_network.py

**Used by.** `neural_assemblies.programs.nemo_fsm.NemoArcFSM`

## SEQ-TRANSDUCER

**Status.** PROVED. **Source.** Dabagia et al. (arXiv:2306.03812), Remark 5.

**Claim.** Prediction/output is an FSM with one more area, fired together with the state update during training -- a transducer.

**Requires.**
- [[SEQ-FSM]]

**Caveat.** NOT YET BUILT HERE. Our stateless next-token model scoring exactly the bigram optimum is the degenerate one-state case.

## SEQ-TM

**Status.** PROVED. **Source.** Dabagia et al. (arXiv:2306.03812), Thm 7.

**Claim.** A Turing machine is simulable by an FSM plus three-area tape cycles, about ten areas in total.

**Requires.**
- [[SEQ-FSM]]
- unbounded tape areas

**Caveat.** NOT BUILT HERE. [[SEQ-EXACT-RECOVERY]] gives unbounded TIME with fixed memory, which is the control half only -- it does not by itself confer more than finite-automaton power.

## SEQ-REGIME-CLIFF

**Status.** MEASURED. **Source.** This repository.

**Engine / substrate.** numpy_sparse; sampled arc in the original sweep; materialized reruns require their own artifact provenance

**Claim.** Crossing the kp >= 3 ln n floor is a CLIFF, not a slope: below it recovery is almost never exact and the machine fails; above it every seed runs correctly.

**Evidence.**
- research/experiments/seq_a1_exactness_sweep.py: state kp 14 -> 4/100 exact steps and 4/10 trajectories; kp 21 -> 80/100 and 10/10; kp 28 -> 100/100 and 10/10, with the transition at the predicted p = 18.6/70 = 0.266

**Used by.** `neural_assemblies.diagnostics.regime_audit`

**Caveat.** The original sweep used the sampled numpy arc; its sequence-dynamics numbers are void under PREREG_sampler_audit.md until reproduced on a materialized or hashed substrate. Mean overlap read 0.812 at the failing point while exactness was 4/100 -- the mean hides this mechanism entirely.

## SEQ-EXACT-RECOVERY

**Status.** MEASURED. **Source.** This repository.

**Engine / substrate.** mixed: vendored nemo_numpy reference, numpy_sparse sampled/materialized, hashed ArcFSM and soft-census organs; see per-evidence caveats

**Claim.** The state area is a DISCRETE attractor: k-WTA maps a whole neighbourhood onto exactly one stored assembly in one step. Recovery must be EXACT -- 69 of 70 neurons is a failure, not a near-miss -- because the arc amplifies any residual ~8x per step.

**Evidence.**
- research/experiments/seq_a1_arc_transfer.py: d(output loss)/d(input loss) = 8.17 ours, 10.33 reference
- research/experiments/seq_a1_horizon.py: 2000 steps, 5/5 seeds, zero errors, exact recovery every step (p = 0.4); at p = 0.3 the one short horizon (seed 1, step 759) was the SAMPLER's: materialized, that seed runs 2000 steps exact
- research/experiments/seq_a1_horizon_hashed.py (GATE-3 of DESIGN_sequence_port.md): the hashed organ, 20 brains, 2000 digits, p = 0.3 and 0.4: 40/40 brains never err; exact recovery 0.92-1.00 at p = 0.3, 1.000 at 0.4; the numpy machine materialized agrees 5/5
- research/experiments/seq_a1_limit_cycle.py: constant input gives an orbit closing bit-identically, 30/30
- research/experiments/seq_s5_soft_census.py: at 60-120 states, ~0.25-0.6% of transitions are SOFT (correct label, one intruder neuron); first lattice exit equals first soft-pair visit, 40/40 seeds, zero parameters (SAMPLED arc)
- seq_s5_soft_census_hashed.py (explicit substrate, 40 organs): soft pairs persist at 0.067% (4/6000, one intruder each, non-abelian/product groups only), the zero-parameter law 4/4 -- and none of the four deviations derailed in 500 steps, where the sampled arc derailed 12/40 words

**Findings and caveats.**
1. Expansion and quantization are a PAIR: amplification alone is chaos, benign only because a quantizing area follows it; composition steps without a re-quantizing stage drift.
2. THE MAP HAS SOFT SPOTS: a rare transition emits 69/70 of the right block (one intruder), and exactness holds until the word first visits one -- the measured 'horizon' is a first-hitting time, not a decay constant (zero-parameter law, 40/40 numpy seeds and 36/36 hashed organs).
3. THE DERAILMENTS WERE THE SAMPLER'S: on the sampled arc post-visit trajectories derailed into an absorbing off-lattice regime (12/18) and 12/40 words went wrong within 500 steps; on the explicit substrate the soft rate is 7.5x lower and no deviation derailed (Addendum 3 of PREREG_s5_cliff_anatomy.md). Apparent group differences at fixed L were the pair count |G| x |gens| at a flat per-pair rate, not solvability.
4. WHAT A SOFT SPOT IS (Addenda 4-5, 800 organs): the collision of the target block's weakest member with the area's best-connected outsider. The outsider is a Binomial(k, p) upper tail (43-50 of 70 rows at p = 0.4), unrelated to the Cayley graph (intruder 'other' 31/33), identical in rate across three groups of order 60 and rising with n_state; the block side is how much of the test-time arc was potentiated onto the block.
5. PRESENTATIONS HAVE A WINDOW: 8 unformed (95% soft), 15 in the window (0.06%, the two tails just touching), 30 RELOCATED (soft rate doubled, 12% of words derail). Relocation is the CLIP: at strength = beta refraction cancels potentiation exactly, a member's net drive is pinned at its base value, and the cancellation ends when its weight clips -- the best-connected members fall first (99% of lost arc neurons clipped vs 12% of kept); c* = ln(w_max max(1, k p) / base) / ln(1 + beta) ~ 28-30 here.
6. STRENGTH IS PINNED AT BETA (Addendum 6, 300 organs): below it the cross-context bias no longer cancels the state-shared neurons' potentiation and the arc collapses onto the state conjunct (0.05: across-symbol overlap 0.63; 0.08: soft rate 18x); above it a member's net decays and relocates. The transducer at half beta is worse on every count (A3 Amendment 2), so its null is structural.
7. THE FIX IS THE GAIN (Addendum 7): trained just below the clip edge (20 and 24 presentations, ~0.7-0.85 c*) the organ is EXACT -- 0 soft pairs in 24,000 and 0 derailments on 100 S5 organs at each. In-degree normalisation cut the 15-presentation rate 3.7x by reweighting the conjuncts (STATE -> ARC in-degree 3,360 vs the stimulus's 16,000), not through the tail.

## SEQ-TEMPORAL-CARRY

**Status.** MEASURED. **Source.** This repository; PREREG_temporal_memory.md (cells A and B, Amendments 1-2, bars TM-1 to TM-9); the design is the literature's temporal memory (predictive cells win) rebuilt on the refracted-arc transducer.

**Engine / substrate.** hashed transducer / temporal organ (20 brains per cell)

**Claim.** A transducer whose STATE is its previous arc (state_mode='copy') and whose PREDICTED arc neurons win (the lateral ARC -> ARC fiber's top-k above half its maximum get (1 + g) x drive, g = 1) carries a feature across distractors by local rules alone. On the agreement chain corpus it beats a bigram by 0.149 MRR with two distractors between the agreeing words (72% of the oracle gap; twenty fresh seeds replicate +0.148 to the third digit) and by 0.106 with three (60%); the copy state alone carries nothing (-0.002), and reading with the state area empty loses the whole gain (+0.146 full minus blind). MECHANISM AS MEASURED: at a distractor position the arc shares 0.25 of k with a sentence of the same subject number and another distractor, against 0.03 for the other number -- number-specific, distractor-invariant cells; that contrast is already 0.11 at g = 0, so the copy state puts the cells there (a conjunction inherits part of its state conjunct) and predicted-win doubles their share and brings them within the ARC -> OUT readout's reach. ORDER: two twelve-word sequences sharing ten words are continued correctly on 20 of 20 brains by the induced state and by the copy state at 20 presentations (19 of 20 with predicted-win at g = 4), and forgotten past the clip edge c* = ln(20) / ln(1.1) = 31.4 presentations, where the sequence stimuli relocate the arcs.

**Requires.**
- hashed substrate, twenty brains per cell, n = n_arc = 10,000, k = 200, organ_p = 0.2, beta = 0.1, arc strength at beta
- presentations inside the clip window for the order result (20 of c* = 31.4)
- a corpus whose oracle gap is large enough to measure: 0.21 at gap 2, 0.18 at gap 3 (PREREG_agreement_corpus.md)

**Evidence.**
- seq_a3_transducer.py --temporal, gap 2, seeds 42-61: g = 1 +0.148 +/- 0.009, g = 4 +0.140, g = 0 -0.001 (results_temporal_chain_gap2.json)
- fresh seeds 62-81 (_amend2_fresh): +0.149 +/- 0.009, lower bound 0.141 against the 0.085 bar; same-number minus different-number arc overlap 0.220 +/- 0.020 at g = 1 (bar 0.10), 0.111 +/- 0.003 at g = 0 (registered 'within 0.02': FAIL)
- gap 3, seeds 42-61 (_amend2_gap3): +0.106 +/- 0.011, lower bound 0.095 against the 0.070 bar
- seq_tm_high_order.py --presentations 20: sets I-III exact on the induced and copy arms; at 40 presentations set III falls to 1 of 20 on every arm (results in research/results/sequence/)

**Used by.** `neural_assemblies/core/torch_engine/_hashed_transducer.py`

**Caveat.** A mechanism result on a synthetic corpus, not a language model: no baseline beyond bigram and oracle, no scaling curve in n, and the carry's decay with the gap has two points (72%, 60%). The registered mechanism clause said the predicted set CREATES the number-specific cells; it does not (they exist at g = 0), it amplifies them -- the entry states the measured version. The predicted-win rule is a gain on a lateral fiber applied at selection, not a plasticity rule. The successor-state construction (state teacher-forced toward the next h words) carries nothing on the same corpus (PREREG_successor_state.md).

## ARC-CONJUNCT-EXPOSURE

**Status.** MEASURED. **Source.** This repository; the same law as the role-binding gain result.

**Engine / substrate.** vendored reference/nemo_numpy (explicit NumPy matrices)

**Claim.** A conjunction area collapses onto whichever conjunct is exposed more often, unless an opposing force (refraction) is present.

**Requires.**
- BOTH overlap directions measured -- one alone cannot distinguish a conjunction from collapse onto the other conjunct

**Evidence.**
- research/experiments/seq_arc_refraction_reference.py: ablating refraction takes across-symbol overlap 0.000 -> 0.989 and the task 3/3 -> 0/3

**Caveat.** Task #92 measured one direction, read 0.90-0.99, and concluded a conjunctive arc has no operating point. It has one.

## REFRACTION-PROPORTIONAL

**Status.** MEASURED. **Source.** This repository.

**Engine / substrate.** vendored reference/nemo_numpy (explicit NumPy matrices)

**Claim.** Refraction must charge in proportion to the winner's raw drive. A constant increment is not an equivalent parameterization: Hebbian growth multiplies drive while a constant grows linearly, so its operating point MOVES with training duration.

**Evidence.**
- research/experiments/seq_arc_refraction_reference.py: the constant winning at 15 presentations fails at 30, while the proportional rule passes both untouched

**Used by.** `neural_assemblies/core/_homeostasis.py`

## REFRACTION-NEEDS-LOAD

**Status.** MEASURED. **Source.** This repository.

**Engine / substrate.** numpy_sparse, sampled versus explicitly materialized arc; sampled load floor is retracted

**Claim.** A refracted conjunction area has a CEILING in load M*k/n: above ~1.3 its conjunctions do not fit (10/10 correct at load 1.26, 0/10 at 1.80). RE-SCOPED 2026-09-09 (PREREG_sampler_audit.md): the lower edge this entry was named for -- 'below ~0.2 its assemblies never converge' -- was the numpy sampler's; with the arc materialized a 3-conjunction arc is 10/10 correct at every load from 0.04 to 0.60. An arc must be sized so its conjunctions fit; it need not be filled.

**Requires.**
- refraction active -- this is a statement about what refraction needs, not about k-WTA generally

**Evidence.**
- research/experiments/seq_a2_refraction_load.py, SAMPLED arc: a 3-conjunction arc goes 1/10 at load 0.04 to 10/10 at 0.21; a 9-conjunction arc holds 10/10 from 0.32 to 1.26 and collapses to 1/10 at 1.80
- the same script with the arc MATERIALIZED (NEMO_MATERIALIZE=1, 10 seeds): 3-conjunction arc 10/10 at every load 0.04-0.60; 9-conjunction arc 10/10 to 1.26, 0/10 at 1.80
- arc assembly stability across training: 0.286 from presentation 5 to 15 under-loaded, 0.957 from 10 to 15 loaded

**Caveat.** The 'silent failure under load' this entry once described -- assemblies that never stop moving while every diagnostic reads healthy -- was measured on the sampled engine and does not occur materialized; treat it as a property of lazily drawn areas, not of refraction. The ceiling is [[AC-CAP]]'s and is not independent of it. One task, 10 seeds.

## REFRACTION-ANTI-MERGING

**Status.** MEASURED. **Source.** This repository; PREREG_refraction_memory.md (bars R1-R7, N1-N3, Q1-Q4, G1-G6, S8-S9).

**Engine / substrate.** hashed AssemblyMemory; materialized numpy_sparse mirror with summed stimulus parts (not an identical stimulus protocol)

**Claim.** A recurrent k-WTA area refracted at HALF beta and read with the refraction bias MASKED holds ~25x the Hebbian ceiling: at n/k = 67 M* ~ 1600-2200 stored assemblies against 64-89 for the control, x34-38 at n/k = 33, >= x13-16 at n/k = 133 (censored). Both ceilings are functions of n/k ALONE (n/k-matched cells agree within 25%), so refraction multiplies the assembly capacity law [[AC-CAP]] rather than changing its form. The mechanism is ANTI-MERGING, not orthogonalization: the stored assemblies are orthogonal (0.00x chance) only while the area has unvisited neurons; past fill 1.0 they overlap at chance like random subsets, yet remain DISTINCT (1.000) and recoverable from a half cue, where the Hebbian control collapses into hubs (distinct 0.63, overlap 15x chance) long before the area is full. The intrinsic bias vetoes recall (the net readout reads chance): a refracted memory is read through the veto or not at all. Fewer rounds per item raise the ceiling (T = 8 > T = 16) while the items still converge; refraction's benefit requires GRADED stimulus drive. SHAPE OF THE LAW (Amendment 4): in regime (k p >= 3 ln n) the refracted ceiling is ~0.40 (n/k)^2 at n/k = 67 and 133 (6995 at (8000, 60)), doubling exponents 2.1 then 1.8 -- Willshaw-like, not adopted as a power law; the control is ~0.017 (n/k)^2 from n/k = 67 on, so the multiplier is ~23-25x there. Out of regime (k p = 15 < 3 ln n) the cells fall 20-32% below their n/k pairs and do not converge at low load. GATED ROUNDS (Amendment 5): ending an item's rounds at its first repeated winner set under T_max = 8 raises the ceiling by a CONSTANT fraction of n/k: +34% at n/k = 67 (2645 vs 1978) and +24% at 133 (8666 vs 6995), both resolved; at both cells the ceiling sits where items stop converging inside T_max (the fraction converging is a U in load, 0.98 at mid load, 0 at the ceiling); the same gate STARVES the Hebbian control, whose winners settle in ~4 rounds before its memory is written; T_max = 16 under the gate costs the out-of-regime cell 13x. STRENGTH IS A SWITCH, NOT A DIAL (Amendment 6): 0.3, 0.4, 0.5 and 0.6 beta give the same ceiling (1919-1993, one bracket) at n/k = 67 -- the ceiling is the synaptic memory's, refraction only has to prevent merging while it is written; the churn transition is in (0.6, 0.7] beta at T = 8.

**Requires.**
- recurrent k-WTA area, weight clip, norm_init, no column scaling (arm B); refraction strength 0.5 beta; T = 8 rounds per item from an inhibited area; readout = half-cue recall with the bias masked
- GRADED stimulus drive: a zero-or-size stimulus (the engine's into a materialized area, at one Bernoulli draw) makes the refracted item rotate through its tied connected set and the benefit vanishes

**Evidence.**
- seq_capacity_scaling.py, arm B on the organ fiber, 20 brains, M grid to 4096, s = 0.5 beta masked vs control: n/k = 33: 431 / 383 vs 11.3; n/k = 67: 1978 / 1589 / 2230 vs 83 / 64 / 89; n/k = 133: >= 4096 vs 307 / 263
- distinct 1.000 at every refracted ceiling; control 0.63 at M = 128 (n = 4000), pairwise 15x chance
- net readout at n = 4000: M* = 8 (chance at every M)
- T = 16: M* = 203 against T = 8's >= 1024 (n = 4000); T = 4 does not converge at low load and does at high load
- numpy_sparse gate (refraction_memory_numpy.py, materialized, 5 brains, summed Binomial stimuli): refracted >= 512 (censored, rank-1 1.000 throughout, distinct 1.000) vs control 34 -- the claim holds on the engine whose k-WTA has no selector defect
- grid to 16384, 20 brains: (8000, 60) REF 6995 [6144, 8192) = 0.395 (n/k)^2, CTL 307; (4000, 30) REF 4749 [4096, 6144), CTL 263 (out of regime)
- (4000, 60) gated T_max 8: REF 2645 [2560, 2816) vs 1978; CTL gated rank-1 0.32 at M = 8 (no memory formed); (4000, 30) gated T_max 16: 362 vs 4749
- (8000, 60) gated T_max 8: 8666 [8192, 10240) vs 6995 (x1.24)
- strength 0.3 / 0.4 / 0.5 / 0.6 beta at (4000, 60): 1986 / 1919 / 1978 / 1993, all [1536, 2048)

**Caveat.** Found by re-measuring PREREG_refraction_capacity.md after the hashed selector's sign defect (1b475fc) -- its Amendment 1 ('spends the substrate') was that defect. The n/k law was counted as holding at n/k = 133 while both cells were censored; resolved, they disagree by 0.68 and the k = 30 one is out of regime -- k p >= 3 ln n is a precondition, not a footnote. Three ratios do not fix an exponent that is falling (2.1 -> 1.8). The multiplier is at 0.5 beta; 0.7 beta converges only given T = 16 and holds ~185 at n/k = 67, but strength is otherwise a plateau whose lower edge (below 0.3 beta) is unmeasured. The gate is two in-regime cells (n/k = 67, 133); n/k = 33 and out of regime are unmeasured, and the gated ceiling's doubling exponent (1.71) may keep falling.

## REFRACTION-CANCELS-CONVERGENCE

**Status.** MEASURED. **Source.** This repository; PREREG_refraction_capacity.md.

**Engine / substrate.** hashed substrate (HashedArea with AreaFiber/StimulusFiber)

**Claim.** [RE-MEASURED 2026-09-04 with the selector fixed (1b475fc): the churn above ~0.75 beta stands; the intermediate-strength rows were a selector artefact -- at 0.5 beta the recurrent assembly converges, relocates once when the clip binds (~round 40-60, the registered P2 prediction) and holds; at 0.7 beta most brains no longer converge. The transition lies in 0.5-0.7 beta. And BELOW it, with the bias-MASKED readout, a refracted recurrent area at 0.5 beta stored every assembly of the grid (rank-1 1.000 to M = 32, above the bar to 256, pairwise 0.00 x chance) against a Hebbian ceiling of 23.5 -- capacity becomes FILL-limited far above the interference limit; the earlier 'spends the substrate' reading was the defect. Post hoc; a registration is owed. PREREG_refraction_capacity.md.] Refraction at strength s is the anti-Hebbian counterweight on a neuron's own repeated input: raw*(1+beta)^t minus the charged bias leaves net drive growing by (beta - s)*raw per win, so at s = beta it is CONSTANT. A feedforward area needs no convergence force (its input ranking is fixed) and holds; a RECURRENT assembly converges only through rich-get-richer, and above s ~ 0.75 beta it never converges and churns through the whole area -- refraction there is a firing-rate equalizer, and firing-rate homeostasis is incompatible with attractor memory in a recurrent k-WTA area. Below the transition it is the anti-merging force of [[REFRACTION-ANTI-MERGING]]: ~25x the Hebbian ceiling, read with the bias masked.

**Requires.**
- recurrent area; strength quoted relative to beta; T=8 rounds per item in the capacity protocol
- the reference uses RefractedArea only as a FEEDFORWARD conjunction area driven by its full input at recall, where none of this applies

**Evidence.**
- seq_refraction_wander.py at n=4000 k=100 p=0.5 beta=0.1 w_max=20, 16 brains, 240 rounds: s/beta = 0.5, 0.7 converge (rounds 48, 45 vs control 4; late stability 1.000); 0.8, 0.9, 0.95, 1.0 never converge (late stability <= 0.22, fill 1.000); feedforward at s = beta holds (late 0.993)
- the w_max saturation arithmetic ln(w_max)/ln(1+beta) + (1-1/w_max)/beta ~ 41 appears as a transient re-ranking at rounds 44-48 below the transition, which the assembly survives
- capacity protocol at s = 0.5 beta, re-measured with the selector fixed: see [[REFRACTION-ANTI-MERGING]] (the earlier 'M* 19.6 vs 23.5' reading was the defect)
- bias-on partial-cue recall 0.250 vs bias-masked 0.984 at M=8, same training: the intrinsic bias vetoes recall from a partial cue, as the identity predicts

**Used by.** `neural_assemblies/core/_homeostasis.py`; `neural_assemblies/core/torch_engine/_hashed.py`

**Caveat.** The critical ratio is bracketed in (0.7, 0.8) at one operating point; a transient-handicap estimate gives ~2/3. Whether REFRACTION-NEEDS-LOAD's under-loaded non-convergence is this mechanism (the arc's state input is itself changing) is suggested, not established. TWO FURTHER LIMITS (AUDIT_refraction_scaling.md): refraction + synaptic scaling on the same area is INCOMPATIBLE -- a feedforward arc that holds under either alone loses its assemblies within ~10 presentations under both (late stability 0.12 vs 1.00), because scaling moves the raw drive the bias is charged against; and with no clip the identity dies of float32 cancellation at ~100-150 wins. Never scale a refracted area.

## AC-CAP

**Status.** MEASURED. **Source.** This repository (critical-load measurement).

**Engine / substrate.** incompletely recorded: graded-similarity evidence compares explicit, materialized and sampled numpy_sparse; capacity-note run provenance remains unresolved

**Claim.** Assembly capacity is EXTENSIVE: about M_max ~ 1.15 n/k distinct assemblies per area.

**Evidence.**
- research/notes/categories/capacity_is_not_the_constraint_separation_is.md
- research/notes/substrate/graded_similarity_and_sampler_load.md

**Caveat.** The capacity-note run's engine provenance remains unresolved; this entry does not certify the numerical capacity claim. This is why k and p are not interchangeable routes to a regime: raising k to reach kp spends capacity and forces n up with it.

## RATE-HETEROGENEITY

**Status.** MEASURED. **Source.** This repository.

**Engine / substrate.** UNRECORDED: evidence contains numbers but no identifiable run or engine

**Claim.** Learning rate is settable PER FIBER and genuinely bites: two fibers into the same area, driven by the same projections, diverge by more than an order of magnitude in weight.

**Evidence.**
- beta=0.5 vs beta=0.01 into one target over 10 rounds: max weight 20.000 (at the w_max clamp) vs 1.094

**Used by.** `neural_assemblies.core.brain.Brain.update_plasticity`

**Caveat.** The numerical evidence has no identified run or engine; provenance must be recovered before adopting it as a reproduced measurement. Density is NOT yet settable per fiber on the production engine -- see [[SEQ-ORGAN-EMBEDS]]. And a fast fiber saturates against w_max, so 'fast' has a ceiling that 'slow' does not: an A/B across rates is confounded unless w_max is checked.

## DUAL-RATE

**Status.** EXTENSION. **Source.** Proposed here; not proved and not measured.

**Claim.** Running fast and slow pathways at once is FUNCTIONALLY useful: a high-beta fiber binds in one shot (episodic) while a low-beta fiber accumulates statistics (semantic), and a system with both does something neither does alone.

**Requires.**
- [[RATE-HETEROGENEITY]] -- the mechanism exists
- [[SEQ-BETA-WINDOW]] -- each rate must sit inside its own window, and the windows may not overlap

**Caveat.** UNTESTED. That the knob exists is measured; that turning it buys anything is not. Two known tensions to design against: beta trades capacity against depth (low favours capacity, high favours depth), and the afferent count flips the SIGN of beta's effect, so a rate that helps one fiber can hurt another at a different kp. Falsified if a dual-rate organ matches the better of its two single-rate controls.

## SEQ-ORGAN-EMBEDS

**Status.** MEASURED. **Source.** This repository.

**Engine / substrate.** numpy_sparse (original organ-density experiment; sampled-arc provenance limitation)

**Claim.** A sequence organ runs at its own regime INSIDE a brain whose ambient density is far lower, given per-fiber p.

**Requires.**
- per-fiber p (`Brain.add_connectivity`, numpy_exact and numpy_sparse)
- EVERY per-fiber quantity scaled by the fiber's p, not the global one -- see the caveat

**Evidence.**
- research/experiments/seq_a1_local_regime.py: ambient p=0.05 with organ fibers at p=0.4 gives 10/10 correct trajectories, matching the uniform p=0.4 result, while the same organ left at the ambient density gives 0/10

**Caveat.** The original organ-density experiment used the sampled numpy arc; its sequence-dynamics numbers are void under PREREG_sampler_audit.md until reproduced materialized or hashed. The heterogeneous path found a defect that the regime audit could NOT see: the stimulus weight clamp was scaled by the global p while the weights were drawn at the fiber's p, so a dense fiber in a sparse brain saturated at a sparse ceiling. The organ read 1/10 while every area reported comfortably in-regime. Any NEW per-fiber quantity is a candidate for the same class. The pooled candidate draw also remains moment-matched -- exact in the first two moments, an approximation beyond them.

## SEQ-STATE-CODE-EMERGENT

**Status.** EXTENSION. **Source.** Not proved and not measured, here or in the source paper.

**Claim.** The state alphabet can be INDUCED from data rather than assigned.

**Caveat.** THE LOAD-BEARING GAP. Both our FSM and the reference assign state assemblies as disjoint blocks by hand. Every result under [[SEQ-FSM]] and [[SEQ-EXACT-RECOVERY]] is therefore about running a machine over a GIVEN alphabet, not about discovering one. Language needs the latter.

## KWTA-TIE-FRAGILE

**Status.** MEASURED. **Source.** The base drive is a Bernoulli COUNT, hence integer-valued, so exact ties are the common case rather than an edge case. Measured 5-18 columns tied at the bar per brain at n=4000-50000 (research/experiments/gpu_radix_select_prototype.py). An ulp-level change to a summation once moved sixteen cells of an exact table with no direction to it.

**Engine / substrate.** torch/CUDA selector prototype; not a Brain-engine conformance claim

**Claim.** The k-WTA bar is routinely TIED, so anything that perturbs the drive in its last bits -- a change of summation order, of arithmetic, or of tie-break policy -- can change WHICH neurons fire, not merely their order.

**Requires.**
- integer or near-integer drive, i.e. before heavy potentiation spreads the values
- k-WTA selecting at a bar that several columns reach

**Evidence.**
- research/experiments/gpu_radix_select_prototype.py

**Used by.** `neural_assemblies.core.numpy_engine._kwta_prune`

**Caveat.** `_kwta_prune` records the operational rule this implies: making the selector's tie-break CANONICAL is a science-affecting change needing its own registration, not an optimisation to smuggle in. The fused selector folds the index into the sort key so that ties break to the smallest index BY CONSTRUCTION, which is exactly why it is a separate entry point.

## HEBB-OUTER-PRODUCT

**Status.** PROVED. **Source.** Immediate from the update rule: `w[i,j] *= (1+beta)` fires exactly when i is a source winner and j a target winner, so the exponent counts co-firing rounds and nothing else.

**Claim.** The Hebbian co-firing count is a SUM OF RANK-1 OUTER PRODUCTS: count = SUM_t x_{t-1} x_t^T, with x_t the 0/1 winner indicator at round t. Restricted to the rows and columns a window of T rounds touches it is a matrix product R^T C of two thin 0/1 matrices; equivalently count[i,j] = popcount(rm[i] & cm[j]) with rm, cm the per-neuron round bitmasks.

**Requires.**
- plasticity purely MULTIPLICATIVE, so the exponent is the whole state
- the pairing must match the engine: source winners x target winners, i.e. prev x new for a recurrent fiber

**Evidence.**
- research/experiments/gpu_writeback_gemm_prototype.py

**Used by.** `neural_assemblies.core.torch_engine._batched`; `neural_assemblies.core.torch_engine._fused_cuda`

**Caveat.** An event carries 2k numbers, so materialising the k x k cross product writes the same information k/2 times over -- a redundancy created BEFORE any sort is reached, which is why sorting strategies cannot recover it. The two forms answer different questions: the GEMM materialises a block, the popcount evaluates one cell. `batched_project_independent` uses new x new rather than prev x new and is therefore NOT interchangeable with the engine.

## DRIVE-SPLIT

**Status.** PROVED. **Source.** Algebraic identity, given that the base is 0/1: an ABSENT cell stays absent however often it is potentiated, and a present one starts at exactly 1.0, so chain(base, c) = base * tab[c].

**Claim.** With a Bernoulli 0/1 base B and G[i,j] = chain(1, count[i,j]), the drive splits as 1_S^T (B (.) G) = 1_S^T B + SUM_{i in S, j} B[i,j] D[i,j] where D = G - 1 is nonzero only on potentiated cells. The correction is a SPARSE MATVEC over D restricted to |S| = k rows, so its intrinsic cost is the number of stored deviations in those rows and nothing else.

**Requires.**
- base strictly 0/1 -- recruitment OVERRIDES cells, so the effective base is 1 wherever a cell was written
- tab replayed with the engine's own per-step multiply-and-clip, NOT min((1+beta)^c, w_max)

**Evidence.**
- research/experiments/gpu_hashed_deviations_prototype.py

**Used by.** `neural_assemblies.core.torch_engine._fused_cuda`

**Caveat.** A representation that cannot say WHICH cells are nonzero must visit every (row, column) pair: the bitmask form costs O(k n W) with W = ceil(rounds/64) against the intrinsic O(sum_{i in S} nnz_i), a ratio n^2 T / (64 k^2) that is INDEPENDENT of M -- 8894x at n=16000, k=60, T=8. Changing the representation changes the SUMMATION ORDER, and float32 addition is not associative, so ties at the k-WTA bar can flip ([[KWTA-TIE-FRAGILE]]).

## CAP-RATIO

**Status.** MEASURED. **Source.** Held-out test registered in research/notes/memory/PREREG_capacity_nk_law.md (bar CS1) before the data existed. Holding n/k = 66.67 while n varies four-fold gives M* = 68.8 / 73.1 / 67.6 at n = 4000 / 8000 / 16000 on the exact count-then-apply path (engine-parity verified) -- constant to +/-4%, where any law M* = f(n) predicts ~4x. All three cells uncensored (fill 0.69-0.76), and all inside the band registered before the data existed.

**Engine / substrate.** hashed AssemblyMemory / exact count-then-apply path

**Claim.** The assembly-capacity ceiling M* is a function of n/k ALONE, not of n and k separately.

**Requires.**
- in regime, kp >= 3 ln n [[SEQ-REGIME]]
- ceiling read from the CURVE, gated on distinctness
- fill at the ceiling below 0.95, else the tiling limit is what is being measured
- AT FIXED p AND beta: M* moved 1.6x across p in [0.3, 0.7] and ~6x across beta in [0.05, 0.20] at fixed n/k = 40 (PREREG_crosstalk_mechanism.md), so the ratio law holds within an operating point, not across them

**Evidence.**
- research/experiments/seq_capacity_scaling.py

**Caveat.** This result survived a WRONG RETRACTION: an intermediate CSR deviation store applied the potentiation table per count FRAGMENT, and (tab[c0]-1)+(tab[c1]-1) != tab[c0+c1]-1, inflating ceilings ~25% (88.3/92.1/83.2); those numbers were briefly recorded here as the verified ones. Multi-episode ENGINE parity exposed it (rel 2e-3) and the exact path lands back on the first run's values. The EXPONENT remains unestablished: b = 2.19 +/- 0.05, above 2, NO mechanism -- reported, never quoted. A synapse bound M ~ n^2 p / (k ln(n/k)) is REFUTED by CS1: it is not a function of n/k alone -- and so is plain second-order CROSSTALK, refuted by registered test: the crosstalk ratio cancels p and beta, but M* ~ p^-0.6 and ~ beta^-1.3 at fixed n/k. No mechanism is adopted. Operational rule from the wrong retraction: when two implementations disagree, a test they BOTH pass verifies neither -- arbitrate with engine parity on the DRIVE.

## CAP-ANCHOR-RATIO

**Status.** MEASURED. **Source.** This repository; PREREG_formation_interference.md (F2) and PREREG_anchor_ratio.md (A1-A3).

**Engine / substrate.** hashed AssemblyMemory / capacity-scaling protocol

**Claim.** The capacity ceiling is set at FORMATION by the ratio of the stimulus anchor to the trained recurrent pull. Density p and gain beta enter through that ratio, so an excursion in either is undone by a computed change in anchor size.

**Requires.**
- n=4000, k=100, T=8, w_max=20, arm B, 16 brains; one operating-point neighbourhood
- exponents s^1.52 p^-0.6 beta^-1.3 are two- and three-point fits used ONLY to place cells; none is adopted

**Evidence.**
- F2: anchor 100 -> 200 lifts M* 23.5 -> 67.1 (2.86x), while retrieval never sees the stimulus
- A1-A3: beta 0.20 / p 0.7 / p 0.3 excursions with uncompensated M* 8.0 / 17.6 / 28.8 land at 29.4 / 25.0 / 25.3 once the anchor is set from the measured exponents -- a 3.6x spread collapses to 1.18x, all within 25% of 23.5
- F3: past the cliff, erosion is retroactive and diffuse (early items fail worst), not one-shot capture

**Caveat.** A1 sits on the numeric band's edge (29.4 vs < 29.4) and passes on the registered prose criterion; all three compensated cells overshoot upward, so the anchor exponent is probably slightly high. The (n/k)^2 dependence of CAP-RATIO is the pull's chance-overlap term and is NOT derived here.

## CAP-CLIFF

**Status.** MEASURED. **Source.** research/experiments/seq_capacity_scaling.py, on the exact count-then-apply path (engine-parity verified). At n=8000, k=60, 16 brains: M=256 gives rank-1 0.938 at pairwise overlap 1.37x chance with every assembly distinct; M=320 gives 0.486; M=384 gives 0.014 at overlap 4.75x. One doubling (256 -> 512) takes rank-1 from 0.938 to 0.000. Ceiling M* = 284 at fill 0.944, uncensored.

**Engine / substrate.** hashed AssemblyMemory / exact count-then-apply path

**Claim.** Capacity failure is a CLIFF, not a slope: past the ceiling the assemblies shatter rather than degrading gracefully.

**Requires.**
- half-cue rank-1 readout against ALL M stored items
- distinctness gate applied, so a collapsed set scores 0

**Evidence.**
- research/experiments/seq_capacity_scaling.py

**Caveat.** There is no soft capacity margin to trade against: a design must know where the ceiling is and stay under it. Sharing itself is healthy -- at M=256 the load Mk/n is 1.9, nearly two assemblies per neuron, with overlap still 1.37x chance -- so the cliff is not caused by sharing. The transition occupies roughly one 1.5x step in M (256 -> 384).

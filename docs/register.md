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
| [`SEQ-INTEGER-ARC-LOAD`](#seq-integer-arc-load) | MEASURED | hashed_arc_fsm (HashedArcFSM), 20 brains a cell, chain-order training | Autonomous chain recall PEAKS WHEN THE ARC LOAD DIVIDES EVENLY. |
| [`SEQ-STATE-COLLISION-TOLERATED`](#seq-state-collision-tolerated) | MEASURED | hashed_arc_fsm (HashedArcFSM, membership readout), 20 brains a cell | The autonomous chain tolerates state collision ONLY WHERE IT HAS MARGIN, and the margin is doing the work. |
| [`SEQ-REGIME-CLIFF`](#seq-regime-cliff) | MEASURED | numpy_sparse, arc materialized (retained runner replay, 10 seeds); the original sweep ran on the sampled arc | Crossing the kp >= 3 ln n floor is a CLIFF, not a slope: below it recovery is rarely exact (0.22 of steps at kp = 14, arc materialized) and a long run derails at its first non-exact step; above it recovery is exact on every seed (0.98 at kp = 21, 1.00 at kp = 28). |
| [`SEQ-EXACT-RECOVERY`](#seq-exact-recovery) | MEASURED | mixed: vendored nemo_numpy reference, numpy_sparse sampled/materialized, hashed ArcFSM and soft-census organs; see per-evidence caveats | The state area is a DISCRETE attractor: k-WTA maps a whole neighbourhood onto exactly one stored assembly in one step. |
| [`SEQ-TEMPORAL-CARRY`](#seq-temporal-carry) | MEASURED | hashed transducer / temporal organ (20 brains per cell) | A transducer whose STATE is its previous arc (state_mode='copy') and whose PREDICTED arc neurons win (the lateral ARC -> ARC fiber's top-k above half its maximum get (1 + g) x drive, g = 1) improves next-token prediction on the synthetic agreement chain. |
| [`ARC-CONJUNCT-EXPOSURE`](#arc-conjunct-exposure) | MEASURED | reference_nemo_numpy (the vendored explicit-matrix FSM, declared profile; retained runner study of 20 seeds, plus the original 3-seed log) | A conjunction area collapses onto whichever conjunct is exposed more often, unless an opposing force (refraction) is present. |
| [`REFRACTION-PROPORTIONAL`](#refraction-proportional) | MEASURED | reference_nemo_numpy (the vendored explicit-matrix FSM, declared profile; retained runner study of 20 seeds, plus the original 3-seed log) | Refraction must charge in proportion to the winner's raw drive. |
| [`REFRACTION-NEEDS-LOAD`](#refraction-needs-load) | MEASURED | numpy_sparse, arc materialized (retained runner replay, 10 seeds); the sampled-arc sweep's load floor is retracted | A refracted conjunction area has a CEILING in load M*k/n: above ~1.3 its conjunctions do not fit (10/10 correct at load 1.26, 0/10 at 1.80). |
| [`REFRACTION-ANTI-MERGING`](#refraction-anti-merging) | MEASURED | hashed AssemblyMemory; materialized numpy_sparse mirror with summed stimulus parts (not an identical stimulus protocol) | A recurrent k-WTA area refracted at HALF beta and read with the refraction bias MASKED holds ~25x the Hebbian ceiling: at n/k = 67 M* ~ 1600-2200 stored assemblies against 64-89 for the control, x34-38 at n/k = 33, >= x13-16 at n/k = 133 (censored). |
| [`SEQUENCE-MEMORY-ROBUSTNESS`](#sequence-memory-robustness) | MEASURED | hashed AssemblyMemory, store_sequence one round per element, refraction 0.5 beta recovering over 33 rounds, beta = theta | The refracted sequence memory's length limit is a budget of elements, spent alike on one long sequence or many short ones (0.81-1.08 of the single-sequence limit); replay loses nothing with up to 7% of its MOST DRIVEN winners replaced by random neurons at every step, derails at a horizon at 10% that a twice-larger area pushes out about sixfold, and starts from its strongest half cue a quarter wrong. |
| [`SEQUENCE-LOAD-LAW`](#sequence-load-law) | MEASURED | hashed AssemblyMemory, store_sequence one round per element, refraction 0.5 beta recovering over 64 rounds, beta = theta | One area's single-sequence replay fails at a critical interference load: with rho = L k ln n / (n^2 p), every cell with k p >= 3 ln n and n/k <= 150 replays a sequence whole from a random half cue on every brain up to rho ~ 0.11, and on none a factor 1.07-1.15 past its 90% point. |
| [`SEQUENCES-OF-SEQUENCES-ACROSS-AREAS`](#sequences-of-sequences-across-areas) | MEASURED | two hashed AssemblyMemory areas (sequence S, chunk C) and a DenseOrganFiber C -> S; recall on two clocks | Two areas replay sequences of sequences: chunks written once into a sequence area are replayed in whatever order a plan in a chunk area dictates -- every plan whole, every brain, from the plan's first state alone -- with plans that share chunks kept apart by the chunk area and the content shared through the sequence area. |
| [`BIDIRECTIONAL-RECALL-BY-LRI`](#bidirectional-recall-by-lri) | MEASURED | hashed AssemblyMemory, store_sequence with forward_counts/reverse_counts, LRI period 4 at recall with the cue and came-from state primed | One area replays a stored 200-element sequence forward, backward, and either way from its middle, on every brain, when its two directions are written about equally and long-range inhibition at recall vetoes the state just left; without LRI it goes nowhere, and with the directions unequal the stronger one wins whatever LRI does. |
| [`RECOVERY-SETS-SEQUENCE-LENGTH`](#recovery-sets-sequence-length) | MEASURED | hashed AssemblyMemory, store_sequence one round per element, refraction 0.5 beta decaying as exp(-1/tau) per writing round, beta = theta | How long a sequence one area recalls by itself is set by how fast its refraction recovers: the limit rises from the unrefracted area's merging cliff as recovery slows, peaks at an interior recovery time (32 to 64 writing rounds at four cells), and falls back to the tiling deadline -- 6890 elements at (8000, 60, 0.5) against 861 unrefracted and 264 never recovering. |
| [`SEQUENCE-TILING-DEADLINE`](#sequence-tiling-deadline) | MEASURED | hashed AssemblyMemory, store_sequence one round per element, refraction 0.5 beta, beta = theta | A refracted area writes a single sequence onto neurons that have never fired until none remain: the sequence tiles the area, the fresh pool is empty at element n/k, and replay breaks at multiples of n/k (49 of 51 breaks within two steps). |
| [`ORDERED-RECALL-BY-TRANSITIONS`](#ordered-recall-by-transitions) | MEASURED | numpy_sparse, materialized connectome (the assembly calculus's own sequence_memorize and ordered_recall) | The calculus's sequence operation recalls an eight-element sequence autonomously -- 7 of 7 steps after the cue on 17 of 20 seeds, 6 on the rest -- when each element is written in ONE stimulus-and-recurrence round at a write equal to the convergence threshold theta; eight rounds per element at the same write, and one round at beta = 0.10, advance 0 on every seed. |
| [`ADAPTATION-SWITCHES-MEMORY-TYPE`](#adaptation-switches-memory-type) | MEASURED | hashed AssemblyMemory (arm B, ungated, masked readout), write_rule round / deferred, refraction strength s | One recurrent area and one causal Hebbian rule store attractors or sequences according to whether the activity holds still while it is written. |
| [`WRITE-TIMING-DECIDES-ATTRACTOR`](#write-timing-decides-attractor) | MEASURED | hashed AssemblyMemory (refracted, arm B, ungated), write_rule round / online_burst / deferred / burst | In the refracted assembly memory WHEN the write happens decides what is stored. |
| [`REFRACTION-CANCELS-CONVERGENCE`](#refraction-cancels-convergence) | MEASURED | hashed_assembly_memory (HashedArea with AreaFiber/StimulusFiber); the registered twenty-brain run is a runner artifact, the wander diagnostic and the bias-readout numbers are logs | [MEASURED 2026-09-13 on two disjoint blocks of twenty registered brains, PREREG_refraction_convergence.md: a refracted RECURRENT area below the transition does not converge and hold. |
| [`AC-CAP`](#ac-cap) | MEASURED | original capacity-note run: incompletely recorded (graded-similarity evidence compares explicit, materialized and sampled numpy_sparse); retained bracket: hashed_assembly_memory, the unrefracted Hebbian control arm of the paired capacity replay at (n, k) = (4000, 60) | Assembly capacity is EXTENSIVE: about M_max ~ 1.15 n/k distinct assemblies per area. |
| [`RATE-HETEROGENEITY`](#rate-heterogeneity) | MEASURED | numpy_explicit (materialized copied-fiber protocol) | Learning rate is settable PER FIBER: copied fibers with identical pre/post activity at beta 0.06 and 0.005 diverge 14.35-fold after 50 updates without reaching the weight clip. |
| [`DUAL-RATE`](#dual-rate) | EXTENSION | Not an empirical entry | Running fast and slow pathways at once is FUNCTIONALLY useful: a high-beta fiber binds in one shot (episodic) while a low-beta fiber accumulates statistics (semantic), and a system with both does something neither does alone. |
| [`SEQ-ORGAN-EMBEDS`](#seq-organ-embeds) | MEASURED | numpy_sparse, arc materialized (retained runner replay, 10 seeds); the original organ-density experiment ran on the sampled arc | A sequence organ runs at its own regime INSIDE a brain whose ambient density is far lower, given per-fiber p. |
| [`SEQ-STATE-CODE-EMERGENT`](#seq-state-code-emergent) | EXTENSION | Not an empirical entry | The state alphabet can be INDUCED from data rather than assigned. |
| [`KWTA-TIE-FRAGILE`](#kwta-tie-fragile) | MEASURED | tie census: numpy_explicit (retained runner study, 20 seeds, PREREG_kwta_tie_fragility.md); the last-bit split under a change of arithmetic: torch/CUDA selector prototype (no retained artifact) and the fused-kernel gate of 2026-09-10 | The k-WTA bar is routinely TIED, so anything that perturbs the drive in its last bits -- a change of summation order, of arithmetic, or of tie-break policy -- can change WHICH neurons fire, not merely their order. |
| [`HEBB-OUTER-PRODUCT`](#hebb-outer-product) | PROVED | Not an empirical entry | The Hebbian co-firing count is a SUM OF RANK-1 OUTER PRODUCTS: count = SUM_t x_{t-1} x_t^T, with x_t the 0/1 winner indicator at round t. |
| [`DRIVE-SPLIT`](#drive-split) | PROVED | Not an empirical entry | With a Bernoulli 0/1 base B and G[i,j] = chain(1, count[i,j]), the drive splits as 1_S^T (B (.) G) = 1_S^T B + SUM_{i in S, j} B[i,j] D[i,j] where D = G - 1 is nonzero only on potentiated cells. |
| [`CAP-RATIO`](#cap-ratio) | MEASURED | hashed AssemblyMemory / exact count-then-apply path | The assembly-capacity ceiling M* is a function of n/k ALONE, not of n and k separately. |
| [`CAP-ANCHOR-RATIO`](#cap-anchor-ratio) | MEASURED | hashed_assembly_memory, capacity-scaling protocol (retained paired-anchor replay of 20 brains; the original excursions are logs without run records) | The capacity ceiling is set at FORMATION by the ratio of the stimulus anchor to the trained recurrent pull. |
| [`CAP-CLIFF`](#cap-cliff) | MEASURED | hashed AssemblyMemory / exact count-then-apply path | Capacity failure is a CLIFF, not a slope: past the ceiling the assemblies shatter rather than degrading gracefully. |

## SEQ-TIME-IN-WEIGHTS

**Status.** PROVED. **Source.** Dabagia, Papadimitriou & Vempala, 'Computation with Sequences in a Model of the Brain' (arXiv:2306.03812), Thm 1.

**Claim.** Sequence/temporal structure is carried by DIRECTED inter-assembly weights, not by an accumulator or a decaying trace.

**Requires.**
- plasticity on the directed fiber
- assemblies stable enough to be re-presented

**Evidence.**
- demonstrated locally by the autonomous chain (PREREG_autonomous_chain.md, autonomous-chain-20260913): a chain driven by ONE CONSTANT symbol is recalled exactly to 128 assemblies on 20 of 20 brains. The symbol carries no information, so every advance is carried by the directed weights alone; there is no accumulator and no decaying trace anywhere in the construction

**Evidence files.**
- [research/results/runs/sequence.autonomous-chain/autonomous-chain-20260913/results.json](../research/results/runs/sequence.autonomous-chain/autonomous-chain-20260913/results.json) (artifact)
- [research/notes/sequence/PREREG_autonomous_chain.md](../research/notes/sequence/PREREG_autonomous_chain.md) (registration)

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

**Provenance gap.** Nothing in this repository measures the window. The entry is the paper's analysis; the non-monotone recall it explains is an observation in a note rather than a registered measurement, and no bar has been set on where either edge of the window lies.

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

**Evidence.**
- neural_assemblies/programs/sequence_transducer.py: LEX + SEQ_STATE -> SEQ_ARC -> SEQ_STATE with SEQ_ARC -> OUT, the FSM plus one output area fired with the state update, state INDUCED rather than assigned; tested by test_sequence_transducer.py and ported to the hashed substrate with its own parity test
- measured through [[SEQ-TEMPORAL-CARRY]], whose adopted numbers come from this construction

**Used by.** `neural_assemblies/programs/sequence_transducer.py`; `neural_assemblies/core/torch_engine/_hashed_transducer.py`

**Caveat.** Caveat corrected 2026-09-13 by a provenance pass: it read 'NOT YET BUILT HERE', which was stale. The transducer IS built, tested and ported, and [[SEQ-TEMPORAL-CARRY]] is measured on it. What remains true is that our STATELESS next-token model, which scores exactly the bigram optimum, is the degenerate one-state case.

## SEQ-TM

**Status.** PROVED. **Source.** Dabagia et al. (arXiv:2306.03812), Thm 7.

**Claim.** A Turing machine is simulable by an FSM plus three-area tape cycles, about ten areas in total.

**Requires.**
- [[SEQ-FSM]]
- unbounded tape areas

**Evidence.**
- neural_assemblies/programs/tm_demo.py: a MINIMAL unary increment machine over FSMNetwork plus FiberCircuit tape areas, exercised by test_literature_parity.py -- the shape of the construction at its smallest nontrivial size, not the general theorem

**Used by.** `neural_assemblies/programs/tm_demo.py`

**Caveat.** Caveat corrected 2026-09-13 by a provenance pass: it read 'NOT BUILT HERE'. A minimal unary-increment demo IS built and tested; the general roughly-ten-area construction is not. [[SEQ-EXACT-RECOVERY]] gives unbounded TIME with fixed memory, which is the control half only -- it does not by itself confer more than finite-automaton power.

## SEQ-INTEGER-ARC-LOAD

**Status.** MEASURED. **Source.** This repository.

**Engine / substrate.** hashed_arc_fsm (HashedArcFSM), 20 brains a cell, chain-order training

**Claim.** Autonomous chain recall PEAKS WHEN THE ARC LOAD DIVIDES EVENLY. With L transitions, k winners and n_arc arc neurons, the mean number of arcs per neuron is L k / n_arc, and arc assembly stability is a function of that quantity's FRACTIONAL PART alone -- a V minimised at the half: 0.748 at frac 0, 0.420, 0.307, 0.250, 0.215 at 0.5, then 0.263, 0.356, 0.583. At integer load every brain recalls every step; at frac 0.375 and 0.625 recall falls to 0.62 and 0.53 of the chain. So sequence capacity here is not a packing limit. The mechanism is an INTERACTION between the load and the SCHEDULE and neither alone suffices: under a fixed sweep exactly L transitions fall between consecutive presentations of a transition, so every arc neuron accrues exactly L k / n_arc charges in between; at an integer the bias landscape shifts UNIFORMLY, a constant offset leaves the k-WTA ranking alone, and the same winners are re-selected. Measured 2-by-2 at n_arc = 1600: chain+integer 1.000, chain+half 0.991, shuffled+integer 0.030, shuffled+half 0.033. An integer load buys NOTHING under a shuffled schedule.

**Requires.**
- above roughly 19 arcs per neuron; below it the margin absorbs the imbalance and every fractional part works
- A FIXED, REPEATED presentation order. Amendment 10 measured it and the effect does NOT survive shuffling: the outcome advantage vanishes entirely (0.030 against 0.033), though a residual stability gap of 0.078 remains against 0.517 under chain order. The effect is about 85 percent schedule and 15 percent load

**Evidence.**
- chain-iload-20260913 (n_arc=1600, load step 0.125, seeds 62..81): two complete periods repeat cell for cell, all three integer cells exact on all twenty brains, stability within a fractional class agreeing to about 0.005 across chain lengths sixteen apart; IL-1, IL-2, IL-4, IL-5 pass
- IL-3 FAILS and is recorded: it required the outcome below 0.99 at half-integer load and measured 0.991, because its premise that the half is the worst case is false -- the quarter-ish fractions are
- found on diagnostics at n_arc 1000 and 1500 and confirmed out of sample at 1600 on the block those diagnostics did not use

**Evidence files.**
- [research/results/runs/sequence.autonomous-chain/chain-iload-20260913/results.json](../research/results/runs/sequence.autonomous-chain/chain-iload-20260913/results.json) (artifact) â€” 17 cells at load step 0.125, per-arm arc stability and per-brain recall
- [research/results/runs/sequence.autonomous-chain/chain-capacity-20260913/results.json](../research/results/runs/sequence.autonomous-chain/chain-capacity-20260913/results.json) (artifact) â€” the 32-cell grid whose holes this explains
- [research/results/runs/sequence.autonomous-chain/chain-bphase-20260913/results.json](../research/results/runs/sequence.autonomous-chain/chain-bphase-20260913/results.json) (artifact) â€” the 2-by-2 showing the effect needs the SCHEDULE, not the load alone
- [research/experiments/autonomous_chain.py](../research/experiments/autonomous_chain.py) (producer)
- [research/notes/sequence/PREREG_autonomous_chain.md](../research/notes/sequence/PREREG_autonomous_chain.md) (registration)

**Mechanism sensitivity.**
- a chain at INTEGER arc load (L=336, 21.00 arcs per neuron) recalls more ABSOLUTE steps on every brain than a chain ten steps LONGER at fractional load (L=346, 21.625): 336 against 178-279, minimum paired margin 57. The longer chain recalling less is the whole content of the law and is not trivially true: `observations/arms/iload-L336/rows/*/correct` all-greater `observations/arms/iload-L346/rows/*/correct` by at least 50, retained in [research/results/runs/sequence.autonomous-chain/chain-iload-20260913/results.json](../research/results/runs/sequence.autonomous-chain/chain-iload-20260913/results.json) and paired by `observations/arms/iload-L336/rows/*/seed`.

**Sensitivity gap.** The half-integer recovery is now explained by a diagnostic and not by a retained arm: the arc ROTATES by about frac(L k / n_arc) per presentation, so overlap at lag l peaks when l * frac is near a whole turn (r = -0.816 over 35 points). At frac 0.5 that is ALTERNATION -- odd lags 0.216-0.252, even lags 0.525-0.694 -- two assemblies each as well formed as the single one at integer load, ten presentations each. The stability metric was measuring alternation and calling it instability. The proportional prediction was then TESTED AND REFUTED: thresholds are 16, 20, 24, 24 at b = 1, 2, 4, 8 where proportional predicts 16, 32, 64, 128. The currency is the RECRUITMENT UNION -- the neurons a transition ever recruits, 3.40 k at b = 1 from DRIFT alone and only 4.81 k at b = 8 -- which predicts the threshold at r = 0.993 against b's 0.815 and saturates where it does. What sets the drift envelope is unmeasured.

**Caveat.** BEWARE ALIASING when sampling. Step 8 in L samples the load axis at 0.800 (n_arc=1000) and 0.533 (n_arc=1500), and a period-1.0 signal aliases there to apparent periods of 40 and 17 in L -- both of which were measured and reported before the aliasing was noticed. Sample the LOAD axis, not L. A corollary for the literature: any chain-length sweep is walking through this oscillation, so a capacity number depends on which lengths were tried. That is a reason to re-examine reported sequence limits, not a demonstration that any of them is wrong. And because the effect needs a rigid repeated order, it is substantially a statement about how these experiments PRESENT data rather than about the substrate alone: no system with irregular input has a fixed sweep, so the biological reading is weak.

## SEQ-STATE-COLLISION-TOLERATED

**Status.** MEASURED. **Source.** This repository.

**Engine / substrate.** hashed_arc_fsm (HashedArcFSM, membership readout), 20 brains a cell

**Claim.** The autonomous chain tolerates state collision ONLY WHERE IT HAS MARGIN, and the margin is doing the work. At a cell with room (L = 160, n_arc = 3000, 20/20 exact) all six arms stay 20/20 exact across load (L+1)k/n_state from 0.25 to 4.03, where at the tightest area a disjoint code is impossible. At a MARGINAL cell (L = 256, n_arc = 2000, 14/20 exact) the same sweep is devastating: mean consecutive-correct falls 251.8, 220.8, 180.8, 142.0, 110.0 as load runs 0.40 to 6.42, losing 142 of 256 steps. Measured state overlap tracks the k/n arithmetic exactly in all three runs. LOAD IS NOT THE VARIABLE: at matched load 6.4 the roomy cell is at 1.000 of L and the marginal cell at 0.430, and the roomy cell holds at 1.000 to load 13.42 -- more than twice the killing load, 161 states in 1200 neurons at pairwise overlap 0.083. So state collision costs nothing until the chain is already marginal and then costs enormously: an AMPLIFIER of an existing limit, not a limit of its own. AND THE LIMIT IT AMPLIFIES IS THE ARC: holding the chain and the state crowding fixed at the combination that collapsed to 0.430 of L and giving the arc 1.5x the neurons restores 0.997, and 2x restores every brain, while a disjoint control sits at 0.998 to 1.000 at both arc sizes. State collision is not a failure mode of the state area; it is a way of SPENDING ARC CAPACITY, free until the arc has none left -- which is also why crowding the states moved the arc overlap in every run. Separately and much smaller, a random code costs 3.7 steps of 256 against the contiguous block code even when roomy, which is unexplained.

**Evidence.**
- --states at the roomy cell: six arms all 20/20 exact; SC-3 FAILED, which the registration named in advance as the stronger outcome
- --margin at the marginal cell: mean correct 255.5 (blocks), 251.8, 220.8, 180.8, 142.0, 110.0 (random, load 0.40 to 6.42) -- a monotone dose-response
- --load drives the ROOMY cell to 13.42 and it stays at 1.000 of L throughout (19/20 exact at the tightest, one brain losing one step); LM-4 FAILED because that cell has no breaking point in the sweep, and the registration's pre-declared reading of that bar combination is WITHDRAWN in the note, with the reasoning set out
- the marginal run's BARS cannot carry this: all five read exact/20, every random arm is 0/20 roomy and crowded alike, so MC-3 passed vacuously (0 <= 0) and MC-2 passed by comparing the tightest random arm against BLOCKS, mixing crowding with the randomness cost. The dose-response is on mean correct, which no bar reads
- arc overlap FALLS as the state area shrinks in both runs (0.0921 to 0.0447 roomy; 0.1493 to 0.0625 marginal): crowding the states made the arcs more distinct, which is unexplained and which breaks the isolation bar

**Evidence files.**
- [research/results/runs/sequence.autonomous-chain/chain-states-20260913/results.json](../research/results/runs/sequence.autonomous-chain/chain-states-20260913/results.json) (artifact) â€” roomy cell: six arms, per-brain correct/arc_overlap, per-arm state overlap and load
- [research/results/runs/sequence.autonomous-chain/chain-margin-20260913/results.json](../research/results/runs/sequence.autonomous-chain/chain-margin-20260913/results.json) (artifact) â€” marginal cell: the same six arms, where the dose-response appears
- [research/results/runs/sequence.autonomous-chain/chain-load-20260913/results.json](../research/results/runs/sequence.autonomous-chain/chain-load-20260913/results.json) (artifact) â€” the roomy cell driven to load 13.42, which separates margin from load
- [research/results/runs/sequence.autonomous-chain/chain-arcb-20260913/results.json](../research/results/runs/sequence.autonomous-chain/chain-arcb-20260913/results.json) (artifact) â€” arc size swept at fixed chain and fixed state crowding: the rescue
- [research/experiments/autonomous_chain.py](../research/experiments/autonomous_chain.py) (producer)
- [research/notes/sequence/PREREG_autonomous_chain.md](../research/notes/sequence/PREREG_autonomous_chain.md) (registration)

**Mechanism sensitivity.**
- the collidable code demonstrably reaches the organ: arc overlap falls on EVERY brain (0.034 to 0.068) between the disjoint code and the 4000-neuron random code, while consecutive-correct stays at 160 on every brain in both -- the knob turns and the outcome does not, which is what makes this null a measurement rather than a saturation: `observations/arms/random-n4000/rows/*/arc_overlap` all-less `observations/arms/blocks-n64000/rows/*/arc_overlap` by at least 0.03, retained in [research/results/runs/sequence.autonomous-chain/chain-states-20260913/results.json](../research/results/runs/sequence.autonomous-chain/chain-states-20260913/results.json) and paired by `observations/arms/blocks-n64000/rows/*/seed`.

**Sensitivity gap.** What remains unexplained is narrower than the headline: why a ROOMY random code still costs 3.7 steps of 256 against a contiguous block code at pairwise overlap 0.0016, and whether CORRELATED collision behaves like the uncorrelated kind measured here -- projection makes assembly overlap track input similarity and random k-subsets do not.

**Caveat.** Random k-subsets are UNCORRELATED. Projection forms assemblies whose overlap tracks input similarity, which is structured interference neither sweep produces, and the papers' states are projection-formed. This measures collision per se, not correlated collision. The id is kept for citation stability but reads too strongly on its own: the tolerance is conditional on margin, and what the crowding cost is ATTRIBUTABLE to is open.

## SEQ-REGIME-CLIFF

**Status.** MEASURED. **Source.** This repository.

**Engine / substrate.** numpy_sparse, arc materialized (retained runner replay, 10 seeds); the original sweep ran on the sampled arc

**Claim.** Crossing the kp >= 3 ln n floor is a CLIFF, not a slope: below it recovery is rarely exact (0.22 of steps at kp = 14, arc materialized) and a long run derails at its first non-exact step; above it recovery is exact on every seed (0.98 at kp = 21, 1.00 at kp = 28). The cliff is in EXACTNESS: on the short decision strings the trajectory is still correct below the floor once the arc is materialized (10/10 at kp = 14); the sampled arc had made it a cliff in decisions as well (4/10).

**Evidence.**
- research/experiments/seq_a1_exactness_sweep.py: state kp 14 -> 4/100 exact steps and 4/10 trajectories; kp 21 -> 80/100 and 10/10; kp 28 -> 100/100 and 10/10, with the transition at the predicted p = 18.6/70 = 0.266

**Evidence files.**
- [research/results/sequence/seq_a1_exactness_sweep_results.json](../research/results/sequence/seq_a1_exactness_sweep_results.json) (artifact) â€” sampled numpy arc; sequence verdict void under sampler audit
- [research/results/sequence/seq_a1_exactness_sweep_results_materialized.json](../research/results/sequence/seq_a1_exactness_sweep_results_materialized.json) (artifact)
- [research/results/runs/sequence.a1-exactness-sweep/sens-exactness-sweep-v3-20260912/results.json](../research/results/runs/sequence.a1-exactness-sweep/sens-exactness-sweep-v3-20260912/results.json) (artifact) â€” shared-runner replay with source archive and per-seed vectors per density
- [research/experiments/seq_a1_exactness_sweep.py](../research/experiments/seq_a1_exactness_sweep.py) (producer)

**Provenance gap.** the two legacy result files have no runner/source record; the 2026-09-12 replay has both

**Mechanism sensitivity.**
- exact recovery steps of 10 per seed at kp = 28 (above the 3 ln n floor of 18.6) against kp = 14 (below it), arc materialized: `observations/by_p/0.4/exact_steps` all-greater `observations/by_p/0.2/exact_steps` by at least 4, retained in [research/results/runs/sequence.a1-exactness-sweep/sens-exactness-sweep-v3-20260912/results.json](../research/results/runs/sequence.a1-exactness-sweep/sens-exactness-sweep-v3-20260912/results.json) and paired by `observations/by_p/0.4/seeds`.

**Sensitivity gap.** The retained contrast crosses the floor by changing p; no retained arm disables plasticity at fixed p, so the cliff's dependence on learning as such rests on the learning-disabled null of [[SEQ-EXACT-RECOVERY]].

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

**Evidence files.**
- [research/results/sequence/seq_a1_horizon_results_hashed_int8_timing.json](../research/results/sequence/seq_a1_horizon_results_hashed_int8_timing.json) (artifact)
- [research/results/sequence/seq_a1_horizon_materialized_check.json](../research/results/sequence/seq_a1_horizon_materialized_check.json) (artifact)
- [research/results/sequence/seq_s5_soft_census_results.json](../research/results/sequence/seq_s5_soft_census_results.json) (artifact) â€” sampled numpy arc
- [research/results/sequence/seq_s5_soft_census_results_hashed.json](../research/results/sequence/seq_s5_soft_census_results_hashed.json) (artifact)
- [research/notes/sequence/PREREG_s5_cliff_anatomy.md](../research/notes/sequence/PREREG_s5_cliff_anatomy.md) (registration)
- [research/results/runs/sequence.a1-learning-null/a1-learning-null-20260910/results.json](../research/results/runs/sequence.a1-learning-null/a1-learning-null-20260910/results.json) (artifact) â€” paired learning-disabled null for the A1 exact-recovery subclaim; hashed organ, seeds 1-20

**Provenance gap.** some legacy subclaims have no source archive or raw runner artifact

**Mechanism sensitivity.**
- exact state recovery with arc->state learning (beta and refraction) disabled, hashed organ at p = 0.3: `observations/cells/0/ensembles/exact_fraction/trained/values` all-greater `observations/cells/0/ensembles/exact_fraction/null/values` by at least 0.5, retained in [research/results/runs/sequence.a1-learning-null/a1-learning-null-20260910/results.json](../research/results/runs/sequence.a1-learning-null/a1-learning-null-20260910/results.json) and paired by `observations/cells/0/ensembles/exact_fraction/trained/keys`.
- exact state recovery with arc->state learning (beta and refraction) disabled, hashed organ at p = 0.4: `observations/cells/1/ensembles/exact_fraction/trained/values` all-greater `observations/cells/1/ensembles/exact_fraction/null/values` by at least 0.5, retained in [research/results/runs/sequence.a1-learning-null/a1-learning-null-20260910/results.json](../research/results/runs/sequence.a1-learning-null/a1-learning-null-20260910/results.json) and paired by `observations/cells/1/ensembles/exact_fraction/trained/keys`.

**Sensitivity gap.** The retained learning-disabled null covers the A1 exact-recovery subclaim on the hashed organ (both densities, 20 seeds each). The soft-census subclaims (rate, first-hitting law, presentation window) still have no retained mechanism-disabled null artifact.

**Findings and caveats.**
1. Expansion and quantization are a PAIR: amplification alone is chaos, benign only because a quantizing area follows it; composition steps without a re-quantizing stage drift.
2. THE MAP HAS SOFT SPOTS: a rare transition emits 69/70 of the right block (one intruder), and exactness holds until the word first visits one -- the measured 'horizon' is a first-hitting time, not a decay constant (zero-parameter law, 40/40 numpy seeds and 36/36 hashed organs).
3. THE DERAILMENTS WERE THE SAMPLER'S: on the sampled arc post-visit trajectories derailed into an absorbing off-lattice regime (12/18) and 12/40 words went wrong within 500 steps; on the explicit substrate the soft rate is 7.5x lower and no deviation derailed (Addendum 3 of PREREG_s5_cliff_anatomy.md). Apparent group differences at fixed L were the pair count |G| x |gens| at a flat per-pair rate, not solvability.
4. WHAT A SOFT SPOT IS (Addenda 4-5, 800 organs): the collision of the target block's weakest member with the area's best-connected outsider. The outsider is a Binomial(k, p) upper tail (43-50 of 70 rows at p = 0.4), unrelated to the Cayley graph (intruder 'other' 31/33), identical in rate across three groups of order 60 and rising with n_state; the block side is how much of the test-time arc was potentiated onto the block.
5. PRESENTATIONS HAVE A WINDOW: 8 unformed (95% soft), 15 in the window (0.06%, the two tails just touching), 30 RELOCATED (soft rate doubled, 12% of words derail). Relocation is the CLIP: at strength = beta refraction cancels potentiation exactly, a member's net drive is pinned at its base value, and the cancellation ends when its weight clips -- the best-connected members fall first (99% of lost arc neurons clipped vs 12% of kept); c* = ln(w_max max(1, k p) / base) / ln(1 + beta) ~ 28-30 here.
6. STRENGTH IS PINNED AT BETA (Addendum 6, 300 organs): below it the cross-context bias no longer cancels the state-shared neurons' potentiation and the arc collapses onto the state conjunct (0.05: across-symbol overlap 0.63; 0.08: soft rate 18x); above it a member's net decays and relocates. The transducer at half beta is worse on every count (A3 Amendment 2), so its null is structural.
7. THE FIX IS THE GAIN (Addendum 7): trained just below the clip edge (20 and 24 presentations, ~0.7-0.85 c*) the organ is EXACT -- 0 soft pairs in 24,000 and 0 derailments on 100 S5 organs at each. In-degree normalisation cut the 15-presentation rate 3.7x by reweighting the conjuncts (STATE -> ARC in-degree 3,360 vs the stimulus's 16,000), not through the tail.

## SEQ-TEMPORAL-CARRY

**Status.** MEASURED. **Source.** This repository; PREREG_temporal_memory.md (cells A and B, Amendments 1-2, bars TM-1 to TM-9); PREREG_temporal_positions.md (bars TP-1 to TP-4); the design is the literature's temporal memory (predictive cells win) rebuilt on the refracted-arc transducer.

**Engine / substrate.** hashed transducer / temporal organ (20 brains per cell)

**Claim.** A transducer whose STATE is its previous arc (state_mode='copy') and whose PREDICTED arc neurons win (the lateral ARC -> ARC fiber's top-k above half its maximum get (1 + g) x drive, g = 1) improves next-token prediction on the synthetic agreement chain. On the agreement chain corpus it beats a bigram by 0.149 MRR with two distractors between the agreeing words (72% of the oracle gap; twenty fresh seeds replicate +0.148 to the third digit) and by 0.106 with three (60%); the copy-state native readout gains nothing (-0.002), and reading with the state area empty loses the whole gain (+0.146 full minus blind). POSITION-SPECIFIC MECHANISM: on twenty further fresh seeds, mean subject-number contrast over the six distractor positions is 0.0258 at g = 0, 0.1849 at g = 1, and 0.0022 with state blinded. The paired g = 1 minus g = 0 amplification is 0.1591 [0.1400, 0.1781]. At g = 0 the contrast is present after the first distractor (0.0479) but absent after the second (0.0037); at g = 1 it remains 0.2032 then 0.1666. Thus predicted-win amplifies and preserves the state-dependent representation through the second distractor. DECAY LAW (gaps 3-6, twenty brains each, PREREG_temporal_positions.md Amendment 1): the carry falls geometrically with distractor position, D_j = A r^(j-1) with a POOLED r = 0.709 [0.689, 0.729] whose per-gap fits (0.78, 0.74, 0.69, 0.73) all contain it; the SHAPE is one constant across gaps and the AMPLITUDE is not (A = 0.207, 0.213, 0.161, 0.124 as the chain lengthens), which is the whole of the pooled fit's residual. HORIZON: the carry clears the registered 0.02 bar at the last distractor through gap 5 (D_5 = 0.042 [0.034, 0.049]) and not at gap 6 (D_6 = 0.024 [0.016, 0.032], positive on 20 of 20 brains but above the bar on 10) -- five distractors, measured, not extrapolated. The plain conjunction decays at r = 0.008: gone after one position at every gap. ORDER: two twelve-word sequences sharing ten words are continued correctly on 20 of 20 brains by the induced state and by the copy state at 20 presentations (19 of 20 with predicted-win at g = 4), and forgotten past the clip edge c* = ln(20) / ln(1.1) = 31.4 presentations, where the sequence stimuli relocate the arcs.

**Requires.**
- hashed substrate, twenty brains per cell, n = n_arc = 10,000, k = 200, organ_p = 0.2, beta = 0.1, arc strength at beta
- presentations inside the clip window for the order result (20 of c* = 31.4)
- a corpus whose oracle gap is large enough to measure: 0.21 at gap 2, 0.18 at gap 3 (PREREG_agreement_corpus.md)

**Evidence.**
- seq_a3_transducer.py --temporal, gap 2, seeds 42-61: g = 1 +0.148 +/- 0.009, g = 4 +0.140, g = 0 -0.001 (results_temporal_chain_gap2.json)
- fresh seeds 62-81 (_amend2_fresh): +0.149 +/- 0.009, lower bound 0.141 against the 0.085 bar; same-number minus different-number arc overlap 0.220 +/- 0.020 at g = 1 (bar 0.10), 0.111 +/- 0.003 at g = 0; pooled-position values VOID for TM-9 (see research/notes/sequence/AUDIT_temporal_position_pooling.md)
- sequence.temporal-positions/temporal-positions-study-20260910, seeds 82-101: distractor contrast g = 0 0.0258 [0.0226, 0.0290], g = 1 0.1849 [0.1660, 0.2038], blind 0.0022 [-0.0003, 0.0046]; paired amplification 0.1591 [0.1400, 0.1781]; all TP-1 to TP-4 bars pass
- gap 3, seeds 42-61 (_amend2_gap3): +0.106 +/- 0.011, lower bound 0.095 against the 0.070 bar
- gaps 3-6 under the position instrument (protocol version 2, seeds 82-101 each): D_j on the g = 1 arm 0.207/0.160/0.127 at gap 3, 0.215/0.151/0.117/0.087 at gap 4, 0.168/0.098/0.076/0.058/0.042 at gap 5, 0.128/0.083/0.065/0.052/0.037/0.024 at gap 6; DL-1, DL-2, DL-4 and DL-5 pass at every gap, DL-3 fails at gap 6 only, and TP-3 fails from gap 3 on because its estimand is a mean over all positions and the plain conjunction carries at one
- seq_tm_high_order.py --presentations 20: sets I-III exact on the induced and copy arms; at 40 presentations set III falls to 1 of 20 on every arm (results in research/results/sequence/)

**Evidence files.**
- [research/results/runs/sequence.temporal-positions/temporal-positions-study-20260910/results.json](../research/results/runs/sequence.temporal-positions/temporal-positions-study-20260910/results.json) (artifact)
- [research/results/runs/sequence.temporal-positions/temporal-positions-gap2-20260912/results.json](../research/results/runs/sequence.temporal-positions/temporal-positions-gap2-20260912/results.json) (artifact)
- [research/results/runs/sequence.temporal-positions/temporal-positions-gap3-20260913/results.json](../research/results/runs/sequence.temporal-positions/temporal-positions-gap3-20260913/results.json) (artifact)
- [research/results/runs/sequence.temporal-positions/temporal-positions-gap4-20260913/results.json](../research/results/runs/sequence.temporal-positions/temporal-positions-gap4-20260913/results.json) (artifact)
- [research/results/runs/sequence.temporal-positions/temporal-positions-gap5-20260913/results.json](../research/results/runs/sequence.temporal-positions/temporal-positions-gap5-20260913/results.json) (artifact)
- [research/results/runs/sequence.temporal-positions/temporal-positions-gap6-20260913/results.json](../research/results/runs/sequence.temporal-positions/temporal-positions-gap6-20260913/results.json) (artifact)
- [research/results/sequence/seq_a3_transducer_results_temporal_chain_gap2.json](../research/results/sequence/seq_a3_transducer_results_temporal_chain_gap2.json) (artifact)
- [research/results/sequence/seq_a3_transducer_results_temporal_chain_gap2_amend2_fresh.json](../research/results/sequence/seq_a3_transducer_results_temporal_chain_gap2_amend2_fresh.json) (artifact)
- [research/results/sequence/seq_a3_transducer_results_temporal_chain_gap3_amend2_gap3.json](../research/results/sequence/seq_a3_transducer_results_temporal_chain_gap3_amend2_gap3.json) (artifact)
- [research/results/sequence/seq_tm_high_order_results_p20.json](../research/results/sequence/seq_tm_high_order_results_p20.json) (artifact)
- [research/notes/sequence/PREREG_temporal_positions.md](../research/notes/sequence/PREREG_temporal_positions.md) (registration)

**Provenance gap.** pre-schema-4 prediction/order artifacts lack complete source and environment capture

**Mechanism sensitivity.**
- state-dependent distractor carry versus state-blind g=1: `observations/summaries/g1/D/values` all-greater `observations/summaries/blind_g1/D/values` by at least 0.05, retained in [research/results/runs/sequence.temporal-positions/temporal-positions-study-20260910/results.json](../research/results/runs/sequence.temporal-positions/temporal-positions-study-20260910/results.json) and paired by `run/seeds`.
- the decay itself on the longest chain: at gap 6 the carry at the first distractor exceeds the carry at the sixth on every one of twenty brains (minimum 0.0687), so the positional decay is a per-brain fact and not an ensemble average: `observations/summaries/g1/D1/values` all-greater `observations/summaries/g1/D6/values` by at least 0.05, retained in [research/results/runs/sequence.temporal-positions/temporal-positions-gap6-20260913/results.json](../research/results/runs/sequence.temporal-positions/temporal-positions-gap6-20260913/results.json) and paired by `run/seeds`.

**Used by.** `neural_assemblies/core/torch_engine/_hashed_transducer.py`

**Caveat.** A prediction result on a synthetic corpus, not a language model: no baseline beyond bigram and oracle, no scaling curve in n. The decay law rests on offset MEANS over twenty brains at four gaps: at gaps 5 and 6 the last offset's carry does not exceed the state-blind control on every brain (minimum paired difference -0.002 and -0.004), so the tail is an ensemble statement with an interval, never a per-brain one. The falling amplitude is unexplained: a larger gap both lengthens the sentence and repeats the same noun-to-noun transitions more often, and nothing here separates the two. The historical TM-9 pooled-position values remain void; the separately preregistered position-specific run replaces their mechanism interpretation. Its g=0 curve reaches only the first distractor, while g=1 remains represented after the second; no alternative readout was tested. The predicted-win rule is a gain on a lateral fiber applied at selection, not a plasticity rule. The successor-state construction (state teacher-forced toward the next h words) carries nothing on the same corpus (PREREG_successor_state.md).

## ARC-CONJUNCT-EXPOSURE

**Status.** MEASURED. **Source.** This repository; the same law as the role-binding gain result.

**Engine / substrate.** reference_nemo_numpy (the vendored explicit-matrix FSM, declared profile; retained runner study of 20 seeds, plus the original 3-seed log)

**Claim.** A conjunction area collapses onto whichever conjunct is exposed more often, unless an opposing force (refraction) is present.

**Requires.**
- BOTH overlap directions measured -- one alone cannot distinguish a conjunction from collapse onto the other conjunct

**Evidence.**
- research/experiments/seq_arc_refraction_reference.py: ablating refraction takes across-symbol overlap 0.000 -> 0.989 and the task 3/3 -> 0/3
- arc-ref-study-20260912 (20 seeds, PREREG_arc_refraction_reference.md AR-2): refraction off, across-symbol overlap 0.977-0.997 on every seed (mean 0.988) against 0.000 with the proportional rule; across-state 0.098; decided 0/20 against 18/20

**Evidence files.**
- [research/results/runs/sequence.autonomous-chain/autonomous-chain-20260913/results.json](../research/results/runs/sequence.autonomous-chain/autonomous-chain-20260913/results.json) (artifact)
- [research/notes/sequence/PREREG_autonomous_chain.md](../research/notes/sequence/PREREG_autonomous_chain.md) (registration)
- [research/experiments/autonomous_chain.py](../research/experiments/autonomous_chain.py) (producer)
- [research/results/logs/seq_arc_refraction_reference.log](../research/results/logs/seq_arc_refraction_reference.log) (log) â€” three seeds, no run record
- [research/results/runs/sequence.arc-refraction-reference/arc-ref-study-20260912/results.json](../research/results/runs/sequence.arc-refraction-reference/arc-ref-study-20260912/results.json) (artifact)
- [research/notes/sequence/PREREG_arc_refraction_reference.md](../research/notes/sequence/PREREG_arc_refraction_reference.md) (registration)
- [research/experiments/seq_arc_refraction_reference.py](../research/experiments/seq_arc_refraction_reference.py) (producer)

**Mechanism sensitivity.**
- the collapse itself, at the extreme: a 32-transition chain over ONE constant symbol, so the symbol is 32 times more exposed than any state. With refraction off every state drives the IDENTICAL arc (overlap 1.0000 on every brain) and the chain takes zero correct steps; with refraction at beta the arcs are disjoint (0.0000) and all 32 steps are exact: `observations/arms/L32-no-refraction/rows/*/arc_overlap` all-greater `observations/arms/L32/rows/*/arc_overlap` by at least 0.9, retained in [research/results/runs/sequence.autonomous-chain/autonomous-chain-20260913/results.json](../research/results/runs/sequence.autonomous-chain/autonomous-chain-20260913/results.json) and paired by `run/seeds`.
- refraction ablated (the arc's bias never accumulates): across-symbol arc overlap at 15 presentations, every seed against the drive-proportional reference: `observations/arms/off-p15/rows/*/across_symbol` all-greater `observations/arms/proportional-p15/rows/*/across_symbol` by at least 0.5, retained in [research/results/runs/sequence.arc-refraction-reference/arc-ref-study-20260912/results.json](../research/results/runs/sequence.arc-refraction-reference/arc-ref-study-20260912/results.json) and paired by `observations/arms/off-p15/rows/*/seed`.

**Caveat.** Task #92 measured one direction, read 0.90-0.99, and concluded a conjunctive arc has no operating point. It has one.

## REFRACTION-PROPORTIONAL

**Status.** MEASURED. **Source.** This repository.

**Engine / substrate.** reference_nemo_numpy (the vendored explicit-matrix FSM, declared profile; retained runner study of 20 seeds, plus the original 3-seed log)

**Claim.** Refraction must charge in proportion to the winner's raw drive. A constant increment is not an equivalent parameterization: Hebbian growth multiplies drive while a constant grows linearly, so its operating point MOVES with training duration.

**Evidence.**
- research/experiments/seq_arc_refraction_reference.py: the constant winning at 15 presentations fails at 30, while the proportional rule passes both untouched
- arc-ref-study-20260912 (20 seeds, AR-3 and AR-4): constant 10 decides 20/20 at 15 presentations and 0/20 at 30 (across-symbol 0.000 -> 0.369); constant 30 decides 0/20 at 15 and 20/20 at 30; proportional decides 18/20 at 15 and 20/20 at 30 with both overlaps 0.000 throughout

**Evidence files.**
- [research/results/logs/seq_arc_refraction_reference.log](../research/results/logs/seq_arc_refraction_reference.log) (log) â€” three seeds, no run record
- [research/results/runs/sequence.arc-refraction-reference/arc-ref-study-20260912/results.json](../research/results/runs/sequence.arc-refraction-reference/arc-ref-study-20260912/results.json) (artifact)
- [research/notes/sequence/PREREG_arc_refraction_reference.md](../research/notes/sequence/PREREG_arc_refraction_reference.md) (registration)
- [research/experiments/seq_arc_refraction_reference.py](../research/experiments/seq_arc_refraction_reference.py) (producer)

**Mechanism sensitivity.**
- the constant increment that decides every seed at 15 presentations (s* = 10) against the proportional rule at 30 presentations: the constant's operating point has moved, the proportional rule's has not: `observations/arms/proportional-p30/rows/*/decided` all-greater `observations/arms/constant-s10-p30/rows/*/decided` by at least 1, retained in [research/results/runs/sequence.arc-refraction-reference/arc-ref-study-20260912/results.json](../research/results/runs/sequence.arc-refraction-reference/arc-ref-study-20260912/results.json) and paired by `observations/arms/proportional-p30/rows/*/seed`.

**Used by.** `neural_assemblies/core/_homeostasis.py`

## REFRACTION-NEEDS-LOAD

**Status.** MEASURED. **Source.** This repository.

**Engine / substrate.** numpy_sparse, arc materialized (retained runner replay, 10 seeds); the sampled-arc sweep's load floor is retracted

**Claim.** A refracted conjunction area has a CEILING in load M*k/n: above ~1.3 its conjunctions do not fit (10/10 correct at load 1.26, 0/10 at 1.80). RE-SCOPED 2026-09-09 (PREREG_sampler_audit.md): the lower edge this entry was named for -- 'below ~0.2 its assemblies never converge' -- was the numpy sampler's; with the arc materialized a 3-conjunction arc is 10/10 correct at every load from 0.04 to 0.60. An arc must be sized so its conjunctions fit; it need not be filled.

**Requires.**
- refraction active -- this is a statement about what refraction needs, not about k-WTA generally

**Evidence.**
- research/experiments/seq_a2_refraction_load.py, SAMPLED arc: a 3-conjunction arc goes 1/10 at load 0.04 to 10/10 at 0.21; a 9-conjunction arc holds 10/10 from 0.32 to 1.26 and collapses to 1/10 at 1.80
- the same script with the arc MATERIALIZED (NEMO_MATERIALIZE=1, 10 seeds): 3-conjunction arc 10/10 at every load 0.04-0.60; 9-conjunction arc 10/10 to 1.26, 0/10 at 1.80
- arc assembly stability across training: 0.286 from presentation 5 to 15 under-loaded, 0.957 from 10 to 15 loaded

**Evidence files.**
- [research/results/sequence/seq_a2_refraction_load_results.json](../research/results/sequence/seq_a2_refraction_load_results.json) (artifact) â€” sampled arc; lower-edge inference retracted
- [research/results/sequence/seq_a2_refraction_load_results_materialized.json](../research/results/sequence/seq_a2_refraction_load_results_materialized.json) (artifact)
- [research/results/runs/sequence.a2-refraction-load/sens-refraction-load-v3-20260912/results.json](../research/results/runs/sequence.a2-refraction-load/sens-refraction-load-v3-20260912/results.json) (artifact) â€” shared-runner replay with source archive and per-seed outcomes
- [research/experiments/seq_a2_refraction_load.py](../research/experiments/seq_a2_refraction_load.py) (producer)

**Provenance gap.** the two legacy artifacts lack runner records and source archives; the 2026-09-12 replay has both

**Mechanism sensitivity.**
- nine-conjunction arc at load 1.26 (n_arc 500) against load 1.80 (n_arc 350): the ceiling, arc materialized: `observations/rows/8/correct_by_seed` all-greater `observations/rows/9/correct_by_seed` by at least 1, retained in [research/results/runs/sequence.a2-refraction-load/sens-refraction-load-v3-20260912/results.json](../research/results/runs/sequence.a2-refraction-load/sens-refraction-load-v3-20260912/results.json) and paired by `observations/rows/8/seeds`.

**Sensitivity gap.** The retained contrast is the ceiling (load 1.26 against 1.80). No retained arm disables refraction itself at a fitting load; the refraction ablation this entry's docstring cites (0/10, arc collapsed to 0.988) predates the runner.

**Caveat.** The 'silent failure under load' this entry once described -- assemblies that never stop moving while every diagnostic reads healthy -- was measured on the sampled engine and does not occur materialized; treat it as a property of lazily drawn areas, not of refraction. The ceiling is [[AC-CAP]]'s and is not independent of it. One task, 10 seeds.

## REFRACTION-ANTI-MERGING

**Status.** MEASURED. **Source.** This repository; PREREG_refraction_memory.md (bars R1-R7, N1-N3, Q1-Q4, G1-G6, S8-S9).

**Engine / substrate.** hashed AssemblyMemory; materialized numpy_sparse mirror with summed stimulus parts (not an identical stimulus protocol)

**Claim.** A recurrent k-WTA area refracted at HALF beta and read with the refraction bias MASKED holds ~25x the Hebbian ceiling: at n/k = 67 M* ~ 1600-2200 stored assemblies against 64-89 for the control, x34-38 at n/k = 33, >= x13-16 at n/k = 133 (censored). Both ceilings are functions of n/k ALONE (n/k-matched cells agree within 25%), so refraction multiplies the assembly capacity law [[AC-CAP]] rather than changing its form. The mechanism is ANTI-MERGING, not orthogonalization: the stored assemblies are orthogonal (0.00x chance) only while the area has unvisited neurons; past fill 1.0 they overlap at chance like random subsets, yet remain DISTINCT (1.000) and recoverable from a half cue, where the Hebbian control collapses into hubs (distinct 0.63, overlap 15x chance) long before the area is full. The intrinsic bias vetoes recall (the net readout reads chance): a refracted memory is read through the veto or not at all. Fewer rounds per item raise the ceiling (T = 8 > T = 16) while the items still converge; refraction's benefit requires GRADED stimulus drive. SHAPE OF THE LAW (Amendment 4): in regime (k p >= 3 ln n) the refracted ceiling is ~0.40 (n/k)^2 at n/k = 67 and 133 (6995 at (8000, 60)), doubling exponents 2.1 then 1.8 -- Willshaw-like, not adopted as a power law; the control is ~0.017 (n/k)^2 from n/k = 67 on, so the multiplier is ~23-25x there. Out of regime (k p = 15 < 3 ln n) the cells fall 20-32% below their n/k pairs and do not converge at low load. GATED ROUNDS (Amendment 5): ending an item's rounds at its first repeated winner set under T_max = 8 raises the ceiling by a CONSTANT fraction of n/k: +35% at n/k = 67 (2645 vs 1961) and +24% at 133 (8666 vs 6995), both resolved; at both cells the ceiling sits where items stop converging inside T_max (the fraction converging is a U in load, 0.98 at mid load, 0 at the ceiling); the same gate STARVES the Hebbian control, whose winners settle in ~4 rounds before its memory is written; T_max = 16 under the gate costs the out-of-regime cell 13x. STRENGTH IS A SWITCH, NOT A DIAL (Amendment 6): 0.3, 0.4, 0.5 and 0.6 beta give the same ceiling (1919-1993, one bracket) at n/k = 67 -- the ceiling is the synaptic memory's, refraction only has to prevent merging while it is written; the churn transition is in (0.6, 0.7] beta at T = 8. WHAT THE CONSTANT IS (Amendment 9): independent random k-subsets written into the SAME circuit at the memory's own write strength fail at 0.69-0.92 (n/k)^2 with half of the present synapses potentiated (0.49-0.60), Willshaw's operating point -- the n/k law is the load M (k/n)^2; the refracted memory realises a constant 0.47-0.56 of that (PATTERN EFFICIENCY), the Hebbian control 0.014-0.023. The reference is conditional on write strength: two potentiation counts weaker moves the random-pattern ceiling +66-89% at k = 60 and +17-29% at k = 120. The refracted assemblies share synapse pairs 2.1-2.8x as much as random subsets and fail at 1.14-1.43x random's pair load; the gate's gain is carried by its assemblies (rewritten cleanly at equal write strength they store +35-51%). WRITE STRENGTH (Amendment 11, a replication of Amendment 10's void record): on the same circuit COMPLETION (>= 0.8 of an item from half of it) needs a strong write and peaks at the clipped binary end, ~g(k p) (n/k)^2 with g = 0.29 / 0.56 / 0.80 at k p = 15 / 30 / 60, where the n/k law fails (k = 120 stores 1.42x its n/k pair); IDENTIFICATION (rank-1) grows as the write weakens, to 4.3-4.5 (n/k)^2 at one count per item, inside a LOAD WINDOW whose lower edge also rises. The operating write (5-6 counts) sits between: its rank-1 ceiling obeys the n/k law and it completes little; the smallest write that completes falls with k p (11 / 6 / 4 counts at k = 30 / 60 / 120). LEARNING RATE (Amendments 12-13, confirmed on new brains and two unseen cells): the refracted memory's DISTINCT-completion-optimal beta obeys ln(1 + beta*) = 0.29 / sqrt(k p / 2) (gamma* 0.257-0.303 at seven cells, k p = 30-120; exponent 1/2 collapses beta* to CV 0.05 against 0.21 for beta itself) -- a fan-in-scaled learning rate, the counterpart of muP's per-layer rate for one area. The adopted beta = 0.1 is above it everywhere (1.3-2.7x), so the ceilings above are identification ceilings at an over-strong write; at the optimum the best distinct completion grows with k p (0.41-2.46 (n/k)^2 at n/k = 33). Completion must be DISTINCT (recovered and rank-1): above beta ~ 0.2 merged assemblies make 'completion' exceed identification. ON RECALL (Amendment 14, ten (n, k, p) cells, new brains): with each memory at its own best learning rate the refracted memory completes 14-57x as many distinct items as the Hebbian control at every cell; at equal (n/k, k p) capacity is the same at p = 0.125, 0.25, 0.5 (within 8%) but sparse p wants a higher learning rate; whether capacity at the optimum scales as n^2 p / (k ln(n/k)) or (n/k)^2 is NOT settled. THE LAW WITH CONNECTIVITY NOISE (Amendment 16, six cells, p = 0.125-0.75): ln(1 + beta*) = 0.285 sqrt(2 (1 - p)) / sqrt(k p / 2) -- the write clears the spread of the cue synapses a neuron receives -- within 15% everywhere, including the dense side (0.707 predicted 0.71); capacity at the optimum is the same across p at equal fan-in. DEPTH (Amendment 15): at each number of write rounds' own best rate capacity is the same (T = 6-16); the per-round optimum falls faster than 1/T. THE CONVERGENCE-THRESHOLD RULE (Amendment 17, ten cells, new brains): above the floor k p >= 3 ln n the best rate is 0.18-0.19 of sqrt((1 - p) ln n / (p k)) (k p = 30-80); below it the fraction holds at p = 0.5 down to k p = 5 but climbs at p = 0.05 (0.21 / 0.27 / 0.30 at k p = 20 / 10 / 5), capacity is flat in k (1356-1452 at k = 10-40) and no longer p-invariant (201 vs 1356 at k p = 5), and the PNAS 2020 cell (k p = 1) completes at no rate (rank-1 at most 112). THE ONSET (Amendment 18, ten cells, a 12-per-octave grid): the weakest completing write is 0.16-0.18 theta everywhere (CV 0.041) -- dense and sparse, above and below the floor; in dense areas it completes only after 158-1098 stored items and the best write sits within a step of it, in sparse ones it completes at once and the best write is 1.6-1.8x higher. REGIMES (Amendment 19): at fixed n and p, capacity at each cell's optimum is 1059-1513 over k = 10-160 -- nearly k-independent, no law change at the floor. RECOGNITION vs RECALL (Amendment 20): the ratio grows as fan-in falls (Spearman -0.88); at the PNAS parameters recall appears only from k p = 4 while recognition holds 114-207 items from k p = 1. THE IN-DEGREE LAW (Amendment 21, nine new cells): at the best write capacity is set by d = n p alone -- at equal d the cells agree within 11% while n/k varies fourfold -- and grows as d^1.5-1.7 (refit on 55 cells: C = 0.0099 d^1.554, x1.15); the registered constants missed by 26% at d = 6000. TWO MEMORIES (Amendment 23): each at its own best write, recognition holds 15-86x as many items as recall, at a write 6-40x weaker (int16 counts); at fixed d recall stays put while recognition scales with n/k and its best write as 1/k.

**Requires.**
- recurrent k-WTA area, weight clip, norm_init, no column scaling (arm B); refraction strength 0.5 beta; T = 8 rounds per item from an inhibited area; readout = half-cue recall with the bias masked
- GRADED stimulus drive: a zero-or-size stimulus (the engine's into a materialized area, at one Bernoulli draw) makes the refracted item rotate through its tied connected set and the benefit vanishes

**Evidence.**
- seq_capacity_scaling.py, arm B on the organ fiber, 20 brains, M grid to 4096, s = 0.5 beta masked vs control: n/k = 33: 431 / 383 vs 11.3; n/k = 67: 1961 / 1589 / 2230 vs 83 / 64 / 89; n/k = 133: >= 4096 vs 307 / 263
- distinct 1.000 at every refracted ceiling; control 0.63 at M = 128 (n = 4000), pairwise 15x chance
- net readout at n = 4000: M* = 8 (chance at every M)
- T = 16: M* = 203 against T = 8's >= 1024 (n = 4000); T = 4 does not converge at low load and does at high load
- numpy_sparse gate (refraction_memory_numpy.py, materialized, 5 brains, summed Binomial stimuli): refracted >= 512 (censored, rank-1 1.000 throughout, distinct 1.000) vs control 34 -- the claim holds on the engine whose k-WTA has no selector defect
- grid to 16384, 20 brains: (8000, 60) REF 6995 [6144, 8192) = 0.395 (n/k)^2, CTL 307; (4000, 30) REF 4749 [4096, 6144), CTL 263 (out of regime)
- (4000, 60) gated T_max 8: REF 2645 [2560, 2816) vs 1961; CTL gated rank-1 0.32 at M = 8 (no memory formed); (4000, 30) gated T_max 16: 362 vs 4749
- (8000, 60) gated T_max 8: 8666 [8192, 10240) vs 6995 (x1.24)
- strength 0.3 / 0.4 / 0.5 / 0.6 beta at (4000, 60): 1986 / 1919 / 1961 / 1993, all [1536, 2048)
- memory_pattern_efficiency.py, 20 brains, seven cells (A9): M*/(n/k)^2 real / random / balanced at (2000,60) 0.394 / 0.736 / 1.629, (4000,120) 0.359 / 0.686 / 3.078, (4000,60) 0.432 / 0.806 / 1.384, (8000,120) 0.512 / 0.920 / 2.296, (8000,60) 0.403 / 0.860 / 1.167; gated 9078 and 2593
- memory_write_strength.py --scan, 20 brains, seven cells (A11): completion at the binary write / (n/k)^2 0.562 / 0.563 / 0.560 at k = 60, 0.798 / 0.798 at k = 120; rank-1 window at one count per item [3310, 76028) at (8000, 60)
- memory_learning_rate.py --distinct, 20 new brains, seven cells (A13): beta* 0.0705 / 0.0558 / 0.0532 / 0.0444 / 0.0377 at n/k = 33 (k p 30 / 45 / 60 / 90 / 120), 0.0752 / 0.0569 at n/k = 67; predicted 0.063 and 0.044 at the unseen (3000,90), (6000,180)
- memory_threshold_law.py, 20 new brains, ten cells (A17): beta*/theta 0.180 / 0.189 / 0.180 above the floor; 0.216 / 0.193 / 0.213 (p 0.5) and 0.211 / 0.270 / 0.296 (p 0.05) below; (10000, 100, 0.01) distinct completion 0 at every rate
- memory_onset.py, 20 new brains, ten cells (A18): onset/theta 0.170 x5, 0.180 x3, 0.160 x2; dense onset windows open at 158-1098 items; best/onset 1.00-1.06 (dense, 5 of 6), 1.59-1.78 (p 0.05)
- memory_regimes.py, 20 new brains, fifteen cells (A19-20): capacity 1314/1148/1513/1383/1473/1392/1468/1239/1059 at n = 4000, k = 10-160; 508/1513/3222 at n = 2000/4000/8000, k = 20; rank-1 at its own best write >= 32768 at k p <= 10, 2511 at k p 80
- memory_degree_law.py, 20 new brains, nine cells (A21): d = 1000: 417/439; d = 1500: 971/811/944 (n/k 200/100/50); d = 3000: 2871/2739; d = 6000: 8605; slope 1.669
- memory_recognition.py v2, 20 new brains, seven cells (A23): recognition 81290/109640/51316/27917/12521 vs recall 1350/1292/1339/1299/837 at n = 4000, k = 10-160

**Evidence files.**
- [research/notes/memory/PREREG_refraction_memory.md](../research/notes/memory/PREREG_refraction_memory.md) (registration)
- [research/results/runs/memory.capacity-scaling/capacity-record-consumed-20260910/results.json](../research/results/runs/memory.capacity-scaling/capacity-record-consumed-20260910/results.json) (artifact) â€” registered protocol-consumption replay, not the whole historical grid
- [research/results/memory/refraction_memory_numpy_results.json](../research/results/memory/refraction_memory_numpy_results.json) (artifact)
- [research/results/memory/capacity_scaling_results_figure_ref.json](../research/results/memory/capacity_scaling_results_figure_ref.json) (artifact)
- [research/results/memory/capacity_scaling_results_figure_ctl.json](../research/results/memory/capacity_scaling_results_figure_ctl.json) (artifact)
- [research/results/runs/memory.capacity-scaling/refraction-paired-sensitivity-20260911/results.json](../research/results/runs/memory.capacity-scaling/refraction-paired-sensitivity-20260911/results.json) (artifact) â€” paired version-3 reproduction and sensitivity run
- [research/results/comparisons/refraction-paired-sensitivity-20260911.json](../research/results/comparisons/refraction-paired-sensitivity-20260911.json) (comparison) â€” 2090-scalar migration comparison receipt
- [research/results/runs/memory.capacity-scaling/capacity-law-replay-L1-20260930/results.json](../research/results/runs/memory.capacity-scaling/capacity-law-replay-L1-20260930/results.json) (artifact) â€” Amendment 8 replay: (2000,30), (4000,120), (8000,120), paired
- [research/results/runs/memory.capacity-scaling/capacity-law-replay-L2-20260930/results.json](../research/results/runs/memory.capacity-scaling/capacity-law-replay-L2-20260930/results.json) (artifact) â€” Amendment 8 replay: (8000,60), (4000,30), paired, grid to 16384
- [research/results/runs/memory.capacity-scaling/capacity-law-replay-L3-20260930/results.json](../research/results/runs/memory.capacity-scaling/capacity-law-replay-L3-20260930/results.json) (artifact) â€” Amendment 8 replay: (2000,60), paired
- [research/results/comparisons/capacity-law-replay-L1-20260930.json](../research/results/comparisons/capacity-law-replay-L1-20260930.json) (comparison) â€” 3615-scalar per-seed comparison with the surviving k-sweep control file
- [research/results/comparisons/capacity-law-replay-L2-20260930.json](../research/results/comparisons/capacity-law-replay-L2-20260930.json) (comparison) â€” 3050-scalar per-seed comparison with the surviving n/k = 133 refracted file
- [research/results/runs/memory.pattern-efficiency/pattern-efficiency-20260930/results.json](../research/results/runs/memory.pattern-efficiency/pattern-efficiency-20260930/results.json) (artifact) â€” Amendment 9: real, clean, random, balanced and gated memories on one circuit, seven cells
- [research/results/runs/memory.write-strength/write-strength-scan-20261001/results.json](../research/results/runs/memory.write-strength/write-strength-scan-20261001/results.json) (artifact) â€” Amendment 11: identification and completion load windows across write strength, seven cells
- [research/results/runs/memory.learning-rate/learning-rate-20261001/results.json](../research/results/runs/memory.learning-rate/learning-rate-20261001/results.json) (artifact) â€” Amendment 12: beta sweep on the real memory, five cells (T1-T3 failed: merged recall)
- [research/results/runs/memory.learning-rate/learning-rate-distinct-20261001/results.json](../research/results/runs/memory.learning-rate/learning-rate-distinct-20261001/results.json) (artifact) â€” Amendment 13: distinct-completion beta sweep, new brains, seven cells incl. two unseen
- [research/results/runs/memory.recall-law/recall-law-20261001/results.json](../research/results/runs/memory.recall-law/recall-law-20261001/results.json) (artifact) â€” Amendment 14: refracted vs Hebbian on distinct completion, ten (n, k, p) cells
- [research/results/runs/memory.time-depth/time-depth-20261001/results.json](../research/results/runs/memory.time-depth/time-depth-20261001/results.json) (artifact) â€” Amendment 15: write rounds T = 6-16, recall at 8 rounds
- [research/results/runs/memory.sparse-law/sparse-law-20261001/results.json](../research/results/runs/memory.sparse-law/sparse-law-20261001/results.json) (artifact) â€” Amendment 16: the learning-rate law with connectivity noise, p = 0.125-0.75
- [research/results/runs/memory.threshold-law/threshold-law-20261001/results.json](../research/results/runs/memory.threshold-law/threshold-law-20261001/results.json) (artifact) â€” Amendment 17: the best rate as a fraction of the convergence threshold, k p = 1-80
- [research/results/runs/memory.onset/onset-20261002-r3/results.json](../research/results/runs/memory.onset/onset-20261002-r3/results.json) (artifact) â€” Amendment 18: the completion onset on a 12-per-octave grid, ten cells
- [research/results/runs/memory.regimes/regimes-20261002-r3/results.json](../research/results/runs/memory.regimes/regimes-20261002-r3/results.json) (artifact) â€” Amendments 19-20: capacity across k and n, recognition against recall
- [research/results/runs/memory.degree-law/degree-law-20261002/results.json](../research/results/runs/memory.degree-law/degree-law-20261002/results.json) (artifact) â€” Amendment 21: capacity at the best write against the in-degree d = n p
- [research/results/runs/memory.recognition/recognition-20261003-a23/results.json](../research/results/runs/memory.recognition/recognition-20261003-a23/results.json) (artifact) â€” Amendment 23: recognition and recall each at its own best write
- [research/results/runs/memory.criticality/criticality-20261003/results.json](../research/results/runs/memory.criticality/criticality-20261003/results.json) (artifact) â€” Amendment 24: per-brain onsets at n = 2000-16000; onset converges at 0.163-0.170 theta

**Provenance gap.** the seven cells the law is fitted to are replayed under the runner (Amendment 8 and A7): exact per-seed reproduction wherever legacy data survives, every lost ceiling reproduced; the two gated ceilings are replayed within 5% (Amendment 9, PE-R); the strength (Amendment 6) and T-sweep grids and the numpy mirror still predate immutable source records

**Mechanism sensitivity.**
- refracted memory versus paired Hebbian control at M=128: `observations/conditions/refracted/cells/B~14000~160/checkpoints/128/rank1` all-greater `observations/conditions/control/cells/B~14000~160/checkpoints/128/rank1` by at least 0.5, retained in [research/results/runs/memory.capacity-scaling/refraction-paired-sensitivity-20260911/results.json](../research/results/runs/memory.capacity-scaling/refraction-paired-sensitivity-20260911/results.json) and paired by `run/seeds`.
- refracted memory versus paired Hebbian control at (2000, 60), M=64 (Amendment 8, RP-4): `observations/conditions/refracted/cells/B~12000~160/checkpoints/64/rank1` all-greater `observations/conditions/control/cells/B~12000~160/checkpoints/64/rank1` by at least 0.5, retained in [research/results/runs/memory.capacity-scaling/capacity-law-replay-L3-20260930/results.json](../research/results/runs/memory.capacity-scaling/capacity-law-replay-L3-20260930/results.json) and paired by `run/seeds`.
- refracted memory versus paired Hebbian control at (4000, 120), M=64 (Amendment 8, RP-4): `observations/conditions/refracted/cells/B~14000~1120/checkpoints/64/rank1` all-greater `observations/conditions/control/cells/B~14000~1120/checkpoints/64/rank1` by at least 0.5, retained in [research/results/runs/memory.capacity-scaling/capacity-law-replay-L1-20260930/results.json](../research/results/runs/memory.capacity-scaling/capacity-law-replay-L1-20260930/results.json) and paired by `run/seeds`.
- refracted memory versus paired Hebbian control at (2000, 30), M=384 (Amendment 8, RP-4): `observations/conditions/refracted/cells/B~12000~130/checkpoints/384/rank1` all-greater `observations/conditions/control/cells/B~12000~130/checkpoints/384/rank1` by at least 0.5, retained in [research/results/runs/memory.capacity-scaling/capacity-law-replay-L1-20260930/results.json](../research/results/runs/memory.capacity-scaling/capacity-law-replay-L1-20260930/results.json) and paired by `run/seeds`.
- refracted memory versus paired Hebbian control at (8000, 120), M=384 (Amendment 8, RP-4): `observations/conditions/refracted/cells/B~18000~1120/checkpoints/384/rank1` all-greater `observations/conditions/control/cells/B~18000~1120/checkpoints/384/rank1` by at least 0.5, retained in [research/results/runs/memory.capacity-scaling/capacity-law-replay-L1-20260930/results.json](../research/results/runs/memory.capacity-scaling/capacity-law-replay-L1-20260930/results.json) and paired by `run/seeds`.
- refracted memory versus paired Hebbian control at (8000, 60), M=1536 (Amendment 8, RP-4): `observations/conditions/refracted/cells/B~18000~160/checkpoints/1536/rank1` all-greater `observations/conditions/control/cells/B~18000~160/checkpoints/1536/rank1` by at least 0.5, retained in [research/results/runs/memory.capacity-scaling/capacity-law-replay-L2-20260930/results.json](../research/results/runs/memory.capacity-scaling/capacity-law-replay-L2-20260930/results.json) and paired by `run/seeds`.
- refracted memory versus paired Hebbian control at (4000, 30), M=1536 (Amendment 8, RP-4): `observations/conditions/refracted/cells/B~14000~130/checkpoints/1536/rank1` all-greater `observations/conditions/control/cells/B~14000~130/checkpoints/1536/rank1` by at least 0.5, retained in [research/results/runs/memory.capacity-scaling/capacity-law-replay-L2-20260930/results.json](../research/results/runs/memory.capacity-scaling/capacity-law-replay-L2-20260930/results.json) and paired by `run/seeds`.

**Sensitivity gap.** Paired refracted-versus-control checks now cover all seven cells the law is fitted to (A7 at (4000, 60); Amendment 8 at the other six). The masked-vs-net veto, the convergence gate, the strength plateau and the cross-engine mirror still lack retained paired checks.

**Caveat.** Found by re-measuring PREREG_refraction_capacity.md after the hashed selector's sign defect (1b475fc) -- its Amendment 1 ('spends the substrate') was that defect. The n/k law was counted as holding at n/k = 133 while both cells were censored; resolved, they disagree by 0.68 and the k = 30 one is out of regime -- k p >= 3 ln n is a precondition, not a footnote. Three ratios do not fix an exponent that is falling (2.1 -> 1.8). The multiplier is at 0.5 beta; 0.7 beta converges only given T = 16 and holds ~185 at n/k = 67, but strength is otherwise a plateau whose lower edge (below 0.3 beta) is unmeasured. The gate is two in-regime cells (n/k = 67, 133); n/k = 33 and out of regime are unmeasured, and the gated ceiling's doubling exponent (1.71) may keep falling.

## SEQUENCE-MEMORY-ROBUSTNESS

**Status.** MEASURED. **Source.** PREREG_refraction_memory.md Amendment 34 (N1-N4 PASS); Amendment 36 (U1-U4 PASS)

**Engine / substrate.** hashed AssemblyMemory, store_sequence one round per element, refraction 0.5 beta recovering over 33 rounds, beta = theta

**Claim.** The refracted sequence memory's length limit is a budget of elements, spent alike on one long sequence or many short ones (0.81-1.08 of the single-sequence limit); replay loses nothing with up to 7% of its MOST DRIVEN winners replaced by random neurons at every step, derails at a horizon at 10% that a twice-larger area pushes out about sixfold, and starts from its strongest half cue a quarter wrong. Under UNIFORMLY random replacement (Amendment 36) it is markedly more tolerant: at 10% the smaller area replays 2.5 times further (103.6 against 41.2 steps) and the larger does not derail in 400 steps (20 of 20 brains); a random half cue works as well as the strongest.

**Requires.**
- noise = replacement of the first slots of each replay step's winners, which the k-WTA orders strongest first (Amendment 36 erratum); the cue is the strongest half of the first state
- two cells, k = 60, p = 0.5

**Evidence.**
- research/notes/memory/PREREG_refraction_memory.md#amendment-34-result-2026-10-07
- research/notes/memory/PREREG_refraction_memory.md#amendment-36-result-2026-10-08

**Evidence files.**
- [research/results/runs/memory.robustness/robustness-20261007/results.json](../research/results/runs/memory.robustness/robustness-20261007/results.json) (artifact) â€” Amendment 34: many sequences, activity noise, corrupted cues, two cells, 20 brains
- [research/results/runs/memory.noise/noise-20261008/results.json](../research/results/runs/memory.noise/noise-20261008/results.json) (artifact) â€” Amendment 36: top-slot and uniformly random noise, strongest and random half cues, two cells, 20 brains

**Mechanism sensitivity.**
- replay survives 5% activity noise per step and derails within a few steps at 20%, at (4000, 60, 0.5) (Amendment 34): `observations/cells/4000~160~10.5/noise/0.05/values` all-greater `observations/cells/4000~160~10.5/noise/0.2/values` by at least 300, retained in [research/results/runs/memory.robustness/robustness-20261007/results.json](../research/results/runs/memory.robustness/robustness-20261007/results.json) and paired by `run/seeds`.
- replay survives 5% activity noise per step and derails within a few steps at 20%, at (8000, 60, 0.5) (Amendment 34): `observations/cells/8000~160~10.5/noise/0.05/values` all-greater `observations/cells/8000~160~10.5/noise/0.2/values` by at least 300, retained in [research/results/runs/memory.robustness/robustness-20261007/results.json](../research/results/runs/memory.robustness/robustness-20261007/results.json) and paired by `run/seeds`.

**Caveat.** ERRATUM (Amendment 36): Amendment 34's noise was described as uniform random replacement but replaced the most driven winners (the k-WTA orders them strongest first), nearly all true members; Amendment 36 measured both. The mechanism of the horizon is not identified. Synaptic noise and noise during writing are untested.

## SEQUENCE-LOAD-LAW

**Status.** MEASURED. **Source.** PREREG_refraction_memory.md Amendment 37 (R1-R3 PASS); its constant fails at n/k = 300, Amendment 38 (K1, D1, S1 FAIL; S2 PASS)

**Engine / substrate.** hashed AssemblyMemory, store_sequence one round per element, refraction 0.5 beta recovering over 64 rounds, beta = theta

**Claim.** One area's single-sequence replay fails at a critical interference load: with rho = L k ln n / (n^2 p), every cell with k p >= 3 ln n and n/k <= 150 replays a sequence whole from a random half cue on every brain up to rho ~ 0.11, and on none a factor 1.07-1.15 past its 90% point. Registered from an exploratory survey (rho_50 = 0.142, 0.119-0.171 at seven cells) and confirmed at two cells never run: rho_50 = 0.141 at (6000, 90, 0.4) and 0.115 at (12000, 80, 0.5), rho_90 = 0.132 and 0.111. Below the floor it fails earlier (0.086, 0.089).

**Requires.**
- k p >= 3 ln n; a recovering refraction (tau = 64) at beta = theta; one sequence per area, noiseless one-round recall
- n/k <= 150: nine in-regime cells, n 2000-16000; at n/k = 300 the cliff is at rho = 0.081 (Amendment 38)

**Evidence.**
- research/notes/memory/PREREG_refraction_memory.md#amendment-37-result-2026-10-08
- research/notes/memory/PREREG_refraction_memory.md#amendment-38-result-2026-10-08

**Evidence files.**
- [research/results/runs/memory.load-law/load-law-20261008/results.json](../research/results/runs/memory.load-law/load-law-20261008/results.json) (artifact) â€” Amendment 37: the load ladder at two held-out cells and one out-of-regime cell, 20 brains
- [research/results/runs/memory.load-drift/load-drift-20261008/results.json](../research/results/runs/memory.load-drift/load-drift-20261008/results.json) (artifact) â€” Amendment 38: the load ladder at two cells with n/k = 300, 20 brains

**Mechanism sensitivity.**
- every brain replays a sequence whole at rho = 0.101 and derails within ~70 steps at rho = 0.156, at (6000, 90, 0.4) (Amendment 37): `observations/cells/6000~190~10.4/ladder/1856/steps` all-greater `observations/cells/6000~190~10.4/ladder/2862/steps` by at least 1000, retained in [research/results/runs/memory.load-law/load-law-20261008/results.json](../research/results/runs/memory.load-law/load-law-20261008/results.json) and paired by `run/seeds`.
- every brain replays a sequence whole at rho = 0.093 and none at rho = 0.120, at (12000, 80, 0.5) (Amendment 37): `observations/cells/12000~180~10.5/ladder/8866/steps` all-greater `observations/cells/12000~180~10.5/ladder/11498/steps` by at least 1000, retained in [research/results/runs/memory.load-law/load-law-20261008/results.json](../research/results/runs/memory.load-law/load-law-20261008/results.json) and paired by `run/seeds`.

**Caveat.** SEMI-EMPIRICAL: the variable is derived, the constant is fitted to the survey; a parameter-free reduced model puts the cliff about twice too high. THE CONSTANT IS NOT A CONSTANT: rho_50 is lower at n/k >= 133 (0.115-0.123) than at n/k <= 67 (0.131-0.171), and at n/k = 300 it is 0.081 at both cells of Amendment 38 -- below the constant, below a post hoc power-law drift, and below the safe rule rho <= 0.09, which does not hold there. Do not extrapolate past n/k = 150. Many sequences, noise, tau other than 64 and the Hebbian (non-refracted) memory are not covered.

## SEQUENCES-OF-SEQUENCES-ACROSS-AREAS

**Status.** MEASURED. **Source.** PREREG_refraction_memory.md Amendment 33 (H1-H4 PASS); rerun on the corrected cross-area divisor, Amendment 35 (H1-H4 PASS)

**Engine / substrate.** two hashed AssemblyMemory areas (sequence S, chunk C) and a DenseOrganFiber C -> S; recall on two clocks

**Claim.** Two areas replay sequences of sequences: chunks written once into a sequence area are replayed in whatever order a plan in a chunk area dictates -- every plan whole, every brain, from the plan's first state alone -- with plans that share chunks kept apart by the chunk area and the content shared through the sequence area. The links from plan positions to chunk starts carry it, and under load they give way first; doubling both areas relieves them.

**Requires.**
- chunk starts linked by construction (the circuit does not find chunk boundaries itself); fixed chunk length on a two-clock recall
- each area as in ADAPTATION-SWITCHES-MEMORY-TYPE: beta = theta, refraction 0.5 beta recovering over 33 rounds

**Evidence.**
- research/notes/memory/PREREG_refraction_memory.md#amendment-33-result-2026-10-07
- research/notes/memory/PREREG_refraction_memory.md#amendment-35-result-2026-10-07

**Evidence files.**
- [research/results/runs/memory.hierarchy/hierarchy-20261007-a35/results.json](../research/results/runs/memory.hierarchy/hierarchy-20261007-a35/results.json) (artifact) â€” Amendment 35: Amendment 33 on the corrected C -> S divisor, the same 20 brains
- [research/results/runs/memory.hierarchy/hierarchy-20261007/results.json](../research/results/runs/memory.hierarchy/hierarchy-20261007/results.json) (artifact) â€” Amendment 33: three loads, links 0 and 2, two area pairs, 20 brains (C -> S divided by the in-degree over S's rows; see Amendment 35)

**Mechanism sensitivity.**
- plan 0 replays whole with the chunk-area to sequence-area links and not without them, at (S, C) = (4000, 2000) (Amendments 33, 35): `observations/cells/4000~12000/links/8~16~12/0/whole/values` all-greater `observations/cells/4000~12000/links/8~16~10/0/whole/values` by at least 0.5, retained in [research/results/runs/memory.hierarchy/hierarchy-20261007-a35/results.json](../research/results/runs/memory.hierarchy/hierarchy-20261007-a35/results.json) and paired by `run/seeds`.
- plan 0 replays whole with the chunk-area to sequence-area links and not without them, at (S, C) = (8000, 4000) (Amendments 33, 35): `observations/cells/8000~14000/links/8~16~12/0/whole/values` all-greater `observations/cells/8000~14000/links/8~16~10/0/whole/values` by at least 0.5, retained in [research/results/runs/memory.hierarchy/hierarchy-20261007-a35/results.json](../research/results/runs/memory.hierarchy/hierarchy-20261007-a35/results.json) and paired by `run/seeds`.

**Caveat.** Two levels, p = 0.5, no noise, chunks of 12 and plans of 5 or 8; the recall's clock (one chunk-area step per chunk) is given, not generated.

## BIDIRECTIONAL-RECALL-BY-LRI

**Status.** MEASURED. **Source.** PREREG_refraction_memory.md Amendments 31 (R1, R2, R4 PASS; R3 FAIL) and 32 (Q1-Q3 PASS)

**Engine / substrate.** hashed AssemblyMemory, store_sequence with forward_counts/reverse_counts, LRI period 4 at recall with the cue and came-from state primed

**Claim.** One area replays a stored 200-element sequence forward, backward, and either way from its middle, on every brain, when its two directions are written about equally and long-range inhibition at recall vetoes the state just left; without LRI it goes nowhere, and with the directions unequal the stronger one wins whatever LRI does.

**Requires.**
- reverse transitions written after the sequence (a stand-in for a post-before-pre rule from a trace); forward and reverse counts within about a third of one write of each other
- the recall cue and, from the middle, the came-from state entered into the LRI history

**Evidence.**
- research/notes/memory/PREREG_refraction_memory.md#amendment-32-result-2026-10-07

**Evidence files.**
- [research/results/runs/memory.bidirectional/bidirectional-balanced-20261007/results.json](../research/results/runs/memory.bidirectional/bidirectional-balanced-20261007/results.json) (artifact) â€” Amendment 32: balanced and unbalanced arms, two cells, 20 brains
- [research/results/runs/memory.bidirectional/bidirectional-20261007/results.json](../research/results/runs/memory.bidirectional/bidirectional-20261007/results.json) (artifact) â€” Amendment 31: reverse links alone -- backward recall, forward lost

**Mechanism sensitivity.**
- a balanced chain (2 + 3) replays backward under LRI where a forward-heavy one (2 + 2) does not, at (4000, 60, 0.5) (Amendment 32): `observations/cells/4000~160~10.5/reverse/2+3/backward_lri/values` all-greater `observations/cells/4000~160~10.5/reverse/2+2/backward_lri/values` by at least 0.5, retained in [research/results/runs/memory.bidirectional/bidirectional-balanced-20261007/results.json](../research/results/runs/memory.bidirectional/bidirectional-balanced-20261007/results.json) and paired by `run/seeds`.
- a balanced chain (2 + 3) replays backward under LRI where a forward-heavy one (2 + 2) does not, at (8000, 60, 0.5) (Amendment 32): `observations/cells/8000~160~10.5/reverse/2+3/backward_lri/values` all-greater `observations/cells/8000~160~10.5/reverse/2+2/backward_lri/values` by at least 0.5, retained in [research/results/runs/memory.bidirectional/bidirectional-balanced-20261007/results.json](../research/results/runs/memory.bidirectional/bidirectional-balanced-20261007/results.json) and paired by `run/seeds`.

**Caveat.** The reverse links are written after the sequence, not by a modelled trace rule; the balance window is narrow and was set by counts, not learned. Two cells, L = 200, p = 0.5.

## RECOVERY-SETS-SEQUENCE-LENGTH

**Status.** MEASURED. **Source.** PREREG_refraction_memory.md Amendment 29 (B2 PASS; B1, B3 FAIL at (2000, 60) only; B4 FAIL)

**Engine / substrate.** hashed AssemblyMemory, store_sequence one round per element, refraction 0.5 beta decaying as exp(-1/tau) per writing round, beta = theta

**Claim.** How long a sequence one area recalls by itself is set by how fast its refraction recovers: the limit rises from the unrefracted area's merging cliff as recovery slows, peaks at an interior recovery time (32 to 64 writing rounds at four cells), and falls back to the tiling deadline -- 6890 elements at (8000, 60, 0.5) against 861 unrefracted and 264 never recovering. At the densest code (n/k = 33) the gain is only 1.27x.

**Requires.**
- one chosen sequence per brain, one stimulus-and-recurrence round per element; replay from half of element 0 by frozen masked rounds
- recovery modelled as bias *= exp(-1/tau) at every writing round

**Evidence.**
- research/notes/memory/PREREG_refraction_memory.md#amendment-29-result-2026-10-07

**Evidence files.**
- [research/results/runs/memory.recovery/recovery-20261007/results.json](../research/results/runs/memory.recovery/recovery-20261007/results.json) (artifact) â€” Amendment 29: length limits at twelve recovery times, four cells, 20 brains

**Mechanism sensitivity.**
- a bias recovering over 64 writing rounds replays a 724-element sequence whole where the never-recovering bias breaks, at (8000, 60, 0.5) (Amendment 29): `observations/cells/8000~160~10.5/taus/64/curve/724/values` all-greater `observations/cells/8000~160~10.5/taus/inf/curve/724/values` by at least 0.5, retained in [research/results/runs/memory.recovery/recovery-20261007/results.json](../research/results/runs/memory.recovery/recovery-20261007/results.json) and paired by `run/seeds`.
- a bias recovering over 64 writing rounds replays a 1448-element sequence whole where the never-recovering bias breaks, at (4000, 30, 0.5) (Amendment 29): `observations/cells/4000~130~10.5/taus/64/curve/1448/values` all-greater `observations/cells/4000~130~10.5/taus/inf/curve/1448/values` by at least 0.5, retained in [research/results/runs/memory.recovery/recovery-20261007/results.json](../research/results/runs/memory.recovery/recovery-20261007/results.json) and paired by `run/seeds`.

**Caveat.** The best limit follows the in-degree n p (post hoc: 1.5-1.9x the attractor memory's 0.0135 d^1.51), not (n/k)^2 as registered (B4 failed). One sequence per brain, no noise, p = 0.5; recall reads the synapses with the bias masked. Mapping a writing round onto physical time is an assumption.

## SEQUENCE-TILING-DEADLINE

**Status.** MEASURED. **Source.** PREREG_refraction_memory.md Amendment 28 (D1, D2, D3 PASS)

**Engine / substrate.** hashed AssemblyMemory, store_sequence one round per element, refraction 0.5 beta, beta = theta

**Claim.** A refracted area writes a single sequence onto neurons that have never fired until none remain: the sequence tiles the area, the fresh pool is empty at element n/k, and replay breaks at multiples of n/k (49 of 51 breaks within two steps). A bias that recovers -- zeroed every n/(2k) elements -- removes the deadline: every brain replays 4 n/k elements (532 at n/k = 133, where the never-decaying bias breaks near 130).

**Requires.**
- one chosen sequence per brain, one stimulus-and-recurrence round per element; replay from half of element 0 by frozen masked rounds
- cumulative refraction bias (bias += s raw at every win, never decaying)

**Evidence.**
- research/notes/memory/PREREG_refraction_memory.md#amendment-28-result-2026-10-06

**Evidence files.**
- [research/results/runs/memory.tiling/tiling-20261006/results.json](../research/results/runs/memory.tiling/tiling-20261006/results.json) (artifact) â€” Amendment 28: fresh share per element and replay breaks, refracted and reset arms, four cells
- [research/results/runs/memory.sequence-length/sequence-length-20261006/results.json](../research/results/runs/memory.sequence-length/sequence-length-20261006/results.json) (artifact) â€” Amendment 27: the refracted length limit, where the deadline was first seen (post hoc)

**Mechanism sensitivity.**
- a bias zeroed every n/(2k) elements replays past the step-n/k wrap where the never-decaying bias breaks, at (8000, 60, 0.5) (Amendment 28): `observations/cells/8000~160~10.5/arms/reset/break/values` all-greater `observations/cells/8000~160~10.5/arms/refracted/break/values` by at least 100, retained in [research/results/runs/memory.tiling/tiling-20261006/results.json](../research/results/runs/memory.tiling/tiling-20261006/results.json) and paired by `run/seeds`.
- a bias zeroed every n/(2k) elements replays past the step-n/k wrap where the never-decaying bias breaks, at (4000, 30, 0.5) (Amendment 28): `observations/cells/4000~130~10.5/arms/reset/break/values` all-greater `observations/cells/4000~130~10.5/arms/refracted/break/values` by at least 100, retained in [research/results/runs/memory.tiling/tiling-20261006/results.json](../research/results/runs/memory.tiling/tiling-20261006/results.json) and paired by `run/seeds`.

**Caveat.** Four cells, p = 0.5, one write strength (theta) and one refraction strength (0.5 beta); the reset is an instantaneous zeroing, not a decay with a time constant. Why replay breaks at the wrap while the one-step transition there is not weak (probe) is not yet measured; n/k = 33 survives its wraps.

## ORDERED-RECALL-BY-TRANSITIONS

**Status.** MEASURED. **Source.** PREREG_ordered_recall_reproduction.md Amendment 2 (OR-1, OR-2, OR-6, OR-7, OR-8 PASS; OR-3, OR-4, OR-5 FAIL)

**Engine / substrate.** numpy_sparse, materialized connectome (the assembly calculus's own sequence_memorize and ordered_recall)

**Claim.** The calculus's sequence operation recalls an eight-element sequence autonomously -- 7 of 7 steps after the cue on 17 of 20 seeds, 6 on the rest -- when each element is written in ONE stimulus-and-recurrence round at a write equal to the convergence threshold theta; eight rounds per element at the same write, and one round at beta = 0.10, advance 0 on every seed. The route is the written bridge, not long-range inhibition (4 to 6 steps with inhibition off).

**Requires.**
- n = 4000, k = 50, p = 0.05, w_max 20, L = 8; LRI period 3, strength 100 set after memorizing; recall from the first stimulus
- MATERIALIZED connectome: on the sampled one recall advances 0 steps (OR-5)

**Evidence.**
- research/notes/sequence/PREREG_ordered_recall_reproduction.md#amendment-2-result-2026-10-06

**Evidence files.**
- [research/results/runs/sequence.ordered-recall-repair/ordered-recall-repair-20261006/results.json](../research/results/runs/sequence.ordered-recall-repair/ordered-recall-repair-20261006/results.json) (artifact) â€” Amendment 2: five arms on twenty seeds; steps after the cue, order, cue retrieval

**Sensitivity gap.** The record keys rows by seed (a mapping), which the register's list paths cannot address; the per-seed contrast it holds -- one round per element 6-7 steps against the diagnostic's construction 0 on every one of 20 seeds at the same write -- is read by ordered_recall_repair.evaluate (OR-7) and pinned by test_ordered_recall_advances_when_each_element_is_written_as_a_transition.

**Caveat.** Not a reproduction of the paper's inhibition-driven mechanism (OR-4 fails; Amendment 1 could not locate that mechanism in either reference). One cell, L = 8. The last step loops back toward the most recent element free of the period-3 inhibition on 8 of 20 seeds (OR-3, post hoc). The write is near theta, where a synapse clips within one or two presentations.

## ADAPTATION-SWITCHES-MEMORY-TYPE

**Status.** MEASURED. **Source.** PREREG_refraction_memory.md Amendment 26 (S1, S2 PASS; S3, S3d FAIL)

**Engine / substrate.** hashed AssemblyMemory (arm B, ungated, masked readout), write_rule round / deferred, refraction strength s

**Claim.** One recurrent area and one causal Hebbian rule store attractors or sequences according to whether the activity holds still while it is written. Online under refraction weaker than plasticity (s = 0.5 beta) the item's rounds converge (consecutive overlap 0.71-0.85) and the area stores attractors; online under refraction stronger than plasticity (s = 1.5 beta), or written after the item, the activity moves every round (overlap <= 0.05) and the area replays the stored trajectory from half of its first state, all seven steps, by itself: 207-950 sequences of eight states at three cells.

**Requires.**
- T = 8 rounds per item, w_max 20, norm_init; replay = frozen masked rounds from half of round 0, each fed the previous round's winners
- write at least ~0.7-1.0 theta: no sequence is stored at 0.5 theta

**Evidence.**
- research/notes/memory/PREREG_refraction_memory.md#amendment-26-result-2026-10-06

**Evidence files.**
- [research/results/runs/memory.sequences/sequences-20261006/results.json](../research/results/runs/memory.sequences/sequences-20261006/results.json) (artifact) â€” Amendment 26: three arms at three cells, 20 brains, replay capacity and during-write overlap
- [research/results/runs/memory.write_rules/write-rules-20261003/results.json](../research/results/runs/memory.write_rules/write-rules-20261003/results.json) (artifact) â€” Amendment 25: the deferred write's one-step trajectory reading
- [research/results/runs/memory.sequence-length/sequence-length-20261006/results.json](../research/results/runs/memory.sequence-length/sequence-length-20261006/results.json) (artifact) â€” Amendment 27: one chosen sequence per brain; Hebbian limit 0.063-0.106 p (n/k)^2, refracted limit set by a break at step n/k

**Mechanism sensitivity.**
- online write at s = 0.5 beta holds the item's rounds, at s = 1.5 beta they move, at (4000, 60, 0.5) (Amendment 26): `observations/cells/4000~160~10.5/arms/round-s0.5/rates/0.07436/own/values` all-greater `observations/cells/4000~160~10.5/arms/round-s1.5/rates/0.52052/own/values` by at least 0.5, retained in [research/results/runs/memory.sequences/sequences-20261006/results.json](../research/results/runs/memory.sequences/sequences-20261006/results.json) and paired by `run/seeds`.
- online write at s = 0.5 beta holds the item's rounds, at s = 1.5 beta they move, at (2000, 60, 0.5) (Amendment 26): `observations/cells/2000~160~10.5/arms/round-s0.5/rates/0.07118/own/values` all-greater `observations/cells/2000~160~10.5/arms/round-s1.5/rates/0.35592/own/values` by at least 0.5, retained in [research/results/runs/memory.sequences/sequences-20261006/results.json](../research/results/runs/memory.sequences/sequences-20261006/results.json) and paired by `run/seeds`.
- deferred write replays the stored trajectory at 1.4 theta, not at 0.5 theta, at (4000, 60, 0.5), 32 items (Amendment 26): `observations/cells/4000~160~10.5/arms/deferred-s0.5/rates/0.52052/ensembles/32/replay/values` all-greater `observations/cells/4000~160~10.5/arms/deferred-s0.5/rates/0.1859/ensembles/32/replay/values` by at least 0.5, retained in [research/results/runs/memory.sequences/sequences-20261006/results.json](../research/results/runs/memory.sequences/sequences-20261006/results.json) and paired by `run/seeds`.

**Caveat.** Sequence capacity follows neither n/k nor the in-degree alone (S3, S3d fail); three cells give C ~ d^1.05 (n/k)^0.75 exactly, untested. The replay criterion is overlap with the item's own state, not distinct against other items' -- a distinctness read is owed. The control and the online arm differ in rate as well as s. All cells p = 0.5, T = 8.

## WRITE-TIMING-DECIDES-ATTRACTOR

**Status.** MEASURED. **Source.** PREREG_refraction_memory.md Amendment 25 (CV, W1-W4 PASS)

**Engine / substrate.** hashed AssemblyMemory (refracted, arm B, ungated), write_rule round / online_burst / deferred / burst

**Claim.** In the refracted assembly memory WHEN the write happens decides what is stored. The round write, whose counts feed into the item's next round, converges the item onto one assembly and stores attractors (451-1442 items). The same counts written after the item store no attractor at any of 21 rates over 0.1-3.2 theta: refraction relocates the unwritten rounds every round, and what is stored is the item's TRAJECTORY (one frozen round from half of round t recovers 64-79% of round t+1). A burst-timing rule (deferred, symmetric, between neurons that fired twice) and a burst-gated online write store nothing.

**Requires.**
- refraction 0.5 beta charged every round, T = 8, w_max 20, norm_init; half-cue masked readout
- burst = fired in at least 2 of the item's 8 rounds

**Evidence.**
- research/notes/memory/PREREG_refraction_memory.md#amendment-25-result-2026-10-03

**Evidence files.**
- [research/results/runs/memory.write_rules/write-rules-20261003/results.json](../research/results/runs/memory.write_rules/write-rules-20261003/results.json) (artifact) â€” Amendment 25: four write rules at three cells, 20 brains, capacity and trajectory readings

**Mechanism sensitivity.**
- deferred write: one round from round t recovers round t+1, not round t at (2000, 60, 0.5) (Amendment 25): `observations/cells/2000~160~10.5/rules/deferred/trajectory/next/values` all-greater `observations/cells/2000~160~10.5/rules/deferred/trajectory/same/values` by at least 0.5, retained in [research/results/runs/memory.write_rules/write-rules-20261003/results.json](../research/results/runs/memory.write_rules/write-rules-20261003/results.json) and paired by `run/seeds`.
- deferred write: one round from round t recovers round t+1, not round t at (4000, 20, 0.5) (Amendment 25): `observations/cells/4000~120~10.5/rules/deferred/trajectory/next/values` all-greater `observations/cells/4000~120~10.5/rules/deferred/trajectory/same/values` by at least 0.5, retained in [research/results/runs/memory.write_rules/write-rules-20261003/results.json](../research/results/runs/memory.write_rules/write-rules-20261003/results.json) and paired by `run/seeds`.
- online write holds the item's rounds together, the deferred write does not at (4000, 160, 0.5) (Amendment 25): `observations/cells/4000~1160~10.5/rules/round/trajectory/own/values` all-greater `observations/cells/4000~1160~10.5/rules/deferred/trajectory/own/values` by at least 0.5, retained in [research/results/runs/memory.write_rules/write-rules-20261003/results.json](../research/results/runs/memory.write_rules/write-rules-20261003/results.json) and paired by `run/seeds`.

**Caveat.** Three cells, all p = 0.5; the trajectory reading is at one rate (1.0 theta). Refraction's strength is tied to the write's (0.5 beta), so a burst rule with a decoupled or weaker refraction -- or a window short against the relocation period -- is untested. Developing retinogeniculate BTDP refines maps rather than storing items; the claim is about item storage in this circuit.

## REFRACTION-CANCELS-CONVERGENCE

**Status.** MEASURED. **Source.** This repository; PREREG_refraction_capacity.md.

**Engine / substrate.** hashed_assembly_memory (HashedArea with AreaFiber/StimulusFiber); the registered twenty-brain run is a runner artifact, the wander diagnostic and the bias-readout numbers are logs

**Claim.** [MEASURED 2026-09-13 on two disjoint blocks of twenty registered brains, PREREG_refraction_convergence.md: a refracted RECURRENT area below the transition does not converge and hold. It forms, then RELOCATES ON A FIXED PERIOD equal to the clip arithmetic ln(w_max)/ln(1+beta) + (1-1/w_max)/beta = 40.93 rounds -- measured mean spacing 41.60 and 41.52 on the two blocks (1.6% and 1.4% error against a registered 10% tolerance), with the first relocation on round 42 on ALL FORTY brains and no spread, because the deadline is set by the potentiation schedule and not by the connectome. The formula is a LAW AT s = 0.5 beta, the adopted operating strength, and ONLY there: a four-arm strength sweep at one operating point measured 59.49, 47.15, 41.60 and 40.78 rounds at s/beta = 0.25, 0.375, 0.5 and 0.625, a 25.9% spread, so the period depends strongly on strength and the formula carries no strength term. A 1/2s erosion term, which the mechanism suggests and which coincides with the written form exactly at 0.5 beta, is closer in three arms of four and still wrong by 16.6% at the slowest; NEITHER closed form is adopted and the dependence stands as four measured points. Capacity and tenure part company here: strength is a capacity SWITCH with one plateau over 0.3-0.6 beta, yet tenure moves 13% between 0.375 and 0.5 beta inside it. At the adopted strength the law was swept across beta in {0.05, 0.10, 0.20} and w_max in {5, 20, 100} on twenty brains per cell it holds in every cell to 4.1%, with the measured w_max ratio 2.354 against a predicted 2.34. The per-brain intervals are 0.1 rounds wide, tight enough to show the plain form is an approximation: it falls OUTSIDE the interval in four of five cells. DISCRETISING it, ceil(ln(w_max)/ln(1+beta)) + (1-1/w_max)/beta -- a weight needing 31.43 rounds of growth clips on round 32 -- cuts the mean error from 4.05% to 0.78% on three further cells chosen before the run and sharing no coordinate with the first five, closer in 3 of 3. Neither form lands inside an interval: the measurement is sharper than either approximation. For small beta the period is within half a percent of (ln w_max + 1)/beta, so TENURE IS BOUGHT WITH BETA AND NOT WITH THE WEIGHT CEILING: a fourfold beta change moved the period by 3.6, a twentyfold w_max change by 2.4. The unrefracted control had ZERO relocations on every brain of every cell though its weights clip on the same schedule, so the clip alone does not move an assembly. Between relocations the winners are unchanged: 5 relocations of 6-10 rounds each in 240, stable fraction 0.80-0.84. A FEEDFORWARD area at s = beta does the same more often (11-32 events) rather than holding. At s >= 0.8 beta there is no stability to relocate from: one event covers the whole run, stable fraction 0.000, fill 1.000. The 0.7 arm is a third regime, churning for about half the run in one 97-120 round event and restabilising, so the transition lies between 0.5 and 0.7 beta. The earlier readings 'converges and holds' and 'feedforward holds' were a shorter window: their bars failed and are retained.] [RE-MEASURED 2026-09-04 with the selector fixed (1b475fc): the churn above ~0.75 beta stands; the intermediate-strength rows were a selector artefact -- at 0.5 beta the recurrent assembly converges, relocates once when the clip binds (~round 40-60, the registered P2 prediction) and holds; at 0.7 beta most brains no longer converge. The transition lies in 0.5-0.7 beta. And BELOW it, with the bias-MASKED readout, a refracted recurrent area at 0.5 beta stored every assembly of the grid (rank-1 1.000 to M = 32, above the bar to 256, pairwise 0.00 x chance) against a Hebbian ceiling of 23.5 -- capacity becomes FILL-limited far above the interference limit; the earlier 'spends the substrate' reading was the defect. Post hoc; a registration is owed. PREREG_refraction_capacity.md.] Refraction at strength s is the anti-Hebbian counterweight on a neuron's own repeated input: raw*(1+beta)^t minus the charged bias leaves net drive growing by (beta - s)*raw per win, so at s = beta it is CONSTANT. A feedforward area needs no convergence force (its input ranking is fixed) and holds; a RECURRENT assembly converges only through rich-get-richer, and above s ~ 0.75 beta it never converges and churns through the whole area -- refraction there is a firing-rate equalizer, and firing-rate homeostasis is incompatible with attractor memory in a recurrent k-WTA area. Below the transition it is the anti-merging force of [[REFRACTION-ANTI-MERGING]]: ~25x the Hebbian ceiling, read with the bias masked.

**Requires.**
- recurrent area; strength quoted relative to beta; T=8 rounds per item in the capacity protocol
- the reference uses RefractedArea only as a FEEDFORWARD conjunction area driven by its full input at recall, where none of this applies

**Evidence.**
- period-law-strength-20260913 (four strengths at w_max 20, beta 0.10, twenty brains each): 59.49 [59.46, 59.51], 47.15 [47.10, 47.20], 41.60 [41.54, 41.66], 40.78 [38.98, 42.58] at s/beta 0.25, 0.375, 0.5, 0.625; ST-1, ST-2 and ST-3 all FAIL, so the period is neither strength-independent nor of the 1/2s form
- period-law-amendment2-20260913 (three cells unseen by the refinement, twenty brains each): measured 13.57, 20.83, 30.23 at (w_max, beta) = (8, 0.25), (12, 0.18), (25, 0.15) against plain 12.82, 20.11, 29.43 and discretised 13.50, 21.09, 30.40; mean relative error 4.05% plain against 0.78% discretised; zero control relocations
- period-law-v2-20260913 (five cells, twenty brains each, per-brain spacing with intervals): 22.05 [22.02, 22.08], 41.60 [41.54, 41.66], 80.05 [78.66, 81.43], 25.06 [25.01, 25.10], 58.98 [58.93, 59.03]
- period-law-20260913 (the same five cells, pooled spacing superseded by the per-brain statistic): measured mean spacing 22.05, 41.60, 79.92 at beta 0.20, 0.10, 0.05 (w_max 20) and 25.06, 41.60, 58.98 at w_max 5, 20, 100 (beta 0.10), against predictions 21.18, 40.93, 80.40, 24.89, 58.22; one doublet in 681 spacings; control relocations 0 of 0 in every cell with stable fraction 1.000
- refraction-convergence-fresh-20260913 (the confirmatory block, seeds 62-81, protocol version 3): all ten Amendment 2 bars pass -- 5 relocations per brain, first at round 42, spacings 40-43 (mean 41.52 against the predicted 40.93), relocation length 6-10 rounds, stable fraction 0.801-0.837; feedforward 11-32 events at 0.796-0.860; 0.7 beta 0.249-0.290 with one event of 97-120 rounds
- refraction-convergence-20260913 (twenty brains, seeds 42-61, 240 one-round episodes, seven arms): late consecutive overlap control 1.000; feedforward 0.912-0.975; 0.5 beta 0.904-0.932 (conv 217-220 on every brain, fill 0.28); 0.7 beta 0.632-0.966 (fill 0.91); 0.8 beta <= 0.277; 0.9 and 1.0 beta <= 0.010 (fill 1.000)
- seq_refraction_wander.py at n=4000 k=100 p=0.5 beta=0.1 w_max=20, 16 brains, 240 rounds: s/beta = 0.5, 0.7 converge (rounds 48, 45 vs control 4; late stability 1.000); 0.8, 0.9, 0.95, 1.0 never converge (late stability <= 0.22, fill 1.000); feedforward at s = beta holds (late 0.993)
- the w_max saturation arithmetic ln(w_max)/ln(1+beta) + (1-1/w_max)/beta ~ 41 appears as a transient re-ranking at rounds 44-48 below the transition, which the assembly survives
- capacity protocol at s = 0.5 beta, re-measured with the selector fixed: see [[REFRACTION-ANTI-MERGING]] (the earlier 'M* 19.6 vs 23.5' reading was the defect)
- bias-on partial-cue recall 0.250 vs bias-masked 0.984 at M=8, same training: the intrinsic bias vetoes recall from a partial cue, as the identity predicts

**Evidence files.**
- [research/results/runs/memory.refraction-convergence/refraction-convergence-20260913/results.json](../research/results/runs/memory.refraction-convergence/refraction-convergence-20260913/results.json) (artifact)
- [research/results/runs/memory.refraction-convergence/refraction-convergence-v2-20260913/results.json](../research/results/runs/memory.refraction-convergence/refraction-convergence-v2-20260913/results.json) (artifact)
- [research/results/runs/memory.refraction-convergence/refraction-convergence-fresh-20260913/results.json](../research/results/runs/memory.refraction-convergence/refraction-convergence-fresh-20260913/results.json) (artifact)
- [research/results/runs/memory.refraction-period-law/period-law-20260913/results.json](../research/results/runs/memory.refraction-period-law/period-law-20260913/results.json) (artifact)
- [research/results/runs/memory.refraction-period-law/period-law-v2-20260913/results.json](../research/results/runs/memory.refraction-period-law/period-law-v2-20260913/results.json) (artifact)
- [research/results/runs/memory.refraction-period-law/period-law-amendment2-20260913/results.json](../research/results/runs/memory.refraction-period-law/period-law-amendment2-20260913/results.json) (artifact)
- [research/results/runs/memory.refraction-period-law/period-law-strength-20260913/results.json](../research/results/runs/memory.refraction-period-law/period-law-strength-20260913/results.json) (artifact)
- [research/notes/memory/PREREG_refraction_period_law.md](../research/notes/memory/PREREG_refraction_period_law.md) (registration)
- [research/experiments/refraction_period_law.py](../research/experiments/refraction_period_law.py) (producer)
- [research/notes/memory/PREREG_refraction_convergence.md](../research/notes/memory/PREREG_refraction_convergence.md) (registration)
- [research/experiments/refraction_convergence.py](../research/experiments/refraction_convergence.py) (producer)
- [research/experiments/seq_refraction_wander.py](../research/experiments/seq_refraction_wander.py) (producer)
- [research/notes/memory/PREREG_refraction_capacity.md](../research/notes/memory/PREREG_refraction_capacity.md) (registration)
- [research/notes/memory/AUDIT_refraction_scaling.md](../research/notes/memory/AUDIT_refraction_scaling.md) (analysis)

**Provenance gap.** the wander diagnostic's and the bias-readout numbers have no immutable result artifact; the convergence contrast now does (refraction-convergence-20260913)

**Mechanism sensitivity.**
- refraction strength 0.5 beta against beta on the same recurrent area (n 4000, k 100, p 0.5, beta 0.10, w_max 20), late consecutive-round overlap over rounds 200 to 240: 0.904 to 0.932 against 0.003 to 0.007 on every paired brain; the s = beta area reshuffles its winners every round: `observations/arms/s0.5/rows/*/late` all-greater `observations/arms/s1.0/rows/*/late` by at least 0.5, retained in [research/results/runs/memory.refraction-convergence/refraction-convergence-20260913/results.json](../research/results/runs/memory.refraction-convergence/refraction-convergence-20260913/results.json) and paired by `run/seeds`.
- refraction on against off at the same operating point and the same weight clip: 5 relocations on every refracted brain against 0 on every control brain, so the clip alone does not move an assembly and the bias charging past it is the whole mechanism: `observations/cells/w20b0.1/arms/refracted/rows/*/n_relocations` all-greater `observations/cells/w20b0.1/arms/control/rows/*/n_relocations` by at least 3, retained in [research/results/runs/memory.refraction-period-law/period-law-20260913/results.json](../research/results/runs/memory.refraction-period-law/period-law-20260913/results.json) and paired by `run/seeds`.
- on the fresh seed block, the fraction of rounds 20 to 240 whose winners are unchanged: 0.801 to 0.837 at 0.5 beta against 0.000 on every brain at 0.8 beta -- relocation and churn separated per brain, not by an average: `observations/arms/s0.5/rows/*/stable_fraction` all-greater `observations/arms/s0.8/rows/*/stable_fraction` by at least 0.7, retained in [research/results/runs/memory.refraction-convergence/refraction-convergence-fresh-20260913/results.json](../research/results/runs/memory.refraction-convergence/refraction-convergence-fresh-20260913/results.json) and paired by `run/seeds`.

**Sensitivity gap.** The retained check is the strength contrast on the convergence protocol; the bias-masked capacity numbers below the transition remain logs (PREREG_refraction_capacity.md).

**Used by.** `neural_assemblies/core/_homeostasis.py`; `neural_assemblies/core/torch_engine/_hashed.py`

**Caveat.** The critical ratio is bracketed in (0.5, 0.7) at one operating point on forty brains (the earlier (0.7, 0.8) came from the 16-brain diagnostic); a transient-handicap estimate gives ~2/3. The period law holds across EIGHT cells (beta 0.05-0.25, w_max 5-100) in its discretised form AT s = 0.5 beta ONLY: a strength sweep refuted both the strength-independent form and the 1/2s form, and the dependence on strength stands as four measured points with no adopted closed form. The 0.625 arm sits near the churn transition and carries an interval 70x wider than the others, so it is the least trustworthy of the four. Everything is at ONE density and ONE refraction strength: the formula carries no strength term and strength was not varied, so the period's independence of s is suggested by the algebra and NOT measured. Three bars of Amendment 1 failed on mis-set thresholds (formation counted as an event; relocations run ~8 rounds, not 2-4) and are retained. Whether REFRACTION-NEEDS-LOAD's under-loaded non-convergence is this mechanism (the arc's state input is itself changing) is suggested, not established. TWO FURTHER LIMITS (AUDIT_refraction_scaling.md): refraction + synaptic scaling on the same area is INCOMPATIBLE -- a feedforward arc that holds under either alone loses its assemblies within ~10 presentations under both (late stability 0.12 vs 1.00), because scaling moves the raw drive the bias is charged against; and with no clip the identity dies of float32 cancellation at ~100-150 wins. Never scale a refracted area.

## AC-CAP

**Status.** MEASURED. **Source.** This repository (critical-load measurement).

**Engine / substrate.** original capacity-note run: incompletely recorded (graded-similarity evidence compares explicit, materialized and sampled numpy_sparse); retained bracket: hashed_assembly_memory, the unrefracted Hebbian control arm of the paired capacity replay at (n, k) = (4000, 60)

**Claim.** Assembly capacity is EXTENSIVE: about M_max ~ 1.15 n/k distinct assemblies per area.

**Evidence.**
- research/notes/categories/capacity_is_not_the_constraint_separation_is.md
- research/notes/substrate/graded_similarity_and_sampler_load.md
- refraction-paired-sensitivity-20260911, control arm (4000, 60), 20 brains: rank-1 recall 1.000 on every brain at M = 32 and at M = 64 on 16 of 20 (mean 0.81, min 0.03); at M = 128 the maximum over brains is 0.031

**Evidence files.**
- [research/notes/categories/capacity_is_not_the_constraint_separation_is.md](../research/notes/categories/capacity_is_not_the_constraint_separation_is.md) (analysis)
- [research/notes/substrate/graded_similarity_and_sampler_load.md](../research/notes/substrate/graded_similarity_and_sampler_load.md) (analysis)
- [research/results/runs/memory.capacity-scaling/refraction-paired-sensitivity-20260911/results.json](../research/results/runs/memory.capacity-scaling/refraction-paired-sensitivity-20260911/results.json) (artifact) â€” Hebbian control arm only; brackets the extensive ceiling at one (n, k) cell

**Provenance gap.** capacity-note run has no identifiable immutable artifact

**Mechanism sensitivity.**
- Hebbian control arm: half-cue rank-1 recall at M = 32 (0.48 n/k, below the extensive ceiling) against M = 128 (1.9 n/k, above it); the ceiling lies between them on every brain: `observations/conditions/control/cells/B~14000~160/checkpoints/32/rank1` all-greater `observations/conditions/control/cells/B~14000~160/checkpoints/128/rank1` by at least 0.9, retained in [research/results/runs/memory.capacity-scaling/refraction-paired-sensitivity-20260911/results.json](../research/results/runs/memory.capacity-scaling/refraction-paired-sensitivity-20260911/results.json) and paired by `run/seeds`.

**Sensitivity gap.** The retained bracket shows the Hebbian ceiling lies between 0.48 and 1.9 n/k on twenty brains at one cell; it does not pin the 1.15 constant, and the original capacity-note run remains unidentified.

**Caveat.** The capacity-note run's engine provenance remains unresolved; this entry does not certify the 1.15 constant. What the retained control arm certifies is the extensive form at one cell: the Hebbian area holds every item at half n/k and none at twice n/k. This is why k and p are not interchangeable routes to a regime: raising k to reach kp spends capacity and forces n up with it.

## RATE-HETEROGENEITY

**Status.** MEASURED. **Source.** This repository.

**Engine / substrate.** numpy_explicit (materialized copied-fiber protocol)

**Claim.** Learning rate is settable PER FIBER: copied fibers with identical pre/post activity at beta 0.06 and 0.005 diverge 14.35-fold after 50 updates without reaching the weight clip.

**Evidence.**
- 20 seeds: fast geometric mean 18.4201012, slow 1.2832253, ratio 14.3545340; swapped labels reproduce the ratio, equal-beta null is 1.0, and maximum weight 18.4201012 is below w_max=20

**Evidence files.**
- [research/notes/memory/PREREG_per_fiber_plasticity.md](../research/notes/memory/PREREG_per_fiber_plasticity.md) (registration)
- [research/experiments/per_fiber_plasticity.py](../research/experiments/per_fiber_plasticity.py) (producer)
- [research/results/runs/mechanism.per-fiber-plasticity/per-fiber-plasticity-20260910/results.json](../research/results/runs/mechanism.per-fiber-plasticity/per-fiber-plasticity-20260910/results.json) (artifact)

**Mechanism sensitivity.**
- per-fiber rate contrast versus equal-rate null: `observations/rows/*/cells/forward/ratio` all-greater `observations/rows/*/cells/equal/ratio` by at least 9, retained in [research/results/runs/mechanism.per-fiber-plasticity/per-fiber-plasticity-20260910/results.json](../research/results/runs/mechanism.per-fiber-plasticity/per-fiber-plasticity-20260910/results.json) and paired by `observations/rows/*/seed`.

**Used by.** `neural_assemblies.core.brain.Brain.update_plasticity`; `research.experiments.per_fiber_plasticity`

**Caveat.** This isolates the multiplicative rate on copied materialized NumPy fibers; it does not measure assembly quality or biological heterogeneity, and it does not establish sampled or hashed backend equivalence. Near-zero intervals reflect the controlled arithmetic, not population certainty. The old saturated beta=0.5 number remains unreproduced and is not evidence for this entry.

## DUAL-RATE

**Status.** EXTENSION. **Source.** Proposed here; not proved and not measured.

**Claim.** Running fast and slow pathways at once is FUNCTIONALLY useful: a high-beta fiber binds in one shot (episodic) while a low-beta fiber accumulates statistics (semantic), and a system with both does something neither does alone.

**Requires.**
- [[RATE-HETEROGENEITY]] -- the mechanism exists
- [[SEQ-BETA-WINDOW]] -- each rate must sit inside its own window, and the windows may not overlap

**Caveat.** UNTESTED. That the knob exists is measured; that turning it buys anything is not. Two known tensions to design against: beta trades capacity against depth (low favours capacity, high favours depth), and the afferent count flips the SIGN of beta's effect, so a rate that helps one fiber can hurt another at a different kp. Falsified if a dual-rate organ matches the better of its two single-rate controls.

## SEQ-ORGAN-EMBEDS

**Status.** MEASURED. **Source.** This repository.

**Engine / substrate.** numpy_sparse, arc materialized (retained runner replay, 10 seeds); the original organ-density experiment ran on the sampled arc

**Claim.** A sequence organ runs at its own regime INSIDE a brain whose ambient density is far lower, given per-fiber p.

**Requires.**
- per-fiber p (`Brain.add_connectivity`, numpy_exact and numpy_sparse)
- EVERY per-fiber quantity scaled by the fiber's p, not the global one -- see the caveat

**Evidence.**
- research/experiments/seq_a1_local_regime.py: ambient p=0.05 with organ fibers at p=0.4 gives 10/10 correct trajectories, matching the uniform p=0.4 result, while the same organ left at the ambient density gives 0/10

**Evidence files.**
- [research/results/sequence/seq_a1_local_regime_results.json](../research/results/sequence/seq_a1_local_regime_results.json) (artifact) â€” sampled arc; sequence verdict void
- [research/results/sequence/seq_a1_local_regime_results_materialized.json](../research/results/sequence/seq_a1_local_regime_results_materialized.json) (artifact)
- [research/results/runs/sequence.a1-local-regime/sens-local-regime-v3-20260912/results.json](../research/results/runs/sequence.a1-local-regime/sens-local-regime-v3-20260912/results.json) (artifact) â€” shared-runner replay with source archive; the ambient-only arm is the disabled-organ null
- [research/experiments/seq_a1_local_regime.py](../research/experiments/seq_a1_local_regime.py) (producer)

**Provenance gap.** the two legacy artifacts lack runner records and source archives; the 2026-09-12 replay has both

**Mechanism sensitivity.**
- organ fibers at p = 0.4 inside an ambient p = 0.05 brain against the same organ left at the ambient density (kp 3.5 against a floor of 18.6), arc materialized: `observations/arms/0/rows/*/both_correct` all-greater `observations/arms/1/rows/*/both_correct` by at least 1, retained in [research/results/runs/sequence.a1-local-regime/sens-local-regime-v3-20260912/results.json](../research/results/runs/sequence.a1-local-regime/sens-local-regime-v3-20260912/results.json) and paired by `observations/arms/0/rows/*/seed`.

**Caveat.** The original organ-density experiment used the sampled numpy arc; its sequence-dynamics numbers are void under PREREG_sampler_audit.md until reproduced materialized or hashed. The heterogeneous path found a defect that the regime audit could NOT see: the stimulus weight clamp was scaled by the global p while the weights were drawn at the fiber's p, so a dense fiber in a sparse brain saturated at a sparse ceiling. The organ read 1/10 while every area reported comfortably in-regime. Any NEW per-fiber quantity is a candidate for the same class. The pooled candidate draw also remains moment-matched -- exact in the first two moments, an approximation beyond them.

## SEQ-STATE-CODE-EMERGENT

**Status.** EXTENSION. **Source.** Not proved and not measured, here or in the source paper.

**Claim.** The state alphabet can be INDUCED from data rather than assigned.

**Caveat.** THE LOAD-BEARING GAP. Both our FSM and the reference assign state assemblies as disjoint blocks by hand. Every result under [[SEQ-FSM]] and [[SEQ-EXACT-RECOVERY]] is therefore about running a machine over a GIVEN alphabet, not about discovering one. Language needs the latter.

## KWTA-TIE-FRAGILE

**Status.** MEASURED. **Source.** The base drive is a Bernoulli COUNT, hence integer-valued, so exact ties are the common case rather than an edge case. Measured 5-18 columns tied at the bar per brain at n=4000-50000 (research/experiments/gpu_radix_select_prototype.py). An ulp-level change to a summation once moved sixteen cells of an exact table with no direction to it.

**Engine / substrate.** tie census: numpy_explicit (retained runner study, 20 seeds, PREREG_kwta_tie_fragility.md); the last-bit split under a change of arithmetic: torch/CUDA selector prototype (no retained artifact) and the fused-kernel gate of 2026-09-10

**Claim.** The k-WTA bar is routinely TIED, so anything that perturbs the drive in its last bits -- a change of summation order, of arithmetic, or of tie-break policy -- can change WHICH neurons fire, not merely their order.

**Requires.**
- integer or near-integer drive, i.e. before heavy potentiation spreads the values
- k-WTA selecting at a bar that several columns reach

**Evidence.**
- research/experiments/gpu_radix_select_prototype.py
- kwta-study-20260912 (numpy_explicit, 20 seeds): candidates tied at the bar of the untrained stimulus drive 26-73 per seed (KT-1 PASS); with in-degree normalization 1-12, at least 2 on 17 of 20 (KT-3 PASS); a one-ulp raise of a tied outsider changes the winner set on 19 of 20 seeds and never on the jittered drive (KT-2 as registered FAIL: seed 49 has 19 candidates above the bar and 31 at it, so all 50 are inside k and no outsider is tied); float32 division, float32 reciprocal and float64 division select the same winners on every seed

**Evidence files.**
- [research/experiments/gpu_radix_select_prototype.py](../research/experiments/gpu_radix_select_prototype.py) (producer) â€” the split half; no retained artifact
- [research/results/runs/substrate.kwta-tie-fragility/kwta-study-20260912/results.json](../research/results/runs/substrate.kwta-tie-fragility/kwta-study-20260912/results.json) (artifact)
- [research/notes/substrate/PREREG_kwta_tie_fragility.md](../research/notes/substrate/PREREG_kwta_tie_fragility.md) (registration)
- [research/experiments/kwta_tie_fragility.py](../research/experiments/kwta_tie_fragility.py) (producer)

**Provenance gap.** the selector prototype's split measurement has no retained raw artifact; the tie census does

**Mechanism sensitivity.**
- candidates tied at the k-th drive of the untrained 0/1 stimulus fiber (n = 1000, k = 50, p = 0.1) against the same drive with a deterministic per-neuron jitter that leaves exactly one at the bar: `observations/rows/*/ties_raw` all-greater `observations/rows/*/ties_jittered` by at least 4, retained in [research/results/runs/substrate.kwta-tie-fragility/kwta-study-20260912/results.json](../research/results/runs/substrate.kwta-tie-fragility/kwta-study-20260912/results.json) and paired by `observations/rows/*/seed`.

**Sensitivity gap.** The retained check covers the counting half (the bar is tied on every seed, 26 to 73 candidates). The other half, that a change of arithmetic splits those ties into different winners, is retained only as the GPU gate's log and the selector prototype: on the CPU explicit engine's own float32 reciprocal and division paths no winner differed on any of 20 seeds, and the one-ulp construction flipped the winner set on 19 of 20 (the twentieth is the exact-fit case, every tied candidate already inside the top k).

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

**Evidence files.**
- [research/results/runs/memory.capacity-scaling/capacity-record-consumed-20260910/results.json](../research/results/runs/memory.capacity-scaling/capacity-record-consumed-20260910/results.json) (artifact)
- [research/results/runs/memory.capacity-scaling/nk-replay-20260913/results.json](../research/results/runs/memory.capacity-scaling/nk-replay-20260913/results.json) (artifact)
- [research/notes/memory/PREREG_capacity_nk_law.md](../research/notes/memory/PREREG_capacity_nk_law.md) (registration)
- [research/experiments/seq_capacity_scaling.py](../research/experiments/seq_capacity_scaling.py) (producer)

**Mechanism sensitivity.**
- the same n = 4000 at k = 30 against k = 60 (ratio 133 against 67), unrefracted, at M = 128: every k-30 brain still recalls (at least 0.938) where every k-60 brain has collapsed (at most 0.031); the ceilings are 267.0 and 74.7, ratio 3.58 against the law's 4: `observations/cells/1/checkpoints/128/rank1` all-greater `observations/cells/0/checkpoints/128/rank1` by at least 0.9, retained in [research/results/runs/memory.capacity-scaling/nk-replay-20260913/results.json](../research/results/runs/memory.capacity-scaling/nk-replay-20260913/results.json) and paired by `run/seeds`.

**Sensitivity gap.** The retained check is the ratio contrast at one n; the equal-ratio cell at n = 8000 in the same run reproduces the n = 4000 ceiling within 3.8% but is a cliff interpolation (no grid point inside its transition band), and the n = 16000 cell of CS1 has no run record.

**Caveat.** This result survived a WRONG RETRACTION: an intermediate CSR deviation store applied the potentiation table per count FRAGMENT, and (tab[c0]-1)+(tab[c1]-1) != tab[c0+c1]-1, inflating ceilings ~25% (88.3/92.1/83.2); those numbers were briefly recorded here as the verified ones. Multi-episode ENGINE parity exposed it (rel 2e-3) and the exact path lands back on the first run's values. The EXPONENT remains unestablished: b = 2.19 +/- 0.05, above 2, NO mechanism -- reported, never quoted. A synapse bound M ~ n^2 p / (k ln(n/k)) is REFUTED by CS1: it is not a function of n/k alone -- and so is plain second-order CROSSTALK, refuted by registered test: the crosstalk ratio cancels p and beta, but M* ~ p^-0.6 and ~ beta^-1.3 at fixed n/k. No mechanism is adopted. Operational rule from the wrong retraction: when two implementations disagree, a test they BOTH pass verifies neither -- arbitrate with engine parity on the DRIVE.

## CAP-ANCHOR-RATIO

**Status.** MEASURED. **Source.** This repository; PREREG_formation_interference.md (F2) and PREREG_anchor_ratio.md (A1-A3).

**Engine / substrate.** hashed_assembly_memory, capacity-scaling protocol (retained paired-anchor replay of 20 brains; the original excursions are logs without run records)

**Claim.** The capacity ceiling is set at FORMATION by the ratio of the stimulus anchor to the trained recurrent pull. Density p and gain beta enter through that ratio, so an excursion in either is undone by a computed change in anchor size.

**Requires.**
- n=4000, k=100, T=8, w_max=20, arm B, 16 brains; one operating-point neighbourhood
- exponents s^1.52 p^-0.6 beta^-1.3 are two- and three-point fits used ONLY to place cells; none is adopted

**Evidence.**
- F2: anchor 100 -> 200 lifts M* 23.5 -> 67.1 (2.86x), while retrieval never sees the stimulus
- A1-A3: beta 0.20 / p 0.7 / p 0.3 excursions with uncompensated M* 8.0 / 17.6 / 28.8 land at 29.4 / 25.0 / 25.3 once the anchor is set from the measured exponents -- a 3.6x spread collapses to 1.18x, all within 25% of 23.5
- F3: past the cliff, erosion is retroactive and diffuse (early items fail worst), not one-shot capture
- anchor-pair-20260912 (20 brains, half-cue rank-1 readout, PREREG_anchor_ratio.md sensitivity replay, AN-1 to AN-3 PASS): M* 23.3 at anchor 100 against 70.5 at anchor 200, ratio 3.02 (the law's (200/100)^1.52 = 2.87); at M = 48 every anchor-200 brain recalls 1.000 while anchor-100 is at most 0.03

**Evidence files.**
- [research/notes/memory/PREREG_anchor_ratio.md](../research/notes/memory/PREREG_anchor_ratio.md) (registration)
- [research/notes/memory/PREREG_formation_interference.md](../research/notes/memory/PREREG_formation_interference.md) (registration)
- [research/results/runs/memory.capacity-scaling/anchor-pair-20260912/results.json](../research/results/runs/memory.capacity-scaling/anchor-pair-20260912/results.json) (artifact) â€” paired anchors 100 and 200 at one cell; not the p and beta excursions
- [research/experiments/seq_capacity_scaling.py](../research/experiments/seq_capacity_scaling.py) (producer)

**Provenance gap.** the A1-A3 excursion outcomes are logs without a run record; the anchor contrast itself has one

**Mechanism sensitivity.**
- stimulus anchor 200 against 100 at (4000, 100), unrefracted, at M = 48: the largest checkpoint every anchor-200 brain still recalls (1.000), where anchor-100 has collapsed (at most 0.03): `observations/conditions/anchor_high/cells/B~14000~1100/checkpoints/48/rank1` all-greater `observations/conditions/anchor_low/cells/B~14000~1100/checkpoints/48/rank1` by at least 0.3, retained in [research/results/runs/memory.capacity-scaling/anchor-pair-20260912/results.json](../research/results/runs/memory.capacity-scaling/anchor-pair-20260912/results.json) and paired by `run/seeds`.

**Sensitivity gap.** The retained check is the anchor contrast at one cell; the p and beta excursions (A1-A3) and their computed compensation remain logs without a run record.

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

**Evidence files.**
- [research/results/runs/memory.capacity-scaling/capacity-record-consumed-20260910/results.json](../research/results/runs/memory.capacity-scaling/capacity-record-consumed-20260910/results.json) (artifact) â€” maintained-run replay at (4000,60), not the registered cliff cell
- [research/results/runs/memory.capacity-scaling/capacity-cliff-sensitivity-20260911/results.json](../research/results/runs/memory.capacity-scaling/capacity-cliff-sensitivity-20260911/results.json) (artifact) â€” fresh exact-path reproduction and retained sensitivity vectors
- [research/experiments/seq_capacity_scaling.py](../research/experiments/seq_capacity_scaling.py) (producer)

**Mechanism sensitivity.**
- safe-load versus overloaded capacity readout: `observations/cells/*/checkpoints/192/rank1` all-greater `observations/cells/*/checkpoints/512/rank1` by at least 0.5, retained in [research/results/runs/memory.capacity-scaling/capacity-cliff-sensitivity-20260911/results.json](../research/results/runs/memory.capacity-scaling/capacity-cliff-sensitivity-20260911/results.json) and paired by `observations/cells/*/seeds`.

**Caveat.** There is no soft capacity margin to trade against: a design must know where the ceiling is and stay under it. Sharing itself is healthy -- at M=256 the load Mk/n is 1.9, nearly two assemblies per neuron, with overlap still 1.37x chance -- so the cliff is not caused by sharing. The transition occupies roughly one 1.5x step in M (256 -> 384). A 20-brain maintained-path reproduction gave 0.909 / 0.433 / 0.0188 at those three checkpoints; its interpolated M* = 310.1 remains fill-censored.

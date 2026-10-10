"""The sequence organ, measured in this repository: arcs, state collision, recovery, carry.

Moved from neural_assemblies/theory.py unchanged; theory.py lists these in its order."""
from __future__ import annotations

from typing import List

from .types import EvidenceRef, Result, SensitivityCheck, Status

CLAIMS: List[Result] = [
    # ------------------------------------------------- measured in this repo
    Result(
        id="SEQ-INTEGER-ARC-LOAD",
        engine="hashed_arc_fsm (HashedArcFSM), 20 brains a cell, chain-order training",
        status=Status.MEASURED,
        sensitivity_checks=(SensitivityCheck(
            artifact="research/results/runs/sequence.autonomous-chain/chain-iload-20260913/results.json",
            sample_path="observations/arms/iload-L336/rows/*/seed",
            treatment_path="observations/arms/iload-L336/rows/*/correct",
            control_path="observations/arms/iload-L346/rows/*/correct",
            relation="all-greater", minimum_effect=50,
            mechanism="a chain at INTEGER arc load (L=336, 21.00 arcs per neuron) recalls more ABSOLUTE steps on every brain than a chain ten steps LONGER at fractional load (L=346, 21.625): 336 against 178-279, minimum paired margin 57. The longer chain recalling less is the whole content of the law and is not trivially true",
        ),),
        sensitivity_gap="The half-integer recovery is now explained by a "
                        "diagnostic and not by a retained arm: the arc ROTATES by "
                        "about frac(L k / n_arc) per presentation, so overlap at "
                        "lag l peaks when l * frac is near a whole turn (r = -0.816 "
                        "over 35 points). At frac 0.5 that is ALTERNATION -- odd "
                        "lags 0.216-0.252, even lags 0.525-0.694 -- two assemblies "
                        "each as well formed as the single one at integer load, ten "
                        "presentations each. The stability metric was measuring "
                        "alternation and calling it instability. The proportional "
                        "prediction was then TESTED AND REFUTED: thresholds are 16, "
                        "20, 24, 24 at b = 1, 2, 4, 8 where proportional predicts "
                        "16, 32, 64, 128. The currency is the RECRUITMENT UNION -- "
                        "the neurons a transition ever recruits, 3.40 k at b = 1 "
                        "from DRIFT alone and only 4.81 k at b = 8 -- which predicts "
                        "the threshold at r = 0.993 against b's 0.815 and saturates "
                        "where it does. What sets the drift envelope is unmeasured.",
        claim="Autonomous chain recall PEAKS WHEN THE ARC LOAD DIVIDES EVENLY. "
              "With L transitions, k winners and n_arc arc neurons, the mean "
              "number of arcs per neuron is L k / n_arc, and arc assembly "
              "stability is a function of that quantity's FRACTIONAL PART alone "
              "-- a V minimised at the half: 0.748 at frac 0, 0.420, 0.307, "
              "0.250, 0.215 at 0.5, then 0.263, 0.356, 0.583. At integer load "
              "every brain recalls every step; at frac 0.375 and 0.625 recall "
              "falls to 0.62 and 0.53 of the chain. So sequence capacity here is "
              "not a packing limit. The mechanism is an INTERACTION between the "
              "load and the SCHEDULE and neither alone suffices: under a fixed "
              "sweep exactly L transitions fall between consecutive "
              "presentations of a transition, so every arc neuron accrues "
              "exactly L k / n_arc charges in between; at an integer the bias "
              "landscape shifts UNIFORMLY, a constant offset leaves the k-WTA "
              "ranking alone, and the same winners are re-selected. Measured "
              "2-by-2 at n_arc = 1600: chain+integer 1.000, chain+half 0.991, "
              "shuffled+integer 0.030, shuffled+half 0.033. An integer load "
              "buys NOTHING under a shuffled schedule.",
        source="This repository.",
        preconditions=(
            "above roughly 19 arcs per neuron; below it the margin absorbs the "
            "imbalance and every fractional part works",
            "A FIXED, REPEATED presentation order. Amendment 10 measured it "
            "and the effect does NOT survive shuffling: the outcome advantage "
            "vanishes entirely (0.030 against 0.033), though a residual "
            "stability gap of 0.078 remains against 0.517 under chain order. "
            "The effect is about 85 percent schedule and 15 percent load",
        ),
        evidence=("chain-iload-20260913 (n_arc=1600, load step 0.125, seeds "
                  "62..81): two complete periods repeat cell for cell, all "
                  "three integer cells exact on all twenty brains, stability "
                  "within a fractional class agreeing to about 0.005 across "
                  "chain lengths sixteen apart; IL-1, IL-2, IL-4, IL-5 pass",
                  "IL-3 FAILS and is recorded: it required the outcome below "
                  "0.99 at half-integer load and measured 0.991, because its "
                  "premise that the half is the worst case is false -- the "
                  "quarter-ish fractions are",
                  "found on diagnostics at n_arc 1000 and 1500 and confirmed "
                  "out of sample at 1600 on the block those diagnostics did "
                  "not use",),
        evidence_refs=(
            EvidenceRef("research/results/runs/sequence.autonomous-chain/chain-iload-20260913/results.json", "artifact",
                        "17 cells at load step 0.125, per-arm arc stability and per-brain recall"),
            EvidenceRef("research/results/runs/sequence.autonomous-chain/chain-capacity-20260913/results.json", "artifact",
                        "the 32-cell grid whose holes this explains"),
            EvidenceRef("research/results/runs/sequence.autonomous-chain/chain-bphase-20260913/results.json", "artifact",
                        "the 2-by-2 showing the effect needs the SCHEDULE, not the load alone"),
            EvidenceRef("research/experiments/autonomous_chain.py", "producer"),
            EvidenceRef("research/notes/sequence/PREREG_autonomous_chain.md", "registration"),
        ),
        caveat="BEWARE ALIASING when sampling. Step 8 in L samples the load axis "
               "at 0.800 (n_arc=1000) and 0.533 (n_arc=1500), and a period-1.0 "
               "signal aliases there to apparent periods of 40 and 17 in L -- "
               "both of which were measured and reported before the aliasing was "
               "noticed. Sample the LOAD axis, not L. A corollary for the "
               "literature: any chain-length sweep is walking through this "
               "oscillation, so a capacity number depends on which lengths were "
               "tried. That is a reason to re-examine reported sequence limits, "
               "not a demonstration that any of them is wrong. And because the "
               "effect needs a rigid repeated order, it is substantially a "
               "statement about how these experiments PRESENT data rather than "
               "about the substrate alone: no system with irregular input has a "
               "fixed sweep, so the biological reading is weak.",
    ),

    Result(
        id="SEQ-STATE-COLLISION-TOLERATED",
        engine="hashed_arc_fsm (HashedArcFSM, membership readout), 20 brains a cell",
        status=Status.MEASURED,
        sensitivity_checks=(SensitivityCheck(
            artifact="research/results/runs/sequence.autonomous-chain/chain-states-20260913/results.json",
            sample_path="observations/arms/blocks-n64000/rows/*/seed",
            treatment_path="observations/arms/random-n4000/rows/*/arc_overlap",
            control_path="observations/arms/blocks-n64000/rows/*/arc_overlap",
            relation="all-less", minimum_effect=0.03,
            mechanism="the collidable code demonstrably reaches the organ: arc overlap falls on EVERY brain (0.034 to 0.068) between the disjoint code and the 4000-neuron random code, while consecutive-correct stays at 160 on every brain in both -- the knob turns and the outcome does not, which is what makes this null a measurement rather than a saturation",
        ),),
        sensitivity_gap="What remains unexplained is narrower than the "
                        "headline: why a ROOMY random code still costs 3.7 "
                        "steps of 256 against a contiguous block code at "
                        "pairwise overlap 0.0016, and whether CORRELATED "
                        "collision behaves like the uncorrelated kind measured "
                        "here -- projection makes assembly overlap track input "
                        "similarity and random k-subsets do not.",
        claim="The autonomous chain tolerates state collision ONLY WHERE IT HAS "
              "MARGIN, and the margin is doing the work. At a cell with room "
              "(L = 160, n_arc = 3000, 20/20 exact) all six arms stay 20/20 "
              "exact across load (L+1)k/n_state from 0.25 to 4.03, where at "
              "the tightest area a disjoint code is impossible. At a MARGINAL "
              "cell (L = 256, n_arc = 2000, 14/20 exact) the same sweep is "
              "devastating: mean consecutive-correct falls 251.8, 220.8, "
              "180.8, 142.0, 110.0 as load runs 0.40 to 6.42, losing 142 of "
              "256 steps. Measured state overlap tracks the k/n arithmetic "
              "exactly in all three runs. LOAD IS NOT THE VARIABLE: at matched "
              "load 6.4 the roomy cell is at 1.000 of L and the marginal cell "
              "at 0.430, and the roomy cell holds at 1.000 to load 13.42 -- "
              "more than twice the killing load, 161 states in 1200 neurons at "
              "pairwise overlap 0.083. So state collision costs nothing until "
              "the chain is already marginal and then costs enormously: an "
              "AMPLIFIER of an existing limit, not a limit of its own. AND "
              "THE LIMIT IT AMPLIFIES IS THE ARC: holding the chain and the "
              "state crowding fixed at the combination that collapsed to 0.430 "
              "of L and giving the arc 1.5x the neurons restores 0.997, and 2x "
              "restores every brain, while a disjoint control sits at 0.998 to "
              "1.000 at both arc sizes. State collision is not a failure mode "
              "of the state area; it is a way of SPENDING ARC CAPACITY, free "
              "until the arc has none left -- which is also why crowding the "
              "states moved the arc overlap in every run. "
              "Separately and much smaller, a random code costs 3.7 steps of "
              "256 against the contiguous block code even when roomy, which is "
              "unexplained.",
        source="This repository.",
        evidence=("--states at the roomy cell: six arms all 20/20 exact; SC-3 "
                  "FAILED, which the registration named in advance as the "
                  "stronger outcome",
                  "--margin at the marginal cell: mean correct 255.5 (blocks), "
                  "251.8, 220.8, 180.8, 142.0, 110.0 (random, load 0.40 to "
                  "6.42) -- a monotone dose-response",
                  "--load drives the ROOMY cell to 13.42 and it stays at 1.000 "
                  "of L throughout (19/20 exact at the tightest, one brain "
                  "losing one step); LM-4 FAILED because that cell has no "
                  "breaking point in the sweep, and the registration's "
                  "pre-declared reading of that bar combination is WITHDRAWN "
                  "in the note, with the reasoning set out",
                  "the marginal run's BARS cannot carry this: all five read "
                  "exact/20, every random arm is 0/20 roomy and crowded alike, "
                  "so MC-3 passed vacuously (0 <= 0) and MC-2 passed by "
                  "comparing the tightest random arm against BLOCKS, mixing "
                  "crowding with the randomness cost. The dose-response is on "
                  "mean correct, which no bar reads",
                  "arc overlap FALLS as the state area shrinks in both runs "
                  "(0.0921 to 0.0447 roomy; 0.1493 to 0.0625 marginal): "
                  "crowding the states made the arcs more distinct, which is "
                  "unexplained and which breaks the isolation bar",),
        evidence_refs=(
            EvidenceRef("research/results/runs/sequence.autonomous-chain/chain-states-20260913/results.json", "artifact",
                        "roomy cell: six arms, per-brain correct/arc_overlap, per-arm state overlap and load"),
            EvidenceRef("research/results/runs/sequence.autonomous-chain/chain-margin-20260913/results.json", "artifact",
                        "marginal cell: the same six arms, where the dose-response appears"),
            EvidenceRef("research/results/runs/sequence.autonomous-chain/chain-load-20260913/results.json", "artifact",
                        "the roomy cell driven to load 13.42, which separates margin from load"),
            EvidenceRef("research/results/runs/sequence.autonomous-chain/chain-arcb-20260913/results.json", "artifact",
                        "arc size swept at fixed chain and fixed state crowding: the rescue"),
            EvidenceRef("research/experiments/autonomous_chain.py", "producer"),
            EvidenceRef("research/notes/sequence/PREREG_autonomous_chain.md", "registration"),
        ),
        caveat="Random k-subsets are UNCORRELATED. Projection forms assemblies "
               "whose overlap tracks input similarity, which is structured "
               "interference neither sweep produces, and the papers' states are "
               "projection-formed. This measures collision per se, not "
               "correlated collision. The id is kept for citation stability but "
               "reads too strongly on its own: the tolerance is conditional on "
               "margin, and what the crowding cost is ATTRIBUTABLE to is open.",
    ),

    Result(
        id="SEQ-REGIME-CLIFF",
        engine="numpy_sparse, arc materialized (retained runner replay, 10 seeds); the original sweep ran on the sampled arc",
        status=Status.MEASURED,
        sensitivity_checks=(SensitivityCheck(
            artifact="research/results/runs/sequence.a1-exactness-sweep/sens-exactness-sweep-v3-20260912/results.json",
            sample_path="observations/by_p/0.4/seeds",
            treatment_path="observations/by_p/0.4/exact_steps",
            control_path="observations/by_p/0.2/exact_steps",
            relation="all-greater", minimum_effect=4,
            mechanism="exact recovery steps of 10 per seed at kp = 28 (above the 3 ln n floor of 18.6) against kp = 14 (below it), arc materialized",
        ),),
        sensitivity_gap="The retained contrast crosses the floor by changing p; "
                        "no retained arm disables plasticity at fixed p, so the "
                        "cliff's dependence on learning as such rests on the "
                        "learning-disabled null of [[SEQ-EXACT-RECOVERY]].",
        claim="Crossing the kp >= 3 ln n floor is a CLIFF, not a slope: below it "
              "recovery is rarely exact (0.22 of steps at kp = 14, arc "
              "materialized) and a long run derails at its first non-exact "
              "step; above it recovery is exact on every seed (0.98 at kp = 21, "
              "1.00 at kp = 28). The cliff is in EXACTNESS: on the short "
              "decision strings the trajectory is still correct below the "
              "floor once the arc is materialized (10/10 at kp = 14); the "
              "sampled arc had made it a cliff in decisions as well (4/10).",
        source="This repository.",
        evidence=("research/experiments/seq_a1_exactness_sweep.py: state kp "
                  "14 -> 4/100 exact steps and 4/10 trajectories; kp 21 -> "
                  "80/100 and 10/10; kp 28 -> 100/100 and 10/10, with the "
                  "transition at the predicted p = 18.6/70 = 0.266",),
        evidence_refs=(
            EvidenceRef("research/results/sequence/seq_a1_exactness_sweep_results.json", "artifact",
                        "sampled numpy arc; sequence verdict void under sampler audit"),
            EvidenceRef("research/results/sequence/seq_a1_exactness_sweep_results_materialized.json", "artifact"),
            EvidenceRef("research/results/runs/sequence.a1-exactness-sweep/sens-exactness-sweep-v3-20260912/results.json", "artifact",
                        "shared-runner replay with source archive and per-seed vectors per density"),
            EvidenceRef("research/experiments/seq_a1_exactness_sweep.py", "producer"),
        ),
        provenance_gap="the two legacy result files have no runner/source record; the 2026-09-12 replay has both",
        implemented_by=("neural_assemblies.diagnostics.regime_audit",),
        caveat="The original sweep used the sampled numpy arc; its sequence-dynamics "
               "numbers are void under PREREG_sampler_audit.md until reproduced "
               "on a materialized or hashed substrate. Mean overlap read 0.812 at the failing point while exactness "
               "was 4/100 -- the mean hides this mechanism entirely.",
    ),
    Result(
        id="SEQ-EXACT-RECOVERY",
        engine="mixed: vendored nemo_numpy reference, numpy_sparse sampled/materialized, hashed ArcFSM and soft-census organs; see per-evidence caveats",
        status=Status.MEASURED,
        sensitivity_checks=(
            SensitivityCheck(
                artifact="research/results/runs/sequence.a1-learning-null/a1-learning-null-20260910/results.json",
                sample_path="observations/cells/0/ensembles/exact_fraction/trained/keys",
                treatment_path="observations/cells/0/ensembles/exact_fraction/trained/values",
                control_path="observations/cells/0/ensembles/exact_fraction/null/values",
                relation="all-greater", minimum_effect=0.5,
                mechanism="exact state recovery with arc->state learning (beta and refraction) disabled, hashed organ at p = 0.3",
            ),
            SensitivityCheck(
                artifact="research/results/runs/sequence.a1-learning-null/a1-learning-null-20260910/results.json",
                sample_path="observations/cells/1/ensembles/exact_fraction/trained/keys",
                treatment_path="observations/cells/1/ensembles/exact_fraction/trained/values",
                control_path="observations/cells/1/ensembles/exact_fraction/null/values",
                relation="all-greater", minimum_effect=0.5,
                mechanism="exact state recovery with arc->state learning (beta and refraction) disabled, hashed organ at p = 0.4",
            ),
        ),
        sensitivity_gap="The retained learning-disabled null covers the A1 "
                        "exact-recovery subclaim on the hashed organ (both "
                        "densities, 20 seeds each). The soft-census subclaims "
                        "(rate, first-hitting law, presentation window) still "
                        "have no retained mechanism-disabled null artifact.",
        claim="The state area is a DISCRETE attractor: k-WTA maps a whole "
              "neighbourhood onto exactly one stored assembly in one step. "
              "Recovery must be EXACT -- 69 of 70 neurons is a failure, not a "
              "near-miss -- because the arc amplifies any residual ~8x per step.",
        source="This repository.",
        evidence=("research/experiments/seq_a1_arc_transfer.py: "
                  "d(output loss)/d(input loss) = 8.17 ours, 10.33 reference",
                  "research/experiments/seq_a1_horizon.py: 2000 steps, 5/5 "
                  "seeds, zero errors, exact recovery every step (p = 0.4); at "
                  "p = 0.3 the one short horizon (seed 1, step 759) was the "
                  "SAMPLER's: materialized, that seed runs 2000 steps exact",
                  "research/experiments/seq_a1_horizon_hashed.py (GATE-3 of "
                  "DESIGN_sequence_port.md): the hashed organ, 20 brains, 2000 "
                  "digits, p = 0.3 and 0.4: 40/40 brains never err; exact "
                  "recovery 0.92-1.00 at p = 0.3, 1.000 at 0.4; the numpy "
                  "machine materialized agrees 5/5",
                  "research/experiments/seq_a1_limit_cycle.py: constant input "
                  "gives an orbit closing bit-identically, 30/30",
                  "research/experiments/seq_s5_soft_census.py: at 60-120 "
                  "states, ~0.25-0.6% of transitions are SOFT (correct label, "
                  "one intruder neuron); first lattice exit equals first "
                  "soft-pair visit, 40/40 seeds, zero parameters (SAMPLED arc)",
                  "seq_s5_soft_census_hashed.py (explicit substrate, 40 "
                  "organs): soft pairs persist at 0.067% (4/6000, one intruder "
                  "each, non-abelian/product groups only), the zero-parameter "
                  "law 4/4 -- and none of the four deviations derailed in 500 "
                  "steps, where the sampled arc derailed 12/40 words"),
        evidence_refs=(
            EvidenceRef("research/results/sequence/seq_a1_horizon_results_hashed_int8_timing.json", "artifact"),
            EvidenceRef("research/results/sequence/seq_a1_horizon_materialized_check.json", "artifact"),
            EvidenceRef("research/results/sequence/seq_s5_soft_census_results.json", "artifact",
                        "sampled numpy arc"),
            EvidenceRef("research/results/sequence/seq_s5_soft_census_results_hashed.json", "artifact"),
            EvidenceRef("research/notes/sequence/PREREG_s5_cliff_anatomy.md", "registration"),
            EvidenceRef("research/results/runs/sequence.a1-learning-null/a1-learning-null-20260910/results.json", "artifact",
                        "paired learning-disabled null for the A1 exact-recovery subclaim; hashed organ, seeds 1-20"),
        ),
        provenance_gap="some legacy subclaims have no source archive or raw runner artifact",
        caveat="(1) Expansion and quantization are a PAIR: amplification alone "
               "is chaos, benign only because a quantizing area follows it; "
               "composition steps without a re-quantizing stage drift. "
               "(2) THE MAP HAS SOFT SPOTS: a rare transition emits 69/70 of "
               "the right block (one intruder), and exactness holds until the "
               "word first visits one -- the measured 'horizon' is a "
               "first-hitting time, not a decay constant (zero-parameter law, "
               "40/40 numpy seeds and 36/36 hashed organs). "
               "(3) THE DERAILMENTS WERE THE SAMPLER'S: on the sampled arc "
               "post-visit trajectories derailed into an absorbing off-lattice "
               "regime (12/18) and 12/40 words went wrong within 500 steps; on "
               "the explicit substrate the soft rate is 7.5x lower and no "
               "deviation derailed (Addendum 3 of PREREG_s5_cliff_anatomy.md). "
               "Apparent group differences at fixed L were the pair count "
               "|G| x |gens| at a flat per-pair rate, not solvability. "
               "(4) WHAT A SOFT SPOT IS (Addenda 4-5, 800 organs): the "
               "collision of the target block's weakest member with the area's "
               "best-connected outsider. The outsider is a Binomial(k, p) upper "
               "tail (43-50 of 70 rows at p = 0.4), unrelated to the Cayley "
               "graph (intruder 'other' 31/33), identical in rate across three "
               "groups of order 60 and rising with n_state; the block side is "
               "how much of the test-time arc was potentiated onto the block. "
               "(5) PRESENTATIONS HAVE A WINDOW: 8 unformed (95% soft), 15 in "
               "the window (0.06%, the two tails just touching), 30 RELOCATED "
               "(soft rate doubled, 12% of words derail). Relocation is the "
               "CLIP: at strength = beta refraction cancels potentiation "
               "exactly, a member's net drive is pinned at its base value, and "
               "the cancellation ends when its weight clips -- the "
               "best-connected members fall first (99% of lost arc neurons "
               "clipped vs 12% of kept); c* = ln(w_max max(1, k p) / base) / "
               "ln(1 + beta) ~ 28-30 here. "
               "(6) STRENGTH IS PINNED AT BETA (Addendum 6, 300 organs): below "
               "it the cross-context bias no longer cancels the state-shared "
               "neurons' potentiation and the arc collapses onto the state "
               "conjunct (0.05: across-symbol overlap 0.63; 0.08: soft rate "
               "18x); above it a member's net decays and relocates. The "
               "transducer at half beta is worse on every count (A3 Amendment "
               "2), so its null is structural. "
               "(7) THE FIX IS THE GAIN (Addendum 7): trained just below the "
               "clip edge (20 and 24 presentations, ~0.7-0.85 c*) the organ is "
               "EXACT -- 0 soft pairs in 24,000 and 0 derailments on 100 S5 "
               "organs at each. In-degree normalisation cut the 15-presentation "
               "rate 3.7x by reweighting the conjuncts (STATE -> ARC in-degree "
               "3,360 vs the stimulus's 16,000), not through the tail.",
    ),
    Result(
        id="SEQ-TEMPORAL-CARRY",
        engine="hashed transducer / temporal organ (20 brains per cell)",
        status=Status.MEASURED,
        sensitivity_checks=(SensitivityCheck(
            artifact="research/results/runs/sequence.temporal-positions/temporal-positions-study-20260910/results.json",
            sample_path="run/seeds",
            treatment_path="observations/summaries/g1/D/values",
            control_path="observations/summaries/blind_g1/D/values",
            relation="all-greater", minimum_effect=0.05,
            mechanism="state-dependent distractor carry versus state-blind g=1",
        ), SensitivityCheck(
            artifact="research/results/runs/sequence.temporal-positions/temporal-positions-gap6-20260913/results.json",
            sample_path="run/seeds",
            treatment_path="observations/summaries/g1/D1/values",
            control_path="observations/summaries/g1/D6/values",
            relation="all-greater", minimum_effect=0.05,
            mechanism="the decay itself on the longest chain: at gap 6 the carry at the first distractor exceeds the carry at the sixth on every one of twenty brains (minimum 0.0687), so the positional decay is a per-brain fact and not an ensemble average",
        )),
        claim="A transducer whose STATE is its previous arc (state_mode='copy') "
              "and whose PREDICTED arc neurons win (the lateral ARC -> ARC "
              "fiber's top-k above half its maximum get (1 + g) x drive, g = 1) "
              "improves next-token prediction on the synthetic agreement chain. On "
              "the agreement chain corpus it beats a bigram by 0.149 MRR with "
              "two distractors between the agreeing words (72% of the oracle "
              "gap; twenty fresh seeds replicate +0.148 to the third digit) and "
              "by 0.106 with three (60%); the copy-state native readout gains nothing "
              "(-0.002), and reading with the state area empty loses the whole "
              "gain (+0.146 full minus blind). POSITION-SPECIFIC MECHANISM: on "
              "twenty further fresh seeds, mean subject-number contrast over the "
              "six distractor positions is 0.0258 at g = 0, 0.1849 at g = 1, "
              "and 0.0022 with state blinded. The paired g = 1 minus g = 0 "
              "amplification is 0.1591 [0.1400, 0.1781]. At g = 0 the contrast "
              "is present after the first distractor (0.0479) but absent after "
              "the second (0.0037); at g = 1 it remains 0.2032 then 0.1666. "
              "Thus predicted-win amplifies and preserves the state-dependent "
              "representation through the second distractor. DECAY LAW "
              "(gaps 3-6, twenty brains each, PREREG_temporal_positions.md "
              "Amendment 1): the carry falls geometrically with distractor "
              "position, D_j = A r^(j-1) with a POOLED r = 0.709 [0.689, "
              "0.729] whose per-gap fits (0.78, 0.74, 0.69, 0.73) all contain "
              "it; the SHAPE is one constant across gaps and the AMPLITUDE is "
              "not (A = 0.207, 0.213, 0.161, 0.124 as the chain lengthens), "
              "which is the whole of the pooled fit's residual. HORIZON: the "
              "carry clears the registered 0.02 bar at the last distractor "
              "through gap 5 (D_5 = 0.042 [0.034, 0.049]) and not at gap 6 "
              "(D_6 = 0.024 [0.016, 0.032], positive on 20 of 20 brains but "
              "above the bar on 10) -- five distractors, measured, not "
              "extrapolated. The plain conjunction decays at r = 0.008: gone "
              "after one position at every gap. ORDER: two twelve-word "
              "sequences sharing ten words are continued correctly on 20 of 20 "
              "brains by the induced state and by the copy state at 20 "
              "presentations (19 of 20 with predicted-win at g = 4), and "
              "forgotten past the clip edge c* = ln(20) / ln(1.1) = 31.4 "
              "presentations, where the sequence stimuli relocate the arcs.",
        source="This repository; PREREG_temporal_memory.md (cells A and B, "
               "Amendments 1-2, bars TM-1 to TM-9); "
               "PREREG_temporal_positions.md (bars TP-1 to TP-4); the design is the "
               "literature's temporal memory (predictive cells win) rebuilt on "
               "the refracted-arc transducer.",
        preconditions=("hashed substrate, twenty brains per cell, n = n_arc = "
                       "10,000, k = 200, organ_p = 0.2, beta = 0.1, arc "
                       "strength at beta",
                       "presentations inside the clip window for the order "
                       "result (20 of c* = 31.4)",
                       "a corpus whose oracle gap is large enough to measure: "
                       "0.21 at gap 2, 0.18 at gap 3 (PREREG_agreement_corpus.md)"),
        evidence=("seq_a3_transducer.py --temporal, gap 2, seeds 42-61: g = 1 "
                  "+0.148 +/- 0.009, g = 4 +0.140, g = 0 -0.001 (results_temporal_chain_gap2.json)",
                  "fresh seeds 62-81 (_amend2_fresh): +0.149 +/- 0.009, lower "
                  "bound 0.141 against the 0.085 bar; same-number minus "
                  "different-number arc overlap 0.220 +/- 0.020 at g = 1 (bar "
                  "0.10), 0.111 +/- 0.003 at g = 0; pooled-position values VOID for TM-9 "
                  "(see research/notes/sequence/AUDIT_temporal_position_pooling.md)",
                  "sequence.temporal-positions/temporal-positions-study-20260910, "
                  "seeds 82-101: distractor contrast g = 0 0.0258 [0.0226, "
                  "0.0290], g = 1 0.1849 [0.1660, 0.2038], blind 0.0022 "
                  "[-0.0003, 0.0046]; paired amplification 0.1591 [0.1400, "
                  "0.1781]; all TP-1 to TP-4 bars pass",
                  "gap 3, seeds 42-61 (_amend2_gap3): +0.106 +/- 0.011, lower "
                  "bound 0.095 against the 0.070 bar",
                  "gaps 3-6 under the position instrument (protocol version 2, "
                  "seeds 82-101 each): D_j on the g = 1 arm 0.207/0.160/0.127 "
                  "at gap 3, 0.215/0.151/0.117/0.087 at gap 4, "
                  "0.168/0.098/0.076/0.058/0.042 at gap 5, "
                  "0.128/0.083/0.065/0.052/0.037/0.024 at gap 6; DL-1, DL-2, "
                  "DL-4 and DL-5 pass at every gap, DL-3 fails at gap 6 only, "
                  "and TP-3 fails from gap 3 on because its estimand is a mean "
                  "over all positions and the plain conjunction carries at one",
                  "seq_tm_high_order.py --presentations 20: sets I-III exact on "
                  "the induced and copy arms; at 40 presentations set III falls "
                  "to 1 of 20 on every arm (results in research/results/sequence/)"),
        evidence_refs=(
            EvidenceRef("research/results/runs/sequence.temporal-positions/temporal-positions-study-20260910/results.json", "artifact"),
            EvidenceRef("research/results/runs/sequence.temporal-positions/temporal-positions-gap2-20260912/results.json", "artifact"),
            EvidenceRef("research/results/runs/sequence.temporal-positions/temporal-positions-gap3-20260913/results.json", "artifact"),
            EvidenceRef("research/results/runs/sequence.temporal-positions/temporal-positions-gap4-20260913/results.json", "artifact"),
            EvidenceRef("research/results/runs/sequence.temporal-positions/temporal-positions-gap5-20260913/results.json", "artifact"),
            EvidenceRef("research/results/runs/sequence.temporal-positions/temporal-positions-gap6-20260913/results.json", "artifact"),
            EvidenceRef("research/results/sequence/seq_a3_transducer_results_temporal_chain_gap2.json", "artifact"),
            EvidenceRef("research/results/sequence/seq_a3_transducer_results_temporal_chain_gap2_amend2_fresh.json", "artifact"),
            EvidenceRef("research/results/sequence/seq_a3_transducer_results_temporal_chain_gap3_amend2_gap3.json", "artifact"),
            EvidenceRef("research/results/sequence/seq_tm_high_order_results_p20.json", "artifact"),
            EvidenceRef("research/notes/sequence/PREREG_temporal_positions.md", "registration"),
        ),
        provenance_gap="pre-schema-4 prediction/order artifacts lack complete source and environment capture",
        implemented_by=("neural_assemblies/core/torch_engine/_hashed_transducer.py",),
        caveat="A prediction result on a synthetic corpus, not a language model: "
               "no baseline beyond bigram and oracle, no scaling curve in n. "
               "The decay law rests on offset MEANS over twenty brains at four "
               "gaps: at gaps 5 and 6 the last offset's carry does not exceed "
               "the state-blind control on every brain (minimum paired "
               "difference -0.002 and -0.004), so the tail is an ensemble "
               "statement with an interval, never a per-brain one. The falling "
               "amplitude is unexplained: a larger gap both lengthens the "
               "sentence and repeats the same noun-to-noun transitions more "
               "often, and nothing here separates the two. "
               "The historical TM-9 pooled-position values remain void; the "
               "separately preregistered position-specific run replaces their "
               "mechanism interpretation. Its g=0 curve reaches only the first "
               "distractor, while g=1 remains represented after the second; no "
               "alternative readout was tested. The "
               "predicted-win rule is a gain on a lateral fiber applied at "
               "selection, not a plasticity rule. The successor-state "
               "construction (state teacher-forced toward the next h words) "
               "carries nothing on the same corpus (PREREG_successor_state.md).",
    ),
    Result(
        id="ARC-CONJUNCT-EXPOSURE",
        engine="reference_nemo_numpy (the vendored explicit-matrix FSM, declared profile; retained runner study of 20 seeds, plus the original 3-seed log)",
        status=Status.MEASURED,
        sensitivity_checks=(SensitivityCheck(
            artifact="research/results/runs/sequence.autonomous-chain/autonomous-chain-20260913/results.json",
            sample_path="run/seeds",
            treatment_path="observations/arms/L32-no-refraction/rows/*/arc_overlap",
            control_path="observations/arms/L32/rows/*/arc_overlap",
            relation="all-greater", minimum_effect=0.9,
            mechanism="the collapse itself, at the extreme: a 32-transition chain over ONE constant symbol, so the symbol is 32 times more exposed than any state. With refraction off every state drives the IDENTICAL arc (overlap 1.0000 on every brain) and the chain takes zero correct steps; with refraction at beta the arcs are disjoint (0.0000) and all 32 steps are exact",
        ), SensitivityCheck(
            artifact="research/results/runs/sequence.arc-refraction-reference/arc-ref-study-20260912/results.json",
            sample_path="observations/arms/off-p15/rows/*/seed",
            treatment_path="observations/arms/off-p15/rows/*/across_symbol",
            control_path="observations/arms/proportional-p15/rows/*/across_symbol",
            relation="all-greater", minimum_effect=0.5,
            mechanism="refraction ablated (the arc's bias never accumulates): across-symbol arc overlap at 15 presentations, every seed against the drive-proportional reference",
        ),),
        sensitivity_gap="",
        claim="A conjunction area collapses onto whichever conjunct is exposed "
              "more often, unless an opposing force (refraction) is present.",
        source="This repository; the same law as the role-binding gain result.",
        evidence=("research/experiments/seq_arc_refraction_reference.py: "
                  "ablating refraction takes across-symbol overlap 0.000 -> "
                  "0.989 and the task 3/3 -> 0/3",
                  "arc-ref-study-20260912 (20 seeds, PREREG_arc_refraction_reference.md "
                  "AR-2): refraction off, across-symbol overlap 0.977-0.997 on every "
                  "seed (mean 0.988) against 0.000 with the proportional rule; "
                  "across-state 0.098; decided 0/20 against 18/20"),
        evidence_refs=(
            EvidenceRef("research/results/runs/sequence.autonomous-chain/autonomous-chain-20260913/results.json", "artifact"),
            EvidenceRef("research/notes/sequence/PREREG_autonomous_chain.md", "registration"),
            EvidenceRef("research/experiments/autonomous_chain.py", "producer"),

            EvidenceRef("research/results/logs/seq_arc_refraction_reference.log", "log",
                        "three seeds, no run record"),
            EvidenceRef("research/results/runs/sequence.arc-refraction-reference/arc-ref-study-20260912/results.json", "artifact"),
            EvidenceRef("research/notes/sequence/PREREG_arc_refraction_reference.md", "registration"),
            EvidenceRef("research/experiments/seq_arc_refraction_reference.py", "producer"),
        ),
        provenance_gap="",
        preconditions=("BOTH overlap directions measured -- one alone cannot "
                       "distinguish a conjunction from collapse onto the other "
                       "conjunct",),
        caveat="Task #92 measured one direction, read 0.90-0.99, and concluded "
               "a conjunctive arc has no operating point. It has one.",
    ),
]

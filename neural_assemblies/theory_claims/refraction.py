"""Refraction: the anti-Hebbian bias, its proportion, its load, and anti-merging.

Moved from neural_assemblies/theory.py unchanged; theory.py lists these in its order."""
from __future__ import annotations

from typing import List

from .types import EvidenceRef, Result, SensitivityCheck, Status

CLAIMS: List[Result] = [
    Result(
        id="REFRACTION-PROPORTIONAL",
        engine="reference_nemo_numpy (the vendored explicit-matrix FSM, declared profile; retained runner study of 20 seeds, plus the original 3-seed log)",
        status=Status.MEASURED,
        sensitivity_checks=(SensitivityCheck(
            artifact="research/results/runs/sequence.arc-refraction-reference/arc-ref-study-20260912/results.json",
            sample_path="observations/arms/proportional-p30/rows/*/seed",
            treatment_path="observations/arms/proportional-p30/rows/*/decided",
            control_path="observations/arms/constant-s10-p30/rows/*/decided",
            relation="all-greater", minimum_effect=1,
            mechanism="the constant increment that decides every seed at 15 presentations (s* = 10) against the proportional rule at 30 presentations: the constant's operating point has moved, the proportional rule's has not",
        ),),
        sensitivity_gap="",
        claim="Refraction must charge in proportion to the winner's raw drive. "
              "A constant increment is not an equivalent parameterization: "
              "Hebbian growth multiplies drive while a constant grows linearly, "
              "so its operating point MOVES with training duration.",
        source="This repository.",
        evidence=("research/experiments/seq_arc_refraction_reference.py: the "
                  "constant winning at 15 presentations fails at 30, while the "
                  "proportional rule passes both untouched",
                  "arc-ref-study-20260912 (20 seeds, AR-3 and AR-4): constant 10 "
                  "decides 20/20 at 15 presentations and 0/20 at 30 (across-symbol "
                  "0.000 -> 0.369); constant 30 decides 0/20 at 15 and 20/20 at 30; "
                  "proportional decides 18/20 at 15 and 20/20 at 30 with both "
                  "overlaps 0.000 throughout"),
        evidence_refs=(
            EvidenceRef("research/results/logs/seq_arc_refraction_reference.log", "log",
                        "three seeds, no run record"),
            EvidenceRef("research/results/runs/sequence.arc-refraction-reference/arc-ref-study-20260912/results.json", "artifact"),
            EvidenceRef("research/notes/sequence/PREREG_arc_refraction_reference.md", "registration"),
            EvidenceRef("research/experiments/seq_arc_refraction_reference.py", "producer"),
        ),
        provenance_gap="",
        implemented_by=("neural_assemblies/core/_homeostasis.py",),
    ),
    Result(
        id="REFRACTION-NEEDS-LOAD",
        engine="numpy_sparse, arc materialized (retained runner replay, 10 seeds); the sampled-arc sweep's load floor is retracted",
        status=Status.MEASURED,
        sensitivity_checks=(SensitivityCheck(
            artifact="research/results/runs/sequence.a2-refraction-load/sens-refraction-load-v3-20260912/results.json",
            sample_path="observations/rows/8/seeds",
            treatment_path="observations/rows/8/correct_by_seed",
            control_path="observations/rows/9/correct_by_seed",
            relation="all-greater", minimum_effect=1,
            mechanism="nine-conjunction arc at load 1.26 (n_arc 500) against load 1.80 (n_arc 350): the ceiling, arc materialized",
        ),),
        sensitivity_gap="The retained contrast is the ceiling (load 1.26 against 1.80). "
                        "No retained arm disables refraction itself at a fitting load; "
                        "the refraction ablation this entry's docstring cites (0/10, arc "
                        "collapsed to 0.988) predates the runner.",
        claim="A refracted conjunction area has a CEILING in load M*k/n: "
              "above ~1.3 its conjunctions do not fit (10/10 correct at load "
              "1.26, 0/10 at 1.80). RE-SCOPED 2026-09-09 (PREREG_sampler_audit.md): "
              "the lower edge this entry was named for -- 'below ~0.2 its "
              "assemblies never converge' -- was the numpy sampler's; with the "
              "arc materialized a 3-conjunction arc is 10/10 correct at every "
              "load from 0.04 to 0.60. An arc must be sized so its conjunctions "
              "fit; it need not be filled.",
        source="This repository.",
        evidence=("research/experiments/seq_a2_refraction_load.py, SAMPLED arc: "
                  "a 3-conjunction arc goes 1/10 at load 0.04 to 10/10 at 0.21; "
                  "a 9-conjunction arc holds 10/10 from 0.32 to 1.26 and "
                  "collapses to 1/10 at 1.80",
                  "the same script with the arc MATERIALIZED (NEMO_MATERIALIZE=1, "
                  "10 seeds): 3-conjunction arc 10/10 at every load 0.04-0.60; "
                  "9-conjunction arc 10/10 to 1.26, 0/10 at 1.80",
                  "arc assembly stability across training: 0.286 from "
                  "presentation 5 to 15 under-loaded, 0.957 from 10 to 15 loaded"),
        evidence_refs=(
            EvidenceRef("research/results/sequence/seq_a2_refraction_load_results.json", "artifact",
                        "sampled arc; lower-edge inference retracted"),
            EvidenceRef("research/results/sequence/seq_a2_refraction_load_results_materialized.json", "artifact"),
            EvidenceRef("research/results/runs/sequence.a2-refraction-load/sens-refraction-load-v3-20260912/results.json", "artifact",
                        "shared-runner replay with source archive and per-seed outcomes"),
            EvidenceRef("research/experiments/seq_a2_refraction_load.py", "producer"),
        ),
        provenance_gap="the two legacy artifacts lack runner records and source archives; the 2026-09-12 replay has both",
        preconditions=("refraction active -- this is a statement about what "
                       "refraction needs, not about k-WTA generally",),
        caveat="The 'silent failure under load' this entry once described -- "
               "assemblies that never stop moving while every diagnostic reads "
               "healthy -- was measured on the sampled engine and does not occur "
               "materialized; treat it as a property of lazily drawn areas, not "
               "of refraction. The ceiling is [[AC-CAP]]'s and is not independent "
               "of it. One task, 10 seeds.",
    ),
    Result(
        id="REFRACTION-ANTI-MERGING",
        engine="hashed AssemblyMemory; materialized numpy_sparse mirror with summed stimulus parts (not an identical stimulus protocol)",
        status=Status.MEASURED,
        sensitivity_checks=(SensitivityCheck(
            artifact="research/results/runs/memory.capacity-scaling/refraction-paired-sensitivity-20260911/results.json",
            sample_path="run/seeds",
            treatment_path="observations/conditions/refracted/cells/B~14000~160/checkpoints/128/rank1",
            control_path="observations/conditions/control/cells/B~14000~160/checkpoints/128/rank1",
            relation="all-greater", minimum_effect=0.5,
            mechanism="refracted memory versus paired Hebbian control at M=128",
        ), SensitivityCheck(
            artifact="research/results/runs/memory.capacity-scaling/capacity-law-replay-L3-20260930/results.json",
            sample_path="run/seeds",
            treatment_path="observations/conditions/refracted/cells/B~12000~160/checkpoints/64/rank1",
            control_path="observations/conditions/control/cells/B~12000~160/checkpoints/64/rank1",
            relation="all-greater", minimum_effect=0.5,
            mechanism="refracted memory versus paired Hebbian control at (2000, 60), M=64 (Amendment 8, RP-4)",
        ), SensitivityCheck(
            artifact="research/results/runs/memory.capacity-scaling/capacity-law-replay-L1-20260930/results.json",
            sample_path="run/seeds",
            treatment_path="observations/conditions/refracted/cells/B~14000~1120/checkpoints/64/rank1",
            control_path="observations/conditions/control/cells/B~14000~1120/checkpoints/64/rank1",
            relation="all-greater", minimum_effect=0.5,
            mechanism="refracted memory versus paired Hebbian control at (4000, 120), M=64 (Amendment 8, RP-4)",
        ), SensitivityCheck(
            artifact="research/results/runs/memory.capacity-scaling/capacity-law-replay-L1-20260930/results.json",
            sample_path="run/seeds",
            treatment_path="observations/conditions/refracted/cells/B~12000~130/checkpoints/384/rank1",
            control_path="observations/conditions/control/cells/B~12000~130/checkpoints/384/rank1",
            relation="all-greater", minimum_effect=0.5,
            mechanism="refracted memory versus paired Hebbian control at (2000, 30), M=384 (Amendment 8, RP-4)",
        ), SensitivityCheck(
            artifact="research/results/runs/memory.capacity-scaling/capacity-law-replay-L1-20260930/results.json",
            sample_path="run/seeds",
            treatment_path="observations/conditions/refracted/cells/B~18000~1120/checkpoints/384/rank1",
            control_path="observations/conditions/control/cells/B~18000~1120/checkpoints/384/rank1",
            relation="all-greater", minimum_effect=0.5,
            mechanism="refracted memory versus paired Hebbian control at (8000, 120), M=384 (Amendment 8, RP-4)",
        ), SensitivityCheck(
            artifact="research/results/runs/memory.capacity-scaling/capacity-law-replay-L2-20260930/results.json",
            sample_path="run/seeds",
            treatment_path="observations/conditions/refracted/cells/B~18000~160/checkpoints/1536/rank1",
            control_path="observations/conditions/control/cells/B~18000~160/checkpoints/1536/rank1",
            relation="all-greater", minimum_effect=0.5,
            mechanism="refracted memory versus paired Hebbian control at (8000, 60), M=1536 (Amendment 8, RP-4)",
        ), SensitivityCheck(
            artifact="research/results/runs/memory.capacity-scaling/capacity-law-replay-L2-20260930/results.json",
            sample_path="run/seeds",
            treatment_path="observations/conditions/refracted/cells/B~14000~130/checkpoints/1536/rank1",
            control_path="observations/conditions/control/cells/B~14000~130/checkpoints/1536/rank1",
            relation="all-greater", minimum_effect=0.5,
            mechanism="refracted memory versus paired Hebbian control at (4000, 30), M=1536 (Amendment 8, RP-4)",
        ),),
        sensitivity_gap="Paired refracted-versus-control checks now cover all "
                        "seven cells the law is fitted to (A7 at (4000, 60); "
                        "Amendment 8 at the other six). The masked-vs-net veto, "
                        "the convergence gate, the strength plateau and the "
                        "cross-engine mirror still lack retained paired checks.",
        claim="A recurrent k-WTA area refracted at HALF beta and read with the "
              "refraction bias MASKED holds ~25x the Hebbian ceiling: at n/k = 67 "
              "M* ~ 1600-2200 stored assemblies against 64-89 for the control, "
              "x34-38 at n/k = 33, >= x13-16 at n/k = 133 (censored). Both "
              "ceilings are functions of n/k ALONE (n/k-matched cells agree "
              "within 25%), so refraction multiplies the assembly capacity law "
              "[[AC-CAP]] rather than changing its form. The mechanism is "
              "ANTI-MERGING, not orthogonalization: the stored assemblies are "
              "orthogonal (0.00x chance) only while the area has unvisited "
              "neurons; past fill 1.0 they overlap at chance like random subsets, "
              "yet remain DISTINCT (1.000) and recoverable from a half cue, where "
              "the Hebbian control collapses into hubs (distinct 0.63, overlap "
              "15x chance) long before the area is full. The intrinsic bias "
              "vetoes recall (the net readout reads chance): a refracted memory "
              "is read through the veto or not at all. Fewer rounds per item "
              "raise the ceiling (T = 8 > T = 16) while the items still "
              "converge; refraction's benefit requires GRADED stimulus drive. "
              "SHAPE OF THE LAW (Amendment 4): in regime (k p >= 3 ln n) the "
              "refracted ceiling is ~0.40 (n/k)^2 at n/k = 67 and 133 (6995 at "
              "(8000, 60)), doubling exponents 2.1 then 1.8 -- Willshaw-like, "
              "not adopted as a power law; the control is ~0.017 (n/k)^2 from "
              "n/k = 67 on, so the multiplier is ~23-25x there. Out of regime "
              "(k p = 15 < 3 ln n) the cells fall 20-32% below their n/k pairs "
              "and do not converge at low load. GATED ROUNDS (Amendment 5): "
              "ending an item's rounds at its first repeated winner set under "
              "T_max = 8 raises the ceiling by a CONSTANT fraction of n/k: +35% at "
              "n/k = 67 (2645 vs 1961) and +24% at 133 (8666 vs 6995), both "
              "resolved; at both cells the ceiling sits where items stop "
              "converging inside T_max (the fraction converging is a U in load, "
              "0.98 at mid load, 0 at the ceiling); the same gate STARVES the "
              "Hebbian control, whose winners settle in ~4 rounds before its "
              "memory is written; T_max = 16 under the gate costs the "
              "out-of-regime cell 13x. STRENGTH IS A SWITCH, NOT A DIAL "
              "(Amendment 6): 0.3, 0.4, 0.5 and 0.6 beta give the same ceiling "
              "(1919-1993, one bracket) at n/k = 67 -- the ceiling is the "
              "synaptic memory's, refraction only has to prevent merging while "
              "it is written; the churn transition is in (0.6, 0.7] beta at "
              "T = 8. WHAT THE CONSTANT IS (Amendment 9): independent random "
              "k-subsets written into the SAME circuit at the memory's own write "
              "strength fail at 0.69-0.92 (n/k)^2 with half of the present "
              "synapses potentiated (0.49-0.60), Willshaw's operating point -- "
              "the n/k law is the load M (k/n)^2; the refracted memory realises "
              "a constant 0.47-0.56 of that (PATTERN EFFICIENCY), the Hebbian "
              "control 0.014-0.023. The reference is conditional on write "
              "strength: two potentiation counts weaker moves the random-pattern "
              "ceiling +66-89% at k = 60 and +17-29% at k = 120. The refracted "
              "assemblies share synapse pairs 2.1-2.8x as much as random subsets "
              "and fail at 1.14-1.43x random's pair load; the gate's gain is "
              "carried by its assemblies (rewritten cleanly at equal write "
              "strength they store +35-51%). WRITE STRENGTH (Amendment 11, a "
              "replication of Amendment 10's void record): on the same circuit "
              "COMPLETION (>= 0.8 of an item from half of it) needs a strong "
              "write and peaks at the clipped binary end, ~g(k p) (n/k)^2 with "
              "g = 0.29 / 0.56 / 0.80 at k p = 15 / 30 / 60, where the n/k law "
              "fails (k = 120 stores 1.42x its n/k pair); IDENTIFICATION (rank-1) "
              "grows as the write weakens, to 4.3-4.5 (n/k)^2 at one count per "
              "item, inside a LOAD WINDOW whose lower edge also rises. The "
              "operating write (5-6 counts) sits between: its rank-1 ceiling "
              "obeys the n/k law and it completes little; the smallest write that "
              "completes falls with k p (11 / 6 / 4 counts at k = 30 / 60 / 120). "
              "LEARNING RATE (Amendments 12-13, confirmed on new brains and two "
              "unseen cells): the refracted memory's DISTINCT-completion-optimal "
              "beta obeys ln(1 + beta*) = 0.29 / sqrt(k p / 2) (gamma* 0.257-0.303 "
              "at seven cells, k p = 30-120; exponent 1/2 collapses beta* to CV "
              "0.05 against 0.21 for beta itself) -- a fan-in-scaled learning "
              "rate, the counterpart of muP's per-layer rate for one area. The "
              "adopted beta = 0.1 is above it everywhere (1.3-2.7x), so the "
              "ceilings above are identification ceilings at an over-strong "
              "write; at the optimum the best distinct completion grows with k p "
              "(0.41-2.46 (n/k)^2 at n/k = 33). Completion must be DISTINCT "
              "(recovered and rank-1): above beta ~ 0.2 merged assemblies make "
              "'completion' exceed identification. ON RECALL (Amendment 14, ten "
              "(n, k, p) cells, new brains): with each memory at its own best "
              "learning rate the refracted memory completes 14-57x as many "
              "distinct items as the Hebbian control at every cell; at equal "
              "(n/k, k p) capacity is the same at p = 0.125, 0.25, 0.5 (within "
              "8%) but sparse p wants a higher learning rate; whether capacity at "
              "the optimum scales as n^2 p / (k ln(n/k)) or (n/k)^2 is NOT "
              "settled. THE LAW WITH CONNECTIVITY NOISE (Amendment 16, six "
              "cells, p = 0.125-0.75): ln(1 + beta*) = 0.285 sqrt(2 (1 - p)) / "
              "sqrt(k p / 2) -- the write clears the spread of the cue synapses "
              "a neuron receives -- within 15% everywhere, including the dense "
              "side (0.707 predicted 0.71); capacity at the optimum is the same "
              "across p at equal fan-in. DEPTH (Amendment 15): at each number of "
              "write rounds' own best rate capacity is the same (T = 6-16); the "
              "per-round optimum falls faster than 1/T. THE CONVERGENCE-THRESHOLD "
              "RULE (Amendment 17, ten cells, new brains): above the floor "
              "k p >= 3 ln n the best rate is 0.18-0.19 of sqrt((1 - p) ln n / "
              "(p k)) (k p = 30-80); below it the fraction holds at p = 0.5 down "
              "to k p = 5 but climbs at p = 0.05 (0.21 / 0.27 / 0.30 at k p = 20 "
              "/ 10 / 5), capacity is flat in k (1356-1452 at k = 10-40) and "
              "no longer p-invariant (201 vs 1356 at k p = 5), and the PNAS 2020 "
              "cell (k p = 1) completes at no rate (rank-1 at most 112). THE "
              "ONSET (Amendment 18, ten cells, a 12-per-octave grid): the "
              "weakest completing write is 0.16-0.18 theta everywhere (CV "
              "0.041) -- dense and sparse, above and below the floor; in dense "
              "areas it completes only after 158-1098 stored items and the best "
              "write sits within a step of it, in sparse ones it completes at "
              "once and the best write is 1.6-1.8x higher. REGIMES (Amendment "
              "19): at fixed n and p, capacity at each cell's optimum is 1059-"
              "1513 over k = 10-160 -- nearly k-independent, no law change at "
              "the floor. RECOGNITION vs RECALL (Amendment 20): the ratio grows "
              "as fan-in falls (Spearman -0.88); at the PNAS parameters recall "
              "appears only from k p = 4 while recognition holds 114-207 items "
              "from k p = 1. THE IN-DEGREE LAW (Amendment 21, nine new cells): at "
              "the best write capacity is set by d = n p alone -- at equal d the "
              "cells agree within 11% while n/k varies fourfold -- and grows as "
              "d^1.5-1.7 (refit on 55 cells: C = 0.0099 d^1.554, x1.15); the "
              "registered constants missed by 26% at d = 6000. TWO MEMORIES "
              "(Amendment 23): each at its own best write, recognition holds "
              "15-86x as many items as recall, at a write 6-40x weaker (int16 "
              "counts); at fixed d recall stays put while recognition scales "
              "with n/k and its best write as 1/k.",
        source="This repository; PREREG_refraction_memory.md (bars R1-R7, N1-N3, "
               "Q1-Q4, G1-G6, S8-S9).",
        evidence=("seq_capacity_scaling.py, arm B on the organ fiber, 20 brains, "
                  "M grid to 4096, s = 0.5 beta masked vs control: n/k = 33: "
                  "431 / 383 vs 11.3; n/k = 67: 1961 / 1589 / 2230 vs 83 / 64 / "
                  "89; n/k = 133: >= 4096 vs 307 / 263",
                  "distinct 1.000 at every refracted ceiling; control 0.63 at "
                  "M = 128 (n = 4000), pairwise 15x chance",
                  "net readout at n = 4000: M* = 8 (chance at every M)",
                  "T = 16: M* = 203 against T = 8's >= 1024 (n = 4000); T = 4 "
                  "does not converge at low load and does at high load",
                  "numpy_sparse gate (refraction_memory_numpy.py, materialized, "
                  "5 brains, summed Binomial stimuli): refracted >= 512 "
                  "(censored, rank-1 1.000 throughout, distinct 1.000) vs "
                  "control 34 -- the claim holds on the engine whose k-WTA has "
                  "no selector defect",
                  "grid to 16384, 20 brains: (8000, 60) REF 6995 [6144, 8192) = "
                  "0.395 (n/k)^2, CTL 307; (4000, 30) REF 4749 [4096, 6144), "
                  "CTL 263 (out of regime)",
                  "(4000, 60) gated T_max 8: REF 2645 [2560, 2816) vs 1961; CTL "
                  "gated rank-1 0.32 at M = 8 (no memory formed); (4000, 30) "
                  "gated T_max 16: 362 vs 4749",
                  "(8000, 60) gated T_max 8: 8666 [8192, 10240) vs 6995 (x1.24)",
                  "strength 0.3 / 0.4 / 0.5 / 0.6 beta at (4000, 60): 1986 / "
                  "1919 / 1961 / 1993, all [1536, 2048)",
                  "memory_pattern_efficiency.py, 20 brains, seven cells (A9): "
                  "M*/(n/k)^2 real / random / balanced at (2000,60) 0.394 / "
                  "0.736 / 1.629, (4000,120) 0.359 / 0.686 / 3.078, (4000,60) "
                  "0.432 / 0.806 / 1.384, (8000,120) 0.512 / 0.920 / 2.296, "
                  "(8000,60) 0.403 / 0.860 / 1.167; gated 9078 and 2593",
                  "memory_write_strength.py --scan, 20 brains, seven cells (A11): "
                  "completion at the binary write / (n/k)^2 0.562 / 0.563 / 0.560 "
                  "at k = 60, 0.798 / 0.798 at k = 120; rank-1 window at one count "
                  "per item [3310, 76028) at (8000, 60)",
                  "memory_learning_rate.py --distinct, 20 new brains, seven cells "
                  "(A13): beta* 0.0705 / 0.0558 / 0.0532 / 0.0444 / 0.0377 at n/k = 33 "
                  "(k p 30 / 45 / 60 / 90 / 120), 0.0752 / 0.0569 at n/k = 67; "
                  "predicted 0.063 and 0.044 at the unseen (3000,90), (6000,180)",
                  "memory_threshold_law.py, 20 new brains, ten cells (A17): "
                  "beta*/theta 0.180 / 0.189 / 0.180 above the floor; 0.216 / "
                  "0.193 / 0.213 (p 0.5) and 0.211 / 0.270 / 0.296 (p 0.05) below; "
                  "(10000, 100, 0.01) distinct completion 0 at every rate",
                  "memory_onset.py, 20 new brains, ten cells (A18): onset/theta "
                  "0.170 x5, 0.180 x3, 0.160 x2; dense onset windows open at "
                  "158-1098 items; best/onset 1.00-1.06 (dense, 5 of 6), 1.59-1.78 "
                  "(p 0.05)",
                  "memory_regimes.py, 20 new brains, fifteen cells (A19-20): "
                  "capacity 1314/1148/1513/1383/1473/1392/1468/1239/1059 at "
                  "n = 4000, k = 10-160; 508/1513/3222 at n = 2000/4000/8000, k = 20; "
                  "rank-1 at its own best write >= 32768 at k p <= 10, 2511 at k p 80",
                  "memory_degree_law.py, 20 new brains, nine cells (A21): d = 1000: "
                  "417/439; d = 1500: 971/811/944 (n/k 200/100/50); d = 3000: "
                  "2871/2739; d = 6000: 8605; slope 1.669",
                  "memory_recognition.py v2, 20 new brains, seven cells (A23): "
                  "recognition 81290/109640/51316/27917/12521 vs recall 1350/1292/"
                  "1339/1299/837 at n = 4000, k = 10-160"),
        evidence_refs=(
            EvidenceRef("research/notes/memory/PREREG_refraction_memory.md", "registration"),
            EvidenceRef("research/results/runs/memory.capacity-scaling/capacity-record-consumed-20260910/results.json", "artifact",
                        "registered protocol-consumption replay, not the whole historical grid"),
            EvidenceRef("research/results/memory/refraction_memory_numpy_results.json", "artifact"),
            EvidenceRef("research/results/memory/capacity_scaling_results_figure_ref.json", "artifact"),
            EvidenceRef("research/results/memory/capacity_scaling_results_figure_ctl.json", "artifact"),
            EvidenceRef("research/results/runs/memory.capacity-scaling/refraction-paired-sensitivity-20260911/results.json", "artifact",
                        "paired version-3 reproduction and sensitivity run"),
            EvidenceRef("research/results/comparisons/refraction-paired-sensitivity-20260911.json", "comparison",
                        "2090-scalar migration comparison receipt"),
            EvidenceRef("research/results/runs/memory.capacity-scaling/capacity-law-replay-L1-20260930/results.json", "artifact",
                        "Amendment 8 replay: (2000,30), (4000,120), (8000,120), paired"),
            EvidenceRef("research/results/runs/memory.capacity-scaling/capacity-law-replay-L2-20260930/results.json", "artifact",
                        "Amendment 8 replay: (8000,60), (4000,30), paired, grid to 16384"),
            EvidenceRef("research/results/runs/memory.capacity-scaling/capacity-law-replay-L3-20260930/results.json", "artifact",
                        "Amendment 8 replay: (2000,60), paired"),
            EvidenceRef("research/results/comparisons/capacity-law-replay-L1-20260930.json", "comparison",
                        "3615-scalar per-seed comparison with the surviving k-sweep control file"),
            EvidenceRef("research/results/comparisons/capacity-law-replay-L2-20260930.json", "comparison",
                        "3050-scalar per-seed comparison with the surviving n/k = 133 refracted file"),
            EvidenceRef("research/results/runs/memory.pattern-efficiency/pattern-efficiency-20260930/results.json", "artifact",
                        "Amendment 9: real, clean, random, balanced and gated memories on one circuit, seven cells"),
            EvidenceRef("research/results/runs/memory.write-strength/write-strength-scan-20261001/results.json", "artifact",
                        "Amendment 11: identification and completion load windows across write strength, seven cells"),
            EvidenceRef("research/results/runs/memory.learning-rate/learning-rate-20261001/results.json", "artifact",
                        "Amendment 12: beta sweep on the real memory, five cells (T1-T3 failed: merged recall)"),
            EvidenceRef("research/results/runs/memory.learning-rate/learning-rate-distinct-20261001/results.json", "artifact",
                        "Amendment 13: distinct-completion beta sweep, new brains, seven cells incl. two unseen"),
            EvidenceRef("research/results/runs/memory.recall-law/recall-law-20261001/results.json", "artifact",
                        "Amendment 14: refracted vs Hebbian on distinct completion, ten (n, k, p) cells"),
            EvidenceRef("research/results/runs/memory.time-depth/time-depth-20261001/results.json", "artifact",
                        "Amendment 15: write rounds T = 6-16, recall at 8 rounds"),
            EvidenceRef("research/results/runs/memory.sparse-law/sparse-law-20261001/results.json", "artifact",
                        "Amendment 16: the learning-rate law with connectivity noise, p = 0.125-0.75"),
            EvidenceRef("research/results/runs/memory.threshold-law/threshold-law-20261001/results.json", "artifact",
                        "Amendment 17: the best rate as a fraction of the convergence threshold, k p = 1-80"),
            EvidenceRef("research/results/runs/memory.onset/onset-20261002-r3/results.json", "artifact",
                        "Amendment 18: the completion onset on a 12-per-octave grid, ten cells"),
            EvidenceRef("research/results/runs/memory.regimes/regimes-20261002-r3/results.json", "artifact",
                        "Amendments 19-20: capacity across k and n, recognition against recall"),
            EvidenceRef("research/results/runs/memory.degree-law/degree-law-20261002/results.json", "artifact",
                        "Amendment 21: capacity at the best write against the in-degree d = n p"),
            EvidenceRef("research/results/runs/memory.recognition/recognition-20261003-a23/results.json", "artifact",
                        "Amendment 23: recognition and recall each at its own best write"),
            EvidenceRef("research/results/runs/memory.criticality/criticality-20261003/results.json", "artifact",
                        "Amendment 24: per-brain onsets at n = 2000-16000; onset converges at 0.163-0.170 theta"),
        ),
        provenance_gap=("the seven cells the law is fitted to are replayed under the runner "
                        "(Amendment 8 and A7): exact per-seed reproduction wherever legacy "
                        "data survives, every lost ceiling reproduced; the two gated ceilings "
                        "are replayed within 5% (Amendment 9, PE-R); the strength "
                        "(Amendment 6) and T-sweep grids and the numpy mirror still "
                        "predate immutable source records"),
        preconditions=("recurrent k-WTA area, weight clip, norm_init, no column "
                       "scaling (arm B); refraction strength 0.5 beta; T = 8 "
                       "rounds per item from an inhibited area; readout = "
                       "half-cue recall with the bias masked",
                       "GRADED stimulus drive: a zero-or-size stimulus (the "
                       "engine's into a materialized area, at one Bernoulli "
                       "draw) makes the refracted item rotate through its tied "
                       "connected set and the benefit vanishes"),
        caveat="Found by re-measuring PREREG_refraction_capacity.md after the "
               "hashed selector's sign defect (1b475fc) -- its Amendment 1 "
               "('spends the substrate') was that defect. The n/k law was "
               "counted as holding at n/k = 133 while both cells were censored; "
               "resolved, they disagree by 0.68 and the k = 30 one is out of "
               "regime -- k p >= 3 ln n is a precondition, not a footnote. Three "
               "ratios do not fix an exponent that is falling (2.1 -> 1.8). The "
               "multiplier is at 0.5 beta; 0.7 beta converges only given T = 16 "
               "and holds ~185 at n/k = 67, but strength is otherwise a plateau "
               "whose lower edge (below 0.3 beta) is unmeasured. The gate is "
               "two in-regime cells (n/k = 67, 133); n/k = 33 and out of regime "
               "are unmeasured, and the gated ceiling's doubling exponent (1.71) "
               "may keep falling.",
    ),
]

"""The representation algebra: k-WTA ties, the Hebbian outer product, drive split, capacity ratios.

Moved from neural_assemblies/theory.py unchanged; theory.py lists these in its order."""
from __future__ import annotations

from typing import List

from .types import EvidenceRef, Result, SensitivityCheck, Status

CLAIMS: List[Result] = [
    # ------------------------------------------------- representation algebra
    Result(
        id="KWTA-TIE-FRAGILE",
        engine="tie census: numpy_explicit (retained runner study, 20 seeds, PREREG_kwta_tie_fragility.md); the last-bit split under a change of arithmetic: torch/CUDA selector prototype (no retained artifact) and the fused-kernel gate of 2026-09-10",
        status=Status.MEASURED,
        sensitivity_checks=(SensitivityCheck(
            artifact="research/results/runs/substrate.kwta-tie-fragility/kwta-study-20260912/results.json",
            sample_path="observations/rows/*/seed",
            treatment_path="observations/rows/*/ties_raw",
            control_path="observations/rows/*/ties_jittered",
            relation="all-greater", minimum_effect=4,
            mechanism="candidates tied at the k-th drive of the untrained 0/1 stimulus fiber (n = 1000, k = 50, p = 0.1) against the same drive with a deterministic per-neuron jitter that leaves exactly one at the bar",
        ),),
        sensitivity_gap="The retained check covers the counting half (the bar is "
                        "tied on every seed, 26 to 73 candidates). The other half, "
                        "that a change of arithmetic splits those ties into "
                        "different winners, is retained only as the GPU gate's "
                        "log and the selector prototype: on the CPU explicit "
                        "engine's own float32 reciprocal and division paths no "
                        "winner differed on any of 20 seeds, and the one-ulp "
                        "construction flipped the winner set on 19 of 20 (the "
                        "twentieth is the exact-fit case, every tied candidate "
                        "already inside the top k).",
        claim="The k-WTA bar is routinely TIED, so anything that perturbs the "
              "drive in its last bits -- a change of summation order, of "
              "arithmetic, or of tie-break policy -- can change WHICH neurons "
              "fire, not merely their order.",
        source="The base drive is a Bernoulli COUNT, hence integer-valued, so "
               "exact ties are the common case rather than an edge case. "
               "Measured 5-18 columns tied at the bar per brain at n=4000-50000 "
               "(research/experiments/gpu_radix_select_prototype.py). An ulp-level "
               "change to a summation once moved sixteen cells of an exact "
               "table with no direction to it.",
        preconditions=("integer or near-integer drive, i.e. before heavy "
                       "potentiation spreads the values",
                       "k-WTA selecting at a bar that several columns reach"),
        evidence=("research/experiments/gpu_radix_select_prototype.py",
                  "kwta-study-20260912 (numpy_explicit, 20 seeds): candidates tied at the "
                  "bar of the untrained stimulus drive 26-73 per seed (KT-1 PASS); with "
                  "in-degree normalization 1-12, at least 2 on 17 of 20 (KT-3 PASS); a "
                  "one-ulp raise of a tied outsider changes the winner set on 19 of 20 "
                  "seeds and never on the jittered drive (KT-2 as registered FAIL: seed "
                  "49 has 19 candidates above the bar and 31 at it, so all 50 are "
                  "inside k and no outsider is tied); float32 division, float32 "
                  "reciprocal and float64 division select the same winners on every seed"),
        evidence_refs=(
            EvidenceRef("research/experiments/gpu_radix_select_prototype.py", "producer",
                        "the split half; no retained artifact"),
            EvidenceRef("research/results/runs/substrate.kwta-tie-fragility/kwta-study-20260912/results.json", "artifact"),
            EvidenceRef("research/notes/substrate/PREREG_kwta_tie_fragility.md", "registration"),
            EvidenceRef("research/experiments/kwta_tie_fragility.py", "producer"),
        ),
        provenance_gap="the selector prototype's split measurement has no retained raw artifact; the tie census does",
        implemented_by=("neural_assemblies.core.numpy_engine._kwta_prune",),
        caveat="`_kwta_prune` records the operational rule this implies: making "
               "the selector's tie-break CANONICAL is a science-affecting "
               "change needing its own registration, not an optimisation to "
               "smuggle in. The fused selector folds the index into the sort "
               "key so that ties break to the smallest index BY CONSTRUCTION, "
               "which is exactly why it is a separate entry point.",
    ),

    Result(
        id="HEBB-OUTER-PRODUCT",
        status=Status.PROVED,
        claim="The Hebbian co-firing count is a SUM OF RANK-1 OUTER PRODUCTS: "
              "count = SUM_t x_{t-1} x_t^T, with x_t the 0/1 winner indicator "
              "at round t. Restricted to the rows and columns a window of T "
              "rounds touches it is a matrix product R^T C of two thin 0/1 "
              "matrices; equivalently count[i,j] = popcount(rm[i] & cm[j]) "
              "with rm, cm the per-neuron round bitmasks.",
        source="Immediate from the update rule: `w[i,j] *= (1+beta)` fires "
               "exactly when i is a source winner and j a target winner, so "
               "the exponent counts co-firing rounds and nothing else.",
        preconditions=("plasticity purely MULTIPLICATIVE, so the exponent is "
                       "the whole state",
                       "the pairing must match the engine: source winners x "
                       "target winners, i.e. prev x new for a recurrent fiber"),
        evidence=("research/experiments/gpu_writeback_gemm_prototype.py",),
        implemented_by=("neural_assemblies.core.torch_engine._batched",
                        "neural_assemblies.core.torch_engine._fused_cuda"),
        caveat="An event carries 2k numbers, so materialising the k x k cross "
               "product writes the same information k/2 times over -- a "
               "redundancy created BEFORE any sort is reached, which is why "
               "sorting strategies cannot recover it. The two forms answer "
               "different questions: the GEMM materialises a block, the "
               "popcount evaluates one cell. `batched_project_independent` "
               "uses new x new rather than prev x new and is therefore NOT "
               "interchangeable with the engine.",
    ),
    Result(
        id="DRIVE-SPLIT",
        status=Status.PROVED,
        claim="With a Bernoulli 0/1 base B and G[i,j] = chain(1, count[i,j]), "
              "the drive splits as "
              "1_S^T (B (.) G) = 1_S^T B + SUM_{i in S, j} B[i,j] D[i,j] "
              "where D = G - 1 is nonzero only on potentiated cells. The "
              "correction is a SPARSE MATVEC over D restricted to |S| = k "
              "rows, so its intrinsic cost is the number of stored deviations "
              "in those rows and nothing else.",
        source="Algebraic identity, given that the base is 0/1: an ABSENT cell "
               "stays absent however often it is potentiated, and a present "
               "one starts at exactly 1.0, so chain(base, c) = base * tab[c].",
        preconditions=("base strictly 0/1 -- recruitment OVERRIDES cells, so "
                       "the effective base is 1 wherever a cell was written",
                       "tab replayed with the engine's own per-step "
                       "multiply-and-clip, NOT min((1+beta)^c, w_max)"),
        evidence=("research/experiments/gpu_hashed_deviations_prototype.py",),
        implemented_by=("neural_assemblies.core.torch_engine._fused_cuda",),
        caveat="A representation that cannot say WHICH cells are nonzero must "
               "visit every (row, column) pair: the bitmask form costs "
               "O(k n W) with W = ceil(rounds/64) against the intrinsic "
               "O(sum_{i in S} nnz_i), a ratio n^2 T / (64 k^2) that is "
               "INDEPENDENT of M -- 8894x at n=16000, k=60, T=8. Changing the "
               "representation changes the SUMMATION ORDER, and float32 "
               "addition is not associative, so ties at the k-WTA bar can "
               "flip ([[KWTA-TIE-FRAGILE]]).",
    ),
    Result(
        id="CAP-RATIO",
        engine="hashed AssemblyMemory / exact count-then-apply path",
        status=Status.MEASURED,
        sensitivity_checks=(SensitivityCheck(
            artifact="research/results/runs/memory.capacity-scaling/nk-replay-20260913/results.json",
            sample_path="run/seeds",
            treatment_path="observations/cells/1/checkpoints/128/rank1",
            control_path="observations/cells/0/checkpoints/128/rank1",
            relation="all-greater", minimum_effect=0.9,
            mechanism="the same n = 4000 at k = 30 against k = 60 (ratio 133 against 67), unrefracted, at M = 128: every k-30 brain still recalls (at least 0.938) where every k-60 brain has collapsed (at most 0.031); the ceilings are 267.0 and 74.7, ratio 3.58 against the law's 4",
        ),),
        sensitivity_gap="The retained check is the ratio contrast at one n; "
                        "the equal-ratio cell at n = 8000 in the same run "
                        "reproduces the n = 4000 ceiling within 3.8% but is a "
                        "cliff interpolation (no grid point inside its "
                        "transition band), and the n = 16000 cell of CS1 has "
                        "no run record.",
        claim="The assembly-capacity ceiling M* is a function of n/k ALONE, "
              "not of n and k separately.",
        source="Held-out test registered in "
               "research/notes/memory/PREREG_capacity_nk_law.md (bar CS1) before the "
               "data existed. Holding n/k = 66.67 while n varies four-fold "
               "gives M* = 68.8 / 73.1 / 67.6 at n = 4000 / 8000 / 16000 on "
               "the exact count-then-apply path (engine-parity verified) -- "
               "constant to +/-4%, where any law M* = f(n) predicts ~4x. All "
               "three cells uncensored (fill 0.69-0.76), and all inside the "
               "band registered before the data existed.",
        preconditions=("in regime, kp >= 3 ln n [[SEQ-REGIME]]",
                       "ceiling read from the CURVE, gated on distinctness",
                       "fill at the ceiling below 0.95, else the tiling limit "
                       "is what is being measured",
                       "AT FIXED p AND beta: M* moved 1.6x across p in "
                       "[0.3, 0.7] and ~6x across beta in [0.05, 0.20] at "
                       "fixed n/k = 40 (PREREG_crosstalk_mechanism.md), so "
                       "the ratio law holds within an operating point, not "
                       "across them"),
        evidence=("research/experiments/seq_capacity_scaling.py",),
        evidence_refs=(
            EvidenceRef("research/results/runs/memory.capacity-scaling/capacity-record-consumed-20260910/results.json", "artifact"),
            EvidenceRef("research/results/runs/memory.capacity-scaling/nk-replay-20260913/results.json", "artifact"),
            EvidenceRef("research/notes/memory/PREREG_capacity_nk_law.md", "registration"),
            EvidenceRef("research/experiments/seq_capacity_scaling.py", "producer"),
        ),
        caveat="This result survived a WRONG RETRACTION: an intermediate "
               "CSR deviation store applied the potentiation table per count "
               "FRAGMENT, and (tab[c0]-1)+(tab[c1]-1) != tab[c0+c1]-1, "
               "inflating ceilings ~25% (88.3/92.1/83.2); those numbers were "
               "briefly recorded here as the verified ones. Multi-episode "
               "ENGINE parity exposed it (rel 2e-3) and the exact path lands "
               "back on the first run's values. The EXPONENT remains "
               "unestablished: b = 2.19 +/- 0.05, above 2, NO mechanism -- "
               "reported, never quoted. A synapse bound "
               "M ~ n^2 p / (k ln(n/k)) is REFUTED by CS1: it is not a "
               "function of n/k alone -- and so is plain second-order "
               "CROSSTALK, refuted by registered test: the crosstalk ratio "
               "cancels p and beta, but M* ~ p^-0.6 and ~ beta^-1.3 at fixed "
               "n/k. No mechanism is adopted. Operational rule from the wrong "
               "retraction: when two implementations disagree, a test they "
               "BOTH pass verifies neither -- arbitrate with engine parity on "
               "the DRIVE.",
    ),
    Result(
        id="CAP-ANCHOR-RATIO",
        engine="hashed_assembly_memory, capacity-scaling protocol (retained paired-anchor replay of 20 brains; the original excursions are logs without run records)",
        status=Status.MEASURED,
        sensitivity_checks=(SensitivityCheck(
            artifact="research/results/runs/memory.capacity-scaling/anchor-pair-20260912/results.json",
            sample_path="run/seeds",
            treatment_path="observations/conditions/anchor_high/cells/B~14000~1100/checkpoints/48/rank1",
            control_path="observations/conditions/anchor_low/cells/B~14000~1100/checkpoints/48/rank1",
            relation="all-greater", minimum_effect=0.3,
            mechanism="stimulus anchor 200 against 100 at (4000, 100), unrefracted, at M = 48: the largest checkpoint every anchor-200 brain still recalls (1.000), where anchor-100 has collapsed (at most 0.03)",
        ),),
        sensitivity_gap="The retained check is the anchor contrast at one cell; "
                        "the p and beta excursions (A1-A3) and their computed "
                        "compensation remain logs without a run record.",
        claim="The capacity ceiling is set at FORMATION by the ratio of the "
              "stimulus anchor to the trained recurrent pull. Density p and "
              "gain beta enter through that ratio, so an excursion in either "
              "is undone by a computed change in anchor size.",
        source="This repository; PREREG_formation_interference.md (F2) and "
               "PREREG_anchor_ratio.md (A1-A3).",
        evidence=("F2: anchor 100 -> 200 lifts M* 23.5 -> 67.1 (2.86x), while "
                  "retrieval never sees the stimulus",
                  "A1-A3: beta 0.20 / p 0.7 / p 0.3 excursions with "
                  "uncompensated M* 8.0 / 17.6 / 28.8 land at 29.4 / 25.0 / "
                  "25.3 once the anchor is set from the measured exponents -- "
                  "a 3.6x spread collapses to 1.18x, all within 25% of 23.5",
                  "F3: past the cliff, erosion is retroactive and diffuse "
                  "(early items fail worst), not one-shot capture",
                  "anchor-pair-20260912 (20 brains, half-cue rank-1 readout, "
                  "PREREG_anchor_ratio.md sensitivity replay, AN-1 to AN-3 PASS): "
                  "M* 23.3 at anchor 100 against 70.5 at anchor 200, ratio 3.02 "
                  "(the law's (200/100)^1.52 = 2.87); at M = 48 every anchor-200 "
                  "brain recalls 1.000 while anchor-100 is at most 0.03"),
        evidence_refs=(
            EvidenceRef("research/notes/memory/PREREG_anchor_ratio.md", "registration"),
            EvidenceRef("research/notes/memory/PREREG_formation_interference.md", "registration"),
            EvidenceRef("research/results/runs/memory.capacity-scaling/anchor-pair-20260912/results.json", "artifact",
                        "paired anchors 100 and 200 at one cell; not the p and beta excursions"),
            EvidenceRef("research/experiments/seq_capacity_scaling.py", "producer"),
        ),
        provenance_gap="the A1-A3 excursion outcomes are logs without a run record; the anchor contrast itself has one",
        preconditions=("n=4000, k=100, T=8, w_max=20, arm B, 16 brains; "
                       "one operating-point neighbourhood",
                       "exponents s^1.52 p^-0.6 beta^-1.3 are two- and "
                       "three-point fits used ONLY to place cells; none is "
                       "adopted"),
        caveat="A1 sits on the numeric band's edge (29.4 vs < 29.4) and "
               "passes on the registered prose criterion; all three "
               "compensated cells overshoot upward, so the anchor exponent is "
               "probably slightly high. The (n/k)^2 dependence of CAP-RATIO is "
               "the pull's chance-overlap term and is NOT derived here.",
    ),
    Result(
        id="CAP-CLIFF",
        engine="hashed AssemblyMemory / exact count-then-apply path",
        status=Status.MEASURED,
        sensitivity_checks=(SensitivityCheck(
            artifact="research/results/runs/memory.capacity-scaling/"
                     "capacity-cliff-sensitivity-20260911/results.json",
            sample_path="observations/cells/*/seeds",
            treatment_path="observations/cells/*/checkpoints/192/rank1",
            control_path="observations/cells/*/checkpoints/512/rank1",
            relation="all-greater", minimum_effect=0.5,
            mechanism="safe-load versus overloaded capacity readout",
        ),),
        claim="Capacity failure is a CLIFF, not a slope: past the ceiling the "
              "assemblies shatter rather than degrading gracefully.",
        source="research/experiments/seq_capacity_scaling.py, on the exact "
               "count-then-apply path (engine-parity verified). At n=8000, "
               "k=60, 16 brains: M=256 gives rank-1 0.938 at pairwise overlap "
               "1.37x chance with every assembly distinct; M=320 gives 0.486; "
               "M=384 gives 0.014 at overlap 4.75x. One doubling (256 -> 512) "
               "takes rank-1 from 0.938 to 0.000. Ceiling M* = 284 at fill "
               "0.944, uncensored.",
        preconditions=("half-cue rank-1 readout against ALL M stored items",
                       "distinctness gate applied, so a collapsed set scores 0"),
        evidence=("research/experiments/seq_capacity_scaling.py",),
        evidence_refs=(
            EvidenceRef("research/results/runs/memory.capacity-scaling/capacity-record-consumed-20260910/results.json", "artifact",
                        "maintained-run replay at (4000,60), not the registered cliff cell"),
            EvidenceRef("research/results/runs/memory.capacity-scaling/capacity-cliff-sensitivity-20260911/results.json", "artifact",
                        "fresh exact-path reproduction and retained sensitivity vectors"),
            EvidenceRef("research/experiments/seq_capacity_scaling.py", "producer"),
        ),
        caveat="There is no soft capacity margin to trade against: a design "
               "must know where the ceiling is and stay under it. Sharing "
               "itself is healthy -- at M=256 the load Mk/n is 1.9, nearly "
               "two assemblies per neuron, with overlap still 1.37x chance -- "
               "so the cliff is not caused by sharing. The transition occupies "
               "roughly one 1.5x step in M (256 -> 384). A 20-brain maintained-"
               "path reproduction gave 0.909 / 0.433 / 0.0188 at those three "
               "checkpoints; its interpolated M* = 310.1 remains fill-censored.",
    ),
]

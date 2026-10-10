"""The refracted sequence memory (PREREG_refraction_memory.md): its laws, its reuse edges, and what write-time separation and sleep do to them.

Moved from neural_assemblies/theory.py unchanged; theory.py lists these in its order."""
from __future__ import annotations

from typing import List

from .types import EvidenceRef, Result, SensitivityCheck, Status

CLAIMS: List[Result] = [
    Result(
        id="SEQUENCE-MEMORY-ROBUSTNESS",
        engine="hashed AssemblyMemory, store_sequence one round per element, refraction 0.5 beta recovering over 33 rounds, beta = theta",
        status=Status.MEASURED,
        claim="The refracted sequence memory's length limit is a budget of elements, spent "
              "alike on one long sequence or many short ones (0.81-1.08 of the single-"
              "sequence limit); replay loses nothing with up to 7% of its MOST DRIVEN winners "
              "replaced by random neurons at every step, derails at a horizon at 10% that a "
              "twice-larger area pushes out about sixfold, and starts from its strongest half "
              "cue a quarter wrong. Under UNIFORMLY random replacement (Amendment 36) it is "
              "markedly more tolerant: at 10% the smaller area replays 2.5 times further "
              "(103.6 against 41.2 steps) and the larger does not derail in 400 steps (20 of "
              "20 brains); a random half cue works as well as the strongest.",
        source="PREREG_refraction_memory.md Amendment 34 (N1-N4 PASS); Amendment 36 "
               "(U1-U4 PASS)",
        preconditions=("noise = replacement of the first slots of each replay step's "
                       "winners, which the k-WTA orders strongest first (Amendment 36 erratum); "
                       "the cue is the strongest half of the first state",
                       "two cells, k = 60, p = 0.5"),
        evidence=("research/notes/memory/PREREG_refraction_memory.md#amendment-34-result-2026-10-07",
                  "research/notes/memory/PREREG_refraction_memory.md#amendment-36-result-2026-10-08"),
        evidence_refs=(
            EvidenceRef("research/results/runs/memory.robustness/robustness-20261007/results.json", "artifact",
                        "Amendment 34: many sequences, activity noise, corrupted cues, two cells, 20 brains"),
            EvidenceRef("research/results/runs/memory.noise/noise-20261008/results.json", "artifact",
                        "Amendment 36: top-slot and uniformly random noise, strongest and random half cues, two cells, 20 brains"),
        ),
        sensitivity_checks=(SensitivityCheck(
            artifact="research/results/runs/memory.robustness/robustness-20261007/results.json",
            sample_path="run/seeds",
            treatment_path="observations/cells/4000~160~10.5/noise/0.05/values",
            control_path="observations/cells/4000~160~10.5/noise/0.2/values",
            relation="all-greater", minimum_effect=300,
            mechanism="replay survives 5% activity noise per step and derails within a few steps at 20%, at (4000, 60, 0.5) (Amendment 34)",
        ), SensitivityCheck(
            artifact="research/results/runs/memory.robustness/robustness-20261007/results.json",
            sample_path="run/seeds",
            treatment_path="observations/cells/8000~160~10.5/noise/0.05/values",
            control_path="observations/cells/8000~160~10.5/noise/0.2/values",
            relation="all-greater", minimum_effect=300,
            mechanism="replay survives 5% activity noise per step and derails within a few steps at 20%, at (8000, 60, 0.5) (Amendment 34)",
        )),
        caveat="ERRATUM (Amendment 36): Amendment 34's noise was described as uniform "
               "random replacement but replaced the most driven winners (the k-WTA orders "
               "them strongest first), nearly all true members; Amendment 36 measured both. "
               "The mechanism of the horizon is not identified. Synaptic noise and noise "
               "during writing are untested.",
    ),
    Result(
        id="SEQUENCE-LOAD-LAW",
        engine="hashed AssemblyMemory, store_sequence one round per element, refraction 0.5 beta recovering over 64 rounds, beta = theta",
        status=Status.MEASURED,
        claim="One area's single-sequence replay fails at a critical interference load: "
              "with rho = L k ln n / (n^2 p), every cell with k p >= 3 ln n and n/k <= 150 replays a "
              "sequence whole from a random half cue on every brain up to rho ~ 0.11, and "
              "on none a factor 1.07-1.15 past its 90% point. Registered from an "
              "exploratory survey (rho_50 = 0.142, 0.119-0.171 at seven cells) and "
              "confirmed at two cells never run: rho_50 = 0.141 at (6000, 90, 0.4) and "
              "0.115 at (12000, 80, 0.5), rho_90 = 0.132 and 0.111. Below the floor it "
              "fails earlier (0.086, 0.089).",
        source="PREREG_refraction_memory.md Amendment 37 (R1-R3 PASS); its constant fails "
               "at n/k = 300, Amendment 38 (K1, D1, S1 FAIL; S2 PASS)",
        preconditions=("k p >= 3 ln n; a recovering refraction (tau = 64) at beta = theta; "
                       "one sequence per area, noiseless one-round recall",
                       "n/k <= 150: nine in-regime cells, n 2000-16000; at n/k = 300 "
                       "the cliff is at rho = 0.081 (Amendment 38)"),
        evidence=("research/notes/memory/PREREG_refraction_memory.md#amendment-37-result-2026-10-08",
                  "research/notes/memory/PREREG_refraction_memory.md#amendment-38-result-2026-10-08"),
        evidence_refs=(
            EvidenceRef("research/results/runs/memory.load-law/load-law-20261008/results.json", "artifact",
                        "Amendment 37: the load ladder at two held-out cells and one out-of-regime cell, 20 brains"),
            EvidenceRef("research/results/runs/memory.load-drift/load-drift-20261008/results.json", "artifact",
                        "Amendment 38: the load ladder at two cells with n/k = 300, 20 brains"),
        ),
        sensitivity_checks=(SensitivityCheck(
            artifact="research/results/runs/memory.load-law/load-law-20261008/results.json",
            sample_path="run/seeds",
            treatment_path="observations/cells/6000~190~10.4/ladder/1856/steps",
            control_path="observations/cells/6000~190~10.4/ladder/2862/steps",
            relation="all-greater", minimum_effect=1000,
            mechanism="every brain replays a sequence whole at rho = 0.101 and derails within ~70 steps at rho = 0.156, at (6000, 90, 0.4) (Amendment 37)",
        ), SensitivityCheck(
            artifact="research/results/runs/memory.load-law/load-law-20261008/results.json",
            sample_path="run/seeds",
            treatment_path="observations/cells/12000~180~10.5/ladder/8866/steps",
            control_path="observations/cells/12000~180~10.5/ladder/11498/steps",
            relation="all-greater", minimum_effect=1000,
            mechanism="every brain replays a sequence whole at rho = 0.093 and none at rho = 0.120, at (12000, 80, 0.5) (Amendment 37)",
        )),
        caveat="SEMI-EMPIRICAL: the variable is derived, the constant is fitted to the "
               "survey; a parameter-free reduced model puts the cliff about twice too high. "
               "THE CONSTANT IS NOT A CONSTANT: rho_50 is lower at n/k >= 133 (0.115-0.123) "
               "than at n/k <= 67 (0.131-0.171), and at n/k = 300 it is 0.081 at both cells "
               "of Amendment 38 -- below the constant, below a post hoc power-law drift, and "
               "below the safe rule rho <= 0.09, which does not hold there. Do not "
               "extrapolate past n/k = 150. Many sequences, noise, tau other than "
               "64 and the Hebbian (non-refracted) memory are not covered.",
    ),
    Result(
        id="RECOVERY-SCALES-WITH-AREA",
        engine="hashed AssemblyMemory, store_sequence one round per element, refraction 0.5 beta recovering over tau rounds, beta = theta",
        status=Status.MEASURED,
        claim="The refraction's recovery time must scale with the steps between a "
              "neuron's uses (n/k): at a fixed tau = 64 the single-sequence cliff falls "
              "to rho = L k ln n / (n^2 p) = 0.081 at n/k = 300 (three cells), below the "
              "load law's safe rule; with tau = n/k / 2 it rises 29% at n/k = 300 and "
              "18% at n/k = 200 on the same brains, to rho_50 = 0.105 and 0.114, and "
              "replay is reliable again to rho = 0.10. Too slow a recovery (tau ~ 1.7 "
              "n/k, probed) brings back the tiling deadline. A sizing rule for a "
              "sequence area: tau = n/k / 2, L k ln n / (n^2 p) <= 0.09.",
        source="PREREG_refraction_memory.md Amendment 39 (T1-T4 PASS); Amendment 38 "
               "(the tau = 64 cliff at n/k = 300)",
        preconditions=("k p >= 3 ln n; beta = theta, refraction 0.5 beta; one sequence "
                       "per area, noiseless one-round recall from a random half cue",
                       "two cells judged (n/k = 300, 200), one probed; the window of good "
                       "tau (0.43-0.85 n/k) from one exploratory cell"),
        evidence=("research/notes/memory/PREREG_refraction_memory.md#amendment-39-result-2026-10-08",
                  "research/notes/memory/PREREG_refraction_memory.md#amendment-38-result-2026-10-08"),
        evidence_refs=(
            EvidenceRef("research/results/runs/memory.load-tau/load-tau-20261008/results.json", "artifact",
                        "Amendment 39: tau = n/k / 2 against tau = 64, two cells, the same 20 brains"),
            EvidenceRef("research/results/runs/memory.load-drift/load-drift-20261008/results.json", "artifact",
                        "Amendment 38: the tau = 64 cliff at n/k = 300, two cells, 20 brains"),
        ),
        sensitivity_checks=(SensitivityCheck(
            artifact="research/results/runs/memory.load-tau/load-tau-20261008/results.json",
            sample_path="run/seeds",
            treatment_path="observations/cells/21000~170~10.5/tau/150/ladder/29288/steps",
            control_path="observations/cells/21000~170~10.5/tau/64/ladder/29288/steps",
            relation="all-greater", minimum_effect=1000,
            mechanism="at rho = 0.093 and n/k = 300 every brain replays the whole sequence with tau = n/k / 2 and derails within a few steps with tau = 64 (Amendment 39)",
        ), SensitivityCheck(
            artifact="research/results/runs/memory.load-tau/load-tau-20261008/results.json",
            sample_path="run/seeds",
            treatment_path="observations/cells/16000~180~10.45/tau/100/ladder/15011/steps",
            control_path="observations/cells/16000~180~10.45/tau/64/ladder/15011/steps",
            relation="all-greater", minimum_effect=1000,
            mechanism="at rho = 0.101 and n/k = 200 every brain replays whole with tau = n/k / 2 and none with tau = 64 (Amendment 39)",
        )),
        caveat="The rule's constant (one half) was chosen from one exploratory cell's "
               "window (tau 128 and 256 equal at n/k = 300); the optimum is not located. "
               "Even under the rule the critical load at n/k >= 200 (0.105-0.114) is "
               "below the 0.142 of smaller areas: the residual n/k dependence is not "
               "explained. Many sequences, noise and the write-side mechanism (whether "
               "fast recovery acts through over-dispersed reuse) are untested.",
    ),
    Result(
        id="SEQUENCE-BUDGET-ANY-SPLIT",
        engine="hashed AssemblyMemory, one store_sequence per sequence, refraction 0.5 beta recovering over tau = n/k / 2 rounds, beta = theta",
        status=Status.MEASURED,
        claim="One area's sequence budget is safe however it is split: with tau = "
              "n/k / 2, every sequence of 16, 64 or thousands of elements replays whole "
              "while the TOTAL load L k ln n / (n^2 p) <= 0.09 (rho_90 0.111-0.132 in "
              "every arm). Past the cliff one long sequence fails all or none, while many "
              "short ones fail one by one with every brain losing the same share (each "
              "sequence an independent trial), outlasting the single cliff by 13-20% in "
              "rho_50.",
        source="PREREG_refraction_memory.md Amendment 40 (A1-A3 PASS)",
        preconditions=("k p >= 3 ln n, n/k = 100 and 171; sequences written one after "
                       "another, not linked, the refraction carrying over",
                       "noiseless one-round replay from a random half of each "
                       "sequence's first element"),
        evidence=("research/notes/memory/PREREG_refraction_memory.md#amendment-40-result-2026-10-08",),
        evidence_refs=(
            EvidenceRef("research/results/runs/memory.load-many/load-many-20261008/results.json", "artifact",
                        "Amendment 40: one sequence, l = 64 and l = 16 at equal total load, two cells, 20 brains"),
        ),
        sensitivity_checks=(SensitivityCheck(
            artifact="research/results/runs/memory.load-many/load-many-20261008/results.json",
            sample_path="run/seeds",
            treatment_path="observations/cells/8000~180~10.5/arms/16/ladder/6352/whole",
            control_path="observations/cells/8000~180~10.5/arms/single/ladder/6351/whole",
            relation="all-greater", minimum_effect=0.5,
            mechanism="at rho = 0.143 every brain replays most of its 16-element sequences whole and not the single sequence of the same total load, at (8000, 80, 0.5) (Amendment 40)",
        ), SensitivityCheck(
            artifact="research/results/runs/memory.load-many/load-many-20261008/results.json",
            sample_path="run/seeds",
            treatment_path="observations/cells/12000~170~10.5/arms/16/ladder/14336/whole",
            control_path="observations/cells/12000~170~10.5/arms/single/ladder/14330/whole",
            relation="all-greater", minimum_effect=0.5,
            mechanism="at rho = 0.131 every brain replays three quarters of its 16-element sequences whole and not the single sequence, at (12000, 70, 0.5) (Amendment 40)",
        )),
        caveat="The split into cue capture (c) and per-step hazard (h) is post hoc, "
               "from two lengths. Sequences sharing elements, noise during replay and "
               "sequences linked across areas are not covered; two cells.",
    ),
    Result(
        id="RECOVERY-PEAK-AT-SMALL-AREAS",
        engine="hashed AssemblyMemory, store_sequence one round per element, refraction 0.5 beta recovering over tau rounds, beta = theta",
        status=Status.MEASURED,
        claim="The single-sequence critical load peaks when the refraction recovers over "
              "the steps between a neuron's uses, tau = n/k, but only in small areas: "
              "against tau = n/k / 2 the gain in rho_50 is 1.65 at n/k = 20, 1.24 at 50, "
              "0.98 at 100 and 0.91 at 200 (four cells differing in n/k alone). Under "
              "tau = n/k / 2 the critical load is nearly constant, rho_50 = 0.122 at "
              "n/k = 20-100 and 0.112 at 200. Compiler rule: tau = n/k to n/k ~ 50, "
              "n/k / 2 from ~ 100; replay is reliable to rho = 0.09 under either.",
        source="PREREG_refraction_memory.md Amendment 41 (P1-P4 PASS)",
        preconditions=("k p ~ 35, k p / ln n ~ 4: the linear regime, mean count per synapse "
                       "pair L (k/n)^2 below ~ 0.5 (probed: at k p / ln n = 24 replay fails "
                       "at its first step)",
                       "one sequence per area, noiseless one-round replay from a random half cue"),
        evidence=("research/notes/memory/PREREG_refraction_memory.md#amendment-41-result-2026-10-08",),
        evidence_refs=(
            EvidenceRef("research/results/runs/memory.load-peak/load-peak-20261008/results.json", "artifact",
                        "Amendment 41: tau = n/k against n/k / 2 at n/k = 20, 50, 100, 200, the same 20 brains"),
        ),
        sensitivity_checks=(SensitivityCheck(
            artifact="research/results/runs/memory.load-peak/load-peak-20261008/results.json",
            sample_path="run/seeds",
            treatment_path="observations/cells/6000~1300~10.12/arms/1.0/ladder/221/steps",
            control_path="observations/cells/6000~1300~10.12/arms/0.5/ladder/221/steps",
            relation="all-greater", minimum_effect=100,
            mechanism="at rho = 0.134 and n/k = 20 every brain replays whole with tau = n/k and derails within ~ 18 steps with tau = n/k / 2 (Amendment 41)",
        ), SensitivityCheck(
            artifact="research/results/runs/memory.load-peak/load-peak-20261008/results.json",
            sample_path="run/seeds",
            treatment_path="observations/cells/5000~1100~10.35/arms/1.0/ladder/1371/steps",
            control_path="observations/cells/5000~1100~10.35/arms/0.5/ladder/1371/steps",
            relation="all-greater", minimum_effect=500,
            mechanism="at rho = 0.133 and n/k = 50 every brain replays whole with tau = n/k and none with tau = n/k / 2 (Amendment 41)",
        )),
        caveat="The peak's location (1.1-1.25 n/k at one probed cell) and the crossover "
               "(between n/k = 50 and 100) are coarse. Why large areas lose the peak is "
               "read, not tested: their tau = n/k failures come at step ~ n/k, the tiling "
               "deadline's signature. One kp; many sequences and noise not covered.",
    ),
    Result(
        id="RECURRING-WORDS-CODED-AS-TOKENS",
        engine="hashed AssemblyMemory, one store_sequence per 16-element sequence, refraction 0.5 beta recovering over tau by the Amendment 41 rule, beta = theta",
        status=Status.MEASURED,
        claim="Inside the load law's safe budget one area holds sequences whose elements "
              "recur (up to 5 uses per word) at no cost, and replays them through 5% "
              "uniform activity noise per step without loss, because it codes every "
              "occurrence of a word apart: two occurrences share ~ 0.1 of their neurons "
              "(a type code would share most), still 50-100 times two different words' "
              "overlap (0.000-0.002) -- a token code with a trace of the type. Noise and "
              "load multiply: at 10% noise the loss grows from 3-10% at rho = 0.05 to "
              "34-63% at 0.11.",
        source="PREREG_refraction_memory.md Amendment 42 (W1, W2, N1, N2 PASS)",
        preconditions=("words drawn i.i.d. per brain from a vocabulary of L / U; 16-element "
                       "sequences, one store_sequence each",
                       "two cells (n/k = 40, 83); uniform replacement noise during replay only"),
        evidence=("research/notes/memory/PREREG_refraction_memory.md#amendment-42-result-2026-10-08",),
        evidence_refs=(
            EvidenceRef("research/results/runs/memory.reuse-noise/reuse-noise-20261008/results.json", "artifact",
                        "Amendment 42: recurring words and replay noise, two cells, 20 brains"),
        ),
        sensitivity_checks=(SensitivityCheck(
            artifact="research/results/runs/memory.reuse-noise/reuse-noise-20261008/results.json",
            sample_path="run/seeds",
            treatment_path="observations/cells/4000~1100~10.35/reuse/0.08/5/same",
            control_path="observations/cells/4000~1100~10.35/reuse/0.08/5/different",
            relation="all-greater", minimum_effect=0.05,
            mechanism="two occurrences of one word share ~ 0.1 of their neurons in every brain, two different words ~ 0.001, at (4000, 100, 0.35) (Amendment 42)",
        ), SensitivityCheck(
            artifact="research/results/runs/memory.reuse-noise/reuse-noise-20261008/results.json",
            sample_path="run/seeds",
            treatment_path="observations/cells/10000~1120~10.3/reuse/0.08/5/same",
            control_path="observations/cells/10000~1120~10.3/reuse/0.08/5/different",
            relation="all-greater", minimum_effect=0.05,
            mechanism="the same token-with-a-type-trace code at (10000, 120, 0.3) (Amendment 42)",
        )),
        caveat="Heavy reuse (20 uses per word) costs up to 36% and fails brain by brain, "
               "apparently by each brain's draw (repeated bigrams, within-sequence "
               "repeats): untested. Whether a readout can use the type trace is untested. "
               "Noise during writing is not covered.",
    ),
    Result(
        id="LENGTH-COSTS-LOGARITHMICALLY",
        engine="hashed AssemblyMemory, one store_sequence per sequence, refraction 0.5 beta recovering over tau = n/k / 2 rounds, beta = theta",
        status=Status.MEASURED,
        claim="The critical load of a sequence area depends on the length of what it "
              "holds, logarithmically: at equal total load, rho_50 falls through sequences "
              "of 16, 64, 256, 1024 elements and one of the whole load (~10^4) at both "
              "cells, linearly in ln l (0.0027 and 0.0032 per e-fold, no point 0.002 off "
              "the line, fitted after the run). A capture-and-constant-hazard model fitted "
              "on 16 and 64 elements places 256 and 1024 within 5% in rho_50 but a "
              "whole-load sequence 5-10% low: long sequences outlive a constant hazard.",
        source="PREREG_refraction_memory.md Amendment 43 (K1, K2, K4 PASS; K3 FAIL)",
        preconditions=("k p >= 3 ln n, n/k = 143 and 160; tau = n/k / 2",
                       "noiseless one-round replay from a random half cue"),
        evidence=("research/notes/memory/PREREG_refraction_memory.md#amendment-43-result-2026-10-08",),
        evidence_refs=(
            EvidenceRef("research/results/runs/memory.load-hazard/load-hazard-20261008/results.json", "artifact",
                        "Amendment 43: five sequence lengths at equal total load, two cells, 20 brains"),
        ),
        sensitivity_checks=(SensitivityCheck(
            artifact="research/results/runs/memory.load-hazard/load-hazard-20261008/results.json",
            sample_path="run/seeds",
            treatment_path="observations/cells/10000~170~10.5/arms/16/ladder/10240/whole",
            control_path="observations/cells/10000~170~10.5/arms/single/ladder/10240/whole",
            relation="all-greater", minimum_effect=0.5,
            mechanism="at rho = 0.132 every brain replays most of its 16-element sequences and not the single whole-load sequence, at (10000, 70, 0.5) (Amendment 43)",
        ), SensitivityCheck(
            artifact="research/results/runs/memory.load-hazard/load-hazard-20261008/results.json",
            sample_path="run/seeds",
            treatment_path="observations/cells/8000~150~10.7/arms/16/ladder/14336/whole",
            control_path="observations/cells/8000~150~10.7/arms/single/ladder/14336/whole",
            relation="all-greater", minimum_effect=0.5,
            mechanism="at rho = 0.144 the same at (8000, 50, 0.7) (Amendment 43)",
        )),
        caveat="The ln l slope is a post hoc fit over five lengths at two cells. The "
               "registered ladder was coarsened by rounding to multiples of 1024 (9 and 12 "
               "distinct loads of 17); at the first cell 256, 1024 and one sequence cross inside one ladder "
               "step, so their order there rests on interpolation. Why long sequences outlive a constant hazard (an "
               "early transient from the cue is the candidate) is untested.",
    ),
    Result(
        id="TRANSITION-REPETITION-BREAKS-REPLAY",
        engine="hashed AssemblyMemory, one store_sequence per 16-element sequence, refraction 0.5 beta recovering over tau by the Amendment 41 rule, beta = theta",
        status=Status.MEASURED,
        claim="Inside the load law's safe budget (rho = 0.05), a token store loses word-level "
              "replay entirely -- every sequence, every brain, two cells -- once each "
              "transition (bigram) is stored about five times, whatever the words' frequency "
              "(10, 20 or 40 uses per word), while random successors at the same frequency keep "
              "0.77 and 0.95 of sequences whole; repeated transitions also raise the overlap of "
              "a word's tokens (+0.05-0.06). Repetition of transitions, not recurrence of "
              "elements, is the dominant cost of reuse in this range.",
        source="PREREG_refraction_memory.md Amendment 44 (G1, G4 PASS; G3 PASS but vacuous; G2 FAIL)",
        preconditions=("per-brain grammars with b successors per word; 16-element random walks",
                       "word-level score: the read-out's nearest stored token's word"),
        evidence=("research/notes/memory/PREREG_refraction_memory.md#amendment-44-result-2026-10-08",),
        evidence_refs=(
            EvidenceRef("research/results/runs/memory.reuse-grammar/reuse-grammar-20261008/results.json", "artifact",
                        "Amendment 44: uses per word crossed with repeats per bigram, two cells, 20 brains"),
        ),
        sensitivity_checks=(SensitivityCheck(
            artifact="research/results/runs/memory.reuse-grammar/reuse-grammar-20261008/results.json",
            sample_path="run/seeds",
            treatment_path="observations/cells/12000~180~10.45/arms/U20~1b2/same",
            control_path="observations/cells/12000~180~10.45/arms/U20~1bV/same",
            relation="all-greater", minimum_effect=0.02,
            mechanism="in every brain a word's tokens overlap more when each transition is stored ~10 times than with random successors, at (12000, 80, 0.45) (Amendment 44)",
        ),),
        caveat="Where between one and five repetitions the failure sets in is unmeasured, and "
               "whether recurrence matters at fixed repetition is untested (every repeated arm "
               "is on the floor; G3 vacuous). Recurrence alone at twenty uses fails whole brains "
               "(1 and 4 of 20), unexplained by the draws' bigram statistics. Small vocabularies "
               "(33-430 words) and 16-element walks: not a corpus.",
    ),
    Result(
        id="REUSE-BUDGET-TWO-EDGES",
        engine="hashed AssemblyMemory, one store_sequence per 16-element walk, refraction 0.5 beta recovering over tau by the Amendment 41 rule, beta = theta",
        status=Status.MEASURED,
        claim="A token store's reuse budget has two edges, at a safe total load (rho = 0.05) "
              "and two new cells: a REPETITION edge -- word-level replay 1.00 at one repeat "
              "per transition, ~0.35-0.68 at 2.8, ~0 at 3.7 -- and a RECURRENCE edge -- with "
              "random successors, 1.00 at 10-20 uses per word, ~0.5 at 40, ~0-0.09 at 60, "
              "failing whole brains at a time. Both lie where an exploratory probe at a third "
              "cell put them (n/k 117-150). The product of the two marginal curves predicts "
              "mixed arms to a mean error of 0.05-0.08 (registered tolerance 0.15 missed at one "
              "arm of ten by 0.006).",
        source="PREREG_refraction_memory.md Amendment 45 (S1 PASS; S2 FAIL)",
        preconditions=("per-brain grammars, 16-element walks, word-level score (nearest stored token's word)",
                       "n/k = 117-150 at tau = n/k / 2; a cell at n/k = 40, tau = n/k, is more tolerant (probed)"),
        evidence=("research/notes/memory/PREREG_refraction_memory.md#amendment-45-result-2026-10-08",),
        evidence_refs=(
            EvidenceRef("research/results/runs/memory.reuse-budget/reuse-budget-20261008/results.json", "artifact",
                        "Amendment 45: recurrence and repetition marginals and five interior arms, two cells, 20 brains"),
        ),
        sensitivity_checks=(SensitivityCheck(
            artifact="research/results/runs/memory.reuse-budget/reuse-budget-20261008/results.json",
            sample_path="run/seeds",
            treatment_path="observations/cells/10000~175~10.48/arms/U10~1bV/word",
            control_path="observations/cells/10000~175~10.48/arms/U10~1b3/word",
            relation="all-greater", minimum_effect=0.5,
            mechanism="every brain replays its random-successor sequences and none of its 3.7-repeat sequences at the word level, at (10000, 75, 0.48) (Amendment 45)",
        ), SensitivityCheck(
            artifact="research/results/runs/memory.reuse-budget/reuse-budget-20261008/results.json",
            sample_path="run/seeds",
            treatment_path="observations/cells/10000~175~10.48/arms/U10~1bV/word",
            control_path="observations/cells/10000~175~10.48/arms/U60~1bV/word",
            relation="all-greater", minimum_effect=0.5,
            mechanism="every brain replays at 10 uses per word and none at 60, random successors, at (10000, 75, 0.48) (Amendment 45)",
        )),
        caveat="The separable product form is approximate, not established (S2 failed by 0.006 "
               "at one arm). The recurrence failure is whole-brain collapse of unknown mechanism "
               "(not predicted by the draws' bigram statistics, Amendment 44). Edges measured at "
               "n/k 117-150 only; small vocabularies, 16-element walks.",
    ),
    Result(
        id="BIRTH-SETPOINT-GATES-SLEEP",
        engine="hashed AssemblyMemory as Amendment 44; standard and comparator store loops, then contrast-gated sleep",
        status=Status.MEASURED,
        claim="Sleep's contrast gate needs no reference brains: each brain can fix it once, before it "
              "learns anything, at 1.02 x the largest contrast its own empty network's dreams reach. "
              "Learning a healthy store raises dream contrast by only 1.3-1.5%, so at two new cells "
              "(n/k 120 and 145) healthy stores keep 1.000 replay, a collapsed U = 50 store is repaired "
              "from 0.11-0.15 to 0.80-0.83 and a comparator U = 100 store reaches 0.72, no brain "
              "collapsed, at least as well as gates calibrated on twenty reference brains.",
        source="PREREG_refraction_memory.md Amendment 55 (G1-G6 PASS; G7 FAIL)",
        preconditions=("random-successor reuse at 50 and 100 uses per word, rho = 0.05, n/k 120-145",
                       "sleep 300 episodes; set point from 300 dreams of the brain's own empty network"),
        evidence=("research/notes/memory/PREREG_refraction_memory.md#amendment-55-result-2026-10-10",),
        evidence_refs=(
            EvidenceRef("research/results/runs/memory.setpoint_sleep/setpoint-sleep-20261010/results.json", "artifact",
                        "Amendment 55: set-point and reference-median gates on the same stores, two cells, 20 brains"),
        ),
        sensitivity_checks=(SensitivityCheck(
            artifact="research/results/runs/memory.setpoint_sleep/setpoint-sleep-20261010/results.json",
            sample_path="run/seeds",
            treatment_path="observations/cells/8400~170~10.52/comparator/setpoint",
            control_path="observations/cells/8400~170~10.52/comparator/before",
            relation="all-greater", minimum_effect=0.15,
            mechanism="every brain's 100-use comparator store gains from set-point-gated sleep, at (8400, 70, 0.52) (Amendment 55)",
        ), SensitivityCheck(
            artifact="research/results/runs/memory.setpoint_sleep/setpoint-sleep-20261010/results.json",
            sample_path="run/seeds",
            treatment_path="observations/cells/11600~180~10.44/comparator/setpoint",
            control_path="observations/cells/11600~180~10.44/comparator/before",
            relation="all-greater", minimum_effect=0.2,
            mechanism="the same at (11600, 80, 0.44) (Amendment 55)",
        )),
        caveat="The mechanism bar failed: at each cell one brain's healthy store dreamt just past its own "
               "set point (by 0.16% and 0.01%) and the gate removed 5-7 healthy counts per million without "
               "cost to replay. The 2% margin barely covers learning's rise in dream contrast, and a smoke "
               "at n/k = 33 showed it reversed; the margin's dependence on n/k is unmeasured.",
    ),
    Result(
        id="ROBUST-SLEEP-THRESHOLD",
        engine="hashed AssemblyMemory as Amendment 44; comparator store loop, then contrast-gated sleep",
        status=Status.MEASURED,
        claim="Sleep's contrast gate set from the MEDIAN over reference brains of each brain's maximum "
              "dream contrast (x 1.02), which fewer than half of them cannot move, is safe and reaches "
              "the bar the maximum missed: at two new cells healthy stores stay at 1.000 with nothing "
              "removed, and 100 uses per word written through the comparator replay at 0.66-0.71 "
              "(0.33-0.41 before sleep) with no brain collapsed, removing 1.3-1.4% of counts. On "
              "Amendment 52's brains, where one reference brain was an outlier, it lifts the failed "
              "cell from 0.37 to 0.63 (exploratory).",
        source="PREREG_refraction_memory.md Amendment 54 (R1-R5 PASS)",
        preconditions=("random-successor reuse at 100 uses per word, rho = 0.05, n/k 131-132",
                       "comparator threshold 0.5; sleep 300 episodes, 20 reference brains"),
        evidence=("research/notes/memory/PREREG_refraction_memory.md#amendment-54-result-2026-10-10",),
        evidence_refs=(
            EvidenceRef("research/results/runs/memory.robust_sleep/robust-sleep-20261009/results.json", "artifact",
                        "Amendment 54: max- and median-gated sleep on the same stores, two cells, 20 brains"),
        ),
        sensitivity_checks=(SensitivityCheck(
            artifact="research/results/runs/memory.robust_sleep/robust-sleep-20261009/results.json",
            sample_path="run/seeds",
            treatment_path="observations/cells/9000~168~10.5/reuse/median",
            control_path="observations/cells/9000~168~10.5/reuse/comparator",
            relation="all-greater", minimum_effect=0.08,
            mechanism="every brain's 100-use store gains from median-gated sleep, at (9000, 68, 0.5) (Amendment 54)",
        ), SensitivityCheck(
            artifact="research/results/runs/memory.robust_sleep/robust-sleep-20261009/results.json",
            sample_path="run/seeds",
            treatment_path="observations/cells/10500~180~10.45/reuse/median",
            control_path="observations/cells/10500~180~10.45/reuse/comparator",
            relation="all-greater", minimum_effect=0.1,
            mechanism="the same at (10500, 80, 0.45) (Amendment 54)",
        )),
        caveat="No outlying reference brain occurred at the registered cells, so the two rules nearly "
               "coincided there; the rescue from an outlier rests on an exploratory re-reading of "
               "Amendment 52's brains.",
    ),
    Result(
        id="REPETITION-EDGE-YIELDS-TO-LIFECYCLE",
        engine="hashed AssemblyMemory as Amendment 44; standard and comparator store loops, then contrast-gated sleep",
        status=Status.MEASURED,
        claim="Amendment 45's second reuse edge -- repeated transitions -- is the same disease as the first "
              "and yields to the same treatment. At 3.3 repeats per transition (10 uses per word, 3 "
              "successors) every standard brain collapses (0.000-0.001); sleep alone recovers 0.50-0.59, "
              "the comparator alone 0.80-0.85 (flagging 4-5% of writes against 0.06-0.23% with random "
              "successors), and both 0.88-0.92 with no brain below 0.8. The tokens written at a repeated "
              "word pair's first two occurrences overlap only at the same-word level (0.11-0.12): "
              "repetition seeds the capture cascade by excess drive, it does not write tokens together.",
        source="PREREG_refraction_memory.md Amendment 53 (P1-P7 PASS)",
        preconditions=("10 uses per word with 3 or 5 successors, or random; rho = 0.05, n/k 133-134",
                       "comparator threshold 0.5; sleep 300 episodes, threshold from reference brains"),
        evidence=("research/notes/memory/PREREG_refraction_memory.md#amendment-53-result-2026-10-09",),
        evidence_refs=(
            EvidenceRef("research/results/runs/memory.repetition_reach/repetition-reach-20261009/results.json", "artifact",
                        "Amendment 53: standard, sleep, comparator and both at b = 3, 5 and random, two cells, 20 brains"),
        ),
        sensitivity_checks=(SensitivityCheck(
            artifact="research/results/runs/memory.repetition_reach/repetition-reach-20261009/results.json",
            sample_path="run/seeds",
            treatment_path="observations/cells/8500~164~10.52/b3/both",
            control_path="observations/cells/8500~164~10.52/b3/standard",
            relation="all-greater", minimum_effect=0.7,
            mechanism="every brain's repeated-transition store replays with the lifecycle and not without it, at (8500, 64, 0.52) (Amendment 53)",
        ), SensitivityCheck(
            artifact="research/results/runs/memory.repetition_reach/repetition-reach-20261009/results.json",
            sample_path="run/seeds",
            treatment_path="observations/cells/11000~182~10.44/b3/both",
            control_path="observations/cells/11000~182~10.44/b3/sleep",
            relation="all-greater", minimum_effect=0.15,
            mechanism="every brain gains from the comparator before sleep over sleep alone, at (11000, 82, 0.44) (Amendment 53)",
        )),
        caveat="One dose of repetition past the edge (3.3 repeats) and one near it (2); heavier repetition "
               "is untested. The sleep threshold was again raised by an outlying reference brain at one "
               "cell, where the sleep after the comparator added only +0.03.",
    ),
    Result(
        id="WRITE-SEPARATION-AND-SLEEP-COMPOSE",
        engine="hashed AssemblyMemory as Amendment 44; comparator store loop, then contrast-gated sleep",
        status=Status.MEASURED,
        claim="The local comparator at write (Amendment 50) and contrast-gated sleep (Amendment 51) "
              "compose and do more together than either alone. At two new cells, at 100 uses per "
              "word -- where a standard store replays nothing, sleep alone 0.004-0.064 and the "
              "comparator alone 0.18-0.24 -- both give 0.37-0.64 with no brain collapsed; at 80 uses "
              "both give 0.65-0.79. The sleep that follows the comparator is sparing (<= 1.4% of "
              "counts): separation keeps the store near enough to healthy for it to finish the repair.",
        source="PREREG_refraction_memory.md Amendment 52 (L1, L2, L4, L5 PASS; L3 FAIL)",
        preconditions=("random-successor reuse at 80 and 100 uses per word, rho = 0.05, n/k 127-139",
                       "comparator threshold 0.5; sleep 300 episodes, threshold from reference brains"),
        evidence=("research/notes/memory/PREREG_refraction_memory.md#amendment-52-result-2026-10-09",),
        evidence_refs=(
            EvidenceRef("research/results/runs/memory.lifecycle/lifecycle-20261009/results.json", "artifact",
                        "Amendment 52: standard, sleep, comparator and both on the same memories, two cells, 20 brains"),
        ),
        sensitivity_checks=(SensitivityCheck(
            artifact="research/results/runs/memory.lifecycle/lifecycle-20261009/results.json",
            sample_path="run/seeds",
            treatment_path="observations/cells/9500~175~10.46/100/both",
            control_path="observations/cells/9500~175~10.46/100/comparator",
            relation="all-greater", minimum_effect=0.15,
            mechanism="every brain's 100-use store replays better with sleep after the comparator than with the comparator alone, at (9500, 75, 0.46) (Amendment 52)",
        ), SensitivityCheck(
            artifact="research/results/runs/memory.lifecycle/lifecycle-20261009/results.json",
            sample_path="run/seeds",
            treatment_path="observations/cells/12500~190~10.4/100/both",
            control_path="observations/cells/12500~190~10.4/100/comparator",
            relation="all-greater", minimum_effect=0.05,
            mechanism="the same at (12500, 90, 0.4) (Amendment 52)",
        )),
        caveat="The REACH at 100 uses missed its registered 0.45 at one cell (0.368): a reference "
               "brain's outlying dream contrast raised the sleep threshold and the sleep that followed "
               "removed a tenth as much. The threshold's dependence on the reference maximum is a "
               "known weakness; a quantile threshold is untested.",
    ),
    Result(
        id="SELF-LIMITING-SLEEP-REPAIRS-CAPTURE",
        engine="hashed AssemblyMemory as Amendment 44; standard store_sequence, then unlearning on the count matrix",
        status=Status.MEASURED,
        claim="Unlearning from noise (Hopfield, Feinstein & Palmer 1983; Crick & Mitchison's proposed "
              "function of REM sleep) gated by settling contrast -- a dream transition is depressed "
              "only if its winners' mean drive over the area's exceeds a healthy reference store's "
              "maximum -- repairs a captured store and spares a healthy one. At two new cells: 50 uses "
              "per word from 0.04-0.15 to 0.73-0.80 after 300 episodes and 0.76-0.84 after 3000, no "
              "brain left collapsed; the gate closes as the clusters dissolve (0.1-0.2% of steps at "
              "the end; 3-5% of counts removed in all); a healthy store unchanged, nothing removed. "
              "Ungated, the same unlearning erases any store past a window (exploratory).",
        source="PREREG_refraction_memory.md Amendment 51 (S1-S6 PASS)",
        preconditions=("random-successor reuse at 50 and 10 uses per word, rho = 0.05, n/k 129-138",
                       "threshold 1.02 x the maximum contrast of 300 dreams of a U = 10 store on separate reference brains"),
        evidence=("research/notes/memory/PREREG_refraction_memory.md#amendment-51-result-2026-10-09",),
        evidence_refs=(
            EvidenceRef("research/results/runs/memory.sleep/sleep-20261009/results.json", "artifact",
                        "Amendment 51: calibration, and replay of U = 50 and U = 10 stores after cumulative sleep doses, two cells, 20 brains"),
        ),
        sensitivity_checks=(SensitivityCheck(
            artifact="research/results/runs/memory.sleep/sleep-20261009/results.json",
            sample_path="run/seeds",
            treatment_path="observations/cells/9000~165~10.5/50/doses/2/reliability",
            control_path="observations/cells/9000~165~10.5/50/doses/0/reliability",
            relation="all-greater", minimum_effect=0.1,
            mechanism="every brain's 50-use store replays better after 300 episodes of gated sleep, at (9000, 65, 0.5) (Amendment 51)",
        ), SensitivityCheck(
            artifact="research/results/runs/memory.sleep/sleep-20261009/results.json",
            sample_path="run/seeds",
            treatment_path="observations/cells/11000~185~10.42/50/doses/2/reliability",
            control_path="observations/cells/11000~185~10.42/50/doses/0/reliability",
            relation="all-greater", minimum_effect=0.1,
            mechanism="the same at (11000, 85, 0.42) (Amendment 51)",
        )),
        caveat="The threshold needs a healthy reference store at the same cell; a rare settling dream "
               "in it raises the threshold (the small-area smoke: max 2.18 vs mean 1.35) and makes the "
               "gate conservative. Repair plateaus at 0.76-0.84, below write-time separation's 0.94-0.99. "
               "One reuse kind; n/k 129-138.",
    ),
    Result(
        id="LOCAL-COMPARATOR-SEPARATES-AT-WRITE",
        engine="hashed AssemblyMemory as Amendment 44; store_sequence's loop written out with a comparator",
        status=Status.MEASURED,
        claim="A local comparator prevents the recurrence collapse as an oracle separation does. "
              "Before each write, the recall projection (recurrence alone: what memory predicts) and "
              "the write projection are compared, as CA1 is held to compare CA3's recall with "
              "cortical input; if they share >= 0.5 of their winners, the predicted cells are "
              "inhibited for that write. With no stored-token lookup and no word identity, at two new "
              "cells: 50 uses per word from 0.10 to 0.94-0.96, 60 uses from 0.00-0.01 to 0.87-0.90, "
              "no brain left collapsed, a healthy store untouched, 2-3% of writes flagged, 78-84% of "
              "the oracle's captures caught, no captured token left.",
        source="PREREG_refraction_memory.md Amendment 50 (C1-C6 PASS); Amendment 49 for the oracle",
        preconditions=("random-successor reuse at 10, 50, 60 uses per word, rho = 0.05, n/k 131-141",
                       "comparator threshold 0.5 (calibrated to n/k ~130-145); masked replay, word-level"),
        evidence=("research/notes/memory/PREREG_refraction_memory.md#amendment-50-result-2026-10-09",),
        evidence_refs=(
            EvidenceRef("research/results/runs/memory.comparator/comparator-20261009/results.json", "artifact",
                        "Amendment 50: standard and comparator stores of the same memories, three arms, two cells, 20 brains"),
        ),
        sensitivity_checks=(SensitivityCheck(
            artifact="research/results/runs/memory.comparator/comparator-20261009/results.json",
            sample_path="run/seeds",
            treatment_path="observations/cells/8500~165~10.5/arms/60/comparator/reliability",
            control_path="observations/cells/8500~165~10.5/arms/60/standard/reliability",
            relation="all-greater", minimum_effect=0.4,
            mechanism="every brain's 60-use store replays better with the comparator, at (8500, 65, 0.5) (Amendment 50)",
        ), SensitivityCheck(
            artifact="research/results/runs/memory.comparator/comparator-20261009/results.json",
            sample_path="run/seeds",
            treatment_path="observations/cells/12000~185~10.43/arms/60/comparator/reliability",
            control_path="observations/cells/12000~185~10.43/arms/60/standard/reliability",
            relation="all-greater", minimum_effect=0.4,
            mechanism="the same at (12000, 85, 0.43) (Amendment 50)",
        )),
        caveat="The fixed threshold is calibrated to n/k 130-145 and over-fires in a small area "
               "(n/k = 33: 27% of healthy writes flagged, smoke). Sequence-initial writes are not "
               "judged (they have no prior state). One reuse kind; the repetition failure is not "
               "covered; global cholinergic suppression of recurrence at write destroyed the store "
               "in the probes (exploratory).",
    ),
    Result(
        id="WRITE-SEPARATION-PREVENTS-COLLAPSE",
        engine="hashed AssemblyMemory as Amendment 44; store_sequence's loop written out with a write-time check",
        status=Status.MEASURED,
        claim="The recurrence collapse -- whole brains failing when words recur in many contexts -- "
              "is written into the store: tokens of different words are laid onto one assembly in a "
              "cascade (exploratory). A label-free pattern separation at write prevents it: each write "
              "is previewed, and the neurons of any stored token it would overlap by >= 0.3 are "
              "inhibited for that write only. At two new cells, 50 uses per word go from 0.07-0.17 to "
              "0.99 and 60 from 0.00 to 0.91, no brain is left collapsed, a healthy store is untouched "
              "(1.000) and under 2% of writes are intervened on. The reuse edge of about 40 uses per "
              "word was a property of the write policy, not of the substrate.",
        source="PREREG_refraction_memory.md Amendment 49 (W1-W6 PASS)",
        preconditions=("random-successor reuse at 10, 50, 60 uses per word, rho = 0.05, n/k 136-144",
                       "capture threshold 0.3, label-free; masked replay, word-level score"),
        evidence=("research/notes/memory/PREREG_refraction_memory.md#amendment-49-result-2026-10-09",),
        evidence_refs=(
            EvidenceRef("research/results/runs/memory.write-separation/write-separation-20261009/results.json", "artifact",
                        "Amendment 49: standard and separated stores of the same memories, three arms, two cells, 20 brains"),
        ),
        sensitivity_checks=(SensitivityCheck(
            artifact="research/results/runs/memory.write-separation/write-separation-20261009/results.json",
            sample_path="run/seeds",
            treatment_path="observations/cells/9500~170~10.5/arms/60/separated/reliability",
            control_path="observations/cells/9500~170~10.5/arms/60/standard/reliability",
            relation="all-greater", minimum_effect=0.3,
            mechanism="every brain's 60-use store replays better separated than standard, at (9500, 70, 0.5) (Amendment 49)",
        ), SensitivityCheck(
            artifact="research/results/runs/memory.write-separation/write-separation-20261009/results.json",
            sample_path="run/seeds",
            treatment_path="observations/cells/11500~180~10.45/arms/60/separated/reliability",
            control_path="observations/cells/11500~180~10.45/arms/60/standard/reliability",
            relation="all-greater", minimum_effect=0.3,
            mechanism="the same at (11500, 80, 0.45) (Amendment 49)",
        )),
        caveat="The check is an ORACLE: it compares each previewed write with every stored token. "
               "A circuit that computes the familiarity signal is not tested here. One reuse kind "
               "(random successors); the repetition failure is not covered; n/k 136-144; at 60 uses "
               "a few brains remain partly failing (lowest 0.40), so a new edge begins there.",
    ),
    Result(
        id="LOGIT-MARGIN-DECIDES-REPLAY",
        engine="hashed AssemblyMemory as Amendment 44; replay frozen, read through a logit lens",
        status=Status.MEASURED,
        claim="Read each replay step's net drive as a logit for every stored token (mean drive "
              "over its k neurons, in units of the k-WTA threshold). A brain's slack -- the correct "
              "next word's margin over the best other word, on the steps it wins -- relative to a "
              "healthy reference brain's at the same cell places every brain on one logistic: the "
              "masked read holds above a relative margin of 0.75-0.76 at four cells (two new), and a "
              "read-time session adaptation lowers that to 0.64-0.65. Collapsed brains, all at 0.00 "
              "masked, are rescued in the order of their slack (Spearman 0.86, 0.87), and the size of "
              "their rescue is predicted before it is applied (within 0.03 and 0.05). The recurrence "
              "collapse is a loss of signal margin, not a deep false well.",
        source="PREREG_refraction_memory.md Amendment 48 (M1-M4 PASS)",
        preconditions=("random-successor reuse at 30-70 uses per word, rho = 0.05, n/k 136-147",
                       "reference arm U = 10; adaptation c = 0.1, T = 50; word-level score"),
        evidence=("research/notes/memory/PREREG_refraction_memory.md#amendment-48-result-2026-10-09",),
        evidence_refs=(
            EvidenceRef("research/results/runs/memory.signal-margin/signal-margin-20261009/results.json", "artifact",
                        "Amendment 48: slack, masked and adapted replay per brain, six arms, two cells, 20 brains"),
        ),
        sensitivity_checks=(SensitivityCheck(
            artifact="research/results/runs/memory.signal-margin/signal-margin-20261009/results.json",
            sample_path="run/seeds",
            treatment_path="observations/cells/7500~155~10.6/arms/10/slack",
            control_path="observations/cells/7500~155~10.6/arms/50/slack",
            relation="all-greater", minimum_effect=0.04,
            mechanism="every brain's logit margin falls with reuse, U = 10 above U = 50, at (7500, 55, 0.6) (Amendment 48)",
        ), SensitivityCheck(
            artifact="research/results/runs/memory.signal-margin/signal-margin-20261009/results.json",
            sample_path="run/seeds",
            treatment_path="observations/cells/12500~185~10.42/arms/50/habit",
            control_path="observations/cells/12500~185~10.42/arms/50/masked",
            relation="all-greater", minimum_effect=0.01,
            mechanism="every brain at U = 50 replays more with read-time adaptation than masked, at (12500, 85, 0.42) (Amendment 48)",
        )),
        caveat="One substrate (k-WTA, no separate inhibition), one reuse kind (random "
               "successors), one adaptation setting, n/k 127-147. The relative margin needs a "
               "reference arm at the same cell; the predicted rescue ran slightly high at both "
               "new cells (0.025, 0.052). The repetition failure is not covered.",
    ),
    Result(
        id="READ-TIME-ADAPTATION-HELPS-FAILING-BRAINS",
        engine="hashed AssemblyMemory as Amendment 44; replay frozen, with a session adaptation in place of the masked read",
        status=Status.MEASURED,
        claim="Replay here normally runs with no adaptation (masked). A session adaptation "
              "during replay -- every replay winner charged 0.1 of its raw drive, decaying over "
              "50 steps, kept across a brain's sequences -- helps every brain the recurrence "
              "collapse leaves failing and harms none (52 of 54 failing brain-arms gained, none "
              "lost, two new cells), leaves a healthy memory at 1.000, and barely moves the "
              "repetition failure (+0.01-0.02). The collapse is a spurious attractor (failed "
              "replays funnel into a few shared states, exploratory) and the force replay lacks "
              "pulls brains out of it.",
        source="PREREG_refraction_memory.md Amendment 47 (Q1, Q3 PASS; Q2, Q4 FAIL); Amendment 46 (H2 PASS)",
        preconditions=("random-successor reuse at 40 and 50 uses per word, rho = 0.05, n/k 129-144",
                       "c = 0.1, T = 50 replay steps; word-level score"),
        evidence=("research/notes/memory/PREREG_refraction_memory.md#amendment-47-result-2026-10-09",
                  "research/notes/memory/PREREG_refraction_memory.md#amendment-46-result-2026-10-09"),
        evidence_refs=(
            EvidenceRef("research/results/runs/memory.read-rescue/read-rescue-20261009/results.json", "artifact",
                        "Amendment 47: masked, habit and strong replay of the same memories, two cells, 20 brains"),
            EvidenceRef("research/results/runs/memory.read-adaptation/read-adaptation-20261008/results.json", "artifact",
                        "Amendment 46: the first registration of the control, two cells, 20 brains"),
        ),
        sensitivity_checks=(SensitivityCheck(
            artifact="research/results/runs/memory.read-rescue/read-rescue-20261009/results.json",
            sample_path="run/seeds",
            treatment_path="observations/cells/9000~170~10.5/arms/recurrence40/modes/habit",
            control_path="observations/cells/9000~170~10.5/arms/recurrence40/modes/masked",
            relation="all-greater", minimum_effect=0.01,
            mechanism="every brain replays more of its 40-use sequences with read-time adaptation than masked, at (9000, 70, 0.5) (Amendment 47)",
        ), SensitivityCheck(
            artifact="research/results/runs/memory.read-rescue/read-rescue-20261009/results.json",
            sample_path="run/seeds",
            treatment_path="observations/cells/13000~190~10.4/arms/recurrence40/modes/habit",
            control_path="observations/cells/13000~190~10.4/arms/recurrence40/modes/masked",
            relation="all-greater", minimum_effect=0.01,
            mechanism="the same at (13000, 90, 0.4) (Amendment 47)",
        )),
        caveat="The SIZE of the lift is not established: collapsed brains gain 0.18-0.23 here "
               "(registered >= 0.2, missed at one cell) against 0.38-0.39 in Amendment 46's "
               "shallower collapses; deeper collapses are lifted less. Its window does not "
               "transfer in absolute charge across cells (Amendment 46, H4). One setting tested.",
    ),
    Result(
        id="SEQUENCES-OF-SEQUENCES-ACROSS-AREAS",
        engine="two hashed AssemblyMemory areas (sequence S, chunk C) and a DenseOrganFiber C -> S; recall on two clocks",
        status=Status.MEASURED,
        claim="Two areas replay sequences of sequences: chunks written once into a "
              "sequence area are replayed in whatever order a plan in a chunk area "
              "dictates -- every plan whole, every brain, from the plan's first state "
              "alone -- with plans that share chunks kept apart by the chunk area and the "
              "content shared through the sequence area. The links from plan positions "
              "to chunk starts carry it, and under load they give way first; doubling "
              "both areas relieves them.",
        source="PREREG_refraction_memory.md Amendment 33 (H1-H4 PASS); rerun on the "
               "corrected cross-area divisor, Amendment 35 (H1-H4 PASS)",
        preconditions=("chunk starts linked by construction (the circuit does not find "
                       "chunk boundaries itself); fixed chunk length on a two-clock recall",
                       "each area as in ADAPTATION-SWITCHES-MEMORY-TYPE: beta = theta, "
                       "refraction 0.5 beta recovering over 33 rounds"),
        evidence=("research/notes/memory/PREREG_refraction_memory.md#amendment-33-result-2026-10-07",
                  "research/notes/memory/PREREG_refraction_memory.md#amendment-35-result-2026-10-07"),
        evidence_refs=(
            EvidenceRef("research/results/runs/memory.hierarchy/hierarchy-20261007-a35/results.json", "artifact",
                        "Amendment 35: Amendment 33 on the corrected C -> S divisor, the same 20 brains"),
            EvidenceRef("research/results/runs/memory.hierarchy/hierarchy-20261007/results.json", "artifact",
                        "Amendment 33: three loads, links 0 and 2, two area pairs, 20 brains "
                        "(C -> S divided by the in-degree over S's rows; see Amendment 35)"),
        ),
        sensitivity_checks=(SensitivityCheck(
            artifact="research/results/runs/memory.hierarchy/hierarchy-20261007-a35/results.json",
            sample_path="run/seeds",
            treatment_path="observations/cells/4000~12000/links/8~16~12/0/whole/values",
            control_path="observations/cells/4000~12000/links/8~16~10/0/whole/values",
            relation="all-greater", minimum_effect=0.5,
            mechanism="plan 0 replays whole with the chunk-area to sequence-area links and not without them, at (S, C) = (4000, 2000) (Amendments 33, 35)",
        ), SensitivityCheck(
            artifact="research/results/runs/memory.hierarchy/hierarchy-20261007-a35/results.json",
            sample_path="run/seeds",
            treatment_path="observations/cells/8000~14000/links/8~16~12/0/whole/values",
            control_path="observations/cells/8000~14000/links/8~16~10/0/whole/values",
            relation="all-greater", minimum_effect=0.5,
            mechanism="plan 0 replays whole with the chunk-area to sequence-area links and not without them, at (S, C) = (8000, 4000) (Amendments 33, 35)",
        )),
        caveat="Two levels, p = 0.5, no noise, chunks of 12 and plans of 5 or 8; the "
               "recall's clock (one chunk-area step per chunk) is given, not generated.",
    ),
    Result(
        id="BIDIRECTIONAL-RECALL-BY-LRI",
        engine="hashed AssemblyMemory, store_sequence with forward_counts/reverse_counts, LRI period 4 at recall with the cue and came-from state primed",
        status=Status.MEASURED,
        claim="One area replays a stored 200-element sequence forward, backward, and "
              "either way from its middle, on every brain, when its two directions are "
              "written about equally and long-range inhibition at recall vetoes the "
              "state just left; without LRI it goes nowhere, and with the directions "
              "unequal the stronger one wins whatever LRI does.",
        source="PREREG_refraction_memory.md Amendments 31 (R1, R2, R4 PASS; R3 FAIL) and 32 (Q1-Q3 PASS)",
        preconditions=("reverse transitions written after the sequence (a stand-in for a "
                       "post-before-pre rule from a trace); forward and reverse counts within "
                       "about a third of one write of each other",
                       "the recall cue and, from the middle, the came-from state entered into "
                       "the LRI history"),
        evidence=("research/notes/memory/PREREG_refraction_memory.md#amendment-32-result-2026-10-07",),
        evidence_refs=(
            EvidenceRef("research/results/runs/memory.bidirectional/bidirectional-balanced-20261007/results.json", "artifact",
                        "Amendment 32: balanced and unbalanced arms, two cells, 20 brains"),
            EvidenceRef("research/results/runs/memory.bidirectional/bidirectional-20261007/results.json", "artifact",
                        "Amendment 31: reverse links alone -- backward recall, forward lost"),
        ),
        sensitivity_checks=(SensitivityCheck(
            artifact="research/results/runs/memory.bidirectional/bidirectional-balanced-20261007/results.json",
            sample_path="run/seeds",
            treatment_path="observations/cells/4000~160~10.5/reverse/2+3/backward_lri/values",
            control_path="observations/cells/4000~160~10.5/reverse/2+2/backward_lri/values",
            relation="all-greater", minimum_effect=0.5,
            mechanism="a balanced chain (2 + 3) replays backward under LRI where a forward-heavy one (2 + 2) does not, at (4000, 60, 0.5) (Amendment 32)",
        ), SensitivityCheck(
            artifact="research/results/runs/memory.bidirectional/bidirectional-balanced-20261007/results.json",
            sample_path="run/seeds",
            treatment_path="observations/cells/8000~160~10.5/reverse/2+3/backward_lri/values",
            control_path="observations/cells/8000~160~10.5/reverse/2+2/backward_lri/values",
            relation="all-greater", minimum_effect=0.5,
            mechanism="a balanced chain (2 + 3) replays backward under LRI where a forward-heavy one (2 + 2) does not, at (8000, 60, 0.5) (Amendment 32)",
        )),
        caveat="The reverse links are written after the sequence, not by a modelled "
               "trace rule; the balance window is narrow and was set by counts, not "
               "learned. Two cells, L = 200, p = 0.5.",
    ),
    Result(
        id="RECOVERY-SETS-SEQUENCE-LENGTH",
        engine="hashed AssemblyMemory, store_sequence one round per element, refraction 0.5 beta decaying as exp(-1/tau) per writing round, beta = theta",
        status=Status.MEASURED,
        claim="How long a sequence one area recalls by itself is set by how fast its "
              "refraction recovers: the limit rises from the unrefracted area's merging "
              "cliff as recovery slows, peaks at an interior recovery time (32 to 64 "
              "writing rounds at four cells), and falls back to the tiling deadline -- "
              "6890 elements at (8000, 60, 0.5) against 861 unrefracted and 264 never "
              "recovering. At the densest code (n/k = 33) the gain is only 1.27x.",
        source="PREREG_refraction_memory.md Amendment 29 (B2 PASS; B1, B3 FAIL at (2000, 60) only; B4 FAIL)",
        preconditions=("one chosen sequence per brain, one stimulus-and-recurrence round per "
                       "element; replay from half of element 0 by frozen masked rounds",
                       "recovery modelled as bias *= exp(-1/tau) at every writing round"),
        evidence=("research/notes/memory/PREREG_refraction_memory.md#amendment-29-result-2026-10-07",),
        evidence_refs=(
            EvidenceRef("research/results/runs/memory.recovery/recovery-20261007/results.json", "artifact",
                        "Amendment 29: length limits at twelve recovery times, four cells, 20 brains"),
        ),
        sensitivity_checks=(SensitivityCheck(
            artifact="research/results/runs/memory.recovery/recovery-20261007/results.json",
            sample_path="run/seeds",
            treatment_path="observations/cells/8000~160~10.5/taus/64/curve/724/values",
            control_path="observations/cells/8000~160~10.5/taus/inf/curve/724/values",
            relation="all-greater", minimum_effect=0.5,
            mechanism="a bias recovering over 64 writing rounds replays a 724-element sequence whole where the never-recovering bias breaks, at (8000, 60, 0.5) (Amendment 29)",
        ), SensitivityCheck(
            artifact="research/results/runs/memory.recovery/recovery-20261007/results.json",
            sample_path="run/seeds",
            treatment_path="observations/cells/4000~130~10.5/taus/64/curve/1448/values",
            control_path="observations/cells/4000~130~10.5/taus/inf/curve/1448/values",
            relation="all-greater", minimum_effect=0.5,
            mechanism="a bias recovering over 64 writing rounds replays a 1448-element sequence whole where the never-recovering bias breaks, at (4000, 30, 0.5) (Amendment 29)",
        )),
        caveat="The best limit follows the in-degree n p (post hoc: 1.5-1.9x the attractor "
               "memory's 0.0135 d^1.51), not (n/k)^2 as registered (B4 failed). One sequence "
               "per brain, no noise, p = 0.5; recall reads the synapses with the bias masked. "
               "Mapping a writing round onto physical time is an assumption.",
    ),
    Result(
        id="SEQUENCE-TILING-DEADLINE",
        engine="hashed AssemblyMemory, store_sequence one round per element, refraction 0.5 beta, beta = theta",
        status=Status.MEASURED,
        claim="A refracted area writes a single sequence onto neurons that have never "
              "fired until none remain: the sequence tiles the area, the fresh pool is "
              "empty at element n/k, and replay breaks at multiples of n/k (49 of 51 "
              "breaks within two steps). A bias that recovers -- zeroed every n/(2k) "
              "elements -- removes the deadline: every brain replays 4 n/k elements "
              "(532 at n/k = 133, where the never-decaying bias breaks near 130).",
        source="PREREG_refraction_memory.md Amendment 28 (D1, D2, D3 PASS)",
        preconditions=("one chosen sequence per brain, one stimulus-and-recurrence round per "
                       "element; replay from half of element 0 by frozen masked rounds",
                       "cumulative refraction bias (bias += s raw at every win, never decaying)"),
        evidence=("research/notes/memory/PREREG_refraction_memory.md#amendment-28-result-2026-10-06",),
        evidence_refs=(
            EvidenceRef("research/results/runs/memory.tiling/tiling-20261006/results.json", "artifact",
                        "Amendment 28: fresh share per element and replay breaks, refracted and reset arms, four cells"),
            EvidenceRef("research/results/runs/memory.sequence-length/sequence-length-20261006/results.json", "artifact",
                        "Amendment 27: the refracted length limit, where the deadline was first seen (post hoc)"),
        ),
        sensitivity_checks=(SensitivityCheck(
            artifact="research/results/runs/memory.tiling/tiling-20261006/results.json",
            sample_path="run/seeds",
            treatment_path="observations/cells/8000~160~10.5/arms/reset/break/values",
            control_path="observations/cells/8000~160~10.5/arms/refracted/break/values",
            relation="all-greater", minimum_effect=100,
            mechanism="a bias zeroed every n/(2k) elements replays past the step-n/k wrap where the never-decaying bias breaks, at (8000, 60, 0.5) (Amendment 28)",
        ), SensitivityCheck(
            artifact="research/results/runs/memory.tiling/tiling-20261006/results.json",
            sample_path="run/seeds",
            treatment_path="observations/cells/4000~130~10.5/arms/reset/break/values",
            control_path="observations/cells/4000~130~10.5/arms/refracted/break/values",
            relation="all-greater", minimum_effect=100,
            mechanism="a bias zeroed every n/(2k) elements replays past the step-n/k wrap where the never-decaying bias breaks, at (4000, 30, 0.5) (Amendment 28)",
        )),
        caveat="Four cells, p = 0.5, one write strength (theta) and one refraction strength "
               "(0.5 beta); the reset is an instantaneous zeroing, not a decay with a time "
               "constant. Why replay breaks at the wrap while the one-step transition there "
               "is not weak (probe) is not yet measured; n/k = 33 survives its wraps.",
    ),
    Result(
        id="ORDERED-RECALL-BY-TRANSITIONS",
        engine="numpy_sparse, materialized connectome (the assembly calculus's own sequence_memorize and ordered_recall)",
        status=Status.MEASURED,
        claim="The calculus's sequence operation recalls an eight-element sequence "
              "autonomously -- 7 of 7 steps after the cue on 17 of 20 seeds, 6 on the "
              "rest -- when each element is written in ONE stimulus-and-recurrence round "
              "at a write equal to the convergence threshold theta; eight rounds per "
              "element at the same write, and one round at beta = 0.10, advance 0 on "
              "every seed. The route is the written bridge, not long-range inhibition "
              "(4 to 6 steps with inhibition off).",
        source="PREREG_ordered_recall_reproduction.md Amendment 2 (OR-1, OR-2, OR-6, OR-7, OR-8 PASS; OR-3, OR-4, OR-5 FAIL)",
        preconditions=("n = 4000, k = 50, p = 0.05, w_max 20, L = 8; LRI period 3, strength 100 "
                       "set after memorizing; recall from the first stimulus",
                       "MATERIALIZED connectome: on the sampled one recall advances 0 steps (OR-5)"),
        evidence=("research/notes/sequence/PREREG_ordered_recall_reproduction.md#amendment-2-result-2026-10-06",),
        evidence_refs=(
            EvidenceRef("research/results/runs/sequence.ordered-recall-repair/ordered-recall-repair-20261006/results.json", "artifact",
                        "Amendment 2: five arms on twenty seeds; steps after the cue, order, cue retrieval"),
        ),
        sensitivity_gap="The record keys rows by seed (a mapping), which the register's "
                        "list paths cannot address; the per-seed contrast it holds -- one "
                        "round per element 6-7 steps against the diagnostic's construction 0 "
                        "on every one of 20 seeds at the same write -- is read by "
                        "ordered_recall_repair.evaluate (OR-7) and pinned by "
                        "test_ordered_recall_advances_when_each_element_is_written_as_a_transition.",
        caveat="Not a reproduction of the paper's inhibition-driven mechanism (OR-4 fails; "
               "Amendment 1 could not locate that mechanism in either reference). One cell, "
               "L = 8. The last step loops back toward the most recent element free of the "
               "period-3 inhibition on 8 of 20 seeds (OR-3, post hoc). The write is near "
               "theta, where a synapse clips within one or two presentations.",
    ),
    Result(
        id="ADAPTATION-SWITCHES-MEMORY-TYPE",
        engine="hashed AssemblyMemory (arm B, ungated, masked readout), write_rule round / deferred, refraction strength s",
        status=Status.MEASURED,
        claim="One recurrent area and one causal Hebbian rule store attractors or "
              "sequences according to whether the activity holds still while it is "
              "written. Online under refraction weaker than plasticity (s = 0.5 beta) "
              "the item's rounds converge (consecutive overlap 0.71-0.85) and the area "
              "stores attractors; online under refraction stronger than plasticity "
              "(s = 1.5 beta), or written after the item, the activity moves every round "
              "(overlap <= 0.05) and the area replays the stored trajectory from half of "
              "its first state, all seven steps, by itself: 207-950 sequences of eight "
              "states at three cells.",
        source="PREREG_refraction_memory.md Amendment 26 (S1, S2 PASS; S3, S3d FAIL)",
        preconditions=("T = 8 rounds per item, w_max 20, norm_init; replay = frozen masked "
                       "rounds from half of round 0, each fed the previous round's winners",
                       "write at least ~0.7-1.0 theta: no sequence is stored at 0.5 theta"),
        evidence=("research/notes/memory/PREREG_refraction_memory.md#amendment-26-result-2026-10-06",),
        evidence_refs=(
            EvidenceRef("research/results/runs/memory.sequences/sequences-20261006/results.json", "artifact",
                        "Amendment 26: three arms at three cells, 20 brains, replay capacity and during-write overlap"),
            EvidenceRef("research/results/runs/memory.write_rules/write-rules-20261003/results.json", "artifact",
                        "Amendment 25: the deferred write's one-step trajectory reading"),
            EvidenceRef("research/results/runs/memory.sequence-length/sequence-length-20261006/results.json", "artifact",
                        "Amendment 27: one chosen sequence per brain; Hebbian limit 0.063-0.106 p (n/k)^2, refracted limit set by a break at step n/k"),
        ),
        sensitivity_checks=(SensitivityCheck(
            artifact="research/results/runs/memory.sequences/sequences-20261006/results.json",
            sample_path="run/seeds",
            treatment_path="observations/cells/4000~160~10.5/arms/round-s0.5/rates/0.07436/own/values",
            control_path="observations/cells/4000~160~10.5/arms/round-s1.5/rates/0.52052/own/values",
            relation="all-greater", minimum_effect=0.5,
            mechanism="online write at s = 0.5 beta holds the item's rounds, at s = 1.5 beta they move, at (4000, 60, 0.5) (Amendment 26)",
        ), SensitivityCheck(
            artifact="research/results/runs/memory.sequences/sequences-20261006/results.json",
            sample_path="run/seeds",
            treatment_path="observations/cells/2000~160~10.5/arms/round-s0.5/rates/0.07118/own/values",
            control_path="observations/cells/2000~160~10.5/arms/round-s1.5/rates/0.35592/own/values",
            relation="all-greater", minimum_effect=0.5,
            mechanism="online write at s = 0.5 beta holds the item's rounds, at s = 1.5 beta they move, at (2000, 60, 0.5) (Amendment 26)",
        ), SensitivityCheck(
            artifact="research/results/runs/memory.sequences/sequences-20261006/results.json",
            sample_path="run/seeds",
            treatment_path="observations/cells/4000~160~10.5/arms/deferred-s0.5/rates/0.52052/ensembles/32/replay/values",
            control_path="observations/cells/4000~160~10.5/arms/deferred-s0.5/rates/0.1859/ensembles/32/replay/values",
            relation="all-greater", minimum_effect=0.5,
            mechanism="deferred write replays the stored trajectory at 1.4 theta, not at 0.5 theta, at (4000, 60, 0.5), 32 items (Amendment 26)",
        )),
        caveat="Sequence capacity follows neither n/k nor the in-degree alone (S3, S3d "
               "fail); three cells give C ~ d^1.05 (n/k)^0.75 exactly, untested. The "
               "replay criterion is overlap with the item's own state, not distinct "
               "against other items' -- a distinctness read is owed. The control and "
               "the online arm differ in rate as well as s. All cells p = 0.5, T = 8.",
    ),
    Result(
        id="WRITE-TIMING-DECIDES-ATTRACTOR",
        engine="hashed AssemblyMemory (refracted, arm B, ungated), write_rule round / online_burst / deferred / burst",
        status=Status.MEASURED,
        claim="In the refracted assembly memory WHEN the write happens decides what is "
              "stored. The round write, whose counts feed into the item's next round, "
              "converges the item onto one assembly and stores attractors (451-1442 "
              "items). The same counts written after the item store no attractor at any "
              "of 21 rates over 0.1-3.2 theta: refraction relocates the unwritten rounds "
              "every round, and what is stored is the item's TRAJECTORY (one frozen round "
              "from half of round t recovers 64-79% of round t+1). A burst-timing rule "
              "(deferred, symmetric, between neurons that fired twice) and a burst-gated "
              "online write store nothing.",
        source="PREREG_refraction_memory.md Amendment 25 (CV, W1-W4 PASS)",
        preconditions=("refraction 0.5 beta charged every round, T = 8, w_max 20, "
                       "norm_init; half-cue masked readout",
                       "burst = fired in at least 2 of the item's 8 rounds"),
        evidence=("research/notes/memory/PREREG_refraction_memory.md#amendment-25-result-2026-10-03",),
        evidence_refs=(
            EvidenceRef("research/results/runs/memory.write_rules/write-rules-20261003/results.json", "artifact",
                        "Amendment 25: four write rules at three cells, 20 brains, capacity and trajectory readings"),
        ),
        sensitivity_checks=(SensitivityCheck(
            artifact="research/results/runs/memory.write_rules/write-rules-20261003/results.json",
            sample_path="run/seeds",
            treatment_path="observations/cells/2000~160~10.5/rules/deferred/trajectory/next/values",
            control_path="observations/cells/2000~160~10.5/rules/deferred/trajectory/same/values",
            relation="all-greater", minimum_effect=0.5,
            mechanism="deferred write: one round from round t recovers round t+1, not round t at (2000, 60, 0.5) (Amendment 25)",
        ), SensitivityCheck(
            artifact="research/results/runs/memory.write_rules/write-rules-20261003/results.json",
            sample_path="run/seeds",
            treatment_path="observations/cells/4000~120~10.5/rules/deferred/trajectory/next/values",
            control_path="observations/cells/4000~120~10.5/rules/deferred/trajectory/same/values",
            relation="all-greater", minimum_effect=0.5,
            mechanism="deferred write: one round from round t recovers round t+1, not round t at (4000, 20, 0.5) (Amendment 25)",
        ), SensitivityCheck(
            artifact="research/results/runs/memory.write_rules/write-rules-20261003/results.json",
            sample_path="run/seeds",
            treatment_path="observations/cells/4000~1160~10.5/rules/round/trajectory/own/values",
            control_path="observations/cells/4000~1160~10.5/rules/deferred/trajectory/own/values",
            relation="all-greater", minimum_effect=0.5,
            mechanism="online write holds the item's rounds together, the deferred write does not at (4000, 160, 0.5) (Amendment 25)",
        )),
        caveat="Three cells, all p = 0.5; the trajectory reading is at one rate (1.0 "
               "theta). Refraction's strength is tied to the write's (0.5 beta), so "
               "a burst rule with a decoupled or weaker refraction -- or a window "
               "short against the relocation period -- is untested. Developing "
               "retinogeniculate BTDP refines maps rather than storing items; the "
               "claim is about item storage in this circuit.",
    ),
    Result(
        id="REFRACTION-CANCELS-CONVERGENCE",
        engine="hashed_assembly_memory (HashedArea with AreaFiber/StimulusFiber); the registered twenty-brain run is a runner artifact, the wander diagnostic and the bias-readout numbers are logs",
        status=Status.MEASURED,
        sensitivity_checks=(SensitivityCheck(
            artifact="research/results/runs/memory.refraction-convergence/refraction-convergence-20260913/results.json",
            sample_path="run/seeds",
            treatment_path="observations/arms/s0.5/rows/*/late",
            control_path="observations/arms/s1.0/rows/*/late",
            relation="all-greater", minimum_effect=0.5,
            mechanism="refraction strength 0.5 beta against beta on the same recurrent area (n 4000, k 100, p 0.5, beta 0.10, w_max 20), late consecutive-round overlap over rounds 200 to 240: 0.904 to 0.932 against 0.003 to 0.007 on every paired brain; the s = beta area reshuffles its winners every round",
        ), SensitivityCheck(
            artifact="research/results/runs/memory.refraction-period-law/period-law-20260913/results.json",
            sample_path="run/seeds",
            treatment_path="observations/cells/w20b0.1/arms/refracted/rows/*/n_relocations",
            control_path="observations/cells/w20b0.1/arms/control/rows/*/n_relocations",
            relation="all-greater", minimum_effect=3,
            mechanism="refraction on against off at the same operating point and the same weight clip: 5 relocations on every refracted brain against 0 on every control brain, so the clip alone does not move an assembly and the bias charging past it is the whole mechanism",
        ), SensitivityCheck(
            artifact="research/results/runs/memory.refraction-convergence/refraction-convergence-fresh-20260913/results.json",
            sample_path="run/seeds",
            treatment_path="observations/arms/s0.5/rows/*/stable_fraction",
            control_path="observations/arms/s0.8/rows/*/stable_fraction",
            relation="all-greater", minimum_effect=0.7,
            mechanism="on the fresh seed block, the fraction of rounds 20 to 240 whose winners are unchanged: 0.801 to 0.837 at 0.5 beta against 0.000 on every brain at 0.8 beta -- relocation and churn separated per brain, not by an average",
        )),
        sensitivity_gap="The retained check is the strength contrast on the "
                        "convergence protocol; the bias-masked capacity "
                        "numbers below the transition remain logs "
                        "(PREREG_refraction_capacity.md).",
        claim="[MEASURED 2026-09-13 on two disjoint blocks of twenty "
              "registered brains, PREREG_refraction_convergence.md: a "
              "refracted RECURRENT area below the transition does not "
              "converge and hold. It forms, then RELOCATES ON A FIXED PERIOD "
              "equal to the clip arithmetic ln(w_max)/ln(1+beta) + "
              "(1-1/w_max)/beta = 40.93 rounds -- measured mean spacing 41.60 "
              "and 41.52 on the two blocks (1.6% and 1.4% error against a "
              "registered 10% tolerance), with the first relocation on round "
              "42 on ALL FORTY brains and no spread, because the deadline is "
              "set by the potentiation schedule and not by the connectome. "
              "The formula is a LAW AT s = 0.5 beta, the adopted operating "
              "strength, and ONLY there: a four-arm strength sweep at one "
              "operating point measured 59.49, 47.15, 41.60 and 40.78 rounds "
              "at s/beta = 0.25, 0.375, 0.5 and 0.625, a 25.9% spread, so the "
              "period depends strongly on strength and the formula carries no "
              "strength term. A 1/2s erosion term, which the mechanism "
              "suggests and which coincides with the written form exactly at "
              "0.5 beta, is closer in three arms of four and still wrong by "
              "16.6% at the slowest; NEITHER closed form is adopted and the "
              "dependence stands as four measured points. Capacity and tenure "
              "part company here: strength is a capacity SWITCH with one "
              "plateau over 0.3-0.6 beta, yet tenure moves 13% between 0.375 "
              "and 0.5 beta inside it. At the adopted strength the law was "
              "swept across "
              "beta in {0.05, 0.10, 0.20} and w_max in {5, 20, 100} on twenty "
              "brains per cell it holds in every cell to 4.1%, with the "
              "measured w_max ratio 2.354 against a predicted 2.34. The "
              "per-brain intervals are 0.1 rounds wide, tight enough to show "
              "the plain form is an approximation: it falls OUTSIDE the "
              "interval in four of five cells. DISCRETISING it, "
              "ceil(ln(w_max)/ln(1+beta)) + (1-1/w_max)/beta -- a weight "
              "needing 31.43 rounds of growth clips on round 32 -- cuts the "
              "mean error from 4.05% to 0.78% on three further cells chosen "
              "before the run and sharing no coordinate with the first five, "
              "closer in 3 of 3. Neither form lands inside an interval: the "
              "measurement is sharper than either approximation. For small "
              "beta the period is within half a percent of "
              "(ln w_max + 1)/beta, so TENURE IS BOUGHT WITH BETA AND NOT "
              "WITH THE WEIGHT CEILING: a fourfold beta change moved the "
              "period by 3.6, a twentyfold w_max change by 2.4. The "
              "unrefracted control had ZERO relocations on every brain of "
              "every cell though its weights clip on the same schedule, so "
              "the clip alone does not move an assembly. "
              "Between relocations the winners are unchanged: 5 relocations of "
              "6-10 rounds each in 240, stable fraction 0.80-0.84. A "
              "FEEDFORWARD area at s = beta does the same more often (11-32 "
              "events) rather than holding. At s >= 0.8 beta there is no "
              "stability to relocate from: one event covers the whole run, "
              "stable fraction 0.000, fill 1.000. The 0.7 arm is a third "
              "regime, churning for about half the run in one 97-120 round "
              "event and restabilising, so the transition lies between 0.5 "
              "and 0.7 beta. The earlier readings 'converges and holds' and "
              "'feedforward holds' were a shorter window: their bars failed "
              "and are retained.] "
              "[RE-MEASURED 2026-09-04 with the selector fixed (1b475fc): the "
              "churn above ~0.75 beta stands; the intermediate-strength rows "
              "were a selector artefact -- at 0.5 beta the recurrent assembly "
              "converges, relocates once when the clip binds (~round 40-60, "
              "the registered P2 prediction) and holds; at 0.7 beta most "
              "brains no longer converge. The transition lies in 0.5-0.7 "
              "beta. And BELOW it, with the bias-MASKED readout, a refracted "
              "recurrent area at 0.5 beta stored every assembly of the grid "
              "(rank-1 1.000 to M = 32, above the bar to 256, pairwise 0.00 x "
              "chance) against a Hebbian ceiling of 23.5 -- capacity becomes "
              "FILL-limited far above the interference limit; the earlier "
              "'spends the substrate' reading was the defect. Post hoc; a "
              "registration is owed. PREREG_refraction_capacity.md.] "
              "Refraction at strength s is the anti-Hebbian counterweight on a "
              "neuron's own repeated input: raw*(1+beta)^t minus the charged "
              "bias leaves net drive growing by (beta - s)*raw per win, so at "
              "s = beta it is CONSTANT. A feedforward area needs no convergence "
              "force (its input ranking is fixed) and holds; a RECURRENT "
              "assembly converges only through rich-get-richer, and above "
              "s ~ 0.75 beta it never converges and churns through the whole "
              "area -- refraction there is a firing-rate equalizer, and "
              "firing-rate homeostasis is incompatible with attractor memory "
              "in a recurrent k-WTA area. Below the transition it is the "
              "anti-merging force of [[REFRACTION-ANTI-MERGING]]: ~25x the "
              "Hebbian ceiling, read with the bias masked.",
        source="This repository; PREREG_refraction_capacity.md.",
        evidence=("period-law-strength-20260913 (four strengths at w_max 20, "
                  "beta 0.10, twenty brains each): 59.49 [59.46, 59.51], "
                  "47.15 [47.10, 47.20], 41.60 [41.54, 41.66], 40.78 "
                  "[38.98, 42.58] at s/beta 0.25, 0.375, 0.5, 0.625; ST-1, "
                  "ST-2 and ST-3 all FAIL, so the period is neither "
                  "strength-independent nor of the 1/2s form",
                  "period-law-amendment2-20260913 (three cells unseen by the "
                  "refinement, twenty brains each): measured 13.57, 20.83, "
                  "30.23 at (w_max, beta) = (8, 0.25), (12, 0.18), (25, 0.15) "
                  "against plain 12.82, 20.11, 29.43 and discretised 13.50, "
                  "21.09, 30.40; mean relative error 4.05% plain against "
                  "0.78% discretised; zero control relocations",
                  "period-law-v2-20260913 (five cells, twenty brains each, "
                  "per-brain spacing with intervals): 22.05 [22.02, 22.08], "
                  "41.60 [41.54, 41.66], 80.05 [78.66, 81.43], 25.06 "
                  "[25.01, 25.10], 58.98 [58.93, 59.03]",
                  "period-law-20260913 (the same five cells, pooled spacing "
                  "superseded by the per-brain statistic): "
                  "measured mean spacing 22.05, 41.60, 79.92 at beta 0.20, "
                  "0.10, 0.05 (w_max 20) and 25.06, 41.60, 58.98 at w_max 5, "
                  "20, 100 (beta 0.10), against predictions 21.18, 40.93, "
                  "80.40, 24.89, 58.22; one doublet in 681 spacings; control "
                  "relocations 0 of 0 in every cell with stable fraction "
                  "1.000",
                  "refraction-convergence-fresh-20260913 (the confirmatory "
                  "block, seeds 62-81, protocol version 3): all ten "
                  "Amendment 2 bars pass -- 5 relocations per brain, first at "
                  "round 42, spacings 40-43 (mean 41.52 against the predicted "
                  "40.93), relocation length 6-10 rounds, stable fraction "
                  "0.801-0.837; feedforward 11-32 events at 0.796-0.860; "
                  "0.7 beta 0.249-0.290 with one event of 97-120 rounds",
                  "refraction-convergence-20260913 (twenty brains, seeds "
                  "42-61, 240 one-round episodes, seven arms): late "
                  "consecutive overlap control 1.000; feedforward 0.912-0.975; "
                  "0.5 beta 0.904-0.932 (conv 217-220 on every brain, fill "
                  "0.28); 0.7 beta 0.632-0.966 (fill 0.91); 0.8 beta <= 0.277; "
                  "0.9 and 1.0 beta <= 0.010 (fill 1.000)",
                  "seq_refraction_wander.py at n=4000 k=100 p=0.5 beta=0.1 "
                  "w_max=20, 16 brains, 240 rounds: s/beta = 0.5, 0.7 converge "
                  "(rounds 48, 45 vs control 4; late stability 1.000); 0.8, "
                  "0.9, 0.95, 1.0 never converge (late stability <= 0.22, fill "
                  "1.000); feedforward at s = beta holds (late 0.993)",
                  "the w_max saturation arithmetic ln(w_max)/ln(1+beta) + "
                  "(1-1/w_max)/beta ~ 41 appears as a transient re-ranking at "
                  "rounds 44-48 below the transition, which the assembly "
                  "survives",
                  "capacity protocol at s = 0.5 beta, re-measured with the "
                  "selector fixed: see [[REFRACTION-ANTI-MERGING]] (the "
                  "earlier 'M* 19.6 vs 23.5' reading was the defect)",
                  "bias-on partial-cue recall 0.250 vs bias-masked 0.984 at "
                  "M=8, same training: the intrinsic bias vetoes recall from "
                  "a partial cue, as the identity predicts"),
        evidence_refs=(
            EvidenceRef("research/results/runs/memory.refraction-convergence/refraction-convergence-20260913/results.json", "artifact"),
            EvidenceRef("research/results/runs/memory.refraction-convergence/refraction-convergence-v2-20260913/results.json", "artifact"),
            EvidenceRef("research/results/runs/memory.refraction-convergence/refraction-convergence-fresh-20260913/results.json", "artifact"),
            EvidenceRef("research/results/runs/memory.refraction-period-law/period-law-20260913/results.json", "artifact"),
            EvidenceRef("research/results/runs/memory.refraction-period-law/period-law-v2-20260913/results.json", "artifact"),
            EvidenceRef("research/results/runs/memory.refraction-period-law/period-law-amendment2-20260913/results.json", "artifact"),
            EvidenceRef("research/results/runs/memory.refraction-period-law/period-law-strength-20260913/results.json", "artifact"),
            EvidenceRef("research/notes/memory/PREREG_refraction_period_law.md", "registration"),
            EvidenceRef("research/experiments/refraction_period_law.py", "producer"),
            EvidenceRef("research/notes/memory/PREREG_refraction_convergence.md", "registration"),
            EvidenceRef("research/experiments/refraction_convergence.py", "producer"),
            EvidenceRef("research/experiments/seq_refraction_wander.py", "producer"),
            EvidenceRef("research/notes/memory/PREREG_refraction_capacity.md", "registration"),
            EvidenceRef("research/notes/memory/AUDIT_refraction_scaling.md", "analysis"),
        ),
        provenance_gap="the wander diagnostic's and the bias-readout numbers have no immutable result artifact; the convergence contrast now does (refraction-convergence-20260913)",
        preconditions=("recurrent area; strength quoted relative to beta; "
                       "T=8 rounds per item in the capacity protocol",
                       "the reference uses RefractedArea only as a FEEDFORWARD "
                       "conjunction area driven by its full input at recall, "
                       "where none of this applies"),
        implemented_by=("neural_assemblies/core/_homeostasis.py",
                        "neural_assemblies/core/torch_engine/_hashed.py"),
        caveat="The critical ratio is bracketed in (0.5, 0.7) at one operating "
               "point on forty brains (the earlier (0.7, 0.8) came from the "
               "16-brain diagnostic); a transient-handicap estimate gives "
               "~2/3. The period law holds across EIGHT cells (beta "
               "0.05-0.25, w_max 5-100) in its discretised form AT s = 0.5 "
               "beta ONLY: a strength sweep refuted both the "
               "strength-independent form and the 1/2s form, and the "
               "dependence on strength stands as four measured points with no "
               "adopted closed form. The 0.625 arm sits near the churn "
               "transition and carries an interval 70x wider than the others, "
               "so it is the least trustworthy of the four. Everything is at "
               "ONE "
               "density and ONE refraction strength: the formula carries no "
               "strength term and strength was not varied, so the period's "
               "independence of s is suggested by the algebra and NOT "
               "measured. Three bars of Amendment 1 failed on "
               "mis-set thresholds (formation counted as an event; "
               "relocations run ~8 rounds, not 2-4) and are retained. Whether "
               "REFRACTION-NEEDS-LOAD's under-loaded non-convergence is this "
               "mechanism (the arc's state input is itself changing) is "
               "suggested, not established. TWO FURTHER LIMITS (AUDIT_"
               "refraction_scaling.md): refraction + synaptic scaling on the "
               "same area is INCOMPATIBLE -- a feedforward arc that holds "
               "under either alone loses its assemblies within ~10 "
               "presentations under both (late stability 0.12 vs 1.00), "
               "because scaling moves the raw drive the bias is charged "
               "against; and with no clip the identity dies of float32 "
               "cancellation at ~100-150 wins. Never scale a refracted area.",
    ),
]

# Redistribution needs a budget worth redistributing — Zipf lost every exam while the mass mechanism replicated inside it

**Task #142 (E13). Experiment: `research/experiments/zipf_synthesis.py` (bars A1/A2/B/C registered before any Zipf corpus trained; per-item collection extended to the arm's own attested forms after the first run exposed a blind spot — bars unchanged). {UNIFORM, ZIPF} × seeds 42–51, SLOW-scaled R1, n=3000.**

## Verdicts

| bar | value | verdict |
|---|---|---|
| A1 attested exam, paired ZIPF−UNIFORM | −0.067 ± 0.072 (0.583 vs 0.650) | **FAILS** |
| A2 fixed probe, paired | **−0.280 ± 0.066** (0.370 vs 0.650) | decisively negative |
| B image guard (shared SG∩PL cols) | 2.7 vs 2.6 | passes — no merging |
| C mass mechanism, within ZIPF | ρ(mass Δ, correct) = **0.761 ± 0.070** | **CONFIRMS E12** on a different corpus |
| guards | UNIFORM = E9's 0.650 exactly; tense flat; roles 40/40 | clean |

## Why it lost — the per-item table, not a theory

The zipf corpus (deterministic per arm — the generator re-seeds
internally) attests **9 PL forms fully disjoint from the fixed probe's
10**: all ten uniform-stable items sit at ZERO exposure and 0.00
accuracy, which is what A2's −0.28 is made of. And within its own
forms, subject-level concentration never became form-level
concentration: best form 2.0 exposures ("children" → 1.00, exactly on
the mass curve; "papers" 2.0 → 0.50), seven forms at 1.0 (acc
0.20–0.80). The uniform realization was *accidentally more concentrated
in PL space* ("girls" 4.0 → 1.00) — the PLURAL_RATE=0.30 coin thins and
decorrelates subject draws from plural surfaces, and one realization of
~42 draws is noisy enough that rank-frequency weighting on subjects
adds nothing downstream of the coin.

The deeper arithmetic: at 50 frames × ~0.3 plural, the whole corpus
contains **~a dozen PL episodes** in either arm. No allocation of ~12
episodes over ~9–10 forms can put more than one or two forms into the
reliable (≥2–4 exposure) regime — the SUM is the binding constraint,
and redistribution moves shares of a sum that is too small to matter.

## Deviation from the registered decision rule, stated

The registration said "A1 fails → the readout suspect is next." That
branch assumed concentration would HAPPEN and fail to pay. Measured:
concentration never happened at form level, while bar C shows the
readout's input (per-item afferent mass) explaining outcomes at 0.76 —
the readout is not the suspect, the lever's delivery is. The
A1-fail branch is therefore superseded by the per-item evidence, and
saying so here is the pre-registration discipline, not an escape from
it.

## What survives, and the one composition left

1. **The mechanism chain is now corpus-general**: mass → correctness at
   ρ 0.75–0.87 on every corpus tested (E12 uniform, E13 zipf).
2. **Exposure is the input to mass** and ~2–4 episodes is the reliable
   regime (children/girls at 1.00; 1-exposure forms scatter 0.2–0.8).
3. **E7's exposure-pinning invariant is a uniform-sampling artifact**:
   under uniform, coverage widens as fast as the budget, pinning
   exposure at ~1.5 forever. Under Zipf, coverage is BOUNDED (~9 forms
   at any budget) — so per-form exposure must rise roughly linearly
   with budget. Zipf at fixed budget reallocates nothing; **Zipf ×
   budget is the composition that breaks the pinning**, and neither
   axis alone can (E7 measured budget-alone; E13 measured Zipf-alone).

**E14 (to register): the 2×2 that ends the arc.** {UNIFORM, ZIPF} ×
FRAMES_PER_STAGE {50, 200}, SLOW-scaled R1, 5+ seeds. Pre-run
arithmetic: zipf at 200 frames ≈ 170 subject draws ≈ 50 PL episodes
over ~9–12 bounded forms ≈ 3–5 exposures each — the whole attested set
enters the reliable regime. Bars: (a) the interaction term — zipf's
budget slope exceeds uniform's (the pinning break is the claim, so the
INTERACTION is the bar, not any single cell); (b) zipf-200 attested
balanced ≥ 0.75 (the E-series bar, at last); (c) image guard at floor;
(d) mass mechanism replication. If (a) holds, the arc closes: number
reliability = per-form mass = exposure = budget × allocation-shape, and
the corpus-side scaling story (Zipfian real text at real size) is what
the substrate has been waiting for all along.

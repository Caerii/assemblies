# Uniform corpora pin per-form exposure at ~1.5 — at every size — and the exam grows with the corpus

**Task #136 (E7). Experiment: `research/experiments/episode_budget_recall.py` (pre-registered B1–B5). Budgets {100, 200} × {OFF, SCALED, BOTH}, 10 seeds, vs measured budget-50 references.**

## Verdicts

| config | b50 (ref) | b100 | b200 |
|---|---|---|---|
| OFF | 0.530 ± 0.038 | 0.506 ± 0.018 | 0.505 ± 0.032 |
| SCALED | 0.650 ± 0.058 | 0.617 ± 0.047 | 0.621 ± 0.041 |
| BOTH | 0.710 ± 0.055 | 0.614 ± 0.053 | **0.586 ± 0.046** |
| distinct PL forms | 10 | 18 | **40** |

- **B5 PASSES emphatically** — the corpus diversifies fast with budget.
- **B1/B2 REFUTED**: nothing rises; BOTH declines monotonically. (The b100 smoke cell's 0.778 was single-realization optimism, the third time a smoke cell has over-promised — smoke is for API breakage only, never direction.)
- **B3 borderline**: tense holds at b100, degrades at b200.

## Two readings, both true

**1. Uniform sampling pins exposure.** Plural events scale with budget (15 → 30 → 60) but with-replacement coverage widens just as fast (10 → 18 → 40 forms), so per-form exposure sits at ~1.5 **at every corpus size**. E6's conjugacy is not a fixed-budget artifact — it is an invariant of uniform sampling. No budget raises reliability; the corpus grows sideways.

**2. The exam grows with the corpus.** Balanced accuracy is scored over *attested* forms, so b200's flat 0.62 is measured over 4× the items — the system learned **four times as many associations at constant per-form reliability**, which is extensive scaling of the feature lexicon reported as stagnation. The metric conflates reliability with load; every cross-budget comparison in E7 carries this confound, stated here rather than hidden. (BOTH's decline survives the caveat — it is worse on the growing exam *and* was built for a regime, few-forms-high-contrast, that diversification removes.)

## What actually raises reliability: repetition

Per-form reliability is set by per-form exposure. Uniform corpora cannot raise it at any size; **real corpora raise it through Zipf** (frequent forms recur — and the Zipf memory warns frequency must be g-compensated for composition), and **children raise it through repetition** (the same utterances, many times). The developmentally faithful, label-free knob is *epochs*: repeat the morph-training phases over the same corpus, raising exposure at fixed diversity. E8 (registered next) sweeps morph-phase repetitions R ∈ {1, 2, 4} at budget 50, scoring BOTH the full attested set and a fixed common probe subset so reliability and load are finally measured separately.

Note the arc's irony: E1's original "the residual is exposure" was right at the per-form level — E4 rejected it because *novelty gain* couldn't create exposure contrast that wasn't there; repetition creates the exposures themselves.

## Standing

Defaults unchanged everywhere (budget 50, uniform, mechanisms off). The E-series has now excluded: mechanism caps, mechanism formulas, substrate size, form diversity, and corpus size — each pre-registered, each survived by exactly one hypothesis. Exposure-via-repetition is the last one standing, and it is E8's to confirm or kill.

# Diversity without data is dilution — form count and per-form exposure are conjugate at fixed budget

**Task #135 (E6). Experiment: `research/experiments/diverse_forms_recall.py` (pre-registered D1–D4). Diverse corpus (coverage sampling + object plurals) at OFF/SCALED/BOTH, 10 seeds, vs the measured uniform references.**

## Verdicts

| config | uniform (measured) | diverse | tense uniform → diverse |
|---|---|---|---|
| OFF | 0.530 ± 0.038 | 0.544 ± 0.047 | 0.678 → 0.559 |
| SCALED | 0.650 ± 0.058 | **0.578 ± 0.088** | 0.642 → 0.578 |
| BOTH | 0.710 ± 0.055 | **0.566 ± 0.092** | 0.694 → 0.595 |

- **D1 PASSES** — 16 distinct PL forms in every cell (bar 13): the manipulation reached the training corpus. Which is what makes the rest interpretable.
- **D2 REFUTED, inverted**: diversity made the mechanized configs substantially WORSE. The BOTH config lost 0.14.
- **D3 REFUTED**: tense fell 0.05–0.12 everywhere — the diluted corpus weakens every per-form association, not just number's.

## The corrected model, three experiments deep

E5 localized the ceiling to "image structure — too few distinct plural forms." E6 tested the direct implication and refuted it: **adding forms at a fixed episode budget subtracts exposure per form, and the net is negative.** The two quantities are conjugate; the true constraint is their product — the **total episode budget** (~50 frames/stage → ~15 plural events). Diversity and strength can only rise together if the *corpus grows*.

This also retroactively explains E6's design-time observations: the uniform corpus's 10 forms at ~1.5 exposures each was, apparently, near the optimum for a 50-frame budget — the generator's with-replacement sampling was accidentally trading diversity for strength in roughly the right proportion.

## E7 (registered next): the episode budget itself

`generate_generic` caps at `min(50, |nouns|·|verbs|)` frames per stage — a constant nobody has ever varied. E7 sweeps frames/stage (50 → 100 → 200) at unchanged rates and unchanged (uniform) sampling: total plural episodes scale linearly, per-form exposure and form diversity BOTH rise (with-replacement sampling naturally widens coverage with more draws). Registered expectation: if the budget is the true constraint, SCALED and BOTH climb with frame count and the 0.75 bar falls at some budget; training cost scales linearly and the speed stack absorbs it. More sentences per stage is also the single most realism-faithful change available — a child hears thousands of utterances per day, not fifty.

## Standing

Defaults unchanged (uniform corpus, mechanisms off) — E6's arm is not adopted, which is exactly why it was an arm and not a default. The E-series scoreboard: scaling remains the only established mechanism; the ceiling's cause is now pinned to the episode budget, pending E7.

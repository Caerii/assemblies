# The cap is innocent under scaling — and OFF's gains were partly cap-mediated compression

**Task #139 (E10). Experiment: `research/experiments/wmax_census.py` (pure measurement, arithmetic done before the run). 12 cells, fraction of class-image column weights ≥ 0.9·w_max.**

## The census

| config | NUMBER:SG | NUMBER:PL | TENSE:PRESENT | TENSE:PAST |
|---|---|---|---|---|
| OFF R1 | 0.009 | 0.001 | 0.000 | 0.001 |
| OFF R4 | **0.236** | 0.131 | 0.049 | 0.062 |
| SLOW R1 | 0.000 | 0.000 | 0.000 | 0.000 |
| SLOW R4 | **0.000** | 0.000 | 0.000 | 0.000 |

- **The saturation suspect is DEAD for the scaled configurations**: zero pile-up, max weights 8–14 against cap 20, means at the setpoint. Scaled-number's 0.65 ceiling is not w_max. The suspect list from E9 is now empty; per the registration, the ceiling is declared **unexplained** — but the census supplies the next hypothesis (below).
- **The pre-run arithmetic was right where scaling is absent**: OFF-R4 shows exactly the predicted asymmetry (SG 23.6% ≫ PL 13.1%). Which yields an unregistered but important reinterpretation: **OFF-R4's improvement (E8's baseline lift) was partly cap-mediated compression** — the weight ceiling truncates the frequent class's runaway, acting as an accidental homeostat. The "raw" baseline was never mechanism-free; w_max is a hidden normalizer, and the rounds-buy-convergence memory documented the cap's regulating role long ago without this connection.

## The new hypothesis, born from exclusion

Nine experiments excluded mass, exposure contrast, substrate size, form diversity, corpus size, scheduling, and now the cap. What remains is structural, and it was in our memory all along (k-WTA hardens shared grounding): **a plural form's core assembly substantially overlaps its own lemma's** — they share a `GroundingContext` verbatim, separated only by phonology at phon_weight 6 — and the lemma trained SG many times. The PL→NUMBER probe therefore drives through an assembly that is largely the SG-trained one; the number signal rides only on the form's *non-shared* fraction, which no learning-rule or corpus manipulation can enlarge. That would cap PL accuracy near the non-shared fraction and explain the entire E-series pattern: every manipulation moved everything *except* the PL ceiling.

**E11 (to register): item-level test.** Correlate per-form PL accuracy with measured core-assembly overlap(form, lemma) across seeds. Confirmed if the failures are the high-overlap forms, stable across configs. If confirmed, the levers are representational (larger phon share for inflected forms, or a dedicated inflection route — the papers' LEX-layer territory), which would close the arc by relocating the problem from learning to *representation* — the one layer the E-series never touched.

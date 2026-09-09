# The ceiling does not move with n — it is image structure, and the substrate is exonerated

**Task #134 (E5). Experiment: `research/experiments/ceiling_vs_n.py` (pre-registered C1–C3). OFF/SCALED/BOTH at n=6000, seeds 42–51, against the measured n=3000 reference.**

## Readings

| config | n=3000 | n=6000 | registered verdict |
|---|---|---|---|
| OFF | 0.530 ± 0.038 | 0.600 ± 0.041 | **MOVES** — crowding relieved by neurons |
| SCALED | 0.650 ± 0.058 | 0.640 ± 0.055 | flat |
| BOTH | 0.710 ± 0.055 | 0.695 ± 0.080 | flat |

The pattern is more informative than either single row: **scaling and n-doubling are partially substitutable.** The unscaled baseline gains from extra neurons exactly the component scaling already removes at fixed n — both attack aggregate class-mass crowding. Once scaling has harvested that, more substrate buys nothing: the mechanized ceiling (~0.70) is invariant to n.

## Conclusion of the E1–E5 arc

The 0.71 ceiling is **not** a learning-rule problem and **not** a substrate-size problem. Per the registered C2 reading, it is **image structure**: the corpus attests ~10 distinct plural forms, and a PL image built from few distinct forms is weak and narrow at any n. The frequency-imbalance program's residual bottleneck has been chased, by exclusion with pre-registered experiments at every step, out of the substrate entirely and into **corpus form diversity**.

The next unit is therefore a diversity change, not a mechanism or a rate: subject selection in the generator currently samples uniformly WITH replacement over the noun pool, which under-covers the vocabulary relative to natural long-tail text — many nouns never appear as plural subjects in a 50-frame stage. Coverage-biased subject sampling (favor unseen nouns; rates untouched) is a defensible realism improvement — real corpora have long-tail type diversity — and directly widens the set of distinct plural forms the PL image is built from. To be registered as its own unit with the same readouts.

## Arc summary (E1–E5, one day, all pre-registered)

1. **E1**: scoped homeostatic scaling — real, +0.12 established; removes accumulated mass.
2. **E2**: gain composes with scaling, perfect ordering; bars missed.
3. **E3**: the cap axis was vacuous (sqrt gain self-caps at 1.2); E2's gain reframed as familiarity suppression.
4. **E4**: linear gain is *worse*; per-form novelty cannot see class mass (every form is individually equally rare) — E1's exposure diagnosis corrected.
5. **E5**: the mechanized ceiling is n-invariant; scaling ≈ substrate-doubling for this readout; the residual is corpus form diversity.

Standing recommendations: scaling is the established mechanism (default still OFF pending adoption discussion); gain stays default-off as a documented negative; forcing-rate retirement remains gated; the diversity unit is the path to the 0.75 bar.

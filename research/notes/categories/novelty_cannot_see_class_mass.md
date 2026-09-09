# Per-form novelty cannot see class mass — the E1 "exposure residual" diagnosis was wrong

**Task #133 (E4). Experiment: `research/experiments/linear_gain_recall.py` (pre-registered L1–L5). 4 configs × 10 seeds.**

## Verdicts

- **L5 FAILED — predictably, from a number already in hand.** LIN:2 and LIN:4 are bit-identical because the maximum linear gain is the square of E3's measured max sqrt gain: 1.215² = 1.476 < 2. The cap never binds under ANY (mean_count/count)-type formula on this corpus. E3's instrumentation contained this bound; registering L5 caught it, but computing 1.215² *before* running would have caught it for free. Lesson recorded: when an instrumented quantity bounds a proposed manipulation, do the arithmetic first.
- **L1 REFUTED**: LIN best is 0.665 ± 0.077 — *below* sqrt-BOTH's 0.710. The linear form's stronger familiarity suppression (frequent forms down-weighted to ~0.47× vs sqrt's ~0.68×) **hurt**.
- **L2 REFUTED**: LIN−SCALED = +0.015 ± 0.079, seeds on both signs — the linear form adds nothing over scaling alone.
- **L3 passes** (+0.005 tense vs OFF — the no-tense-cost property again).

## The insight: the imbalance is not where the gain mechanism looks

Mean per-form exposure is 1.41 — meaning *every* surface form, plural and singular alike, is individually rare. There is no per-form frequency contrast for a novelty signal to exploit; "surprise" has nothing to see. The actual imbalance is **aggregate class mass**: many *distinct* singular forms all potentiate the same SG-image post-neurons, while few distinct plural forms feed PL's. That asymmetry lives per-*neuron* (column mass), not per-*form* (exposure count) — which is exactly why:

- scaling (per-neuron column normalization) produces the one established margin (+0.12 over OFF at n=10),
- per-form gain barely moves anything in either formula, and
- the sqrt form's tiny effect was likely its *gentle* suppression accidentally approximating a small extra mass correction, which the linear form overdid.

E1's closing diagnosis ("the residual is exposure — a form seen twice wrote only two updates") conflated per-form write count with per-class mass. Correcting the record: the writes exist and are visible under scaling; the residual gap to 0.75+ is something else — candidate suspects, in order: PL-image quality (few distinct forms → weaker, narrower image), k-WTA crowding between the two number images at k=30, and readout margin structure. The **ceiling-vs-n branch** (registered next, E5) discriminates: if 0.71 moves at n=6000, it's crowding/capacity; if not, it's image structure.

## Standing after E1–E4

| mechanism | number balanced (n=10) | verdict |
|---|---|---|
| OFF | 0.530 ± 0.038 | baseline |
| SCALED | 0.650 ± 0.058 | **the established mechanism** |
| BOTH (sqrt gain) | 0.710 ± 0.055 | best point estimate; margin over SCALED not established |
| BOTH (linear gain) | 0.665 ± 0.077 | worse than sqrt — suppression overdone |

Gain mechanisms as parameterized are **exhausted**: cap vacuous (E3), formula counterproductive (E4), and the mechanism is aimed at a contrast the corpus does not contain. `novelty_gain_max` stays default-off; the code stays (it is the honest record of what was tried, and the exp parameter documents the finding at its own docstring). The retirement A/B remains gated; E5 (ceiling vs n) is the live branch.

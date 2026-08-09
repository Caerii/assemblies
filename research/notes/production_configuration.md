# The production configuration — what the E-series fixed, at which level, and what you set to get it

**Task #149, the consolidation/adoption unit of the number arc (E1–E19b,
tasks #130–#148, closed at `897da37` with terminal 0.727 ± 0.037 against
the 0.75 bar). Every default flip is its own commit citing the run that
measured it, and every flip passed a gate: structural (byte-identical
default path) or measured (a paired ensemble at the default regime).
This note is the one place the whole configuration is stated together.**

## The five-level law (what was learned)

1. **Per-item reliability is afferent mass** into the right label's
   neurons (ρ 0.50–0.87 across five corpora/architectures — E11/E12/E15).
2. **Mass is per-form exposure**: ~2–4 episodes make a form reliable
   (E12; the residue of every terminal number is items at 1–2 episodes).
3. **Exposure = budget × allocation**: Zipfian allocation at a budget
   worth redistributing raises head-form exposure at constant total cost;
   neither alone works (E13 null, E14 interaction).
4. **Architecture**: two label values sharing one k-WTA area MERGE as
   label projections grow (2.6 → 17.8 shared image columns of 30 at 200
   frames — E14); one area per value inside a mutual-inhibition group
   removes this by construction and by intervention (E15 D1).
5. **Homeostatic schedule**: deferred synaptic scaling must flush on a
   RATE, not at phase boundaries. The right wall (within-interval mass
   concentration, which column normalization cannot undo) is
   unconditional; the left wall (colliding with fast Hebbian dynamics)
   is conditional on repetition-style training (E18, E19, E19b).

## The adopted defaults (what you get without setting anything)

- **`split_feature_areas=True`** — per-value feature areas
  (NUMBER_SG/NUMBER_PL, TENSE×5) in MI groups, created lazily by
  `train_tense`/`train_number`. Adopted through an n=10 PAIRED gate at
  the default corpus (bars registered before data): balanced tense delta
  −0.039 ± 0.058 (no measured harm), SG 0.920, and **PL 0.415 vs 0.085
  shared — the split takes plural recall off the chance floor at the
  default corpus**, not only at scale. The gate also re-taught the
  ensemble lesson in both directions: seed 42 alone read PAST −0.26
  (a floor-test failure that nearly refused the flip), seeds 43/44 read
  −0.26/+0.17 — noise-dominated, decided only at n=10. The named cost:
  tense PAST trends down under the split (seed-42 pin 0.826 → 0.565);
  the slow-test floors were re-pinned accordingly.
  `False` = the legacy shared-area path, byte-identical to pre-E15,
  kept reachable for parity reproductions.
- **`morph_flush_every=40`** — interim deferred-scaling flushes every 40
  morph episodes. Inert unless `synaptic_scaling_deferred` is on (the
  structural gate). The rule is FLUSH OFTEN: any interval well below the
  concentration scale is fine (K=1 tied K=40 at n=10); only
  repetition-style training (`morph_repetitions>1`) needs an interval
  floor (E9).
- **`morph_readout="mi"`** — the returned recall answer is the paper's
  mutual-inhibition drive competition. The readouts CROSS (E15): MI wins
  at the default 50-frame budget (0.620 vs 0.540 on number), the
  within-area overlap readout wins at ≥200 frames (0.700 vs 0.637;
  terminal 0.727 is overlap's). MI is a COMMIT device, not an
  accumulator — latching holds a decision exactly and never changes one
  (E16, accuracy delta 0.0000). Both answers always ride in diag.

## The paper-regime recipe — production at scale on real corpora (#151)

**Adopted by the re-run adoption gate (`145288c`): min(SG, PL) on the
E≥2 exam = 1.000 ± 0.000 across all ten seeds, balanced 0.955 ± 0.019,
PL E=1 0.848 ± 0.064, on Brown at n=10⁵. REQUIRES the fixed engine
(≥ `53e9808`) — on the pre-fix engine the same recipe read PL E≥2
0.779 ± 0.227 with a ~1-in-10 silent dead-fiber death
(the_dead_fiber_was_a_growth_ratchet.md). Figures:
research/figures/fig_151_*.**

For morph-feature learning from a REAL corpus (natural Zipf/imbalance),
on top of the defaults:

```python
p = EmergentParser(n=100_000, k=30, ...)   # papers' scale: collision
                                            # load V*k^2/n ~ 3 at V=370
p.morph_label_stim = False   # ROUTING-ONLY: the teacher picks the
                             # area, the WORD picks the winners -- the
                             # papers' construction; the label stimulus
                             # is an attractor that competes with the
                             # word at recall (the 2x2, 90f12e3)
p.morph_beta_gain = 4.0      # COLT22 Remark-2 margin lever at kp=1.5
p.train_number(sentences, labels=corpus_teacher)  # corpus annotation
# scaling stays OFF (the #150 inversion, below); readout stays "mi"
```

Split value areas are a PREREQUISITE of routing-only, not an
independent axis: with one shared area the routing IS the label, so
there is nothing to route. Reporting is STRATIFIED BY EXPOSURE — the
E=1 stratum is priced (0.85), not hidden.

Scope notes from the Phase 1 close-out (childes_phase1_recipe,
`fd8b1c4`): the exposure law transfers attenuated at ceiling
(ρ = 0.26 ± 0.05 vs 0.61 pre-fix; interpretation rule pre-registered);
ALL residual failures are E=1 words carrying 64% more SG-shared row
mass (10.3 vs 6.3) — collision load is the residual, feeding the open
theory unit on why E=1 clears the naive margin at all. Deferred
scaling is DORMANT in this recipe (P4 liveness), so the E19 schedule
claims below are scoped to the synthetic configuration.

## The measured-best (production/scale) configuration

For a morph-feature learner at scale — and for the CHILDES graduation —
set, on top of the defaults:

```python
generation.SUBJECT_SAMPLING = "zipf"   # allocation
generation.FRAMES_PER_STAGE = 200      # budget (the interaction partner)
parser = EmergentParser(..., synaptic_scaling={value areas},
                        synaptic_scaling_deferred=True)
parser.morph_readout = "overlap"       # the >=200-frame winner
```

This is the E19b terminal cell: zipf-200 × split × K=40 × overlap =
**0.727 ± 0.037** (n=10 seeds).

**SCOPE LIMIT measured by the #150 graduation (Brown, 90/10 class
imbalance): deferred scaling INVERTS the number decision on real CDS —
per-column normalization equalizes value areas whose form loads differ
~8x and the rare class's area becomes a constant responder (SG 0.036
scaled vs 0.513 unscaled, causal by intervention). Until the per-form
normalization unit lands, deferred scaling on feature value areas is
recommended ONLY for corpora near the synthetic 70/30 imbalance; on
natural-imbalance corpora run scaling OFF. See
research/notes/the_real_corpus_inverts_the_collapse.md.**

The corpus default stays
`SUBJECT_SAMPLING="uniform"` because zipf at the default 50-frame budget
was measured a null (E13) and the uniform-50 corpus underlies every
non-morph arc's standing measurement; the zipf knob dissolves entirely
under a real corpus, whose statistics are Zipfian by nature (#30/#150).

## Named residuals (why 0.75 was not cleared)

- Exposure tail: most failing items sit at 1–2 episodes, below the ~3–4
  reliable regime — a corpus-statistics fact, not a substrate defect.
- Cross-area drive comparison is the model class's weak primitive
  (margins pinned at 7–10%; third independent measurement of #24) —
  visible again in the adoption gate as the split's tense-PAST trend.
- The non-monotone budget curve (0.700@200 → 0.570@400 per-phase) is
  largely but not fully recovered by the schedule (0.658@400 at K=40).

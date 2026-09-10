# PREREG: the sampler audit

Registered 2026-09-09, before running. The sequence port found that the
numpy engine's lazily drawn areas produced a false horizon (A1, p = 0.3),
a soft-transition rate seven times too high, and every derailment in the
S5 census (DESIGN_sequence_port.md; PREREG_s5_cliff_anatomy.md Addendum 3).
Three register entries rest on experiments that ran with the arc area
drawn lazily. Each is re-run with the arc area materialized, the only
change, on the same seeds.

    entry                 script                         what it claims
    SEQ-REGIME-CLIFF      seq_a1_exactness_sweep.py      exact recovery jumps between p = 0.2 and 0.3, where k p crosses 3 ln n_state
    REFRACTION-NEEDS-LOAD seq_a2_refraction_load.py      the single-mood arm's correctness rises as the arc shrinks (load M k / n rises)
    SEQ-ORGAN-EMBEDS      seq_a1_local_regime.py         an organ at its own density works inside a brain at p = 0.05

    SA-1  SEQ-REGIME-CLIFF stands: materialized, the exact-step fraction at
          p = 0.2 is below 0.5 and at p >= 0.3 above 0.9 (a cliff, not a
          slope).  PREDICTION: PASSES; the materialized engine is quieter
          than the sampled one everywhere so far, and the floor is a
          theorem's.
    SA-2  REFRACTION-NEEDS-LOAD stands: materialized, correctness at the
          smallest arc (load ~1) exceeds correctness at the largest (load
          ~0.1) by at least 3 seeds of 10 in the single-mood arm.
          PREDICTION: uncertain. The load window was measured on a sampled
          arc; if convergence failures at low load were the sampler's, the
          window disappears and the entry is re-scoped.
    SA-3  SEQ-ORGAN-EMBEDS stands: materialized, the organ at organ_p works
          embedded (decided >= 8/10) and at ambient p does not (<= 2/10).
          PREDICTION: PASSES.

Each script takes `NEMO_MATERIALIZE=1` in the environment and writes its
results with a `_materialized` suffix. A FAIL re-scopes the entry to the
sampled engine and records the materialized numbers beside it.

## Result (2026-09-09, 10 seeds each, arc materialized; sampled originals beside)

    SA-1  SEQ-REGIME-CLIFF.  exact-step fraction by p (materialized | sampled):
          p 0.2: 0.22 | 0.04     p 0.3: 0.98 | 0.80     p 0.4: 1.00 | 1.00     p 0.5: 1.00 | 1.00
          Below 0.5 at 0.2, above 0.9 from 0.3: the cliff stands.              PASS
          Materialized, every trajectory is correct even at p = 0.2 (10/10
          against the sampled 4/10): the cliff is in exactness, and the
          sampler had made it a cliff in decisions as well.
    SA-2  REFRACTION-NEEDS-LOAD.  single-mood arm, correct of 10 by load
          (materialized | sampled): 0.04: 10 | 1    0.10: 10 | 7    0.21: 10 | 10
          0.42: 10 | 10    0.60: 10 | 10.  No rise with load.                   FAIL
          multi-mood arm: 10/10 to load 1.26 both ways; at 1.80, 0/10 | 1/10.
          The window's LOWER edge -- refraction inert below load ~0.2, the
          entry's claim -- was the sampler's. The UPPER edge (collapse near
          load 1.8) stands. The entry is re-scoped from a window to a
          ceiling.
    SA-3  SEQ-ORGAN-EMBEDS.  organ at organ_p: 10/10; ambient only: 0/10.     PASS

**Reading.** Two of three entries stand and are sharper materialized. The
third loses half its claim: the low-load failures that named the entry
were convergence failures of the sampled engine, and on the explicit
substrate a refracted conjunction converges at any load below the
ceiling. That is consistent with everything the port found: the sampler
adds failures, never successes.

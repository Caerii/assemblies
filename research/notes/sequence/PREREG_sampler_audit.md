# PREREG: the sampler audit

Registered 2026-09-09, before running. The sequence port found that the
numpy engine's lazily drawn areas produced a false horizon (A1, p = 0.3),
a soft-transition rate five times too high, and every derailment in the
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

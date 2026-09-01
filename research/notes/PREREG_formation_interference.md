# PREREG: is the capacity ceiling set at FORMATION, not at retrieval?

Registered before running. Successor to `PREREG_crosstalk_mechanism.md`, whose
retrieval-side mechanism was refuted by its own held-out tests. Any mechanism
now has three constraints to satisfy:

    C1  M* ~ (n/k)^2 at a fixed operating point            (CS1, verified)
    C2  dM*/dp < 0, roughly p^-0.6 over [0.3, 0.7]          (X1)
    C3  dM*/dbeta < 0, roughly beta^-1.3 over [0.05, 0.2]   (X2)

## The hypothesis

The ceiling is set DURING TRAINING: while item M forms, recurrence pulls the
forming assembly toward already-stored attractors through trained foreign
edges, against the stimulus anchor. Capture strengthens with beta (trained
edges deeper) and with p (more shared edges) -- the C2/C3 signs -- and the
pull is second-order in the chance overlap, giving C1. Consistent with
[[recurrence-is-the-collapse-channel]] and [[beta-opposes-capacity-and-depth]],
and with the cliff: attractor capture is catastrophic, not gradual. Suggestive
prior evidence: at the cliff, DISTINCTNESS falls (0.86 at n=8000 M=384) --
exact duplicates form, which a readout-side mechanism cannot produce at all.

## The discriminator: the anchor

Formation capture is a fight between the stimulus anchor and the recurrent
pull, so STRENGTHENING THE ANCHOR should raise M*. A readout-side mechanism
cannot care: retrieval is cued by half the assembly with NO stimulus present.

**F2 (primary).** n=4000, k=100, p=0.5, beta=0.10, T=8, w_max=20, 16 brains,
arm B, exact path. Baseline M* = 23.5 (measured, bracket [20, 24)). Re-run
with `stim_size = 2k = 200` (anchor drive doubles; nothing else changes --
retrieval remains half-assembly-cued, no stimulus).

    PASS (formation): M*(2k) / M*(k) >= 1.3
    REFUTE (readout-side after all): ratio <= 1.1
    between -> inconclusive, reported as such

**F3 (secondary -- capture is CONCENTRATED).** At M = 32 (~1.4x M*), per-item
forensics: for each stored item, rank-1 hit/miss and its MAX overlap with any
other stored assembly.

    PASS: median over FAILED items of max-overlap >= 2x the median over
          RECALLED items, and the distinctness losses pair a LATER item with
          an EARLIER one (capture has a direction).
    FAIL: failed items' max-overlap indistinguishable from recalled items'
          (diffuse failure, not capture).

## Interpretation, stated now

* F2 and F3 pass -> formation-capture ADOPTED as the mechanism CLASS (the
  quantitative law stays open; no exponent is quoted from this).
* F2 refutes -> the anchor does not protect; formation-capture in this form is
  dropped, and the C2/C3 signs need another source.
* F2 passes, F3 fails -> the anchor matters but failure is not concentrated
  capture; reported as partial support, not adopted.

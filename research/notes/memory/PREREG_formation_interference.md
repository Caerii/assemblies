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

---

## Result (2026-08-25): F2 PASSES 2.86x, F3 FAILS -- partial support, NOT adopted

**F2 -- PASS, decisively.** Doubling the stimulus anchor (stim_size 100 -> 200,
nothing else changed, retrieval still anchor-free):

    M*(k anchor)  = 23.5   [20, 24)   fill 0.53
    M*(2k anchor) = 67.1   [64, 80)   fill 0.83   uncensored

Ratio 2.86 against a PASS bar of 1.3. A readout-side mechanism cannot produce
this -- retrieval never sees the stimulus. The ceiling is set at FORMATION.

**F3 -- FAIL as registered, and the failure shape is the finding.** At M = 32
(1.36x M*, past the cliff), per-item forensics over 512 (item, brain) cells:

    overall rank-1                    0.031
    max-overlap median, FAILED items  0.190   (vs bar: >= 2x recalled)
    max-overlap median, RECALLED      1.000
    rank-1 by formation half          first 0.000   second 0.062

Failure is DIFFUSE (median max-overlap 0.19 -- 7.6x the k/n chance floor of
0.025, but nowhere near duplication), not concentrated capture. The registered
capture direction INVERTS: EARLY items fail worse than late ones, and the only
recalled cells are the EARLIER members of exact-duplicate pairs (their
max-overlap is 1.000; argmax ties resolve to the lower index, so the earlier
twin "wins" the readout). The partner-direction statistic is not over-read:
failed items concentrate early, whose item-conditional null for an earlier
partner is itself small.

**Per the registered interpretation: partial support, NOT adopted.** What is
established: the ceiling is FORMATION-side (F2), and the failure mode past the
cliff is RETROACTIVE -- later training erodes earlier attractors diffusely,
with occasional exact duplication whose earlier twin survives the argmax. What
is not established: the one-shot capture picture, whose concentration
signature is absent. Consistent with [[reinforcement-tradeoff]] (deep
reinforcement MERGES multi-assembly areas) more than with prospective capture.

Constraint ledger for the next mechanism attempt: C1-C3 as before, plus
C4 (anchor strength raises M*, ratio ~2.9 at 2x anchor) and C5 (retroactive,
diffuse, early-items-first).

# DESIGN: the sequence organ on the hashed substrate -- registered, not yet built

The aligner's port (DESIGN_hashed_aligner.md -> DESIGN_present_only.md) took
a five-cell sweep from hours to a minute and, on the way, found four
defects in the FORMULATION and none in the kernels. The sequence organ is
the next port because it is the state machine the parser and the census
stand on, and because its claims -- horizon as a hitting time, the arc's
operating window in load, state code emergence -- are exactly the ones a
20-seed distribution at width would settle.

## What is ported

`programs/sequence_transducer.py`, unchanged in its clock and its areas:

    stim[w]  -> LEX                 word            (StimulusFiber, anchor)
    gstim[w] -> OUT                 grounding       (StimulusFiber, anchor)
    LEX + STATE -> ARC              refracted conjunction  (two PresentFibers)
    ARC -> STATE                    feed-forward state update
    ARC -> OUT                      prediction

`HashedArea` already carries `refracted_strength` and is gated against the
numpy engine's refraction (`test_hashed_substrate_parity`, refracted
parity); `PresentFiber` carries the max-relative pricing and column scaling
the organ's fibers use; the organ's own density (`organ_p`) is a per-fiber
p, which the hashed fibers take by construction. One tick = one word; the
write repeats onto a fixed arc; `emit` reads OUT with no teacher.

## Gates (pre-registered; in the order the repo trusts them)

    GATE-1  DRIVE REPLAY. The numpy transducer runs `train_sentence` on a
            tiny vocabulary with `record_activation`; its winner trajectory
            is replayed through the hashed fibers and the pre-k-WTA drive on
            ARC, STATE and OUT compared every projection, INCLUDING the
            refraction bias's contribution on ARC (bias charged on the
            replayed winners). Tolerance: the aligner's 5e-6 on drives.
            Stimulus bases injected from the engine; area fibers generated
            from the engine's pair seeds.
    GATE-2  DECISIONS. On the same tiny corpus, `emit` after training ranks
            the same next word first wherever the numpy organ's margin is
            above one count (ties are recorded, not compared:
            [[exact-tables-are-tie-fragile]]).
    GATE-3  THE HITTING TIME. seq_a1_horizon's registered curve -- first
            divergence index over 2000-step mod-3 runs -- on 20 brains at
            p = 0.3 and p = 0.4: the hashed organ's distribution of first
            exits must contain the numpy organ's five seeds inside its
            central 90% at both p ([[horizon-is-a-hitting-time]]).
    GATE-4  IDENTITY ACROSS WIDTH. A brain's trajectory in a launch of 20
            equals its trajectory alone (the scheduled aligner's second
            gate, restated for the organ).

## Targets

    T1  A1's horizon experiment (5 seeds, 2000 steps, two p) in under a
        minute; 20 seeds in under five.
    T2  The arc's load window ([[arc-needs-a-load-window]]) re-measured as a
        curve over M k / n at 20 seeds per point.

## What to expect to find (the aligner's precedent)

Each port so far surfaced formulation facts the numpy path had hidden:
the 0-or-size stimulus weight ([[add-stimulus-zero-or-size]]), the anchor
gain, tie correlation, potentiation-count table sizing. For the organ the
candidates are named now so a discrepancy is recognised, not debugged:

* Refraction is charged by `project`, not by `probe`; the replay must
  charge exactly where the numpy engine charges and nowhere else
  ([[probe-isolation-required]]).
* The arc is a CONJUNCTION with an operating window in load; the hashed
  organ's per-fiber p must reproduce the numpy organ's `add_connectivity`
  scoping exactly (organ_p on four fibers, ambient p elsewhere).
* `ground` forms LEX and OUT with recurrence off; the OUT signature is in
  neuron ids ([[two-index-spaces-compact-vs-neuron-id]]); the hashed area
  is compact-indexed and the readout must map.

## Not in scope

The parser and the census (their organs stack on this one); the
`NemoArcFSM` unification the transducer's docstring owes; any change to
the organ's clock, densities or refraction strengths -- those are protocol.

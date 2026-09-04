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

## Progress (2026-09-04, evening)

**GATE-1 PASSED on the first run** (`test_hashed_transducer_parity.py`):
`HashedTransducer` replays the engine's winners through the four fibers and
matches the net drive within 5e-6 on every projection into LEX, ARC, STATE
and OUT, and the accumulated ARC refraction bias within 5e-6. The substrate
already carried refraction parity from the capacity work; the organ
inherits it. Committed 3d4aca6.

**The present fiber gained an ABSOLUTE mode** (5fccbae): the organ's
regime -- weight clip, no column scaling -- priced by count from the
engine's chain table on the same kernels, gated equal to the store fiber.

**And then the regime, in numbers.** A3's registered parameters are
n = 10000, k = 200, organ_p = 0.2, n_arc in (2000, 10000, 50000):

    n_arc     LEX->ARC row degree   drive vector (shared)   dense int16 counts / brain   visits per ARC projection
    2,000        400 entries            8 KB                    38 MB                        0.1 M
    10,000     2,000                   39 KB                   191 MB                        0.4 M
    50,000    10,000                  195 KB                   954 MB                        2.0 M

The present-only warp kernel was built for the aligner's regime (p = 0.05,
n <= ~8000): its per-row lists cap at 512 entries and its drive vector
lives in shared memory. At organ_p = 0.2 a row has 2,000-10,000 present
columns and an n_arc = 50,000 drive vector is 195 KB. So the organ's
registered protocol does not fit the layout that carries the aligner --
and at 20% density the DENSE count layout (predicated loads, ~full sector
utilisation) that was deleted today as dead code is the right one, priced
absolutely, with K = 200 rows and a global-memory drive. "One canonical
representation" turned out to be one per density regime: present-only
below ~10%, dense above. That is a substrate fact worth the day.

**Decision to register before building (next):**

* A dense-count kernel for the organ's regime: int16 counts [n_pre, n_post]
  per fiber per brain (191 MB at n_arc = 10,000; the 50,000 cell needs 1 GB
  per fiber and a smaller width), K up to 256 rows, absolute chain pricing
  with clip, norm_init, refraction on the target; the drive as a global
  [B, N] vector written by a block per brain in row order per column (a
  thread per column keeps that order for free -- the deleted kernel's
  structure).
* Per-brain schedules for the organ (each seed its own corpus): stacked
  stimuli indexed by word, -1 words and -1 rows as the dead-brain
  convention (the kernels skip them; a per-brain inhibit sets rows to -1).
* Then A3 at 20 seeds, paired against the numpy JSON's three, as the
  first width measurement; GATE-3 (the horizon hitting time) needs the FSM
  organ's assigned-state core on the same kernel.

## Built (2026-09-04, night): the organ at width, and what its first run found

* `DenseOrganFiber` (int16 counts, presence mask, absolute chain pricing,
  gated == store fiber; -1 rows/winners skipped) and the batched
  `HashedTransducer` (stacked per-word stimuli, per-brain schedules with
  idle steps and per-brain resets). GATE-1 (drive + refraction, 5e-6) and
  GATE-4 (identity across width, exact) pass. GATE-2 was superseded by the
  A3 study itself; GATE-3 (the FSM horizon) awaits the assigned-state core.
* A3 (PREREG_seq_a3_transducer.md) at 20 seeds in ~45 minutes: H5 pass,
  H1 PASS (+0.10 paired against #14, which three numpy seeds had judged the
  other way), H4 pass (0.07), H2/H3 fail, state-blind delta zero -- the
  state is distinct and uninformative.
* THE SUBSTRATE DEFECT the run found: `topk_select` ranked negative net
  drives above positives (raw float bits as key). Refraction is the only
  producer of negative drives, and the selector had only been gated on
  replayed winners; the organ's arc collapsed onto its most-biased neurons
  (bias 66 against a drive of 0.6, cross-prefix overlap 1.0). Fixed in
  1b475fc with a negatives unit test. Lesson for every port to come: a
  drive-replay gate cannot see a selection defect -- add a SELECTION gate
  on drives that go negative. The refraction registration's hashed numbers
  are suspended and re-running (PREREG_refraction_capacity.md CAVEAT).
* Also adopted: `StimulusFiber(zero_or_size=True)`, the engine's stimulus
  model, as the organ's default (not the cause, but the model).

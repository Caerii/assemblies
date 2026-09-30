"""Role binding driven by the inhibition state machine, not by Python.

THE CLAIM FIBER GATING IS ACTUALLY FOR
---------------------------------------
Gating spent the 2026-07-28 session solving problems that turned out to be
artifacts -- slot addressing, the two-phase read -- and each time the honest
outcome was that the substrate did not need it. This is the job the reference
does need it for, and it has never been measured here.

`research/PRIMITIVES_AUDIT.md` and `core/inhibition.py` state the gap: NEMO
gates WHICH PROJECTIONS HAPPEN (fiber/area state), while this repo names
targets in Python and applies winner-take-all AFTER firing. Task #24 measured
the consequence -- 1373 `project()` calls, ZERO of which co-target a group, so
the paper's mutual inhibition never runs and role exclusivity is symbolic.

The test of whether that matters is sequential role binding. Three words arrive
one at a time and must land in three different role areas. Under gating that is
automatic: only one fiber is open per step, so exclusivity holds because no
other projection was possible. Without gating the lexical area reaches every
role at once and the last word overwrites all three.

WHY THIS IS NOW WORTH RUNNING
------------------------------
The substrate underneath it is finally sound. Earlier attempts would have been
measuring collapse: feed-forward lexicon has no measured ceiling (1.0000 to
M=256 at n=1000), `ops.merge` gained channel gating, depth 3 is flat at 1.0000
with margin 4.58x, and 512 composed constituents hold at margin 6.67x.

DESIGN, and the two properties that make it honest
---------------------------------------------------
RULES ARE DATA-INDEPENDENT. The controller opens LEX<->ROLE_i at step i and
closes everything else. The sequence depends on POSITION only -- it never
consults the word. A controller that looked at the word would be smuggling in
the answer, which is the failure this line has had to guard against repeatedly.
The projection map is DERIVED by `InhibitionState.project_map`, not written by
hand, so the state machine is genuinely in the loop.

ROLE AREAS ARE SHARED. Every sentence binds into the same three areas. An
input-independent read therefore scores 1/M by construction, so the wrong-cue
control is architectural rather than bolted on -- the methodological lesson from
`depth_corrected_substrate.py`. With one area per binding the measurement would
be free and meaningless.

THE CONTROL ARM is the same protocol with every fiber open, which is what this
repo does today. It is not a strawman: it is the current mechanism.

THE HARD CONTROL is `swap`. The same word appears as SUBJECT in one sentence
and OBJECT in another. If role assignment came from the word rather than the
gate, one of the two must be wrong. This is the measurement that separates
"binding" from "lookup", and it is the reason the sentence set is constructed
rather than random.

PRE-REGISTERED
--------------
R1 Gated binding clears chance by a wide margin at all three positions.
R2 The ungated arm is at or near 1/3 of the roles correct -- the last word
   overwrites the earlier ones, because LEX reached every role simultaneously.
   If ungated ALSO works, gating buys nothing here and the symbolic role route
   is adequate; that would settle task #33 in the opposite direction and is the
   outcome that would make this file worth having anyway.
R3 swap holds: a word bound as SUBJ in one sentence and OBJ in another is read
   correctly in BOTH. Role comes from the gate, not from the word.
R4 Accuracy is stable as the number of sentences grows, since the role areas
   are shared and therefore crowding -- reported with margin, because at
   M=256 accuracy read 0.9544 while margin was 1.47x.
"""

from __future__ import annotations

import os
import statistics
import sys

os.environ.setdefault("TRAIN_PROGRESS", "0")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _substrate import MIN_TRIALS, probe, read, similarity  # noqa: E402

N, K, P, BETA = 2000, 50, 0.05, 0.10
BUILD_ROUNDS, BIND_ROUNDS = 6, 2
SEEDS = (42, 7, 123, 2024, 5, 99)
LEX = "LEX"
ROLES = ("ROLE_SUBJ", "ROLE_VERB", "ROLE_OBJ")
VOCAB = (16, 32, 64)


def build_ff(brain, stim, area, rounds):
    """Feed-forward lexicon build. See lexicon_capacity_law.py."""
    for _ in range(rounds):
        brain.project({stim: [area]}, {})
    return read(brain, area)


def sentences(m_words):
    """Sentences whose word-role assignments cross, so `swap` is measurable.

    Sentence i is (i, i+1, i+2) mod M. Every word therefore appears once in
    each of the three positions across the set, which makes role-from-word
    impossible to fake: the same word must read as SUBJ, VERB and OBJ in
    different sentences.
    """
    return [tuple((i + d) % m_words for d in range(3)) for i in range(m_words)]


def make_state(gated):
    """Inhibition state over LEX + the three role areas.

    Everything starts closed (the `InhibitionState` default, which is the
    correct one -- a fiber nobody opened must not carry signal). The ungated
    arm opens every fiber ONCE and never touches them again, which is what
    naming all targets in Python amounts to.
    """
    from neural_assemblies.core.inhibition import InhibitionState

    areas = [LEX, *ROLES]
    st = InhibitionState(areas, initial_areas=areas)
    if not gated:
        for r in ROLES:
            st.disinhibit_fiber(LEX, r)
    return st


def bind_sentence(brain, state, sent, stim_of, gated):
    """Present three words in order. The rules never look at the words."""
    for pos, word in enumerate(sent):
        if gated:
            # Data-independent: position -> role. Open exactly one fiber.
            for r in ROLES:
                state.inhibit_fiber(LEX, r)
            state.disinhibit_fiber(LEX, ROLES[pos])
        proj = state.project_map(brain, lex_area=LEX)
        # The reference's guard: LEX must never reach two role slots at once.
        if gated:
            state.check_war_of_fibers(proj, LEX)
        stim = {stim_of[word]: [LEX]}
        for _ in range(BIND_ROUNDS):
            brain.project(stim, proj)


def trial(m_words, gated, seed):
    from neural_assemblies.core.brain import Brain

    brain = Brain(p=P, seed=seed)
    for area in (LEX, *ROLES):
        brain.add_area(area, N, K, beta=BETA)
    stim_of = {w: f"ph{w}" for w in range(m_words)}
    for w in range(m_words):
        brain.add_stimulus(stim_of[w], K)
    for w in range(m_words):
        build_ff(brain, stim_of[w], LEX, BUILD_ROUNDS)

    # Reference: what word w looks like IN role area r.
    #
    # Built OUTSIDE `probe`, deliberately. An earlier version built it inside,
    # and every cell read EXACTLY chance with a margin of EXACTLY 1.00 --
    # the signature of a dead probe, not of a negative result. `read_only()`
    # blocks recruitment (that is its job), and the role areas had never been
    # materialised, so LEX->ROLE was never written and every read returned the
    # same thing. This pass IS the training of those fibers, the same way the
    # depth experiments merge for real and only then read under a probe.
    ref = {r: {} for r in ROLES}
    for r in ROLES:
        for w in range(m_words):
            build_ff(brain, stim_of[w], LEX, BUILD_ROUNDS)
            for _ in range(BIND_ROUNDS):
                brain.project({}, {LEX: [r]})
            ref[r][w] = read(brain, r)

    state = make_state(gated)
    sents = sentences(m_words)
    hits = [0, 0, 0]
    margins = []
    seen = {}          # word -> set of positions it was read correctly in
    for sent in sents:
        with probe(brain):
            bind_sentence(brain, state, sent, stim_of, gated)
            for pos, r in enumerate(ROLES):
                live = read(brain, r)
                sims = sorted(((similarity(live, a), w)
                               for w, a in ref[r].items()), reverse=True)
                ok = sims[0][1] == sent[pos]
                hits[pos] += ok
                if ok:
                    seen.setdefault(sent[pos], set()).add(pos)
                if len(sims) > 1 and sims[1][0] > 0:
                    margins.append(sims[0][0] / sims[1][0])

    # R3: words read correctly in MORE THAN ONE distinct position. Role cannot
    # be a property of the word if the same word reads as two different roles.
    multi = sum(1 for v in seen.values() if len(v) > 1)
    return (hits, len(sents), statistics.mean(margins) if margins else float("nan"),
            multi, len(seen))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print(f"\n  n={N} k={K} beta={BETA}, feed-forward lexicon, SHARED role "
          f"areas, bindT={BIND_ROUNDS}")
    print(f"  rules are POSITION -> ROLE and never consult the word; the "
          f"projection map is derived by")
    print(f"  InhibitionState.project_map, so the state machine is in the "
          f"loop rather than described.")
    print(f"  ungated = every LEX->ROLE fiber open at once, i.e. what naming "
          f"targets in Python does.\n")
    print(f"  {'M':>5}{'arm':>9}{'chance':>9}{'subj':>8}{'verb':>8}{'obj':>8}"
          f"{'all3':>8}{'margin':>8}{'multi':>8}{'trials':>8}")

    rows = {}
    for m_words in VOCAB:
        for gated in (True, False):
            res = [trial(m_words, gated, s) for s in SEEDS]
            tot = sum(x[1] for x in res)
            per = [sum(x[0][i] for x in res) / tot for i in range(3)]
            mar = statistics.mean(x[2] for x in res)
            multi = statistics.mean(x[3] / max(x[4], 1) for x in res)
            rows[(m_words, gated)] = (per, mar, multi, tot)
            flag = "" if tot >= MIN_TRIALS else "  [UNDER-POWERED]"
            print(f"  {m_words:>5}{'gated' if gated else 'ungated':>9}"
                  f"{1 / m_words:>9.4f}{per[0]:>8.4f}{per[1]:>8.4f}"
                  f"{per[2]:>8.4f}{statistics.mean(per):>8.4f}{mar:>8.2f}"
                  f"{multi:>8.2f}{tot:>8}{flag}")
        print()

    print("  READING")
    g16 = rows[(16, True)]
    u16 = rows[(16, False)]
    r1 = min(g16[0]) > (1 / 16) + 0.20
    print(f"    R1 gated binds all three positions:  {r1}   "
          f"subj/verb/obj {g16[0][0]:.4f}/{g16[0][1]:.4f}/{g16[0][2]:.4f}")
    r2 = statistics.mean(u16[0]) < statistics.mean(g16[0]) - 0.20
    print(f"    R2 ungated collapses:                {r2}   "
          f"all3 {statistics.mean(u16[0]):.4f} vs gated "
          f"{statistics.mean(g16[0]):.4f}")
    r3 = g16[2] > 0.5
    print(f"    R3 same word reads in >1 role:       {r3}   "
          f"{g16[2]:.2f} of words (role is from the GATE, not the word)")
    accs = [statistics.mean(rows[(m, True)][0]) for m in VOCAB]
    r4 = min(accs) > 0.90
    print(f"    R4 stable as vocabulary grows:       {r4}   " +
          "  ".join(f"M={m}:{a:.4f}" for m, a in zip(VOCAB, accs)) +
          f"   margin {rows[(VOCAB[-1], True)][1]:.2f}x")

    print()
    if r1 and r2 and r3:
        print(f"    GATING IS LOAD-BEARING FOR ROLE BINDING. Exclusivity holds")
        print(f"    because only one fiber was ever open -- no cross-area")
        print(f"    comparison, no winner-take-all after firing, and no rule")
        print(f"    that consults the word. The same word reads as a different")
        print(f"    role in a different position, so this is binding rather")
        print(f"    than lookup. Bears directly on tasks #24 and #33.")
    elif r1 and not r2:
        print(f"    Gating WORKS but so does the ungated arm, so gating buys")
        print(f"    nothing measurable here and the symbolic role route is")
        print(f"    adequate for this task. That settles #33 the other way and")
        print(f"    is worth the file -- but check `multi` before concluding:")
        print(f"    an ungated arm that scores well WITHOUT swapping roles is")
        print(f"    doing lookup, and the two should not be equated.")
    else:
        print(f"    Gated binding does not clear chance, so nothing here is")
        print(f"    interpretable yet. Check the per-position columns: if only")
        print(f"    the LAST position works, the fibers are not actually being")
        print(f"    closed between steps and the state machine is not driving.")


if __name__ == "__main__":
    main()

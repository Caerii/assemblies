"""Overnight envelope mapping: the axes this project has never varied.

WHY THIS SWEEP
--------------
Every result in the 2026-07-28/29 arc was measured at p=0.05, beta=0.1, k=50.
Three of the four parameters that define the substrate were held fixed while
conclusions were drawn about capacity, composition and depth. Those conclusions
may be laws or they may be facts about one corner of the space, and nothing so
far distinguishes the two.

This maps the envelope. It is deliberately broad rather than deep: each study
answers "where does this stop working", and the point is coverage of axes
nobody has moved, not another decimal place on a number already measured.

FIVE STUDIES, cheapest first so a short night still yields the useful part:

  A  LEXICON     capacity vs (n, k, beta). The base of the stack. Feed-forward
                 had no measured ceiling at k=50 -- does that survive changing
                 the sparsity k/n and the learning rate?
  B  SPARSITY    capacity vs p, the connection probability. Never varied at
                 all. p sets how much evidence each neuron sees, so it should
                 set the floor on how many items can be told apart.
  C  MERGE       composition capacity vs (n, beta, T). Whether the T=2-3
                 operating point is a property of beta=0.1 or of the operation.
  D  BINDING     role binding vs (vocabulary, sentences). Gating gave 1.0000
                 at 64 words; the question is what it costs as the corpus grows.
  E  BETA        the training-window walls vs beta, directly. The (1+beta)^T law
                 predicts the walls MOVE with beta, which has been asserted from
                 one beta and never checked.

METHOD
------
Capacity is reported as the largest M whose rank-1 identity clears 0.90, found
by doubling. Margin at that M is reported alongside, because a capacity number
without a margin cannot distinguish "comfortably fine" from "one step from
failing" -- measured, accuracy read 0.9544 at a margin of 1.47x.

Every cell runs through `neural_assemblies.diagnostics`, so a collapsed or
dead-probe cell is FLAGGED rather than silently reported as a low number. That
is the difference between this and the sweeps that produced six retractions.

Results are flushed per row. A cell that raises is recorded and the sweep
continues.
"""

from __future__ import annotations

import os
import statistics
import sys
import traceback

os.environ.setdefault("TRAIN_PROGRESS", "0")

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from neural_assemblies.core.measurement import defined_values  # noqa: E402
from neural_assemblies.diagnostics import area_health, read_assembly  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "overnight_characterization_results.txt")
SEEDS = (42, 7, 123)
CAP_LADDER = (16, 32, 64, 128, 256, 512, 1024)


def emit(line, fh):
    print(line, flush=True)
    fh.write(line + "\n")
    fh.flush()
    os.fsync(fh.fileno())


# --------------------------------------------------------------------------
# Primitives
# --------------------------------------------------------------------------

def lexicon_cell(n, k, p, beta, m_words, rounds, seed):
    """Feed-forward lexicon of *m_words*; returns an AreaHealth."""
    from neural_assemblies.core.brain import Brain

    brain = Brain(p=p, seed=seed)
    brain.add_area("L", n, k, beta=beta)
    for m in range(m_words):
        brain.add_stimulus(f"w{m}", k)

    def drive(m):
        for _ in range(rounds):
            brain.project({f"w{m}": ["L"]}, {})

    stored = {}
    for m in range(m_words):
        drive(m)
        stored[m] = read_assembly(brain, "L")
    cues = {m: (lambda mm=m: drive(mm)) for m in range(m_words)}
    return area_health(brain, "L", stored, cues)


def merge_cell(n, k, p, beta, m_items, rounds, seed):
    from neural_assemblies.assembly_calculus.ops import merge
    from neural_assemblies.core.brain import Brain

    brain = Brain(p=p, seed=seed)
    for a in ("A", "B", "C"):
        brain.add_area(a, n, k, beta=beta)
    for m in range(m_items):
        brain.add_stimulus(f"a{m}", k)
        brain.add_stimulus(f"b{m}", k)

    def parent(stim, area):
        for _ in range(6):
            brain.project({stim: [area]}, {})

    for m in range(m_items):
        parent(f"a{m}", "A")
        parent(f"b{m}", "B")
    stored = {}
    for m in range(m_items):
        merge(brain, "A", "B", "C", stim_a=f"a{m}", stim_b=f"b{m}",
              rounds=rounds, parent_self=False, target_self=False,
              back_project=False)
        stored[m] = read_assembly(brain, "C")

    def cue(m):
        parent(f"a{m}", "A")
        brain.project({}, {"A": ["C"]})

    cues = {m: (lambda mm=m: cue(mm)) for m in range(m_items)}
    return area_health(brain, "C", stored, cues)


def capacity(fn, **kw):
    """Largest ladder M clearing 0.90, plus its margin and any flags."""
    best, best_margin, flags = 0, float("nan"), []
    for m in CAP_LADDER:
        try:
            hs = [fn(m_words=m, seed=s, **kw) if "m_words" in
                  fn.__code__.co_varnames else fn(m_items=m, seed=s, **kw)
                  for s in SEEDS]
        except Exception as exc:
            flags.append(f"{type(exc).__name__}@M={m}")
            break
        # `h.accuracy` and `h.margin` are `Measured`, not floats. The old code
        # averaged accuracy unguarded and hand-filtered margin with the NaN
        # idiom `h.margin == h.margin` on the very next line -- and that filter
        # dropped exactly the seeds where separation was PERFECT, so the
        # reported margin was a mean over the worse half. `defined_values`
        # drops only genuinely-unmeasurable readings, and says how many.
        accs = defined_values([h.accuracy for h in hs])
        mars = defined_values([h.margin for h in hs])
        if len(accs) < len(hs):
            flags.append(f"ACC_UNDEFINED@M={m}({len(hs) - len(accs)}/{len(hs)})")
            break
        acc = statistics.mean(accs)
        mar = statistics.mean(mars) if mars else float("nan")
        if len(mars) < len(hs):
            flags.append(f"margin over {len(mars)}/{len(hs)} seeds@M={m}")
        if any(not h.trustworthy for h in hs):
            flags.append(f"UNTRUSTWORTHY@M={m}")
            break
        if acc > 0.90:
            best, best_margin = m, mar
        else:
            break
    censored = best == CAP_LADDER[-1]
    return best, best_margin, censored, flags


# --------------------------------------------------------------------------
# Studies
# --------------------------------------------------------------------------

def study_lexicon(fh):
    emit("\n  A. LEXICON capacity vs (n, k, beta), feed-forward, p=0.05", fh)
    emit(f"  {'n':>7}{'k':>5}{'beta':>7}{'k/n':>8}{'M_max':>8}"
         f"{'margin':>9}{'note':>12}", fh)
    for n in (1000, 4000):
        for k in (25, 50, 100):
            for beta in (0.05, 0.10, 0.20):
                m, mar, cens, flags = capacity(
                    lexicon_cell, n=n, k=k, p=0.05, beta=beta, rounds=6)
                note = ("CENSORED" if cens else "") + " ".join(flags)
                emit(f"  {n:>7}{k:>5}{beta:>7.2f}{k / n:>8.4f}{m:>8}"
                     f"{mar:>9.2f}{note:>12}", fh)


def study_sparsity(fh):
    emit("\n  B. LEXICON capacity vs p (connection probability), never varied "
         "before", fh)
    emit(f"  {'n':>7}{'k':>5}{'p':>7}{'M_max':>8}{'margin':>9}{'note':>12}", fh)
    for n in (1000, 4000):
        for p in (0.01, 0.02, 0.05, 0.10, 0.20):
            m, mar, cens, flags = capacity(
                lexicon_cell, n=n, k=50, p=p, beta=0.10, rounds=6)
            note = ("CENSORED" if cens else "") + " ".join(flags)
            emit(f"  {n:>7}{50:>5}{p:>7.2f}{m:>8}{mar:>9.2f}{note:>12}", fh)


def study_merge(fh):
    emit("\n  C. MERGE capacity vs (n, beta, T), all channels gated", fh)
    emit(f"  {'n':>7}{'beta':>7}{'T':>4}{'M_max':>8}{'margin':>9}"
         f"{'note':>12}", fh)
    for n in (1000, 4000):
        for beta in (0.05, 0.10, 0.20):
            for t in (2, 3, 5):
                m, mar, cens, flags = capacity(
                    merge_cell, n=n, k=50, p=0.05, beta=beta, rounds=t)
                note = ("CENSORED" if cens else "") + " ".join(flags)
                emit(f"  {n:>7}{beta:>7.2f}{t:>4}{m:>8}{mar:>9.2f}"
                     f"{note:>12}", fh)


def study_beta_window(fh):
    """E. The (1+beta)^T law says the walls MOVE with beta. Check directly."""
    emit("\n  E. Training window vs beta: rounds needed for an assembly to "
         "survive its own recurrence", fh)
    emit(f"  {'n':>7}{'beta':>7}{'T':>4}{'(1+b)^T':>10}{'self_overlap':>14}", fh)
    from neural_assemblies.assembly_calculus.ops import project
    from neural_assemblies.core.brain import Brain
    from neural_assemblies.diagnostics import assembly_overlap
    for n in (2000, 10000):
        for beta in (0.05, 0.10, 0.20):
            for t in (4, 6, 10, 16):
                vals = []
                for s in SEEDS:
                    b = Brain(p=0.05, seed=s)
                    b.add_area("W", n, 50, beta=beta)
                    b.add_stimulus("w", 50)
                    project(b, "w", "W", rounds=t, recurrent=True)
                    trained = read_assembly(b, "W")
                    for _ in range(8):
                        b.project({}, {"W": ["W"]})
                    vals.append(assembly_overlap(read_assembly(b, "W"), trained))
                emit(f"  {n:>7}{beta:>7.2f}{t:>4}{(1 + beta) ** t:>10.2f}"
                     f"{statistics.mean(vals):>14.4f}", fh)


STUDIES = [("A lexicon", study_lexicon), ("B sparsity", study_sparsity),
           ("C merge", study_merge), ("E beta window", study_beta_window)]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    with open(OUT, "w", encoding="utf-8") as fh:
        emit("  OVERNIGHT CHARACTERIZATION -- axes never varied before", fh)
        emit(f"  capacity = largest M with rank-1 > 0.90, ladder "
             f"{CAP_LADDER}, {len(SEEDS)} seeds", fh)
        emit("  CENSORED = the ladder ran out before the substrate did, so "
             "M_max is a LOWER BOUND", fh)
        emit("  every cell passes through diagnostics; UNTRUSTWORTHY means the "
             "measurement failed, not the substrate", fh)
        for name, fn in STUDIES:
            try:
                fn(fh)
            except Exception as exc:
                emit(f"\n  {name} FAILED: {type(exc).__name__}: {exc}", fh)
                traceback.print_exc()
        emit("\n  DONE.", fh)


if __name__ == "__main__":
    main()

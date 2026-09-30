"""Which wired pathways actually carry drive? (task #50)

An area->area weight block is materialised lazily, so a pathway that has never
successfully carried drive has shape (0, 0) and delivers exactly zero -- while
the projection naming it still returns k winners and looks like it worked. Two
instances are on record:

  * a role area fed by LEX_NOUN then LEX_VERB used only the FIRST; every item
    arriving through the second "stored" the target's stale assembly, and
    swapping the training order swapped which category died.
  * `reset_area_connections` zeroing a connectome so k-WTA fell through to its
    index tie-break and returned identical winners for every input.

WHY THIS IS AN AUDIT AND NOT HOUSEKEEPING. The signature of a dead pathway in
results is "mechanism X turns out to have surprisingly little effect", which is
indistinguishable by inspection from a real negative result. Several standing
conclusions in this repo have exactly that shape and were reached BEFORE the
bug was known:

  * multi-mood word order -- SYN receives from ROLE *and* MOOD, two sources
    introduced at different times, which is the precise precondition. The
    recorded finding is that MOOD controls ~4% of SYN's drive and that five
    interventions failed to move it.
  * mutual inhibition never firing (#24)
  * function-word bootstrapping not clearing chance (#29)
  * `associate()` marginal at ~3.8x chance (#39)

This does not assert any of those is wrong. It converts "could a dead pathway
explain this?" from an argument into a measurement, per construction, which is
cheap and has to be done before the conclusions are built on further.

`driven=` is the load-bearing argument: it names the pathways the caller
BELIEVES it uses, so a dead one becomes a failed verdict rather than one row in
a long listing.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("TRAIN_PROGRESS", "0")
os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from neural_assemblies.diagnostics import (                  # noqa: E402
    FiberState, fiber_census, format_report,
)

SEED = int(os.environ.get("FA_SEED", "42"))


def _report(tag, brain, driven=None):
    rows = fiber_census(brain, driven=driven)
    fibers = [r for r in rows if isinstance(r, FiberState)]
    verdicts = [r for r in rows if not isinstance(r, FiberState)]
    ignored = [f for f in fibers if f.silently_ignored]
    live = [f for f in fibers if not f.dead]
    print(f"\n{'=' * 74}\n  {tag}")
    print(f"  {len(fibers)} pathways: {len(live)} carrying weight, "
          f"{len(fibers) - len(live)} empty, "
          f"{len(ignored)} EMPTY INTO A LIVE AREA")
    if live:
        print(format_report(sorted(live, key=lambda f: (f.dst, f.src))))
    if ignored:
        print("\n  -- empty pathways into areas that have materialised "
              "neurons --")
        print(format_report(sorted(ignored, key=lambda f: (f.dst, f.src))))
    if verdicts:
        print("\n  -- pathways this model BELIEVES it drives --")
        print(format_report(verdicts))
    else:
        if driven:
            print(f"\n  all {sum(len(v) for v in driven.values())} claimed "
                  f"pathways carry weight")
    return fibers, verdicts


def _live_set(brain):
    return {(f.src, f.dst) for f in fiber_census(brain)
            if isinstance(f, FiberState) and not f.dead}


def audit_nemo_parser():
    """NemoParser: LEX_NOUN/LEX_VERB -> three ROLE areas, plus SEQ.

    SNAPSHOTTED PER TRAINING PHASE, because a single end-of-training census
    cannot answer the question. `reset_area_connections` sets a block to
    `empty((0, 0))`, which is byte-identical to never-materialised, so a
    final-state reading of "dead" conflates:

        NEVER USED  -- the projection never ran, or ran with no drive
        RESET       -- it ran, learned, and was then zeroed

    Those have opposite implications and the difference is only visible as a
    transition between phases. Comparing consecutive snapshots separates them.
    """
    from neural_assemblies.assembly_calculus.parser import NemoParser
    from neural_assemblies.core.brain import Brain
    import corpus

    nouns, verbs = corpus.build(12, 6)
    sents = [list(s) for s in
             corpus.occurrences(nouns, verbs, n_sentences=40, seed=0)]
    brain = Brain(p=0.2, save_winners=True, seed=SEED, engine="numpy_sparse")
    parser = NemoParser(brain, n=1000, k=50, beta=0.0718, rounds=10)
    parser.setup_areas()
    for w in nouns:
        parser.register_word(w, "noun", f"vis_{w}")
    for w in verbs:
        parser.register_word(w, "verb", f"mot_{w}")

    # INSTRUMENT THE RESET ITSELF. Phase boundaries are too coarse: the reset
    # fires after EVERY word, so a role fiber is only ever alive mid-word and a
    # snapshot between phases always reads zero. Recording liveness in the
    # instant BEFORE each reset is what distinguishes "never learned" from
    # "learned and then destroyed" -- and only the second is a reset problem.
    resets = []
    _orig = brain._engine.reset_area_connections

    def _spy(area, *a, **k):
        if isinstance(area, str):
            live = {(s, d) for (s, d) in _live_set(brain) if d == area}
            resets.append((area, sorted(live)))
        return _orig(area, *a, **k)

    brain._engine.reset_area_connections = _spy

    phases = []
    for tag, fn in (("setup_areas", lambda: None),
                    ("train_lexicon", parser.train_lexicon),
                    ("train_roles", lambda: parser.train_roles(sents)),
                    ("train_word_order",
                     lambda: parser.train_word_order(sents))):
        fn()
        phases.append((tag, _live_set(brain)))
    brain._engine.reset_area_connections = _orig

    destroyed = [(a, live) for a, live in resets if live]
    print(f"\n{'=' * 74}\n  reset_area_connections fired {len(resets)} times; "
          f"{len(destroyed)} of those zeroed a LIVE pathway")
    seen = set()
    for area, live in destroyed:
        for pair in live:
            if pair not in seen:
                seen.add(pair)
                print(f"      {pair[0]} -> {pair[1]}  was carrying weight when "
                      f"reset({area}) zeroed it")

    print(f"\n{'=' * 74}\n  NemoParser -- live area->area pathways per phase")
    prev = set()
    for tag, live in phases:
        gained, lost = sorted(live - prev), sorted(prev - live)
        print(f"\n  after {tag:<18} {len(live)} live")
        for s, d in gained:
            print(f"      + {s} -> {d}")
        for s, d in lost:
            print(f"      - {s} -> {d}    LEARNED THEN ZEROED (reset), not "
                  f"unused")
        prev = live

    # What train_roles drives: subject/object are nouns, verb is a verb, so
    # each role area is fed by exactly ONE lexical area. That is why the
    # per-area capacity sweep was unaffected by the deferred-init bug --
    # asserted here rather than assumed.
    driven = {"LEX_NOUN": ["ROLE_AGENT", "ROLE_PATIENT"],
              "LEX_VERB": ["ROLE_ACTION"]}
    return _report("NemoParser, FINAL state (what parse() would use)",
                   brain, driven)


def audit_emergent_parser():
    """EmergentParser: the helper-area architecture, where the risk is real."""
    from neural_assemblies.assembly_calculus.emergent.evaluation.sweep import (
        get_parser_cache,
    )
    cache = get_parser_cache()
    parser = cache.get("SENTENCES", seed=SEED, calibrate=False)
    brain = getattr(parser, "brain", None)
    if brain is None:
        print("\n  (EmergentParser exposes no .brain -- skipped)")
        return [], []
    return _report("EmergentParser SENTENCES (helper areas)", brain)


def audit_gated_parser():
    """ParserBrain: the fiber-gated TACL parser. Gating decides which fire."""
    try:
        from neural_assemblies.language.parser import ParserBrain
    except Exception as exc:                                 # noqa: BLE001
        print(f"\n  (ParserBrain unavailable: {exc})")
        return [], []
    try:
        pb = ParserBrain(p=0.1)
        brain = getattr(pb, "brain", pb)
        return _report("ParserBrain (fiber-gated TACL)", brain)
    except Exception as exc:                                 # noqa: BLE001
        print(f"\n  (ParserBrain needs constructor args: {exc})")
        return [], []


if __name__ == "__main__":
    which = os.environ.get("FA_WHICH", "nemo,emergent,gated").split(",")
    total_ignored, total_bad = 0, 0
    for name, fn in (("nemo", audit_nemo_parser),
                     ("emergent", audit_emergent_parser),
                     ("gated", audit_gated_parser)):
        if name not in which:
            continue
        try:
            fibers, verdicts = fn()
        except Exception as exc:                             # noqa: BLE001
            import traceback
            print(f"\n  {name}: AUDIT FAILED -- {type(exc).__name__}: {exc}")
            traceback.print_exc()
            continue
        total_ignored += sum(1 for f in fibers if f.silently_ignored)
        total_bad += len(verdicts)
    print(f"\n{'=' * 74}")
    print(f"  TOTAL: {total_ignored} empty pathways into live areas, "
          f"{total_bad} claimed-but-dead pathways")
    if total_bad:
        print("  A claimed-but-dead pathway means a projection in that model "
              "delivers no\n  drive at all. Any result attributing a small "
              "effect to that source is void.")

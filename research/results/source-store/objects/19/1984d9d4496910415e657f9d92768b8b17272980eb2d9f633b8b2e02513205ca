"""Are the parser's LEX assemblies reachable from PHON alone? (#46)

FOUND WHILE FIXING THE BRIDGE READOUT, and it is a question about shipped code
rather than about the experiment. `train_lexicon` builds each word's LEX
assembly by driving PHON *and* the grounding stimulus together, with
self-recurrence, on a freshly zeroed connectome. Anything that later wants to
re-activate that word has only PHON -- grounding is a learning signal, not
something available at parse time -- and `classify_word` duly projects PHON
alone.

Measured overlap between the stored assembly and the one PHON alone recovers
was 0.020 at n=1000, k=50, i.e. BELOW the chance floor k/n = 0.05. If that is
right it is not a small discrepancy: it says the lexical layer cannot re-enter
its own learned representations from the only cue it will ever have.

WHY THAT MIGHT STILL BE HARMLESS, and why this has to be measured rather than
argued: `classify_word` never needs the assembly itself. It compares the best
overlap against the noun lexicon with the best against the verb lexicon, so any
distortion shared by both sides cancels, and a systematic signal far below the
floor could still order the two correctly. Category classification working is
therefore fully compatible with assembly reproduction failing.

So the two things are measured separately:

  1. `fidelity`  -- overlap of PHON-only recovery with the stored assembly, per
     word, against the floor k/n.
  2. `classify`  -- the accuracy `classify_word` actually delivers.

A gap between them localises the problem precisely: category information
survives in the lexical layer while token identity does not.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("TRAIN_PROGRESS", "0")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

import corpus                                                # noqa: E402

N = int(os.environ.get("LR_N", "1000"))
K = int(os.environ.get("LR_K", "50"))
P = float(os.environ.get("LR_P", "0.2"))
ROUNDS = int(os.environ.get("LR_ROUNDS", "10"))
BETA = float(os.environ.get("LR_BETA", "0.07177"))
SEEDS = [int(x) for x in os.environ.get("LR_SEEDS", "42,43,44").split(",")]


def run(seed, nouns, verbs):
    from _substrate import probe
    from neural_assemblies.assembly_calculus.ops import _snap, project
    from neural_assemblies.assembly_calculus.parser import NemoParser
    from neural_assemblies.core.brain import Brain

    brain = Brain(p=P, save_winners=True, seed=seed, engine="numpy_sparse")
    parser = NemoParser(brain, n=N, k=K, beta=BETA, rounds=ROUNDS)
    parser.setup_areas()
    for w in nouns:
        parser.register_word(w, "noun", f"vis_{w}")
    for w in verbs:
        parser.register_word(w, "verb", f"mot_{w}")
    parser.train_lexicon()

    fid, fid_grounded = {}, {}
    for lex_area, lexicon in (("LEX_NOUN", parser.noun_lexicon),
                              ("LEX_VERB", parser.verb_lexicon)):
        for word, asm in lexicon.items():
            stored = set(int(x) for x in asm.winners)
            # (a) PHON only, the cue actually available at parse time.
            with probe(brain):
                brain._engine.reset_area_connections(lex_area)
                got = project(brain, parser.stim_map[word], lex_area,
                              rounds=ROUNDS)
            fid[word] = len(stored & set(int(x) for x in got.winners)) / len(stored)
            # (b) PHON + grounding + recurrence -- train_lexicon's own
            # protocol, as an upper bound. If even this fails to reproduce the
            # assembly the problem is the probe or the reset, not the cue.
            with probe(brain):
                brain._engine.reset_area_connections(lex_area)
                for _ in range(ROUNDS):
                    brain.project(
                        {parser.stim_map[word]: [lex_area],
                         parser.grounding_map[word]: [lex_area]},
                        {lex_area: [lex_area]})
                got2 = _snap(brain, lex_area)
            fid_grounded[word] = len(
                stored & set(int(x) for x in got2.winners)) / len(stored)

    correct = sum(parser.classify_word(w) == parser.word_categories[w]
                  for w in list(nouns) + list(verbs))

    # THE THIRD ARM, and it is the one that localises the bridge's precondition
    # failure. Re-measure PHON-only fidelity AFTER role training of the kind
    # `train_roles` does -- project PHON into LEX, with plasticity, WITHOUT
    # resetting between words. If fidelity is intact before and destroyed
    # after, the corruption is caused by role training rather than by the cue,
    # the probe, or the reset.
    from neural_assemblies.assembly_calculus.parser import ROLE_AREAS
    for i, word in enumerate(list(nouns) + list(verbs)):
        lex_area = ("LEX_NOUN" if parser.word_categories[word] == "noun"
                    else "LEX_VERB")
        role_area = ROLE_AREAS[i % 3]
        project(brain, parser.stim_map[word], lex_area, rounds=ROUNDS)
        brain.areas[lex_area].fix_assembly()
        for _ in range(ROUNDS):
            brain.project({}, {lex_area: [role_area]})
        brain.areas[lex_area].unfix_assembly()

    fid_after = {}
    for lex_area, lexicon in (("LEX_NOUN", parser.noun_lexicon),
                              ("LEX_VERB", parser.verb_lexicon)):
        for word, asm in lexicon.items():
            stored = set(int(x) for x in asm.winners)
            with probe(brain):
                brain._engine.reset_area_connections(lex_area)
                got = project(brain, parser.stim_map[word], lex_area,
                              rounds=ROUNDS)
            fid_after[word] = len(
                stored & set(int(x) for x in got.winners)) / len(stored)

    return fid, fid_grounded, correct / (len(nouns) + len(verbs)), fid_after


def main():
    nouns, verbs = corpus.build(int(os.environ.get("LR_NOUNS", "40")),
                                int(os.environ.get("LR_VERBS", "20")))
    floor = K / N
    print(f"\n  LEX REPRODUCIBILITY   n={N} k={K} p={P} rounds={ROUNDS} "
          f"beta={BETA}")
    print(f"  {len(nouns)} nouns, {len(verbs)} verbs, chance floor k/n "
          f"= {floor:.4f}")
    print(f"\n  {'seed':>5} {'PHON-only':>11} {'PHON+ground':>12} "
          f"{'classify':>9} {'PHON after roles':>18}")
    all_phon, all_gr, accs, all_after = [], [], [], []
    for seed in SEEDS:
        fid, fidg, acc, fid_after = run(seed, nouns, verbs)
        mp = sum(fid.values()) / len(fid)
        mg = sum(fidg.values()) / len(fidg)
        ma = sum(fid_after.values()) / len(fid_after)
        all_phon += list(fid.values())
        all_gr += list(fidg.values())
        all_after += list(fid_after.values())
        accs.append(acc)
        print(f"  {seed:>5} {mp:>11.4f} {mg:>12.4f} {acc:>9.4f} "
              f"{ma:>18.4f}")
    mp = sum(all_phon) / len(all_phon)
    mg = sum(all_gr) / len(all_gr)
    ma = sum(all_after) / len(all_after)
    print(f"\n  PHON only, clean LEX : {mp:.4f}   ({mp / floor:.2f}x the "
          f"{floor:.4f} floor)")
    print(f"  PHON+grounding       : {mg:.4f}   ({mg / floor:.2f}x floor)")
    print(f"  classify_word        : {sum(accs) / len(accs):.4f}  "
          f"(chance 0.5 for a two-way decision)")
    print(f"  PHON only, AFTER role training : {ma:.4f}   "
          f"({ma / floor:.2f}x floor)")
    print("\n  READING. High in the first column: the lexicon IS recoverable "
          "from the only\n  cue parse time has, so there is no defect in "
          "`train_lexicon` or in the probe.\n  A collapse in the last column "
          "isolates ROLE TRAINING as what destroys it --\n  it projects PHON "
          "into LEX with plasticity and never resets, so the LEX\n  pattern "
          "presented to the role areas drifts from word to word.")


if __name__ == "__main__":
    main()

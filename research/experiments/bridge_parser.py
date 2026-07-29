"""Does the phase map predict the PARSER? (task #46, the bridge experiment)

THE MAP WAS BUILT ON A SYNTHETIC TASK: uniform item frequency, uniform depth, a
chain of shared areas. Real grammar is skewed, reuses constituents, and is
hierarchical. So the map may simply be about a different system, and the cheapest
way to find that out is one experiment rather than a quarter of building on it.

THE TEST. Read the parser's REAL operating point per area -- n, k, p, the number
of assemblies it is asked to hold, and above all its cumulative gain -- then
place each area on the measured map and predict which ones collapse. Then measure
them and see.

WHY THE PREDICTION IS SHARP HERE. The parser uses rounds=10, so a single training
step applies (1+beta)^10 = 2.594 at beta=0.1, already above a crowding wall
measured at roughly 1.9-2.2. Worse, plasticity is cumulative and every training
step repeats it, so an item appearing c times in the corpus reaches

    g_eff = (1 + beta)^(rounds * c)

which for beta=0.1, rounds=10, c=4 is 1.1^40 = 45. That is not near a boundary,
it is an order of magnitude beyond one.

PREDICTION, registered before measuring:
  1. Areas driven repeatedly across sentences (role areas, SEQ) should show
     COLLAPSE -- low distinctness, assemblies merged.
  2. The lexicon should be comparatively FINE, because feed-forward selection
     cannot be reordered by uniform potentiation, which is the same reason
     frequency has no effect there.
  3. The prescription g = g_c^(1/(rounds*c_max)) implies beta of order 0.003 --
     roughly thirty times smaller than the 0.1 in use.

If (1) and (2) hold, the map transfers and the known parser failures are
instances of it, not separate bugs. Two are already on record and were diagnosed
independently: mood-specific chains merging after about twenty sentences, and 94
distinct verb phrases collapsing onto a single assembly.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("TRAIN_PROGRESS", "0")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

N = int(os.environ.get("BR_N", "1000"))
K = int(os.environ.get("BR_K", "50"))
P = float(os.environ.get("BR_P", "0.05"))
ROUNDS = int(os.environ.get("BR_ROUNDS", "10"))
BETAS = [float(x) for x in os.environ.get("BR_BETAS", "0.1").split(",")]
SEEDS = [int(x) for x in os.environ.get("BR_SEEDS", "42,43,44").split(",")]

NOUNS = ["dog", "cat", "bird"]
VERBS = ["chases", "sees", "catches"]
TRAIN = [["dog", "chases", "cat"], ["cat", "sees", "bird"],
         ["bird", "catches", "dog"], ["dog", "sees", "bird"]]


def build(beta, seed, no_reset=False):
    from neural_assemblies.assembly_calculus.parser import NemoParser
    from neural_assemblies.core.brain import Brain

    brain = Brain(p=P, save_winners=True, seed=seed, engine="numpy_sparse")
    if no_reset:
        # NEUTRALISE `reset_area_connections`, which train_roles calls after
        # EVERY word. It zeroes the role area's connectome, so k-WTA falls
        # through to its index tie-break and returns the same lowest-index
        # winners for every word -- one assembly, spread exactly 1.0000,
        # regardless of anything else.
        #
        # That is why the first run of this experiment looked like a confirmed
        # prediction and was not: the role areas were collapsed, as the map
        # predicts they should be, but the collapse did not budge across a 25x
        # change in beta (0.1 -> 0.004, spread 1.0000 and distinct 0.333 at
        # every step). A crowding effect cannot be gain-invariant. The
        # invariance is the tell, and it matches a failure mode already on
        # record for this call.
        #
        # So the map's prediction is UNTESTABLE on these areas until this is
        # out of the way -- a structural bug masks any gain effect entirely.
        brain._engine.reset_area_connections = lambda *a, **k: None
    parser = NemoParser(brain, n=N, k=K, beta=beta, rounds=ROUNDS)
    parser.setup_areas()
    for w in NOUNS:
        parser.register_word(w, "noun", f"vis_{w}")
    for w in VERBS:
        parser.register_word(w, "verb", f"mot_{w}")
    parser.train_lexicon()
    parser.train_roles(TRAIN)
    parser.train_word_order(TRAIN)
    return parser, brain


def word_counts():
    c = {}
    for s in TRAIN:
        for w in s:
            c[w] = c.get(w, 0) + 1
    return c


def area_report(brain, parser):
    """Per-area distinctness, using whatever assemblies the parser stored."""
    from neural_assemblies.diagnostics import area_health

    lexicons = {"LEX_NOUN": parser.noun_lexicon,
                "LEX_VERB": parser.verb_lexicon}
    lexicons.update({name: lex for name, lex in parser.role_lexicons.items()})

    out = {}
    for area, lex in lexicons.items():
        if not lex or area not in brain.areas:
            continue
        stored = {}
        for word, asm in lex.items():
            wn = getattr(asm, "winners", asm)
            stored[word] = list(wn)
        if len(stored) < 2:
            continue
        h = area_health(brain, area, stored)
        out[area] = h
    return out


if __name__ == "__main__":
    counts = word_counts()
    c_max = max(counts.values())
    print(f"\n  BRIDGE: parser vs phase map   n={N} k={K} p={P} "
          f"rounds={ROUNDS}")
    print(f"  corpus: {len(TRAIN)} sentences, counts={counts}, "
          f"c_max={c_max}, kp={K * P:g}")
    print(f"\n  {'beta':>6} {'g/step':>8} {'g_eff(c_max)':>13}  "
          f"prescription g_c^(1/(rounds*c_max))")
    for beta in BETAS:
        g_step = (1 + beta) ** ROUNDS
        g_eff = (1 + beta) ** (ROUNDS * c_max)
        print(f"  {beta:>6.4f} {g_step:>8.3f} {g_eff:>13.3g}")
    presc = 2.0 ** (1.0 / (ROUNDS * c_max)) - 1.0
    print(f"\n  prescribed beta for g_c=2.0: {presc:.5f}  "
          f"(vs 0.1 in use -> {0.1 / presc:.0f}x smaller)")

    # TWO FACTORS. Varying beta alone cannot separate "the map is right" from
    # "a structural bug pins these areas", because both produce collapse. The
    # reset arm is what makes the beta arm interpretable.
    for no_reset in (False, True):
        tag = "reset NEUTRALISED" if no_reset else "reset AS SHIPPED"
        for beta in BETAS:
            print(f"\n  === {tag}, beta={beta} "
                  f"(g/step={(1 + beta) ** ROUNDS:.3f}) ===")
            agg = {}
            for seed in SEEDS:
                parser, brain = build(beta, seed, no_reset=no_reset)
                for area, h in area_report(brain, parser).items():
                    agg.setdefault(area, []).append(h)
            if not agg:
                print("    (no area held >=2 assemblies -- nothing to score)")
                continue
            print(f"    {'area':<16} {'items':>6} {'distinct':>9} "
                  f"{'spread':>8} {'floor':>8}  verdict")
            for area, hs in sorted(agg.items()):
                mean = lambda f: sum(f(h) for h in hs) / len(hs)  # noqa: E731
                dfrac = mean(lambda h: h.distinct_frac)
                spr = mean(lambda h: h.spread)
                floor = hs[0].floor
                collapsed = sum(h.collapsed for h in hs)
                print(f"    {area:<16} {hs[0].n_items:>6} {dfrac:>9.3f} "
                      f"{spr:>8.4f} {floor:>8.4f}  "
                      f"{'COLLAPSED' if collapsed > len(hs) / 2 else 'ok'} "
                      f"({collapsed}/{len(hs)} seeds)")

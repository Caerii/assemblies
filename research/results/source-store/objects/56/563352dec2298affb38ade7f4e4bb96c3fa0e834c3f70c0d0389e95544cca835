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

import csv
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


def build(beta, seed, no_reset=False, ff_roles=False, dedup=False,
          nouns=None, verbs=None, train=None, roles_fn=None):
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
        # SUPPRESS IT FOR ROLE AREAS ONLY. A global suppression was wrong and
        # produced a misleading run: train_lexicon resets LEX before every
        # word, so a LEX assembly is only reproducible if the reader resets
        # too -- which is exactly what the parser's own classify_word does.
        # Removing the reset everywhere therefore broke LEX reproduction, so
        # the role area was driven by an input that never occurred in training,
        # and role retrieval read at chance in every arm for a reason that had
        # nothing to do with the role areas.
        #
        # The reset is genuinely LOAD-BEARING for the lexicon and harmful for
        # the roles, which is why the fix has to be per-call-site.
        _orig_reset = brain._engine.reset_area_connections

        def _selective_reset(area, *a, **k):
            if isinstance(area, str) and area.startswith("ROLE_"):
                return None
            return _orig_reset(area, *a, **k)

        brain._engine.reset_area_connections = _selective_reset
    nouns = NOUNS if nouns is None else nouns
    verbs = VERBS if verbs is None else verbs
    train = TRAIN if train is None else train
    parser = NemoParser(brain, n=N, k=K, beta=beta, rounds=ROUNDS)
    parser.setup_areas()
    for w in nouns:
        parser.register_word(w, "noun", f"vis_{w}")
    for w in verbs:
        parser.register_word(w, "verb", f"mot_{w}")
    parser.train_lexicon()
    if roles_fn is not None:
        # Caller supplies the whole role-training schedule. Used by
        # bridge_capacity.py, which needs the training and readout drives to be
        # the SAME code path -- see its docstring.
        roles_fn(parser)
    elif ff_roles:
        train_roles_ff(parser, train, dedup=dedup)
        parser.train_word_order(train)
    else:
        parser.train_roles(train)
        parser.train_word_order(train)
    return parser, brain


def word_counts():
    c = {}
    for s in TRAIN:
        for w in s:
            c[w] = c.get(w, 0) + 1
    return c


def train_roles_ff(parser, sentences, dedup=False):
    """train_roles WITHOUT the role-area self-recurrence.

    The shipped version drives {lex: [role], role: [role]}. Self-recurrence in
    a SHARED area during training is this project's documented collapse
    channel, and the recorded general fix is to build shared areas
    FEED-FORWARD rather than to reset after each item. The shipped code does
    the opposite on both counts: it keeps the recurrence AND resets.

    That combination explains what the reset arm alone could not. Removing the
    reset restores distinctness -- assemblies differ, overlap falls below the
    chance floor -- and role retrieval still reads at chance, because the
    recurrent fiber is shared across every word and pulls the area toward a
    common attractor whatever the lexical drive. Distinct storage and
    retrievable storage are not the same property.
    """
    from neural_assemblies.assembly_calculus.ops import project
    from neural_assemblies.assembly_calculus.parser import (
        ROLE_ACTION, ROLE_AGENT, ROLE_PATIENT, _snap,
    )

    seq = [ROLE_AGENT, ROLE_ACTION, ROLE_PATIENT]
    # DEDUP drives c_max to 1. Both walls must hold across the corpus frequency
    # range at once: the rarest item needs (1+b)^(rounds*c_min) >= `need` to be
    # re-selectable at all, while the most frequent needs
    # (1+b)^(rounds*c_max) <= g_c to avoid crowding. A single beta therefore
    # exists only if need^(c_max/c_min) <= g_c, and at every shipped operating
    # point it does not -- 12.53 against 2.24 at n=10000 k=100 p=0.05.
    # Training each (word, role) binding ONCE removes the spread, which is the
    # only change that makes a single beta possible at all.
    seen = set()
    for sentence in sentences:
        for word, role_area in zip(sentence, seq):
            if dedup:
                if (word, role_area) in seen:
                    continue
                seen.add((word, role_area))
            lex_area = ("LEX_NOUN"
                        if parser.word_categories[word] == "noun"
                        else "LEX_VERB")
            project(parser.brain, parser.stim_map[word], lex_area,
                    rounds=parser.rounds)
            parser.brain.areas[lex_area].fix_assembly()
            for _ in range(parser.rounds):
                parser.brain.project({}, {lex_area: [role_area]})
            parser.role_lexicons.setdefault(role_area, {})[word] = _snap(
                parser.brain, role_area)
            parser.brain.areas[lex_area].unfix_assembly()


def role_retrieval(brain, parser, recur=True):
    """TASK-LEVEL metric: can a role area say WHICH word it is holding?

    Distinctness says the stored assemblies differ; it does not say the area
    can be driven back to the right one. This drives each word's LEX assembly
    into the role area and checks that the resulting activity best matches that
    word's stored role assembly rather than another word's -- the direct
    analogue of retrieval accuracy in the synthetic sweeps.

    `parser.assign_role` cannot be used for this. Its own docstring says it is
    bookkeeping: it reports which lexicon contains the word and consults
    nothing about the brain's state, because LEX and ROLE have separate neuron
    populations and cross-area overlap is structurally ~0. So a role metric
    has to be measured WITHIN a role area, which is what this does.

    Readout runs inside `probe` so it cannot itself train the area -- reading
    with plasticity live would let the measurement create the structure it is
    trying to detect.
    """
    from _substrate import probe

    out = {}
    for role_area, lex in parser.role_lexicons.items():
        words = [w for w in lex if w in parser.word_categories]
        if len(words) < 2:
            continue
        stored = {w: list(getattr(lex[w], "winners", lex[w])) for w in words}
        hits = 0
        for w in words:
            lex_area = ("LEX_NOUN" if parser.word_categories[w] == "noun"
                        else "LEX_VERB")
            with probe(brain):
                from neural_assemblies.assembly_calculus.ops import project
                # Reset LEX first: train_lexicon built each word's assembly on
                # a zeroed LEX area driven by that word's own grounding
                # stimulus, so this is the only protocol that reproduces it.
                # classify_word does the same thing for the same reason.
                brain._engine.reset_area_connections(lex_area)
                project(brain, parser.stim_map[w], lex_area,
                        rounds=parser.rounds)
                brain.areas[lex_area].fix_assembly()
                # MATCH THE TRAINING PROTOCOL EXACTLY. train_roles drives
                # {lex_area: [role_area], role_area: [role_area]} -- with
                # self-recurrence. A readout that omits the recurrence is
                # running different dynamics from the ones that built the
                # assembly, and will read at chance no matter how healthy the
                # representation is. That is a defect in the measurement, not
                # in the parser, and it is worth being explicit about because
                # the first version of this function made exactly that mistake.
                for _ in range(parser.rounds):
                    tgt = {lex_area: [role_area]}
                    if recur:
                        tgt[role_area] = [role_area]
                    brain.project({}, tgt)
                # THE INDEX SPACES MUST MATCH, and this line is why every arm
                # of this experiment read at chance. `stored` holds NEURON IDs
                # (an `Assembly` snapshot); `brain.areas[x].winners` holds
                # COMPACT ENGINE INDICES. Intersecting the two compares
                # unrelated coordinate systems, so the overlap is whatever two
                # arbitrary integer sets share -- which is to say chance, no
                # matter how healthy the representation is, and invariant to
                # every parameter. `_snap` is the one-way door between the two
                # spaces and has to be used on both sides.
                #
                # Set BR_COMPACT_READOUT=1 to restore the bug as a negative
                # control; it should reproduce chance in every arm.
                if os.environ.get("BR_COMPACT_READOUT") == "1":
                    live = set(int(x)
                               for x in brain.areas[role_area].winners)
                else:
                    from neural_assemblies.assembly_calculus.ops import _snap
                    live = set(int(x) for x in _snap(brain, role_area).winners)
                brain.areas[lex_area].unfix_assembly()
            best, best_ov = None, -1.0
            for other, asm in stored.items():
                ov = len(live & set(int(x) for x in asm)) / max(len(asm), 1)
                if ov > best_ov:
                    best, best_ov = other, ov
            hits += (best == w)
        out[role_area] = hits / len(words)
    return out


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

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            os.environ.get("BR_OUT", "bridge_parser.csv"))
    new = not os.path.exists(out_path)
    fh = open(out_path, "a", newline="", encoding="utf-8")
    wcsv = csv.writer(fh)
    if new:
        wcsv.writerow(["cut", "reset", "n", "k", "p", "rounds", "beta",
                       "g_step", "g_eff", "area", "kind", "n_items", "seed",
                       "distinct_frac", "spread", "floor", "collapsed"])
        fh.flush()

    # TWO FACTORS. Varying beta alone cannot separate "the map is right" from
    # "a structural bug pins these areas", because both produce collapse. The
    # reset arm is what makes the beta arm interpretable.
    ARMS = [(False, False, False, "shipped (reset + recurrence)"),
            (True, True, False, "no reset, feed-forward, c_max=3"),
            (True, True, True, "no reset, feed-forward, DEDUP c_max=1")]
    for no_reset, ff_roles, dedup, tag in ARMS:
        for beta in BETAS:
            print(f"\n  === {tag}, beta={beta} "
                  f"(g/step={(1 + beta) ** ROUNDS:.3f}) ===")
            agg, task = {}, {}
            for seed in SEEDS:
                parser, brain = build(beta, seed, no_reset=no_reset,
                                      ff_roles=ff_roles, dedup=dedup)
                for ra, acc in role_retrieval(brain, parser,
                                              recur=not ff_roles).items():
                    task.setdefault(ra, []).append(acc)
                for area, h in area_report(brain, parser).items():
                    agg.setdefault(area, []).append(h)
                    wcsv.writerow([
                        f"bridge{'_ff' if ff_roles else ''}", int(no_reset), N, K, P, ROUNDS,
                        f"{beta:.5f}", f"{(1 + beta) ** ROUNDS:.5f}",
                        f"{(1 + beta) ** (ROUNDS * c_max):.5f}", area,
                        "lexicon" if area.startswith("LEX") else "role",
                        h.n_items, seed, f"{h.distinct_frac:.6f}",
                        f"{h.spread:.6f}", f"{h.floor:.6f}",
                        int(h.collapsed)])
            fh.flush()
            if not agg:
                print("    (no area held >=2 assemblies -- nothing to score)")
                continue
            print(f"    {'area':<16} {'items':>6} {'distinct':>9} "
                  f"{'spread':>8} {'floor':>8}  verdict")
            for area, hs in sorted(agg.items()):
                # These are `Measured`. Averaging over a mixture would be the
                # aggregate form of the ⊥-invention this module exists to
                # catch, so undefined readings are dropped and COUNTED -- a
                # mean over 2 of 9 seeds is a different claim from one over 9.
                from neural_assemblies.core.measurement import defined_values

                def mean(f, hs=hs):
                    vals = defined_values([f(h) for h in hs])
                    return (sum(vals) / len(vals) if vals else float("nan"),
                            len(hs) - len(vals))
                dfrac, dfrac_drop = mean(lambda h: h.distinct_frac)
                spr, spr_drop = mean(lambda h: h.spread)
                dropped = max(dfrac_drop, spr_drop)
                floor = hs[0].floor.or_else(float("nan"))
                collapsed = sum(h.collapsed for h in hs)
                print(f"    {area:<16} {hs[0].n_items:>6} {dfrac:>9.3f} "
                      f"{spr:>8.4f} {floor:>8.4f}  "
                      f"{'COLLAPSED' if collapsed > len(hs) / 2 else 'ok'} "
                      f"({collapsed}/{len(hs)} seeds)"
                      + (f"  [{dropped} undefined]" if dropped else ""))
            if task:
                # Chance is 1/n_items, so 0.333 for three words -- printed so a
                # metric sitting exactly at chance is recognisable as such
                # rather than read as a number.
                n_items = max(len(v) for v in [list(task.values())[0]]) and 3
                print(f"    {'ROLE RETRIEVAL':<16} "
                      + "  ".join(f"{ra.replace('ROLE_', '')} "
                                  f"{sum(v) / len(v):.3f}"
                                  for ra, v in sorted(task.items()))
                      + f"   (chance {1 / n_items:.3f})")

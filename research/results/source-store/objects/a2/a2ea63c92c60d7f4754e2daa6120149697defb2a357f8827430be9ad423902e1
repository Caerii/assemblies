"""Does role retrieval fail for lack of a KEY, or for lack of CAPACITY? (#46)

This is the bridge experiment at a vocabulary size where its metric has
resolution. The toy run measured role retrieval at 0.444 against a chance level
of 0.333 on three words -- one hit either way -- and from that no conclusion of
any kind follows. `corpus.py` now supplies 40 nouns and 20 verbs, putting chance
at 0.025 in the PATIENT area.

TWO HYPOTHESES, and they disagree sharply. Registered before measuring.

H_key   (what role_binding_design.md concluded). The demo parser projects
        LEX -> ROLE from ONE parent, so it stores content with no addressable
        key, whereas the synthetic merge task bound TWO parents and reached
        1.000. If that is right, retrieval sits at chance at EVERY vocabulary
        size and in every area, because no amount of scale creates a key.

H_cap   (the alternative this run exists to give a fair shot). The word's LEX
        assembly IS an addressable key -- k=100 of n neurons, distinct per
        word -- and a feed-forward LEX -> ROLE fiber is then an ordinary
        associative memory. On that reading the earlier chance readings were a
        protocol artifact plus a metric with no resolution, and retrieval
        should work up to the measured capacity bound M_max ~ 1.15 n/k.

WHAT MAKES THIS DECISIVE rather than another inconclusive run: H_cap does not
merely predict "better than chance", it predicts WHERE the failure returns. The
three role areas hold different numbers of items by a fact of the grammar --
every transitive verb requires an animate agent, so AGENT sees 16 types, ACTION
20, PATIENT 40 -- and two operating points straddle the bound differently:

    n=1000,  k=50,  M_max ~ 23  ->  AGENT 16 and ACTION 20 INSIDE the bound,
                                    PATIENT 40 well OUTSIDE it
    n=10000, k=100, M_max ~ 115 ->  all three areas inside

So H_cap predicts a DISSOCIATION ACROSS AREAS WITHIN A SINGLE BRAIN at n=1000
-- AGENT and ACTION retrieve, PATIENT degrades -- and no dissociation at
n=10000. H_key predicts flat chance in all six cells. A within-brain
dissociation cannot be produced by a global protocol bug, which is what makes
this worth running: the two hypotheses cannot both survive.

kp IS HELD AT 10, NOT p. Afferent count is the control parameter -- it flips
the sign of beta's effect -- so p = 10/k gives p=0.2 at k=50 and p=0.1 at
k=100. Matching p instead would confound the comparison with a change in
afferent number, and it would also violate feasibility outright: the condition
need <= g_c fails at kp=5 (need 2.32 vs g_c 2.0) and holds at kp=10.

BETA IS DERIVED, NOT TUNED. With c_max driven to 1 by the binding schedule the
prescription is g = g_c^(1/rounds), i.e. beta = 2^(1/10) - 1 = 0.0718.

PRECONDITION, CHECKED BEFORE ANY ROLE NUMBER IS REPORTED. The readout has to
reproduce each word's LEX assembly, because the role area is being asked to
respond to the same input it was trained on. `train_lexicon` builds each LEX
assembly on a zeroed connectome, so the readout resets too -- but the readout
runs inside `probe`, which blocks weight change, and whether a reset survives
that is a question about the engine and not something to assume. This script
measures LEX reproduction fidelity FIRST and refuses to interpret role
retrieval if it is not essentially 1.0. A role metric read through an input
that never occurred in training is void, and that exact defect has already
produced one round of misleading chance readings here.
"""

from __future__ import annotations

import csv
import math
import os
import sys

os.environ.setdefault("TRAIN_PROGRESS", "0")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

import corpus                                                # noqa: E402

ROUNDS = int(os.environ.get("BC_ROUNDS", "10"))
G_C = float(os.environ.get("BC_GC", "2.0"))
SEEDS = [int(x) for x in os.environ.get("BC_SEEDS", "42,43,44").split(",")]
N_NOUNS = int(os.environ.get("BC_NOUNS", "40"))
N_VERBS = int(os.environ.get("BC_VERBS", "20"))
KP = float(os.environ.get("BC_KP", "10"))
ZIPF_S = float(os.environ.get("BC_ZIPF", "1.0"))
#: (n, k) pairs. The default pair straddles M_max differently -- see docstring.
POINTS = [tuple(int(v) for v in pt.split(","))
          for pt in os.environ.get("BC_POINTS", "1000,50;10000,100").split(";")]

ROLE_NAMES = ["ROLE_AGENT", "ROLE_ACTION", "ROLE_PATIENT"]


def derived_beta(rounds=ROUNDS, c_max=1, g_c=G_C):
    """beta such that cumulative gain on the most frequent item equals g_c."""
    return g_c ** (1.0 / (rounds * c_max)) - 1.0


def need(n, k, p):
    """Drive an assembly must beat to be re-selected against the tail.

    1 + sqrt(2 ln(n/k)) * sqrt((1-p)/(kp)): the expected extreme value of the
    unpotentiated candidate pool, in units of the mean afferent count.
    """
    return 1.0 + math.sqrt(2.0 * math.log(n / k)) * math.sqrt((1 - p) / (k * p))


def wilson(hits, total, z=1.96):
    """Wilson score interval. Correct near 0, unlike the normal approximation.

    Retrieval is a proportion of a small number of trials and will sit near the
    chance floor in at least some cells, which is exactly where a normal
    interval produces negative lower bounds and invites over-reading.
    """
    if total == 0:
        return float("nan"), float("nan"), float("nan")
    ph = hits / total
    d = 1 + z * z / total
    c = (ph + z * z / (2 * total)) / d
    h = z * math.sqrt(ph * (1 - ph) / total + z * z / (4 * total * total)) / d
    return ph, max(0.0, c - h), min(1.0, c + h)


# --------------------------------------------------------------------------
# The LEX -> ROLE drive. ONE implementation, called by training and by readout.
#
# WHY INJECTION RATHER THAN A PHON PROJECTION -- with the wrong reason removed.
#
# The first version of this script measured LEX reproduction at 0.020 and the
# precondition gate refused to report role numbers. The gate was right to fire
# and my reading of it was wrong: I attributed the 0.020 to a mismatch between
# `train_lexicon` (PHON + grounding, with recurrence) and a PHON-only readout.
# `lex_reproducibility.py` then measured that directly and it is not so --
# PHON-only recovery of the stored assembly is 0.977, nineteen times the k/n
# floor, and unchanged after role training. The lexicon is fine, and so is
# `classify_word` at 1.000.
#
# The 0.020 was MY OWN METRIC comparing two coordinate systems: `stored` held
# neuron IDs from an `Assembly` snapshot while the live read used
# `area.winners`, which is compact engine indices. See bridge_parser's
# `role_retrieval` for the same bug with much larger consequences.
#
# Injection is still the right choice, for its own reasons rather than that
# one. `activate_assembly` is the repo's existing primitive for replaying a
# stored snapshot -- its docstring calls it "exactly reproducible" -- so the
# LEX side becomes stationary and order-independent, and training and readout
# run the SAME code path by construction rather than by two call sites
# agreeing. It also removes the residual 2.3% drift of the PHON route and
# matches the paper's design, where LEX assemblies are fixed disjoint
# index-addressed blocks.
#
# SCOPE LIMIT, since injection is an idealisation. This measures the LEX->ROLE
# binding in isolation, assuming a perfect lexical layer. That is the right
# decomposition for a capacity question, but it licenses no claim about the
# end-to-end parser, where LEX is driven by PHON at 0.977 rather than 1.000.
# --------------------------------------------------------------------------

def lex_area_of(parser, word):
    return ("LEX_NOUN" if parser.word_categories[word] == "noun"
            else "LEX_VERB")


def drive(parser, word, role_area):
    """Inject word's LEX assembly, project LEX -> ROLE, return neuron IDs.

    NO role-area self-recurrence: recurrence in a shared area is this
    project's documented collapse channel. Snapshots via `_snap`, so the
    returned set is in NEURON ID space -- `area.winners` is compact indices and
    comparing the two spaces reads exactly chance, which has silently voided a
    result here before.
    """
    from neural_assemblies.assembly_calculus.ops import _snap, activate_assembly

    brain = parser.brain
    lex_area = lex_area_of(parser, word)
    lex = (parser.noun_lexicon if lex_area == "LEX_NOUN"
           else parser.verb_lexicon)
    activate_assembly(brain, lex[word])
    brain.areas[lex_area].fix_assembly()
    for _ in range(parser.rounds):
        brain.project({}, {lex_area: [role_area]})
    asm = _snap(brain, role_area)
    brain.areas[lex_area].unfix_assembly()
    return asm


def make_trainer(binds):
    """Role training from an explicit (word, slot) schedule, via `drive`."""
    def train(parser):
        for word, slot in binds:
            role_area = ROLE_NAMES[slot]
            asm = drive(parser, word, role_area)
            parser.role_lexicons.setdefault(role_area, {})[word] = asm
    return train


def retrieval(parser, determinism_check=True):
    """Can each role area say WHICH word it holds?

    Reports the confusion matrix's two halves separately, because "at chance"
    has two mechanically different causes and reporting only accuracy cannot
    tell them apart:

      * `diag` -- overlap of the readout with the word's OWN stored assembly.
        Low diag means the drive does not reproduce what training stored, i.e.
        a protocol defect.
      * `offdiag` -- mean overlap with OTHER words' stored assemblies. High
        offdiag with high diag means crowding: storage is fine, the assemblies
        are simply not separable.
    """
    from _substrate import probe

    out = {}
    for role_area, lex in parser.role_lexicons.items():
        words = sorted(lex)
        if len(words) < 2:
            continue
        stored = {w: set(int(x) for x in lex[w].winners) for w in words}
        hits, diag, off, nondet = 0, [], [], 0
        for w in words:
            with probe(parser.brain):
                live = set(int(x) for x in drive(parser, w, role_area).winners)
                if determinism_check:
                    again = set(int(x)
                                for x in drive(parser, w, role_area).winners)
                    nondet += (again != live)
            best, best_ov = None, -1.0
            for other, asm in stored.items():
                ov = len(live & asm) / max(len(asm), 1)
                if other == w:
                    diag.append(ov)
                else:
                    off.append(ov)
                if ov > best_ov:
                    best, best_ov = other, ov
            hits += (best == w)
        out[role_area] = {
            "acc": hits / len(words), "hits": hits, "trials": len(words),
            "diag": sum(diag) / max(len(diag), 1),
            "offdiag": sum(off) / max(len(off), 1),
            "nondet": nondet,
        }
    return out


def _duplicate_report(parser, order, nouns):
    """Who lost their assembly: late-trained items, or verbs?

    Groups the role area's stored assemblies by exact winner set. Any group of
    size > 1 is a collapse. Reports the collapsed items' positions in TRAINING
    ORDER and their noun/verb split, which is what distinguishes the two
    candidate mechanisms.
    """
    lex = parser.role_lexicons.get("ROLE_AGENT", {})
    groups = {}
    for w in order:
        if w not in lex:
            continue
        groups.setdefault(frozenset(int(x) for x in lex[w].winners),
                          []).append(w)
    dup = [w for ws in groups.values() if len(ws) > 1 for w in ws[1:]]
    pos = [order.index(w) for w in dup]
    nset = set(nouns)
    return {
        "n_distinct": len(groups),
        "n_dup": len(dup),
        "first_dup_pos": min(pos) if pos else None,
        "dup_nouns": sum(w in nset for w in dup),
        "dup_verbs": sum(w not in nset for w in dup),
        "mean_dup_pos": (sum(pos) / len(pos)) if pos else float("nan"),
    }


def load_sweep():
    """alpha varied by LOAD ALONE, at fixed n, k, p -- the unconfounded test.

    WHY THIS EXISTS. The POINTS sweep varied k at fixed kp, so p = 10/k and
    k/n moved with it, from 0.01 to 0.40. That cannot separate load from
    density, and the data says so plainly: alpha = 1.74 occurs twice with
    margins 0.694 and 0.464, so alpha does not collapse the curve. Worse, at
    k/n = 0.4 an assembly is 40% of the area, so an off-diagonal overlap of
    0.977 is forced by pigeonhole and has nothing to do with how many items
    are stored. The apparent agreement of retrieval accuracy at matched alpha
    was agreement between saturated and degenerate values.

    So: hold n=1000, k=50, p=0.2 -- hence k/n = 0.05 and kp = 10 fixed -- and
    vary only M, the number of words trained into ONE role area. alpha = Mk/n
    = M/20, so M from 10 to 60 sweeps alpha from 0.5 to 3.0 and brackets the
    wall suspected between 1.74 and 2.78 with load as the only moving part.

    ONLY M <= 40 IS CLEAN, and the data announces it. The word list is 40 nouns
    then 20 verbs, so M > 40 starts adding VERBS -- which arrive through
    LEX_VERB, a SECOND source fibre. That changes the number of afferent
    fibres, not just the load, and it shows up as a non-monotonic margin:

        M      10     20     30     40  |    50     60
        margin  .950   .905   .841   .723 |  .799   .831

    The rise is not capacity recovering. Cross-fibre pairs have near-zero
    overlap because their weights are independent, so adding verbs pulls the
    MEAN off-diagonal down while the within-noun pairs are unchanged. LOAD IS
    PER-FIBRE, NOT PER-AREA: at M=60 the busiest fibre still carries only 40
    items, so alpha_eff is 2.0 rather than 3.0.

    Read the curve at M <= 40 only: single fibre, retrieval 1.000 throughout,
    margin decaying smoothly 0.950 -> 0.723 as alpha goes 0.5 -> 2.0. Pushing
    a single fibre past alpha = 2.0 needs MORE SAME-CATEGORY WORDS -- a lexicon
    task, not a parameter change -- so alpha* at depth 1 is bounded below by
    2.0 here and not yet located.
    """
    import bridge_parser as bp

    n = int(os.environ.get("BC_LN", "1000"))
    k = int(os.environ.get("BC_LK", "50"))
    p = KP / k
    m_max = 1.15 * n / k
    beta = derived_beta()
    nouns, verbs = corpus.build(N_NOUNS, N_VERBS)
    words = list(nouns) + list(verbs)
    Ms = [int(x) for x in os.environ.get(
        "BC_MS", "10,20,30,40,50,60").split(",")]

    print(f"\n  LOAD SWEEP -- alpha by M alone   n={n} k={k} p={p:g} "
          f"(kp={KP:g}, k/n={k / n:g} FIXED)  beta={beta:.5f}")
    print(f"  M_max = 1.15n/k = {m_max:.0f}, so alpha = M/{m_max:.0f}; "
          f"one shared role area holds all M words")
    print(f"\n  {'M':>4} {'alpha':>7} {'distinct':>9} {'spread':>8} "
          f"{'diag':>7} {'offdiag':>8} {'margin':>8} {'retrieval':>10} "
          f"{'95% CI':>15} {'chance':>7}")

    out_path = os.path.join(HERE, os.environ.get("BC_LOUT",
                                                 "bridge_load.csv"))
    new = not os.path.exists(out_path)
    fh = open(out_path, "a", newline="", encoding="utf-8")
    wcsv = csv.writer(fh)
    if new:
        wcsv.writerow(["n", "k", "p", "kp", "beta", "rounds", "M", "m_max",
                       "alpha", "acc", "lo", "hi", "chance", "diag",
                       "offdiag", "margin", "spread", "distinct_frac",
                       "nondet", "seeds"])
    for M in Ms:
        sub = words[:M]
        binds = [(w, 0) for w in sub]          # every word into ROLE_AGENT
        if os.environ.get("BC_SHUFFLE") == "1":
            # WHICH items lose their assembly? Two mechanisms predict
            # different answers and the default order cannot tell them apart,
            # because it happens to train all 40 nouns before any verb.
            #
            #   CATEGORY -- verbs collapse because LEX_VERB -> ROLE_AGENT is a
            #     second fiber arriving into an already-trained area.
            #   ORDER    -- LATE items collapse, whatever they are, because the
            #     area has accumulated high-in-degree hub neurons that win for
            #     any input. This project's documented collapse mechanism.
            #
            # Both give exactly 40 distinct at M=50 and M=60 under the default
            # ordering. Shuffling separates them: under CATEGORY the survivors
            # stay the nouns, under ORDER they become the first 40 trained.
            import random as _r
            sub = list(sub)
            _r.Random(12345).shuffle(sub)
            binds = [(w, 0) for w in sub]
        bp.N, bp.K, bp.P, bp.ROUNDS = n, k, p, ROUNDS
        hits = trials = nondet = 0
        dg, od, spr, dfr = [], [], [], []
        diag_words = {"n_dup": 0}
        for seed in SEEDS:
            parser, brain = bp.build(beta, seed, nouns=nouns, verbs=verbs,
                                     roles_fn=make_trainer(binds))
            st = retrieval(parser)["ROLE_AGENT"]
            hits += st["hits"]
            trials += st["trials"]
            nondet += st["nondet"]
            dg.append(st["diag"])
            od.append(st["offdiag"])
            h = bp.area_report(brain, parser)["ROLE_AGENT"]
            spr.append(h.spread)
            dfr.append(h.distinct_frac)
            if seed == SEEDS[0]:
                diag_words = _duplicate_report(parser, sub, nouns)
        ph, lo, hi = wilson(hits, trials)
        mn = lambda v: sum(v) / len(v)                        # noqa: E731
        d, o = mn(dg), mn(od)
        print(f"  {M:>4} {M * k / n:>7.2f} {mn(dfr):>9.3f} {mn(spr):>8.4f} "
              f"{d:>7.3f} {o:>8.4f} {d - o:>8.3f} {ph:>10.3f} "
              f"[{lo:.3f},{hi:.3f}]".rjust(0) + f" {1 / M:>7.4f}")
        if diag_words["n_dup"]:
            print(f"       -> {diag_words['n_distinct']} distinct, "
                  f"{diag_words['n_dup']} collapsed; first at training "
                  f"position {diag_words['first_dup_pos']}, mean position "
                  f"{diag_words['mean_dup_pos']:.1f}; "
                  f"{diag_words['dup_nouns']} nouns / "
                  f"{diag_words['dup_verbs']} verbs")
        wcsv.writerow([n, k, f"{p:g}", f"{KP:g}", f"{beta:.6f}", ROUNDS, M,
                       f"{m_max:.1f}", f"{M * k / n:.4f}", f"{ph:.6f}",
                       f"{lo:.6f}", f"{hi:.6f}", f"{1 / M:.6f}", f"{d:.6f}",
                       f"{o:.6f}", f"{d - o:.6f}", f"{mn(spr):.6f}",
                       f"{mn(dfr):.6f}", nondet,
                       "|".join(str(s) for s in SEEDS)])
        fh.flush()
    fh.close()
    print(f"\n  wrote {out_path}")


def main():
    import bridge_parser as bp

    nouns, verbs = corpus.build(N_NOUNS, N_VERBS)
    sents = corpus.occurrences(nouns, verbs, n_sentences=400, zipf_s=ZIPF_S,
                               seed=0)
    binds = corpus.bindings(sents)
    info = corpus.summary(nouns, verbs, sents)
    beta = derived_beta()

    per_role = {ROLE_NAMES[s]: len({w for w, sl in binds if sl == s})
                for s in range(3)}

    print(f"\n  ROLE RETRIEVAL vs CAPACITY   rounds={ROUNDS} "
          f"zipf_s={ZIPF_S} seeds={SEEDS}")
    print(f"  corpus: {info['n_sentences']} sentences, {info['types']} types, "
          f"{len(binds)} bindings (c_max={info['c_max']} in the stream, "
          f"1 in training)")
    print(f"  derived beta = g_c^(1/rounds) - 1 = {beta:.5f} "
          f"(g/step = {(1 + beta) ** ROUNDS:.3f} = g_c = {G_C})")
    print(f"  items per role area: " + "  ".join(
        f"{r.replace('ROLE_', '')}={m}" for r, m in per_role.items()))

    print(f"\n  {'n':>6} {'k':>5} {'p':>6} {'kp':>5} {'M_max':>7} "
          f"{'need':>7} {'feasible':>9}")
    for n, k in POINTS:
        p = KP / k
        m_max = 1.15 * n / k
        nd = need(n, k, p)
        print(f"  {n:>6} {k:>5} {p:>6.3f} {KP:>5.0f} {m_max:>7.0f} "
              f"{nd:>7.3f} {'YES' if nd <= G_C else 'NO':>9}")
        for role, m in per_role.items():
            print(f"           {role.replace('ROLE_', ''):<8} M={m:<4} "
                  f"alpha=M/M_max={m / m_max:.2f}  "
                  f"predicted {'OK' if m < m_max else 'DEGRADED'}")

    out_path = os.path.join(HERE, os.environ.get("BC_OUT",
                                                 "bridge_capacity.csv"))
    new = not os.path.exists(out_path)
    fh = open(out_path, "a", newline="", encoding="utf-8")
    wcsv = csv.writer(fh)
    if new:
        wcsv.writerow(["n", "k", "p", "kp", "beta", "rounds", "seeds", "area",
                       "n_items", "m_max", "alpha", "acc", "lo", "hi",
                       "chance", "diag", "offdiag", "distinct_frac", "spread",
                       "floor", "nondet", "collapsed"])
        fh.flush()

    for n, k in POINTS:
        p = KP / k
        m_max = 1.15 * n / k
        print(f"\n  === n={n} k={k} p={p:g} (kp={KP:g}) "
              f"M_max~{m_max:.0f} beta={beta:.5f} ===")
        bp.N, bp.K, bp.P, bp.ROUNDS = n, k, p, ROUNDS
        agg, task = {}, {}
        for seed in SEEDS:
            parser, brain = bp.build(beta, seed, nouns=nouns, verbs=verbs,
                                     roles_fn=make_trainer(binds))
            res = retrieval(parser)
            for ra, st in res.items():
                task.setdefault(ra, []).append(st)
            for area, h in bp.area_report(brain, parser).items():
                agg.setdefault(area, []).append(h)
            print(f"    seed {seed}: " + "  ".join(
                f"{ra.replace('ROLE_', '')} {st['acc']:.3f}"
                for ra, st in sorted(res.items())))

        nondet = sum(st["nondet"] for sts in task.values() for st in sts)
        if nondet:
            print(f"\n    WARNING: the readout drive was NON-DETERMINISTIC on "
                  f"{nondet} words. Retrieval below is not a stable "
                  f"measurement.")

        print(f"\n    {'area':<14} {'M':>4} {'a=M/Mmax':>9} {'distinct':>9} "
              f"{'spread':>8} {'diag':>6} {'offdiag':>8} {'retrieval':>10} "
              f"{'95% CI':>15} {'chance':>7}")
        for area in ROLE_NAMES + ["LEX_NOUN", "LEX_VERB"]:
            hs = agg.get(area) or []
            if not hs:
                continue
            mean = lambda f: sum(f(h) for h in hs) / len(hs)  # noqa: E731
            dfrac, spr = mean(lambda h: h.distinct_frac), mean(lambda h: h.spread)
            m = hs[0].n_items
            sts = task.get(area, [])
            if sts:
                hits = sum(st["hits"] for st in sts)
                trials = sum(st["trials"] for st in sts)
                ph, lo, hi = wilson(hits, trials)
                dg = sum(st["diag"] for st in sts) / len(sts)
                od = sum(st["offdiag"] for st in sts) / len(sts)
                ci, acc_s, ch = f"[{lo:.3f},{hi:.3f}]", f"{ph:.3f}", 1.0 / m
                dg_s, od_s = f"{dg:.3f}", f"{od:.4f}"
            else:
                ph = lo = hi = dg = od = float("nan")
                ci = acc_s = dg_s = od_s = "-"
                ch = float("nan")
            print(f"    {area:<14} {m:>4} {m / m_max:>9.2f} {dfrac:>9.3f} "
                  f"{spr:>8.4f} {dg_s:>6} {od_s:>8} {acc_s:>10} {ci:>15} "
                  f"{('%.4f' % ch) if ch == ch else '-':>7}")
            wcsv.writerow([n, k, f"{p:g}", f"{KP:g}", f"{beta:.6f}", ROUNDS,
                           "|".join(str(s) for s in SEEDS), area, m,
                           f"{m_max:.1f}", f"{m / m_max:.4f}",
                           f"{ph:.6f}", f"{lo:.6f}", f"{hi:.6f}",
                           f"{ch:.6f}", f"{dg:.6f}", f"{od:.6f}",
                           f"{dfrac:.6f}", f"{spr:.6f}",
                           f"{hs[0].floor:.6f}", nondet,
                           sum(h.collapsed for h in hs)])
        fh.flush()
    fh.close()
    print(f"\n  wrote {out_path}")


if __name__ == "__main__":
    if os.environ.get("BC_MODE") == "load":
        load_sweep()
    else:
        main()

"""Cross-situational learning: reference and structure solved together.

Implements `research/notes/PREREG_unaligned_scenes.md` (+ Amendment 1).

TODAY the grounded pipeline gives each word its own feature bundle, so word ->
referent alignment is a given and only the role mapping is learned. HERE the
learner sees a sentence as WORDS ONLY and a scene as UNLABELLED feature
bundles (participants in causal order, plus the action). Which word denotes
which participant has to come from cross-situational statistics: "dog"
co-occurs with [DOG, ANIMAL] every time and with [BALL, OBJECT] only sometimes.

THE LEARNER never touches a word's grounding. `align_train` receives
`(words, scene_features)` and nothing else; the deleted groundings are read
only by the SCORER. That separation is enforced by construction, not by
discipline: the scorer's inputs are built in `main`, the learner's in
`experience_of`.

    stage 1  alignment: phon(word) + every scene feature -> LEX / FEAT,
             reciprocal fibers, Hebbian. Readout is RECONSTRUCTION
             (Amendment 1): cue phon alone under a probe, LEX -> FEAT, compare
             the reconstructed feature assembly with each candidate bundle's
             own feature-driven assembly.
    stage 2  roles from the ALIGNED bundle's position in the scene's causal
             order, contexts rebuilt from the ALIGNED features, then the
             unchanged word-order induction.

    python research/experiments/unaligned_scenes.py [--seeds 42,1,2,3,4] [--no-u2]
"""
from __future__ import annotations

import argparse
import copy
import os
import random
import sys
import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _HERE)

from neural_assemblies.assembly_calculus.ops import _snap                # noqa: E402
from neural_assemblies.core.brain import Brain                           # noqa: E402
from neural_assemblies.diagnostics import ensemble_from_values           # noqa: E402
from neural_assemblies.assembly_calculus.emergent.core.scene import (    # noqa: E402
    CAUSAL_ROLE_ORDER, _word_features)

LEX, FEAT = "LEX", "FEAT"
N, K, P, BETA = 1000, 50, 0.05, 0.10
ROUNDS_WORD = 5          # rounds per word occurrence during alignment
ROUNDS_READ = 3
MIN_EXPOSURES = 3
_MODALITIES = ("visual", "motor", "properties", "spatial", "social",
               "temporal", "emotional")


def _key(bundle: Sequence[str]) -> Tuple[str, ...]:
    return tuple(sorted(set(bundle)))


# ---------------------------------------------------------------------------
# what the learner is allowed to see
# ---------------------------------------------------------------------------

def experience_of(corpus) -> List[Tuple[List[str], List[Tuple[str, ...]]]]:
    """(words, scene bundles) per sentence that has a scene. NO groundings."""
    out = []
    for s in corpus:
        ev = getattr(s, "event", None)
        if ev is None:
            continue
        bundles = [_key(b) for b in ev.participants] + [_key(ev.action)]
        out.append((list(s.words), bundles))
    return out


def shuffle_scenes(exp, seed):
    """U3: permute scenes across sentences with the SAME participant count."""
    rng = random.Random(seed + 777)
    by_n = defaultdict(list)
    for i, (_w, bundles) in enumerate(exp):
        by_n[len(bundles)].append(i)
    out = list(exp)
    for idxs in by_n.values():
        perm = idxs[:]
        rng.shuffle(perm)
        for src, dst in zip(idxs, perm):
            out[dst] = (exp[dst][0], exp[src][1])
    return out


# ---------------------------------------------------------------------------
# stage 1: alignment
# ---------------------------------------------------------------------------

class Aligner:
    def __init__(self, seed, words, features, scaling=True, *,
                 n=None, k=None, stim_size=None):
        # Area size and phon stimulus size are PARAMETERS so the capacity
        # study (word_capacity.py) can sweep them; defaults reproduce U1-U3.
        n = N if n is None else int(n)
        k = K if k is None else int(k)
        stim_size = k if stim_size is None else int(stim_size)
        random.seed(seed)
        np.random.seed(seed)
        # SYNAPTIC SCALING ON THE FEATURE AREA (Amendment 5). Raw Hebbian mass
        # follows the BASE RATE of a bundle, not its association with a word
        # ([[hebbian-mass-follows-frequency]]): before this, every word's
        # reconstruction pointed at the corpus's most frequent bundle
        # (dog 0.28, ball 0.26, chases 0.28 -- all at ('ANIMAL','DOG')).
        # Column renormalization divides each FEAT neuron's incoming mass by
        # its own total, which is exactly the base-rate correction
        # cross-situational learning needs. Scoped to FEAT; no refracted area
        # exists here (AUDIT_refraction_scaling.md).
        self.b = Brain(p=P, seed=seed, engine="numpy_sparse",
                       synaptic_scaling=frozenset({FEAT}) if scaling else False)
        self.b.add_area(LEX, n, k, BETA)
        self.b.add_area(FEAT, n, k, BETA)
        # MATERIALIZED, for the reason `NemoArcFSM` materializes its state
        # area: while an area is nearly empty the lazy candidate sampler
        # flattens DISJOINT inputs into overlapping winners
        # ([[sampler-merges-at-low-load]]). Measured here before any bar was
        # read -- every feature bundle's FEAT assembly was IDENTICAL (pairwise
        # 1.00) although the bundles share no features at all, so alignment
        # sat at exactly inventory chance with every word mapped to the
        # alphabetically first bundle. Not sparsity: raising n does not fix
        # it, materializing does ([[sampler-is-the-whole-discrepancy]]).
        self.b.materialize_area(LEX)
        self.b.materialize_area(FEAT)
        self.phon = {}
        for w in sorted(words):
            self.phon[w] = f"phon_{w}"
            self.b.add_stimulus(self.phon[w], stim_size)
        self.feat = {}
        for f in sorted(features):
            self.feat[f] = f"feat_{f}"
            self.b.add_stimulus(self.feat[f], k)

    def train(self, exp, rng):
        """One co-presentation per (word, PERCEIVED OBJECT) pair.

        Amendment 3. Firing the whole scene's features at once made FEAT hold
        one "scene soup" assembly, so nothing could bind a word to a
        PARTICULAR participant and the cross-fiber learned the same thing for
        every word in the sentence. Participants are distinct perceived
        objects -- the scene already supplies them separately -- so each is
        presented in its own step. Nothing about WHICH pairing is correct is
        supplied: every word is paired with every bundle in its scene, and
        only the cross-situational statistics separate them (a word's true
        referent is in every one of its scenes; each distractor is in some).

        ONE DIRECTION, LEX -> FEAT (Amendment 4). LEX must be driven by its
        phonological stimulus ALONE, because that is the only cue the readout
        has: with a FEAT -> LEX fiber in the loop the trained LEX assembly
        overlapped the phon-cued one by 0.30, so the conjunction was written on
        cells the readout never activates. Same lesson as
        [[writer-and-reader-must-share-the-lookup]]. FEAT stays pinned by its
        feature stimuli, so the write lands on LEX(word) x FEAT(bundle) and
        the readout cues exactly those two sets.

        FEEDFORWARD (Amendment 2): with LEX -> LEX and FEAT -> FEAT recurrence
        both areas collapsed to a single attractor.
        """
        order = list(range(len(exp)))
        rng.shuffle(order)
        for i in order:
            words, bundles = exp[i]
            for w in words:
                for b in bundles:
                    stim = {self.feat[f]: [FEAT] for f in b}
                    stim[self.phon[w]] = [LEX]
                    self.b.inhibit_areas([LEX, FEAT])
                    self.b.project(stim, {})
                    for _ in range(ROUNDS_WORD):
                        self.b.project(stim, {LEX: [FEAT]})

    # -- probe-isolated readouts ------------------------------------------
    # `probe()`, NOT `read_only()`: read_only freezes the winners, so every
    # snapshot returns the same set and every overlap reads 1.000 -- the
    # fake-perfect signature ([[fake-perfect-probe-signatures]]). Measured
    # here first: 13 words, 13 bundles, all pairwise 1.00. probe() lets
    # winners move without learning or recruitment.
    def bundle_assembly(self, bundle):
        with self.b.probe():
            self.b.inhibit_areas([LEX, FEAT])
            stim = {self.feat[f]: [FEAT] for f in bundle}
            self.b.project(stim, {})     # SAME cue the training step used
            return _snap(self.b, FEAT)

    def reconstruct(self, word):
        """Cue phon alone, let LEX settle, then LEX -> FEAT. Amendment 1."""
        with self.b.probe():
            self.b.inhibit_areas([LEX, FEAT])
            self.b.project({self.phon[word]: [LEX]}, {})   # SAME LEX cue
            self.b.project({}, {LEX: [FEAT]})
            return _snap(self.b, FEAT)


def align(seed, exp, words, features, scaling=True):
    al = Aligner(seed, words, features, scaling=scaling)
    al.train(exp, random.Random(seed + 11))
    inventory = sorted({b for _w, bs in exp for b in bs})
    inv_asm = {b: al.bundle_assembly(b) for b in inventory}
    recon = {w: al.reconstruct(w) for w in words}
    scores = {w: {b: recon[w].overlap(inv_asm[b]) for b in inventory}
              for w in words}
    return scores, inventory


# ---------------------------------------------------------------------------
# scoring (the ONLY place the deleted groundings are read)
# ---------------------------------------------------------------------------

def targets_of(corpus) -> Dict[str, Tuple[str, ...]]:
    """word -> its own feature bundle, from the grounding the learner never saw.
    Words whose bundle is not a scene bundle anywhere (determiners, adjectives)
    have no alignment target and are excluded."""
    t = {}
    for s in corpus:
        ev = getattr(s, "event", None)
        if ev is None:
            continue
        scene = {_key(b) for b in ev.participants} | {_key(ev.action)}
        for w, c in zip(s.words, s.contexts):
            k = _key(_word_features(c))
            if k and k in scene:
                if w in t and t[w] != k:
                    raise AssertionError(f"{w!r} grounds two ways: {t[w]} {k}")
                t[w] = k
    return t


def score(scores, inventory, exp, targets, exposures):
    """Per-type (against the whole inventory) and per-scene (registered)."""
    words = [w for w in targets if exposures[w] >= MIN_EXPOSURES]
    type_hits = 0
    for w in words:
        best = max(inventory, key=lambda b: scores[w][b])
        type_hits += int(best == targets[w])
    type_acc = type_hits / max(len(words), 1)

    occ = hit = 0
    chance = 0.0
    for sent_words, bundles in exp:
        for w in sent_words:
            if w not in targets or exposures[w] < MIN_EXPOSURES:
                continue
            best = max(bundles, key=lambda b: scores[w][b])
            occ += 1
            hit += int(best == targets[w])
            chance += 1.0 / len(bundles)
    return dict(type_acc=type_acc, n_types=len(words),
                scene_acc=hit / max(occ, 1), n_occ=occ,
                scene_chance=chance / max(occ, 1),
                inv_chance=1.0 / len(inventory))


def two_person_words(corpus, targets):
    ws = set()
    for s in corpus:
        ev = getattr(s, "event", None)
        if ev is None:
            continue
        persons = [b for b in ev.participants if "PERSON" in b]
        if len(persons) >= 2:
            for w, c in zip(s.words, s.contexts):
                if w in targets and "PERSON" in _word_features(c):
                    ws.add(w)
    return sorted(ws)


# ---------------------------------------------------------------------------
# stage 2: roles from ALIGNED bundles, then word-order induction
# ---------------------------------------------------------------------------

def aligned_corpus(corpus, scores, inventory, modality_of):
    """Rebuild every scene sentence's roles AND contexts from the alignment."""
    from neural_assemblies.assembly_calculus.emergent.core.grounding import (
        GroundingContext)
    out = []
    for s in corpus:
        ev = getattr(s, "event", None)
        if ev is None:
            continue
        s2 = copy.deepcopy(s)
        parts = [_key(b) for b in ev.participants]
        act = _key(ev.action)
        roles, ctxs = [], []
        for w in s.words:
            best = max(inventory, key=lambda b: scores[w][b]) \
                if w in scores else None
            role, feats = None, ()
            if best is not None and scores[w][best] > 0:
                if best == act:
                    role, feats = "action", best
                elif best in parts:
                    i = parts.index(best)
                    role = CAUSAL_ROLE_ORDER[i] if i < len(CAUSAL_ROLE_ORDER) else None
                    feats = best
            kw = {m: [] for m in _MODALITIES}
            for f in feats:
                kw[modality_of.get(f, "visual")].append(f)
            roles.append(role)
            ctxs.append(GroundingContext(**kw))
        s2.roles = roles
        s2.contexts = ctxs
        out.append(s2)
    return out


def induce_order(seed, corpus_aligned):
    os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
    os.environ.setdefault("TRAIN_PROGRESS", "0")
    from lesion_aphasia import build_corpus
    from neural_assemblies.assembly_calculus.emergent.curriculum.data import (
        create_training_sentences)
    from neural_assemblies.assembly_calculus.emergent.nemo_parse import (
        NemoParser, infer_transitive_verbs)
    from neural_assemblies.assembly_calculus.emergent.parser import EmergentParser
    from word_order_induction import (
        AGENTY, ORDERS, PATIENTY, VERBS, _permute, lexical_preference)

    transitive = infer_transitive_verbs(create_training_sentences() + build_corpus())
    parser = EmergentParser(n=N, k=K, p=P, beta=BETA, seed=seed, rounds=10)
    parser.train(create_training_sentences() + corpus_aligned)
    prefs = {w: lexical_preference(parser, w, transitive)
             for w in AGENTY + PATIENTY}
    items = [(s, v, o) for s in AGENTY for v in VERBS for o in PATIENTY]
    hits, rows = 0, []
    for true_order in ORDERS:
        sentences = [_permute(s, v, o, true_order) for s, v, o in items]
        agreement = {}
        for hyp in ORDERS:
            agree = judged = 0
            for words in sentences:
                pred = NemoParser(copy.deepcopy(parser),
                                  transitive_verbs=transitive,
                                  sequential=True, word_order_type=hyp,
                                  ).parse(list(words))
                for w in words:
                    if prefs.get(w) in ("AGENT", "PATIENT"):
                        judged += 1
                        agree += int(pred.get(w) == prefs[w])
            agreement[hyp] = agree / max(judged, 1)
        best = [h for h in ORDERS if agreement[h] == max(agreement.values())]
        ok = best == [true_order]
        hits += int(ok)
        rows.append((true_order, ",".join(best), ok))
    return hits, rows, prefs


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,1,2,3,4")
    ap.add_argument("--no-u2", action="store_true")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    from grounded_corpus import build
    corpus = build()
    exp = experience_of(corpus)
    targets = targets_of(corpus)
    exposures = Counter(w for ws, _b in exp for w in ws)
    words = sorted({w for ws, _b in exp for w in ws})
    features = sorted({f for _w, bs in exp for b in bs for f in b})
    modality_of = {}
    for s in corpus:
        for c in s.contexts:
            for m in _MODALITIES:
                for f in getattr(c, m, ()) or ():
                    modality_of.setdefault(f, m)
    two_person = two_person_words(corpus, targets)
    mean_p = float(np.mean([len(b) for _w, b in exp]))
    print(f"UNALIGNED SCENES  sentences with scenes {len(exp)}/{len(corpus)}  "
          f"word types {len(words)}  features {len(features)}  "
          f"alignable types {len(targets)}  mean bundles/scene {mean_p:.2f}")
    print(f"  n={N} k={K} p={P} beta={BETA} rounds/word {ROUNDS_WORD}  "
          f"min exposures {MIN_EXPOSURES}  two-person words {two_person}")

    type_accs, scene_accs, shuf_accs, tp_accs = [], [], [], []
    u2_hits = []
    for seed in seeds:
        t0 = time.perf_counter()
        scores, inventory = align(seed, exp, words, features)
        r = score(scores, inventory, exp, targets, exposures)
        type_accs.append(r["type_acc"])
        scene_accs.append(r["scene_acc"])
        tp = [w for w in two_person if exposures[w] >= MIN_EXPOSURES]
        tp_hit = sum(int(max(inventory, key=lambda b: scores[w][b])
                         == targets[w]) for w in tp)
        tp_accs.append(tp_hit / max(len(tp), 1))
        # U3: decorrelate word and scene
        s_scores, s_inv = align(seed, shuffle_scenes(exp, seed), words, features)
        rs = score(s_scores, s_inv, exp, targets, exposures)
        shuf_accs.append(rs["type_acc"])
        print(f"\n  seed {seed}: type-acc {r['type_acc']:.3f} "
              f"(n={r['n_types']}, inventory chance {r['inv_chance']:.3f})  "
              f"scene-acc {r['scene_acc']:.3f} (n={r['n_occ']}, chance "
              f"{r['scene_chance']:.3f})  two-person {tp_accs[-1]:.3f} "
              f"(n={len(tp)})  SHUFFLED type-acc {rs['type_acc']:.3f}  "
              f"[{time.perf_counter() - t0:.0f}s]")
        wrong = [(w, targets[w], max(inventory, key=lambda b: scores[w][b]))
                 for w in targets if exposures[w] >= MIN_EXPOSURES
                 and max(inventory, key=lambda b: scores[w][b]) != targets[w]]
        for w, t, g in wrong[:6]:
            print(f"      miss {w!r}: target {t} -> got {g}")
        if not args.no_u2:
            t1 = time.perf_counter()
            hits, rows, prefs = induce_order(
                seed, aligned_corpus(corpus, scores, inventory, modality_of))
            u2_hits.append(hits)
            print(f"    U2 prefs {prefs}")
            for o, b, ok in rows:
                print(f"    U2 {o:<5} -> {b:<20} {'YES' if ok else 'no'}")
            print(f"    U2 recovered {hits}/6   [{time.perf_counter() - t1:.0f}s]")

    print("\n=== BARS (PREREG_unaligned_scenes.md) ===")
    e1 = ensemble_from_values(type_accs, label="U1 type-acc vs inventory")
    e1s = ensemble_from_values(scene_accs, label="U1 scene-acc (registered)")
    e3 = ensemble_from_values(shuf_accs, label="U3 shuffled type-acc")
    e4 = ensemble_from_values(tp_accs, label="U4 two-person words")
    inv_chance = 1.0 / len(inventory)
    sc_chance = r["scene_chance"]
    print(f"  {e1}   chance {inv_chance:.3f}")
    print(f"  {e1s}   chance {sc_chance:.3f}")
    u1 = e1s.mean >= 0.85 and e1s.beats(2 * sc_chance)
    print(f"  {'PASS' if u1 else ('FAIL' if e1s.mean <= 0.60 else 'INCONCLUSIVE')}"
          f"  U1 scene-acc >= 0.85 and lower bound beats 2x chance")
    print(f"  {e3}   (must be <= 1.5x chance = {1.5 * inv_chance:.3f})")
    u3 = e3.mean <= 1.5 * inv_chance
    print(f"  {'PASS' if u3 else 'FAIL'}  U3 shuffled scenes fall to chance"
          + ("" if u3 else "  -- THE SCORER LEAKS; U1 IS VOID"))
    print(f"  {e4}   U4 two-person words (reported, no bar)")
    if u2_hits:
        good = sum(int(h >= 5) for h in u2_hits)
        print(f"  U2 orders recovered per seed {u2_hits}; >=5/6 on "
              f"{good}/{len(u2_hits)} seeds  "
              f"{'PASS' if good >= max(4, len(u2_hits) - 1) else 'FAIL'}")


if __name__ == "__main__":
    main()

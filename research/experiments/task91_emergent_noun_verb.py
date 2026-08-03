"""#91: does noun-vs-verb emerge WITHOUT a label?

WHAT THIS REPLACES
------------------
`emergent/core/grounding.py` decides a word's core area from its
`dominant_modality`, resolved by a fixed priority order over hand-authored
modality lists, and cites Mitropolsky & Papadimitriou (2025) for it. That paper
has no such map. Cut the three hand-authored routes and the parser answers one
class for everything, exactly at majority baseline -- which is why "the parser
induces categories" is currently not a claim this repository can make.

THE PAPER'S ACTUAL MECHANISM (Sec. 2.1-2.2)
-------------------------------------------
LEX1 and LEX2 are BOTH tabula rasa and EVERY word is presented to BOTH. No label
exists anywhere. The only prior is architectural asymmetry:

    "Four of these 2m + 6 fibers, namely the ones between PHON and the two
     lexical areas, as well as the one between LEX1 and VISUAL, and the one
     between LEX2 and MOTOR, have increased parameters beta AND p, making them
     stronger conduits of synaptic input."

Nouns carry VISUAL grounding, verbs carry MOTOR grounding, and the class of a
word falls out of which lexical area can hold a stable assembly for it.

THE READOUT IS A STABILITY PROBE, NOT A CLASSIFIER (Property 3)

    "firing PHON[w] into LEX2 results in a 'wobbly' set of neurons: firing this
     set again recurrently ... results in a quite different set."

So: fire PHON[w] into both areas, keep firing, and ask which k-cap is
self-sustaining. Note what this repository normally calls that quantity --
recurrent instability is treated as a DEFECT in [[self-recurrence-stability-
window]], [[recurrence-is-the-collapse-channel]] and [[norm-init-stability-
threshold]]. Here it is the signal.

WHY IT WAS UNBUILDABLE UNTIL NOW
--------------------------------
`p` was a Brain-level constant. Per-fiber beta existed (`update_plasticity`);
per-fiber `p` did not, so the paper's four privileged fibers could not be
expressed. `add_connectivity` now implements it on `numpy_exact` (which is also
the engine whose drive is computed rather than sampled, so a graded stability
measurement is not being read off the candidate sampler).

PRE-REGISTERED
--------------
E1 Nouns are more stable in LEX1 than in LEX2, and verbs the reverse. The
   MINIMAL claim -- direction only, no threshold.
E2 Classification by "which area is more stable" beats the 0.5 majority
   baseline over the whole lexicon. This is the claim that would actually
   replace the annotation, and the baseline is stated because the repo's
   existing category result sits exactly at majority.
E3 The gap GROWS with the number of distinct complements a word occurred with
   (paper Fig. 3d: "dog" with 7 verbs reads ~30% stability in LEX2 against
   100% in LEX1). This is the one that distinguishes a real mechanism from a
   lucky initialization, because it is a claim about co-occurrence statistics
   rather than about the wiring.

CONTROL, and it decides whether any of the above means anything: the SAME
protocol with the asymmetry REMOVED (all fibers at the base p and beta). If
classification still beats baseline there, the split is coming from something
other than the modality asymmetry -- a hidden regularity in how words are
indexed, most likely -- and E1-E3 are void. A result that survives its own
ablation is the only kind worth reporting here.

DEVIATIONS FROM THE PAPER, stated rather than buried: n=10^4 not 10^5 (speed);
no C_i context areas (m=0, which the paper says its algorithm tolerates); and
intransitive two-word sentences only, which is the paper's own Sec. 2.1 setting.
"""

from __future__ import annotations

import os
import statistics
import sys

os.environ.setdefault("TRAIN_PROGRESS", "0")

import numpy as np  # noqa: E402

from neural_assemblies.core.brain import Brain  # noqa: E402
from neural_assemblies.diagnostics import (  # noqa: E402
    assembly_overlap, read_assembly)

PHON, VISUAL, MOTOR, LEX1, LEX2 = "PHON", "VISUAL", "MOTOR", "LEX1", "LEX2"

N, K = 10000, 50
P_BASE, P_STRONG = 0.05, 0.20
BETA_BASE, BETA_STRONG = 0.06, 0.20
TAU = 2

NOUNS = ["dog", "cat", "bird", "man", "girl", "ball", "tree", "fish"]
VERBS = ["runs", "jumps", "eats", "sleeps"]


class Lexicon:
    """PHON + VISUAL/MOTOR -> LEX1/LEX2, tabula rasa, no labels."""

    def __init__(self, seed: int, asymmetric: bool = True):
        self.asymmetric = asymmetric
        self.brain = Brain(p=P_BASE, seed=seed, engine="numpy_exact",
                           norm_init=True)
        words = NOUNS + VERBS
        self.index = {w: i for i, w in enumerate(words)}
        for area in (PHON, VISUAL, MOTOR):
            self.brain.add_explicit_area(area, len(words) * K, K, BETA_BASE)
        for area in (LEX1, LEX2):
            self.brain.add_area(area, N, K, BETA_BASE)

        # THE ONLY PRIOR. Under `asymmetric=False` this block is skipped and
        # every fiber keeps the base p and beta -- the ablation that decides
        # whether the split comes from the modality asymmetry or from
        # somewhere else.
        if asymmetric:
            for lex in (LEX1, LEX2):
                self._strengthen(PHON, lex)
            self._strengthen(VISUAL, LEX1)
            self._strengthen(MOTOR, LEX2)

    def _strengthen(self, src: str, dst: str) -> None:
        """Raise BOTH p and beta on the src -> dst fiber.

        The two calls take the SAME argument order -- `add_connectivity(source,
        target)` and `update_plasticity(from_area, to_area)` -- and the first
        version of this method wrote the second one reversed. p landed on
        VISUAL -> LEX1 while beta landed on LEX1 -> VISUAL, and the experiment
        read accuracy 0.000: a PERFECT inversion, nouns preferring the
        MOTOR-linked area and verbs the VISUAL-linked one. Worth noting that
        the symptom was not noise but an exact anti-correlation, which is what
        finally made it obvious.
        """
        self.brain.add_connectivity(src, dst, P_STRONG)
        self.brain.update_plasticity(src, dst, BETA_STRONG)

    def present(self, noun: str, verb: str) -> None:
        """One grounded sentence.

        The scene fires THROUGHOUT: as in the paper, VISUAL[noun] and
        MOTOR[verb] are both active while each word is heard, so co-occurrence
        -- not any label -- is what ties a word to a modality.
        """
        self.brain.inhibit_areas([LEX1, LEX2])
        for w in (noun, verb):
            self.brain.activate(VISUAL, self.index[noun])
            self.brain.activate(MOTOR, self.index[verb])
            self.brain.activate(PHON, self.index[w])
            for _ in range(TAU):
                self.brain.project({}, {
                    PHON: [LEX1, LEX2],
                    VISUAL: [LEX1, LEX2],
                    MOTOR: [LEX1, LEX2],
                    LEX1: [LEX1], LEX2: [LEX2],
                })

    def stability(self, word: str, lex: str, rounds: int = 3) -> float:
        """Property 1/3: is the k-cap PHON[w] elects self-sustaining?

        Fire PHON[w] into `lex`, snapshot, then keep firing (PHON[w] plus the
        area's own recurrence) and compare. A stable assembly returns itself;
        the paper's "wobbly" set does not.
        """
        with self.brain.frozen():
            self.brain.inhibit_areas([lex])
            self.brain.activate(PHON, self.index[word])
            self.brain.project({}, {PHON: [lex]})
            first = read_assembly(self.brain, lex)
            for _ in range(rounds):
                self.brain.project({}, {PHON: [lex], lex: [lex]})
            return assembly_overlap(first, read_assembly(self.brain, lex))

    def recall(self, word: str, lex: str) -> float:
        """The paper's own algorithmic test (Property 3), not the stability one.

            "try firing PHON[w] into LEX1 and LEX2, and see which of the two
             k-caps activates PHON[w] when firing back into PHON"

        Preferred over `stability` as the classifier because stability
        saturates: PHON keeps firing through the recurrent rounds and is itself
        a strengthened fiber, so it can pin the winners on its own. The
        round-trip has no such ceiling -- a k-cap that is not really this word's
        assembly cannot reconstruct this word's phonology.
        """
        with self.brain.frozen():
            self.brain.inhibit_areas([lex])
            self.brain.activate(PHON, self.index[word])
            target = read_assembly(self.brain, PHON)
            self.brain.project({}, {PHON: [lex]})
            # PHON MUST BE UNFIXED BEFORE THE RETURN LEG. `Brain.activate`
            # calls `fix_assembly`, and a fixed target short-circuits
            # `project_into` -- it returns its existing winners untouched. The
            # first version of this probe left it fixed and read EXACTLY 1.000
            # for every word in both areas: a round trip that never travelled.
            self.brain.areas[PHON].unfix_assembly()
            self.brain._engine.unfix_assembly(PHON)
            self.brain.project({}, {lex: [PHON]})
            got = read_assembly(self.brain, PHON)
            self.brain.activate(PHON, self.index[word])   # restore
            return assembly_overlap(target, got)

    def classify(self, word: str) -> str:
        r1, r2 = self.recall(word, LEX1), self.recall(word, LEX2)
        return LEX1 if r1 >= r2 else LEX2


def train(lex: Lexicon, sentences: int, seed: int, complements=None):
    """`complements` caps how many distinct verbs each noun is paired with.

    That cap is the E3 axis: the paper's Fig. 3d says a noun's assembly in the
    WRONG area destabilises as the variety of its complements grows.
    """
    rng = np.random.default_rng(seed)
    for _ in range(sentences):
        noun = NOUNS[rng.integers(len(NOUNS))]
        pool = VERBS if complements is None else VERBS[:complements]
        lex.present(noun, pool[rng.integers(len(pool))])


def arm(asymmetric: bool, seed: int, sentences: int, complements=None):
    lx = Lexicon(seed, asymmetric=asymmetric)
    train(lx, sentences, seed, complements)
    noun_gap = statistics.mean(
        lx.stability(w, LEX1) - lx.stability(w, LEX2) for w in NOUNS)
    verb_gap = statistics.mean(
        lx.stability(w, LEX2) - lx.stability(w, LEX1) for w in VERBS)
    correct = (sum(lx.classify(w) == LEX1 for w in NOUNS)
               + sum(lx.classify(w) == LEX2 for w in VERBS))
    # ABSOLUTE level, not just the gap. A smoke test at 20 sentences read
    # 0.80-0.94 in BOTH areas for BOTH classes: with PHON still firing during
    # the recurrent rounds, and PHON->LEX among the strengthened fibers, PHON
    # alone can pin the winners and the "stability" is then a property of the
    # input rather than of anything learned. A gap computed on top of two
    # saturated numbers is noise, so the level has to be visible next to it.
    level = statistics.mean(
        [lx.stability(w, LEX1) for w in NOUNS + VERBS]
        + [lx.stability(w, LEX2) for w in NOUNS + VERBS])
    return noun_gap, verb_gap, correct / (len(NOUNS) + len(VERBS)), level


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    seeds = (1, 2, 3)
    sentences = 10 * (len(NOUNS) + len(VERBS))   # the paper's ~10 per word
    print(f"\n  #91 -- does noun/verb emerge with NO label?")
    print(f"  n={N} k={K}, p {P_BASE}->{P_STRONG} and beta "
          f"{BETA_BASE}->{BETA_STRONG} on 4 fibers, engine numpy_exact")
    print(f"  {len(NOUNS)} nouns + {len(VERBS)} verbs, {sentences} sentences "
          f"(~10/word), {len(seeds)} seeds, majority baseline "
          f"{max(len(NOUNS), len(VERBS)) / (len(NOUNS) + len(VERBS)):.2f}\n")

    print(f"  {'arm':>14} {'noun gap':>10} {'verb gap':>10} {'accuracy':>10} "
          f"{'stability':>10}")
    out = {}
    for label, asym in (("asymmetric", True), ("CONTROL flat", False)):
        r = [arm(asym, s, sentences) for s in seeds]
        ng = statistics.mean(x[0] for x in r)
        vg = statistics.mean(x[1] for x in r)
        acc = statistics.mean(x[2] for x in r)
        lvl = statistics.mean(x[3] for x in r)
        out[label] = (ng, vg, acc, lvl)
        print(f"  {label:>14} {ng:>10.3f} {vg:>10.3f} {acc:>10.3f} "
              f"{lvl:>10.3f}", flush=True)

    print(f"\n  E3 -- gap vs number of distinct complements (asymmetric arm)")
    print(f"  {'complements':>12} {'noun gap':>10}")
    e3 = []
    for c in (1, 2, 4):
        r = [arm(True, s, sentences, complements=c) for s in seeds]
        g = statistics.mean(x[0] for x in r)
        e3.append(g)
        print(f"  {c:>12} {g:>10.3f}", flush=True)

    ng, vg, acc, lvl = out["asymmetric"]
    _cng, _cvg, cacc, clvl = out["CONTROL flat"]
    base = max(len(NOUNS), len(VERBS)) / (len(NOUNS) + len(VERBS))

    print("\n  READING\n")
    e1 = ng > 0 and vg > 0
    print(f"    E1 both classes lean the right way:  {str(e1):>5}   "
          f"noun {ng:+.3f}, verb {vg:+.3f}")
    e2 = acc > base
    print(f"    E2 accuracy beats majority:          {str(e2):>5}   "
          f"{acc:.3f} vs {base:.2f}")
    e3ok = e3[-1] > e3[0]
    print(f"    E3 gap grows with complements:       {str(e3ok):>5}   "
          f"{e3[0]:.3f} -> {e3[-1]:.3f}")
    ctrl = cacc <= base
    print(f"    CONTROL flat arm stays at baseline:  {str(ctrl):>5}   "
          f"{cacc:.3f} vs {base:.2f}")
    sat = lvl > 0.75
    print(f"    stability is SATURATED (>0.75):       {str(sat):>5}   "
          f"asym {lvl:.3f}, control {clvl:.3f}")

    print()
    if sat:
        print("    STABILITY IS SATURATED IN BOTH AREAS, so the gap above is a")
        print("    difference between two numbers that are both near ceiling.")
        print("    PHON keeps firing through the recurrent rounds AND is one of")
        print("    the strengthened fibers, so it can pin the winners on its")
        print("    own -- the probe would then read 'stable' for a word that")
        print("    was never learned. Fix the PROBE before reading E1-E3: drop")
        print("    PHON after the first round, or lower its p, and re-check that")
        print("    an untrained word reads LOW. A saturated metric's null is not")
        print("    a null (the repo's N400 lesson, #28).")
        print()
    if not ctrl:
        print("    THE CONTROL FAILED. With every fiber at the same p and beta")
        print("    there is no modality asymmetry to carry a noun/verb split,")
        print("    so an above-baseline reading there means the score is coming")
        print("    from somewhere else -- most likely a regularity in how words")
        print("    are indexed, since NOUNS occupy the low indices. E1-E3 are")
        print("    VOID until that is explained; do not report them.")
    elif e1 and e2:
        print("    The split is EMERGENT here: no label anywhere, and removing")
        print("    the modality asymmetry removes the effect. That is the")
        print("    mechanism grounding.py cites but does not implement.")
    else:
        print("    The asymmetry alone does not produce the split at this size.")
        print("    Before concluding anything about the substrate, check that")
        print("    the LEX assemblies are forming at all (Property 1) -- a")
        print("    stability of ~1.0 in BOTH areas means the probe is measuring")
        print("    a fixed point that PHON alone determines, and a stability of")
        print("    ~0 in both means nothing was learned.")


if __name__ == "__main__":
    main()

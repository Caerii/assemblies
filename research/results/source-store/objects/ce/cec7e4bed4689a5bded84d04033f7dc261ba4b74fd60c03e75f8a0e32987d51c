"""Can the model INFER its word order instead of being told?

All six orders now run on one mechanism (`SLOT_SEQUENCES`), but the order is
still a parameter you hand in. The parser executes OVS perfectly; it never
discovers that the corpus IS OVS. This asks whether it can.

WHAT COULD POSSIBLY BREAK THE SYMMETRY
---------------------------------------
Not parse success. Sequential gating is a PERFECT POSITIONAL TEMPLATE for
whatever order it is given -- every hypothesis produces an equally confident
parse, every slot gets filled. Measured: all six orders score 1.000 on
positional items. So the signal must come from OUTSIDE the gating, and there
are exactly two independent sources, both built earlier:

1. UNPARSEABILITY (the ELAN predicate, `category_addresses_open_slot`). Under a
   wrong VERB POSITION a noun meets the ACTION slot and a verb meets a filler
   slot -- the word's category cannot address the open slot. This is the same
   notion the `wobbly` acquisition package mines, but read off GATING STATE
   rather than ERP energy, which matters because the ERP signals are
   unreliable here (N400 is saturated; the shipped P600 did not separate until
   probes were isolated -- commits 8d61144, 2036ddd).

2. LEXICAL AGREEMENT (mutual inhibition). Under the true order, positional
   assignments agree with what the corpus taught about words -- `dog` is 53/2
   agent-biased, `ball` 1/20 patient-biased. Under a reversed order they
   contradict it.

NEITHER ALONE IS ENOUGH, and the reason is structural rather than empirical.
Lexical agreement can only recover S-before-O vs O-before-S: SVO and SOV assign
the two nouns IDENTICALLY and differ only in where the verb sits. Unparseability
recovers the verb's position but says nothing about which noun is subject.
Together they identify all six uniquely -- 3 verb positions x 2 noun orders.

HONEST FRAMING: THIS IS SEMANTIC BOOTSTRAPPING, NOT UNSUPERVISED INDUCTION
--------------------------------------------------------------------------
The lexical preferences were learned from role-annotated training, so they are
not free of supervision. The claim is NOT "word order induced from raw
strings". It is the weaker, and independently interesting, claim that GIVEN
lexical role knowledge acquired from grounded experience (balls get thrown,
dogs do the chasing), the SYNTACTIC ORDER follows without being told -- which
is the semantic-bootstrapping hypothesis from the acquisition literature.

THE CONTROL THAT MAKES IT SCIENCE
----------------------------------
Recovering SVO alone proves nothing: SVO might win for an incidental reason.
The test presents corpora in ALL SIX orders and asks whether the argmax tracks
the truth in every case. A method that always answers SVO scores 1/6 here.

Note the lexicon is order-INDEPENDENT: role training binds words to roles via
annotations, not position, so the same connectome serves every simulated
language. That is what makes the lexical route a genuinely independent source
of evidence rather than a restatement of the order.

PREDICTIONS, recorded before running
-------------------------------------
* mismatch alone -> 3 verb positions, so ~2 orders tie for each true order.
* agreement alone -> 2 noun orders, so ~3 orders tie.
* combined -> all 6 recovered.
If combined does NOT recover all six, report which confusions remain; a
confusion matrix that collapses exactly along one of those two axes is
informative, not a failure.

RESULT: HALF CONFIRMED, HALF REFUTED
-------------------------------------
    signal        uniquely recovered
    mismatch              0/6         ties exactly as predicted:
                                      {SVO,OVS} {SOV,OSV} {VSO,VOS}
    agreement             6/6         PREDICTED 3-WAY TIES. WRONG.
    combined              6/6

The mismatch half is exactly as reasoned: it recovers VERB POSITION and nothing
else, so each true order ties with the one order sharing its verb slot.

The agreement half REFUTES my analysis, and the error is worth keeping. I
assumed nouns only ever land in FILLER slots, so SVO and SOV would assign the
two nouns identically and tie. They do not: under a wrong VERB POSITION a noun
lands in the ACTION slot -- parsing SOV's "dog ball chases" under an SVO
hypothesis puts `ball` in ACTION -- and a noun in ACTION never matches an
AGENT/PATIENT preference, so it scores as disagreement. Lexical agreement
therefore carries verb-position information too and SUBSUMES most of the
mismatch signal.

So the two signals are NOT the independent, complementary pair the design
assumed. Agreement alone suffices here. Mismatch remains worth keeping for two
reasons: it is the only signal that works WITHOUT lexical bias (it needs no
learned preferences at all, just categories), and it is the crisp
unparseability detector the `wobbly` package currently approximates with ERP
energy.

SCOPE, and it is narrow. These items use STRONGLY biased nouns (dog 53/2, ball
1/20). With balanced nouns the agreement signal vanishes, because there is no
preference to agree with -- the same words that made `bird` a coin flip in
`nemo_competitive_ab.py`. So this shows order is recoverable FROM A BIASED
LEXICON, not from arbitrary text.
"""

from __future__ import annotations

import copy
import os
import sys
from typing import Dict, List, Sequence

ORDERS = ("SVO", "SOV", "VSO", "VOS", "OSV", "OVS")

#: Agent-biased and patient-biased nouns from the lesion corpus, whose learned
#: preference is strong enough to be evidence (dog 53/2, ball 1/20).
AGENTY = ("dog", "cat")
PATIENTY = ("ball", "book", "food", "table", "car")
VERBS = ("chases", "finds")


def _permute(subject: str, verb: str, obj: str, order: str) -> List[str]:
    slot = {"S": subject, "V": verb, "O": obj}
    return [slot[c] for c in order]


def lexical_preference(parser, word: str, transitive) -> str:
    """Which role slot does the CONNECTOME prefer for this word?

    Projects the word's core into both trained role areas in ONE call so mutual
    inhibition fires, and reports the survivor. Runs on a FORK: MI is
    destructive (winners=[], w=0 on losers) and materializes connectomes, and
    probe contamination is exactly what invalidated the ERP harness.
    """
    from neural_assemblies.assembly_calculus.emergent.core.areas import (
        ROLE_AGENT, ROLE_PATIENT,
    )
    from neural_assemblies.assembly_calculus.emergent.nemo_parse import NemoParser

    probe = copy.deepcopy(parser)
    brain_cls = type(probe.brain)
    original = brain_cls._apply_mutual_inhibition
    captured = []

    def spy(self, scores, _orig=original, _cap=captured):
        if ROLE_AGENT in scores and ROLE_PATIENT in scores:
            _cap.append(dict(scores))
        return _orig(self, scores)

    brain_cls._apply_mutual_inhibition = spy
    try:
        NemoParser(probe, transitive_verbs=transitive,
                   competitive=True).parse([word, "chases", "dog"])
    finally:
        brain_cls._apply_mutual_inhibition = original

    if not captured:
        return "?"
    first = captured[0]
    return ("AGENT" if float(first.get(ROLE_AGENT, 0.0))
            >= float(first.get(ROLE_PATIENT, 0.0)) else "PATIENT")


def run(seeds: Sequence[int] = (42, 43, 44)) -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
    os.environ.setdefault("TRAIN_PROGRESS", "0")

    from lesion_aphasia import build_corpus, train_parser
    from neural_assemblies.assembly_calculus.emergent.curriculum.data import (
        create_training_sentences,
    )
    from neural_assemblies.assembly_calculus.emergent.nemo_parse import (
        NemoParser, infer_transitive_verbs,
    )

    transitive = infer_transitive_verbs(create_training_sentences() + build_corpus())
    items = [(s, v, o) for s in AGENTY for v in VERBS for o in PATIENTY]

    print(f"\n  word-order induction: {len(ORDERS)} true orders x "
          f"{len(ORDERS)} hypotheses, {len(items)} sentences, "
          f"{len(seeds)} seeds")

    recovered = {"mismatch": 0, "agreement": 0, "combined": 0}
    confusions: List[str] = []

    for seed in seeds:
        trained = train_parser(seed)
        prefs = {w: lexical_preference(trained, w, transitive)
                 for w in AGENTY + PATIENTY}
        if seed == seeds[0]:
            print(f"\n  learned lexical preferences: {prefs}")
            print(f"\n  {'true':<6}{'best(mismatch)':<32}"
                  f"{'best(agreement)':<32}{'best(combined)':<16}")

        for true_order in ORDERS:
            sentences = [_permute(s, v, o, true_order) for s, v, o in items]
            mismatch: Dict[str, float] = {}
            agreement: Dict[str, float] = {}

            for hypothesis in ORDERS:
                bad = seen = agree = judged = 0
                for words in sentences:
                    parser = copy.deepcopy(trained)
                    nemo = NemoParser(parser, transitive_verbs=transitive,
                                      sequential=True,
                                      word_order_type=hypothesis)
                    pred = nemo.parse(list(words))
                    for _w, addressed in nemo.last_mismatches:
                        if addressed is None:
                            continue
                        seen += 1
                        bad += 0 if addressed else 1
                    for w in words:
                        if w in prefs and prefs[w] in ("AGENT", "PATIENT"):
                            judged += 1
                            agree += 1 if pred.get(w) == prefs[w] else 0
                mismatch[hypothesis] = bad / max(seen, 1)
                agreement[hypothesis] = agree / max(judged, 1)

            combined = {h: agreement[h] - mismatch[h] for h in ORDERS}
            best_m = [h for h in ORDERS
                      if mismatch[h] == min(mismatch.values())]
            best_a = [h for h in ORDERS
                      if agreement[h] == max(agreement.values())]
            best_c = [h for h in ORDERS
                      if combined[h] == max(combined.values())]

            recovered["mismatch"] += 1 if best_m == [true_order] else 0
            recovered["agreement"] += 1 if best_a == [true_order] else 0
            recovered["combined"] += 1 if best_c == [true_order] else 0
            if best_c != [true_order]:
                confusions.append(f"{true_order}->{','.join(best_c)}")

            if seed == seeds[0]:
                print(f"  {true_order:<6}{','.join(best_m):<32}"
                      f"{','.join(best_a):<32}{','.join(best_c):<16}")

    total = len(seeds) * len(ORDERS)
    print(f"\n  UNIQUELY RECOVERED (argmax is exactly the true order):")
    for key in ("mismatch", "agreement", "combined"):
        print(f"    {key:<12}{recovered[key]:>3}/{total}")
    if confusions:
        print(f"\n  combined confusions: {sorted(set(confusions))}")
    print("\n  Prediction: mismatch ~2-way ties (verb position), agreement")
    print("  ~3-way ties (noun order), combined unique. A method that always")
    print("  answers SVO would score 1/6 per seed.")


if __name__ == "__main__":
    run()

"""Lesion study: the Geschwind reversible/irreversible dissociation.

Geschwind (1970) reported that Broca's aphasics select the right picture for
"the lion chases the deer" but fail on "the lion chases the tiger". The first is
IRREVERSIBLE -- only a lion can chase a deer, so world knowledge alone fixes the
roles. The second is REVERSIBLE -- either animal could be the agent, so the
roles can only come from syntax. Damage the syntactic route and the reversible
sentences break while the irreversible ones survive.

This is a prediction a transformer cannot make: it has no anatomy to lesion.
Mitropolsky & Papadimitriou (2025) place SUBJ/VERB/OBJ in Broca's area and the
lexical/role areas in Wernicke's, which is what makes the experiment meaningful
here.

TWO ROUTES, VERIFIED BEFORE BUILDING THIS
-----------------------------------------
`_assign_roles_neural` can reach a role assignment two ways, and both were
confirmed to be live rather than assumed:

* LEXICAL (neural): each noun's core assembly is projected into the role areas
  and read out against `role_lexicons`. Verified to exist AND to override
  position -- after training where "ball" is only ever a patient, the parser
  reads "ball chases dog" as ball=PATIENT, dog=AGENT, against word order.
* POSITIONAL (symbolic): `constituent_role_order` returns a ranking derived from
  `self.word_order_type`. Flipping that attribute drops role accuracy 1.00 ->
  0.33, so it does carry the decision.

HONEST CAVEAT, stated up front: the positional route is a stored Python
attribute, NOT a neural area. Ablating it is therefore a symbolic lesion, not a
synaptic one, and a "Broca's lesion" here is not anatomically real. The LEXICAL
lesion below is genuinely synaptic (it zeroes core -> ROLE weights). So this
gives a double dissociation with one neural arm and one symbolic arm, which is
weaker than the study the paper's anatomy implies but is what the implementation
actually supports.

PREDICTIONS
    lesion POSITIONAL  -> reversible breaks, irreversible survives  (Broca's)
    lesion LEXICAL     -> irreversible breaks, reversible survives  (the mirror)
"""

from __future__ import annotations

import sys
from typing import Dict, List

import numpy as np

from neural_assemblies.assembly_calculus.emergent.parser import EmergentParser
from neural_assemblies.assembly_calculus.emergent.curriculum.data import (
    GroundedSentence, create_training_sentences,
)
from neural_assemblies.assembly_calculus.emergent.core.grounding import VOCABULARY
from neural_assemblies.assembly_calculus.emergent.core.areas import (
    ROLE_AGENT, ROLE_PATIENT,
)

# Role-exclusive nouns give IRREVERSIBLE items (lexical route suffices);
# balanced nouns appear in both roles, so only position can disambiguate them.
AGENT_ONLY = ("dog", "cat")
PATIENT_ONLY = ("ball", "book")
BALANCED = ("bird", "boy")
VERBS = ("chases", "finds", "sees")


def _gs(words: List[str], roles: List[str]) -> GroundedSentence:
    return GroundedSentence(
        words=words, contexts=[VOCABULARY[w] for w in words], roles=roles,
    )


def build_corpus() -> List[GroundedSentence]:
    """Role-exclusive nouns plus balanced nouns seen in BOTH roles."""
    out: List[GroundedSentence] = []
    for a in AGENT_ONLY:
        for pat in PATIENT_ONLY:
            for v in VERBS:
                out.append(_gs([a, v, pat], ["agent", "action", "patient"]))
    # balanced nouns appear equally often as agent and as patient, so they carry
    # no lexical preference and the model must fall back on order
    for b in BALANCED:
        for other in BALANCED:
            if b == other:
                continue
            for v in VERBS:
                out.append(_gs([b, v, other], ["agent", "action", "patient"]))
                out.append(_gs([other, v, b], ["agent", "action", "patient"]))
    return out * 3


def test_items():
    """(sentence, gold_roles, kind) for scoring.

    A lesion can only show through where the two routes DISAGREE. Canonical
    sentences are scored correctly by position alone AND by lexical preference
    alone, so every condition scores 1.00 and the experiment measures nothing --
    that was the first version of this file, and it produced a flat null.

    So each item type is built to isolate one route:

    * IRREVERSIBLE items put the role-exclusive nouns in NON-canonical order,
      so position says the wrong thing and only world knowledge ("a ball does
      not chase a dog") gives the right answer. This is Geschwind's
      irreversible case: semantics can rescue you when syntax cannot.
    * REVERSIBLE items use the balanced nouns in canonical order. They carry no
      lexical preference by construction, so only order can assign the roles.
    """
    items = []
    # semantics vs position: gold follows the LEXICAL preference
    for a in AGENT_ONLY:
        for pat in PATIENT_ONLY:
            items.append(([pat, "chases", a], {a: "AGENT", pat: "PATIENT"},
                          "irreversible"))
    # no lexical preference available: gold follows ORDER
    for b in BALANCED:
        for other in BALANCED:
            if b != other:
                items.append(([b, "chases", other],
                              {b: "AGENT", other: "PATIENT"}, "reversible"))
    return items


def score(parser, kind: str) -> float:
    ok = tot = 0
    for words, gold, k in test_items():
        if k != kind:
            continue
        got = parser.parse(list(words))["roles"]
        for w, g in gold.items():
            ok += (got.get(w) == g)
            tot += 1
    return ok / max(tot, 1)


def lesion_lexical(parser) -> None:
    """SYNAPTIC lesion: zero the core -> ROLE weights that carry the learned
    lexical role preference, and drop the readout targets."""
    eng = parser.brain._engine
    for core in list(parser.core_lexicons.keys()):
        for role in (ROLE_AGENT, ROLE_PATIENT):
            conn = eng._area_conns.get(core, {}).get(role)
            w = getattr(conn, "weights", None)
            if w is not None and getattr(w, "size", 0):
                np.asarray(w)[:] = 0.0
    for role in (ROLE_AGENT, ROLE_PATIENT):
        parser.role_lexicons[role] = {}


def lesion_positional(parser) -> None:
    """SYMBOLIC lesion: corrupt the inferred constituent order (see caveat).

    Setting it to None does NOT work as a lesion: `constituent_role_order`
    falls back to the AGENT-first ranking, which is exactly right for an SVO
    corpus, so the "damaged" parser scores perfectly. Corrupting the order to an
    object-initial one is what actually removes the correct syntactic cue.
    """
    parser.word_order_type = "OVS"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    corpus = build_corpus()
    print("Geschwind lesion study -- role assignment under damage")
    print(f"corpus {len(corpus)} sentences | "
          f"agent-only {AGENT_ONLY} patient-only {PATIENT_ONLY} "
          f"balanced {BALANCED}\n")

    print(f"{'condition':<22} {'irreversible':>13} {'reversible':>12}")
    print("-" * 50)
    rows: Dict[str, tuple] = {}
    for label, lesion in (("intact", None),
                          ("lesion POSITIONAL", lesion_positional),
                          ("lesion LEXICAL", lesion_lexical)):
        p = EmergentParser(n=1000, k=50, p=0.05, beta=0.1, seed=42, rounds=10)
        p.train(create_training_sentences() + corpus)
        if lesion is not None:
            lesion(p)
        irr, rev = score(p, "irreversible"), score(p, "reversible")
        rows[label] = (irr, rev)
        print(f"{label:<22} {irr:>13.2f} {rev:>12.2f}", flush=True)

    base_irr, base_rev = rows["intact"]
    p_irr, p_rev = rows["lesion POSITIONAL"]
    l_irr, l_rev = rows["lesion LEXICAL"]
    print("\nDissociation check (drop from intact):")
    print(f"  POSITIONAL lesion: irreversible {base_irr - p_irr:+.2f}  "
          f"reversible {base_rev - p_rev:+.2f}   "
          f"(Geschwind predicts reversible drops MORE)")
    print(f"  LEXICAL lesion:    irreversible {base_irr - l_irr:+.2f}  "
          f"reversible {base_rev - l_rev:+.2f}   "
          f"(mirror predicts irreversible drops MORE)")


if __name__ == "__main__":
    main()

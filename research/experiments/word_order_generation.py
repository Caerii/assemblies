"""Reproduce the paper's word-order / generation experiment -- the task NEMO
is actually built for.

Mitropolsky & Papadimitriou (2025), "Simulated Language Acquisition", Sec 2.4-2.5.
The published model does NOT do next-token prediction from a word prefix. Its
task is: present a SCENE (assemblies in the three ROLE areas) plus a MOOD, and
generate the sentence with the correct CONSTITUENT ORDER. Quoting the paper:

    "we test, upon the presentation of a randomly generated novel scene (e.g.,
    a dog eating a cookie, presented as three assemblies in the three ROLE
    areas), whether the device will generate the appropriate sentence with the
    correct word order"

    "Success ... is defined as the successful generation of a transitive
    sentence and another intransitive sentence that are both sampled randomly
    and withheld during training."

Two consequences for how this should be measured, both of which we got wrong
before reading the paper:

1. The decision is 3-way (which ROLE area fires next), not |V|-way (which word
   comes next). Word order lives in ``SYN[i] -> ROLE[i+1]`` synapses; the
   readout is an argmax of total synaptic input across the role areas, not an
   overlap-readout against a lexicon.
2. The headline metric is SAMPLE EFFICIENCY -- "how many sentence presentations
   are needed" before the withheld scene generates in the right order.

THE CONTROL THAT MATTERS.  A parser with no order training at all still emits
SVO (there is a default ordering), so "generates SVO" is NOT evidence of
learning. The real test is whether the generated order FOLLOWS THE TRAINED
ORDER. We therefore train separate parsers on SVO, SOV and VSO corpora built by
permuting the same sentences, and check each reproduces its own order on a
withheld scene. Measured: SVO->SVO, SOV->SOV, VSO->VSO.
"""

import sys

from neural_assemblies.assembly_calculus.emergent.parser import EmergentParser
from neural_assemblies.assembly_calculus.emergent.curriculum.data import (
    GroundedSentence, create_training_sentences,
)
from neural_assemblies.assembly_calculus.emergent.core.areas import (
    ROLE_ACTION, ROLE_AGENT, ROLE_PATIENT,
)

ORDERS = {
    "SVO": ("agent", "action", "patient"),
    "SOV": ("agent", "patient", "action"),
    "VSO": ("action", "agent", "patient"),
    "OVS": ("patient", "action", "agent"),
}


def transitive_sentences():
    return [s for s in create_training_sentences()
            if s.roles and "patient" in (s.roles or [])]


def reorder(sent, order):
    """Rewrite a sentence's content words into `order` (drops determiners)."""
    trip = {r: (w, c)
            for w, c, r in zip(sent.words, sent.contexts, sent.roles) if r}
    if len(trip) < 3:
        return None
    ws, cs, rs = [], [], []
    for role in order:
        w, c = trip[role]
        ws.append(w); cs.append(c); rs.append(role)
    return GroundedSentence(words=ws, contexts=cs, roles=rs)


def run_order(name, order, *, repetitions=4, seed=42, verbose=True):
    """Train on one word order; generate a WITHHELD scene. Returns produced order."""
    corpus = [x for x in (reorder(s, order) for s in transitive_sentences()) if x]
    held = corpus[2]
    train = [s for s in corpus if s is not held]
    hw = {r: w for w, r in zip(held.words, held.roles)}

    parser = EmergentParser(n=1000, k=50, p=0.05, beta=0.1, seed=seed, rounds=10)
    parser.train(create_training_sentences())          # vocabulary acquisition
    parser.train_constituent_order(train, repetitions=repetitions)

    parser.prepare_scene({
        ROLE_AGENT: hw["agent"],
        ROLE_ACTION: hw["action"],
        ROLE_PATIENT: hw["patient"],
    })
    out = parser.generate_from_roles(max_len=6)
    sym = {hw["agent"]: "S", hw["action"]: "V", hw["patient"]: "O"}
    produced = "".join(sym.get(w, "?") for w in out)
    if verbose:
        ok = "MATCHES" if produced.startswith(name) else "MISMATCH"
        print(f"  trained {name}: withheld scene -> {out} = '{produced}'  {ok}")
    return produced


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("Word-order generation on a WITHHELD scene "
          "(Mitropolsky & Papadimitriou 2025, Sec 2.4-2.5)\n")
    print("A no-training parser still emits SVO by default, so the test is "
          "whether\ngeneration follows the TRAINED order:\n")
    results = {name: run_order(name, order) for name, order in ORDERS.items()}
    hits = sum(1 for n, p in results.items() if p.startswith(n))
    print(f"\n{hits}/{len(results)} word orders reproduced from a withheld scene")


if __name__ == "__main__":
    main()

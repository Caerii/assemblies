"""Sample-efficiency curve for word-order acquisition -- the paper's headline metric.

Mitropolsky & Papadimitriou (2025), Sec 2.5:

    "We determine how many sentence presentations are needed in order for the
    model to be able to generate sentences with the correct word order in each
    mood, for randomly chosen subject, verb, and object, as a function of the
    size of the lexicon and the number of moods. We withhold one particular
    transitive sentence ... from the training set, and test after each training
    sentence whether the model generates the withheld sentence in the correct
    order."

    Figure 5b: the required number of presentations "appears to grow linearly
    with the number of moods, and with the size of the lexicon".

So the measurement is SENTENCES-TO-CRITERION on a WITHHELD scene, swept over
lexicon size. That is the quantity directly comparable to a gradient-trained
baseline's sample efficiency.

Two design decisions that make the number meaningful
----------------------------------------------------
1. RESOLUTION. An untrained parser already emits SVO (there is a default
   ordering), so an SVO curve starts at "already correct" and measures nothing.
   The real curves are the NON-DEFAULT orders (SOV / VSO / OVS), where being
   correct requires the SYN[i] -> ROLE[i+1] synapses to actually override the
   default. SVO is still reported, explicitly labelled as confounded.
2. PROBE PURITY. Probing must not train the model, or the probes themselves
   become a training signal and the curve is meaningless. Verified: 8 repeated
   probes give identical answers with no substrate growth (generation runs under
   ``brain.frozen()``, and unlike scoring it does not materialise new neurons).

Criterion: first presentation count after which the withheld scene generates in
the trained order for ``STABLE_WINDOW`` consecutive probes (a single lucky hit
does not count).
"""

import copy
import random
import statistics
import sys
from typing import Dict, List, Optional

from neural_assemblies.assembly_calculus.emergent.parser import EmergentParser
from neural_assemblies.assembly_calculus.emergent.curriculum.data import (
    VOCABULARY, GroundedSentence, create_training_sentences,
)
from neural_assemblies.assembly_calculus.emergent.core.areas import (
    ROLE_ACTION, ROLE_AGENT, ROLE_PATIENT,
)

ORDERS: Dict[str, tuple] = {
    "SVO": ("agent", "action", "patient"),   # confounded by the default order
    "SOV": ("agent", "patient", "action"),
    "VSO": ("action", "agent", "patient"),
    "OVS": ("patient", "action", "agent"),
}

NOUNS = ["dog", "cat", "bird", "boy", "girl", "ball", "book", "car",
         "food", "table"]
VERBS = ["chases", "sees", "finds", "reads", "eats", "plays"]

MAX_SENTENCES = 80
STABLE_WINDOW = 3


def make_sentence(agent: str, action: str, patient: str, order) -> GroundedSentence:
    """A grounded transitive sentence with its content words in `order`."""
    filler = {"agent": agent, "action": action, "patient": patient}
    words = [filler[r] for r in order]
    return GroundedSentence(
        words=words,
        contexts=[VOCABULARY[w] for w in words],
        roles=list(order),
    )


def sentences_to_criterion(
    base: EmergentParser, order_name: str, order, lexicon_size: int, seed: int,
) -> Optional[int]:
    """Present sentences one at a time; return the count at which the withheld
    scene first generates in `order_name` and stays correct, or None."""
    rng = random.Random(seed)
    nouns = NOUNS[:lexicon_size]
    verbs = VERBS[:max(2, lexicon_size // 2)]

    # Withhold one scene entirely -- it is never trained on.
    held = (rng.choice(nouns), rng.choice(verbs), rng.choice(nouns))
    sym = {held[0]: "S", held[1]: "V", held[2]: "O"}

    parser = copy.deepcopy(base)          # shares vocabulary acquisition
    streak = 0
    for n in range(1, MAX_SENTENCES + 1):
        while True:                        # sample a training scene != withheld
            scene = (rng.choice(nouns), rng.choice(verbs), rng.choice(nouns))
            if scene != held:
                break
        parser.train_constituent_order(
            [make_sentence(*scene, order)], repetitions=1)

        parser.prepare_scene({ROLE_AGENT: held[0], ROLE_ACTION: held[1],
                              ROLE_PATIENT: held[2]})
        out = parser.generate_from_roles(max_len=6)
        produced = "".join(sym.get(w, "?") for w in out)
        streak = streak + 1 if produced.startswith(order_name) else 0
        if streak >= STABLE_WINDOW:
            return n - STABLE_WINDOW + 1
    return None


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    seeds = [1, 2, 3, 4, 5]
    sizes = [4, 6, 8, 10]

    print("Sentences-to-criterion on a WITHHELD scene "
          "(Mitropolsky & Papadimitriou 2025, Fig 5b)")
    print(f"criterion: correct for {STABLE_WINDOW} consecutive probes; "
          f"cap {MAX_SENTENCES}; {len(seeds)} seeds (median [min-max])\n")

    base = EmergentParser(n=1000, k=50, p=0.05, beta=0.1, seed=42, rounds=10)
    base.train(create_training_sentences())      # vocabulary acquisition, once

    header = "  " + "".join(f"{('lex=' + str(s)):>16}" for s in sizes)
    for name, order in ORDERS.items():
        tag = "  (confounded: default order)" if name == "SVO" else ""
        print(f"{name}{tag}")
        print(header)
        row = "  "
        for size in sizes:
            vals = [sentences_to_criterion(base, name, order, size, sd)
                    for sd in seeds]
            got = [v for v in vals if v is not None]
            if not got:
                row += f"{'never':>16}"
            else:
                med = int(statistics.median(got))
                miss = len(vals) - len(got)
                cell = f"{med} [{min(got)}-{max(got)}]"
                if miss:
                    cell += f" {miss}x"
                row += f"{cell:>16}"
        print(row + "\n")


if __name__ == "__main__":
    main()

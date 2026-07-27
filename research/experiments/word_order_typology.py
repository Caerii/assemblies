"""Test the paper's typological predictions about constituent order.

Mitropolsky & Papadimitriou (2025), Fig 5c and surrounding text, make unusually
sharp falsifiable claims:

  H1  "The algorithm succeeds for any order if more transitive sentences are
       presented than intransitive."
  H2  "significantly more sentences are needed to learn OSV and OVS."
  H3  "If the proportion of transitive sentences is less than half ... the
       orders VOS, OSV, or OVS cannot be learned by this system even with
       unbounded input."
  H4  "these are in fact the three rarest constituent orders among world
       languages" -- so the mechanism PREDICTS typological frequency.

The intransitive corpus follows the paper's own observation that "in attested
languages with a rigid or default word order, the intransitive order is almost
always the same ordering with O omitted" (SOV -> SV).

DERIVED CONFLICT STRUCTURE (why some orders should be harder). Learning stores
MOOD -> first constituent and SYN[i] -> next constituent, so:

    SVO  S->V->O / S->V   no clash: intransitive is a strict PREFIX
    VSO  V->S->O / V->S   no clash: strict prefix
    SOV  S->O->V / S->V   TRANSITION clash  SUBJ->{O vs V}
    VOS  V->O->S / V->S   TRANSITION clash  VERB->{O vs S}
    OSV  O->S->V / S->V   FIRST-WORD clash  MOOD->{O vs S}
    OVS  O->V->S / V->S   FIRST-WORD clash  MOOD->{O vs V}

Only SVO and VSO are conflict-free, which is the same conclusion
`parser_mixins/constituent_order.py::_frame_assembly` reached independently.

REPRODUCIBILITY.  This grid was originally unreproducible: global RNG leaked
between Brain constructions and flipped exactly the borderline orders (identical
config and seed, four builds in one process, gave OVS,VSO,OVS,VSO). That is now
fixed at source -- `Brain(seed=)` draws wiring from dedicated seeded streams --
so `_reseed` below is belt-and-braces rather than load-bearing. Keep it: it also
pins torch's global generator, which the GPU sampler still uses.

RESULT (5 seeds, 60 sentences, deterministic substrate)::

    order      30%    50%    70%    90%
    SVO   *    5/5   5/5   5/5   5/5
    VSO   *    5/5   5/5   5/5   5/5
    SOV        3/5   5/5   4/5   5/5
    VOS        3/5   4/5   4/5   5/5
    OSV        4/5   3/5   4/5   4/5
    OVS        3/5   2/5   2/5   2/5

* CONFIRMED, the conflict derivation above: SVO and VSO -- the only orders whose
  intransitive chain is a strict prefix of the transitive one -- are perfect in
  every cell.
* CONFIRMED qualitatively, H4: the two OBJECT-INITIAL orders are hardest, and
  those are the two rarest in WALS (OVS ~1%, OSV ~0.3%).
* NOT CONFIRMED, H3: there is no cliff below 50% transitive (OSV scores 4/5 at
  30%), and VOS behaves like SOV rather than like the other two orders the paper
  groups it with.
* ANOMALY: OVS gets WORSE with more transitive input (3,2,2,2), the opposite of
  H1's direction. The one cell that contradicts rather than merely underperforms.
* CAVEAT: the typology fit is partial. VSO is rated as easy as SVO though it is
  ~7% of languages, and SOV harder than VSO though SOV is the most common order
  (~41%). Object-initial rarity is predicted; the rest of the ranking is not.
"""

from __future__ import annotations

import random
import sys
from typing import Dict, List, Sequence

import numpy as np

from neural_assemblies.reference.word_order_learner import (
    WordOrderLearner, MOOD, PHON, TPJ, HELPER, CONSTITUENTS,
)

ORDERS: Dict[str, Sequence[str]] = {
    "SVO": ("S", "V", "O"), "SOV": ("S", "O", "V"),
    "VSO": ("V", "S", "O"), "VOS": ("V", "O", "S"),
    "OSV": ("O", "S", "V"), "OVS": ("O", "V", "S"),
}
CONFLICT_FREE = ("SVO", "VSO")


def _reseed(seed: int) -> None:
    """Pin every global RNG the stack may consume. See module docstring."""
    np.random.seed(seed)
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


class TransIntransLearner(WordOrderLearner):
    """Word-order learner that also trains INTRANSITIVE sentences (order minus O)."""

    def intransitive_order(self, mood_index: int = 0):
        return tuple(c for c in self.mood_orders[mood_index] if c != "O")

    def train_sentence_intrans(self, mood_index: int = 0) -> None:
        subj = self._rng.randrange(self.num_nouns)
        verb = self._rng.randrange(self.num_nouns, self.num_words)
        self.brain.activate(MOOD, mood_index)
        for c, idx in (("S", subj), ("V", verb)):
            self._activate_role(idx, c)
        prev = None
        for pos, c in enumerate(self.intransitive_order(mood_index)):
            for t in range(self.training_fire_rounds):
                self._project_training(c, t, first_word=(pos == 0), previous=prev)
            prev = c

    def train_mixed(self, n_sentences: int, transitive_fraction: float,
                    mood_index: int = 0) -> None:
        for _ in range(n_sentences):
            if self._rng.random() < transitive_fraction:
                self.train_sentence(mood_index)
            else:
                self.train_sentence_intrans(mood_index)

    def generate_scene(self, roles: List[str], mood_index: int = 0,
                       firings: int = 3) -> List[str]:
        """Generate with only `roles` present -- an intransitive scene has no O."""
        self._mood_now = mood_index
        idxs = {"S": self._rng.randrange(self.num_nouns),
                "O": self._rng.randrange(self.num_nouns),
                "V": self._rng.randrange(self.num_nouns, self.num_words)}
        with self.brain.frozen():
            self.brain.activate(MOOD, mood_index)
            for c in roles:
                self._activate_role(idxs[c], c, firings=firings)
            current = self._strongest(MOOD, roles)
            self.brain.project({}, {MOOD: [HELPER[current]]})
            out = [current]
            for _ in range(len(roles) - 1):
                syn = self._syn(current)
                self.brain.project({}, {HELPER[current]: [syn], MOOD: [syn]})
                rem = [c for c in roles if c not in out]
                self.brain.project({}, {syn: [HELPER[c] for c in rem],
                                        **{TPJ[c]: [HELPER[c]] for c in roles}})
                current = self._strongest(syn, rem)
                out.append(current)
        return out


def trial(order_name: str, transitive_fraction: float, n_sentences: int,
          seed: int):
    """Returns (transitive_ok, intransitive_ok) on WITHHELD scenes."""
    _reseed(seed)
    order = tuple(ORDERS[order_name])
    m = TransIntransLearner(
        num_nouns=4, num_verbs=2, mood_orders={0: order},
        n=1000, k=50, p=0.05, beta=0.06, seed=seed,
    )
    m.train_mixed(n_sentences, transitive_fraction)
    t_ok = tuple(m.generate_scene(list(CONSTITUENTS))) == order
    intr = m.intransitive_order()
    i_ok = tuple(m.generate_scene(list(intr))) == intr
    return t_ok, i_ok


def _wilson(k: int, n: int, z: float = 1.96) -> tuple:
    """(centre, half-width) Wilson score interval for a binomial proportion.

    Wilson rather than the normal approximation because these cells sit near
    0 and 1, where the normal interval is badly wrong (it gives zero width at
    p=0 and can run outside [0,1]). "5/5" is not evidence of a perfect order;
    Wilson says 5/5 is [0.57, 1.00], which is the honest statement.
    """
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = (z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)) / d
    return centre, half


def main(seeds: Sequence[int] = tuple(range(1, 16))) -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    seeds = list(seeds)
    fracs = (0.3, 0.5, 0.7, 0.9)
    n = len(seeds)
    print("Constituent-order typology (Mitropolsky & Papadimitriou 2025, Fig 5c)")
    print(f"success = correct TRANSITIVE and INTRANSITIVE generation on withheld "
          f"scenes; {n} seeds, 60 sentences, globals reseeded per trial")
    print("cells are successes/n with a 95% Wilson interval -- at these n the "
          "intervals are WIDE, and\nthat is the point: a 2/5 vs 4/5 difference "
          "in the older 5-seed grid was not resolvable.\n")
    print(f"{'order':<6}{'':<2}" + "".join(f"{int(f*100):>17}% " for f in fracs))
    print("-" * 82)
    for name in ORDERS:
        mark = "*" if name in CONFLICT_FREE else " "
        row = f"{name:<6}{mark:<2}"
        for frac in fracs:
            ok = sum(all(trial(name, frac, 60, s)) for s in seeds)
            c, h = _wilson(ok, n)
            row += f"{ok:>3}/{n} [{max(0.0, c - h):.2f},{min(1.0, c + h):.2f}]"
        print(row, flush=True)
    print("\n* = conflict-free (intransitive chain is a prefix of the transitive)")


if __name__ == "__main__":
    main()

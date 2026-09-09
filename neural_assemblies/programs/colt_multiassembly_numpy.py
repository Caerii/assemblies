"""[COLT22] Theorems 1/3/4 -- assembly creation, recall, and MULTIPLE assemblies.

Dabagia, Papadimitriou & Vempala, "Assemblies of Neurons Learn to Classify
Well-Separated Distributions" (COLT 2022).

THE PROTOCOL IS THE PAPER'S, not ours. A stimulus class ``A`` is a DISTRIBUTION,
defined by a set ``S_A`` of ``k`` sensory neurons and two scalars ``r > q``: to
draw a stimulus, each ``i in S_A`` fires with probability ``r`` and each
``i not in S_A`` with probability ``qk/n``. An assembly is formed from a STREAM
of such samples over ``O(log k)`` rounds, and ``A*`` is the union of the caps.

This matters because our own lexicon fires ONE FIXED deterministic pattern per
word. Without sampling there is no "core set" for the theorems to talk about,
so a protocol that skips the distribution is not testing these theorems at all.

WHAT EACH THEOREM CLAIMS, and which of them our engine satisfies:

  * Theorem 1 (Creation) bounds the support, ``|A*| <= k / (1 - exp(-(b/b0)^2))``
    with ``b0`` a plasticity LOWER bound. **The bound is vacuous at every beta
    anyone runs** -- 279k for our parser, and 506k at the acquisition paper's
    own beta=0.06 -- so it is reported, not asserted. These are asymptotic
    sufficient conditions with unoptimised constants.

  * Theorem 3 (Recall): a fresh sample from the class produces a cap
    overlapping ``A*`` by at least ``1 - e^{-kpr}``. **This HOLDS here**, with
    wide margin (measured 0.998-1.000 against a 0.989 floor), so it is
    asserted. It also validates the harness: if recall fails, no Theorem 4
    number from the same run means anything.

  * Theorem 4 (Multiple Assemblies): with ``|S_A ∩ S_B| = alpha*k``, the
    assemblies preserve it -- ``|A* ∩ B*| <= alpha*k``. **This DOES NOT hold in
    our implementation.** Overlap is AMPLIFIED, with beta as the gain, and
    raising beta past ``b0`` does not restore it. That divergence is pinned by
    a test rather than asserted away; see
    ``research/notes/substrate/kwta_amplifies_input_overlap.md``.

WHY THE DIVERGENCE IS NOT OBVIOUSLY A BUG. Every practical beta is far below
``b0``, so the theorem's hypothesis never holds in any regime that has been
run -- including the papers' own simulations. "We violate Theorem 4" would be
an overclaim; "the guarantee is not in force and the measured behaviour is
amplification" is the honest statement.

ENGINE IS PINNED to ``numpy_exact``. The sampler on ``numpy_sparse`` INVENTS
drive for neurons that have not fired, so an overlap measurement there is
partly a property of the sampler.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Dict, Sequence, Tuple

import numpy as np

SENSE, TARGET = "SENSE", "B"


@dataclasses.dataclass(frozen=True)
class MultiAssemblyResult:
    """One run of the Theorem 1/3/4 protocol."""

    parameters: Dict[str, Any]
    #: Theorem 3: mean overlap of a fresh sample's cap with A*.
    recall: float
    #: The theorem's own floor, 1 - e^{-kpr}.
    recall_floor: float
    #: alpha values swept, and the measured |A* ∩ B*| / k at each.
    alphas: Tuple[float, ...]
    overlaps: Tuple[float, ...]
    #: |A*||B*|/n / k -- what two INDEPENDENT supports share by default. The
    #: theorem's bound demands exactly 0 at alpha=0, which no finite substrate
    #: can deliver, so alpha=0 must be scored against this and not against 0.
    chance: Tuple[float, ...]
    #: |A*| / k, Theorem 1's support inflation.
    supports: Tuple[float, ...]
    #: Theorem 1's beta lower bound for these parameters.
    beta0: float


def beta0(n: int, k: int, p: float, r: float) -> float:
    """Theorem 1's plasticity lower bound.

    ``(1/r^2) * [sqrt(2-r^2)*sqrt(2 ln(n/k)) + sqrt(6)] / [sqrt(kp) +
    sqrt(2 ln(n/k))]``.
    """
    lg = math.sqrt(2.0 * math.log(n / k))
    return (1.0 / r ** 2) * ((math.sqrt(2.0 - r * r) * lg + math.sqrt(6.0))
                             / (math.sqrt(k * p) + lg))


def support_bound(k: int, beta: float, b0: float) -> float:
    """Theorem 1's support bound ``k / (1 - exp(-(beta/beta0)^2))``."""
    x = 1.0 - math.exp(-((beta / b0) ** 2))
    return k / x if x > 0 else float("inf")


def _sample(core: np.ndarray, n: int, k: int, rng, r: float, q: float):
    """Draw one stimulus from the class: i in S_A w.p. r, else w.p. qk/n."""
    on = core[rng.random(core.size) < r]
    if q > 0:
        outside = rng.random(n) < (q * k / n)
        outside[core] = False
        on = np.union1d(on, np.flatnonzero(outside))
    return on.astype(np.int64)


def _read(brain, area: str) -> np.ndarray:
    from neural_assemblies.assembly_calculus.ops import _snap
    return np.asarray(_snap(brain, area).winners, dtype=np.int64)


def _present(brain, idx) -> None:
    area = brain.areas[SENSE]
    area.winners = np.asarray(idx, dtype=np.int64)
    area.fix_assembly()


def _build(brain, core, n, k, rng, r, q, rounds):
    union = set()
    for _ in range(rounds):
        _present(brain, _sample(core, n, k, rng, r, q))
        brain.project({}, {SENSE: [TARGET]})
        union.update(int(x) for x in _read(brain, TARGET))
    return union


def _recall(brain, core, n, k, rng, r, q, star) -> float:
    with brain.read_only():
        _present(brain, _sample(core, n, k, rng, r, q))
        brain.project({}, {SENSE: [TARGET]})
        cap = set(int(x) for x in _read(brain, TARGET))
    return len(cap & star) / max(len(cap), 1)


def run_colt_multiassembly(
    *,
    n: int = 10000,
    k: int = 100,
    p: float = 0.05,
    beta: float = 0.10,
    r: float = 0.9,
    q: float = 0.0,
    rounds: int = 10,
    alphas: Sequence[float] = (0.0, 0.25, 0.5),
    seeds: Sequence[int] = (42, 7),
) -> MultiAssemblyResult:
    """Run the Theorem 1/3/4 protocol and return the measured quantities."""
    from neural_assemblies.core.brain import Brain

    recalls, ov, ch, sup = [], [], [], []
    for alpha in alphas:
        per_alpha_ov, per_alpha_ch, per_alpha_sup = [], [], []
        for seed in seeds:
            rng = np.random.default_rng(seed)
            brain = Brain(p=p, seed=seed, engine="numpy_exact", norm_init=True)
            brain.add_area(SENSE, n, k, beta=beta)
            brain.add_area(TARGET, n, k, beta=beta)

            shared = int(round(alpha * k))
            pool = rng.permutation(n)
            core_a = np.sort(pool[:k])
            core_b = np.sort(np.concatenate(
                [core_a[:shared], pool[k:2 * k - shared]]))

            a_star = _build(brain, core_a, n, k, rng, r, q, rounds)
            recalls.append(_recall(brain, core_a, n, k, rng, r, q, a_star))
            b_star = _build(brain, core_b, n, k, rng, r, q, rounds)
            recalls.append(_recall(brain, core_b, n, k, rng, r, q, b_star))

            per_alpha_ov.append(len(a_star & b_star) / k)
            per_alpha_ch.append((len(a_star) * len(b_star) / n) / k)
            per_alpha_sup.append(len(a_star) / k)
        ov.append(float(np.mean(per_alpha_ov)))
        ch.append(float(np.mean(per_alpha_ch)))
        sup.append(float(np.mean(per_alpha_sup)))

    return MultiAssemblyResult(
        parameters={"n": n, "k": k, "p": p, "beta": beta, "r": r, "q": q,
                    "rounds": rounds, "seeds": list(seeds),
                    "engine": "numpy_exact"},
        recall=float(np.mean(recalls)),
        recall_floor=1.0 - math.exp(-k * p * r),
        alphas=tuple(float(a) for a in alphas),
        overlaps=tuple(ov),
        chance=tuple(ch),
        supports=tuple(sup),
        beta0=beta0(n, k, p, r),
    )

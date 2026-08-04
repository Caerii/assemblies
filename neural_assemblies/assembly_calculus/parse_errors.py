"""The parser paper's OWN violation detectors: try-project, empty-project, nonsense-assembly.

Mitropolsky, Collins & Papadimitriou (TACL 2021) do not score syntactic
violations with a graded magnitude. They report two STRUCTURAL signals, both
categorical, and one primitive that underpins both:

  * ``try_project(x, B)`` -- fire x into B and accept the result ONLY if the
    k-cap is stable under a repeated firing. If it is not stable it is not an
    assembly. This is the paper's own suggested neural implementation, and it is
    what makes the readout able to say "there is nothing here".

  * EMPTY-PROJECT -- after an intransitive verb, OBJ was never disinhibited, so
    an incoming noun causes ``project*`` to fire NO assemblies. The violation is
    an event, not a small number.

  * NONSENSE-ASSEMBLY -- project an area into LEX; if the resulting cap
    corresponds to no word in the lexicon, the state of the graph could not have
    come from a legal sentence.

WHY THIS MODULE EXISTS. Our shipped P600 detector is ``1 - normalized_energy``
compared against a fixed margin. Measured, the raw quantity occupies 0.7% of
[0,1] in every condition and the margin sits 11.9x above the largest excess ever
observed, so the detector cannot fire -- see research/notes/erp_metric_is_clipped.md
and #104. A continuous quantity acquired a scale, the scale changed, and the
threshold did not. The three signals here cannot suffer that failure: two are
boolean and the third is a set comparison.

THE THRESHOLDS ARE DERIVED, NEVER TUNED, and that is the point of the design.
``chance_overlap(n, k) = k / n`` is the expected overlap of a random k-cap with
a fixed one; ``nonsense_threshold`` puts a detection bar a stated number of
standard deviations above that hypergeometric null. Nothing here contains a
constant fitted to a particular brain, so nothing here can go stale when the
substrate is rescaled. That is design rule 4 from the note above -- a reference
constant must never override a data-derived threshold -- applied at construction
rather than patched afterwards.

A LIMIT WORTH STATING UP FRONT. ``empty_project`` as the paper means it needs
``project*`` to be able to fire nothing. Our k-WTA is TOTAL: ``_select_ordered_
indices`` returns ``min(k, len(values))`` winners whatever the drive, so a
zero-drive projection still yields a full assembly ([[silent-no-op-dead-fibers]]).
Until that changes, this module detects empty-project at the GATING level -- the
derived projection map is empty, so no projection is attempted -- which is
faithful to Algorithm 2 (where the map comes from the inhibition state) but does
not yet catch a projection that runs and should have produced nothing. The
distinction is recorded in ``EmptyProject.detected_by``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .assembly import Assembly, overlap

__all__ = [
    "Stability",
    "LexicalReadout",
    "EmptyProject",
    "chance_overlap",
    "nonsense_threshold",
    "try_project",
    "assembly_stability",
    "lexical_readout",
    "empty_project",
]


# ---------------------------------------------------------------------------
# Derived nulls. No constant below is fitted to a brain.
# ---------------------------------------------------------------------------

def chance_overlap(n: int, k: int) -> float:
    """Expected overlap FRACTION of a random k-cap with a fixed k-cap.

    Draw k neurons uniformly from n; the expected size of the intersection with
    a fixed k-set is k*k/n, which as a fraction of k is ``k / n``. At the
    parser's operating point (n=3000, k=30) that is 0.01, so an overlap of 0.25
    is 25x chance and an overlap of 0.02 is noise.

    This is the null a readout must beat, and it is a property of the AREA, not
    of any measurement.
    """
    if n <= 0 or k <= 0:
        return 0.0
    return min(1.0, float(k) / float(n))


def overlap_sd(n: int, k: int) -> float:
    """SD of the overlap FRACTION under the same hypergeometric null.

    The intersection size is Hypergeometric(N=n, K=k, draws=k) with variance
    ``k * (k/n) * (1 - k/n) * (n - k) / (n - 1)``. Dividing by k puts it on the
    same scale as ``chance_overlap``.
    """
    if n <= 1 or k <= 0:
        return 0.0
    p = float(k) / float(n)
    var_count = k * p * (1.0 - p) * (n - k) / (n - 1.0)
    return float(np.sqrt(max(var_count, 0.0))) / float(k)


def nonsense_threshold(n: int, k: int, z: float = 4.0) -> float:
    """Overlap below which a cap is NOT a lexical item: chance + z sigma.

    ``z=4`` is a false-positive rate, not a fitted parameter: it is the bar a
    random cap must clear by luck, and at n=3000/k=30 it puts the threshold at
    about 0.08 -- far below any real lexical match (measured 0.25+) and far
    above the 0.01 null. Callers wanting a stricter or looser detector should
    move ``z`` and say why; there is nothing else to tune.
    """
    return min(1.0, chance_overlap(n, k) + z * overlap_sd(n, k))


# ---------------------------------------------------------------------------
# try-project: the paper's stability criterion
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Stability:
    """Result of firing an assembly and firing it again WITH self-recurrence.

    Attributes:
        stable: the k-cap did not change AT ALL -- the paper's literal
            criterion ("changes with a repeated firing").
        jaccard: graded agreement between the two caps. Measured on a trained
            pathway this is 1.0000 and on an untrained one 0.3000, so the
            graded form carries real information and is worth keeping.
        pool: how many neurons were eligible to compete in the target.
        k: cap size.
        first: cap after the first firing (neuron IDs).
        second: cap after the repeated firing.
    """

    stable: bool
    jaccard: float
    pool: int
    k: int
    first: Assembly
    second: Assembly

    @property
    def trustworthy(self) -> bool:
        """False when the target had too few candidates for a real contest.

        READ THIS BEFORE ``stable``. If ``pool <= k`` the k-WTA returns every
        eligible neuron, so the cap cannot move and ``stable`` is True for a
        trained and an untrained pathway alike. Measured: with 30 materialised
        neurons and k=30 both read jaccard 1.0000 and the detector carries zero
        information.

        This is the [[fake-perfect-probe-signatures]] guard for this module.
        A detector that cannot distinguish "stable" from "no alternatives" is
        the dead-threshold failure of #104 wearing different clothes.
        """
        return self.pool > self.k

    def __str__(self) -> str:  # pragma: no cover - display only
        if not self.trustworthy:
            return (f"UNTRUSTWORTHY (pool {self.pool} <= k {self.k}: nothing "
                    f"to lose to, so 'stable' is vacuous)")
        verdict = "STABLE" if self.stable else "not an assembly"
        return f"{verdict} (jaccard {self.jaccard:.3f}, pool {self.pool})"


def assembly_stability(brain, source_area: str, target_area: str,
                       *, rounds: int = 1) -> Stability:
    """Fire ``source -> target``, then fire again WITH RECURRENCE; did the cap move?

    THE RECURRENCE IS THE CRITERION, not an implementation detail. The paper's
    "repeated firing" is a ``project*`` step, and ``project*`` fires every
    disinhibited area including the target into itself. The first version of
    this function projected ``{source: [target]}`` twice with no self-fiber, so
    the second firing saw an IDENTICAL input and returned an IDENTICAL cap --
    measured jaccard 1.0000 for a trained AND an untrained pathway, a detector
    with no power at all. With ``target -> target`` on the second firing the
    same brain reads 1.0000 trained versus 0.3000 untrained, because a real
    assembly is densely interconnected and reinforces itself while an arbitrary
    top-k has nothing holding it together. Same trap as
    [[project-rounds-drops-self-recurrence]].

    THE READ IS ISOLATED, under ``brain.read_only()``: no plasticity and no
    recruitment, so the test cannot create the assembly it is asking about. A
    projection under plasticity converges to a stable cap by construction.

    AND THE POOL MUST BE BIGGER THAN k -- check ``result.trustworthy``. Under
    lazy materialisation an untouched area may hold fewer than ``k`` neurons,
    in which case every one of them wins and stability is vacuous. Materialise
    the target first (``engine.materialize_area``) for the substrate the AC
    actually specifies: a fixed n neurons, all competing.

    Args:
        brain: the Brain.
        source_area: area holding the assembly to fire.
        target_area: area to fire into.
        rounds: firings per observation. 1 is the paper's "repeated firing".

    Returns:
        Stability. Read ``trustworthy`` before ``stable``.
    """
    from .ops import _snap

    with brain.read_only():
        for _ in range(max(1, rounds)):
            brain.project({}, {source_area: [target_area]})
        first = _snap(brain, target_area)
        for _ in range(max(1, rounds)):
            brain.project({}, {source_area: [target_area],
                               target_area: [target_area]})
        second = _snap(brain, target_area)

    engine = brain._engine_for(brain.areas[target_area])
    pool = engine.materialized_count(target_area)
    if pool is None:                       # engine does not track it; assume
        pool = int(getattr(brain.areas[target_area], "n", 0))   # the full area
    same = (len(first.winners) == len(second.winners)
            and bool(np.array_equal(np.sort(first.winners),
                                    np.sort(second.winners))))
    return Stability(stable=same, jaccard=float(overlap(first, second)),
                     pool=int(pool), k=int(brain.areas[target_area].k),
                     first=first, second=second)


def try_project(brain, source_area: str, target_area: str,
                *, rounds: int = 1) -> Optional[Assembly]:
    """The paper's ``try-project``: the assembly in B, or None if unstable.

    Returns None where the AC's readout would say "there is nothing here" --
    the ability our substrate otherwise lacks, since k-WTA always hands back k
    winners.

    Raises:
        ValueError: if the target has too few candidates to hold a contest.
            Returning an assembly there would be indistinguishable from a real
            success, and a silently vacuous True is what this module exists to
            prevent.
    """
    st = assembly_stability(brain, source_area, target_area, rounds=rounds)
    if not st.trustworthy:
        raise ValueError(
            f"try_project({source_area!r} -> {target_area!r}) is vacuous: the "
            f"target has {st.pool} materialised neurons for k={st.k}, so every "
            f"candidate wins and the cap cannot move. Materialise the area "
            f"first -- the AC specifies a fixed n neurons all competing.")
    return st.first if st.stable else None


# ---------------------------------------------------------------------------
# nonsense-assembly
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LexicalReadout:
    """What a cap in the lexical area corresponds to, if anything.

    Attributes:
        word: best-matching word, or None when the cap clears no threshold.
        best_overlap: its overlap.
        threshold: the DERIVED bar it had to clear (see nonsense_threshold).
        chance: the null, for reporting a ratio rather than a bare number.
        runner_up: (word, overlap) of the second best, or None.
    """

    word: Optional[str]
    best_overlap: float
    threshold: float
    chance: float
    runner_up: Optional[Tuple[str, float]]

    @property
    def is_nonsense(self) -> bool:
        """The paper's nonsense-assembly error: the cap is no word at all."""
        return self.word is None

    @property
    def times_chance(self) -> float:
        return self.best_overlap / self.chance if self.chance > 0 else 0.0

    @property
    def margin(self) -> float:
        """How far the winner beat the runner-up. Small = ambiguous readout."""
        if self.runner_up is None:
            return self.best_overlap
        return self.best_overlap - self.runner_up[1]

    def __str__(self) -> str:  # pragma: no cover - display only
        if self.word is None:
            return (f"NONSENSE (best {self.best_overlap:.4f} < "
                    f"{self.threshold:.4f}, chance {self.chance:.4f})")
        return (f"{self.word!r} ov={self.best_overlap:.4f} "
                f"({self.times_chance:.1f}x chance, margin {self.margin:.4f})")


def lexical_readout(assembly: Assembly, lexicon: Dict[str, Assembly],
                    *, n: int, k: Optional[int] = None,
                    z: float = 4.0) -> LexicalReadout:
    """Read a cap against the lexicon; report None when it is no word.

    This is the paper's ``getWord()`` with its failure case made explicit and
    its threshold derived rather than chosen. ``getWord()`` failing IS the
    nonsense-assembly error.

    Args:
        assembly: the cap to read (neuron IDs, i.e. Assembly.winners).
        lexicon: word -> reference Assembly, same index space.
        n: area size. Required, because the null depends on it and defaulting
            it would reintroduce the fitted-constant problem this module
            exists to avoid.
        k: cap size; taken from ``assembly`` when omitted.
        z: sigmas above chance required to call it a word.
    """
    kk = int(k if k is not None else len(assembly.winners))
    chance = chance_overlap(n, kk)
    thr = nonsense_threshold(n, kk, z=z)

    scored: List[Tuple[str, float]] = sorted(
        ((w, float(overlap(assembly, ref))) for w, ref in lexicon.items()),
        key=lambda t: (-t[1], t[0]),        # ties broken by NAME, not dict order
    )
    if not scored:
        return LexicalReadout(None, 0.0, thr, chance, None)

    best_word, best_ov = scored[0]
    runner = scored[1] if len(scored) > 1 else None
    return LexicalReadout(
        word=best_word if best_ov >= thr else None,
        best_overlap=best_ov, threshold=thr, chance=chance, runner_up=runner,
    )


# ---------------------------------------------------------------------------
# empty-project
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EmptyProject:
    """Whether a ``project*`` step fires nothing, and how that was established.

    Attributes:
        empty: no assembly would fire.
        detected_by: "gating" when the derived projection map has no target --
            faithful to Algorithm 2, where the map comes from the inhibition
            state. "drive" when a projection ran and produced no winners, which
            our k-WTA cannot currently report (see the module docstring).
        lex_targets: what the lexical area reaches, for the message.
    """

    empty: bool
    detected_by: str
    lex_targets: Tuple[str, ...]

    def __str__(self) -> str:  # pragma: no cover - display only
        if not self.empty:
            return f"project* reaches {list(self.lex_targets)}"
        return f"EMPTY-PROJECT (by {self.detected_by})"


def empty_project(state, brain, lex_area: str) -> EmptyProject:
    """Would ``project*`` fire nothing from *lex_area* under this gating state?

    The paper's example: having processed an intransitive verb, OBJ is not
    disinhibited and every other noun area is inhibited, so the next noun has
    nowhere to go. That is a syntactic violation detected structurally, with no
    threshold anywhere.

    Args:
        state: ``core.inhibition.InhibitionState``.
        brain: the Brain, consulted only for which areas hold winners.
        lex_area: the lexical area.
    """
    proj = state.project_map(brain, lex_area=lex_area)
    # Targets OTHER than lex_area itself: the reference treats LEX->LEX as
    # expected bookkeeping rather than a real destination (see the war-of-fibers
    # bound of 2 in InhibitionState.check_war_of_fibers), so counting it here
    # would make every state look non-empty.
    targets = tuple(t for t in proj.get(lex_area, ()) if t != lex_area)
    return EmptyProject(empty=not targets, detected_by="gating",
                        lex_targets=targets)

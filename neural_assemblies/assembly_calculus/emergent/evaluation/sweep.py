"""Parser cache and sweep helpers — amortize training across experiments.

Use ``ParserCache`` in pytest session fixtures and ``sweep_dynamics`` grids so
serial CPU work (lexicon, curriculum, calibration) runs once per cell key.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..parser import EmergentParser
    from .checkpoint import ParserCheckpoint

DEFAULT_N = 3000
DEFAULT_K = 30
#: Must track `EmergentParser.__init__`. These are cache-key material: they were
#: absent from both the in-memory key and the on-disk filename, so a study that
#: varied one of them silently re-used the other arm's parser.
DEFAULT_BETA = 0.05
DEFAULT_P = 0.05
DEFAULT_ROUNDS = 10
DEFAULT_PHON_WEIGHT = 6.0


def sweep_mode_enabled() -> bool:
    """Fast paths for parameter sweeps (single-pass calibration, mining probes)."""
    return os.environ.get("EMERGENT_SWEEP_MODE", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def wobbly_bootstrap_stages(*, fast: bool = True) -> frozenset:
    """Stages where wobbly episode mining runs (narrower in fast/sweep mode)."""
    if sweep_mode_enabled() or fast:
        return frozenset({"TWO_WORD", "SENTENCES", "COMPLEX_GRAMMAR"})
    return frozenset({
        "VOCABULARY_SPURT", "TWO_WORD", "SENTENCES",
        "COMPLEX_GRAMMAR", "DIALOGUE", "CONVERSATION",
    })


def erp_fast_calibration_enabled() -> bool:
    """Skip redundant calibration re-parse when safe."""
    if sweep_mode_enabled():
        return True
    return os.environ.get("EMERGENT_ERP_FAST", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def backbone_cache_dir():
    """Directory for pickled backbones, or ``None`` when disabled.

    Defaults to ``.cache/backbones`` at the repo root. Set
    ``ASSEMBLIES_BACKBONE_CACHE`` to another path to relocate it, or to ``0``/
    ``off`` to disable -- useful when bisecting, since a disk cache is exactly
    the kind of hidden state that makes a bisect lie.
    """
    from pathlib import Path

    raw = os.environ.get("ASSEMBLIES_BACKBONE_CACHE", "").strip()
    if raw.lower() in ("0", "off", "false", "no"):
        return None
    if raw:
        return Path(raw)
    return Path(__file__).resolve().parents[4] / ".cache" / "backbones"


#: Whole packages hashed RECURSIVELY. Everything numeric lives here -- the
#: engines, the seeding, the pricing law, the samplers, the winner policies --
#: and enumerating them file by file is what went wrong before.
#:
#: The list below used to name individual engine files, on the reasoning that
#: hashing more would "needlessly invalidate hours of training". That trade was
#: wrong in both directions. It was too coarse to be safe: a trace of a MINIMAL
#: projection run (two areas, four rounds) executes 15 modules under these two
#: packages and only THREE were listed -- the misses included
#: `core/numpy_engine/_seeding.py`, which decides every synapse's initial
#: weight, and `core/_pricing.py`, which decides who wins. And it was too
#: expensive to be worth it: `core/` and `compute/` are 39 files, so hashing
#: them wholesale costs 39 `stat` calls per process.
#:
#: A stale backbone does not fail loudly. It returns a plausible number, and it
#: corrupts exactly the A/B comparisons this program depends on. Retraining an
#: extra time is the cheaper error.
#:
#: `assembly_calculus` JOINED THE LIST ON 2026-08-04, and the reason is the
#: third occurrence of one mistake. The language layer used to be ENUMERATED --
#: eight files, chosen by judgement -- and the enumeration missed
#: `acquisition/pos_inference.py` and `parser_mixins/prediction.py`, which are
#: EXACTLY the two files whose edits fixed #80. They decide the order training
#: visits words, so they decide which neurons get recruited; editing them
#: served a stale pickle. Today's ERP re-run was safe only because those files
#: happened to predate the cached backbone by fourteen minutes.
#:
#: Enumerating "the files that matter" has now failed twice (once for the
#: engines, once here). Walking the package is the structural answer, and the
#: cost is 149 `stat` calls per process against 49.
_TRAINING_SOURCE_DIRS = (
    "core",
    "compute",
    "assembly_calculus",
)

#: Carved OUT of the recursive walk above. Everything here READS a trained
#: parser and cannot produce one, so hashing it would retrain ten backbones
#: every time an ERP metric is edited.
#:
#: THE BAR IS "no path from this module reaches plasticity or recruitment
#: during training", not "I think it is only analysis".
#:
#: I FAILED MY OWN BAR ON THE FIRST TRY, and it cost a real fix. This started
#: as the whole of `emergent/evaluation`, on the reasoning that evaluation
#: reads rather than trains. But `evaluation/generalization.py` is where
#: `train_parser_to_depth` lives -- it is the function that BUILDS every cached
#: backbone -- so adding phrase-pathway pre-growth there changed training and
#: did NOT invalidate the cache. The fix worked with the cache disabled and was
#: invisible through `forked_parser`, which is exactly the stale-pickle failure
#: this fingerprint exists to prevent (#106, and [[backbone-fingerprint-gap]]
#: for the third time).
#:
#: `evaluation/erp` is the genuine measurement layer: it probes a finished
#: parser and never trains. Nothing else in `evaluation/` qualifies, because
#: `sweep.py` and `generalization.py` both orchestrate training.
_NOT_TRAINING_DIRS = (
    "assembly_calculus/emergent/evaluation/erp",
)

#: Individual files outside the walked packages, kept for anything that lands
#: elsewhere in future. Empty is the healthy state: a name here is a judgement
#: call, and judgement calls are what the two misses above were.
_TRAINING_SOURCES = ()

_CODE_FINGERPRINT: Optional[str] = None


def training_code_fingerprint() -> str:
    """Short hash of the sources that determine a trained backbone.

    WHY THE CACHE KEY NEEDS THIS. The key was (depth, seed, n, k, holdout) --
    nothing about the CODE. So after changing role training, a warm session
    happily served a pickle trained by the OLD code, and an A/B of an
    intervention returned byte-identical numbers in both arms because neither
    arm trained anything. That reads exactly like a clean negative result.

    It is the same failure as workers re-importing edited source mid-run: the
    thing measured was not the thing under test. Both are silent, and both
    produce a plausible number rather than an error.

    Hashes relative paths and contents, once per process. This detects same-size
    edits with preserved timestamps and identifies identical source across
    checkouts. Run from an immutable checkout; restart after source changes.
    Set ``ASSEMBLIES_IGNORE_CODE_FINGERPRINT=1``
    to opt out when deliberately reusing a backbone across a known-irrelevant
    change.

    COVERAGE is the part that has gone wrong twice, so it is now structural:
    the whole of ``core/`` and ``compute/`` is walked recursively rather than
    enumerated. ``tests/test_training_fingerprint.py`` traces a real projection
    run and asserts every module that EXECUTES is covered, so the next addition
    cannot quietly fall outside it.
    """
    global _CODE_FINGERPRINT
    if _CODE_FINGERPRINT is not None:
        return _CODE_FINGERPRINT
    if os.environ.get("ASSEMBLIES_IGNORE_CODE_FINGERPRINT", "").strip().lower() \
            in ("1", "true", "yes", "on"):
        _CODE_FINGERPRINT = "ignored"
        return _CODE_FINGERPRINT
    import hashlib

    pkg = Path(__file__).resolve().parents[3]
    h = hashlib.sha256()
    for rel in fingerprint_source_files():
        p = pkg / rel
        try:
            content = p.read_bytes()
            h.update(rel.encode() + b"\0")
            h.update(hashlib.sha256(content).digest())
        except OSError:
            h.update(f"{rel}:missing".encode())
    _CODE_FINGERPRINT = h.hexdigest()
    return _CODE_FINGERPRINT


def fingerprint_source_files() -> Tuple[str, ...]:
    """Every package-relative path the fingerprint covers, sorted.

    Sorted so the hash does not depend on filesystem enumeration order, which
    differs between platforms and would give the same tree two fingerprints.
    Exposed rather than inlined so the coverage test can assert against the
    real set instead of re-deriving it.
    """
    pkg = Path(__file__).resolve().parents[3]
    found: Set[str] = set(_TRAINING_SOURCES)
    for d in _TRAINING_SOURCE_DIRS:
        root = pkg / d
        if not root.is_dir():
            continue
        for p in root.rglob("*.py"):
            rel = p.relative_to(pkg).as_posix()
            if any(rel.startswith(x + "/") for x in _NOT_TRAINING_DIRS):
                continue
            found.add(rel)
    return tuple(sorted(found))


#: Environment variables that change WHAT GETS TRAINED, not merely how fast.
#: They must be part of the cache identity: two parsers that differ in any of
#: these are different parsers, however equal their (depth, seed, n, k) look.
#:
#: THE BUG THIS CLOSES. `EMERGENT_DEV_CURRICULUM` turns off the preset skip so
#: babble + early grammar always run -- a different training corpus. It was
#: absent from the key, so a parser trained under it was stored under the SAME
#: key as one trained without, in memory AND on disk. One full-suite run
#: therefore poisoned the on-disk backbone for every later run.
#:
#: It reached the suite by import, not by intent: `tests/test_acquisition.py`
#: set it at MODULE level, and pytest imports every collected module before
#: running anything. So `pytest tests/` silently retrained every parser on a
#: different corpus, while `pytest <explicit files>` did not -- which is exactly
#: the pattern that looked like cross-test leakage and then like a cache defect.
#: Measured: adding `EMERGENT_DEV_CURRICULUM=1` to an otherwise-passing cold run
#: reproduces its 4 ERP failures precisely.
_TRAINING_ENV_VARS = ("EMERGENT_DEV_CURRICULUM",)


def _training_env_signature() -> Tuple:
    """The training-affecting environment, as part of the cache identity.

    Read at call time rather than import time on purpose: a process may legally
    change these between studies, and a signature captured at import would go
    stale in exactly the way this exists to prevent.
    """
    return tuple(
        (name, os.environ.get(name, "").strip().lower())
        for name in _TRAINING_ENV_VARS
    )


def _backbone_disk_path(depth, *, seed, n, k, holdout, params):
    """Full path for one backbone pickle, or ``None`` when caching is off."""
    root = backbone_cache_dir()
    if root is None:
        return None
    from .checkpoint import backbone_cache_filename

    name = backbone_cache_filename(
        depth, seed=seed, n=n, k=k, holdout_words=holdout, params=params,
    )
    # Fingerprint goes in the FILENAME, so a code change misses the cache and
    # retrains instead of loading a stale backbone. Old files simply stop being
    # found; they are cache entries, not data.
    fp = training_code_fingerprint()
    # The training ENVIRONMENT joins the code fingerprint in the filename, for
    # the same reason: a backbone trained under a different curriculum is a
    # different backbone, and must miss rather than load. Without this, one
    # `pytest tests/` run wrote a dev-curriculum parser over the normal one and
    # every later warm run silently used it.
    env = "".join(v for _n, v in _training_env_signature() if v)
    suffix = f".code{fp}" + (f".env{env}" if env else "")
    name = f"{Path(name).stem}{suffix}{Path(name).suffix or '.pkl'}"
    return root / name


def _pristine_copy(parser):
    """Specification: docs/reviews/whole-codebase/SEMANTIC_CARDS.md#contract-parser-fork

    Snapshot failure cannot degrade to a fork of mutable shared state.
    """
    try:
        import copy
        return copy.deepcopy(parser)
    except Exception as exc:
        raise RuntimeError("Cannot create pristine parser snapshot") from exc


@dataclass
class ParserCacheEntry:
    parser: "EmergentParser"
    train_seconds: float
    calibrated: bool = False
    calibration_seconds: float = 0.0
    #: A copy taken BEFORE anything was allowed to touch `parser`, and never
    #: handed out directly. `fork()` clones this rather than the live object.
    #:
    #: WHY. `get()` returns the shared `parser`, and its contract said that was
    #: "only safe to read". Reading is not safe: parsing RECRUITS -- one 5-word
    #: sentence of known words adds 663 neurons to an already-trained parser
    #: (#102). So a test that merely PARSED through `sentences_parser` grew the
    #: shared object, and every later `fork()` cloned the grown version.
    #:
    #: Measured consequence: seed 42 read Cohen's d = -0.26, INVERTED, against
    #: +1.80 from a pristine parser, and two suite tests failed as a function of
    #: which other tests had run first (#103).
    pristine: Optional["EmergentParser"] = None


@dataclass
class ParserCache:
    """Session-scoped cache keyed by training configuration."""

    _entries: Dict[Tuple, ParserCacheEntry] = field(default_factory=dict)
    hits: int = 0
    misses: int = 0
    disk_hits: int = 0

    def _key(
        self,
        depth: str,
        *,
        seed: int,
        holdout_words: Optional[Set[str]],
        n: int,
        k: int,
        fast_training: bool,
        params: Tuple,
    ) -> Tuple:
        holdout = frozenset(holdout_words or ())
        return (depth, seed, holdout, n, k, fast_training, params,
                _training_env_signature())

    def get(
        self,
        depth: str,
        *,
        seed: int = 42,
        holdout_words: Optional[Set[str]] = None,
        n: int = DEFAULT_N,
        k: int = DEFAULT_K,
        beta: float = DEFAULT_BETA,
        p: float = DEFAULT_P,
        rounds: int = DEFAULT_ROUNDS,
        phon_weight: float = DEFAULT_PHON_WEIGHT,
        fast_training: bool = True,
        calibrate: bool = False,
    ) -> "EmergentParser":
        """Return a trained parser, building and caching on first access.

        `beta`, `p` and `rounds` are part of BOTH cache keys. They were in
        neither, so a study that varied one of them re-used the other arm's
        parser -- in-memory within a run, and via the pickled backbone across
        runs. Neither raises, and warm runs do not train, so the only symptom
        was a zero effect.
        """
        params = (("beta", beta), ("p", p), ("rounds", rounds),
                  ("phon_weight", phon_weight))
        key = self._key(
            depth, seed=seed, holdout_words=holdout_words,
            n=n, k=k, fast_training=fast_training, params=params,
        )
        if key in self._entries:
            self.hits += 1
            entry = self._entries[key]
            if calibrate and not entry.calibrated:
                self._calibrate(entry)
            return entry.parser

        self.misses += 1
        from .generalization import default_holdout_set, train_parser_to_depth

        holdout = holdout_words if holdout_words is not None else default_holdout_set()

        # Second and later PROCESSES reuse the pickled backbone instead of
        # retraining: the in-memory dict above only amortizes within one run.
        disk_path = _backbone_disk_path(
            depth, seed=seed, n=n, k=k, holdout=frozenset(holdout),
            params=dict(params),
        )
        if disk_path is not None:
            from .checkpoint import load_backbone_cache

            cached = load_backbone_cache(disk_path)
            if cached is not None:
                self.disk_hits += 1
                entry = ParserCacheEntry(
                    parser=cached.parser,
                    train_seconds=cached.train_seconds,
                    calibrated=cached.calibrated,
                    calibration_seconds=cached.calibration_seconds,
                    pristine=_pristine_copy(cached.parser),
                )
                self._entries[key] = entry
                if calibrate and not entry.calibrated:
                    self._calibrate(entry)
                return entry.parser

        t0 = time.perf_counter()
        parser = train_parser_to_depth(
            depth,
            n=n,
            k=k,
            beta=beta,
            p=p,
            rounds=rounds,
            phon_weight=phon_weight,
            seed=seed,
            holdout_words=holdout,
            fast_training=fast_training,
        )
        train_seconds = time.perf_counter() - t0

        if disk_path is not None:
            from .checkpoint import ParserCheckpoint, save_backbone_cache

            try:
                save_backbone_cache(
                    ParserCheckpoint(
                        parser=parser, depth=depth, seed=seed, n=n, k=k,
                        holdout_words=frozenset(holdout),
                        train_seconds=train_seconds,
                    ),
                    disk_path,
                )
            except (OSError, TypeError, ValueError, AttributeError):
                # A cache that cannot be written must never fail the run.
                pass

        entry = ParserCacheEntry(parser=parser, train_seconds=train_seconds,
                                 pristine=_pristine_copy(parser))
        self._entries[key] = entry
        if calibrate:
            self._calibrate(entry)
        return entry.parser

    def fork(
        self,
        depth: str,
        *,
        wobbly: bool = False,
        **kwargs,
    ) -> "EmergentParser":
        """Return an INDEPENDENT trained parser, training at most once per key.

        ``get()`` hands back the shared cached object, so it is only safe for
        read-only tests. Anything that mutates the parser -- calibration,
        further training, lexicon writes -- must use this instead: the
        expensive curriculum training is still amortized across the session,
        but each caller gets its own copy to scribble on.

        All mutable parser state is isolated for either value of ``wobbly``.

        CLONES THE PRISTINE SNAPSHOT, not the live cached object. `get()` hands
        out a shared parser, and PARSING MUTATES IT (#102: one 5-word sentence
        of known words recruits 663 neurons). So a test that only read through
        `sentences_parser` still grew the shared object, and every fork after it
        inherited the growth -- which is how seed 42 came to read Cohen's d
        = -0.26, inverted, against +1.80 from a pristine parser (#103).
        """
        from .checkpoint import fork_parser_instance

        parser = self.get(depth, **kwargs)
        # `get` has just populated the entry; find it by identity rather than
        # by rebuilding the key, so the two cannot drift apart.
        for entry in self._entries.values():
            if entry.parser is parser:
                if entry.pristine is None:
                    raise RuntimeError("Cannot fork without a pristine parser snapshot")
                return fork_parser_instance(entry.pristine, wobbly=wobbly)
        raise RuntimeError("Cannot locate pristine parser snapshot for cached parser")

    def _calibrate(self, entry: ParserCacheEntry) -> None:
        """Specification: docs/reviews/whole-codebase/SEMANTIC_CARDS.md#contract-parser-fork

        Publish calibrated live/pristine state together, after all work succeeds.
        """
        from .erp import ensure_parser_erp_calibration

        if entry.pristine is None:
            raise RuntimeError("Cannot calibrate without a pristine parser snapshot")
        t0 = time.perf_counter()
        calibrated = _pristine_copy(entry.pristine)
        ensure_parser_erp_calibration(
            calibrated,
            fast=erp_fast_calibration_enabled(),
        )
        pristine = _pristine_copy(calibrated)
        entry.parser = calibrated
        entry.pristine = pristine
        entry.calibration_seconds = time.perf_counter() - t0
        entry.calibrated = True

    def clear(self) -> None:
        """Drop the in-memory cache. Leaves the on-disk backbones alone."""
        self._entries.clear()
        self.hits = 0
        self.misses = 0
        self.disk_hits = 0

    def stats(self) -> Dict[str, object]:
        return {
            "entries": len(self._entries),
            "hits": self.hits,
            "misses": self.misses,
            "disk_hits": self.disk_hits,
        }


# Module-level default for pytest session and sweep scripts
_default_cache: Optional[ParserCache] = None


def get_parser_cache() -> ParserCache:
    global _default_cache
    if _default_cache is None:
        _default_cache = ParserCache()
    return _default_cache


def reset_parser_cache() -> None:
    global _default_cache
    if _default_cache is not None:
        _default_cache.clear()


_SWEEP_LOG_T0 = time.perf_counter()


def reset_sweep_log() -> None:
    """Reset sweep step timer (call at start of a sweep run)."""
    global _SWEEP_LOG_T0
    _SWEEP_LOG_T0 = time.perf_counter()


def sweep_log_enabled() -> bool:
    return os.environ.get("SWEEP_LOG", "1").strip().lower() not in (
        "0", "false", "no", "off",
    )


def sweep_log(message: str) -> None:
    """Flushed per-step sweep log (stdout when ``EMERGENT_SWEEP_MODE=1``)."""
    if not sweep_log_enabled():
        return
    import sys

    elapsed = time.perf_counter() - _SWEEP_LOG_T0
    if sweep_mode_enabled():
        stream = sys.stdout
    else:
        stream = sys.stderr
    print(f"[sweep {elapsed:7.1f}s] {message}", file=stream, flush=True)


def delta_mining_sentences(
    holdout: str,
    *,
    probe_sent: Optional[list] = None,
) -> List[List[str]]:
    """Sentences for wobbly mining in a delta cell.

    Sweep mode uses one probe (holdout in object position); full mode adds
    a subject-position variant for broader wobble coverage.
    """
    probe = probe_sent or ["the", "bird", "finds", holdout]
    if sweep_mode_enabled():
        return [probe]
    return [probe, ["the", holdout, "bird"]]


def ingest_holdout_pattern(
    parser: "EmergentParser",
    pattern: str,
    holdout: str = "small",
) -> int:
    """Apply adversarial holdout exposure geometry before mining."""
    patterns = {
        "balanced": [
            ["the", holdout, "bird"],
            ["the", holdout, "dog"],
            ["the", "bird", "finds", holdout],
        ],
        "subject_only": [
            ["the", holdout, "bird"],
            ["the", holdout, "dog"],
            ["the", holdout, "cat"],
            ["the", "bird", "runs"],
        ],
        "object_only": [
            ["the", "bird", "runs"],
            ["the", "dog", "runs"],
            ["the", "bird", "finds", holdout],
            ["the", "dog", "finds", holdout],
        ],
    }
    sents = patterns.get(pattern, patterns["balanced"])
    for sent in sents:
        parser.ingest_raw_sentence(sent)
    from ..acquisition.pos_inference import ingest_holdout_sentence_stats

    ingest_holdout_sentence_stats(parser, {holdout})
    return len(sents)


def run_delta_cell(
    checkpoint: "ParserCheckpoint",
    *,
    pattern: str,
    wobbly: bool,
    holdout: str = "small",
    probe_sent: Optional[list] = None,
    log_steps: bool = True,
) -> Dict[str, object]:
    """Fork backbone, apply exposure delta + mining — no retrain."""
    from .checkpoint import fork_parser
    from .erp import assess_erp_readiness
    from ..acquisition import (
        classify_word_bootstrapped,
        infer_holdout_categories,
        mine_wobbly_episodes,
    )

    import time

    tag = f"pattern={pattern} wobbly={wobbly}"
    if log_steps:
        sweep_log(f"delta fork ({tag})")

    t_fork = time.perf_counter()
    parser = fork_parser(checkpoint, wobbly=wobbly)
    fork_s = time.perf_counter() - t_fork
    if log_steps:
        sweep_log(f"delta fork done {fork_s:.2f}s ({tag})")

    if log_steps:
        sweep_log(f"delta ingest exposure ({tag})")
    n_ingest = ingest_holdout_pattern(parser, pattern, holdout=holdout)
    holdout_set = set(checkpoint.holdout_words)
    if log_steps:
        sweep_log(f"delta ingest {n_ingest} sents ({tag})")

    mining_sents = delta_mining_sentences(holdout, probe_sent=probe_sent)
    if log_steps:
        sweep_log(
            f"delta mine {len(mining_sents)} probe(s) ({tag})",
        )
    mined_probes: list = []
    t_mine = time.perf_counter()
    mem = mine_wobbly_episodes(
        parser,
        mining_sents,
        target_words={holdout},
        probe_depth="mining",
        collected_probes=mined_probes,
    )
    mine_s = time.perf_counter() - t_mine
    wobble = (
        sum(1 for p in mined_probes if p.wobbly) / len(mined_probes)
        if mined_probes else 0.0
    )
    if log_steps:
        sweep_log(
            f"delta mine done {mine_s:.2f}s "
            f"episodes={len(mem.episodes)} wobble={wobble:.2f} ({tag})",
        )

    if wobbly and mem.episodes:
        if log_steps:
            sweep_log(f"delta wobbly replay x{len(mem.episodes)} ({tag})")
        from ..acquisition.wobbly.replay import replay_wobbly_episodes
        replay_wobbly_episodes(parser, mem)

    if log_steps:
        sweep_log(f"delta classify ({tag})")
    infer_holdout_categories(parser, holdout_set)
    cat, _ = classify_word_bootstrapped(parser, holdout)
    readiness = assess_erp_readiness(parser)

    expected = {"small": "ADJ", "bird": "NOUN", "finds": "VERB"}
    correct = sum(
        1 for w in holdout_set
        if w in expected
        and classify_word_bootstrapped(parser, w)[0] == expected[w]
    )
    total = sum(1 for w in holdout_set if w in expected)

    if log_steps:
        sweep_log(
            f"delta done acc={correct / max(1, total):.2f} "
            f"cat={cat} fork={fork_s:.2f}s mine={mine_s:.2f}s ({tag})",
        )

    return {
        "train_s": round(checkpoint.train_seconds, 3),
        "calibration_s": round(checkpoint.calibration_seconds, 3),
        "fork_s": round(fork_s, 3),
        "mine_s": round(mine_s, 3),
        "n400_cohens_d": 0.0,
        "p600_cohens_d": 0.0,
        "p600_ready": readiness.p600_ready,
        "holdout_acc": round(correct / max(1, total), 4),
        "holdout_small_cat": cat,
        "wobble_rate": round(wobble, 4),
        "wobbly_episodes": len(mem.episodes),
        "cache_hit": True,
        "backbone_key": str(checkpoint.key),
    }

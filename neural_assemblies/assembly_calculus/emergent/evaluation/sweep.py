"""Parser cache and sweep helpers — amortize training across experiments.

Use ``ParserCache`` in pytest session fixtures and ``sweep_dynamics`` grids so
serial CPU work (lexicon, curriculum, calibration) runs once per cell key.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..parser import EmergentParser
    from .checkpoint import ParserCheckpoint

DEFAULT_N = 3000
DEFAULT_K = 30


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


_TRAINING_SOURCES = (
    # Modules whose contents change what a trained backbone contains. Kept
    # explicit rather than hashing the whole package, so an unrelated edit does
    # not needlessly invalidate hours of training.
    "assembly_calculus/ops.py",
    "assembly_calculus/emergent/core/corpus_index.py",
    "assembly_calculus/emergent/parser_mixins/roles.py",
    "assembly_calculus/emergent/parser_mixins/unsupervised.py",
    "assembly_calculus/emergent/parser_mixins/core.py",
    "assembly_calculus/emergent/parser_mixins/classify.py",
    "assembly_calculus/emergent/training/batch.py",
    "assembly_calculus/emergent/training/schedule.py",
    "core/numpy_engine/_sparse.py",
    # The k-WTA CANDIDATE SAMPLER. Omitting it was the exact failure this
    # fingerprint exists to prevent, and it cost a long false-positive hunt:
    # `_binom_ppf_cached` was rewritten to avoid importing scipy.stats, the new
    # value was verified identical on all 2013 calls of a live run, and
    # test_erp_calibration still flipped -- because the backbone being
    # calibrated had been trained under the OLD sampler and the fingerprint
    # never noticed. Forcing a retrain made both arms agree.
    "compute/sparse_simulation.py",
    # Its sibling: the winner-selection policy that consumes those candidates.
    "compute/winner_selection.py",
)

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

    Hashes size and mtime rather than contents -- enough to notice an edit,
    cheap enough to run on every lookup. Set ``ASSEMBLIES_IGNORE_CODE_FINGERPRINT=1``
    to opt out when deliberately reusing a backbone across a known-irrelevant
    change.
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
    h = hashlib.blake2b(digest_size=6)
    for rel in _TRAINING_SOURCES:
        p = pkg / rel
        try:
            st = p.stat()
            h.update(f"{rel}:{st.st_size}:{int(st.st_mtime)}".encode())
        except OSError:
            h.update(f"{rel}:missing".encode())
    _CODE_FINGERPRINT = h.hexdigest()
    return _CODE_FINGERPRINT


def _backbone_disk_path(depth, *, seed, n, k, holdout):
    """Full path for one backbone pickle, or ``None`` when caching is off."""
    root = backbone_cache_dir()
    if root is None:
        return None
    from .checkpoint import backbone_cache_filename

    name = backbone_cache_filename(
        depth, seed=seed, n=n, k=k, holdout_words=holdout,
    )
    # Fingerprint goes in the FILENAME, so a code change misses the cache and
    # retrains instead of loading a stale backbone. Old files simply stop being
    # found; they are cache entries, not data.
    fp = training_code_fingerprint()
    name = f"{Path(name).stem}.code{fp}{Path(name).suffix or '.pkl'}"
    return root / name


@dataclass
class ParserCacheEntry:
    parser: "EmergentParser"
    train_seconds: float
    calibrated: bool = False
    calibration_seconds: float = 0.0


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
    ) -> Tuple:
        holdout = frozenset(holdout_words or ())
        return (depth, seed, holdout, n, k, fast_training)

    def get(
        self,
        depth: str,
        *,
        seed: int = 42,
        holdout_words: Optional[Set[str]] = None,
        n: int = DEFAULT_N,
        k: int = DEFAULT_K,
        fast_training: bool = True,
        calibrate: bool = False,
    ) -> "EmergentParser":
        """Return a trained parser, building and caching on first access."""
        key = self._key(
            depth, seed=seed, holdout_words=holdout_words,
            n=n, k=k, fast_training=fast_training,
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

        entry = ParserCacheEntry(parser=parser, train_seconds=train_seconds)
        self._entries[key] = entry
        if calibrate:
            self._calibrate(entry)
        return parser

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

        Pass ``wobbly=True`` when replay may rewrite the assembly lexicons.
        """
        from .checkpoint import fork_parser_instance

        return fork_parser_instance(self.get(depth, **kwargs), wobbly=wobbly)

    def _calibrate(self, entry: ParserCacheEntry) -> None:
        from .erp import ensure_parser_erp_calibration

        t0 = time.perf_counter()
        ensure_parser_erp_calibration(
            entry.parser,
            fast=erp_fast_calibration_enabled(),
        )
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

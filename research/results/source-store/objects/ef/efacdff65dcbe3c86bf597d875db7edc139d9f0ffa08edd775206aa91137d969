"""Parser checkpointing — train backbone once, fork per sweep cell.

WHY FORKING IS THE ONLY CORRECT WAY TO SWEEP THIS MODEL.  A parameter sweep
normally reuses one trained model and varies a setting at evaluation time.
That is invalid here.  Assembly formation is order-dependent and every
projection applies plasticity, so evaluating cell A leaves the connectome
different from how cell B would have found it.  Reusing one parser across
cells means each cell is measured on a brain the previous cells reshaped, and
the sweep silently measures the ORDER of cells as much as the parameter.

Checkpointing buys back the shared work without that contamination: train the
backbone once, then deep-copy it per cell so each cell starts from the
identical connectome.  The copy is what makes cells independent; it is not an
optimisation and must not be replaced with sharing a reference.

Correspondingly, a checkpoint's identity is the training that produced it --
depth, seed, and the holdout set are all part of the key, because two parsers
trained to the same depth with different holdouts are not interchangeable.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import FrozenSet, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from ..parser import EmergentParser


@dataclass
class ParserCheckpoint:
    """Backbone plus training/calibration metadata; fork before mutation."""

    parser: "EmergentParser"
    depth: str
    seed: int
    n: int
    k: int
    holdout_words: FrozenSet[str]
    train_seconds: float = 0.0
    calibration_seconds: float = 0.0
    calibrated: bool = False
    meta: dict = field(default_factory=dict)

    @property
    def key(self) -> tuple:
        return (self.depth, self.seed, self.holdout_words, self.n, self.k)


def _reset_ephemeral_parser_state(parser: "EmergentParser") -> None:
    """Prepare a copied parser for a new cell using the legacy cursor reset.

    Besides caches, this resets CONTEXT construction counts/IDs while retaining
    learned fibers. It is not an activity-only reset or exact checkpoint restore.
    """
    parser._incremental_circuit = None
    if hasattr(parser, "_wobbly_memory"):
        del parser._wobbly_memory
    if hasattr(parser, "_reset_context_state"):
        parser._reset_context_state()


def fork_parser_instance(
    src: "EmergentParser",
    *,
    wobbly: bool = False,
) -> "EmergentParser":
    """Specification: docs/reviews/whole-codebase/SEMANTIC_CARDS.md#contract-parser-fork

    Copy the complete parser graph, then prepare its sentence-construction state.
    The legacy wobbly flag no longer weakens isolation of any mutable state.
    """
    parser = copy.deepcopy(src)
    _reset_ephemeral_parser_state(parser)
    # The flag controls downstream wobbly-episode replay. Preserve it on the
    # fork so the low-level copy does not erase experiment provenance.
    parser._wobbly_fork = bool(wobbly)
    return parser


def fork_parser(
    checkpoint: ParserCheckpoint,
    *,
    wobbly: bool = False,
) -> "EmergentParser":
    """Return an independent parser copy from a backbone checkpoint."""
    return fork_parser_instance(checkpoint.parser, wobbly=wobbly)


def build_parser_backbone(
    depth: str,
    *,
    seed: int = 42,
    holdout_words: Optional[Set[str]] = None,
    n: int = 3000,
    k: int = 30,
    fast_training: bool = True,
    calibrate: bool = True,
    fast_calibration: Optional[bool] = None,
    show_progress: Optional[bool] = None,
) -> ParserCheckpoint:
    """Train once to *depth*, optionally calibrate ERP, return checkpoint."""
    from .generalization import default_holdout_set, train_parser_to_depth
    from .sweep import erp_fast_calibration_enabled
    from ..train_progress import (
        finish_progress,
        progress_enabled,
        start_progress,
    )
    from ..training.perf import resolve_engine, warn_engine_scale_mismatch

    if show_progress is None:
        show_progress = progress_enabled()

    holdout = frozenset(holdout_words or default_holdout_set())
    engine = resolve_engine("auto", n_hint=n)
    warn_engine_scale_mismatch(engine, n, where="build_parser_backbone")
    if show_progress:
        start_progress(
            f"backbone {depth} seed={seed} n={n:,} k={k} engine={engine}",
        )

    t0 = time.perf_counter()
    parser = train_parser_to_depth(
        depth,
        n=n,
        k=k,
        seed=seed,
        holdout_words=set(holdout),
        fast_training=fast_training,
    )
    train_s = time.perf_counter() - t0

    cal_s = 0.0
    calibrated = False
    if calibrate:
        from .erp import ensure_parser_erp_calibration
        from ..train_progress import current_progress

        if fast_calibration is None:
            fast_calibration = erp_fast_calibration_enabled()
        t_cal = time.perf_counter()
        if show_progress:
            current_progress().phase_start(
                "erp_calibration",
                "fast" if fast_calibration else "full",
            )
        ensure_parser_erp_calibration(parser, fast=fast_calibration)
        cal_s = time.perf_counter() - t_cal
        calibrated = True
        if show_progress:
            current_progress().phase_done("erp_calibration")

    if show_progress:
        finish_progress(
            f"train={train_s:.1f}s cal={cal_s:.1f}s "
            f"engine={getattr(parser, 'engine_name', engine)}",
        )

    return ParserCheckpoint(
        parser=parser,
        depth=depth,
        seed=seed,
        n=n,
        k=k,
        holdout_words=holdout,
        train_seconds=train_s,
        calibration_seconds=cal_s,
        calibrated=calibrated,
    )


# Disk cache for sweep resume (pickle whole checkpoint after backbone build).
BACKBONE_CACHE_VERSION = 1

_SOURCE_FINGERPRINT: Optional[str] = None


# Directories whose contents cannot influence how a parser is TRAINED, and so
# must not invalidate a trained parser. Tests and research scripts import the
# library; the library does not import them.
_FINGERPRINT_EXCLUDED_DIRS = frozenset({"tests", "__pycache__"})


def _semantic_source_digest(path) -> bytes:
    """Structure-only digest of one module: docstrings and comments removed.

    Hashing raw bytes means a documentation pass invalidates every cached
    parser, which is wrong by construction -- a docstring cannot change what a
    projection computes. Comments never reach the AST at all, and docstrings are
    stripped here, so a purely expository edit leaves the digest identical.

    Falls back to raw bytes if the file will not parse, which is the safe
    direction: an unparseable file yields a different digest and simply misses.
    """
    import ast

    raw = path.read_bytes()
    try:
        tree = ast.parse(raw)
    except (SyntaxError, ValueError):
        return raw
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list) or not body:
            continue
        if not isinstance(node, (ast.Module, ast.FunctionDef,
                                 ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        first = body[0]
        if (isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)):
            body.pop(0)
    return ast.dump(tree).encode()


def source_fingerprint() -> str:
    """Structural hash over the library modules that can affect training.

    A pickled backbone is only reusable while the code that produced it is
    unchanged. Without this, editing the simulation path -- winner selection,
    plasticity, the engines -- would silently reuse a parser trained by the OLD
    code and report results that no longer correspond to the source. That is a
    correctness bug, not a stale-cache annoyance, so the fingerprint is part of
    the cache filename: changed source simply misses and retrains.

    SCOPE, and why it is narrower than it looks. The first version hashed every
    ``.py`` file in the repo -- 801 files, of which 421 (53%) were tests and
    research scripts that cannot influence training, because the library does
    not import them. Editing a single test therefore threw away every trained
    parser, and during iteration test edits are the most common edit. Only ~133
    modules are in training's actual import closure.

    Conservatism at the wrong granularity buys no safety while costing
    everything in speed: the safety comes from hashing the right dependency
    closure, not from hashing more files. So this hashes the library package
    minus ``tests/``, which is deterministic and strictly a superset of the
    import closure -- erring toward over-invalidation rather than under.

    Uses a structure-only digest (see ``_semantic_source_digest``) rather than
    stat or raw bytes, so that a ``touch``, a branch switch that restores
    identical content, or a docstring pass all leave the cache valid.
    """
    global _SOURCE_FINGERPRINT
    if _SOURCE_FINGERPRINT is None:
        import hashlib
        from pathlib import Path

        # parents[3] IS the neural_assemblies package directory
        # (.../neural_assemblies/assembly_calculus/emergent/evaluation/this).
        # Appending "neural_assemblies" again yields a path that does not
        # exist, rglob returns nothing, and the fingerprint silently becomes a
        # constant that NEVER invalidates -- the exact failure this mechanism
        # exists to prevent. Guarded below.
        pkg = Path(__file__).resolve().parents[3]
        h = hashlib.blake2b(digest_size=8)
        hashed = 0
        for py in sorted(pkg.rglob("*.py")):
            if _FINGERPRINT_EXCLUDED_DIRS & set(py.parts):
                continue
            try:
                digest = _semantic_source_digest(py)
            except OSError:
                continue
            h.update(py.relative_to(pkg).as_posix().encode())
            h.update(b":")
            h.update(hashlib.blake2b(digest, digest_size=8).digest())
            h.update(b"|")
            hashed += 1
        if hashed == 0:
            # A fingerprint over zero files is a constant, so every backbone
            # would look valid forever and stale parsers would be reused
            # silently across arbitrary code changes. Refuse rather than
            # degrade: a wrong path must not read as "nothing changed".
            raise RuntimeError(
                f"source_fingerprint hashed no files under {pkg!r}; refusing "
                f"to emit a fingerprint that can never invalidate",
            )
        _SOURCE_FINGERPRINT = h.hexdigest()
    return _SOURCE_FINGERPRINT


def training_params_digest(params) -> str:
    """Stable short digest of the training parameters that are not in the name.

    WHY THIS IS A REQUIRED ARGUMENT rather than an optional one with a default.
    The filename used to key on ``depth, seed, n, k, holdout`` only, so two
    parsers trained at DIFFERENT beta collided on one cache file -- the second
    arm of a beta A/B would silently load the first arm's backbone and the
    study would compare a parser against a copy of itself. Nothing raises;
    warm runs do not train ([[backbone-fingerprint-gap]]), so the only symptom
    is an effect size of zero.

    `p`, `rounds` and the vocabulary had the same hole. Making the parameter
    REQUIRED means a new training knob cannot be added without either putting
    it in here or deliberately deciding not to -- the omission has to be
    spelled, which is the point ([[one-canonical-way]]).
    """
    import hashlib

    items = sorted((str(key), repr(value)) for key, value in dict(params).items())
    h = hashlib.sha256("|".join(f"{key}={value}" for key, value in items).encode())
    return h.hexdigest()[:10]


def backbone_cache_filename(
    depth: str,
    *,
    seed: int,
    n: int,
    k: int,
    holdout_words: FrozenSet[str],
    params,
) -> str:
    holdout_tag = "-".join(sorted(holdout_words)) or "none"
    return (
        f"v{BACKBONE_CACHE_VERSION}_{source_fingerprint()}"
        f"_{depth}_s{seed}_n{n}_k{k}_t{training_params_digest(params)}"
        f"_h{holdout_tag}.pkl"
    )


def load_backbone_cache(path) -> Optional[ParserCheckpoint]:
    """Specification: docs/reviews/whole-codebase/SEMANTIC_CARDS.md#contract-checkpoint-storage

    Load a trusted local checkpoint; incompatible or incomplete caches miss.
    """
    import pickle
    from pathlib import Path

    p = Path(path)
    if not p.is_file():
        return None
    try:
        with p.open("rb") as f:
            cp = pickle.load(f)
    except (OSError, pickle.PickleError, EOFError, ImportError, AttributeError, TypeError, ValueError):
        return None
    if not isinstance(cp, ParserCheckpoint):
        return None
    return cp


def save_backbone_cache(checkpoint: ParserCheckpoint, path) -> None:
    """Specification: docs/reviews/whole-codebase/SEMANTIC_CARDS.md#contract-checkpoint-storage

    Publish a complete cache using a private temporary file on the same volume.
    """
    import os
    import pickle
    from pathlib import Path
    from tempfile import NamedTemporaryFile

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with NamedTemporaryFile(mode="wb", dir=p.parent, prefix=f".{p.name}.",
                                suffix=".tmp", delete=False) as f:
            tmp = Path(f.name)
            pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())
        # Windows can briefly deny replacement while another publisher closes
        # the destination. Retry only its contention-capable error codes, with
        # at most 310 ms total backoff; permanent failure still propagates.
        for attempt in range(6):
            try:
                tmp.replace(p)
                break
            except PermissionError as exc:
                if getattr(exc, "winerror", None) not in (5, 32, 33) or attempt == 5:
                    raise
                time.sleep(0.01 * 2 ** attempt)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def backbone_cache_path(
    cache_dir,
    depth: str,
    *,
    seed: int,
    n: int,
    k: int,
    holdout_words: FrozenSet[str],
    params,
):
    from pathlib import Path

    return Path(cache_dir) / backbone_cache_filename(
        depth, seed=seed, n=n, k=k, holdout_words=holdout_words, params=params,
    )

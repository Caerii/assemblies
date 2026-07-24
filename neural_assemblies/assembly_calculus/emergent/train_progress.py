"""Streaming training progress to the console (stderr by default).

Enable/disable with environment variable ``TRAIN_PROGRESS`` (default ``1``).

Usage::

    from neural_assemblies.assembly_calculus.emergent.train_progress import (
        TrainProgress, progress_enabled, start_progress,
    )

    prog = start_progress("train_for_agent")
    with prog.phase("lexicon", f"{n_words} words"):
        parser.train_lexicon()
    prog.finish("ok")
"""

from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager
from typing import Iterator, Optional, TextIO


def progress_enabled() -> bool:
    """True unless ``TRAIN_PROGRESS`` is 0/false/no."""
    return os.environ.get("TRAIN_PROGRESS", "1").strip().lower() not in (
        "0", "false", "no", "off",
    )


def default_progress_stream() -> TextIO:
    """Sweep scripts log to stdout so progress is visible in the default terminal."""
    if os.environ.get("EMERGENT_SWEEP_MODE", "").strip().lower() in (
        "1", "true", "yes", "on",
    ):
        return sys.stdout
    return sys.stderr


class TrainProgress:
    """Line-buffered progress logger with elapsed timestamps."""

    def __init__(
        self,
        title: str,
        *,
        enabled: bool = True,
        stream: Optional[TextIO] = None,
    ):
        self.title = title
        self.enabled = enabled
        self.stream = stream if stream is not None else default_progress_stream()
        self._t0 = time.perf_counter()
        self._phase_t0 = self._t0
        self._depth = 0

    def _elapsed(self) -> float:
        return time.perf_counter() - self._t0

    def _emit(self, level: str, message: str) -> None:
        if not self.enabled:
            return
        indent = "  " * self._depth
        print(
            f"[{self._elapsed():7.1f}s] {self.title} {level} {indent}{message}",
            file=self.stream,
            flush=True,
        )

    def info(self, message: str) -> None:
        self._emit("|", message)

    def start(self, detail: str = "") -> None:
        suffix = f" — {detail}" if detail else ""
        self._emit(">>", f"BEGIN{suffix}")

    def finish(self, summary: str = "") -> None:
        total = self._elapsed()
        suffix = f" — {summary}" if summary else ""
        self._emit("<<", f"DONE ({total:.1f}s total){suffix}")

    def phase_start(self, name: str, detail: str = "") -> None:
        self._phase_t0 = time.perf_counter()
        suffix = f" — {detail}" if detail else ""
        self._emit("..", f"{name}{suffix}")

    def phase_done(self, name: str, detail: str = "") -> None:
        dt = time.perf_counter() - self._phase_t0
        suffix = f" — {detail}" if detail else ""
        self._emit("ok", f"{name} ({dt:.1f}s){suffix}")

    @contextmanager
    def phase(self, name: str, detail: str = "") -> Iterator["TrainProgress"]:
        self.phase_start(name, detail)
        try:
            yield self
        finally:
            self.phase_done(name)

    @contextmanager
    def section(self, name: str) -> Iterator["TrainProgress"]:
        self._emit(">>", name)
        self._depth += 1
        try:
            yield self
        finally:
            self._depth -= 1
            self._emit("<<", name)


class _NullProgress(TrainProgress):
    """No-op progress sink."""

    def __init__(self) -> None:
        super().__init__("", enabled=False)

    def _emit(self, level: str, message: str) -> None:
        return


_NULL = _NullProgress()
_active: Optional[TrainProgress] = None


def start_progress(
    title: str,
    *,
    enabled: Optional[bool] = None,
    stream: Optional[TextIO] = None,
) -> TrainProgress:
    """Start a global progress session (retrieved via ``current_progress()``)."""
    global _active
    if enabled is None:
        enabled = progress_enabled()
    _active = TrainProgress(title, enabled=enabled, stream=stream)
    _active.start()
    return _active


def current_progress() -> TrainProgress:
    """Return the active progress logger, or a no-op stub."""
    return _active if _active is not None else _NULL


def ensure_progress(title: str) -> TrainProgress:
    """Start progress logging if not already active."""
    global _active
    if _active is None:
        return start_progress(title)
    return _active


def finish_progress(summary: str = "") -> None:
    """Finish the active progress session if one is running."""
    global _active
    if _active is not None:
        _active.finish(summary)
        _active = None

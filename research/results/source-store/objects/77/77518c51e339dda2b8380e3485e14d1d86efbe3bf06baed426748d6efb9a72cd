"""Immutable protocol for the registered word-capacity study.

Specification: docs/reviews/whole-codebase/SEMANTIC_CARDS.md#contract-word-capacity-protocol
Registration: research/notes/aligner/PREREG_word_capacity.md
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math


def _positive_int(name: str, value: int) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _finite_number(name: str, value: float) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


@dataclass(frozen=True)
class CapacityCell:
    """One lexical area and its phonological anchor."""

    name: str
    n: int
    k: int
    stimulus_size: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("cell name must be nonempty")
        for field in ("n", "k", "stimulus_size"):
            _positive_int(f"cell {self.name} {field}", getattr(self, field))
        if self.k > self.n or self.stimulus_size > self.n:
            raise ValueError(f"cell {self.name} counts cannot exceed n")

    def to_list(self) -> list[int]:
        return [self.n, self.k, self.stimulus_size]


@dataclass(frozen=True)
class WordCapacityProtocol:
    """Every value that can alter the corpus, learner, readout, or sweep."""

    cells: tuple[str, ...]
    vocabulary_sizes: tuple[int, ...]
    feature_area: tuple[int, int]
    cell_definitions: tuple[CapacityCell, ...]
    connection_probability: float
    plasticity: float
    rounds_per_pair: int
    exposures_per_referent: int
    referents_per_scene: int
    category_count: int
    minimum_exposures: int
    threshold: float
    corpus_seed_offset: int
    corpus_seed_scope: str
    training_seed_offset: int
    connectome_seed_stride: int
    early_stop_margin: float
    ceiling_interior_band: tuple[float, float]
    w2_pass_tolerance: float
    w2_fail_tolerance: float
    w3_pass_ratio: float
    w3_fail_ratio: float
    launch_budget_bytes: int
    feature_ladder: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.cells, tuple) or not isinstance(self.cell_definitions, tuple):
            raise ValueError("cells and cell definitions must be tuples")
        definitions = {cell.name: cell for cell in self.cell_definitions}
        if len(definitions) != len(self.cell_definitions):
            raise ValueError("cell definitions must have unique names")
        if not self.cells or len(set(self.cells)) != len(self.cells):
            raise ValueError("cells must be nonempty and unique")
        if any(name not in definitions for name in self.cells):
            raise ValueError("every selected cell must have a definition")
        sizes = self.vocabulary_sizes
        if (len(sizes) < 2 or any(type(value) is not int or value <= 1 for value in sizes)
                or tuple(sorted(set(sizes))) != sizes):
            raise ValueError("vocabulary sizes must be increasing unique integers above one")
        probability = _finite_number(
            "connection probability", self.connection_probability,
        )
        plasticity = _finite_number("plasticity", self.plasticity)
        threshold = _finite_number("threshold", self.threshold)
        margin = _finite_number("early stop margin", self.early_stop_margin)
        if not 0 < probability <= 1:
            raise ValueError("connection probability must be in (0, 1]")
        if plasticity < 0:
            raise ValueError("plasticity must be nonnegative")
        for field in (
            "rounds_per_pair", "exposures_per_referent", "referents_per_scene",
            "category_count", "minimum_exposures", "launch_budget_bytes",
        ):
            _positive_int(field.replace("_", " "), getattr(self, field))
        if self.minimum_exposures > self.exposures_per_referent:
            raise ValueError("minimum exposures cannot exceed exposures per referent")
        if self.referents_per_scene > min(sizes):
            raise ValueError("referents per scene cannot exceed the smallest vocabulary")
        if not 0 < threshold <= 1:
            raise ValueError("threshold must be in (0, 1]")
        if not 0 <= margin < threshold:
            raise ValueError("early stop margin must be in [0, threshold)")
        if type(self.corpus_seed_offset) is not int or type(self.training_seed_offset) is not int:
            raise ValueError("seed offsets must be integers")
        if self.corpus_seed_scope not in {"per-brain", "shared-batch"}:
            raise ValueError("corpus seed scope must be per-brain or shared-batch")
        _positive_int("connectome seed stride", self.connectome_seed_stride)
        if (not isinstance(self.ceiling_interior_band, tuple)
                or len(self.ceiling_interior_band) != 2):
            raise ValueError("ceiling interior band must be a pair")
        band_lo, band_hi = (
            _finite_number("ceiling interior bound", value)
            for value in self.ceiling_interior_band
        )
        if not 0 <= band_lo < band_hi <= 1:
            raise ValueError("ceiling interior band must lie within [0, 1]")
        w2_pass = _finite_number("W2 pass tolerance", self.w2_pass_tolerance)
        w2_fail = _finite_number("W2 fail tolerance", self.w2_fail_tolerance)
        w3_pass = _finite_number("W3 pass ratio", self.w3_pass_ratio)
        w3_fail = _finite_number("W3 fail ratio", self.w3_fail_ratio)
        if not 0 <= w2_pass < w2_fail:
            raise ValueError("W2 tolerances must satisfy 0 <= pass < fail")
        if not 0 <= w3_fail < w3_pass:
            raise ValueError("W3 ratios must satisfy 0 <= fail < pass")
        if not isinstance(self.feature_ladder, tuple) or not self.feature_ladder:
            raise ValueError("feature ladder must be a nonempty tuple")
        for name, area in (
            ("feature area", self.feature_area),
            *[("feature ladder area", area) for area in self.feature_ladder],
        ):
            if (not isinstance(area, tuple) or len(area) != 2
                    or any(type(value) is not int or value <= 0 for value in area)
                    or area[1] > area[0]):
                raise ValueError(f"{name} must be an (n, k) pair with 0 < k <= n")

    @property
    def definitions(self) -> dict[str, CapacityCell]:
        return {cell.name: cell for cell in self.cell_definitions}

    def cell(self, name: str) -> CapacityCell:
        if name not in self.cells:
            raise ValueError(f"capacity cell {name!r} is not selected by this protocol")
        try:
            return self.definitions[name]
        except KeyError as exc:
            raise ValueError(f"unknown capacity cell: {name}") from exc

    def select(
        self, *, cells: tuple[str, ...] | None = None,
        vocabulary_sizes: tuple[int, ...] | None = None,
        feature_area: tuple[int, int] | None = None,
    ) -> WordCapacityProtocol:
        """Return a validated protocol selection without mutating the base."""
        return replace(
            self,
            cells=self.cells if cells is None else cells,
            vocabulary_sizes=(self.vocabulary_sizes if vocabulary_sizes is None
                              else vocabulary_sizes),
            feature_area=self.feature_area if feature_area is None else feature_area,
        )

    def to_parameters(self) -> dict:
        """Encode the complete protocol as strict JSON-compatible values."""
        return {
            "cells": list(self.cells),
            "vocabulary_sizes": list(self.vocabulary_sizes),
            "feature_area": list(self.feature_area),
            "cell_definitions": {
                cell.name: cell.to_list() for cell in self.cell_definitions
            },
            "connection_probability": self.connection_probability,
            "plasticity": self.plasticity,
            "rounds_per_pair": self.rounds_per_pair,
            "exposures_per_referent": self.exposures_per_referent,
            "referents_per_scene": self.referents_per_scene,
            "category_count": self.category_count,
            "minimum_exposures": self.minimum_exposures,
            "threshold": self.threshold,
            "corpus_seed_offset": self.corpus_seed_offset,
            "corpus_seed_scope": self.corpus_seed_scope,
            "training_seed_offset": self.training_seed_offset,
            "connectome_seed_stride": self.connectome_seed_stride,
            "early_stop_margin": self.early_stop_margin,
            "ceiling_interior_band": list(self.ceiling_interior_band),
            "w2_pass_tolerance": self.w2_pass_tolerance,
            "w2_fail_tolerance": self.w2_fail_tolerance,
            "w3_pass_ratio": self.w3_pass_ratio,
            "w3_fail_ratio": self.w3_fail_ratio,
            "launch_budget_bytes": self.launch_budget_bytes,
            "feature_ladder": [list(area) for area in self.feature_ladder],
        }

    @classmethod
    def from_parameters(cls, raw: dict) -> WordCapacityProtocol:
        """Decode a complete record; unknown and missing fields are errors."""
        expected = set(REGISTERED_PROTOCOL.to_parameters())
        if not isinstance(raw, dict) or set(raw) != expected:
            raise ValueError("word-capacity parameters must contain the complete protocol")
        definitions = raw["cell_definitions"]
        if not isinstance(definitions, dict):
            raise ValueError("cell_definitions must be an object")
        try:
            cells = tuple(
                CapacityCell(name, *values) for name, values in definitions.items()
            )
            return cls(
                cells=tuple(raw["cells"]),
                vocabulary_sizes=tuple(raw["vocabulary_sizes"]),
                feature_area=tuple(raw["feature_area"]),
                cell_definitions=cells,
                connection_probability=raw["connection_probability"],
                plasticity=raw["plasticity"],
                rounds_per_pair=raw["rounds_per_pair"],
                exposures_per_referent=raw["exposures_per_referent"],
                referents_per_scene=raw["referents_per_scene"],
                category_count=raw["category_count"],
                minimum_exposures=raw["minimum_exposures"],
                threshold=raw["threshold"],
                corpus_seed_offset=raw["corpus_seed_offset"],
                corpus_seed_scope=raw["corpus_seed_scope"],
                training_seed_offset=raw["training_seed_offset"],
                connectome_seed_stride=raw["connectome_seed_stride"],
                early_stop_margin=raw["early_stop_margin"],
                ceiling_interior_band=tuple(raw["ceiling_interior_band"]),
                w2_pass_tolerance=raw["w2_pass_tolerance"],
                w2_fail_tolerance=raw["w2_fail_tolerance"],
                w3_pass_ratio=raw["w3_pass_ratio"],
                w3_fail_ratio=raw["w3_fail_ratio"],
                launch_budget_bytes=raw["launch_budget_bytes"],
                feature_ladder=tuple(tuple(area) for area in raw["feature_ladder"]),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid word-capacity protocol: {exc}") from exc


REGISTERED_PROTOCOL = WordCapacityProtocol(
    cells=("A", "B", "C", "D", "E"),
    vocabulary_sizes=(16, 32, 64, 128, 256, 512, 1024),
    feature_area=(4000, 100),
    cell_definitions=(
        CapacityCell("A", 1000, 50, 50),
        CapacityCell("B", 2000, 50, 50),
        CapacityCell("C", 4000, 50, 50),
        CapacityCell("D", 4000, 100, 100),
        CapacityCell("E", 2000, 50, 100),
    ),
    connection_probability=0.05,
    plasticity=0.10,
    rounds_per_pair=2,
    exposures_per_referent=12,
    referents_per_scene=3,
    category_count=4,
    minimum_exposures=3,
    threshold=0.90,
    corpus_seed_offset=9001,
    corpus_seed_scope="per-brain",
    training_seed_offset=11,
    connectome_seed_stride=1000,
    early_stop_margin=0.30,
    ceiling_interior_band=(0.10, 0.90),
    w2_pass_tolerance=0.25,
    w2_fail_tolerance=0.40,
    w3_pass_ratio=1.30,
    w3_fail_ratio=1.10,
    launch_budget_bytes=4 << 30,
    feature_ladder=(
        (1000, 50), (2000, 50), (4000, 50),
        (8000, 50), (4000, 100), (8000, 100),
    ),
)

SMOKE_PROTOCOL = REGISTERED_PROTOCOL.select(
    cells=("A",), vocabulary_sizes=(8, 16), feature_area=(1000, 50),
)

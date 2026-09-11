"""Typed sparse attention over immutable assembly snapshots.

This is the first, deliberately small attention operator. It is a readout
instrument: it does not mutate a ``Brain`` or claim to learn query-key fibers.
Compatibility is assembly overlap, selection is deterministic top-k, and the
value side is an explicit weighted sparse aggregate.

Specification: neural_assemblies/ir/VERIFICATION.md#contract-assembly-attention
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Mapping

import numpy as np

from .assembly import Assembly, overlap
from .contracts import ATTENTION_CONTRACT, AttentionPlan, implements


def _positive_int(label: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return int(value)


@dataclass(frozen=True)
class AttentionCandidate:
    """One immutable key/value match in an attention result."""

    label: str
    compatibility: float
    weight: float


@dataclass(frozen=True)
class AttentionResult:
    """Inspectable sparse attention output and its normalized evidence."""

    query_area: str
    value_area: str
    candidates: tuple[AttentionCandidate, ...]
    selected_labels: tuple[str, ...]
    value: Assembly


@implements(ATTENTION_CONTRACT)
def attend(
    query: Assembly,
    keys: Mapping[str, Assembly],
    values: Mapping[str, Assembly],
    *,
    top_k: int = 1,
    output_size: int | None = None,
    temperature: float = 1.0,
) -> AttentionResult:
    """Attend from a query assembly to keyed value assemblies.

    Specification: neural_assemblies/ir/VERIFICATION.md#contract-assembly-attention

    ``keys`` and ``values`` share labels. Compatibility is
    ``overlap(query, key)``; weights are a stable softmax of compatibility
    divided by ``temperature``. The selected values are aggregated by weighted
    neuron support and truncated deterministically to ``output_size``.

    The function is pure: inputs are never mutated, and the result carries all
    scores needed to audit the readout. It is therefore suitable as a decoder
    baseline while a learned Brain-backed operator is developed separately.
    """
    if not isinstance(query, Assembly):
        raise TypeError("query must be an Assembly snapshot")
    if not query:
        raise ValueError("attention query must contain at least one neuron")
    if not isinstance(keys, Mapping) or not isinstance(values, Mapping):
        raise TypeError("keys and values must be mappings of labels to Assembly")
    if not keys:
        raise ValueError("attention requires at least one key")
    if set(keys) != set(values):
        raise ValueError("keys and values must have exactly the same labels")
    if any(not isinstance(assembly, Assembly)
           for assembly in (*keys.values(), *values.values())):
        raise TypeError("attention keys and values must be Assembly snapshots")
    if any(not assembly for assembly in (*keys.values(), *values.values())):
        raise ValueError("attention keys and values must be nonempty assemblies")
    top_k = _positive_int("top_k", top_k)
    if top_k > len(keys):
        raise ValueError("top_k cannot exceed the number of keys")
    if output_size is None:
        output_size = len(next(iter(values.values())))
    output_size = _positive_int("output_size", output_size)
    if (isinstance(temperature, bool) or not isinstance(temperature, Real)
            or not math.isfinite(float(temperature)) or temperature <= 0):
        raise ValueError("temperature must be a finite positive real")
    temperature = float(temperature)

    for label, assembly in (*keys.items(), *values.items()):
        if not isinstance(label, str) or not label:
            raise ValueError("attention labels must be nonempty strings")
        if not isinstance(assembly, Assembly):
            raise TypeError("attention keys and values must be Assembly snapshots")
        if not assembly:
            raise ValueError("attention keys and values must be nonempty assemblies")
    value_areas = {assembly.area for assembly in values.values()}
    if len(value_areas) != 1:
        raise ValueError("all attention values must belong to one area")
    value_area = next(iter(value_areas))
    key_areas = {assembly.area for assembly in keys.values()}
    if key_areas != {query.area}:
        raise ValueError("attention query and keys must share one area")
    plan = AttentionPlan(
        query=query,
        keys=tuple(keys.items()),
        values=tuple(values.items()),
        top_k=top_k,
        output_size=output_size,
        temperature=temperature,
    )

    scored = [(label, overlap(plan.query, key)) for label, key in plan.keys]
    scored.sort(key=lambda item: (-item[1], item[0]))
    logits = np.asarray([score / temperature for _, score in scored], dtype=float)
    logits -= float(np.max(logits))
    probabilities = np.exp(logits)
    probabilities /= float(np.sum(probabilities))
    candidates = tuple(
        AttentionCandidate(label, float(score), float(weight))
        for (label, score), weight in zip(scored, probabilities)
    )
    selected = candidates[:top_k]

    support: dict[int, float] = {}
    value_by_label = dict(plan.values)
    for candidate in selected:
        for neuron_id in value_by_label[candidate.label].neuron_ids:
            neuron = int(neuron_id)
            support[neuron] = support.get(neuron, 0.0) + candidate.weight
    ranked_neurons = sorted(support, key=lambda neuron: (-support[neuron], neuron))
    output = Assembly(value_area, ranked_neurons[:output_size])
    return AttentionResult(
        query_area=query.area,
        value_area=value_area,
        candidates=candidates,
        selected_labels=tuple(candidate.label for candidate in selected),
        value=output,
    )

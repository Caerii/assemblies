"""Typed sparse attention over immutable assembly snapshots.

This is the first, deliberately small attention operator. It is a readout
instrument: it does not mutate a ``Brain`` or claim to learn query-key fibers.
Compatibility is assembly overlap, selection is deterministic top-k, and the
value side is an explicit weighted sparse aggregate.

Specification: neural_assemblies/ir/VERIFICATION.md#contract-assembly-attention
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .assembly import Assembly, overlap
from ..core.index_spaces import NeuronIds
from .contracts import ATTENTION_CONTRACT, AttentionPlan, implements


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
    if not isinstance(keys, Mapping) or not isinstance(values, Mapping):
        raise TypeError("keys and values must be mappings of labels to Assembly")
    if set(keys) != set(values):
        raise ValueError("keys and values must have exactly the same labels")
    if any(not isinstance(assembly, Assembly)
           for assembly in (*keys.values(), *values.values())):
        raise TypeError("attention keys and values must be Assembly snapshots")
    # Canonicalize the value mapping to key order before constructing the plan.
    # The plan is the single validation boundary; this prevents the executable
    # and contract paths from drifting on labels, areas, and schedules.
    key_entries = tuple(keys.items())
    value_entries = tuple((label, values[label]) for label, _ in key_entries)
    if output_size is None:
        if value_entries and isinstance(value_entries[0][1], Assembly):
            output_size = len(value_entries[0][1])
    plan = AttentionPlan(
        query=query,
        keys=key_entries,
        values=value_entries,
        top_k=top_k,
        output_size=output_size,
        temperature=temperature,
    )
    value_area = plan.values[0][1].area
    temperature = float(plan.temperature)

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
    output = Assembly(value_area, NeuronIds(np.asarray(ranked_neurons[:plan.output_size], dtype=np.uint32)))
    return AttentionResult(
        query_area=query.area,
        value_area=value_area,
        candidates=candidates,
        selected_labels=tuple(candidate.label for candidate in selected),
        value=output,
    )

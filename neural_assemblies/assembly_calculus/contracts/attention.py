"""The attention plan and its contract.

Part of neural_assemblies.assembly_calculus.contracts."""
from dataclasses import dataclass
import math
from numbers import Integral, Real


from ..assembly import Assembly

from .contract import OperationContract


@dataclass(frozen=True)
class AttentionPlan:
    """Immutable readout schedule for sparse assembly attention."""

    query: Assembly
    keys: tuple[tuple[str, Assembly], ...]
    values: tuple[tuple[str, Assembly], ...]
    top_k: int = 1
    output_size: int | None = None
    temperature: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.query, Assembly) or not self.query:
            raise ValueError("attention query must be a nonempty Assembly")
        if not self.keys or not self.values:
            raise ValueError("attention requires nonempty keys and values")
        if tuple(label for label, _ in self.keys) != tuple(label for label, _ in self.values):
            raise ValueError("attention keys and values must have identical ordered labels")
        if len({label for label, _ in self.keys}) != len(self.keys):
            raise ValueError("attention labels must be unique")
        if any(not isinstance(label, str) or not label or not isinstance(assembly, Assembly)
               or not assembly for label, assembly in (*self.keys, *self.values)):
            raise ValueError("attention entries must be nonempty labeled Assemblies")
        if any(assembly.area != self.query.area for _, assembly in self.keys):
            raise ValueError("attention query and keys must share one area")
        if len({assembly.area for _, assembly in self.values}) != 1:
            raise ValueError("attention values must share one area")
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, Integral) or not 1 <= self.top_k <= len(self.keys):
            raise ValueError("attention top_k must be between one and the number of keys")
        if self.output_size is not None and (isinstance(self.output_size, bool)
                                              or not isinstance(self.output_size, Integral)
                                              or self.output_size < 1):
            raise ValueError("attention output_size must be a positive integer")
        if (isinstance(self.temperature, bool) or not isinstance(self.temperature, Real)
                or not math.isfinite(float(self.temperature)) or self.temperature <= 0):
            raise ValueError("attention temperature must be finite and positive")


ATTENTION_CONTRACT = OperationContract(
    operation_id="assembly-attention-v1",
    specification="neural_assemblies/ir/VERIFICATION.md#contract-assembly-attention",
    plan_type=AttentionPlan,
    inputs=("query", "labeled keys", "labeled values", "top_k", "output_size", "temperature"),
    reads=("query and key neuron IDs", "value neuron IDs"),
    mutates=("nothing; pure readout",),
    regime=("nonempty immutable assemblies", "shared query/key area", "shared value area"),
    observed_outcome=("ranked compatibility weights", "selected labels", "bounded sparse value assembly"),
    failure_conditions=("mismatched labels", "mixed areas", "invalid top_k/output_size/temperature"),
    constructed_controls=(
        "neural_assemblies/tests/test_attention_operator.py::test_attention_multi_key_output_is_deterministic_and_bounded",
    ),
    true_negative_controls=(
        "neural_assemblies/tests/test_attention_operator.py::test_attention_rejects_key_value_mismatch_and_mixed_value_areas",
    ),
)

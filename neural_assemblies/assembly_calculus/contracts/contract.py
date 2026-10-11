"""An operation's contract (OperationContract), the protocol of an operation that carries
one, and `implements`, which attaches a contract to an operation.

Part of neural_assemblies.assembly_calculus.contracts.
Specification: neural_assemblies/ir/VERIFICATION.md#contract-operation-objects"""
from dataclasses import dataclass, is_dataclass
from typing import Callable, ParamSpec, Protocol, TypeVar, cast




_P = ParamSpec("_P")


_R_co = TypeVar("_R_co", covariant=True)


@dataclass(frozen=True)
class OperationContract:
    """Reviewable scientific surface attached to an executable operation."""

    operation_id: str
    specification: str
    plan_type: type
    inputs: tuple[str, ...]
    reads: tuple[str, ...]
    mutates: tuple[str, ...]
    regime: tuple[str, ...]
    observed_outcome: tuple[str, ...]
    failure_conditions: tuple[str, ...]
    constructed_controls: tuple[str, ...]
    true_negative_controls: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.operation_id, str) or not self.operation_id:
            raise ValueError("operation contract requires a nonempty string ID")
        if not isinstance(self.specification, str) or "#contract-" not in self.specification:
            raise ValueError("operation contract requires an ID and specification anchor")
        plan_params = getattr(self.plan_type, "__dataclass_params__", None)
        if not is_dataclass(self.plan_type) or not getattr(plan_params, "frozen", False):
            raise ValueError("operation contract plan_type must be a frozen dataclass")
        surfaces = {
            "inputs": self.inputs,
            "reads": self.reads,
            "mutates": self.mutates,
            "regime": self.regime,
            "observed outcome": self.observed_outcome,
            "failure conditions": self.failure_conditions,
            "constructed controls": self.constructed_controls,
            "true-negative controls": self.true_negative_controls,
        }
        invalid = []
        for name, values in surfaces.items():
            if (
                not isinstance(values, tuple)
                or not values
                or any(not isinstance(value, str) or not value for value in values)
                or len(set(values)) != len(values)
            ):
                invalid.append(name)
        if invalid:
            raise ValueError(f"operation contract has invalid surfaces: {invalid}")


class ContractedOperation(Protocol[_P, _R_co]):
    """Callable carrying the inspectable contract attached by ``implements``."""

    operation_contract: OperationContract

    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _R_co: ...


def implements(
    contract: OperationContract,
) -> Callable[[Callable[_P, _R_co]], ContractedOperation[_P, _R_co]]:
    """Attach and validate the exact contract at the implementation boundary.

    A contract link is part of an operation's source-level meaning.  Checking
    it while decorating the function makes a missing link an import-time
    failure instead of allowing a semantically undocumented operation to run
    until a later repository-wide audit.
    """
    def decorate(operation: Callable[_P, _R_co]) -> ContractedOperation[_P, _R_co]:
        specification = contract.specification
        if specification not in (operation.__doc__ or ""):
            raise TypeError(
                f"{operation.__module__}.{operation.__qualname__} must link "
                f"its specification {specification!r} in its docstring"
            )
        contracted = cast(ContractedOperation[_P, _R_co], operation)
        contracted.operation_contract = contract
        return contracted
    return decorate

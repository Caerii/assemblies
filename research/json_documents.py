"""One loss-aware JSON boundary for run records and evidence comparison."""
import json
import math


def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate evidence key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"nonfinite JSON number: {value}")


def _finite_float(text):
    value = float(text)
    if not math.isfinite(value):
        raise ValueError("JSON number overflows binary64")
    if value == 0 and any(c in "123456789" for c in text.lower().split("e")[0]):
        raise ValueError("nonzero JSON number underflows binary64")
    return value


def load_document(path):
    """Specification: neural_assemblies/ir/VERIFICATION.md#contract-evidence-json"""
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_pairs,
                      parse_constant=_reject_constant, parse_float=_finite_float)


def encode_document(value):
    """Deterministic encoding also preserves JSON numeric and boolean types."""
    return json.dumps(value, indent=2, allow_nan=False, sort_keys=True) + "\n"

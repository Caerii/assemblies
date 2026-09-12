"""One loss-aware JSON boundary for run records and evidence comparison."""
import json
import math
import os
import tempfile


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


def decode_document(text):
    """Specification: neural_assemblies/ir/VERIFICATION.md#contract-evidence-json"""
    return json.loads(text, object_pairs_hook=unique_pairs,
                      parse_constant=_reject_constant, parse_float=_finite_float)


def load_document(path):
    return decode_document(path.read_text(encoding="utf-8"))


def encode_document(value):
    """Deterministic encoding also preserves JSON numeric and boolean types."""
    return json.dumps(value, indent=2, allow_nan=False, sort_keys=True) + "\n"


def snapshot_document(value):
    """Return an independent, canonical JSON-safe snapshot of *value*.

    The round trip deliberately goes through the same strict decoder used for
    files, so in-memory run records and observations obey the file boundary's
    finite-number and duplicate-key rules before a study can mutate state.
    """
    return decode_document(encode_document(value))


def write_new_document(path, value):
    """Specification: neural_assemblies/ir/VERIFICATION.md#contract-evidence-json

    Validate before filesystem mutation; exclusively create a UTF-8 document.
    """
    text = encode_document(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(text)
        stream.flush()


def write_checkpoint_document(path, value):
    """Atomically replace a resumable study checkpoint.

    Checkpoints are mutable by protocol: a study may persist after each cell
    and resume from the latest complete list. The explicit name keeps that
    exception separate from immutable result publication. Encoding happens
    before filesystem mutation; the temporary sibling is flushed and
    atomically replaced so readers observe either the old or new document.
    """
    text = encode_document(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w', encoding='utf-8', dir=path.parent,
            prefix=f'.{path.name}.', suffix='.tmp', delete=False,
        ) as stream:
            temporary = stream.name
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

"""R1o: blind atomic learning and held-out noncommutative operator composition.

The teacher is a private affine permutation on Z/nZ for each role:
    T_role(x) = a_role*x + b_role (mod n), gcd(a_role,n)=1.
Only atomic (cue,role,target) examples are disclosed at training time.
No multi-step sequence or composite target is ever trained.

Synthetic hypothesis-aware controls, NOT native Brain/neural compositionality.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np

ROLES = ("A", "B")
SEQUENCES = (("A",), ("B",), ("A", "B"), ("B", "A"),
             ("A", "B", "A"), ("B", "A", "B"),
             ("A", "A", "B", "B"), ("B", "A", "B", "A"))


@dataclass(frozen=True)
class Protocol:
    n: int = 64
    k: int = 8
    atomic_examples: int = 8
    test_cues: int = 12
    data_seed: int = 23        # private benchmark-generation seed
    model_seed: int = 8123     # only seed visible to learners
    roles: tuple[str, str] = ROLES

    def validate(self):
        if any(type(x) is not int for x in
               (self.n, self.k, self.atomic_examples, self.test_cues,
                self.data_seed, self.model_seed)):
            raise ValueError("integer geometry and seeds required")
        if not (16 <= self.n <= 512 and 2 <= self.k < self.n and
                self.atomic_examples >= 2 and self.test_cues >= 2):
            raise ValueError("invalid geometry")
        if self.data_seed < 0 or self.model_seed < 0 or self.data_seed == self.model_seed:
            raise ValueError("private and public seeds must differ and be nonnegative")
        if (type(self.roles) is not tuple or len(self.roles) != 2 or
                any(type(r) is not str or not r for r in self.roles) or
                self.roles[0] == self.roles[1]):
            raise ValueError("exactly two distinct roles required")
        if math.comb(self.n, self.k) < self.atomic_examples * 2 + self.test_cues:
            raise ValueError("insufficient distinct sparse assemblies")
        return self


@dataclass(frozen=True)
class PublicSpec:
    n: int
    k: int
    roles: tuple[str, str]
    seed: int


def public_view(protocol: Protocol):
    protocol.validate()
    return PublicSpec(protocol.n, protocol.k, protocol.roles, protocol.model_seed)


def _codes(seed: int, n: int, k: int, count: int):
    rng = np.random.default_rng(seed)
    generated, used = [], set()
    while len(generated) < count:
        x = np.sort(rng.choice(n, k, replace=False).astype(np.uint32))
        identity = tuple(map(int, x))
        if identity not in used:
            used.add(identity)
            generated.append(x)
    return generated


def _transform(code, parameters, n):
    a, b = parameters
    return np.sort(((a * np.asarray(code, dtype=np.int64) + b) % n).astype(np.uint32))


def _valid_output(x, n: int, k: int):
    if x is None:
        return None
    x = np.asarray(x)
    if (x.ndim != 1 or x.size != k or not np.issubdtype(x.dtype, np.integer)
            or (x < 0).any() or (x >= n).any() or len(np.unique(x)) != k):
        raise ValueError("predictor emitted invalid k-of-n neuron IDs")
    return np.sort(x.astype(np.uint32))


class PrivateDataset:
    def __init__(self, protocol: Protocol):
        self.p = protocol.validate()
        rng = np.random.default_rng(np.random.SeedSequence([protocol.data_seed, 101]))
        units = [a for a in range(1, protocol.n) if math.gcd(a, protocol.n) == 1 and a != 1]
        if not units:
            raise ValueError("no nontrivial affine units")
        # Noncommuting transformations are essential. A shift-only task is
        # commutative and could not detect operator-order mistakes.
        for _ in range(2048):
            params = {role: (int(rng.choice(units)), int(rng.integers(protocol.n)))
                      for role in protocol.roles}
            a, b = params[protocol.roles[0]]
            c, d = params[protocol.roles[1]]
            if (a * d + b - c * b - d) % protocol.n:
                break
        else:
            raise RuntimeError("failed to construct noncommuting role operators")
        self._params = params
        codes = _codes(int(np.random.SeedSequence([protocol.data_seed, 202]).generate_state(1)[0]),
                       protocol.n, protocol.k,
                       2 * protocol.atomic_examples + protocol.test_cues)
        self.atomic = {role: codes[i * protocol.atomic_examples:(i + 1) * protocol.atomic_examples]
                       for i, role in enumerate(protocol.roles)}
        self.test = codes[2 * protocol.atomic_examples:]
        self.sequences = tuple(tuple(protocol.roles[0] if r == "A" else protocol.roles[1] for r in seq)
                               for seq in SEQUENCES)

    def target(self, cue, sequence):
        out = np.asarray(cue).copy()
        for role in sequence:
            out = _transform(out, self._params[role], self.p.n)
        return out

    def manifest(self):
        train_bytes = b"".join(np.ascontiguousarray(x).tobytes()
                               for role in self.p.roles for x in self.atomic[role])
        test_bytes = b"".join(np.ascontiguousarray(x).tobytes() for x in self.test)
        truth_bytes = b"".join(self.target(x, seq).tobytes()
                               for seq in self.sequences for x in self.test)
        return {"train_cues_sha256": hashlib.sha256(train_bytes).hexdigest(),
                "test_cues_sha256": hashlib.sha256(test_bytes).hexdigest(),
                "hidden_targets_sha256": hashlib.sha256(truth_bytes).hexdigest(),
                "operator_commitment_sha256": hashlib.sha256(
                    json.dumps(self._params, sort_keys=True).encode()).hexdigest(),
                "atomic_writes": 2 * self.p.atomic_examples,
                "composite_writes": 0,
                "sequences": self.sequences}


class AffineRuleLearner:
    """Train-only maximum-overlap learner over an explicitly known family."""
    def __init__(self, cfg: PublicSpec):
        self.n, self.k, self.roles = cfg.n, cfg.k, cfg.roles
        aa = np.asarray([a for a in range(1, cfg.n) if math.gcd(a, cfg.n) == 1], dtype=np.int64)
        self.a = np.repeat(aa, cfg.n)
        self.b = np.tile(np.arange(cfg.n, dtype=np.int64), len(aa))
        self.evidence = {r: np.zeros(len(self.a), dtype=np.int64) for r in cfg.roles}
        self.count = {r: 0 for r in cfg.roles}
        self.learned = {r: None for r in cfg.roles}

    def train(self, role, cue, target):
        if role not in self.evidence:
            raise ValueError("unknown role")
        cue, target = _valid_output(cue, self.n, self.k), _valid_output(target, self.n, self.k)
        hits = np.zeros(self.n, dtype=np.int8)
        hits[target] = 1
        projected = (self.a[:, None] * cue[None, :] + self.b[:, None]) % self.n
        self.evidence[role] += hits[projected].sum(axis=1)
        self.count[role] += 1
        idx = int(np.argmax(self.evidence[role]))
        self.learned[role] = int(self.a[idx]), int(self.b[idx])

    def predict(self, roles, cue):
        current = _valid_output(cue, self.n, self.k)
        for role in roles:
            if self.learned.get(role) is None:
                return None
            current = _transform(current, self.learned[role], self.n)
        return current

    def fingerprint(self):
        h = hashlib.sha256()
        for role in self.roles:
            h.update(self.evidence[role].tobytes())
            h.update(str((self.count[role], self.learned[role])).encode())
        return h.hexdigest()

    def bytes_state(self):
        return sum(x.nbytes for x in self.evidence.values())


class OrderBlindAffine(AffineRuleLearner):
    """Correct atomic rule family, but destroys the order of multi-step roles."""
    def predict(self, roles, cue):
        return super().predict(tuple(sorted(roles)), cue)


class ShiftOnly(AffineRuleLearner):
    """Wrong model family (a=1), isolating affine multiplicative structure."""
    def __init__(self, cfg):
        super().__init__(cfg)
        self.a = np.ones(cfg.n, dtype=np.int64)
        self.b = np.arange(cfg.n, dtype=np.int64)
        self.evidence = {r: np.zeros(cfg.n, dtype=np.int64) for r in cfg.roles}


class HebbianPairs:
    """Dense local coactivation, sequential winner-to-winner composition."""
    def __init__(self, cfg):
        self.n, self.k, self.roles = cfg.n, cfg.k, cfg.roles
        self.weights = {r: np.zeros((cfg.n, cfg.n), dtype=np.float32) for r in cfg.roles}

    def train(self, role, cue, target):
        cue, target = _valid_output(cue, self.n, self.k), _valid_output(target, self.n, self.k)
        self.weights[role][np.ix_(cue, target)] += np.float32(1)

    def predict(self, roles, cue):
        current = _valid_output(cue, self.n, self.k)
        for role in roles:
            if role not in self.weights:
                return None
            drive = np.sum(self.weights[role][current], axis=0)
            order = np.lexsort((np.arange(self.n), -drive))
            current = np.sort(order[:self.k].astype(np.uint32))
        return current

    def fingerprint(self):
        return hashlib.sha256(b"".join(self.weights[r].tobytes() for r in self.roles)).hexdigest()

    def bytes_state(self):
        return sum(x.nbytes for x in self.weights.values())


class AtomicKeyMemory:
    """Exact atomic pair lookup: no interpolated outputs, no composed writes."""
    def __init__(self, cfg):
        self.codes = {}
        self.roles = cfg.roles
        self.n, self.k = cfg.n, cfg.k

    def train(self, role, cue, target):
        cue, target = _valid_output(cue, self.n, self.k), _valid_output(target, self.n, self.k)
        self.codes[role, tuple(map(int, cue))] = target.copy()

    def predict(self, roles, cue):
        current = _valid_output(cue, self.n, self.k)
        for role in roles:
            current = self.codes.get((role, tuple(map(int, current))))
            if current is None:
                return None
        return current.copy()

    def fingerprint(self):
        h = hashlib.sha256()
        for key, value in sorted(self.codes.items()):
            h.update(str(key).encode())
            h.update(value.tobytes())
        return h.hexdigest()

    def bytes_state(self):
        return len(self.codes) * self.k * 8 * 2  # lower-bound estimate incl. IDs


class CopyInput:
    def __init__(self, cfg):
        self.n, self.k = cfg.n, cfg.k
    def train(self, role, cue, target):
        pass
    def predict(self, roles, cue):
        return _valid_output(cue, self.n, self.k)
    def fingerprint(self):
        return "no_trainable_state"
    def bytes_state(self):
        return 0


FACTORIES = (("affine_rule", AffineRuleLearner),
             ("order_blind", OrderBlindAffine),
             ("shift_only", ShiftOnly),
             ("hebbian_pairs", HebbianPairs),
             ("atomic_lookup", AtomicKeyMemory),
             ("copy_input", CopyInput))


def run_one(protocol: Protocol, *, factories=None):
    ds = PrivateDataset(protocol)
    factories = FACTORIES if factories is None else tuple(factories)
    public = public_view(protocol)
    if not factories or len({name for name, _ in factories}) != len(factories):
        raise ValueError("nonempty uniquely named model factories required")
    models = {name: factory(public) for name, factory in factories}
    # The ONLY training channel: per-role atomics. No composed trajectories,
    # private operator parameters or evaluation labels are made available.
    for role in protocol.roles:
        for x in ds.atomic[role]:
            label = ds.target(x, (role,))
            for model in models.values():
                model.train(role, x.copy(), label.copy())
    records = []
    for name, model in models.items():
        before = model.fingerprint()
        for partition, cues, sequences in (
            ("atomic_train", [x for r in protocol.roles for x in ds.atomic[r]],
             [(r,) for r in protocol.roles for _ in ds.atomic[r]]),
            ("heldout_atomic", ds.test, ((r,) for r in protocol.roles)),
            ("heldout_composed", ds.test, (seq for seq in ds.sequences if len(seq) >= 2)),
        ):
            # materialize generators before nested loops
            sequences = tuple(sequences)
            if partition == "atomic_train":
                cases = zip(cues, sequences)
            else:
                cases = ((x, seq) for seq in sequences for x in cues)
            for cue, sequence in cases:
                expected = ds.target(cue, sequence)
                predicted = _valid_output(model.predict(sequence, cue.copy()), protocol.n, protocol.k)
                if model.fingerprint() != before:
                    raise AssertionError(f"prediction mutated learned model: {name}")
                overlap = 0.0 if predicted is None else len(np.intersect1d(
                    expected, predicted, assume_unique=True)) / protocol.k
                distractor_cues = (ds.atomic[sequence[0]] if partition == "atomic_train"
                                   else ds.test)
                other_targets = [ds.target(other, sequence) for other in distractor_cues
                                 if not np.array_equal(other, cue)]
                own_hits = 0 if predicted is None else int(np.intersect1d(
                    predicted, expected, assume_unique=True).size)
                best_other = max((int(np.intersect1d(predicted, other,
                                         assume_unique=True).size) for other in other_targets),
                                 default=-1) if predicted is not None else 0
                margin = own_hits - best_other
                records.append({"model": name, "partition": partition,
                                "sequence": ''.join(sequence), "depth": len(sequence),
                                "exact": int(predicted is not None and np.array_equal(predicted, expected)),
                                "overlap": float(overlap),
                                "strict_rank1": int(predicted is not None and margin > 0),
                                "margin": int(margin)})
    summary = []
    for name, model in models.items():
        for partition in ("atomic_train", "heldout_atomic", "heldout_composed"):
            sub = [r for r in records if r["model"] == name and r["partition"] == partition]
            summary.append({"model": name, "partition": partition, "cases": len(sub),
                            "exact": float(np.mean([r["exact"] for r in sub])),
                            "overlap": float(np.mean([r["overlap"] for r in sub])),
                            "strict_rank1": float(np.mean([r["strict_rank1"] for r in sub])),
                            "bytes_state": model.bytes_state()})
    order_pairs = []
    seq_ab = tuple(protocol.roles)
    seq_ba = tuple(reversed(protocol.roles))
    for name, model in models.items():
        state_before = model.fingerprint()
        for cue in ds.test:
            predicted_ab = _valid_output(model.predict(seq_ab, cue.copy()), protocol.n, protocol.k)
            predicted_ba = _valid_output(model.predict(seq_ba, cue.copy()), protocol.n, protocol.k)
            truth_ab = ds.target(cue, seq_ab)
            truth_ba = ds.target(cue, seq_ba)
            assert not np.array_equal(truth_ab, truth_ba), "task lost noncommutativity"
            order_pairs.append({"model": name, "distinguished": int(
                predicted_ab is not None and predicted_ba is not None and
                not np.array_equal(predicted_ab, predicted_ba)),
                "both_exact": int(predicted_ab is not None and predicted_ba is not None and
                    np.array_equal(predicted_ab, truth_ab) and np.array_equal(predicted_ba, truth_ba))})
        if model.fingerprint() != state_before:
            raise AssertionError(f"order probe mutated model: {name}")
    order_summary = [{"model": name,
                      "both_exact": float(np.mean([r["both_exact"] for r in order_pairs if r["model"] == name])),
                      "distinguished": float(np.mean([r["distinguished"] for r in order_pairs if r["model"] == name]))}
                     for name in models]
    manifest = ds.manifest()
    return {"status": "synthetic_noncommutative_composition_not_native_brain",
            "protocol": asdict(protocol), "dataset": manifest,
            "records": records, "summary": summary,
            "order_summary": order_summary,
            "limitations": [
                "Exact affine-role hypothesis is supplied to the affine learner.",
                "No composite targets or test cues are passed to learners during training.",
                "Data/learner code share a process; not an adversarial security boundary.",
                "Model memory budgets are not matched.",
                "No native Brain, Rust, FPGA, or multimodal foundation model results.",
            ]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--atomic-examples", type=int, default=8)
    p.add_argument("--test-cues", type=int, default=12)
    p.add_argument("--data-seed", type=int, default=23)
    p.add_argument("--model-seed", type=int, default=8123)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    obj = run_one(Protocol(n=a.n, k=a.k, atomic_examples=a.atomic_examples,
                           test_cues=a.test_cues, data_seed=a.data_seed,
                           model_seed=a.model_seed))
    payload = json.dumps(obj, indent=2, sort_keys=True) + "\n"
    if a.output:
        a.output.write_text(payload)
    else:
        print(payload)


if __name__ == "__main__":
    main()

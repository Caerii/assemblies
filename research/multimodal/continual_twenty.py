"""Twenty-round prequential shift-learning calibration.

Targets follow a hidden role-specific cyclic shift, changed after round 10.
All probes are unseen lexical codes, never passed into train. These are
synthetic rule-inference controls, NOT native assembly dynamics.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, asdict
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np

ROLES = ('SUBJ', 'OBJ')

@dataclass(frozen=True)
class Protocol:
    n: int = 128
    k: int = 8
    rounds: int = 20
    change_after: int = 10
    probes_per_role: int = 5
    seeds: tuple[int, ...] = tuple(range(20))
    learner_seed: int = 19331
    window: int = 3
    poison_rounds: tuple[int, ...] = (7, 16)

    def validate(self):
        if any(type(x) is not int for x in (self.n, self.k, self.rounds,
                   self.change_after, self.probes_per_role, self.window)):
            raise ValueError('all geometry and timing parameters must be integers')
        if not (2 <= self.k < self.n and self.rounds >= 4 and
                1 <= self.change_after < self.rounds and
                self.probes_per_role >= 2 and self.window >= 1):
            raise ValueError('invalid geometry or timeline')
        if math.comb(self.n, self.k) < self.rounds + len(ROLES) * self.probes_per_role:
            raise ValueError('insufficient distinct sparse codes for probes and train stream')
        if not self.seeds or len(self.seeds) != len(set(self.seeds)):
            raise ValueError('unique nonempty private seeds required')
        if (type(self.learner_seed) is not int or self.learner_seed < 0
                or any(type(s) is not int or s < 0 or s == self.learner_seed
                       for s in self.seeds)):
            raise ValueError('private and public seeds must be distinct')
        if len(set(self.poison_rounds)) != len(self.poison_rounds) or any(
                type(x) is not int or x < 1 or x > self.rounds
                for x in self.poison_rounds):
            raise ValueError('invalid corrupted-label rounds')
        return self


class ShiftLearner:
    """Explicit cyclic-shift hypothesis learner with train-only exemplars.

    Rule scores are consolidated at the WRITE boundary; predict is read-only
    and never silently retrains or mutates persistent inference state.
    """
    def __init__(self, n, k, window=None):
        self.n, self.k, self.window = n, k, window
        self.examples = {role: deque(maxlen=window) for role in ROLES}
        self.best_shift = {role: None for role in ROLES}

    def train(self, role, cue, label):
        self.examples[role].append((np.asarray(cue).copy(), np.asarray(label).copy()))
        scores = np.zeros(self.n, dtype=np.int32)
        candidate_shift = np.arange(self.n, dtype=np.int64)[:, None]
        for src, dst in self.examples[role]:
            mask = np.zeros(self.n, dtype=bool)
            mask[dst] = True
            scores += np.count_nonzero(mask[(src[None, :] + candidate_shift) % self.n], axis=1)
        self.best_shift[role] = int(np.argmax(scores))

    def state_digest(self):
        h = hashlib.sha256()
        for role in ROLES:
            h.update(str(self.best_shift[role]).encode())
            for cue, target in self.examples[role]:
                h.update(cue.tobytes())
                h.update(target.tobytes())
        return h.hexdigest()

    def predict(self, role, cue):
        best = self.best_shift[role]
        if best is None:
            return None
        return np.sort(((cue + best) % self.n).astype(np.uint32))


class HebbianPairs:
    """Plain content-addressed coactivation without shared shift hypothesis."""
    def __init__(self, n, k):
        self.n, self.k = n, k
        self.weights = {r: np.zeros((n, n), dtype=np.float32) for r in ROLES}

    def train(self, role, cue, label):
        self.weights[role][np.ix_(cue, label)] += np.float32(1.)

    def state_digest(self):
        h = hashlib.sha256()
        for role in ROLES:
            h.update(self.weights[role].tobytes())
        return h.hexdigest()

    def predict(self, role, cue):
        scores = np.sum(self.weights[role][cue], axis=0)
        ids = np.lexsort((np.arange(self.n), -scores))[:self.k]
        return np.sort(ids.astype(np.uint32))


class InputCopy:
    def state_digest(self):
        return hashlib.sha256(b"input-copy-no-learning").hexdigest()

    def train(self, role, cue, label):
        pass
    def predict(self, role, cue):
        return np.asarray(cue, dtype=np.uint32).copy()


def exact(prediction, target, k):
    if prediction is None:
        return 0.
    pred = np.asarray(prediction)
    if (pred.ndim != 1 or len(pred) != k or
            not np.issubdtype(pred.dtype, np.integer) or
            len(np.unique(pred)) != k):
        raise ValueError('invalid prediction cardinality')
    return float(np.array_equal(np.sort(pred), np.sort(target)))


def _cue(rng, n, k):
    return np.sort(rng.choice(n, k, replace=False).astype(np.uint32))


def trial(seed: int, protocol: Protocol):
    p = protocol
    # Private data stream belongs exclusively to the evaluator.
    rng = np.random.default_rng(seed)
    shifts = {period: {role: int(rng.integers(1, p.n)) for role in ROLES}
              for period in ('old', 'new')}
    for role in ROLES:
        if shifts['new'][role] == shifts['old'][role]:
            shifts['new'][role] = (shifts['old'][role] % (p.n - 1)) + 1
    # Probe inputs are entirely disjoint from training inputs.
    reserved = set()
    probes = {}
    for role in ROLES:
        probes[role] = []
        while len(probes[role]) < p.probes_per_role:
            candidate = _cue(rng, p.n, p.k)
            key = tuple(map(int, candidate))
            if key not in reserved:
                reserved.add(key)
                probes[role].append(candidate)
    training_cues = []
    observed_training_targets = []
    models = {
        'global_rule': ShiftLearner(p.n, p.k),
        'recent_rule': ShiftLearner(p.n, p.k, p.window),
        'last_rule': ShiftLearner(p.n, p.k, 1),
        'hebbian_pairs': HebbianPairs(p.n, p.k),
        'copy_input': InputCopy(),
    }

    def score_models(period):
        result = {}
        for name, model in models.items():
            state_before = model.state_digest()
            by_role = {}
            for role in ROLES:
                target_shift = shifts[period][role]
                by_role[role] = float(np.mean([
                    exact(model.predict(role, cue),
                          (cue + target_shift) % p.n, p.k)
                    for cue in probes[role]
                ]))
            if model.state_digest() != state_before:
                raise AssertionError(f"evaluation mutated learned state: {name}")
            result[name] = {'mean_exact': float(np.mean(list(by_role.values()))),
                            'per_role_exact': by_role}
        return result

    rows = []
    for round_id in range(1, p.rounds + 1):
        period = 'old' if round_id <= p.change_after else 'new'
        role = ROLES[(round_id - 1) % len(ROLES)]
        before = score_models(period)
        while True:
            cue = _cue(rng, p.n, p.k)
            key = tuple(map(int, cue))
            if key not in reserved:
                reserved.add(key)
                break
        training_cues.append(cue.copy())
        actual = np.sort(((cue + shifts[period][role]) % p.n).astype(np.uint32))
        label = actual.copy()
        poisoned = round_id in p.poison_rounds
        if poisoned:
            wrong = (shifts[period][role] % (p.n - 1)) + 1
            label = np.sort(((cue + wrong) % p.n).astype(np.uint32))
        observed_training_targets.append(label.copy())
        for model in models.values():
            model.train(role, cue.copy(), label.copy())
        after = score_models(period)
        old_recall = score_models('old') if period == 'new' else None
        rows.append({'seed': seed, 'round': round_id, 'period': period,
                     'trained_role': role, 'corrupted_label': poisoned,
                     'before': before, 'after': after,
                     'old_recall_after': old_recall})
    assert len(reserved) == p.rounds + len(ROLES) * p.probes_per_role, (
        'distinct train and withheld probe code invariant failed')
    training_bytes = b''.join(x.tobytes() for x in training_cues + observed_training_targets)
    probe_bytes = b''.join(x.tobytes() for value in probes.values() for x in value)
    return {'seed': seed, 'rows': rows, 'training_sha256': hashlib.sha256(training_bytes).hexdigest(),
            'probe_sha256': hashlib.sha256(probe_bytes).hexdigest(),
            'shifts_sha256': hashlib.sha256(json.dumps(shifts, sort_keys=True).encode()).hexdigest()}


def run(protocol: Protocol):
    p = protocol.validate()
    trials = [trial(s, p) for s in p.seeds]
    summary = []
    names = list(trials[0]['rows'][0]['after'])
    for round_id in range(1, p.rounds + 1):
        stage = [tr['rows'][round_id - 1] for tr in trials]
        for name in names:
            summary.append({'round': round_id,
                            'model': name,
                            'mean_before_exact': float(np.mean([r['before'][name]['mean_exact'] for r in stage])),
                            'mean_after_exact': float(np.mean([r['after'][name]['mean_exact'] for r in stage])),
                            'mean_old_retention_exact': (float(np.mean([
                                r['old_recall_after'][name]['mean_exact'] for r in stage]))
                                if round_id > p.change_after else None)})
    return {'status': 'synthetic_20_round_continual_shift_not_native_brain',
            'protocol': asdict(p), 'trials': trials, 'summary': summary,
            'limitations': [
                'Shared-rule learners explicitly assume cyclic shifts; no native Brain dynamics.',
                'Data and model share one Python process, so this is not adversarially isolated.',
                'All probes are withheld, but the small seed space permits brute force.',
                'The data stream and role shifts are synthetic; no language/vision grounding.',
                'Global and windowed learners have different history capacity and cost.',
            ]}


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--output', type=Path)
    a.add_argument('--seeds', type=int, nargs='+', default=list(range(20)))
    args = a.parse_args()
    result = run(Protocol(seeds=tuple(args.seeds)))
    data = json.dumps(result, sort_keys=True, indent=2) + '\n'
    if args.output:
        args.output.write_text(data)
    else:
        print(data)

if __name__ == '__main__':
    main()

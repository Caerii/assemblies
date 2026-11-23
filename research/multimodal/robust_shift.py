"""R1m: corruption-quarantined shift learning and decayed train-only evidence.

Uses R1l's exact stimulus stream and hidden probe protocol. These are
hypothesis-aware synthetic controls, NOT native Brain measurements.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np
from .continual_twenty import Protocol, ROLES, _cue, exact


def support(n, cue, label):
    """Training-only evidence for each candidate shared cyclic shift."""
    target = np.zeros(n, dtype=bool)
    target[label] = True
    return target[(cue[None, :] + np.arange(n)[:, None]) % n].sum(axis=1)


class Quarantine:
    def __init__(self, n, confirmations=2):
        if n < 3 or type(confirmations) is not int or confirmations < 1:
            raise ValueError('invalid quorum')
        self.n, self.confirmations = n, confirmations
        self.active = dict.fromkeys(ROLES)
        self.pending = dict.fromkeys(ROLES)
        self.count = dict.fromkeys(ROLES, 0)
        self.switches = dict.fromkeys(ROLES, 0)

    def train(self, role, cue, label):
        candidate = int(np.argmax(support(self.n, cue, label)))
        if self.active[role] is None:
            self.active[role] = candidate
        elif candidate == self.active[role]:
            self.pending[role], self.count[role] = None, 0
        elif self.pending[role] == candidate:
            self.count[role] += 1
            if self.count[role] >= self.confirmations:
                self.active[role] = candidate
                self.switches[role] += 1
                self.pending[role], self.count[role] = None, 0
        else:
            self.pending[role], self.count[role] = candidate, 1

    def predict(self, role, cue):
        shift = self.active[role]
        return None if shift is None else np.sort(((cue + shift) % self.n).astype(np.uint32))

    def state_digest(self):
        return hashlib.sha256(str((self.active, self.pending, self.count,
                                   self.switches)).encode()).hexdigest()


class Decayed:
    def __init__(self, n, gamma):
        if n < 3 or not 0 <= gamma < 1:
            raise ValueError('invalid gamma')
        self.n, self.gamma = n, gamma
        self.scores = {r: np.zeros(n, dtype=np.float64) for r in ROLES}
        self.trained = dict.fromkeys(ROLES, False)

    def train(self, role, cue, label):
        self.scores[role] *= self.gamma
        self.scores[role] += support(self.n, cue, label)
        self.trained[role] = True

    def predict(self, role, cue):
        if not self.trained[role]:
            return None
        return np.sort(((cue + int(np.argmax(self.scores[role]))) % self.n).astype(np.uint32))

    def state_digest(self):
        return hashlib.sha256(b''.join(self.scores[r].tobytes() for r in ROLES)
                              + bytes(self.trained.values())).hexdigest()


def trial(seed: int, p: Protocol):
    p.validate()
    rng = np.random.default_rng(seed)
    shifts = {period: {role: int(rng.integers(1, p.n)) for role in ROLES}
              for period in ('old', 'new')}
    for role in ROLES:
        if shifts['new'][role] == shifts['old'][role]:
            shifts['new'][role] = (shifts['old'][role] % (p.n - 1)) + 1
    reserved = set()
    probes = {}
    for role in ROLES:
        probes[role] = []
        while len(probes[role]) < p.probes_per_role:
            cue = _cue(rng, p.n, p.k)
            key = tuple(map(int, cue))
            if key not in reserved:
                reserved.add(key)
                probes[role].append(cue)
    models = {'quarantine2': Quarantine(p.n, 2),
              'decay_0.5': Decayed(p.n, .5),
              'decay_0.8': Decayed(p.n, .8)}
    cues, labels, rows = [], [], []

    def measure(period):
        res = {}
        for name, model in models.items():
            before = model.state_digest()
            by_role = {role: float(np.mean([
                exact(model.predict(role, cue),
                      (cue + shifts[period][role]) % p.n, p.k)
                for cue in probes[role]])) for role in ROLES}
            if before != model.state_digest():
                raise AssertionError('prediction mutated model: ' + name)
            res[name] = {'mean_exact': float(np.mean(list(by_role.values()))),
                         'per_role_exact': by_role}
        return res

    for r in range(1, p.rounds + 1):
        phase = 'old' if r <= p.change_after else 'new'
        role = ROLES[(r - 1) % len(ROLES)]
        before = measure(phase)
        while True:
            cue = _cue(rng, p.n, p.k)
            key = tuple(map(int, cue))
            if key not in reserved:
                reserved.add(key)
                break
        true_shift = shifts[phase][role]
        label = np.sort(((cue + true_shift) % p.n).astype(np.uint32))
        if r in p.poison_rounds:
            wrong = (true_shift % (p.n - 1)) + 1
            label = np.sort(((cue + wrong) % p.n).astype(np.uint32))
        cues.append(cue.copy()); labels.append(label.copy())
        for model in models.values():
            model.train(role, cue.copy(), label.copy())
        rows.append({'round': r, 'period': phase, 'role': role,
                     'corrupted_label': r in p.poison_rounds,
                     'before': before, 'after': measure(phase),
                     'old_recall_after': measure('old') if phase == 'new' else None})
    assert len(reserved) == p.rounds + len(ROLES)*p.probes_per_role
    return {'seed': seed, 'rows': rows,
            'training_sha256': hashlib.sha256(b''.join(x.tobytes() for x in cues + labels)).hexdigest(),
            'probe_sha256': hashlib.sha256(b''.join(x.tobytes() for role in ROLES for x in probes[role])).hexdigest(),
            'shifts_sha256': hashlib.sha256(json.dumps(shifts, sort_keys=True).encode()).hexdigest()}


def run(p: Protocol):
    trials = [trial(seed, p) for seed in p.validate().seeds]
    names = tuple(trials[0]['rows'][0]['after'])
    summary = []
    for r in range(1, p.rounds + 1):
        for name in names:
            values = [t['rows'][r-1] for t in trials]
            summary.append({'round': r, 'model': name,
                            'mean_after_exact': float(np.mean([v['after'][name]['mean_exact'] for v in values])),
                            'mean_old_retention_exact': (float(np.mean([
                                v['old_recall_after'][name]['mean_exact'] for v in values]))
                                if r > p.change_after else None)})
    return {'status': 'synthetic_robust_policy_not_native_brain',
            'protocol': asdict(p), 'trials': trials, 'summary': summary,
            'limitations': [
                'All learners assume cyclic shifts; native Brain is not evaluated.',
                'Quarantine confirmation count and exponential decays are fixed, not tuned blind.',
                'Two consecutive labels can be wrong; quarantine is not an error-correcting guarantee.',
                'The private rule/probe generation runs in the same process as the models.',
            ]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', nargs='+', type=int, default=list(range(20)))
    parser.add_argument('--output', type=Path)
    opts = parser.parse_args()
    data = json.dumps(run(Protocol(seeds=tuple(opts.seeds))), indent=2, sort_keys=True) + '\n'
    if opts.output:
        opts.output.write_text(data)
    else:
        print(data)

if __name__ == '__main__': main()

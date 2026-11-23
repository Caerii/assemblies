"""R1n: predeclared multi-change-point, corruption-sensitive streaming tests.

Synthetic cyclic-shift controls ONLY. No native Brain, NEMO, or multimodal
intelligence claim. Models receive current training (role, cue, noisy label)
but not private shift schedules, the corruption flag, or held-out probes.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .continual_twenty import ROLES, HebbianPairs, ShiftLearner, _cue, exact
from .robust_shift import Quarantine, Decayed


@dataclass(frozen=True)
class Scenario:
    name: str
    changes: tuple[int, ...] = ()  # first round using the new rule
    corrupt: tuple[int, ...] = ()

    def validate(self, rounds: int):
        if type(self.name) is not str or not self.name:
            raise ValueError('scenario name is required')
        for label, values, lo, hi in (
            ('changes', self.changes, 2, rounds),
            ('corrupt', self.corrupt, 1, rounds),
        ):
            if (any(type(v) is not int or v < lo or v > hi for v in values)
                    or tuple(sorted(set(values))) != values):
                raise ValueError(f'{label} must be unique sorted rounds in range')
        return self


SCENARIOS = (
    Scenario('stationary_clean'),
    Scenario('three_shifts_clean', (13, 27, 39)),
    Scenario('three_shifts_isolated', (13, 27, 39), (7, 18, 32, 44)),
    # Consecutive same-role corruptions exercise the failure mode of two-
    # confirmation quarantine, not just its successful one-error condition.
    Scenario('three_shifts_burst', (13, 27, 39), (18, 20, 22, 24)),
)


@dataclass(frozen=True)
class Protocol:
    n: int = 128
    k: int = 8
    rounds: int = 48
    probes_per_role: int = 5
    seeds: tuple[int, ...] = tuple(range(20))
    scenarios: tuple[Scenario, ...] = SCENARIOS

    def validate(self):
        for value in (self.n, self.k, self.rounds, self.probes_per_role):
            if type(value) is not int:
                raise ValueError('geometry must be integer')
        if not (2 <= self.k < self.n and self.rounds >= 4 and self.probes_per_role >= 2):
            raise ValueError('invalid experiment geometry')
        if math.comb(self.n, self.k) < self.rounds + len(ROLES) * self.probes_per_role:
            raise ValueError('insufficient distinct assemblies for disjoint train/probe split')
        if (not self.seeds or len(set(self.seeds)) != len(self.seeds)
                or any(type(s) is not int or s < 0 for s in self.seeds)):
            raise ValueError('unique nonnegative dataset seeds required')
        if (not self.scenarios or len(set(s.name for s in self.scenarios)) != len(self.scenarios)):
            raise ValueError('unique scenarios required')
        for scenario in self.scenarios:
            scenario.validate(self.rounds)
        return self


class DualTimescale:
    """Fast shift evidence with two consecutive confirmations to switch.

    It is a predefined hypothesis-aware synthetic control, not a neural
    controller. Its inference only uses state inferred from training labels.
    """
    def __init__(self, n: int, gamma: float = .5, confirmations: int = 2):
        if (type(n) is not int or n < 3 or not np.isfinite(gamma)
                or not 0 <= gamma < 1 or type(confirmations) is not int
                or confirmations < 1):
            raise ValueError('invalid dual-timescale parameters')
        self.n = n
        self.gamma = gamma
        self.confirmations = confirmations
        self.evidence = {r: np.zeros(n, dtype=np.float64) for r in ROLES}
        self.active = dict.fromkeys(ROLES)
        self.pending = dict.fromkeys(ROLES)
        self.count = dict.fromkeys(ROLES, 0)
        self.switches = dict.fromkeys(ROLES, 0)

    def train(self, role, cue, target):
        if role not in self.evidence:
            raise ValueError('unknown role')
        cue = np.asarray(cue)
        target = np.asarray(target)
        mask = np.zeros(self.n, dtype=np.int8)
        mask[target] = 1
        shifts = np.arange(self.n, dtype=np.int64)[:, None]
        obs = mask[(cue[None, :] + shifts) % self.n].sum(axis=1)
        self.evidence[role] = self.gamma * self.evidence[role] + obs
        proposed = int(np.argmax(self.evidence[role]))
        if self.active[role] is None:
            self.active[role] = proposed
        elif proposed == self.active[role]:
            self.pending[role] = None
            self.count[role] = 0
        elif proposed == self.pending[role]:
            self.count[role] += 1
            if self.count[role] >= self.confirmations:
                self.active[role] = proposed
                self.pending[role] = None
                self.count[role] = 0
                self.switches[role] += 1
        else:
            self.pending[role] = proposed
            self.count[role] = 1
            if self.confirmations == 1:
                self.active[role] = proposed
                self.pending[role] = None
                self.count[role] = 0
                self.switches[role] += 1

    def predict(self, role, cue):
        shift = self.active[role]
        return None if shift is None else np.sort(((cue + shift) % self.n).astype(np.uint32))

    def state_digest(self):
        h = hashlib.sha256()
        for r in ROLES:
            h.update(self.evidence[r].tobytes())
            h.update(str((self.active[r], self.pending[r], self.count[r], self.switches[r])).encode())
        return h.hexdigest()


def learners(n: int, k: int):
    """All hyperparameters fixed before the seed/scenario sweep."""
    return {
        'global_history': ShiftLearner(n, k),
        'last_example': ShiftLearner(n, k, 1),
        'recent_three': ShiftLearner(n, k, 3),
        'quarantine_two': Quarantine(n, 2),
        'decay_half': Decayed(n, .5),
        'decay_eight': Decayed(n, .8),
        'dual_timescale': DualTimescale(n, .5, 2),
        'hebbian_pairs': HebbianPairs(n, k),
    }


def _private_stream(seed: int, p: Protocol):
    """Same train/probe cue streams regardless of scenario and corruption."""
    # Explicitly disjoint RNGs prevent changes in hidden rules or probe count
    # from modifying which cues are shown in the training stream.
    shiftrng = np.random.default_rng(np.random.SeedSequence([seed, 101]))
    proberng = np.random.default_rng(np.random.SeedSequence([seed, 202]))
    trainrng = np.random.default_rng(np.random.SeedSequence([seed, 303]))
    nphase = 1 + max(len(s.changes) for s in p.scenarios)
    shifts = []
    for phase in range(nphase):
        row = {}
        for role in ROLES:
            candidate = int(shiftrng.integers(1, p.n))
            if phase and candidate == shifts[-1][role]:
                candidate = (candidate % (p.n - 1)) + 1
            row[role] = candidate
        shifts.append(row)
    reserved = set()
    probes = {}
    for role in ROLES:
        own = []
        while len(own) < p.probes_per_role:
            c = _cue(proberng, p.n, p.k)
            tup = tuple(map(int, c))
            if tup not in reserved:
                reserved.add(tup)
                own.append(c)
        probes[role] = own
    training = []
    for rnd in range(1, p.rounds + 1):
        while True:
            c = _cue(trainrng, p.n, p.k)
            tup = tuple(map(int, c))
            if tup not in reserved:
                reserved.add(tup)
                training.append(c)
                break
    assert len(reserved) == p.rounds + len(ROLES) * p.probes_per_role
    return shifts, probes, training


def trial(seed: int, p: Protocol, scenario: Scenario):
    p.validate()
    scenario.validate(p.rounds)
    shifts, probes, training = _private_stream(seed, p)
    models = learners(p.n, p.k)
    labels, rows = [], []
    for rnd in range(1, p.rounds + 1):
        role = ROLES[(rnd - 1) % len(ROLES)]
        phase = sum(rnd >= cp for cp in scenario.changes)
        true_shift = shifts[phase][role]
        cue = training[rnd - 1]
        corrupt = rnd in scenario.corrupt
        actual_label_shift = ((true_shift % (p.n - 1)) + 1) if corrupt else true_shift
        label = np.sort(((cue + actual_label_shift) % p.n).astype(np.uint32))
        labels.append(label.copy())
        for model in models.values():
            model.train(role, cue.copy(), label.copy())
        accuracy = {}
        for name, model in models.items():
            before = model.state_digest()
            per_role = {}
            for target_role in ROLES:
                rule = shifts[phase][target_role]
                per_role[target_role] = float(np.mean([
                    exact(model.predict(target_role, x), (x + rule) % p.n, p.k)
                    for x in probes[target_role]
                ]))
            if model.state_digest() != before:
                raise AssertionError(f'read mutated state: {name}')
            accuracy[name] = float(np.mean(list(per_role.values())))
        rows.append({'round': rnd, 'role': role, 'phase': phase,
                     'changed': rnd in scenario.changes,
                     'corrupted_label': corrupt,
                     'after_exact': accuracy})
    return {
        'seed': seed, 'scenario': scenario.name, 'rows': rows,
        'training_cues_sha256': hashlib.sha256(b''.join(x.tobytes() for x in training)).hexdigest(),
        'probe_cues_sha256': hashlib.sha256(b''.join(x.tobytes() for r in ROLES for x in probes[r])).hexdigest(),
        'observed_labels_sha256': hashlib.sha256(b''.join(x.tobytes() for x in labels)).hexdigest(),
        'rule_commitment_sha256': hashlib.sha256(json.dumps(shifts, sort_keys=True).encode()).hexdigest(),
    }


def _mean(values):
    return float(np.mean(values))


def run(p: Protocol):
    p.validate()
    trials = [trial(seed, p, scenario) for scenario in p.scenarios for seed in p.seeds]
    model_names = tuple(trials[0]['rows'][0]['after_exact'])
    summary = []
    for scenario in p.scenarios:
        tt = [t for t in trials if t['scenario'] == scenario.name]
        for name in model_names:
            per_seed = []
            for t in tt:
                post = [r['after_exact'][name] for r in t['rows']
                        if scenario.changes and r['round'] >= scenario.changes[0]]
                all_ = [r['after_exact'][name] for r in t['rows']]
                # For stable-only scenarios, use full-horizon performance.
                per_seed.append({'seed': t['seed'], 'overall_exact': _mean(all_),
                                 'postchange_exact': _mean(post) if post else None,
                                 'at_corruption_exact': _mean([
                                     r['after_exact'][name] for r in t['rows']
                                     if r['corrupted_label']]) if scenario.corrupt else None,
                                 'final_exact': float(all_[-1])})
            summary.append({'scenario': scenario.name, 'model': name,
                            'seeds': len(per_seed), 'overall_exact': _mean([
                                x['overall_exact'] for x in per_seed]),
                            'postchange_exact': (_mean([x['postchange_exact'] for x in per_seed])
                                                 if scenario.changes else None),
                            'final_exact': _mean([x['final_exact'] for x in per_seed]),
                            'per_seed': per_seed})
    return {'status': 'synthetic_multi_drift_stress_not_native_brain',
            'protocol': asdict(p), 'trials': trials, 'summary': summary,
            'limitations': [
                'All shift learners know the cyclic-shift hypothesis class.',
                'Scenario timeline and corruption points are preset, not exposed to any learner.',
                'Models and private dataset run in the same process (not an adversarial security boundary).',
                'Hyperparameters are not optimized on held-out probes; numbers are descriptive.',
                'No native Brain, Rust, or multimodal grounding measurements are made.',
            ]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path)
    ap.add_argument('--seeds', nargs='+', type=int, default=list(range(20)))
    opts = ap.parse_args()
    res = run(Protocol(seeds=tuple(opts.seeds)))
    txt = json.dumps(res, sort_keys=True, indent=2) + '\n'
    if opts.output:
        opts.output.write_text(txt, encoding='utf-8')
    else:
        print(txt)


if __name__ == '__main__':
    main()

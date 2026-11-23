"""R1o locked 20-seed geometry sweep, no native Brain execution."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .noncommutative_composition import Protocol, run_one, FACTORIES


@dataclass(frozen=True)
class Sweep:
    geometries: tuple[tuple[int, int], ...] = ((64, 8), (96, 8), (128, 12))
    seeds: tuple[int, ...] = tuple(range(20))
    model_seed: int = 8123
    atomic_examples: int = 8
    test_cues: int = 8

    def validate(self):
        if (not self.geometries or len(set(self.geometries)) != len(self.geometries)
                or not self.seeds or len(set(self.seeds)) != len(self.seeds)):
            raise ValueError('unique nonempty geometry and seeds required')
        for n, k in self.geometries:
            for seed in self.seeds:
                Protocol(n=n, k=k, atomic_examples=self.atomic_examples,
                         test_cues=self.test_cues, data_seed=seed,
                         model_seed=self.model_seed).validate()
        return self


def run(spec: Sweep):
    spec.validate()
    cells = []
    for n, k in spec.geometries:
        for seed in spec.seeds:
            r = run_one(Protocol(n=n, k=k, atomic_examples=spec.atomic_examples,
                                 test_cues=spec.test_cues, data_seed=seed,
                                 model_seed=spec.model_seed))
            metrics = {(v['model'], v['partition']): v for v in r['summary']}
            order = {v['model']: v for v in r['order_summary']}
            for name, _ in FACTORIES:
                cells.append({
                    'n': n, 'k': k, 'data_seed': seed, 'model': name,
                    'atomic_train_exact': metrics[name, 'atomic_train']['exact'],
                    'heldout_atomic_exact': metrics[name, 'heldout_atomic']['exact'],
                    'composed_exact': metrics[name, 'heldout_composed']['exact'],
                    'composed_overlap': metrics[name, 'heldout_composed']['overlap'],
                    'composed_strict_rank1': metrics[name, 'heldout_composed']['strict_rank1'],
                    'ordered_AB_BA_both_exact': order[name]['both_exact'],
                    'ordered_AB_BA_distinguished': order[name]['distinguished'],
                    'bytes_state': metrics[name, 'heldout_composed']['bytes_state'],
                    'training_cues_sha256': r['dataset']['train_cues_sha256'],
                    'test_cues_sha256': r['dataset']['test_cues_sha256'],
                    'hidden_targets_sha256': r['dataset']['hidden_targets_sha256'],
                    'hidden_operator_commitment_sha256': r['dataset']['operator_commitment_sha256'],
                })
    summary = []
    metric_names = ('atomic_train_exact', 'heldout_atomic_exact',
                    'composed_exact', 'composed_overlap', 'composed_strict_rank1',
                    'ordered_AB_BA_both_exact', 'ordered_AB_BA_distinguished')
    for n, k in spec.geometries:
        for name, _ in FACTORIES:
            subset = [c for c in cells if (c['n'], c['k'], c['model']) == (n,k,name)]
            summary.append({'n': n, 'k': k, 'model': name, 'seeds': len(subset),
                            **{field: {'mean': float(np.mean([row[field] for row in subset])),
                                       'min': float(np.min([row[field] for row in subset])),
                                       'max': float(np.max([row[field] for row in subset]))}
                               for field in metric_names}})
    packed = json.dumps(cells, sort_keys=True, separators=(',', ':')).encode()
    return {'status': 'synthetic_affine_composition_not_native_brain',
            'spec': asdict(spec), 'rows_sha256': hashlib.sha256(packed).hexdigest(),
            'cells': cells, 'summary': summary,
            'limitations': [
                'The affine-rule learner is supplied with the exact teacher hypothesis family.',
                'The teacher parameters and hidden test codes are never supplied to learned models.',
                'This experiment tests structural inference and multi-step composition, not native assembly intelligence.',
                'No memory/time/energy matching between baselines; no native Brain, Rust or FPGA results.',
                'No separate adversarial process isolation; seed ranges are small and deterministic.',
            ]}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seeds', nargs='+', type=int, default=list(range(20)))
    p.add_argument('--output', type=Path)
    a=p.parse_args()
    out=json.dumps(run(Sweep(seeds=tuple(a.seeds))),indent=2,sort_keys=True)+'\n'
    if a.output:
        a.output.write_text(out)
    else:
        print(out)

if __name__ == '__main__': main()

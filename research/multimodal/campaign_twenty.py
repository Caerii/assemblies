"""Exactly twenty preregistered synthetic continual-learning sensitivity stages.

Each stage runs 20 within-stage learning updates over a fixed seed cohort.
Not native Brain evidence. Never select or tune hyperparameters on held-out
results and then claim the resulting scores are independent validation.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import argparse
import hashlib
import json
from pathlib import Path

from research.multimodal.continual_twenty import Protocol, run
from research.multimodal.analyze_twenty import analyze


def plan(base: Protocol):
    variants = [
        ('00_baseline', {}),
        ('01_clean', {'poison_rounds': ()}),
        ('02_old_poison', {'poison_rounds': (7,)}),
        ('03_new_poison', {'poison_rounds': (16,)}),
        ('04_window_1', {'window': 1}),
        ('05_window_2', {'window': 2}),
        ('06_window_4', {'window': 4}),
        ('07_window_5', {'window': 5}),
        ('08_window_8', {'window': 8}),
        ('09_window_10', {'window': 10}),
        ('10_change_early', {'change_after': 6}),
        ('11_change_late', {'change_after': 14}),
        ('12_sparse_64_4', {'n': 64, 'k': 4}),
        ('13_sparse_64_8', {'n': 64, 'k': 8}),
        ('14_sparse_128_4', {'n': 128, 'k': 4}),
        ('15_sparse_128_16', {'n': 128, 'k': 16}),
        ('16_sparse_256_8', {'n': 256, 'k': 8}),
        ('17_sparse_256_16', {'n': 256, 'k': 16}),
        ('18_boundary_poison', {'poison_rounds': (9, 12)}),
        ('19_transition_poison', {'poison_rounds': (11, 12)}),
    ]
    if len(variants) != 20 or len({name for name,_ in variants}) != 20:
        raise AssertionError('campaign requires exactly 20 independent named stages')
    return [(name, replace(base, **changes).validate()) for name, changes in variants]


def campaign(base: Protocol, *, draws=500):
    base.validate()
    if base.rounds != 20 or base.change_after != 10:
        raise ValueError('campaign base must have 20 rounds and midpoint change')
    entries=[]
    for name, spec in plan(base):
        result=run(spec)
        stats=analyze(result, draws=draws)
        raw=json.dumps(result, sort_keys=True, separators=(',',':')).encode()
        by_model={model: [x for x in result['summary'] if x['model']==model]
                  for model in ('global_rule','recent_rule','last_rule','hebbian_pairs','copy_input')}
        entries.append({
            'name': name, 'spec': asdict(spec), 'raw_sha256': hashlib.sha256(raw).hexdigest(),
            'post_change_metrics':stats['per_model_post_change_mean_exact'],
            'paired_recent_minus_global': stats['paired_recent_minus_global'],
            'round_10_current': {model: rows[9]['mean_after_exact'] for model,rows in by_model.items()},
            'round_11_current': {model: rows[10]['mean_after_exact'] for model,rows in by_model.items()},
            'round_20_current': {model: rows[19]['mean_after_exact'] for model,rows in by_model.items()},
            'round_20_old_retention': {model: rows[19]['mean_old_retention_exact'] for model,rows in by_model.items()},
        })
    return {'status':'20_stage_synthetic_ablation_not_native_brain',
            'stages':entries, 'stage_count':len(entries),
            'total_learning_updates':len(entries)*len(base.seeds)*base.rounds,
            'limitations':[
                'All 20 stages are an exploratory parameter sensitivity sweep, not independent discoveries.',
                'The global and windowed shift-rule baselines know the correct hypothesis class.',
                'Paired seed-level bootstrap quantifies local synthetic variability only.',
                'An apparent best window chosen from these stages requires new held-out studies.',
                'Native Brain, Rust and FPGA performance are not tested.',
            ]}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path)
    p.add_argument('--seeds',nargs='+',type=int,default=list(range(20)))
    p.add_argument('--draws',type=int,default=500)
    args=p.parse_args()
    result=campaign(Protocol(seeds=tuple(args.seeds)),draws=args.draws)
    text=json.dumps(result, sort_keys=True,indent=2)+'\n'
    if args.output: args.output.write_text(text)
    else: print(text)

if __name__=='__main__': main()

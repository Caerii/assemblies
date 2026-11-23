"""Paired, seed-level analysis of twenty-round continual-learning diagnostics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np


def _per_seed(result, model, start=11, end=20, field='after'):
    p = result['protocol']
    if not 1 <= start <= end <= p['rounds']:
        raise ValueError('invalid analysis window')
    return np.array([
        np.mean([row[field][model]['mean_exact']
                 for row in trial['rows'] if start <= row['round'] <= end])
        for trial in result['trials']
    ], dtype=np.float64)


def paired_ci(diff, *, draws=2000, seed=0):
    values = np.asarray(diff, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError('bootstrap requires >=2 finite paired seed-level outcomes')
    if type(draws) is not int or draws < 100:
        raise ValueError('insufficient bootstrap draws')
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), size=(draws,len(values)))].mean(axis=1)
    return {'paired_mean_delta': float(values.mean()),
            'percentile_95_ci': [float(x) for x in np.quantile(samples, (.025,.975))],
            'seeds': len(values)}


def analyze(result, *, draws=2000, seed=0):
    required = {'global_rule', 'recent_rule', 'last_rule', 'hebbian_pairs', 'copy_input'}
    for trial in result['trials']:
        if len(trial['rows']) != result['protocol']['rounds']:
            raise ValueError('incomplete trial rows')
        for row in trial['rows']:
            if not required <= set(row['after']):
                raise ValueError('missing baseline arm')
    if len(result['trials']) < 2:
        raise ValueError('multiple seeds required for paired analysis')
    p = result['protocol']
    post_start = p['change_after']+1
    scored = {name: _per_seed(result,name,post_start,p['rounds'])
              for name in sorted(required)}
    return {
        'status':'descriptive_seed_paired_synthetic_not_native_brain',
        'window': [post_start,p['rounds']],
        'per_model_post_change_mean_exact': {
            name: float(values.mean()) for name,values in scored.items()},
        'paired_recent_minus_global': paired_ci(scored['recent_rule']-scored['global_rule'],draws=draws,seed=seed),
        'paired_recent_minus_hebbian': paired_ci(scored['recent_rule']-scored['hebbian_pairs'],draws=draws,seed=seed),
        'paired_recent_minus_last': paired_ci(scored['recent_rule']-scored['last_rule'],draws=draws,seed=seed),
        'limitations': ['Percentile interval is descriptive over sampled synthetic seeds, not a population guarantee.',
                        'Small task family and hypothesis-aware learners; no native Brain accuracy.',
                        'Do not treat per-round values as independent bootstrap samples.']}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('input', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--draws', type=int, default=2000)
    args=p.parse_args()
    report=analyze(json.loads(args.input.read_text()), draws=args.draws)
    rendered=json.dumps(report,sort_keys=True,indent=2)+'\n'
    if args.output: args.output.write_text(rendered)
    else: print(rendered)

if __name__=='__main__': main()

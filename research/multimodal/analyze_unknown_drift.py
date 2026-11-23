"""R1n paired-seed analysis; descriptive intervals, no significance claims."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np


def _index(report):
    observations = {}
    scenarios = {s['name'] for s in report['protocol']['scenarios']}
    seeds = set(report['protocol']['seeds'])
    for row in report['trials']:
        key = (row['scenario'], row['seed'])
        if key in observations or row['scenario'] not in scenarios or row['seed'] not in seeds:
            raise ValueError('missing or duplicate scenario/seed in report')
        observations[key] = row
    if set(observations) != {(scenario, seed) for scenario in scenarios for seed in seeds}:
        raise ValueError('incomplete paired scenario/seed grid')
    for seed in seeds:
        trials = [observations[s, seed] for s in sorted(scenarios)]
        for key in ('training_cues_sha256', 'probe_cues_sha256', 'rule_commitment_sha256'):
            if len({x[key] for x in trials}) != 1:
                raise AssertionError(f'paired scenario mismatch for {key} seed={seed}')
    return observations


def _post(trial, model, change):
    values = [r['after_exact'][model] for r in trial['rows'] if r['round'] >= change]
    if not values:
        raise ValueError('empty post-change window')
    return float(np.mean(values))


def _paired_interval(differences, *, bootstrap_seed=1791, replicates=3000):
    x = np.asarray(differences, dtype=np.float64)
    if x.ndim != 1 or not x.size or not np.isfinite(x).all():
        raise ValueError('nonempty finite paired differences required')
    rng = np.random.default_rng(bootstrap_seed)
    means = x[rng.integers(0, len(x), size=(replicates, len(x)))].mean(axis=1)
    return {'mean_difference': float(x.mean()),
            'descriptive_bootstrap_95pct': [float(v) for v in np.quantile(means, [.025, .975])],
            'seed_level_differences': x.tolist()}


def analyze(report):
    indexed = _index(report)
    scenarios = {s['name']: s for s in report['protocol']['scenarios']}
    require = {'three_shifts_clean', 'three_shifts_isolated', 'three_shifts_burst'}
    if not require <= scenarios.keys():
        raise ValueError('paired corruption comparison needs three drift scenarios')
    if len({tuple(scenarios[name]['changes']) for name in require}) != 1:
        raise ValueError('change points differ; cannot interpret corruption delta')
    first_change = scenarios['three_shifts_clean']['changes'][0]
    seeds = tuple(report['protocol']['seeds'])
    model_names = list(indexed['three_shifts_clean', seeds[0]]['rows'][0]['after_exact'])
    comparisons = []
    for name in model_names:
        clean = [_post(indexed['three_shifts_clean', seed], name, first_change) for seed in seeds]
        for scenario in ('three_shifts_isolated', 'three_shifts_burst'):
            damaged = [_post(indexed[scenario, seed], name, first_change) for seed in seeds]
            comparisons.append({'model': name, 'scenario': scenario, 'reference': 'three_shifts_clean',
                                **_paired_interval([a - b for a, b in zip(damaged, clean)])})
    change_profiles = []
    for name in model_names:
        for scenario_name in require:
            for offset in range(6):
                values = [
                    indexed[scenario_name, seed]['rows'][point - 1 + offset]['after_exact'][name]
                    for seed in seeds for point in scenarios[scenario_name]['changes']
                    if point - 1 + offset < len(indexed[scenario_name, seed]['rows'])
                ]
                change_profiles.append({'model': name, 'scenario': scenario_name,
                                        'rounds_after_change': offset,
                                        'mean_exact': float(np.mean(values)),
                                        'cells': len(values)})
    return {'status': 'descriptive_paired_synthetic_analysis_not_native_brain',
            'seeds': list(seeds), 'paired_corruption_effects': comparisons,
            'change_profiles': change_profiles,
            'limitations': [
                'Bootstrap intervals resample synthetic seeds; they are not proof of real-world generalization.',
                'Corruption schedules are fixed rather than independently randomized per seed.',
                'Per-round observations share models and are not independent bootstrap units.',
                'The shift hypothesis family is known to synthetic learners.',
            ]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('input', type=Path)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    res = analyze(json.loads(args.input.read_text(encoding='utf-8')))
    out = json.dumps(res, sort_keys=True, indent=2) + '\n'
    if args.output:
        args.output.write_text(out, encoding='utf-8')
    else:
        print(out)


if __name__ == '__main__':
    main()

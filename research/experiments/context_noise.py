"""Registered context-readout noise sweep using immutable runner provenance."""
from dataclasses import asdict
from pathlib import Path

from neural_assemblies import Brain
from neural_assemblies.assembly_calculus import AttractorConfig, ContextAttractorChoice, ContextChoiceProtocol
from neural_assemblies.diagnostics import ensemble_from_values
from research.runner import experiment_parser, run_experiment

REGISTRATION = "research/notes/memory/PREREG_context_noise.md"
SEEDS = list(range(101, 121))
NOISE = [0., .1, .3, 1., 3., 10., 30., 100., 1000.]
BASE = ContextChoiceProtocol(400, 100, ('left', 'right'), ((4, 0), (0, 4)),
                             AttractorConfig(2000, 200, 3., rounds_train=10), 1., 3, 0.)
BARS = {'accuracy_low': .90, 'overlap_low': .80, 'null_joint_high': .90,
        'extreme_overlap_high': .40}


def parameters(smoke=False):
    return {'p': .05, 'protocol': asdict(BASE), 'read_seeds': [700, 701] if smoke else list(range(700, 710)),
            'bars': BARS, 'cells': [
                {'id': f'noise-{noise:g}', 'noise': noise, 'coupling_beta': 1., 'context_enabled': True}
                for noise in ([0., 1000.] if smoke else NOISE)] + [
                {'id': 'zero-coupling', 'noise': 0., 'coupling_beta': 0., 'context_enabled': True},
                {'id': 'context-disabled', 'noise': 0., 'coupling_beta': 1., 'context_enabled': False}]}


def summarize(rows, seeds):
    """Specification: research/notes/memory/PREREG_context_noise.md"""
    if [row['seed'] for row in rows] != seeds:
        raise ValueError('rows must preserve the registered unique brain identities')
    summaries = {}
    for metric in ('accuracy', 'target_overlap', 'joint_success'):
        values = [row[metric] for row in rows]
        if any(type(value) not in (int, float) or not 0 <= value <= 1 for value in values):
            raise ValueError('per-brain metrics must be finite proportions')
        result = ensemble_from_values(values, metric, keys=seeds)
        summaries[metric] = {**asdict(result), 'low': result.low, 'high': result.high}
    return summaries


def judge(cells, bars):
    """Specification: research/notes/memory/PREREG_context_noise.md"""
    indexed = {cell['id']: cell['summary'] for cell in cells}
    if len(indexed) != len(cells):
        raise ValueError('duplicate cell identities')
    checks = {}
    for name in ('noise-0', 'noise-1'):
        checks[name + '-accuracy'] = indexed[name]['accuracy']['low'] > bars['accuracy_low']
        checks[name + '-overlap'] = indexed[name]['target_overlap']['low'] > bars['overlap_low']
    for name in ('zero-coupling', 'context-disabled'):
        checks[name] = indexed[name]['joint_success']['high'] < bars['null_joint_high']
    checks['extreme-noise'] = indexed['noise-1000']['target_overlap']['high'] < bars['extreme_overlap_high']
    return checks


def experiment(record):
    config, seeds = record['parameters'], record['seeds']
    cells = []
    for cell in config['cells']:
        values = dict(config['protocol'], noise_std=cell['noise'], coupling_beta=cell['coupling_beta'])
        values['attractors'] = AttractorConfig(**values['attractors'])
        protocol = ContextChoiceProtocol(**values)
        rows = []
        for seed in seeds:
            model = ContextAttractorChoice(Brain(p=config['p'], seed=seed, engine=record['engine']), protocol=protocol)
            observations = []
            for read_seed in config['read_seeds']:
                for target, context in enumerate(protocol.contexts):
                    observed = model.observe(context, seed=read_seed, context_enabled=cell['context_enabled'])
                    observations.append({'read_seed': read_seed, 'context': context, 'target': target,
                                         **asdict(observed), 'margin': observed.margin})
            count = len(observations)
            rows.append({'seed': seed, 'observations': observations,
                         'accuracy': sum(row['label'] == row['target'] for row in observations) / count,
                         'target_overlap': sum(row['overlaps'][row['target']] for row in observations) / count,
                         'joint_success': sum(row['label'] == row['target'] and
                                              row['overlaps'][row['target']] > .8 and row['margin'] > .5
                                              for row in observations) / count})
        cells.append({**cell, 'rows': rows, 'summary': summarize(rows, seeds)})
        print(cell['id'], cells[-1]['summary']['accuracy']['mean'], flush=True)
    checks = {} if record['mode'] == 'smoke' else judge(cells, config['bars'])
    return {'cells': cells, 'checks': checks, 'verdict': 'VOID' if record['mode'] == 'smoke' else
            ('PASS' if all(checks.values()) else 'FAIL'), 'scope': 'preselected noise-1 hypothesis only; other levels descriptive'}


def main(argv=None):
    parser = experiment_parser(__doc__, engines=('torch_sparse', 'numpy_sparse'), default_seeds=tuple(SEEDS))
    args = parser.parse_args(argv)
    if not args.smoke and args.seeds != SEEDS:
        parser.error('study requires brain seeds 101..120 in order')
    if args.smoke and len(args.seeds) != 3:
        parser.error('smoke requires exactly three explicit seeds')
    print(run_experiment(script=Path(__file__), protocol='memory.context-noise', protocol_version='1',
                         registration=REGISTRATION, engine=args.engine, seeds=args.seeds, tag=args.tag,
                         smoke=args.smoke, minimum_study_seeds=20, parameters=parameters(args.smoke), measure=experiment))


if __name__ == '__main__':
    main()

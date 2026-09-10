"""Small CPU-only observations for the audit, not scientific measurements.

Run with an existing Python environment; imports are pinned to this worktree.
No downloads, GPU launches, or experiment output files are used.
"""
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.meta_path = [f for f in sys.meta_path
                 if '__editable__' not in getattr(f, '__module__', '')]
sys.path.insert(0, str(ROOT))

import numpy as np
import neural_assemblies

assert Path(neural_assemblies.__file__).is_relative_to(ROOT)

from neural_assemblies.compute.hyperdimensional import FinalFixedHyperdimensionalAssembly
from neural_assemblies.assembly_calculus.emergent.training.perf import resolve_engine
from neural_assemblies.diagnostics import ensemble_from_values
from research.experiments.base import summarize
from research.harness import study, Criteria


def main():
    observations = {}
    hdc = FinalFixedHyperdimensionalAssembly(None, dimension=16,
                                            rng=np.random.default_rng(42))
    ids = np.array([1, 2, 3, 4])
    observations['hdc_roundtrip'] = hdc.decode_hypervector_to_assembly(
        hdc.encode_assembly_as_hypervector(ids), k=4).tolist()
    with patch.dict(os.environ, {'ASSEMBLIES_ENGINE': 'numpy_sparse'}):
        observations['explicit_numpy_exact_resolves_to'] = resolve_engine('numpy_exact')

    # Stub provenance to avoid parser-cache construction: this probe concerns
    # verdict and seed handling only, not the separate fingerprint finding.
    with patch('research.harness._provenance_snapshot', return_value=('', {})):
        for label, seeds, arms in [
            ('empty_study', [1, 2, 3], {'a': lambda s: {}, 'b': lambda s: {}}),
            ('duplicate_seeds', [1, 1, 1], {'a': lambda s: {'m': float(s)},
                                           'b': lambda s: {'m': float(s + 1)}}),
        ]:
            try:
                result = study(arms=arms, seeds=seeds,
                               criteria={'required_metric': Criteria(above=.5)})
                observations[label] = {'passed': result.passed}
            except ValueError as exc:
                observations[label] = {'refused': str(exc)}
    values = list(range(20))
    old = summarize(values)
    observations['base_summary_ci_halfwidth'] = (old['ci95_hi'] - old['ci95_lo']) / 2
    observations['diagnostics_ci_halfwidth'] = ensemble_from_values(values).ci

    from neural_assemblies.programs import colt_mnist_brain as mnist
    def result(score, backend):
        return SimpleNamespace(mean_accuracy=score, per_class_accuracy=np.array([score]),
                               data_source='audit_stub', parameters={}, backend=backend)
    with patch.object(mnist, 'run_colt_mnist_protocol', return_value=result(.9, 'protocol')), \
         patch.object(mnist, 'run_colt_mnist_brain_explicit', return_value=result(.1, 'explicit')):
        selected = mnist.run_colt_mnist_brain()
        observations['mnist_disagreement_selected'] = {'backend': selected.backend,
                                                       'score': selected.mean_accuracy}
    try:
        from neural_assemblies.assembly_calculus.batched_trainer import BatchedSeqTrainer
        trainer = BatchedSeqTrainer(16, 4, ['a', 'b'], p=0, device='cpu')
        observations['batched_p0_edges_before'] = int(trainer.W.count_nonzero())
        trainer.train([['a', 'b']])
        observations['batched_p0_edges_after'] = int(trainer.W.count_nonzero())
    except ImportError as exc:
        observations['batched_probe_unavailable'] = str(exc)
    print(json.dumps(observations, indent=2))


if __name__ == '__main__':
    main()

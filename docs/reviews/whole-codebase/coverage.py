"""Compact current-commit coverage manifest; does not certify semantic review."""
import ast
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SOURCE = {'.py', '.rs', '.cc', '.cpp', '.cu', '.h', '.hpp', '.jl', '.m', '.ps1', '.bat', '.sh'}
# Conservative: only the additional review's principal source-evidence files.
# Other files were read, but absence here avoids claiming full-file scrutiny.
INSPECTED = '''research/harness.py
research/experiments/base.py
neural_assemblies/diagnostics.py
neural_assemblies/core/brain.py
neural_assemblies/core/area.py
neural_assemblies/core/engine.py
neural_assemblies/core/connectome.py
neural_assemblies/core/_homeostasis.py
neural_assemblies/theory.py
neural_assemblies/assembly_calculus/assembly.py
neural_assemblies/assembly_calculus/ops.py
neural_assemblies/assembly_calculus/batched_trainer.py
neural_assemblies/assembly_calculus/fsm.py
neural_assemblies/assembly_calculus/emergent/parser.py
neural_assemblies/assembly_calculus/emergent/parser_mixins/blocks.py
neural_assemblies/assembly_calculus/emergent/parser_mixins/instructions.py
neural_assemblies/assembly_calculus/emergent/training/perf.py
neural_assemblies/assembly_calculus/emergent/training/schedule.py
neural_assemblies/assembly_calculus/emergent/evaluation/sweep.py
neural_assemblies/assembly_calculus/emergent/evaluation/erp/protocol.py
neural_assemblies/compute/hyperdimensional.py
neural_assemblies/programs/planning.py
neural_assemblies/programs/tm_demo.py
neural_assemblies/programs/colt_mnist_brain.py
neural_assemblies/programs/colt_mnist_data.py
neural_assemblies/programs/vision_data.py
neural_assemblies/nemo/core/brain.py
neural_assemblies/nemo/core/area.py
neural_assemblies/ir/protocol.py
neural_assemblies/parity/paths.py
neural_assemblies/tests/test_ac_conformance.py
neural_assemblies/tests/test_ensemble_helpers.py
crates/assembly-ir/src/lib.rs
crates/na-kernels/src/lib.rs
cpp/python_implementations/billion_scale/billion_scale_cuda_brain.py
cpp/cuda_kernels/simple_cuda_brain.cu
pyproject.toml
.github/workflows/publish.yml
docs/supported_surfaces.md'''.splitlines()


def main():
    paths = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    rows, counts, source_lines = [], Counter(), Counter()
    for rel in sorted(filter(None, paths)):
        path = ROOT / rel
        if not path.is_file():
            rows.append([rel, '', 'missing', '', 'not reviewed'])
            continue
        data = path.read_bytes()
        kind = path.suffix
        status = 'bytes hashed'
        try:
            content = data.decode('utf-8-sig')
            lines = len(content.splitlines())
        except UnicodeDecodeError:
            content, lines = None, 0
        if kind == '.py' and content is not None:
            try:
                ast.parse(content, filename=rel)
                status = 'Python AST parsed'
            except SyntaxError as exc:
                status = 'syntax error: ' + str(exc)
        if kind == '.ipynb' and content is not None:
            cells = json.loads(content).get('cells', [])
            status = 'notebook JSON parsed; code not executed'
            counts['notebook_code_cells'] += sum(c.get('cell_type') == 'code' for c in cells)
        counts[kind or 'extensionless'] += 1
        counts[status] += 1
        if kind in SOURCE:
            source_lines[rel.split('/')[0]] += lines
        scope = 'selected source inspected' if rel in INSPECTED else 'structural inventory only'
        rows.append([rel, hashlib.sha256(data).hexdigest(), status, lines, scope])
    with (OUT / 'coverage.tsv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['path', 'sha256', 'structural_check', 'lines', 'semantic_scope'])
        writer.writerows(rows)
    summary = {'baseline': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
               'tracked_files': len(rows), 'counts': dict(counts),
               'source_lines_by_root': dict(source_lines),
               'scope': 'Current tracked bytes, not new untracked audit artifacts; no full semantic certification.'}
    (OUT / 'coverage-summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

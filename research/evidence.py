"""Validate runner artifacts and inventory unresolved historical evidence links.

Run `python -m research.evidence audit` to list candidate orphan results and
unresolved references. The audit is an inventory, not a validity verdict.
Run `python -m research.evidence validate PATH` for a runner results file.
"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import json
from pathlib import Path
import re
import subprocess

from neural_assemblies.core.environment import ENVIRONMENT_POLICY, ENVIRONMENT_PREFIXES

from research.json_documents import encode_document, load_document
from research.source_archive import validate_source_archive

ROOT = Path(__file__).resolve().parents[1]
_FILE_REF = re.compile(r'(?<![\w/])(?:[\w.-]+/)*[\w.-]+\.(?:py|md|json|csv|ipynb)(?![\w])')


def validate_artifact(path: Path, *, root: Path = ROOT) -> list[str]:
    """Validate run identity and file edges without mistaking completion for adoption."""
    errors = []
    try:
        payload = load_document(path)
        record = payload['run']
        original = load_document(path.parent / 'run.json')
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return [f'{path}: unreadable run artifact: {exc}']
    if not isinstance(record, dict) or encode_document(record) != encode_document(original):
        return ['embedded run record differs from the reserved run.json']
    required = {'schema_version', 'script', 'script_sha256', 'git_commit', 'source_sha256',
                'registration', 'registration_sha256', 'protocol', 'protocol_version',
                'engine', 'seeds', 'tag', 'parameters', 'mode', 'scientific_status'}
    missing = required - record.keys()
    if missing:
        return [f'missing run fields: {sorted(missing)}']
    if type(record['schema_version']) is not int or record['schema_version'] not in (1, 2, 3):
        errors.append('unsupported run schema version')
    if record['schema_version'] in (2, 3) or 'environment' in record:
        environment = record.get('environment')
        if (not isinstance(environment, dict)
                or set(environment) != {'policy', 'variables_sha256'}
                or environment['policy'] != ENVIRONMENT_POLICY
                or not isinstance(environment['variables_sha256'], dict)):
            errors.append('environment must contain the supported policy and variable digests')
        elif any(not name.startswith(ENVIRONMENT_PREFIXES)
                 or not isinstance(digest, str) or not re.fullmatch('[a-f0-9]{64}', digest)
                 for name, digest in environment['variables_sha256'].items()):
            errors.append('environment variables must name repository settings with SHA-256 digests')
    if any(not isinstance(record[field], str) or not record[field]
           for field in ('script', 'registration', 'engine', 'protocol', 'protocol_version',
                         'tag', 'mode', 'scientific_status')):
        return errors + ['file references and run identities must be nonempty strings']
    if not re.fullmatch('[a-f0-9]{40}|[a-f0-9]{64}', str(record['git_commit'])):
        errors.append('git_commit must be a full commit identity')
    if not isinstance(record['parameters'], dict):
        errors.append('parameters must be a mapping')
    if record['engine'] == 'auto':
        errors.append('engine must be resolved, not auto')
    if path.parent.name != record['tag'] or path.parent.parent.name != record['protocol']:
        errors.append('artifact directory does not match protocol and tag')
    for field in ('script', 'registration'):
        target = (root / record[field]).resolve()
        if not target.is_relative_to(root.resolve()) or not target.is_file():
            errors.append(f'dangling {field} edge: {record[field]}')
    for field in ('script_sha256', 'source_sha256', 'registration_sha256'):
        if not re.fullmatch('[a-f0-9]{64}', str(record[field])):
            errors.append(f'{field} is not a SHA-256 digest')
    inputs = record.get('input_artifacts', {})
    if not isinstance(inputs, dict):
        errors.append('input_artifacts must map file paths to content digests')
    else:
        for name, digest in inputs.items():
            target = (root / name).resolve()
            if not target.is_relative_to(root.resolve()) or not target.is_file():
                errors.append(f'dangling input artifact edge: {name}')
            if not re.fullmatch('[a-f0-9]{64}', str(digest)):
                errors.append(f'invalid input artifact digest: {name}')
    if record['schema_version'] == 3 or 'source_archive' in record:
        errors.extend(validate_source_archive(path.parent, record))
    seeds = record['seeds']
    if not isinstance(seeds, list) or any(type(s) is not int for s in seeds):
        errors.append('seeds must be a list of integer identities')
    elif len(seeds) < 3 or len(set(seeds)) != len(seeds):
        errors.append('run needs at least three unique seeds')
    elif record['mode'] == 'study' and record['engine'].startswith('hashed') and len(seeds) < 20:
        errors.append('hashed studies need at least twenty unique seeds')
    expected_status = {'smoke': 'VOID', 'study': 'UNJUDGED'}.get(record['mode'])
    if expected_status is None or record['scientific_status'] != expected_status:
        errors.append('run mode and scientific status are inconsistent')
    if payload.get('status') != 'complete' or not isinstance(payload.get('observations'), dict):
        errors.append('artifact is not a completed observation record')
    return errors


def audit_history(root: Path = ROOT) -> dict:
    """Resolve literal references; ambiguous/dynamic names remain review items.

    A candidate orphan has no literal incoming reference in tracked Python or
    Markdown. This does not establish that no dynamic consumer exists.
    """
    names = subprocess.check_output(['git', 'ls-files', '-z'], cwd=root).decode().split('\0')
    files = {name for name in names if name and (root / name).is_file()}
    by_basename = defaultdict(list)
    for name in files:
        by_basename[Path(name).name].append(name)
    edges, unresolved = [], []
    incoming = set()
    for source in sorted(files):
        if Path(source).suffix not in {'.md', '.py'}:
            continue
        content = (root / source).read_text(encoding='utf-8-sig', errors='replace')
        for ref in sorted(set(_FILE_REF.findall(content))):
            candidates = []
            if ref in files:
                candidates = [ref]
            else:
                target = (root / Path(source).parent / ref).resolve()
                relative = target.relative_to(root.resolve()).as_posix() if target.is_relative_to(root.resolve()) else ''
                if relative in files:
                    candidates = [relative]
                elif '/' not in ref:
                    candidates = by_basename.get(ref, [])
            if len(candidates) == 1:
                edges.append({'from': source, 'to': candidates[0], 'literal': ref})
                incoming.add(candidates[0])
            else:
                unresolved.append({'from': source, 'literal': ref,
                                   'reason': 'ambiguous' if candidates else 'unresolved',
                                   'candidates': candidates})
    results = sorted(name for name in files if name.startswith('research/')
                     and Path(name).suffix in {'.json', '.csv'}
                     and ('/results/' in name or 'result' in Path(name).name))
    preregs = sorted(name for name in files if Path(name).name.startswith('PREREG_')
                     and Path(name).suffix == '.md')
    reports_results = {edge['from'] for edge in edges if edge['to'] in results}
    return {'scope': 'literal-reference inventory, not semantic validity or exhaustive dynamic reachability',
            'tracked_files': len(files), 'resolved_edges': edges, 'unresolved_references': unresolved,
            'candidate_orphan_results': [name for name in results if name not in incoming],
            'preregistrations_without_resolved_result_links': [name for name in preregs if name not in reports_results]}


def specification_links(root: Path = ROOT) -> tuple[list[dict], list[str]]:
    """Check source docstring links to explicit Markdown specification anchors.

    This checks navigation, not whether an implementation satisfies its spec.
    Stable explicit anchors survive prose-heading edits. No package imports are
    needed, including for implementations requiring unavailable GPU hardware.
    """
    root = root.resolve()
    edges, errors = [], []
    package = root / 'neural_assemblies'
    sources = sorted([*package.rglob('*.py'), *package.rglob('*.rs'),
                      *(root / 'formal' / 'AssemblyIR').rglob('*.lean')])
    for source in sources:
        text = source.read_text(encoding='utf-8-sig')
        if 'Specification:' not in text:
            continue
        if source.suffix == '.lean':
            # Only module documentation blocks, not arbitrary proof/source text.
            docs = [("<module>", doc) for doc in
                    re.findall(r'^/-!\s*\n(.*?)^-/', text, re.MULTILINE | re.DOTALL)]
        elif source.suffix == '.rs':
            docs = [("<module>", "\n".join(
                line[3:] for line in text.splitlines() if line.startswith(("//!", "///"))))]
        else:
            tree = ast.parse(text, filename=str(source))
            docs = [(getattr(node, 'name', '<module>'), ast.get_docstring(node) or '')
                    for node in ast.walk(tree)
                    if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))]
        for owner, doc in docs:
            for ref in re.findall(r'^\s*Specification:\s*(\S+)', doc, re.MULTILINE):
                name, separator, anchor = ref.partition('#')
                target = (root / name).resolve()
                origin = f"{source.relative_to(root).as_posix()}:{owner}"
                edges.append({'from': origin, 'to': ref})
                if not separator or not anchor or not name.endswith('.md'):
                    errors.append(f'{origin}: specification needs a Markdown path and anchor: {ref}')
                elif not target.is_relative_to(root) or not target.is_file():
                    errors.append(f'{origin}: dangling specification file: {ref}')
                elif f'<a id="{anchor}"></a>' not in target.read_text(encoding='utf-8'):
                    errors.append(f'{origin}: dangling specification anchor: {ref}')
    return edges, errors


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('audit')
    commands.add_parser('specifications')
    validate = commands.add_parser('validate')
    validate.add_argument('path', type=Path)
    args = parser.parse_args(argv)
    if args.command == 'specifications':
        edges, errors = specification_links()
        print(json.dumps({'edges': edges, 'errors': errors}, indent=2))
        return 1 if errors else 0
    if args.command == 'audit':
        print(json.dumps(audit_history(), indent=2))
        return 0
    errors = validate_artifact(args.path)
    print(json.dumps({'path': str(args.path), 'valid_run_record': not errors, 'errors': errors}, indent=2))
    return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())

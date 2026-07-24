#!/usr/bin/env python3
"""Ground-up developmental training (babble → grammar) with exit gates.

Usage::

    python examples/train_developmental.py
    python examples/train_developmental.py --max-stage SENTENCES --preset core
    python examples/train_developmental.py --max-stage SENTENCES --output-dir research/results/dev_runs
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run developmental acquisition with CDS curriculum and exit gates.",
    )
    parser.add_argument(
        "--preset",
        choices=["core", "medium", "large"],
        default="core",
        help="Vocabulary preset (default: core)",
    )
    parser.add_argument(
        "--max-stage",
        default="SENTENCES",
        help="Last developmental stage to train (default: SENTENCES)",
    )
    parser.add_argument("-n", type=int, default=3000, help="Neurons per area")
    parser.add_argument("-k", type=int, default=30, help="Assembly size")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Write report.txt and metrics.json here",
    )
    parser.add_argument(
        "--no-gates",
        action="store_true",
        help="Disable per-stage exit gate enforcement",
    )
    parser.add_argument(
        "--no-babble",
        action="store_true",
        help="Skip pre-lexical babble stage",
    )
    parser.add_argument(
        "--no-adaptive",
        action="store_true",
        help="Disable adaptive remediation between stages",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Enable EMERGENT_FAST_TRAINING=1",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress training progress (TRAIN_PROGRESS=0)",
    )
    args = parser.parse_args()

    os.environ["EMERGENT_DEV_CURRICULUM"] = "1"
    if args.fast:
        os.environ["EMERGENT_FAST_TRAINING"] = "1"
    if args.quiet:
        os.environ["TRAIN_PROGRESS"] = "0"

    from neural_assemblies.assembly_calculus.emergent import (
        EmergentParser,
        build_vocabulary_preset,
    )
    from neural_assemblies.assembly_calculus.emergent.acquisition import (
        acquisition_report_to_dict,
        format_acquisition_report,
        run_developmental_acquisition,
    )
    from neural_assemblies.assembly_calculus.emergent.evaluation.generalization import (
        default_holdout_set,
    )

    vocab = build_vocabulary_preset(args.preset)
    emergent = EmergentParser(
        n=args.n,
        k=args.k,
        seed=args.seed,
        vocabulary=vocab,
        fast_training=args.fast,
    )

    report = run_developmental_acquisition(
        emergent,
        max_stage=args.max_stage,
        holdout_words=default_holdout_set(),
        babble=not args.no_babble,
        adaptive=not args.no_adaptive,
        gate_enforcement=not args.no_gates,
        seed=args.seed,
    )

    text = format_acquisition_report(report)
    print(text)

    payload = acquisition_report_to_dict(report)
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        stem = f"dev_{args.max_stage.lower()}_{args.seed}"
        (args.output_dir / f"{stem}.txt").write_text(text, encoding="utf-8")
        (args.output_dir / f"{stem}.json").write_text(
            json.dumps(payload, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"Wrote {args.output_dir / stem}.{{txt,json}}", file=sys.stderr)

    return 1 if report.blocked_at_stage else 0


if __name__ == "__main__":
    raise SystemExit(main())

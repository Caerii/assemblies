#!/usr/bin/env python3
"""Interactive chat with a curriculum-trained EmergentParser.

Usage::

    python examples/chat_emergent.py
    python examples/chat_emergent.py --preset medium --stage DIALOGUE
    python examples/chat_emergent.py --novel --corpus-size 500 --preset medium
    python examples/chat_emergent.py --preset discussion --stage CONVERSATION -n 5000 -k 50
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Chat with an EmergentParser trained on scaled vocabulary.",
    )
    parser.add_argument(
        "--preset",
        choices=["core", "medium", "large", "discussion"],
        default="medium",
        help="Vocabulary preset (default: medium)",
    )
    parser.add_argument(
        "--stage",
        choices=["DIALOGUE", "CONVERSATION"],
        default="DIALOGUE",
        help="Last curriculum stage to train (default: DIALOGUE)",
    )
    parser.add_argument("-n", type=int, default=3000, help="Neurons per area")
    parser.add_argument("-k", type=int, default=30, help="Assembly size")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-agent",
        action="store_true",
        help="Skip agent/tool training after conversation curriculum",
    )
    parser.add_argument(
        "--no-learn",
        action="store_true",
        help="Disable online learning during chat",
    )
    parser.add_argument(
        "--novel",
        action="store_true",
        help="Use large-corpus novel-chat training (grammar + bridges + conversation)",
    )
    parser.add_argument(
        "--corpus-size",
        type=int,
        default=400,
        help="Generated training sentences for --novel mode (default: 400)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress training progress on stderr (TRAIN_PROGRESS=0)",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Scale to n=1M neurons / k=100 (auto-selects torch_sparse when CUDA is available)",
    )
    args = parser.parse_args()

    if args.gpu:
        if args.n == 3000:
            args.n = 1_000_000
        if args.k == 30:
            args.k = 100

    if args.quiet:
        import os
        os.environ["TRAIN_PROGRESS"] = "0"

    from neural_assemblies.assembly_calculus.emergent import EmergentSession

    if args.novel:
        session = EmergentSession.bootstrap_novel_chat(
            preset=args.preset,
            n_corpus_sentences=args.corpus_size,
            max_stage=args.stage,
            n=args.n,
            k=args.k,
            seed=args.seed,
        )
    else:
        session = EmergentSession.bootstrap_conversation(
            preset=args.preset,
            max_stage=args.stage,
            n=args.n,
            k=args.k,
            seed=args.seed,
            include_agent=not args.no_agent,
        )
    session.online_learn = not args.no_learn

    vocab_size = len(session.parser.stim_map)
    engine = getattr(session.parser, "engine_name", "?")
    n = getattr(session.parser, "n", args.n)
    k = getattr(session.parser, "k", args.k)
    print(
        f"Ready — {vocab_size} words, engine={engine}, n={n:,}, k={k}. "
        f"Type 'quit' or Ctrl+C to exit.\n",
        file=sys.stderr,
    )

    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.lower() in ("quit", "exit", "q"):
            break
        reply = session.interact(line)
        print(f"bot> {reply}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

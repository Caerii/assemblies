#!/usr/bin/env python3
"""Record coin-flip demo statistics (package PFANetwork + RandomChoiceArea)."""

from __future__ import annotations

import json
from pathlib import Path

from neural_assemblies.programs.markov_coin import CoinFlipModel, train_markov_from_sequences
from neural_assemblies.core.brain import Brain

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "research" / "literature" / "parity" / "golden" / "coin2024_demo.json"

SEED = 42
N_FLIPS = 24
N_FLIPS_BIASED = 20


def main() -> None:
    traces = [("q0", "flip", "q0")] * 5 + [("q0", "flip", "q1")] * 5
    brain = Brain(p=0.05, save_winners=True, seed=SEED, engine="numpy_sparse")
    model = CoinFlipModel(
        brain,
        traces=traces,
        initial_state="q0",
        n=5000,
        k=50,
        beta=0.08,
        rounds=8,
        input_noise_std=0.02,
    )

    fair = model.empirical_flip_counts(N_FLIPS, bias=0.5, seed_base=100)
    biased = model.empirical_flip_counts(N_FLIPS_BIASED, bias=0.85, seed_base=200)

    transitions = train_markov_from_sequences(traces)
    p_q0 = next(t[3] for t in transitions if t[0] == "q0" and t[2] == "q0")
    p_q1 = next(t[3] for t in transitions if t[0] == "q0" and t[2] == "q1")
    golden = {
        "protocol": "coin2024_demo",
        "source": "neural_assemblies CoinFlipModel (dabagia.org describes softmax; impl uses k-split)",
        "demo_url": "http://dabagia.org/nemo/coinflipping/",
        "recorded": "2026-06-23",
        "parameters": {
            "seed": SEED,
            "n_flips_fair": N_FLIPS,
            "n_flips_biased": N_FLIPS_BIASED,
            "fair_seed_base": 100,
            "biased_seed_base": 200,
            "input_noise_std": 0.02,
            "markov_traces": "5×(q0,flip,q0)+5×(q0,flip,q1)",
        },
        "metrics": {
            "fair_p1": round(fair[1] / N_FLIPS, 4),
            "fair_n0": fair[0],
            "fair_n1": fair[1],
            "biased_majority_zero": biased[0] > biased[1],
            "biased_n0": biased[0],
            "biased_n1": biased[1],
            "markov_learned_p_q0": round(p_q0, 4),
            "markov_learned_p_q1": round(p_q1, 4),
        },
        "tolerance": {"fair_p1": 0.08},
        "expected": {
            "fair_both_outcomes": True,
            "biased_majority_zero": True,
            "markov_learned_p_q0": 0.5,
            "markov_learned_p_q1": 0.5,
        },
        "notes": "Live demo has no exported numeric API; golden from package simulation at paper-described Markov protocol.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(golden, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(golden["metrics"], indent=2))


if __name__ == "__main__":
    main()

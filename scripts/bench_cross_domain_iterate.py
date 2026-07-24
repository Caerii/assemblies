#!/usr/bin/env python3
"""Compare cross-domain fusion variants (discovery iteration bench)."""

from __future__ import annotations

import argparse

from neural_assemblies.programs.colt_mnist_tier_util import clear_ventral_bundle_cache
from neural_assemblies.programs.cross_domain_assemblies import run_vision_language_contrastive_hub
from neural_assemblies.programs.cross_domain_profile import profile_cross_domain_hub
from neural_assemblies.programs.cross_domain_assemblies import _train_cross_domain_hub


VARIANTS = {
    "baseline_slots": dict(
        semantic_wiring="slots", use_connectome_lri=False,
        confused_class_reinforcement=False, prototypes_per_digit=3,
    ),
    "full_v2": dict(
        semantic_wiring="views", use_connectome_lri=True,
        confused_class_reinforcement=True, prototypes_per_digit=5,
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-examples", type=int, default=50)
    args = parser.parse_args()

    clear_ventral_bundle_cache()
    print(f"Cross-domain iteration bench (n={args.n_examples}, seed={args.seed})\n")

    for name, kw in VARIANTS.items():
        r = run_vision_language_contrastive_hub(
            seed=args.seed, n_examples=args.n_examples, **kw,
        )
        print(
            f"{name:20s} fused={r.visual_accuracy:.1%}  "
            f"lang={r.language_accuracy:.1%}  "
            f"i2t={r.image_to_text_recall:.1%}  "
            f"lri={r.extra.get('lri_trigger_rate', 0):.1%}"
        )

    print("\n--- Full profile (v2) ---")
    hub = _train_cross_domain_hub(
        seed=args.seed, n_examples=args.n_examples, **VARIANTS["full_v2"],
    )
    p = profile_cross_domain_hub(hub=hub)
    print(p.narrative)


if __name__ == "__main__":
    main()

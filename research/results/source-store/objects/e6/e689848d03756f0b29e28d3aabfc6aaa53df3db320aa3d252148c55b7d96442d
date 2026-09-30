"""What is actually IN the role lexicons, and does it overlap the vocabulary?

`the_role_margin_branch_that_never_fires.md` recorded four of seven role
lexicons empty (THEME/GOAL/SOURCE/LOCATION = 0) and treated it as harmless
because those roles never became candidates. `erp_or_else_census.py` then found
`_role_binding_margin` UNDEFINED on 117 of 147 calls, with reasons naming the
ordinary sentence words:

    'cat' has no stored binding in ROLE_PATIENT
    'dog' has no stored binding in ROLE_AGENT
    'she' has no stored binding in ROLE_AGENT

That is a different and worse claim than "four roles are empty". It says the
POPULATED lexicons do not contain the words the parser is actually parsing --
in which case the lexical role route is dead in exactly the sentences every
published role result is measured on, and `.or_else(0.0)` hands every decision
to the structural prior while looking like a neural readout.

This prints, per role area: size, a sample, and the intersection with the
vocabulary the ERP frames and the training curriculum actually use. Nothing is
inferred from a call trace -- the lexicons are read directly, so a disagreement
with the census is informative rather than confusing.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
os.environ.setdefault("TRAIN_PROGRESS", "0")
os.environ.setdefault("EMERGENT_ERP_FAST", "1")

from neural_assemblies.assembly_calculus.emergent.evaluation.erp.frames import (  # noqa: E402
    DEFAULT_CALIBRATION_FRAMES,
)
from neural_assemblies.assembly_calculus.emergent.evaluation.sweep import (  # noqa: E402
    get_parser_cache,
)

SEEDS = [11, 42]


def main():
    frame_words = sorted({w for _l, _d, ws in DEFAULT_CALIBRATION_FRAMES
                          for w in ws})

    for seed in SEEDS:
        parser = get_parser_cache().fork("SENTENCES", seed=seed)
        lex = getattr(parser, "role_lexicons", {}) or {}
        core = getattr(parser, "core_lexicons", {}) or {}

        print(f"=== seed {seed} ===")
        print(f"stim_map: {len(parser.stim_map)} words")
        print(f"core_lexicons: "
              f"{ {a: len(v) for a, v in sorted(core.items())} }")
        print()
        print(f"{'role area':<16} {'size':>5}  {'frame words present':<40} sample")
        print("-" * 100)
        for area in sorted(lex):
            words = set(lex[area])
            present = sorted(w for w in frame_words if w in words)
            sample = ", ".join(sorted(words)[:6])
            print(f"{area:<16} {len(words):>5}  "
                  f"{(', '.join(present) or '(none)'):<40} {sample}")

        missing_areas = [a for a in lex if not lex[a]]
        if missing_areas:
            print(f"\nEMPTY: {missing_areas}")

        union = set().union(*(set(v) for v in lex.values())) if lex else set()
        print(f"\nframe words in ANY role lexicon: "
              f"{sorted(w for w in frame_words if w in union) or '(none)'}")
        print(f"frame words in stim_map but in NO role lexicon: "
              f"{sorted(w for w in frame_words if w in parser.stim_map and w not in union)}")
        print()


if __name__ == "__main__":
    main()

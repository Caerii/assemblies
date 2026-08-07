"""Does the drive-share fix repair the NEMO-2025 curriculum role-probe golden?

THE BROKEN GOLDEN. `nemo2025_curriculum.json` asserts
``role_probe_accuracy_min = 1.0``. Cold, current code delivers **0.6667** --
two of three role probes. Verified pre-existing: the same 0.6667 reproduces at
commit 6ecfa15 (before anything in this session's arc) with the backbone cache
disabled, so it is not a regression I introduced.

WHY IT WENT UNNOTICED. The golden was recorded 2026-06-23, BEFORE `norm_init`
became the default substrate, and the failure was observed on 2026-07-31. In
between, warm backbones let the test pass: warm runs deserialise, they do not
train, so a golden can rot for weeks while CI stays green
([[backbone-fingerprint-gap]]).

THE HYPOTHESIS THIS FILE TESTS. A role probe asks whether a word is retrievable
from its bound role, and this session measured exactly that quantity rising
under two changes:

    beta 0.10 -> 0.05          role ret@6 0.806/0.733 -> 0.872/0.844
    phon_weight 1 -> 6         role ret@6 0.872/0.844 -> 0.974/0.964

If the golden broke because role binding degraded, those knobs should move it.
If it broke for an unrelated reason -- a changed probe, a vocabulary gap, a
missing word -- they will not, and the accuracy will sit at 0.6667 throughout.

BOTH OUTCOMES ARE USEFUL, and the second is the more likely one worth guarding
against: 2 of 3 is a SMALL DENOMINATOR. A single probe item flipping moves the
metric by 0.333, so any change here must be read as "which item changed", not
as a continuous improvement. The per-probe detail is printed for that reason.

WHAT THIS FILE DOES NOT DO: re-record the golden at 0.6667. Lowering a
threshold until it passes converts a broken claim into a passing test, which is
the failure mode the whole parity programme exists to prevent.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
os.environ.setdefault("TRAIN_PROGRESS", "0")
os.environ.pop("EMERGENT_DEV_CURRICULUM", None)

import json                                                            # noqa: E402

GOLDEN = (Path(__file__).resolve().parents[2] / "research" / "literature"
          / "parity" / "golden" / "nemo2025_curriculum.json")

PROBES = [
    {"words": ["the", "dog", "runs"], "expected_roles": {"dog": "AGENT"}},
    {"words": ["the", "cat", "chases", "the", "bird"],
     "expected_roles": {"cat": "AGENT", "bird": "PATIENT"}},
]

ARMS = [
    ("golden as recorded", {}),
    ("beta=0.05", {"beta": 0.05}),
    ("phon_weight=6", {"phon_weight": 6.0}),
    ("beta=0.05 + phon_weight=6", {"beta": 0.05, "phon_weight": 6.0}),
]


def run(params, overrides):
    from neural_assemblies.assembly_calculus.emergent import EmergentParser
    from neural_assemblies.assembly_calculus.emergent.curriculum import (
        CurriculumTrainer,
    )
    from neural_assemblies.assembly_calculus.emergent.evaluation import (
        EvaluationSuite,
    )

    kw = dict(n=params["n"], k=params["k"], p=params["p"],
              beta=params["beta"], seed=42, rounds=params["rounds"])
    kw.update(overrides)
    parser = EmergentParser(**kw)
    trainer = CurriculumTrainer(parser)
    for stage in params["stages"]:
        trainer.train_stage(stage)

    suite = EvaluationSuite(parser)
    wo = suite.evaluate_word_order(target="SVO")
    roles = suite.evaluate_roles(PROBES)
    sent = next((r for r in trainer.stage_results
                 if r.stage_name == "SENTENCES"), None)
    return {
        "role_acc": roles.get("accuracy"),
        "roles": roles,
        "word_order_correct": wo.get("correct"),
        "sent_acc": getattr(sent, "classification_accuracy", float("nan")),
    }


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    g = json.loads(GOLDEN.read_text(encoding="utf-8"))
    params = g["parameters"]
    want = g["thresholds"]["role_probe_accuracy_min"]
    print("NEMO-2025 curriculum golden: role_probe_accuracy_min = "
          f"{want}, recorded {g.get('recorded', '?')}")
    print(f"parameters: {params}")
    print()
    hdr = (f"{'arm':<28}{'role acc':>10}{'vs golden':>11}"
           f"{'word order':>12}{'sent acc':>10}")
    print(hdr)
    print("-" * len(hdr))
    out = {}
    for label, ov in ARMS:
        r = run(params, ov)
        out[label] = r
        acc = r["role_acc"]
        mark = "PASS" if acc is not None and acc >= want else "fail"
        print(f"{label:<28}{acc:>10.4f}{mark:>11}"
              f"{str(r['word_order_correct']):>12}{r['sent_acc']:>10.3f}")

    print()
    print("PER-PROBE DETAIL -- the denominator is 3, so one item is 0.333 and")
    print("a moved metric must be read as WHICH item changed:")
    for label, r in out.items():
        det = r["roles"].get("details") or r["roles"].get("probes") or r["roles"]
        print(f"  {label}: {str(det)[:220]}")

    print()
    best = max(out.items(), key=lambda kv: kv[1]["role_acc"] or 0.0)
    if best[1]["role_acc"] and best[1]["role_acc"] >= want:
        print(f"REPAIRED by '{best[0]}'. The golden's claim reproduces once the")
        print("drive share is fixed, so the parity break was a SUBSTRATE")
        print("regression and not a stale threshold. Re-record with the new")
        print("parameters recorded explicitly.")
    else:
        print("NOT REPAIRED by either knob. Role binding is not what broke this")
        print("golden -- look at the probe items themselves (vocabulary gaps,")
        print("inflection mismatches) before touching the threshold, and do NOT")
        print("re-record at the lower value to make it pass.")


if __name__ == "__main__":
    main()

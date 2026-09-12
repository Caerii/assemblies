"""#121: does the BINDING channel see the pathway learning the drive channel
provably cannot?

CONTEXT. `erp_pathway_vs_area_control.py` decomposed the shipped P600 into
(a) pathway learning and (b) area identity, and the drive-channel readout
(`anchored_p600_live`, 1 - input_drive energy) measured:

    (a) PATHWAY only  AUC 0.5150 +/- 0.0725   <- chance
    (a+b) SHIPPED     AUC 0.8870 +/- 0.0275
    (b) AREA only     AUC 0.9800 +/- 0.0400

Chance on (a) is a property of the CHANNEL: drive sums over all candidate
neurons and measures HOW MUCH, never WHERE. Meanwhile #52 closed with the
binding channel discriminating stored role bindings at 0.97-1.000 top-1
(role_recipe_2x2, n=3e3 and n=1e5) through `ops.read_binding` -- the same
dynamics training used, under read_only(). `role_binding_deficit`
(erp/adapters.py, registered with this experiment) packages that channel as
a P600-style deficit: 1 - max overlap of the recalled assembly against the
role area's stored bindings.

DESIGN: identical arms to the control -- imported from it, not re-implemented
-- measured on the SAME parsers with BOTH readouts, so the dissociation (if
any) is within-substrate and paired:

    patient_trained   noun bound into ROLE_PATIENT     (zero corpus freq)
    agent_only        noun bound into ROLE_AGENT only  (zero corpus freq)
    verb_object       trained verb                     (zero corpus freq)

REGISTERED PREDICTIONS (before any cell; each can fail):
  P-PATHWAY  binding-channel AUC(agent_only above patient_trained) >= 0.75.
             THE claim: the WHERE channel sees word-level binding history
             where drive reads 0.515. Below 0.75 but above 0.65 = real but
             weak (report as such); 0.5-ish = the channel also cannot see
             it in the parser's trained state; BELOW 0.5 = the #120
             INVERSION signature -- pre-stated diagnostic: the per-item
             `landed_on` census, because capture by a dominant shared
             attractor is the known mechanism that inverts this readout.
  P-SHIPPED  binding-channel AUC(verb_object above patient_trained) >= 0.75
             (the shipped contrast must not regress on the new channel).
  P-DRIVE    paired drive-channel pathway AUC replicates chance: mean in
             [0.40, 0.65] across seeds (the control read 0.515 on 5 seeds).
  VOID RULE  undefined probes are REPORTED per arm; an arm with < N/2
             defined items voids every contrast it enters (no silent
             shrinkage of an arm into a different population).
  DECISION RULE (pre-stated): `role_binding_deficit` becomes the ERP
             role-integration readout -- and the composite flip gets its own
             A/B unit -- ONLY if P-PATHWAY and P-SHIPPED both hold. If
             P-PATHWAY fails at ~0.5 with retrieval still at #52's ceiling
             on these very parsers, the finding is that the exam and the ERP
             probe read DIFFERENT quantities and the divergence point must
             be located before any adoption.

SEEDS: the control's 5 (paired replication) + 5 extension = 10. Within a
seed all arms and both channels run on one parser (within-substrate).
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
os.environ.setdefault("TRAIN_PROGRESS", "0")
os.environ.setdefault("EMERGENT_ERP_FAST", "1")

from research.json_documents import write_new_document                 # noqa: E402

from erp_pathway_vs_area_control import (                               # noqa: E402
    DET, N_PER_ARM, SUBJECT, VERB, _corpus_counts, _pick,
)
from neural_assemblies.assembly_calculus.emergent.core.areas import (   # noqa: E402
    ROLE_PATIENT,
)
from neural_assemblies.assembly_calculus.emergent.evaluation.erp.adapters import (  # noqa: E402
    role_binding_deficit,
)
from neural_assemblies.assembly_calculus.emergent.evaluation.erp.frames import (  # noqa: E402
    collect_frame_samples,
)
from neural_assemblies.assembly_calculus.emergent.evaluation.sweep import (  # noqa: E402
    get_parser_cache,
)
from neural_assemblies.diagnostics import separation                    # noqa: E402

SEEDS = [11, 12, 42, 7, 19, 43, 44, 45, 46, 47]
DEPTH = "SENTENCES"
ARM_CORE = {
    "patient_trained": "NOUN_CORE",
    "agent_only": "NOUN_CORE",
    "verb_object": "VERB_CORE",
}
OUT_PATH = os.path.join(os.path.dirname(__file__),
                        "erp_pathway_vs_area_binding_results.json")

CONTRASTS = (
    ("pathway", "agent_only", "patient_trained"),
    ("shipped", "verb_object", "patient_trained"),
    ("area", "verb_object", "agent_only"),
)


def main():
    counts = _corpus_counts()
    out = {"seeds": {}, "analysis": {}}

    for seed in SEEDS:
        parser = get_parser_cache().fork(DEPTH, seed=seed)
        pools = _pick(parser, counts)
        short = {k: len(v) for k, v in pools.items() if len(v) < N_PER_ARM}
        if short:
            print(f"seed {seed}: POOLS TOO SMALL {short}; skipping")
            continue

        # Channel 1: the binding deficit (context-free cue -> expected slot).
        binding = {}
        undef = {}
        for arm, words in pools.items():
            vals, rows = [], []
            for w in words:
                m = role_binding_deficit(parser, ARM_CORE[arm],
                                         ROLE_PATIENT, w)
                if m.defined:
                    vals.append(float(m))
                    rows.append({"word": w, "deficit": float(m), **m.detail})
                else:
                    rows.append({"word": w, "undefined": m.why})
            binding[arm] = vals
            undef[arm] = sum(1 for r in rows if "undefined" in r)
            out["seeds"].setdefault(str(seed), {}).setdefault(
                "binding_rows", {})[arm] = rows

        # Channel 2: the paired drive p600 at the same probe position.
        frames = [
            (arm, f"{arm}:{w}", [DET, SUBJECT, VERB, w])
            for arm, words in pools.items() for w in words
        ]
        samples = collect_frame_samples(parser, frames)
        drive = {}
        for s in samples:
            drive.setdefault(s.label, []).append(s.p600)

        rec = out["seeds"][str(seed)]
        rec["pools"] = pools
        rec["undefined_per_arm"] = undef
        rec["binding"] = binding
        rec["drive"] = drive

        print(f"=== seed {seed} ===")
        for name, hi, lo in CONTRASTS:
            line = f"  {name:<8}"
            for label, ch in (("BINDING", binding), ("drive", drive)):
                hi_v, lo_v = ch.get(hi, []), ch.get(lo, [])
                voided = (len(hi_v) < N_PER_ARM / 2
                          or len(lo_v) < N_PER_ARM / 2)
                if voided:
                    line += f"  {label} VOID(n={len(hi_v)},{len(lo_v)})"
                    continue
                s = separation(hi_v, lo_v, label=f"{name}:{label}")
                line += f"  {label} AUC={s.auc:.4f}"
                rec.setdefault("auc", {})[f"{name}_{label}"] = s.auc
            print(line + f"  undef={undef}")

    # Across seeds.
    print("=" * 66)
    for name, _hi, _lo in CONTRASTS:
        for label in ("BINDING", "drive"):
            key = f"{name}_{label}"
            aucs = [rec["auc"][key] for rec in out["seeds"].values()
                    if key in rec.get("auc", {})]
            if aucs:
                m = sum(aucs) / len(aucs)
                sd = (sum((a - m) ** 2 for a in aucs)
                      / max(len(aucs) - 1, 1)) ** 0.5
                out["analysis"][key] = {
                    "mean": m, "sd": sd, "n": len(aucs),
                    "per_seed": aucs,
                }
                print(f"  {key:<20} AUC={m:.4f} sd={sd:.4f} n={len(aucs)}")

    write_new_document(Path(OUT_PATH), out)
    print(f"-> {OUT_PATH}")


if __name__ == "__main__":
    main()

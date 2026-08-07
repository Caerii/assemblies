"""Does `bind()` write a pathway that can be read back at all?

WHY THIS IS THE QUESTION. `erp_pathway_vs_area_control.py` found that the P600
cannot distinguish a noun bound into ROLE_PATIENT (36 of them) from a noun never
bound there (29), while swapping the SOURCE AREA reproduces the whole effect.
Two very different explanations, and they have very different blast radii:

  (i)  THE READOUT WASHES IT OUT. The synaptic difference exists but is
       destroyed by area-level normalization, or by overlap between noun
       assemblies. -> an ERP readout problem, and the metric may be rescuable.
  (ii) THE BINDING BARELY HAPPENED. `bind(NOUN_CORE, ROLE_PATIENT, word)` does
       not measurably strengthen that word's pathway. -> the mechanism is
       near-dormant, and this touches EVERY role result in the repo -- role
       retrieval, the lesion double dissociation, word-order induction -- all of
       which assume role binding writes something retrievable.

THE FIRST VERSION OF THIS SCRIPT WAS WRONG, and the way it was wrong is a
finding in its own right. It compared, per word,

    contrast(word) = drive(-> ROLE_PATIENT) - drive(-> ROLE_AGENT)

on the reasoning that `input_drive` drives all targets in ONE projection and its
docstring calls the scores commensurable. They are not, here. That docstring
promises normalization "per candidate" -- "two role areas measured here differed
by 391 vs 449, and comparing raw sums across them reverses the ranking purely on
size" -- and implements it as

    v / max(int(brain.areas[a].w), 1)

`brain.areas[a].w` is the WINNERS-LENGTH ALIAS, not the recruited count. Measured
on this parser:

    ROLE_PATIENT   Area.w = 0      engine.w = 675
    ROLE_AGENT     Area.w = 0      engine.w = 874

Both role areas have empty winners at probe time, so the divisor is 1 for both
and the "normalization" does nothing -- while the areas differ 675 vs 874, which
is precisely the 1.3x size difference the docstring says reverses rankings. The
guard was disabled exactly where it was needed. See [[same-name-two-meanings]].

SO THE COMPARISON IS SINGLE-TARGET. Every word is read into ROLE_PATIENT only.
Whatever the divisor is, it is the SAME for every word, so it cancels in a rank
statistic. Drive to ROLE_AGENT is still reported per group, because it is
informative, but no cross-area difference is scored.

Three groups the parser supplies itself:

    patient_bound   in role_lexicons[ROLE_PATIENT]   drive -> PATIENT should be HIGH
    agent_bound     in [ROLE_AGENT] only             lower
    unbound         in neither -- the clean control  lowest, or indistinguishable

`source_assemblies` IS USED DELIBERATELY, and it closes a trap. The obvious
alternative is to project phon -> NOUN_CORE and read the area's winners. But
`brain.read_only()` FREEZES winners, so under it every word would read the same
assembly and every drive would be identical -- a manufactured null that looks
exactly like outcome (ii). Handing `input_drive` the STORED assembly makes the
source word-specific by construction, with no dependence on whether a projection
took effect. See [[fake-perfect-probe-signatures]].

PRE-REGISTERED, and both outcomes are informative:

  * BINDING IS READABLE: AUC(patient_bound above agent_bound, on drive into
    ROLE_PATIENT) clearly above 0.5, and above `unbound` too.
  * BINDING IS DORMANT: that AUC ~ 0.5 and the three groups are
    indistinguishable.

SANITY CHECKS THAT MUST PASS BEFORE THE RESULT MEANS ANYTHING, because a
degenerate probe produces outcome (ii) for free:
  * the drives must VARY across words -- a constant means the activation never
    reached the target;
  * both target areas must be materialized -- a dead fiber reads 0.0 and would
    make every contrast identical;
  * raw per-area drives are printed, not only the contrast, so a constant is
    visible rather than inferred.

Also measures pairwise overlap between noun core assemblies, because the
crowding account (#52, role binding at overlap 0.15-0.22) predicts that if the
assemblies themselves are not distinct, no per-word pathway could be readable
whatever `bind()` did.
"""
import os
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
os.environ.setdefault("TRAIN_PROGRESS", "0")
os.environ.setdefault("EMERGENT_ERP_FAST", "1")

import numpy as np                                                      # noqa: E402

from neural_assemblies.assembly_calculus.binding import input_drive     # noqa: E402
from neural_assemblies.assembly_calculus.emergent.core.areas import (   # noqa: E402
    NOUN_CORE, ROLE_AGENT, ROLE_PATIENT,
)
from neural_assemblies.assembly_calculus.emergent.evaluation.sweep import (  # noqa: E402
    get_parser_cache,
)
from neural_assemblies.diagnostics import separation                    # noqa: E402

SEEDS = [11, 12, 42]


def _groups(parser):
    core = parser.core_lexicons.get(NOUN_CORE, {})
    patient = set(parser.role_lexicons.get(ROLE_PATIENT, {}))
    agent = set(parser.role_lexicons.get(ROLE_AGENT, {}))
    return {
        "patient_bound": sorted(w for w in core if w in patient),
        "agent_bound": sorted(w for w in core if w in agent and w not in patient),
        "unbound": sorted(w for w in core if w not in patient and w not in agent),
    }


def _neurons(asm):
    """Assembly winners as NEURON IDS -- the stable space, not compact indices."""
    ids = getattr(asm, "neuron_ids", None)
    if ids is None:
        ids = getattr(asm, "winners", None)
    return set(np.asarray(ids).ravel().tolist()) if ids is not None else set()


def main():
    for seed in SEEDS:
        parser = get_parser_cache().fork("SENTENCES", seed=seed)
        brain = parser.brain
        core = parser.core_lexicons.get(NOUN_CORE, {})
        groups = _groups(parser)

        print(f"=== seed {seed} ===")
        eng = getattr(brain, "_engine", None)
        for a in (NOUN_CORE, ROLE_PATIENT, ROLE_AGENT):
            present = a in brain.areas
            alias = int(brain.areas[a].w) if present else -1
            try:
                real = eng._areas[a].w
            except Exception:                              # noqa: BLE001
                real = "?"
            flag = "  <-- ALIAS != RECRUITED, input_drive divides by the alias"                 if present and str(real) != str(alias) else ""
            print(f"  {a:<14} Area.w={alias:<7} engine.w={real}{flag}")

        rows = {}
        for name, words in groups.items():
            recs = []
            for word in words:
                asm = core.get(word)
                if asm is None:
                    continue
                d = input_drive(
                    brain,
                    sources=[NOUN_CORE],
                    target_areas=[ROLE_PATIENT, ROLE_AGENT],
                    source_assemblies={NOUN_CORE: asm},
                )
                if ROLE_PATIENT not in d or ROLE_AGENT not in d:
                    continue
                recs.append((word, d[ROLE_PATIENT], d[ROLE_AGENT]))
            rows[name] = recs

        allrec = [r for recs in rows.values() for r in recs]
        if not allrec:
            print("  NO READINGS -- nothing to report")
            continue
        pats = [p for _w, p, _a in allrec]
        agts = [a for _w, _p, a in allrec]
        print(f"  readings: {len(allrec)} words")
        print(f"  drive->ROLE_PATIENT  min={min(pats):.6f} max={max(pats):.6f}")
        print(f"  drive->ROLE_AGENT    min={min(agts):.6f} max={max(agts):.6f}")
        if max(pats) - min(pats) < 1e-12 and max(agts) - min(agts) < 1e-12:
            print("  ** DEGENERATE: drive is CONSTANT across words. The source "
                  "assembly is not reaching the target; nothing below is a "
                  "measurement. **")
            print()
            continue

        for name in ("patient_bound", "agent_bound", "unbound"):
            recs = rows.get(name, [])
            if not recs:
                print(f"  {name:<15} n=0")
                continue
            cs = [p - a for _w, p, a in recs]
            mp = sum(p for _w, p, _a in recs) / len(recs)
            ma = sum(a for _w, _p, a in recs) / len(recs)
            print(f"  {name:<15} n={len(recs):<3} "
                  f"mean->PATIENT={mp:.6f}  mean->AGENT={ma:.6f}  "
                  f"mean contrast={sum(cs)/len(cs):+.6f}")

        # SINGLE TARGET ONLY. Cross-area differences are not scored -- see the
        # module docstring on the disabled `w` normalization.
        def into_patient(name):
            return [p for _w, p, _a in rows.get(name, [])]

        pb, ab, ub = (into_patient(n) for n in
                      ("patient_bound", "agent_bound", "unbound"))
        if pb and ab:
            s = separation(pb, ab, label="patient_vs_agent_bound")
            print(f"  AUC(patient_bound drive->PATIENT above agent_bound) = "
                  f"{s.auc:.4f}   n_pairs={len(pb)*len(ab)}")
        if pb and ub:
            s = separation(pb, ub, label="patient_vs_unbound")
            print(f"  AUC(patient_bound above UNBOUND)                    = "
                  f"{s.auc:.4f}   n_pairs={len(pb)*len(ub)}")

        # Crowding: are the noun assemblies distinct enough to carry a
        # per-word pathway in the first place?
        sample = sorted(core)[:24]
        ov = []
        for x, y in combinations(sample, 2):
            ax, ay = _neurons(core[x]), _neurons(core[y])
            if ax and ay:
                ov.append(len(ax & ay) / max(len(ax), 1))
        if ov:
            print(f"  NOUN_CORE pairwise overlap over {len(sample)} words: "
                  f"mean={sum(ov)/len(ov):.4f} min={min(ov):.4f} "
                  f"max={max(ov):.4f}")
        print()

    print("READING IT. AUC ~ 0.5 with the sanity checks passing means the")
    print("binding wrote nothing a downstream area can read -- which is a")
    print("claim about the ROLE MECHANISM, not about the ERP metric, and it")
    print("would put every role result in the repo on the same footing.")


if __name__ == "__main__":
    main()

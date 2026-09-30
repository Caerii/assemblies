"""The pre-registered probe for #33: a role readout that consults THIS parse.

THE DESIGN, from the 2021 parser paper's division of labor (gate -> record ->
recall), assembled from parts the repo already has:

  GATE    Voice is detected by the LEARNED gating (`_determine_role_order`,
          MARKER conf 0.96) and selects a SLOT SEQUENCE: the language's filler
          order for an active clause, the REVERSED filler order for a passive.
          A passive of an SVO language is parsed by the OVS sequence -- voice is
          control selecting a different program, which is exactly the papers'
          claim about Broca-like control.

  RECORD  Each content word's core assembly is projected into the open slot
          (the `bind` traversal protocol, T=2). Frozen here, so "record" is the
          role area's WINNERS -- the parse's transient state -- not plasticity.

  RECALL  The RECONSTRUCTION readout. For each role area's current winners,
          ask which candidate word's projection image reproduces them:

              occupant(R) = argmax_w overlap(image(w->R), winners(R))

          No `role_lexicons` anywhere: identity comes from replaying core
          assemblies, so a word NEVER TRAINED in a role can still be read out
          of it -- the exact case (`child` as patient) where the stored-lexicon
          routes fail.

WHAT THE SUBSTRATE CONTRIBUTES, stated so the trivial part is not oversold.
With a frozen parse the occupant's replay reproduces the winners BY
DETERMINISM; occupancy alone is not the finding. The substrate's contribution
is the GAP between the occupant's overlap (~1.0) and the best non-occupant's
(the role area's attractor baseline -- 0.70-0.85 for trained fillers, per
`_role_binding_margin`'s own docstring). If word images in a role area
COLLAPSE (the merge/crowding regime this repo has measured repeatedly), the gap
closes and this readout fails. So the gap is the substrate-dependence metric,
and it must VARY with the seed where the old route's accuracy was bit-identical.

PRE-REGISTERED CRITERIA (written before running):
  A. Active frames: reconstruction recovers agent/patient correctly.
  B. Passive frames: likewise -- INCLUDING for nouns whose training bias is
     the opposite role, where `_role_binding_margin` measurably fails.
  C. Sentence-conditioning: the SAME word reads out differently in the active
     and passive frames of the same event, and the role area's winners differ
     between the two frames.
  D. The occupant-vs-nonoccupant gap is positive with margin, and varies
     across seeds (the substrate is in the loop; zero-width = red flag).
  E. Side-by-side: `_assign_roles_neural` (the margin route) on the same
     items, so the comparison is on identical probes.
"""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
os.environ.setdefault("TRAIN_PROGRESS", "0")

import numpy as np  # noqa: E402

from neural_assemblies.assembly_calculus.assembly import (  # noqa: E402
    overlap as assembly_overlap,
)
from neural_assemblies.assembly_calculus.ops import (  # noqa: E402
    _snap, activate_assembly,
)
from neural_assemblies.assembly_calculus.emergent import (  # noqa: E402
    EmergentParser,
)
from neural_assemblies.assembly_calculus.emergent.core.areas import (  # noqa: E402
    ROLE_ACTION, ROLE_AGENT, ROLE_PATIENT,
)
from neural_assemblies.assembly_calculus.emergent.curriculum import (  # noqa: E402
    CurriculumTrainer,
)
from neural_assemblies.assembly_calculus.emergent.parser_mixins._shared import (  # noqa: E402
    _ROLE_BINDING_ROUNDS,
)
from neural_assemblies.assembly_calculus.emergent.vocabulary_builder import (  # noqa: E402
    build_vocabulary_preset,
)

STAGES = ("FIRST_WORDS", "VOCABULARY_SPURT", "TWO_WORD", "SENTENCES")
N, K = 3000, 30
SEEDS = (11, 23, 37)
FILLER_SLOTS = (ROLE_AGENT, ROLE_PATIENT)

ACTIVE_SEQ = (ROLE_AGENT, ROLE_ACTION, ROLE_PATIENT)
PASSIVE_SEQ = (ROLE_PATIENT, ROLE_ACTION, ROLE_AGENT)


def _bind_traversal(brain, core: str, role: str) -> None:
    """The parse/replay projection: ONE protocol for both, which is what makes
    the reconstruction well-defined (the replay reproduces the parse exactly
    when the source and weights are the same)."""
    brain.project({}, {core: [role]})
    for _ in range(_ROLE_BINDING_ROUNDS - 1):
        brain.project({}, {core: [role], role: [role]})


def _project_word(parser, word: str, role: str) -> bool:
    """Activate `word` in its core area, bind-traverse into `role`.

    The stabilized lexicon assembly when one exists; otherwise the parser's own
    fallback (roles.py): project the phon stimulus. Without the fallback the
    first run showed whole frames collapsing to `{}` -- an inflected form with
    no stored assembly ('enters', 'held') stalled the slot sequence and every
    later word misaligned, exactly the cascade nemo_parse.py warns about.
    Under read_only the fallback projection is frozen, hence deterministic,
    hence replayable -- which is what the reconstruction needs.
    """
    core = parser._word_core_area(word)
    if core is None or core not in parser.brain.areas:
        return False
    stored = parser.core_lexicons.get(core, {}).get(word)
    if stored is not None:
        activate_assembly(parser.brain, stored)
    else:
        phon = parser.stim_map.get(word)
        if phon is None:
            return False
        from neural_assemblies.assembly_calculus.ops import project as _proj
        _proj(parser.brain, phon, core, rounds=parser.rounds)
    parser.brain.areas[core].fix_assembly()
    try:
        _bind_traversal(parser.brain, core, role)
    finally:
        parser.brain.areas[core].unfix_assembly()
    return True


def gated_parse_and_reconstruct(parser, words: List[str]
                                ) -> Tuple[Dict[str, Optional[str]], dict]:
    """Delegates to the CANONICAL implementation, promoted into the package
    after this experiment measured it: `RoleBindingMixin.
    parse_roles_by_reconstruction`. The body below is retained ONLY as the
    reference the promotion was checked against; the delegation keeps this
    harness measuring the shipped code rather than a drifting copy.
    """
    return parser.parse_roles_by_reconstruction(list(words))


def _reference_gated_parse_and_reconstruct(
        parser, words: List[str]) -> Tuple[Dict[str, Optional[str]], dict]:
    """The original experiment-local implementation (promotion reference)."""
    brain = parser.brain
    cats = {w: parser.classify_word_cached(w)[0] for w in words}

    # GATE: learned voice detection selects the slot sequence. This is the
    # measured mechanism from #128 (MARKER conf 0.96, contrastive), consulted
    # through the same lookup its writer uses.
    _order, is_passive = parser._determine_role_order(words, cats)
    # FILLER slots only. Making nouns queue behind an ACTION pivot deadlocks
    # the moment the verb token is unclassifiable ('enters': never generated,
    # so never registered, so no grounding, so no category) -- the noun after
    # it then waits forever. Nouns consume the voice-ordered filler sequence;
    # the FIRST verb-classified token takes ACTION independently, which is how
    # `_assign_roles_neural` is shaped too.
    sequence = ((ROLE_PATIENT, ROLE_AGENT) if is_passive
                else (ROLE_AGENT, ROLE_PATIENT))

    diag: dict = {"is_passive": bool(is_passive), "gaps": [], "winners": {}}
    out: Dict[str, Optional[str]] = {w: None for w in words}

    with brain.read_only():
        # A parse must not inherit residue -- training leaves the LAST trained
        # sentence's winners in the role areas (measured in nemo_parse.py:
        # deterministic 4-of-24 failures until cleared).
        for area in (ROLE_AGENT, ROLE_ACTION, ROLE_PATIENT):
            if area in brain.areas:
                brain.areas[area].unfix_assembly()
        brain.inhibit_areas([a for a in (ROLE_AGENT, ROLE_ACTION, ROLE_PATIENT)
                             if a in brain.areas])

        # RECORD: content words consume slots in the gated order. Nouns take
        # filler slots; the first verb-class token takes ACTION; function
        # words (det, aux, marker, prep) consume nothing -- they are control.
        slot_idx = 0
        action_taken = False
        fillers: List[Tuple[str, str]] = []
        for w in words:
            cat = cats.get(w)
            subcat = parser._func_subcat_of(w)
            if subcat is not None:
                continue  # function word: control, not a filler
            if cat in ("NOUN", "PRON") and slot_idx < len(sequence):
                open_slot = sequence[slot_idx]
                if _project_word(parser, w, open_slot):
                    fillers.append((w, open_slot))
                slot_idx += 1  # consumed even if unrecordable: no deadlock
            elif cat == "VERB" and not action_taken:
                if _project_word(parser, w, ROLE_ACTION):
                    out[w] = "ACTION"
                action_taken = True

        # Capture the parse's transient state BEFORE any replay disturbs it.
        snaps = {role: _snap(brain, role) for _w, role in fillers}
        diag["winners"] = {
            role: tuple(int(x) for x in snap.winners)
            for role, snap in snaps.items()
        }

        # RECALL: reconstruction. Candidates are the sentence's nouns -- the
        # things that could occupy a filler slot.
        nouns = [w for w in words if cats.get(w) in ("NOUN", "PRON")
                 and parser._func_subcat_of(w) is None]
        for role, snap in snaps.items():
            scores = {}
            for w in nouns:
                if not _project_word(parser, w, role):
                    continue
                image = _snap(brain, role)
                scores[w] = float(assembly_overlap(image, snap))
            if not scores:
                continue
            ranked = sorted(scores.items(), key=lambda kv: -kv[1])
            occupant, top = ranked[0]
            runner = ranked[1][1] if len(ranked) > 1 else 0.0
            diag["gaps"].append((role, occupant, top, runner, top - runner))
            if top > runner:  # strict: a tie is a failure to read, not a guess
                out[occupant] = {ROLE_AGENT: "AGENT",
                                 ROLE_PATIENT: "PATIENT"}[role]
    return out, diag


def margin_route(parser, words: List[str]) -> Dict[str, Optional[str]]:
    """The incumbent, on identical items."""
    cats = {w: parser.classify_word_cached(w)[0] for w in words}
    with parser.brain.read_only():
        return parser._assign_roles_neural(list(words), cats)


def _train(seed: int):
    """One trained parser. PHON_WEIGHT/BETA env vars select the substrate arm.

    Phase B (semantic-drive-share) measured phon_weight=6 + beta=0.05 cutting
    core-assembly duplicates 0.32 -> 0.06; the default was deliberately never
    flipped. The reconstruction readout's ties are image collapse through the
    shared-core channel, so this is the REGISTERED substrate lever: if parsing
    accuracy responds to it through the readout, the assembly is deciding.
    """
    random.seed(seed)
    np.random.seed(seed)
    kwargs = {}
    if os.environ.get("PROBE_PHON_WEIGHT"):
        kwargs["phon_weight"] = float(os.environ["PROBE_PHON_WEIGHT"])
    if os.environ.get("PROBE_BETA"):
        kwargs["beta"] = float(os.environ["PROBE_BETA"])
    parser = EmergentParser(
        n=N, k=K, seed=seed, vocabulary=build_vocabulary_preset("core"),
        fast_training=True, **kwargs)
    trainer = CurriculumTrainer(parser)
    for stage in STAGES:
        trainer.train_stage(stage)
    return parser


def _bias(parser, word: str) -> str:
    a = word in parser.role_lexicons.get(ROLE_AGENT, {})
    p = word in parser.role_lexicons.get(ROLE_PATIENT, {})
    return "AGENT" if a and not p else "PATIENT" if p and not a else "both/none"


#: (active frame, passive frame, same_event). Two pair types, DELIBERATELY:
#:
#:   same_event=True   the passive restates the active ("dog chases cat" /
#:                     "cat is chased by dog"). The parse's substrate state
#:                     must be IDENTICAL across the two frames -- the deepest
#:                     available criterion: a VOICE-INVARIANT event
#:                     representation, surface order absorbed by the gating.
#:   same_event=False  the passive states the REVERSED event ("child enters
#:                     mouse" / "child is entered by mouse"). Same surface
#:                     subject, opposite role -- the trained-bias trap the
#:                     margin route fails -- and the substrate state must
#:                     DIFFER, because who-did-what differs.
#:
#: The first registration of criterion C demanded "state differs between
#: frames" for every pair, which is WRONG for same-event pairs; the run itself
#: exposed the mis-registration (identical=True exactly on the restating
#: pairs, 30/30 winners shared, vs 1/30 on the reversed ones).
EVENTS = [
    ("the child enters the mouse", "the child is entered by the mouse", False),
    ("the dog chases the cat", "the cat is chased by the dog", True),
    ("the boy holds the ball", "the ball is held by the boy", True),
    ("the girl sees the horse", "the girl is seen by the horse", False),
]


def _frame_roles(text: str) -> Tuple[str, str]:
    """(agent, patient) read off the frame's construction, for scoring."""
    toks = text.split()
    if "by" in toks:
        return toks[-1], toks[1]     # passive: by-phrase noun is the agent
    return toks[1], toks[-1]         # active SVO


def main() -> None:
    print("Sentence-conditioned reconstruction readout -- pre-registered probe")
    print()
    ok = {"recon_active": 0, "recon_passive": 0,
          "margin_active": 0, "margin_passive": 0}
    n_frames = {"active": 0, "passive": 0}
    all_gaps: List[float] = []
    per_seed_gap_means: List[float] = []
    invariance: List[float] = []   # shared-winner fraction, same-event pairs
    separation: List[float] = []   # shared-winner fraction, reversed pairs

    for seed in SEEDS:
        parser = _train(seed)
        print(f"  === seed {seed} ===")
        seed_gaps: List[float] = []
        for active, passive, same_event in EVENTS:
            recon_by_frame = {}
            for kind, text in (("active", active), ("passive", passive)):
                words = text.split()
                agent, patient = _frame_roles(text)
                recon, diag = gated_parse_and_reconstruct(parser, words)
                marg = margin_route(parser, words)
                recon_by_frame[kind] = (recon, diag)

                n_frames[kind] += 1
                r_ok = (recon.get(agent) == "AGENT"
                        and recon.get(patient) == "PATIENT")
                m_ok = (marg.get(agent) == "AGENT"
                        and marg.get(patient) == "PATIENT")
                ok[f"recon_{kind}"] += r_ok
                ok[f"margin_{kind}"] += m_ok
                for (_role, _occ, _top, _runner, gap) in diag["gaps"]:
                    seed_gaps.append(gap)
                    all_gaps.append(gap)
                subj = words[1]
                recon_show = {k: v for k, v in recon.items() if v}
                marg_show = {k: v for k, v in marg.items() if v}
                print(f"      [{kind:7}] {text!r}"
                      + ("" if r_ok else "   <-- recon WRONG"))
                print(f"          recon  {recon_show}")
                print(f"          margin {marg_show}   "
                      f"(surface subject {subj!r} bias={_bias(parser, subj)})")

            # C, corrected two-sided form. Shared-winner fraction per role
            # area between the two frames:
            #   same event   -> should be ~1 (voice-INVARIANT event state)
            #   reversed     -> should be ~0 (who-did-what separates)
            (_ra, da), (_rp, dp) = (recon_by_frame["active"],
                                    recon_by_frame["passive"])
            common = set(da["winners"]) & set(dp["winners"])
            if common:
                frac = sum(
                    len(set(da["winners"][r]) & set(dp["winners"][r]))
                    / max(1, len(da["winners"][r]))
                    for r in common) / len(common)
                (invariance if same_event else separation).append(frac)

        gap_mean = sum(seed_gaps) / max(1, len(seed_gaps))
        per_seed_gap_means.append(gap_mean)
        print(f"      gap (occupant - runner-up): mean {gap_mean:.4f}  "
              f"min {min(seed_gaps):.4f}  n={len(seed_gaps)}")
        print()

    print("  === PRE-REGISTERED CRITERIA ===")
    print(f"    A recon on actives   : {ok['recon_active']}/{n_frames['active']}"
          f"   (margin route: {ok['margin_active']}/{n_frames['active']})")
    print(f"    B recon on passives  : {ok['recon_passive']}/{n_frames['passive']}"
          f"   (margin route: {ok['margin_passive']}/{n_frames['passive']})")
    inv = sum(invariance) / max(1, len(invariance))
    sep = sum(separation) / max(1, len(separation))
    print(f"    C1 voice INVARIANCE (same event, shared winners): {inv:.4f}"
          f"   (want ~1)")
    print(f"    C2 event SEPARATION (reversed, shared winners)  : {sep:.4f}"
          f"   (want ~0)")
    print(f"    D gap > 0 on all reads: {all(g > 0 for g in all_gaps)}   "
          f"min gap {min(all_gaps):.4f}")
    print(f"      per-seed gap means : "
          + ", ".join(f"{g:.4f}" for g in per_seed_gap_means)
          + ("   VARIES with substrate"
             if len(set(f"{g:.4f}" for g in per_seed_gap_means)) > 1
             else "   ZERO-WIDTH -- red flag"))


if __name__ == "__main__":
    main()

"""Batch projection helpers for compiled training schedules."""

from __future__ import annotations

from typing import TYPE_CHECKING

from neural_assemblies.assembly_calculus.ops import (
    bind,
)

# Kept in sync with parser_mixins.core._ROLE_BINDING_ROUNDS (imported lazily to
# avoid a circular import between the training and parser_mixins packages).
ROLE_BINDING_ROUNDS = 2

if TYPE_CHECKING:
    from ..parser_mixins.core import CoreParserMixin
    from ..core.grounding import GroundingContext


class BatchProjector:
    """Execute role/lexicon projections with shared batching utilities."""

    def __init__(self, parser: "CoreParserMixin"):
        self.parser = parser
        self.brain = parser.brain

    def apply_lexicon_word(
        self,
        word: str,
        ctx: "GroundingContext",
        core_area: str,
        *,
        rounds: int,
    ) -> None:
        """Project phon + grounding into core; preserve connectome between words."""
        p = self.parser
        p._clear_core_activity(core_area)

        phon = p.stim_map[word]
        stim_dict = {phon: [core_area]}
        for gs in p._grounding_stim_names(ctx):
            stim_dict[gs] = [core_area]

        p.brain.project(stim_dict, {})
        if rounds > 1:
            p.brain.project_rounds(
                target=core_area,
                areas_by_stim=stim_dict,
                dst_areas_by_src_area={core_area: [core_area]},
                rounds=rounds - 1,
            )

    def apply_role_update(
        self,
        word: str,
        role_area: str,
        *,
        rounds: int,
        clear_role: bool = True,
    ) -> None:
        """One Hebbian core → role update (matches unsupervised loop)."""
        p = self.parser
        core_area = p._word_core_area(word)
        phon = p.stim_map[word]

        # Prefer the stabilized core assembly; re-projecting phon -> core
        # applies plasticity and drifts the filler representation between
        # storing a binding and reading it back (see train_roles).
        #
        # But a stored assembly can be STALE: consolidation
        # (consolidation.prepare_area_for_replay) deliberately wipes an area's
        # compact_to_neuron_id and re-issues neuron IDs, which orphans every
        # snapshot taken before it -- its own docstring says "do not carry
        # pre-consolidation lexicons across". The novel-chat curriculum does
        # run consolidation between train_lexicon (which fills core_lexicons)
        # and the unsupervised role pass (which replays them here), so some
        # entries no longer map. Rather than crash in activate_assembly, fall
        # back to re-projecting phon -> core, which is exactly what a word with
        # no stored core already does and yields a fresh valid assembly.
        # ONE SHARED IMPLEMENTATION -- `ops.bind`, which now carries the
        # stale-snapshot guard this function used to be the only holder of, plus
        # the stimulus fallback every copy had hand-rolled. See its docstring
        # for why each property matters; the comments that were here are now
        # there, so they cannot drift out of sync with the code again.
        #
        # `clear_role` stays OUTSIDE the primitive on purpose: it interacts with
        # the compiled path's neuron-id pool (a ring-reuse device) and is not
        # part of the binding protocol.
        stored_core = p.core_lexicons.get(core_area, {}).get(word)
        asm = bind(
            p.brain, core_area, role_area, stored_core,
            source_stimulus=phon, project_rounds=rounds,
            tail_rounds=ROLE_BINDING_ROUNDS - 1,
        )

        if role_area not in p.role_lexicons:
            p.role_lexicons[role_area] = {}
        p.role_lexicons[role_area][word] = asm

        if clear_role and hasattr(p, "_clear_role_activity"):
            p._clear_role_activity(role_area)

    def apply_control_projection(
        self,
        stim: str,
        target_area: str,
        *,
        src_area: str | None = None,
        rounds: int,
    ) -> None:
        """Single control-area projection (mood/tense/polarity pattern)."""
        p = self.parser
        if src_area is None:
            p.brain.project({stim: [target_area]}, {target_area: [target_area]})
            areas_by_stim = {stim: [target_area]}
            dst = {target_area: [target_area]}
        else:
            p.brain.project(
                {stim: [target_area], p.stim_map.get(src_area, stim): [src_area]},
                {src_area: [target_area], target_area: [target_area]},
            )
            areas_by_stim = {stim: [target_area]}
            dst = {src_area: [target_area], target_area: [target_area]}

        if rounds > 1:
            p.brain.project_rounds(
                target=target_area,
                areas_by_stim=areas_by_stim,
                dst_areas_by_src_area=dst,
                rounds=rounds - 1,
            )

    def apply_control_batch(
        self,
        items: list[tuple[str, str, int]],
        *,
        src_area: str | None = None,
    ) -> None:
        """Run control-area projections; batch via multi-target when possible."""
        by_target: dict = {}
        for stim, target, rounds in items:
            by_target.setdefault(target, []).append((stim, rounds))

        for target, group in by_target.items():
            if len(group) == 1:
                stim, rounds = group[0]
                self.apply_control_projection(
                    stim, target, src_area=src_area, rounds=rounds,
                )
            else:
                for stim, rounds in group:
                    self.apply_control_projection(
                        stim, target, src_area=src_area, rounds=rounds,
                    )

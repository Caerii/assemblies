"""PhraseStructureMixin -- Composing constituents into phrase assemblies.

Split out of the CoreParserMixin monolith. Bodies are unchanged;
only their address is.
"""


from typing import Dict, List
from neural_assemblies.assembly_calculus.ops import project, merge, _snap

from ..core.areas import VERB_CORE, VP
from ..curriculum.data import GroundedSentence
from ._shared import MERGE_ROUNDS, _ROLE_BINDING_ROUNDS


class PhraseStructureMixin:
    """Composing constituents into phrase assemblies."""

    def train_phrases(self, sentences: List[GroundedSentence]):
        """Phase 3: Phrase structure via merge operations.

        For transitive sentences, merge subject and verb core assemblies
        into the VP area using the merge() operation.
        """
        for sent in sentences:
            subj_word = None
            verb_word = None
            obj_word = None

            for word, role in zip(sent.words, sent.roles):
                if role == "agent":
                    subj_word = word
                elif role == "action":
                    verb_word = word
                elif role == "patient":
                    obj_word = word

            if subj_word and verb_word:
                subj_core = self._word_core_area(subj_word)

                # Activate both source assemblies
                project(
                    self.brain, self.stim_map[subj_word],
                    subj_core, rounds=self.rounds,
                )
                project(
                    self.brain, self.stim_map[verb_word],
                    VERB_CORE, rounds=self.rounds,
                )

                # Merge subject + verb into VP.
                #
                # Shallow rounds, matching `_ROLE_BINDING_ROUNDS` above and for
                # the same documented reason: recurrence merges assemblies that
                # share an area. On the ISOLATED primitive this is decisive --
                # 94 merges into an n=1000 area give mean pairwise overlap
                # 0.752 at rounds=10 but 0.051 at rounds=1 (area size is the
                # other lever: n=3000 -> 0.231, n=10000 -> 0.002).
                vp_asm = merge(
                    self.brain, subj_core, VERB_CORE, VP,
                    rounds=MERGE_ROUNDS,
                )
                vp_key = f"{subj_word}_{verb_word}"
                self.vp_assemblies[vp_key] = vp_asm

                if obj_word:
                    obj_core = self._word_core_area(obj_word)
                    project(
                        self.brain, self.stim_map[obj_word],
                        obj_core, rounds=self.rounds,
                    )
                    # Extend VP with object via additional projection
                    self.brain.areas[obj_core].fix_assembly()
                    for _ in range(self.rounds):
                        self.brain.project(
                            {},
                            {obj_core: [VP], VP: [VP]},
                        )
                    self.brain.areas[obj_core].unfix_assembly()

                    vp_key_full = f"{subj_word}_{verb_word}_{obj_word}"
                    self.vp_assemblies[vp_key_full] = _snap(self.brain, VP)

                # NO reset_area_connections(VP) between sentences.
                #
                # It used to be here, "to reset VP connections for the next
                # sentence", and it was the cause of the phrase-structure
                # collapse -- the SAME failure already documented and fixed for
                # role areas above (see the note by `self.brain.project({},
                # {core_area: [role_area]})`). Resetting empties the sparse
                # connectome, so on the next merge every candidate neuron in VP
                # has equal input and the deterministic index tie-break in
                # winner selection hands back the same k neurons regardless of
                # which words were merged.
                #
                # Measured paired over seeds 42/7/123, means; the arms differ
                # only in this one line, and every seed agrees:
                #
                #                    pairwise    VP    rank-1 retrieval
                #                     overlap     w      subj    verb
                #   with the reset      1.000    74     0.156   0.094
                #   without it          0.492   752     0.781   0.646
                #
                # w is how many VP neurons have EVER fired: with the reset only
                # ~74 of 1000 were ever recruited, so k=50 winners drawn from a
                # 74-neuron pool cannot be distinct -- every phrase was the same
                # assembly, pairwise overlap 1.000.
                #
                # Rank-1 cues VP with ONE parent and asks whether the top-
                # scoring stored constituent contains that parent; chance is
                # 0.031 over the 32-constituent pool. The reset arm's 0.156 is
                # not partial credit, it is a tie-break artifact: when every
                # assembly is identical, `max` returns whichever key it sees
                # first. This is the property merge exists to provide (ops.merge,
                # [PNAS20] sec 3 -- the merged assembly responds to EITHER
                # source alone); without it a constituent is not retrievable
                # from its parts and nothing can be composed further from it.
                #
                # Carrying VP's connectome across sentences is not a leak to be
                # cleaned up; it is where the phrase lexicon LIVES. The role
                # areas reached the same conclusion for the same reason.

    def _identify_phrases(self, words: List[str],
                          categories: Dict[str, str]) -> dict:
        """Identify NP, VP, PP phrase boundaries from category sequence."""
        phrases: dict = {"NP": [], "VP": [], "PP": []}
        current_np: List[str] = []

        for word in words:
            cat = categories.get(word, "UNKNOWN")
            if cat in ("DET", "ADJ", "NOUN", "PRON"):
                current_np.append(word)
            else:
                if current_np:
                    phrases["NP"].append(current_np[:])
                    current_np = []
                if cat == "VERB":
                    phrases["VP"].append([word])
                elif cat == "PREP":
                    phrases["PP"].append([word])

        if current_np:
            phrases["NP"].append(current_np)

        return phrases

"""Incremental and recursive parsing mixin for EmergentParser.

Provides word-by-word incremental parsing with FiberCircuit gating
and recursive clause handling for embedded relative clauses.

WHY INCREMENTAL AT ALL.  Human comprehension does not wait for the end of the
sentence; each word is integrated as it arrives, which is what makes garden
paths and ERP effects observable in the first place.  A parser that consumed a
whole sentence and then produced a structure would have no state to measure at
word ``i``.  This mixin processes one word per step so that per-word
quantities exist.

THE GATING CYCLE.  Each word is handled as PRE-rules, project, POST-rules:

    _apply_pre_rules   open the fibers this word's category needs, so the
                       word is routed to the right syntactic slot
    (projection)       one FiberCircuit step -- only the open fibers fire
    _apply_post_rules  close the fibers this word consumed, so the slot it
                       filled is no longer available to the next word

The closing half is what makes the parse a sequence of commitments rather than
a soup.  Inhibiting NOUN_CORE -> SUBJ after a subject noun is how the parser
records that the subject slot is taken; the next noun finds only the object
route open.  Note the consequence: a wrong routing decision is not
recoverable within a parse, because the fiber that would have carried the
correction is already closed.  That is the mechanistic correlate of a garden
path, not a defect to be patched.

Routing is not hardcoded to English -- ``_get_syntactic_target`` consults the
learned word-order typology, so which slot a noun goes to depends on whether
the verb has been seen and what order the corpus taught.

RING MODE.  CONTEXT is a fixed-size buffer, and ``_init_context_ring``
pre-allocates its connectome columns before use.  The subtlety recorded in
that function is worth reading before touching it: pre-growth runs with
plasticity DISABLED and with CONTEXT self-recurrence OFF, because with
plasticity off every weight is 1, every candidate ties, and the deterministic
index tie-break re-selects the same winners forever -- allocating only ~k
columns however long the sequence.  Padding uses DISTINCT vocabulary words
for the same reason.  Ring capacity is sized to ``capacity * k``, one
assembly per prefix position, because ring mode is a hard cap and an
under-measured capacity permanently prevents CONTEXT from telling prefixes
apart.
"""

from typing import List, Optional, Tuple, TYPE_CHECKING

from neural_assemblies.assembly_calculus.assembly import Assembly
from neural_assemblies.assembly_calculus.consolidation import accumulate_context_step
from neural_assemblies.assembly_calculus.ops import (
    activate_assembly, project, _snap,
)
from neural_assemblies.assembly_calculus.fiber import FiberCircuit

from ..core.areas import (
    NOUN_CORE, VERB_CORE, ADJ_CORE, ADV_CORE,
    PREP_CORE, DET_CORE, PRON_CORE, CONJ_CORE,
    CORE_AREAS, CATEGORY_TO_CORE,
    SUBJ, OBJ, VP, PP,
    TENSE, MOOD, SENT,
    CONTEXT, DEP_CLAUSE,
    ALL_AREAS,
)

if TYPE_CHECKING:
    from ..curriculum.data import GroundedSentence


class IncrementalMixin:
    """Incremental word-by-word parsing and recursive clause handling."""

    _context_ring_capacity: int = 0
    _context_ring_capacity_cols: int = 0
    _prediction_ring_capacity_cols: int = 0
    _role_ring_capacity_cols: dict = {}
    _core_ring_capacity_cols: dict = {}
    _role_paths_bootstrapped: bool = False

    #: Verbs the corpus ever gives a patient. EMPTY MEANS "UNKNOWN", NOT
    #: "NONE": with no transitivity information every verb opens the object
    #: slot, which is the behaviour that shipped before this existed. Populate
    #: it (see `acquisition.transitivity.learn_transitivity`) to switch the
    #: paper's empty-project detector on.
    transitive_verbs: Optional[set] = None

    #: Structural violations seen in the CURRENT sentence, newest last. Reset
    #: whenever the circuit is reset. See `assembly_calculus.parse_errors`.
    _parse_errors: Optional[list] = None

    @property
    def parse_errors(self) -> list:
        """Structural violations detected while parsing the current sentence.

        The parser paper's own signals, not a graded score: an ``EmptyProject``
        here means ``project*`` had nowhere to send a word. Empty list means a
        clean parse -- which is NOT the same as "no detector ran", so check
        `transitive_verbs` if this is always empty.
        """
        if self._parse_errors is None:
            self._parse_errors = []
        return self._parse_errors

    def _verb_takes_an_object(self, word: Optional[str]) -> bool:
        """Does this verb license an object slot?

        UNKNOWN VERBS TAKE OBJECTS. Defaulting the other way would make every
        untrained parser reject every object, turning a missing lexicon into a
        stream of confident violations -- the failure mode where an apparatus
        defect reads as a linguistic result.
        """
        known = self.transitive_verbs
        if not known or word is None:
            return True
        return word in known

    def _set_freeze_connectome_growth(
        self, area_names, *, enabled: bool,
    ) -> None:
        """Skip connectome matrix growth during compiled training."""
        for area_name in area_names:
            engine = self.brain._engine_for(self.brain.areas[area_name])
            if not hasattr(engine, "_areas"):
                continue
            if area_name in engine._areas:
                engine._areas[area_name]._freeze_connectome_growth = enabled

    def _set_compiled_topology_mode(
        self, area_names, *, enabled: bool,
    ) -> None:
        """Fixed-topology projection: top-k on pregrown columns only."""
        for area_name in area_names:
            engine = self.brain._engine_for(self.brain.areas[area_name])
            if not hasattr(engine, "_areas"):
                continue
            if area_name in engine._areas:
                engine._areas[area_name]._plasticity_only_mode = enabled

    def _enable_area_ring_mode(self, area_name: str, capacity_cols: int) -> None:
        """Reuse pregrown connectome columns during training (skip expand)."""
        if capacity_cols <= 0:
            return
        engine = self.brain._engine_for(self.brain.areas[area_name])
        if not hasattr(engine, "_areas") or area_name not in engine._areas:
            return
        st = engine._areas[area_name]
        st._ring_mode = True
        st._ring_capacity_cols = capacity_cols

    def _disable_area_ring_mode(self, area_name: str) -> None:
        engine = self.brain._engine_for(self.brain.areas[area_name])
        if not hasattr(engine, "_areas") or area_name not in engine._areas:
            return
        st = engine._areas[area_name]
        st._ring_mode = False

    def _enable_context_ring_mode(self, capacity_cols: int) -> None:
        """Reuse pregrown CONTEXT columns during bridge training (no expand)."""
        self._enable_area_ring_mode(CONTEXT, capacity_cols)

    def _disable_context_ring_mode(self) -> None:
        self._disable_area_ring_mode(CONTEXT)

    def _enable_prediction_ring_mode(self, capacity_cols: int) -> None:
        """Reuse pregrown PREDICTION columns during lexicon/bridge training."""
        from ..core.areas import PREDICTION
        self._enable_area_ring_mode(PREDICTION, capacity_cols)

    def _disable_prediction_ring_mode(self) -> None:
        from ..core.areas import PREDICTION
        self._disable_area_ring_mode(PREDICTION)

    def _context_compiled_active(self) -> bool:
        """True when CONTEXT ring reuse is enabled for bridge training."""
        engine = self.brain._engine_for(self.brain.areas[CONTEXT])
        if not hasattr(engine, "_areas") or CONTEXT not in engine._areas:
            return False
        return bool(getattr(engine._areas[CONTEXT], "_ring_mode", False))

    def _init_context_ring(self, capacity: int, words: Optional[List[str]] = None) -> None:
        """Reserve CONTEXT connectome depth for fixed-size sentence buffer."""
        if capacity < 2:
            return
        self._context_ring_capacity = capacity
        self._bootstrap_prediction_connectivity()

        # Pad with DISTINCT vocabulary words. Repeating one word makes every
        # pre-growth step project the same core assembly, so the same columns
        # are reused and almost none are allocated.
        vocab = [w for w in self.stim_map if w not in set(words or ())]
        seq = list(words or ())
        i = 0
        while len(seq) < capacity and i < len(vocab):
            seq.append(vocab[i])
            i += 1
        while len(seq) < capacity and self.stim_map:
            seq.append(next(iter(self.stim_map.keys())))

        with self.brain.frozen():
            self._reset_context_state()
            for word in seq[:capacity]:
                if word not in self.stim_map:
                    continue
                # Pre-growth exists to ALLOCATE connectome columns, not to
                # build a meaningful context, so it is input-driven. With
                # plasticity disabled every recurrent weight is 1, so
                # including CONTEXT self-recurrence leaves all candidates on
                # equal input and the deterministic index tie-break re-selects
                # the same winners every step -- the area then allocates only
                # ~k columns no matter how long the sequence is.
                core_area = self._word_core_area(word)
                core_asm = self.core_lexicons.get(core_area, {}).get(word)
                if core_asm is not None:
                    activate_assembly(self.brain, core_asm)
                else:
                    project(self.brain, self.stim_map[word], core_area,
                            rounds=self.inference_rounds)
                self.brain.project({}, {core_area: [CONTEXT]})

        engine = self.brain._engine_for(self.brain.areas[CONTEXT])
        if hasattr(engine, "_areas") and CONTEXT in engine._areas:
            # Size the ring by what the representation REQUIRES -- one
            # k-assembly per prefix position -- not merely by what pre-growth
            # happened to allocate. Ring mode is a hard cap on recruitment, so
            # an under-measured capacity permanently prevents CONTEXT from
            # distinguishing prefixes.
            required_cols = capacity * self.k
            self._context_ring_capacity_cols = max(
                self._context_ring_capacity_cols,
                int(engine._areas[CONTEXT].w),
                required_cols,
            )
        if self._context_ring_capacity_cols >= self.k:
            self._enable_context_ring_mode(self._context_ring_capacity_cols)
        self._reset_context_for_bridge(preserve_topology=True)

    def _pregrow_context_capacity(
        self,
        sentences: List["GroundedSentence"],
        *,
        max_len: Optional[int] = None,
    ) -> None:
        """Pre-expand CONTEXT connectomes to corpus max prefix length."""
        best_words: List[str] = []
        for sent in sentences:
            words = [w for w in sent.words if w in self.stim_map]
            if len(words) > len(best_words):
                best_words = words

        if max_len is not None:
            capacity = max(max_len, len(best_words), 2)
        else:
            capacity = max(len(best_words), 2)

        if capacity < 2:
            return

        if len(best_words) < capacity:
            pad = (
                best_words[-1]
                if best_words
                else next(iter(self.stim_map.keys()))
            )
            seq = list(best_words) + [pad] * (capacity - len(best_words))
        else:
            seq = best_words[:capacity]

        self._init_context_ring(capacity, words=seq)

    def _restore_context_from_cache(
        self,
        assembly: Assembly,
        *,
        word: Optional[str] = None,
        rounds: Optional[int] = None,
    ) -> None:
        """Restore CONTEXT from snapshot; optionally advance one word."""
        from neural_assemblies.assembly_calculus.ops import activate_assembly

        activate_assembly(self.brain, assembly)
        if word is not None:
            self._advance_context_direct(
                word, rounds=rounds, use_lexicon_core=True,
            )

    def _reset_context_winners(self, *, preserve_mapping: bool = False) -> None:
        """Clear CONTEXT activity; ID remapping is forbidden during observation."""
        self._check_context_reset(preserve_mapping=preserve_mapping)
        self.brain.inhibit_areas([CONTEXT])
        if self.brain.is_fixed(CONTEXT):
            self.brain.unfix_assembly(CONTEXT)
        self.brain.reset_area_population_cursor(
            CONTEXT, preserve_mapping=preserve_mapping,
            reset_count=not preserve_mapping,
        )

    def _check_context_reset(self, *, preserve_mapping: bool) -> None:
        engine = self.brain._engine_for(self.brain.areas[CONTEXT])
        if not preserve_mapping and getattr(engine, "_no_recruitment", False):
            raise ValueError(
                "read_only cannot reset CONTEXT population or neuron identities; "
                "use build_context_incremental(..., preserve_topology=True)")

    def _reset_context_state(self) -> None:
        """Reset recruitment for sentence construction, retaining learned fibers.

        This legacy cursor reset is distinct from an activity-only probe reset.
        The shared winner/ID reset rejects it before mutation in read_only.
        """
        self._reset_context_winners(preserve_mapping=False)

    def _reset_context_for_bridge(self, *, preserve_topology: bool = False) -> None:
        """Specification: docs/reviews/whole-codebase/SEMANTIC_CARDS.md#contract-context-bridge-reset

        Clear bridge activity, optionally preserving the allocated population.

        Requested ring capacity is configuration, not evidence that its neurons
        exist. Preserve the backend count and IDs; never replace the count with
        that request. Construction resets remain forbidden during observation.
        """
        self._check_context_reset(preserve_mapping=False)
        if not preserve_topology:
            self._reset_context_state()
            return
        self._reset_context_winners(preserve_mapping=True)

    def _build_circuit(self) -> FiberCircuit:
        """Build a FiberCircuit with all projection channels initially inhibited.

        Declares fibers for:
        - core → SUBJ, core → OBJ (syntactic role routing)
        - core → VP (verb phrase building)
        - core → CONTEXT (running context accumulation)
        - DET_CORE/ADJ_CORE → SUBJ/OBJ (modifier → syntactic role)
        - SUBJ → VP (subject-verb merge)

        All fibers start inhibited; pre-rules disinhibit them as needed.
        """
        circuit = FiberCircuit(self.brain)

        # Core areas → syntactic role areas
        for core in [NOUN_CORE, PRON_CORE]:
            circuit.add(core, SUBJ, active=False)
            circuit.add(core, OBJ, active=False)

        # Modifier cores → syntactic role areas (merge det/adj into NP)
        circuit.add(DET_CORE, SUBJ, active=False)
        circuit.add(DET_CORE, OBJ, active=False)
        circuit.add(ADJ_CORE, SUBJ, active=False)
        circuit.add(ADJ_CORE, OBJ, active=False)

        # Verb → VP
        circuit.add(VERB_CORE, VP, active=False)
        # Subject → VP merge
        circuit.add(SUBJ, VP, active=False)
        # Object → VP merge
        circuit.add(OBJ, VP, active=False)

        # Prep → PP
        circuit.add(PREP_CORE, PP, active=False)

        # Adverb → VP
        circuit.add(ADV_CORE, VP, active=False)

        # Verb → TENSE (tense binding during incremental parse)
        circuit.add(VERB_CORE, TENSE, active=False)

        # MOOD → SENT (mood feeds sentence-level representation)
        circuit.add(MOOD, SENT, active=False)

        # CONJ_CORE → SENT (conjunctions link sentence-level structures)
        circuit.add(CONJ_CORE, SENT, active=False)

        # All cores → CONTEXT (sequential context building)
        for core in CORE_AREAS:
            circuit.add(core, CONTEXT, active=False)
        # CONTEXT self-recurrence
        circuit.add(CONTEXT, CONTEXT, active=False)

        return circuit

    def _get_syntactic_target(self, verb_seen: bool,
                              noun_count: int) -> str:
        """Determine whether to route to SUBJ or OBJ based on word order.

        Args:
            verb_seen: Whether a verb has been processed.
            noun_count: Number of nouns seen so far.

        Returns:
            SUBJ or OBJ area constant.
        """
        from ..core.word_order import WORD_ORDERS, order_slots

        order = self.word_order_type
        if order not in WORD_ORDERS:
            order = "SVO"
        slots = order_slots(order)
        v_at = slots.index("V")
        pre = [s for i, s in enumerate(slots) if s != "V" and i < v_at]
        post = [s for i, s in enumerate(slots) if s != "V" and i > v_at]

        # Which noun slot is this noun filling? Nouns before the verb consume
        # the pre-verbal slots in order, nouns after it the post-verbal ones.
        # For an object-initial order the FIRST noun routes to OBJ, which the
        # old SVO/SOV/VSO ladder could not express.
        seq = post if verb_seen else pre
        if not seq:
            seq = [s for s in slots if s != "V"]
        rank = min(noun_count, len(seq) - 1)
        return SUBJ if seq[rank] == "S" else OBJ

    def _apply_pre_rules(self, circuit: FiberCircuit, category: str,
                         verb_seen: bool, noun_count: int = 0) -> None:
        """Disinhibit fibers before projecting the current word.

        Mirrors recursive_parser.py PRE_RULES: open channels that this
        word category needs for projection. Adapts routing based on
        learned word order typology.
        """
        target = self._get_syntactic_target(verb_seen, noun_count)

        # EMPTY-PROJECT (Mitropolsky et al. 2021 sec. 6). If the slot this word
        # needs is an INHIBITED AREA, project* has nowhere to send it and the
        # paper calls that a detected syntactic violation -- their example is
        # "the dogs lived" followed by "cats", where OBJ was never disinhibited
        # because the verb was intransitive.
        #
        # Returning early rather than opening the fiber anyway is the whole
        # point. The rules used to be TOTAL: `_get_syntactic_target` clamps
        # with min(noun_count, len(seq)-1) and always names SUBJ or OBJ, so
        # every word always had a route and no violation could ever surface.
        # A grammar that cannot reject is not a grammar.
        # AN OPEN PP IS A ROUTE, so a noun after a preposition is NOT stranded
        # and must not be flagged. Measured: without this check, "the dog
        # sleeps on the table" reports 2 errors -- the SAME count as the real
        # violation "the dog sleeps the cat", making the detector useless on
        # any corpus containing prepositions.
        #
        # KNOWN GAP, pinned in test_parse_errors_live.py: the circuit has no
        # `core -> PP` fiber, so "table" is not flagged but does not bind into
        # PP either. Suppressing a false alarm is not the same as routing the
        # word correctly, and this only does the first.
        in_pp = category in ("DET", "ADJ", "NOUN", "PRON") and \
            circuit.is_active(PREP_CORE, PP)

        # EMPTY-PROJECT (Mitropolsky et al. 2021 sec. 6). If the slot this word
        # needs is an INHIBITED AREA, project* has nowhere to send it and the
        # paper calls that a detected syntactic violation -- their example is
        # "the dogs lived" followed by "cats", where OBJ was never disinhibited
        # because the verb was intransitive.
        #
        # Returning early rather than opening the fiber anyway is the whole
        # point. The rules used to be TOTAL: `_get_syntactic_target` clamps
        # with min(noun_count, len(seq)-1) and always names SUBJ or OBJ, so
        # every word always had a route and no violation could ever surface.
        # A grammar that cannot reject is not a grammar.
        if category in ("DET", "ADJ", "NOUN", "PRON") and not in_pp:
            gate = getattr(self.brain, "_inhibition", None)
            if gate is not None and not gate.area_open(target):
                from neural_assemblies.assembly_calculus.parse_errors import (
                    EmptyProject,
                )
                self.parse_errors.append(EmptyProject(
                    empty=True, detected_by="drive", lex_targets=()))
                return

        if category == "DET":
            circuit.disinhibit(DET_CORE, target)

        elif category == "ADJ":
            circuit.disinhibit(ADJ_CORE, target)

        elif category in ("NOUN", "PRON"):
            core = NOUN_CORE if category == "NOUN" else PRON_CORE
            circuit.disinhibit(core, target)
            # Merge DET/ADJ into the same syntactic slot
            circuit.disinhibit(DET_CORE, target)
            circuit.disinhibit(ADJ_CORE, target)

        elif category == "VERB":
            circuit.disinhibit(VERB_CORE, VP)
            circuit.disinhibit(SUBJ, VP)
            # Activate verb → TENSE binding
            circuit.disinhibit(VERB_CORE, TENSE)

        elif category == "PREP":
            circuit.disinhibit(PREP_CORE, PP)

        elif category == "ADV":
            circuit.disinhibit(ADV_CORE, VP)

        elif category == "CONJ":
            circuit.disinhibit(CONJ_CORE, SENT)

    def _apply_post_rules(self, circuit: FiberCircuit, category: str,
                          verb_seen: bool, noun_count: int = 0,
                          word: Optional[str] = None) -> None:
        """Inhibit fibers after projecting, preparing for next word.

        Mirrors recursive_parser.py POST_RULES: close channels that
        were consumed by this word.

        AND, for a verb, decides whether the OBJECT SLOT EXISTS AT ALL. That is
        the paper's mechanism for detecting a syntactic violation: an
        intransitive verb never disinhibits area OBJ, so a following noun has
        nowhere to go and `project*` fires nothing. It needs AREA inhibition,
        not fiber gating -- which is why this could not be expressed before
        `Brain.inhibit_area` existed, and why the FiberCircuit alone (all this
        parser had) could never produce the violation.

        Unknown verbs keep the slot open; see `_verb_takes_an_object`.
        """
        if category in ("NOUN", "PRON"):
            target = self._get_syntactic_target(verb_seen, noun_count)
            core = NOUN_CORE if category == "NOUN" else PRON_CORE
            # Close noun/det/adj → syntactic slot
            circuit.inhibit(core, target)
            circuit.inhibit(DET_CORE, target)
            circuit.inhibit(ADJ_CORE, target)

        elif category == "VERB":
            # Close subject → SUBJ routing (subject slot filled)
            for core in [NOUN_CORE, PRON_CORE]:
                if circuit.is_active(core, SUBJ):
                    circuit.inhibit(core, SUBJ)
            if circuit.is_active(DET_CORE, SUBJ):
                circuit.inhibit(DET_CORE, SUBJ)
            if circuit.is_active(ADJ_CORE, SUBJ):
                circuit.inhibit(ADJ_CORE, SUBJ)
            # Open object routing
            circuit.disinhibit(VERB_CORE, VP)
            circuit.disinhibit(OBJ, VP)

            # ...and decide whether the object slot exists. Only touched when
            # transitivity is actually known, so a parser without it keeps the
            # previous behaviour exactly and no gate is ever allocated.
            if self.transitive_verbs:
                if self._verb_takes_an_object(word):
                    self.brain.disinhibit_area(OBJ)
                else:
                    self.brain.inhibit_area(OBJ)

    def _get_incremental_circuit(self, *, reset: bool = False) -> FiberCircuit:
        """Reuse a FiberCircuit across words/sentences when possible."""
        if reset or self._incremental_circuit is None:
            self._incremental_circuit = self._build_circuit()
            # A new sentence starts with every slot available and no errors.
            # Leaving OBJ inhibited from the last sentence's intransitive verb
            # would make the NEXT sentence's object read as a violation -- a
            # gate that leaks across sentences is worse than no gate, because
            # it produces confident wrong detections rather than none.
            self._parse_errors = []
            # Reopen whenever a gate EXISTS, not only when transitivity is
            # currently known. Gating on `self.transitive_verbs` here left a
            # real bug: turning transitivity back off did not reopen OBJ, so a
            # parser that had once seen an intransitive verb kept flagging
            # every later object -- caught by the test that asserts silence
            # with transitivity disabled.
            if self.brain._inhibition is not None:
                self.brain.disinhibit_area(OBJ)
        return self._incremental_circuit

    def _advance_context_direct(
        self,
        word: str,
        *,
        rounds: Optional[int] = None,
        use_lexicon_core: bool = True,
    ) -> str:
        """Build CONTEXT for one word without FiberCircuit gating."""
        rounds = rounds if rounds is not None else self.inference_rounds

        grounding = self.word_grounding.get(word)
        cat, _ = self.classify_word_cached(word, grounding=grounding)

        phon = self.stim_map.get(word)
        core_area = self._word_core_area(word)
        core_asm = None
        if use_lexicon_core:
            core_asm = self.core_lexicons.get(core_area, {}).get(word)

        if core_asm is not None:
            try:
                accumulate_context_step(
                    self.brain,
                    core_area=core_area,
                    context_area=CONTEXT,
                    core_assembly=core_asm,
                    rounds=rounds,
                )
            except ValueError:
                core_asm = None

        if core_asm is None and phon is not None:
            accumulate_context_step(
                self.brain,
                phon=phon,
                core_area=core_area,
                context_area=CONTEXT,
                rounds=rounds,
            )
        return cat

    def _advance_incremental_word(
        self,
        word: str,
        circuit: FiberCircuit,
        verb_seen: bool,
        noun_count: int,
        *,
        rounds: Optional[int] = None,
        forced_category: Optional[str] = None,
    ) -> Tuple[str, bool, int]:
        """Process one word through incremental gating + context build."""
        rounds = rounds if rounds is not None else self.inference_rounds

        grounding = self.word_grounding.get(word)
        if forced_category is not None:
            cat = forced_category
        else:
            cat, _ = self.classify_word_cached(word, grounding=grounding)

        phon = self.stim_map.get(word)
        core_area = CATEGORY_TO_CORE.get(cat, self._word_core_area(word))
        if phon is not None:
            project(self.brain, phon, core_area, rounds=rounds)

        self._apply_pre_rules(circuit, cat, verb_seen, noun_count)

        for _ in range(rounds):
            circuit.step()

        self.brain.project(
            {},
            {core_area: [CONTEXT], CONTEXT: [CONTEXT]},
        )
        if rounds > 1:
            self.brain.project_rounds(
                target=CONTEXT,
                areas_by_stim={},
                dst_areas_by_src_area={
                    core_area: [CONTEXT],
                    CONTEXT: [CONTEXT],
                },
                rounds=rounds - 1,
            )

        self._apply_post_rules(circuit, cat, verb_seen, noun_count, word=word)

        if cat == "VERB":
            verb_seen = True
        if cat in ("NOUN", "PRON"):
            noun_count += 1

        return cat, verb_seen, noun_count

    def build_context_incremental(
        self,
        words: List[str],
        *,
        reset: bool = True,
        direct: bool = False,
        preserve_topology: bool = False,
    ) -> dict:
        """Specification: docs/reviews/whole-codebase/SEMANTIC_CARDS.md#contract-context-observation

        Lightweight incremental parse: CONTEXT + categories only.

        Skips role assignment, phrase detection, and morphosyntax — used
        by prediction and dialogue hot paths.

        When ``direct=True``, skips FiberCircuit gating (faster hot path).
        ``preserve_topology=True`` resets activity while retaining the existing
        population and neuron mapping. Required for read-only prefix probes.
        The legacy default resets recruitment and can construct new neurons.
        """
        result: dict = {
            "categories": {},
            "steps": [],
        }

        if reset:
            if preserve_topology:
                self._reset_context_winners(preserve_mapping=True)
            else:
                self._reset_context_state()

        verb_seen = False
        noun_count = 0
        steps: List[dict] = []

        with self.brain.frozen():
            if direct:
                for i, word in enumerate(words):
                    cat = self._advance_context_direct(word)
                    result["categories"][word] = cat
                    steps.append({
                        "word": word,
                        "category": cat,
                        "position": i,
                        "context_assembly": _snap(self.brain, CONTEXT),
                    })
            else:
                circuit = self._get_incremental_circuit(reset=reset)
                for i, word in enumerate(words):
                    cat, verb_seen, noun_count = self._advance_incremental_word(
                        word, circuit, verb_seen, noun_count,
                    )
                    result["categories"][word] = cat
                    steps.append({
                        "word": word,
                        "category": cat,
                        "position": i,
                        "context_assembly": _snap(self.brain, CONTEXT),
                        "verb_seen": verb_seen,
                    })

        result["steps"] = steps
        return result

    def parse_incremental(
        self,
        words: List[str],
        *,
        reset: bool = True,
        light: bool = False,
    ) -> dict:
        """Parse a sentence word-by-word with FiberCircuit gating.

        For each word:
        1. Classify (phon + grounding → core area readout)
        2. Project phon → core area to activate word assembly
        3. Apply pre-rules (disinhibit appropriate channels)
        4. Project through active fibers (circuit.step × rounds)
        5. Project core → CONTEXT with recurrence (build context)
        6. Apply post-rules (inhibit consumed channels)
        7. Snapshot state

        Returns same format as parse() plus:
        - "steps": per-word snapshots with category and context assembly

        When ``light=True``, only builds categories and context (no roles,
        phrases, or morphosyntax detection).
        """
        if light:
            result = self.build_context_incremental(words, reset=reset)
            result.setdefault("roles", {})
            result.setdefault("phrases", {})
            return result

        result: dict = {
            "categories": {},
            "roles": {},
            "phrases": {},
            "steps": [],
        }

        if reset:
            self._reset_context_state()

        circuit = self._get_incremental_circuit(reset=reset)
        verb_seen = False
        noun_count = 0
        steps: List[dict] = []

        for i, word in enumerate(words):
            cat, verb_seen, noun_count = self._advance_incremental_word(
                word, circuit, verb_seen, noun_count,
            )
            result["categories"][word] = cat

            ctx_assembly = _snap(self.brain, CONTEXT)
            steps.append({
                "word": word,
                "category": cat,
                "position": i,
                "context_assembly": ctx_assembly,
                "verb_seen": verb_seen,
            })

        result["steps"] = steps

        # Roles from THIS parse (same route as batch parse -- one canonical
        # readout, see core.py parse()).
        result["roles"], result["role_diagnostics"] = (
            self.parse_roles_by_reconstruction(words))

        # Identify phrases
        result["phrases"] = self._identify_phrases(
            words, result["categories"])

        # Detect tense, mood, polarity
        result["tense"] = self.detect_tense(words)
        result["mood"] = self.detect_mood(words)
        result["polarity"] = self.detect_polarity(words)

        return result

    @staticmethod
    def _detect_clause_boundary(word: str, prev_category: Optional[str]
                                ) -> bool:
        """Returns True if word is a relative pronoun after a NOUN."""
        return word in ("that", "which") and prev_category in ("NOUN", "PRON")

    def _save_outer_state(self) -> dict:
        """Save current winner assemblies in all active areas."""
        state = {}
        for area_name in ALL_AREAS:
            area = self.brain.areas[area_name]
            if hasattr(area, 'winners') and area.winners is not None:
                state[area_name] = list(area.winners)
            else:
                state[area_name] = []
        return state

    def _restore_outer_state(self, state: dict) -> None:
        """Restore winner assemblies saved by ``_save_outer_state``."""
        for area_name, winners in state.items():
            if area_name not in self.brain.areas:
                continue
            if winners:
                import numpy as np
                arr = np.asarray(winners, dtype=np.uint32)
                engine = self.brain._engine_for(self.brain.areas[area_name])
                engine.set_winners(area_name, arr)
                self.brain.areas[area_name].winners = arr
            else:
                self.brain.inhibit_areas([area_name])

    def _replay_without_plasticity(self, words: List[str],
                                   circuit: FiberCircuit) -> None:
        """Replay word sequence without plasticity to restore outer context.

        Mirrors recursive_parser.py:850-868: disable plasticity, replay
        each outer word (classify + pre-rules + project + post-rules),
        then re-enable plasticity.
        """
        with self.brain.frozen():
            verb_seen = False
            noun_count = 0
            infer_r = self.inference_rounds
            for word in words:
                grounding = self.word_grounding.get(word)
                cat, _ = self.classify_word_cached(word, grounding=grounding)

                phon = self.stim_map.get(word)
                core_area = self._word_core_area(word)
                if phon is not None:
                    project(self.brain, phon, core_area, rounds=infer_r)

                self._apply_pre_rules(circuit, cat, verb_seen, noun_count)
                circuit.step()
                self._apply_post_rules(circuit, cat, verb_seen, noun_count)

                if cat == "VERB":
                    verb_seen = True
                if cat in ("NOUN", "PRON"):
                    noun_count += 1

    def parse_recursive(self, words: List[str]) -> dict:
        """Parse a sentence that may contain embedded relative clauses.

        Extends incremental parsing with clause boundary detection:

        On "that"/"which" after a NOUN:
        1. Save outer context (fix all area assemblies)
        2. Disinhibit DEP_CLAUSE, project antecedent noun → DEP_CLAUSE
        3. Parse inner clause words incrementally

        On "," or sentence end after embedded clause:
        1. Snapshot DEP_CLAUSE assembly
        2. Replay outer words WITHOUT plasticity to restore context
        3. Inhibit DEP_CLAUSE, continue main parse

        Args:
            words: Full sentence including clause markers ("that", ",").

        Returns:
            Same format as parse() plus:
            - "clauses": {"main": [...], "embedded": [...]}
            - "dep_clause_assembly": Assembly of embedded clause (or None)
        """
        result: dict = {
            "categories": {},
            "roles": {},
            "inner_roles": {},
            "phrases": {},
            "clauses": {"main": [], "embedded": []},
            "dep_clause_assembly": None,
        }

        circuit = self._build_circuit()
        verb_seen = False
        noun_count = 0
        in_clause = False
        outer_words: List[str] = []
        inner_words: List[str] = []
        main_words: List[str] = []
        prev_category: Optional[str] = None

        antecedent_word: Optional[str] = None
        inner_verb_seen = False

        i = 0
        while i < len(words):
            word = words[i]

            # --- Classify early so we can use category for clause decisions ---
            grounding = self.word_grounding.get(word)
            cat, _ = self.classify_word_cached(word, grounding=grounding)
            result["categories"][word] = cat

            # --- Check for clause entry ---
            if (not in_clause
                    and self._detect_clause_boundary(word, prev_category)):
                in_clause = True
                outer_words = list(main_words)  # Words before "that"
                inner_verb_seen = False

                # Remember the antecedent noun for filler-gap binding
                if main_words:
                    antecedent_word = main_words[-1]
                    ant_phon = self.stim_map.get(antecedent_word)
                    if ant_phon:
                        ant_core = self._word_core_area(antecedent_word)
                        project(
                            self.brain, ant_phon, ant_core,
                            rounds=self.rounds,
                        )
                        for _ in range(self.rounds):
                            self.brain.project(
                                {},
                                {ant_core: [DEP_CLAUSE],
                                 DEP_CLAUSE: [DEP_CLAUSE]},
                            )

                # Reset core area connections for clean inner clause
                for core in CORE_AREAS:
                    self.brain.reset_area_connections(core)

                # Build fresh circuit for inner clause
                circuit = self._build_circuit()

                prev_category = cat
                i += 1
                continue

            # --- Check for clause exit ---
            # Exit when: comma, or inner clause verb already seen and
            # current word is a verb (main clause verb), or end of sentence.
            is_clause_exit = False
            if in_clause:
                if word == ",":
                    is_clause_exit = True
                elif inner_verb_seen and cat == "VERB":
                    # Inner clause had its verb; this verb belongs to main
                    is_clause_exit = True
                elif i == len(words) - 1 and inner_verb_seen:
                    # End of sentence; last word goes to main clause
                    is_clause_exit = True

            if is_clause_exit:
                # Snapshot DEP_CLAUSE
                result["dep_clause_assembly"] = _snap(
                    self.brain, DEP_CLAUSE)
                result["clauses"]["embedded"] = list(inner_words)

                # Replay outer words without plasticity to restore context
                circuit = self._build_circuit()
                self._replay_without_plasticity(outer_words, circuit)

                in_clause = False
                verb_seen = any(
                    result["categories"].get(w) == "VERB"
                    for w in outer_words
                )
                noun_count = sum(
                    1 for w in outer_words
                    if result["categories"].get(w) in ("NOUN", "PRON")
                )

                if word == ",":
                    prev_category = cat
                    i += 1
                    continue
                # Otherwise fall through to process this word as main clause

            # --- Normal word processing ---
            if in_clause:
                inner_words.append(word)
                if cat == "VERB":
                    inner_verb_seen = True
            else:
                main_words.append(word)

            # Activate word assembly
            phon = self.stim_map.get(word)
            core_area = self._word_core_area(word)
            if phon is not None:
                project(self.brain, phon, core_area, rounds=self.rounds)

            # Apply gating rules and project
            self._apply_pre_rules(circuit, cat, verb_seen, noun_count)
            for _ in range(self.rounds):
                circuit.step()

            # Build context
            circuit.disinhibit(core_area, CONTEXT)
            circuit.disinhibit(CONTEXT, CONTEXT)
            for _ in range(self.rounds):
                circuit.step()
            circuit.inhibit(core_area, CONTEXT)

            self._apply_post_rules(circuit, cat, verb_seen, noun_count)

            if cat == "VERB":
                verb_seen = True
            if cat in ("NOUN", "PRON"):
                noun_count += 1

            prev_category = cat
            i += 1

        result["clauses"]["main"] = main_words

        # Assign roles for main clause words
        result["roles"], result["role_diagnostics"] = (
            self.parse_roles_by_reconstruction(main_words))

        # Assign roles for inner clause with filler-gap binding:
        # The antecedent noun is the "filler" — it was displaced from its
        # canonical position in the inner clause.  For SRCs the antecedent
        # is the agent; for ORCs it is the patient.
        if inner_words:
            inner_cats = {w: result["categories"][w]
                          for w in inner_words
                          if w in result["categories"]}

            # Determine filler role from inner clause structure:
            # - ORC active: "dog that THE CAT chased" → pre-verb noun → PATIENT
            # - SRC active: "dog that chased THE CAT" → no pre-verb noun → AGENT
            # - Passive RC:  "dog that was chased by ..." → passive → PATIENT
            filler_w = antecedent_word
            filler_r = None
            if filler_w:
                inner_is_passive = self._detect_passive(
                    inner_words, inner_cats)
                inner_has_pre_verb_noun = False
                for w in inner_words:
                    ic = inner_cats.get(w)
                    if ic == "VERB":
                        break
                    if ic in ("NOUN", "PRON"):
                        inner_has_pre_verb_noun = True
                        break
                if inner_is_passive or inner_has_pre_verb_noun:
                    filler_r = "PATIENT"
                else:
                    filler_r = "AGENT"

            inner_roles, inner_diag = self.parse_roles_by_reconstruction(
                inner_words,
                filler_word=filler_w,
                filler_role=filler_r,
            )
            # Store full inner clause roles (including filler) separately.
            # The filler has DUAL roles: e.g. AGENT in the main clause
            # ("the dog ... sees the cat") and PATIENT in the inner
            # clause ("that the bird chases").  The flat ``roles`` dict
            # keeps the main-clause role for the filler; consumers that
            # need the inner-clause role can read ``inner_roles``.
            result["inner_roles"] = dict(inner_roles)
            result["inner_role_diagnostics"] = inner_diag
            for word, role in inner_roles.items():
                if word == filler_w:
                    continue  # keep main-clause role for filler
                result["roles"][word] = role

        result["phrases"] = self._identify_phrases(
            main_words + inner_words, result["categories"])

        # Detect tense, mood, polarity from full word list
        result["tense"] = self.detect_tense(words)
        result["mood"] = self.detect_mood(words)
        result["polarity"] = self.detect_polarity(words)

        return result

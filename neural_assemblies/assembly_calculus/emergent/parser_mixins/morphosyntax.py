"""Morphosyntax mixin: tense, mood, polarity, number detection/training and
conjunction handling.

WHAT THESE FEATURES ARE, ARCHITECTURALLY.  Tense, mood, polarity and number
each get their own brain area rather than being properties of the verb's
assembly.  That is the architectural claim: they are values held CONCURRENTLY
with the clause, not modifications of it -- a sentence is simultaneously about
a particular event and in the past and negated, and separate areas is what
lets all three be active at once.  The same choice is why they can be read out
independently and why a mismatch between them is detectable.

THE TWO HALVES, AND WHICH IS WHICH.  Every feature here comes in a
``detect_*`` / ``train_*`` pair, and the division of labour is important to
state plainly:

    detect_*  Python.  Word-list lookup against the frozensets below --
              ``_NEGATION_WORDS``, ``_FUTURE_MARKERS``, ``_PERFECT_AUX``, and
              so on.  Hand-authored, English-specific, and not learned.
    train_*   Neural.  Uses the detected label to name a STIMULUS, then
              co-fires that stimulus with the verb's assembly so the feature
              area learns which verbs occur under which value.

So ``detect_tense`` is a teacher signal, not a result.  The learned content is
the association between verb assemblies and feature assemblies, and that is
what should be evaluated; the detectors are scaffolding standing in for a
morphological analyser the model does not have.  Applying this mixin to
another language requires replacing every frozenset in this file.

Note also that the detectors run over SURFACE tokens, so they see "will" and
"not" but nothing about affixes -- a language marking tense inflectionally
would register as PRESENT throughout.
"""

from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

from neural_assemblies.assembly_calculus.assembly import (
    overlap as assembly_overlap,
)
from neural_assemblies.assembly_calculus.ops import _snap
from ..core.areas import (
    VERB_CORE, CONJ_CORE, TENSE, MOOD, POLARITY, NUMBER,
    GROUNDING_TO_CORE,
)


class MorphosyntaxMixin:
    """Tense, mood, polarity detection and training; conjunction handling."""

    _NEGATION_WORDS = frozenset([
        "not", "n't", "no", "never", "neither", "nor", "nobody",
        "nothing", "nowhere", "don't", "doesn't", "didn't",
        "won't", "wouldn't", "can't", "couldn't", "shouldn't",
    ])

    _FUTURE_MARKERS = frozenset(["will", "shall", "gonna"])
    _PROGRESSIVE_AUX = frozenset(["is", "am", "are", "was", "were"])
    _PERFECT_AUX = frozenset(["has", "have", "had"])
    _WH_WORDS = frozenset([
        "who", "what", "where", "when", "why", "how", "which",
    ])
    _AUX_WORDS = frozenset([
        "do", "does", "did", "is", "am", "are", "was", "were",
        "has", "have", "had", "will", "shall", "can", "could",
        "would", "should", "may", "might", "must",
    ])
    _CONJUNCTION_WORDS = frozenset([
        "and", "but", "or", "nor", "so", "yet", "for",
        "because", "although", "if", "when", "while", "since",
    ])

    def detect_tense(self, words: List[str]) -> str:
        """Detect tense from verb morphology and auxiliaries.

        Args:
            words: List of word tokens.

        Returns:
            One of: "PRESENT", "PAST", "FUTURE", "PROGRESSIVE", "PERFECT".
        """
        from ..vocabulary_builder import lookup_verb_form

        for i, word in enumerate(words):
            # Check for future markers
            if word.lower() in self._FUTURE_MARKERS:
                return "FUTURE"

            # Check for progressive: aux + -ing form
            if (word.lower() in self._PROGRESSIVE_AUX and
                    i + 1 < len(words)):
                result = lookup_verb_form(words[i + 1])
                if result and result[1] == "PROGRESSIVE":
                    return "PROGRESSIVE"

            # Check for perfect: has/have + past participle
            if (word.lower() in self._PERFECT_AUX and
                    i + 1 < len(words)):
                result = lookup_verb_form(words[i + 1])
                if result and result[1] == "PERFECT":
                    return "PERFECT"

            # Check verb forms
            result = lookup_verb_form(word)
            if result:
                _, tense = result
                if tense == "PAST":
                    return "PAST"

        return "PRESENT"

    def detect_mood(self, words: List[str]) -> str:
        """Detect sentence mood from surface cues.

        Args:
            words: List of word tokens.

        Returns:
            One of: "DECLARATIVE", "INTERROGATIVE", "IMPERATIVE".
        """
        if not words:
            return "DECLARATIVE"

        first = words[0].lower()

        # Question: starts with wh-word or auxiliary
        if first in self._WH_WORDS:
            return "INTERROGATIVE"
        if first in self._AUX_WORDS:
            return "INTERROGATIVE"

        # Imperative: starts with a verb (no subject)
        grounding = self.word_grounding.get(first)
        if grounding and grounding.dominant_modality == "motor":
            return "IMPERATIVE"

        return "DECLARATIVE"

    def detect_polarity(self, words: List[str]) -> str:
        """Detect sentence polarity from negation markers.

        Args:
            words: List of word tokens.

        Returns:
            One of: "AFFIRMATIVE", "NEGATIVE".
        """
        for word in words:
            if word.lower() in self._NEGATION_WORDS:
                return "NEGATIVE"
        return "AFFIRMATIVE"

    def _novelty_gain(self, feature: str, word: str) -> float:
        """Surprise gain for this episode, from the learner's OWN history.

        E2 (task #131): rare events write BIGGER updates -- the
        neuromodulated-plasticity (ACh/NE) analog, and the complement of
        homeostatic scaling, which removes accumulated MASS but cannot
        strengthen an association that was only ever written twice.

        The registered form: after counting this exposure,

            gain = min(novelty_gain_max, (mean_count / count) ** exp)

        Self-normalizing (a uniform corpus gives ~1 everywhere) and
        LABEL-FREE: the counts key on (feature area, surface form), never
        on a linguistic category -- a gain keyed to "is plural" would be
        PLURAL_EVERY in disguise. `novelty_gain_max` <= 1 disables
        (default), preserving exact prior behavior.

        THE EXPONENT IS THE LEVER (E3, #132): at exp=0.5 (E2's original
        sqrt) the raw gain NEVER exceeded 1.215 on this corpus (mean
        exposure 1.41), so every cap >= 1.2 was the same experiment and
        the mechanism acted mostly as mild familiarity suppression.
        exp=1.0 (linear) restores GAIN_MAX to a live parameter.
        """
        gmax = getattr(self, "novelty_gain_max", 1.0)
        counts = self._morph_exposure
        key = f"{feature}:{word}"
        counts[key] = counts.get(key, 0) + 1
        if gmax <= 1.0:
            return 1.0
        prefix = feature + ":"
        feature_counts = [c for k, c in counts.items()
                          if k.startswith(prefix)]
        mean_count = sum(feature_counts) / len(feature_counts)
        exp = getattr(self, "novelty_gain_exp", 0.5)
        return min(gmax, (mean_count / counts[key]) ** exp)

    @contextmanager
    def _gain_on_fiber(self, target: str, source: str, gain: float):
        """Transiently multiply one fiber's beta through the engine's own
        set_beta/get_beta -- the authoritative per-fiber store (the #88
        lesson: writing any OTHER beta bookkeeping is a silent no-op)."""
        if gain == 1.0:
            yield
            return
        eng = self.brain._engine
        base = eng.get_beta(target, source)
        eng.set_beta(target, source, base * gain)
        try:
            yield
        finally:
            eng.set_beta(target, source, base)

    # ------------------------------------------------------------------
    # PER-VALUE FEATURE AREAS (E15, #144)
    # ------------------------------------------------------------------

    def _feature_target_area(self, feature: str, label: str) -> str:
        """Where this (feature, label) episode trains.

        Default: the shared feature area (TENSE/NUMBER) -- byte-identical
        prior behavior. Under `split_feature_areas`: the per-value area
        (NUMBER_SG, TENSE_PAST, ...). E14 (#143) measured the shared-area
        design's label images MERGING as total label projections grow
        (shared SG-PL cols 2.6 -> 17.8 of 30 at 200 frames, through VARIED
        sentences); one area per value makes that structurally impossible.
        """
        if not getattr(self, "split_feature_areas", False):
            return feature
        from ..core.areas import feature_value_area
        return feature_value_area(feature, label)

    def _ensure_value_areas(self, feature: str) -> List[str]:
        """Create `feature`'s per-value areas + their MI group, idempotently.

        LAZY BY DESIGN: called from train_tense/train_number when the split
        is enabled, never at construction, so the flag can be flipped on a
        parser restored from a pre-stage checkpoint. The mutual-inhibition
        group is the paper's device verbatim -- "firing only in the area
        that receives the greatest total synaptic input" (Mitropolsky &
        Papadimitriou 2025, sec. 2; their ROLE triple) -- and it stays
        silent during training because teacher-forced episodes target ONE
        value area per project() call; it fires exactly when recall
        co-targets the group.
        """
        from ..core.areas import FEATURE_VALUE_LABELS, feature_value_area

        names = [feature_value_area(feature, lab)
                 for lab in FEATURE_VALUE_LABELS[feature]]
        for name in names:
            if name not in self.brain.areas:
                self.brain.add_area(name, self.n, self.k, self.beta)
        groups = getattr(self.brain, "_mutual_inhibition_groups", [])
        if not any(set(g) == set(names) for g in groups):
            self.brain.add_mutual_inhibition(names)
        return names

    def train_tense(self, sentences: List[List[str]]) -> None:
        """Train TENSE area from verb morphology in sentences.

        Creates tense stimuli and projects verb+tense → TENSE area.

        Args:
            sentences: List of token lists.
        """
        # Training invalidates the recall readout's label-image cache.
        self._feature_image_cache.clear()
        if getattr(self, "split_feature_areas", False):
            self._ensure_value_areas(TENSE)
        # Register tense stimuli
        tense_stims = {}
        for tense_name in ("PRESENT", "PAST", "FUTURE",
                           "PROGRESSIVE", "PERFECT"):
            stim_name = f"tense_{tense_name}"
            if stim_name not in self.brain.stimuli:
                self.brain.add_stimulus(stim_name, self.k)
            tense_stims[tense_name] = stim_name

        # E8 (#137): epochs raise per-form exposure at fixed diversity --
        # the one quantity uniform corpora pin (~1.5) at every size (E7).
        for _epoch in range(max(1, getattr(self, "morph_repetitions", 1))):
          for sent in sentences:
            tense = self.detect_tense(sent)
            tense_stim = tense_stims[tense]
            # Shared area by default; the per-value area under the split
            # (E15). Teacher-forced: exactly ONE area is targeted, so the
            # MI group stays silent during training.
            tgt = self._feature_target_area(TENSE, tense)

            # Find the verb in this sentence using grounding (fast path)
            for word in sent:
                if word not in self.stim_map:
                    continue
                grounding = self.word_grounding.get(word)
                if grounding and grounding.dominant_modality == "motor":
                    phon = self.stim_map[word]
                    # Project tense + verb → TENSE area. The gain brackets
                    # the AFFERENT fiber only -- the one recall probes.
                    gain = self._novelty_gain("TENSE", word)
                    with self._gain_on_fiber(tgt, VERB_CORE, gain):
                        self.brain.project(
                            {tense_stim: [tgt], phon: [VERB_CORE]},
                            {VERB_CORE: [tgt]},
                        )
                        if self.rounds > 1:
                            self.brain.project_rounds(
                                target=tgt,
                                areas_by_stim={tense_stim: [tgt]},
                                dst_areas_by_src_area={
                                    VERB_CORE: [tgt], tgt: [tgt],
                                },
                                rounds=self.rounds - 1,
                            )
                    break  # One tense per sentence

        # SLOW HOMEOSTASIS boundary (E9): if scaling is deferred, this
        # phase end is where the accumulated mass gets renormalized.
        eng = self.brain._engine
        if getattr(eng, "synaptic_scaling_deferred", False):
            eng.flush_synaptic_scaling()

    def train_mood(self, sentences: List[List[str]]) -> None:
        """Train MOOD area from sentence mood detection.

        Args:
            sentences: List of token lists.
        """
        mood_stims = {}
        for mood_name in ("DECLARATIVE", "INTERROGATIVE", "IMPERATIVE"):
            stim_name = f"mood_{mood_name}"
            if stim_name not in self.brain.stimuli:
                self.brain.add_stimulus(stim_name, self.k)
            mood_stims[mood_name] = stim_name

        for sent in sentences:
            mood = self.detect_mood(sent)
            mood_stim = mood_stims[mood]
            self.brain.project(
                {mood_stim: [MOOD]},
                {MOOD: [MOOD]},
            )
            if self.rounds > 1:
                self.brain.project_rounds(
                    target=MOOD,
                    areas_by_stim={mood_stim: [MOOD]},
                    dst_areas_by_src_area={MOOD: [MOOD]},
                    rounds=self.rounds - 1,
                )

    def train_polarity(self, sentences: List[List[str]]) -> None:
        """Train POLARITY area from negation detection.

        Args:
            sentences: List of token lists.
        """
        pol_stims = {}
        for pol_name in ("AFFIRMATIVE", "NEGATIVE"):
            stim_name = f"polarity_{pol_name}"
            if stim_name not in self.brain.stimuli:
                self.brain.add_stimulus(stim_name, self.k)
            pol_stims[pol_name] = stim_name

        for sent in sentences:
            polarity = self.detect_polarity(sent)
            pol_stim = pol_stims[polarity]
            self.brain.project(
                {pol_stim: [POLARITY]},
                {POLARITY: [POLARITY]},
            )
            if self.rounds > 1:
                self.brain.project_rounds(
                    target=POLARITY,
                    areas_by_stim={pol_stim: [POLARITY]},
                    dst_areas_by_src_area={POLARITY: [POLARITY]},
                    rounds=self.rounds - 1,
                )

    def train_conjunctions(self, sentences: List[List[str]]) -> None:
        """Train CONJ_CORE area with conjunction words.

        Args:
            sentences: List of token lists.
        """
        for sent in sentences:
            for word in sent:
                if word.lower() not in self._CONJUNCTION_WORDS:
                    continue
                # Register if needed
                self.register_word(word) if hasattr(
                    self, 'register_word') else None
                phon = self.stim_map.get(word)
                if phon is None:
                    continue

                # Project phon → CONJ_CORE
                self.brain._engine.reset_area_connections(CONJ_CORE)
                stim_dict = {phon: [CONJ_CORE]}
                self.brain.project(stim_dict, {})
                if self.rounds > 1:
                    self.brain.project_rounds(
                        target=CONJ_CORE,
                        areas_by_stim=stim_dict,
                        dst_areas_by_src_area={CONJ_CORE: [CONJ_CORE]},
                        rounds=self.rounds - 1,
                    )

                if CONJ_CORE not in self.core_lexicons:
                    self.core_lexicons[CONJ_CORE] = {}
                self.core_lexicons[CONJ_CORE][word] = _snap(
                    self.brain, CONJ_CORE)
                self.brain._engine.reset_area_connections(CONJ_CORE)

    def detect_number(self, word: str) -> str:
        """Detect grammatical number, lexicon forms first, then grounding.

        The grounding-feature path was the ONLY path until the variation
        measurement (task #129) found that nothing in the pipeline ever sets
        an SG/PL feature: plural surface forms inherit the LEMMA's grounding
        verbatim (`_register_surface_forms`), so this teacher read "SG" for
        every token in the corpus -- a one-class signal that made
        `train_number` unable to learn a discrimination even where it ran.
        The lexicon check mirrors `lookup_verb_form`: a token equal to a
        NOUN entry's ``forms["plural"]`` is PL, form-level and unambiguous
        for nouns. (Verb agreement is NOT decidable at the form level -- the
        bare lemma serves plural-present agreement and other slots -- so
        verbs fall through to the SG default, and the NUMBER area learns a
        noun contrast only.)

        Noun-verb homographs ("loves", "lives", "hates") are ambiguous at
        the form level: the same surface is a noun plural AND a verb 3sg.
        This teacher sees one token with no context, so it declines to call
        those PL -- corpus verb tokens vastly outnumber plural uses of these
        nouns, and 3sg agreement is semantically singular anyway.  (The old
        first-wins lexicon index instead labelled every such verb token a
        noun plural, contaminating the NUMBER teacher signal.)

        Args:
            word: A single word token.

        Returns:
            "SG" or "PL".
        """
        from ..vocabulary_builder import lookup_lexicon_entries

        candidates = lookup_lexicon_entries(word)
        noun_plural = any(
            pos == "NOUN" and word == entry.get("forms", {}).get("plural")
            for entry, pos in candidates
        )
        verb_reading = any(pos == "VERB" for _entry, pos in candidates)
        if noun_plural and not verb_reading:
            return "PL"

        grounding = self.word_grounding.get(word)
        if grounding:
            for mod in ("visual", "motor", "properties", "spatial",
                        "social", "temporal", "emotional"):
                features = getattr(grounding, mod)
                if "SG" in features:
                    return "SG"
                if "PL" in features:
                    return "PL"
        return "SG"

    def train_number(self, sentences: List[List[str]]) -> None:
        """Train NUMBER area from morphological number features.

        For each content word (noun or verb) in each sentence, detects
        its grammatical number and projects number_stim + phon into the
        NUMBER area. This creates separate SG and PL assemblies in
        NUMBER, each associated with core-area word assemblies.

        Follows the same pattern as train_tense().

        Args:
            sentences: List of token lists.
        """
        # Training invalidates the recall readout's label-image cache.
        self._feature_image_cache.clear()
        if getattr(self, "split_feature_areas", False):
            self._ensure_value_areas(NUMBER)
        number_stims = {}
        for num_name in ("SG", "PL"):
            stim_name = f"number_{num_name}"
            if stim_name not in self.brain.stimuli:
                self.brain.add_stimulus(stim_name, self.k)
            number_stims[num_name] = stim_name

        # Same epoch loop as train_tense -- see E8 note there.
        for _epoch in range(max(1, getattr(self, "morph_repetitions", 1))):
          for sent in sentences:
            for word in sent:
                if word not in self.stim_map:
                    continue
                grounding = self.word_grounding.get(word)
                if grounding is None:
                    continue

                # Only train number for content words (nouns and verbs)
                mod = grounding.dominant_modality
                if mod not in ("visual", "motor"):
                    continue

                num = self.detect_number(word)
                num_stim = number_stims[num]
                core_area = GROUNDING_TO_CORE[mod]
                phon = self.stim_map[word]
                # Shared area by default; per-value area under the split
                # (E15) -- one target per call, MI silent during training.
                tgt = self._feature_target_area(NUMBER, num)

                # Project number_stim + word phon -> NUMBER area, novelty
                # gain on the afferent fiber (see _novelty_gain).
                gain = self._novelty_gain("NUMBER", word)
                with self._gain_on_fiber(tgt, core_area, gain):
                    self.brain.project(
                        {num_stim: [tgt], phon: [core_area]},
                        {core_area: [tgt]},
                    )
                    if self.rounds > 1:
                        self.brain.project_rounds(
                            target=tgt,
                            areas_by_stim={num_stim: [tgt]},
                            dst_areas_by_src_area={
                                core_area: [tgt], tgt: [tgt],
                            },
                            rounds=self.rounds - 1,
                        )

        # SLOW HOMEOSTASIS boundary (E9): if scaling is deferred, this
        # phase end is where the accumulated mass gets renormalized.
        eng = self.brain._engine
        if getattr(eng, "synaptic_scaling_deferred", False):
            eng.flush_synaptic_scaling()

    # ------------------------------------------------------------------
    # RECALL -- the third of the detect/train pair, previously missing.
    # ------------------------------------------------------------------

    def _recall_morph_feature(
            self, word: str, feature_area: str,
            stim_by_label: Dict[str, str],
            candidates: Tuple[str, ...],
    ) -> "tuple[Optional[str], dict]":
        """Read a morph feature for `word` back OUT of its feature area.

        Same shape as `parse_roles_by_reconstruction`'s RECALL step: compare
        the word's image in the feature area (word -> core -> feature, no
        feature stimulus) against each candidate label's stimulus image, and
        answer argmax overlap -- a tie is a failure to read, not a guess.

        This is the readout `train_tense`/`train_number` never had: the
        detectors are Python teachers, the training writes core->feature
        associations, and until now NOTHING read them back, so whether the
        corpus variation bought a recallable contrast was unmeasurable.

        Candidate labels whose stimulus was never REGISTERED are skipped, but
        a registered-yet-untrained stimulus still materializes an image
        (beta=0 is not a null), so callers must restrict `candidates` to
        labels the corpus could actually have taught when scoring accuracy.

        Returns (label_or_None, diag) where diag carries per-label overlap
        `scores`, the `margin` (top - runner), and `image_separation`
        (pairwise overlap between candidate images -- near 1.0 means the
        feature area cannot answer and the readout is untrustworthy).
        """
        from neural_assemblies.assembly_calculus.ops import (
            project as _ops_project,
        )

        brain = self.brain
        diag: dict = {"scores": {}, "margin": None, "image_separation": None}

        core_area = self._word_core_area(word)
        phon = self.stim_map.get(word)
        if (core_area is None or core_area not in brain.areas
                or feature_area not in brain.areas or phon is None):
            return None, diag

        rounds = max(1, int(self.rounds))
        with brain.read_only():
            brain.areas[feature_area].unfix_assembly()

            # Label images are a property of the SUBSTRATE, not the probed
            # word -- identical for every word on an unchanged connectome,
            # and recomputing them per call was ~40% of readout time.
            # train_tense/train_number invalidate the cache; any new writer
            # into a feature area must too.
            images = {}
            for label in candidates:
                stim = stim_by_label.get(label)
                if stim is None or stim not in brain.stimuli:
                    continue
                key = (feature_area, label, rounds)
                cached = self._feature_image_cache.get(key)
                if cached is not None:
                    images[label] = cached
                    continue
                brain.inhibit_areas([feature_area])
                brain.project({stim: [feature_area]}, {})
                for _ in range(rounds - 1):
                    brain.project({stim: [feature_area]},
                                  {feature_area: [feature_area]})
                images[label] = _snap(brain, feature_area)
                self._feature_image_cache[key] = images[label]
            if len(images) < 2:
                return None, diag

            pairs = list(images.items())
            seps = [
                float(assembly_overlap(pairs[i][1], pairs[j][1]))
                for i in range(len(pairs)) for j in range(i + 1, len(pairs))
            ]
            diag["image_separation"] = max(seps)

            # The probe: word -> core (settle), then core -> feature with NO
            # feature stimulus -- exactly what the trained pathway can supply
            # on its own.
            brain.inhibit_areas([feature_area])
            _ops_project(brain, phon, core_area, rounds=rounds)
            brain.areas[core_area].fix_assembly()
            try:
                brain.project({}, {core_area: [feature_area]})
                for _ in range(rounds - 1):
                    brain.project({}, {core_area: [feature_area],
                                       feature_area: [feature_area]})
            finally:
                brain.areas[core_area].unfix_assembly()
            probe = _snap(brain, feature_area)

        diag["scores"] = {
            label: float(assembly_overlap(image, probe))
            for label, image in images.items()
        }
        ranked = sorted(diag["scores"].items(), key=lambda kv: -kv[1])
        top_label, top = ranked[0]
        runner = ranked[1][1] if len(ranked) > 1 else 0.0
        diag["margin"] = top - runner
        if top > runner:
            return top_label, diag
        return None, diag

    def _recall_morph_feature_split(
            self, word: str, feature: str,
            stim_by_label: Dict[str, str],
            candidates: Tuple[str, ...],
    ) -> "tuple[Optional[str], dict]":
        """Per-value-area recall: the paper's mutual inhibition DECIDES.

        The competition is one project() co-targeting every candidate's
        value area from the word's settled core assembly -- exactly the
        condition under which `add_mutual_inhibition` fires, and exactly
        the paper's readout: "there is firing only in the area that
        receives the greatest total synaptic input" (Mitropolsky &
        Papadimitriou 2025). The answer is the winning AREA's label;
        `diag["scores"]` carries each area's total drive (the quantity the
        competition was decided on, exposed via last_activation_scores).

        Label-image merging is structurally impossible here -- the images
        live in different areas -- so `image_separation` is reported as
        0.0 by construction, and E15's experiment must VERIFY that with
        its own counter rather than trust this line.

        A SECONDARY readout rides along in `diag["overlap_scores"]` /
        `diag["overlap_answer"]`: each area probed alone (no co-target, MI
        silent), scored by overlap against its own label image -- the
        E-series shared-area readout translated per-area. Registered as a
        diagnostic, not the answer; #24 measured cross-area drive
        comparison as the less reliable primitive in the ROLE setting, so
        if the two readouts disagree systematically that is a finding to
        report, not to average.
        """
        from neural_assemblies.assembly_calculus.ops import (
            project as _ops_project,
        )
        from ..core.areas import feature_value_area

        brain = self.brain
        diag: dict = {"scores": {}, "margin": None,
                      "image_separation": None, "readout": "mi_split",
                      "overlap_scores": {}, "overlap_answer": None}

        core_area = self._word_core_area(word)
        phon = self.stim_map.get(word)
        if (core_area is None or core_area not in brain.areas
                or phon is None):
            return None, diag

        cand_areas: Dict[str, str] = {}
        for label in candidates:
            stim = stim_by_label.get(label)
            area = feature_value_area(feature, label)
            if (stim is not None and stim in brain.stimuli
                    and area in brain.areas):
                cand_areas[label] = area
        if len(cand_areas) < 2:
            return None, diag

        rounds = max(1, int(self.rounds))
        with brain.read_only():
            # Label images, one per AREA (cache keyed by area name, so
            # split and shared keys never collide; train_* invalidates).
            images = {}
            for label, area in cand_areas.items():
                key = (area, label, rounds)
                cached = self._feature_image_cache.get(key)
                if cached is None:
                    stim = stim_by_label[label]
                    brain.inhibit_areas([area])
                    brain.project({stim: [area]}, {})
                    for _ in range(rounds - 1):
                        brain.project({stim: [area]}, {area: [area]})
                    cached = _snap(brain, area)
                    self._feature_image_cache[key] = cached
                images[label] = cached
            diag["image_separation"] = 0.0  # structural; see docstring

            # Settle the word's core assembly (the recall probe's own
            # activation, unchanged from the shared-area readout).
            brain.inhibit_areas(list(cand_areas.values()))
            _ops_project(brain, phon, core_area, rounds=rounds)
            brain.areas[core_area].fix_assembly()
            try:
                # THE COMPETITION: co-target the MI group in ONE call.
                brain.project(
                    {}, {core_area: list(cand_areas.values())})
            finally:
                brain.areas[core_area].unfix_assembly()
            drive = dict(getattr(brain, "last_activation_scores", {}) or {})
            diag["mi_survivors"] = [
                label for label, area in cand_areas.items()
                if len(getattr(brain.areas[area], "winners", ())) > 0]

            # Secondary: per-area solo probe, overlap vs the label image.
            for label, area in cand_areas.items():
                brain.inhibit_areas([area])
                brain.areas[core_area].fix_assembly()
                try:
                    brain.project({}, {core_area: [area]})
                    for _ in range(rounds - 1):
                        brain.project({}, {core_area: [area],
                                           area: [area]})
                finally:
                    brain.areas[core_area].unfix_assembly()
                probe = _snap(brain, area)
                diag["overlap_scores"][label] = float(
                    assembly_overlap(images[label], probe))

        diag["scores"] = {label: float(drive.get(area, 0.0))
                          for label, area in cand_areas.items()}
        ranked = sorted(diag["scores"].items(), key=lambda kv: -kv[1])
        top_label, top = ranked[0]
        runner = ranked[1][1] if len(ranked) > 1 else 0.0
        diag["margin"] = ((top - runner) / top) if top > 0 else None
        ov = sorted(diag["overlap_scores"].items(), key=lambda kv: -kv[1])
        if ov and (len(ov) < 2 or ov[0][1] > ov[1][1]):
            diag["overlap_answer"] = ov[0][0]
        if top > runner:
            return top_label, diag
        return None, diag

    def recall_tense(self, word: str,
                     candidates: Tuple[str, ...] = ("PRESENT", "PAST"),
                     ) -> "tuple[Optional[str], dict]":
        """Recall the tense associated with a verb FORM from the TENSE area.

        Form-level by construction: `train_tense` co-fires the verb form's
        assembly with the detected tense stimulus, so what is recallable is
        the form->tense association ("chased" -> PAST), not anything about
        the sentence. The default candidates are the two labels the generated
        corpus teaches; FUTURE/PROGRESSIVE/PERFECT stimuli exist but are
        untrained there (see `_recall_morph_feature` on why untrained labels
        must not be scored).
        """
        stims = {t: f"tense_{t}" for t in
                 ("PRESENT", "PAST", "FUTURE", "PROGRESSIVE", "PERFECT")}
        if getattr(self, "split_feature_areas", False):
            return self._recall_morph_feature_split(
                word, TENSE, stims, candidates)
        return self._recall_morph_feature(word, TENSE, stims, candidates)

    def recall_number(self, word: str,
                      candidates: Tuple[str, ...] = ("SG", "PL"),
                      ) -> "tuple[Optional[str], dict]":
        """Recall grammatical number for a noun form from the NUMBER area."""
        stims = {"SG": "number_SG", "PL": "number_PL"}
        if getattr(self, "split_feature_areas", False):
            return self._recall_morph_feature_split(
                word, NUMBER, stims, candidates)
        return self._recall_morph_feature(word, NUMBER, stims, candidates)

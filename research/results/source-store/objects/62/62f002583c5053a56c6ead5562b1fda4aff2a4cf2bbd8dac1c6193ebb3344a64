"""Where training sentences come from -- the corpus-generation concern.

Split out of `CurriculumTrainer` (task #109 / the literate-architecture
directive): the trainer ORCHESTRATES stages; this module decides what
sentences exist. The two grew together until the file needed an "AND" to
describe -- stage scheduling AND realism pools AND passive emission AND
surface registration -- which is the signal to split.

One `SentenceGenerator` owns the whole path from stage vocabulary to
`SentencePlan`s:

  * `cds_corpus_sentences` -- transcribed child-directed lines filtered to
    the stage vocabulary. These carry NO `SceneEvent`, deliberately: nobody
    recorded the scene, and inventing one would fabricate the supervision
    the scene-derived-role path exists to avoid.
  * `generate_generic` -- the deterministic frame generator: realism pools
    licensed by the lexicon (tense/number/determiners/pronouns/adverbs,
    e909c61), selectional restrictions, ditransitive and locative PPs, and
    the passive alternation. Every full clause carries the `SceneEvent` it
    describes, because who-acted-on-whom is not recoverable from the
    string.
  * `generate` -- the stage-facing blend of the two (plus holdout-bridge
    lines), the ONE entry point trainer and tests call.
  * `register_surface_forms` -- inflected surfaces enter `stim_map` sharing
    the lemma's grounding; without this, every inflected token is silently
    dropped from role training.

The rate constants (PASSIVE_EVERY, DITRANSITIVE_EVERY, PAST_RATE,
PLURAL_RATE) live HERE and nowhere else -- experiments that pin an axis
patch this module. They are stated starvation counters, not tuned numbers,
and the standing goal is to RETIRE them in favor of learning-rule
mechanisms (#130 measured scaling; #131 is surprise-gain): a per-phenomenon
corpus knob is exactly what does not scale to a Zipfian corpus.
"""

from typing import Dict, List, Optional, Set

from ..core.scene import SceneEvent
from ..core.sentence import SentencePlan


#: Emit a passive for one in every N eligible transitive clauses. 0 disables.
#:
#: The 0 case is not decoration: an A/B on "corpus with vs without passives"
#: needs an off switch, and the obvious one -- setting N enormous -- does NOT
#: work, because `n_eligible % N == 0` is TRUE at n_eligible == 0 and the first
#: clause still emits. That contaminated the control arm with exactly one
#: passive, which the experiment's own sanity check caught before it scored
#: anything.
#: English runs roughly 2-10% passive. This is deliberately higher: the gating
#: learner is CONTRASTIVE (it compares role order with the marker present
#: against absent), and with only ~50 generated sentences per stage a 5% rate
#: gives it two or three examples to generalise from. Stated rather than tuned;
#: raising it toward 1 would make the corpus passive-dominant and teach the
#: determiner to reverse roles, which is the failure `_learn_gating_patterns`
#: is written to avoid.
PASSIVE_EVERY = 4

#: Force a DITRANSITIVE verb draw for one in every N frames (0 disables).
#: Ditransitives are 8 of ~119 frame verbs, so natural sampling yields ~4
#: draws per 50-frame stage -- too thin for ROLE_GOAL to accumulate bindings,
#: the same starvation argument that set PASSIVE_EVERY. Stated, not tuned.
DITRANSITIVE_EVERY = 6

#: Probability that a complexity>=3 frame is realized in the PAST tense
#: (0.0 disables -- the one-class corpus the variation census flagged).
#: Module-level so an A/B can pin one axis while leaving the others at
#: default; the passive off-switch lesson applies (an "enormous rate"
#: non-value is not an off switch, an explicit 0.0 is).
PAST_RATE = 0.30

#: Probability that a subject with a plural form is realized PLURAL
#: (0.0 disables). Same contract as PAST_RATE.
PLURAL_RATE = 0.30

#: How frame SUBJECTS are drawn from the stage pool.
#:   "uniform"   with-replacement (the measured production corpus).
#:   "coverage"  least-used-first with random tie-break: every eligible
#:               noun surfaces as a subject before any repeats.
#:               Uniform-with-replacement under-covers the vocabulary vs
#:               natural long-tail text, and E5 (#134) localized the
#:               number-recall ceiling to the resulting thin PL image.
#:               E6 (#135) measured it: WORSE -- diversity without data
#:               is dilution (exposure and coverage are conjugate at
#:               fixed budget).
#:   "zipf"      rank-frequency weights 1/rank^ZIPF_EXPONENT over the
#:               stage noun order (the rank assignment is IMPOSED and
#:               arbitrary; what is Zipfian is the distribution, which is
#:               the natural-text shape uniform sampling lacks). E12
#:               (#141) showed per-item afferent mass decides number
#:               recall and that phase REPETITION self-defeats by merging
#:               the label images; Zipf is the REDISTRIBUTION lever --
#:               per-form exposure rises on head forms at CONSTANT total
#:               episode and label-stimulus budget.
#: THE VERDICT IS NOW IN (#149). E13 (#142): zipf at the default 50-frame
#: budget is a NULL -- redistribution needs a budget worth redistributing
#: (~12 total PL episodes cannot be reallocated into reliability). E14
#: (#143): the gain is the zipf x budget INTERACTION. E15-E19b: every
#: terminal number of the arc (0.700 thrice-replicated, 0.727 +/- 0.037)
#: lives on zipf-200. So the DEFAULT STAYS "uniform" *because the default
#: budget is 50*: at that budget zipf was measured to buy nothing, and
#: the uniform-50 corpus is the substrate of every non-morph arc's
#: standing measurement. The production configuration adopts zipf JOINTLY
#: with FRAMES_PER_STAGE=200 -- see
#: research/notes/language/production_configuration.md. The real resolution is the
#: CHILDES corpus (#30/#150), whose statistics are Zipfian without this
#: knob existing at all. Experiments patch this module attribute in
#: their workers.
SUBJECT_SAMPLING = "uniform"

#: Zipf exponent s for SUBJECT_SAMPLING="zipf" (weights 1/rank^s).
#: 1.0 is the classic rank-frequency law; stated, not tuned.
ZIPF_EXPONENT = 1.0

#: PLURAL_RATE for OBJECT NPs (0.0 disables -- the measured production
#: corpus; objects were the last always-singular slot). E6's diverse arm
#: sets this to PLURAL_RATE. Same off-switch contract as PASSIVE_EVERY.
OBJECT_PLURAL_RATE = 0.0

#: Frame budget per complexity>=3 stage. THE constant every experiment
#: through E6 held fixed, and E6 (#135) proved it is the binding one:
#: form diversity and per-form exposure are CONJUGATE at fixed budget
#: (the diverse corpus scored WORSE -- more forms, fewer exposures each),
#: so only a larger budget can raise both. Also the least child-like
#: number in the pipeline: fifty utterances is not a developmental stage.
#: 50 reproduces the measured production corpus byte-for-byte; E7 (#136)
#: sweeps it.
FRAMES_PER_STAGE = 50


class SentenceGenerator:
    """Generates stage corpora for a parser; composed by `CurriculumTrainer`.

    Holds no training state: `parser` supplies grounding and `stim_map`,
    `holdout_words` (shared with the trainer) gates bridge lines. Everything
    else is derived per call from the stage word list.
    """

    def __init__(self, parser, holdout_words: Optional[Set[str]] = None):
        self.parser = parser
        self.holdout_words: Set[str] = (
            holdout_words if holdout_words is not None else set())
        #: The passive role marker ("by") found in the CURRENT stage's own
        #: words -- set by `generate_generic`, read by `passive_of`. A stage
        #: that has not met "by" simply generates no passives.
        self._by_marker: Optional[str] = None

    def cds_corpus_sentences(
        self,
        stage_name: str,
        words: list,
        complexity: int,
    ) -> Optional[List[List[str]]]:
        """CDS-style corpus lines filtered to stage vocabulary."""
        from ..training.perf import developmental_curriculum_enabled

        if not developmental_curriculum_enabled():
            return None

        lemmas = {w.lemma for w in words}
        known = lemmas | set(self.parser.stim_map.keys())

        if stage_name == "FIRST_WORDS":
            from neural_assemblies.lexicon.curriculum.stage1_first_words import (
                STAGE1_CORPUS,
            )
            corpus = STAGE1_CORPUS
            max_len = 2 if complexity <= 1 else 3
        elif stage_name == "VOCABULARY_SPURT":
            from neural_assemblies.lexicon.curriculum.stage1_first_words import (
                STAGE1_CORPUS,
            )
            from neural_assemblies.lexicon.curriculum.stage2_vocabulary_spurt import (
                STAGE2_CORPUS,
            )
            corpus = list(STAGE1_CORPUS) + list(STAGE2_CORPUS)
            max_len = 3
        elif stage_name == "TWO_WORD":
            from neural_assemblies.lexicon.curriculum.stage2_vocabulary_spurt import (
                STAGE2_CORPUS,
            )
            from neural_assemblies.lexicon.curriculum.stage3_two_word import (
                STAGE3_CORPUS,
            )
            corpus = list(STAGE2_CORPUS) + list(STAGE3_CORPUS)
            max_len = 4
        elif stage_name in ("SENTENCES", "COMPLEX_GRAMMAR"):
            from neural_assemblies.lexicon.curriculum.stage4_sentences import (
                STAGE4_CORPUS,
            )
            if complexity >= 4:
                from neural_assemblies.lexicon.curriculum.stage3_two_word import (
                    STAGE3_CORPUS,
                )
                corpus = list(STAGE4_CORPUS) + list(STAGE3_CORPUS)
            else:
                from neural_assemblies.lexicon.curriculum.stage3_two_word import (
                    STAGE3_CORPUS,
                )
                corpus = list(STAGE3_CORPUS)
            max_len = 8 if complexity >= 6 else 6
        else:
            return None

        sentences: List[List[str]] = []
        for line in corpus:
            tokens = line.split()
            if not tokens or len(tokens) > max_len:
                continue
            if not all(t in known for t in tokens):
                continue
            sentences.append(tokens)

        if not sentences:
            return None

        if complexity == 1:
            singles = [s for s in sentences if len(s) == 1]
            if singles:
                return singles + [s for s in sentences if len(s) > 1][:12]
        return sentences

    def scene_features(self, lemma: str) -> List[str]:
        """The perceptual features of a word, as `roles_from_scene` reads them.

        Deliberately the SAME accessor the derivation uses, so a participant
        bundle cannot be built from one view of a word's grounding and matched
        against another. A word with no grounding yields no features, and
        `SceneEvent.role_of_features` then declines to give it a role.
        """
        from ..core.scene import _word_features

        ctx = self.parser.word_grounding.get(lemma)
        return list(_word_features(ctx)) if ctx is not None else []

    def generate_generic(
        self,
        words: list,
        complexity: int,
    ) -> List[SentencePlan]:
        """Deterministic pattern generator (no CDS corpora).

        Returns SENTENCE PLANS, not token lists. Every full clause carries the
        `SceneEvent` it describes, because who-acted-on-whom is not recoverable
        from the string: a passive states the same event in the opposite order,
        and a positional reading of it is inverted, not merely uninformative.
        See `core/sentence.py::SentencePlan`.
        """
        from neural_assemblies.lexicon.lexicon_manager import WordCategory
        import random as _rng

        nouns = [w for w in words if w.category == WordCategory.NOUN]
        verbs = [w for w in words if w.category == WordCategory.VERB]
        adjs = [w for w in words if w.category == WordCategory.ADJECTIVE]
        dets = [w for w in words if w.category == WordCategory.DETERMINER]
        preps = [w for w in words if w.category == WordCategory.PREPOSITION]
        advs = [w for w in words if w.category == WordCategory.ADVERB]
        prons = [w for w in words if w.category == WordCategory.PRONOUN]

        sentences: List[SentencePlan] = []

        # Fragments, not clauses: a bare noun or a determiner phrase describes
        # no event, so there is nothing to attach and no role to derive.
        if complexity == 1:
            for n in nouns[:20]:
                sentences.append(SentencePlan([n.lemma]))
            for v in verbs[:10]:
                sentences.append(SentencePlan([v.lemma]))
            return sentences

        if complexity == 2:
            for n in nouns[:15]:
                if dets:
                    sentences.append(SentencePlan([dets[0].lemma, n.lemma]))
                for v in verbs[:5]:
                    sentences.append(SentencePlan([n.lemma, v.lemma]))
            return sentences

        _rng_state = _rng.getstate()
        _rng.seed(42)
        det_word = dets[0].lemma if dets else "the"

        # GRAMMATICALITY. The previous version emitted
        # `[det, random_noun, verb.LEMMA, det, random_noun]` unconditionally,
        # which produced strings like "the store build the dog": no
        # subject-verb agreement, an object forced onto intransitive verbs, and
        # subjects drawn uniformly over all nouns including abstract ones.
        #
        # That corpus is what role induction learns AGENT/PATIENT from, so the
        # bindings were partly noise -- `dog` got its ONLY role binding from
        # that single sentence, as its object, which is why the curriculum
        # golden asks for dog=AGENT and can never get it.
        #
        # The lexicon already carries everything needed to do this properly:
        # `forms["3sg"]`, `features` (transitive / intransitive /
        # ambitransitive / stative / copula, and animate / abstract on nouns),
        # and `arguments` (the thematic frame). None of it was being read.
        def _feat(word) -> dict:
            return getattr(word, "features", None) or {}

        def _form(word, name: str) -> Optional[str]:
            forms = getattr(word, "forms", None)
            if not isinstance(forms, dict):
                return None
            value = forms.get(name)
            return value if isinstance(value, str) else None

        def _lemma(word) -> str:
            value = getattr(word, "lemma", None)
            if not isinstance(value, str) or not value:
                raise TypeError("lexicon entries must provide a nonempty lemma")
            return value

        def _finite(verb) -> str:
            """3sg present, because every generated subject is `the <noun>`."""
            return (getattr(verb, "forms", None) or {}).get("3sg") or verb.lemma

        def _first_arg(verb) -> str:
            args = getattr(verb, "arguments", None) or []
            return args[0] if args else ""

        def _takes_object(verb) -> bool:
            f = _feat(verb)
            if f.get("transitive") or f.get("ditransitive"):
                return True
            if f.get("intransitive"):
                return False
            if f.get("ambitransitive"):
                return bool(_rng.getrandbits(1))
            # Unannotated: assume intransitive rather than invent an argument.
            return False

        locatives = [
            p for p in preps
            if _feat(p).get("spatial")
            and not _feat(p).get("motion")
            and not _feat(p).get("goal")
            and not _feat(p).get("source")
        ]
        animate = [n for n in nouns if _feat(n).get("animate")]
        concrete = [n for n in nouns if not _feat(n).get("abstract")] or nouns
        # A copula needs a predicate ("the dog is big"), which this frame does
        # not build, so `be` would only ever yield "the dog is". Excluded rather
        # than emitted ungrammatical.
        frame_verbs = [v for v in verbs if not _feat(v).get("copula")]
        # At complexity 3 the frame is `the <noun> <verb>` with no object slot,
        # so a TRANSITIVE verb there is ungrammatical ("the water takes").
        # Prefer verbs that can stand alone; the object test below still fires
        # if one slips through, so grammar wins over sentence length rather
        # than the filter being load-bearing on its own.
        if complexity < 4:
            standalone = [v for v in frame_verbs
                          if _feat(v).get("intransitive")
                          or _feat(v).get("ambitransitive")]
            if standalone:
                frame_verbs = standalone

        # The passive needs an auxiliary and a role marker. Both are looked up
        # in the STAGE's own words, not hardcoded: a stage that has not met
        # "be" or "by" yet simply gets no passives, which is the honest answer
        # rather than teaching a word the learner has never heard.
        aux = next((v for v in verbs if _feat(v).get("copula")), None)
        self._by_marker = next(
            (p.lemma for p in preps if p.lemma == "by"), None)
        # The TRANSFER marker, by feature: goal-taking and non-motion ('to';
        # 'into' is goal+motion and excluded). A stage without it simply
        # generates no ditransitive frames.
        goal_marker = next(
            (pp.lemma for pp in preps
             if _feat(pp).get("goal") and not _feat(pp).get("motion")), None)

        # REALISM POOLS, each licensed by the lexicon rather than hand-listed.
        # The grammar-gap census found every one of these dimensions CONSTANT
        # in the generated corpus -- one tense, one number, one determiner, no
        # adverbs, no pronouns -- and a cue that never varies carries no
        # information, so nothing downstream could learn agreement, tense or
        # determiner statistics however long it trained.
        # NOT interrogative: `whose` is possessive:True AND interrogative:True,
        # and "whose girls grew the boy" is a question wearing a declarative
        # frame -- caught by the variation census, not by the audit (mood is
        # not one of its three checks).
        possessives = [d.lemma for d in dets
                       if _feat(d).get("possessive")
                       and not _feat(d).get("interrogative")]
        manner_advs = [a for a in advs if _feat(a).get("manner")]
        # Subject pronouns: grounded 3rd person only (1st/2nd have no
        # grounding, so their scene roles could never be derived), and
        # animate-safe -- he/she by gender, they by number -- so the animacy
        # rule below cannot be violated by 'it runs'.
        pron_subjects = [
            pr for pr in prons
            if _feat(pr).get("person") == 3
            and self.parser.word_grounding.get(pr.lemma) is not None
            and (_feat(pr).get("gender") in ("m", "f")
                 or _feat(pr).get("number") == "pl")
        ]

        def _choose_det(after: str, plural: bool) -> str:
            """A licensed determiner for the NP whose next token is `after`.

            60% definite, matching the dominance of "the" in real text; the
            remainder exercises possessives (number-neutral) and, for singular
            NPs, the indefinite article -- a/an chosen by the FOLLOWING token's
            initial, which is why the adjective must be chosen before the
            determiner ("an big dog" is what choosing them in the other order
            produces).
            """
            if _rng.random() < 0.6:
                return det_word
            alts = list(possessives)
            if not plural:
                art = "an" if after[:1].lower() in "aeiou" else "a"
                if any(d.lemma == art for d in dets):
                    alts.append(art)
            return _rng.choice(alts) if alts else det_word

        def _derivable(action_feats, parts) -> bool:
            """Would `roles_from_scene` recover every role of this frame?

            Checked AT GENERATION for pronoun frames: a pronoun's features are
            thin ([MALE, PERSON]-ish), so against a same-category noun the
            reference test can tie and the role comes back None. A frame whose
            roles cannot be derived from its own scene must not be emitted --
            that is the wellformedness bar the passive arc established.
            """
            ev = SceneEvent(action=list(action_feats),
                            participants=[list(x) for x in parts])
            if not parts:
                return False
            if ev.role_of_features(parts[0]) != "agent":
                return False
            if len(parts) >= 2 and ev.role_of_features(parts[1]) != "patient":
                return False
            return len(parts) <= 2 or ev.role_of_features(parts[2]) == "goal"

        ditransitives = [v for v in frame_verbs
                         if _feat(v).get("ditransitive")]

        n_eligible = 0
        subj_usage: Dict[str, int] = {}
        # Rank weights for "zipf" (computed unconditionally -- no RNG, and
        # keeping it out of the loop keeps every arm's draw count identical).
        # Ranked over ALL stage nouns so both subject pools are covered.
        zipf_w: Dict[str, float] = {
            w.lemma: 1.0 / (rank + 1) ** ZIPF_EXPONENT
            for rank, w in enumerate(nouns)
        }
        for frame_i in range(min(FRAMES_PER_STAGE,
                                 len(nouns) * len(verbs))):
            # Periodic FORCED ditransitive draw (see DITRANSITIVE_EVERY):
            # without it the recipient construction is too rare for its role
            # area to learn anything.
            if (ditransitives and DITRANSITIVE_EVERY > 0
                    and complexity >= 4
                    and frame_i % DITRANSITIVE_EVERY == DITRANSITIVE_EVERY - 1):
                verb = _rng.choice(ditransitives)
            else:
                verb = _rng.choice(frame_verbs) if frame_verbs else None
            if verb is None or not nouns:
                continue
            obj = None

            # Selectional restriction: an agent or experiencer must be animate.
            # Falls back to the concrete nouns when the lexicon has no animate
            # word at this stage, rather than silently allowing "the anger runs".
            needs_animate = _first_arg(verb) in ("agent", "experiencer")
            subj_pool = (animate if (needs_animate and animate) else concrete)
            if SUBJECT_SAMPLING == "coverage":
                # Least-used first, random tie-break: coverage, not a rate.
                subj = min(subj_pool,
                           key=lambda w: (subj_usage.get(w.lemma, 0),
                                          _rng.random()))
                subj_usage[subj.lemma] = subj_usage.get(subj.lemma, 0) + 1
            elif SUBJECT_SAMPLING == "zipf":
                # Rank-frequency draw (one RNG call, like choice()): head
                # nouns recur, which is where per-form exposure comes from
                # in natural text (E7's pinning is a uniform-sampling
                # artifact, not a corpus-size one).
                subj = _rng.choices(
                    subj_pool,
                    weights=[zipf_w[w.lemma] for w in subj_pool], k=1)[0]
            else:
                subj = _rng.choice(subj_pool)

            # SURFACE REALIZATION, each choice licensed by the lexicon:
            #   tense    ~30% past -- forms["past"], present on all 119+ verbs
            #   number   ~30% plural subject -- forms["plural"], 108/116 nouns;
            #            plural PRESENT agreement is the bare lemma (English),
            #            past is number-invariant
            #   pronoun  ~15% subject -- grounded 3rd person, verified below
            past = complexity >= 3 and _rng.random() < PAST_RATE
            plural_form = (getattr(subj, "forms", None) or {}).get("plural")
            use_plural = bool(plural_form) and _rng.random() < PLURAL_RATE
            pron = (_rng.choice(pron_subjects)
                    if pron_subjects and _rng.random() < 0.15 else None)

            # Object chosen BEFORE the surface is assembled, because the
            # pronoun frame's derivability check needs the object's features.
            # NOT gated on complexity: a transitive verb needs its object to
            # be grammatical at any sentence length.
            obj = None
            obj_lemma = ""
            if _takes_object(verb):
                obj_pool = [o for o in concrete if o.lemma != subj.lemma]
                if obj_pool:
                    obj = _rng.choice(obj_pool)

            afeats = self.scene_features(verb.lemma)
            if pron is not None:
                pfeats = self.scene_features(pron.lemma)
                parts = [pfeats] + (
                    [self.scene_features(obj.lemma)] if obj else [])
                if not (pfeats and _derivable(afeats, parts)):
                    # A frame whose roles its own scene cannot derive is not
                    # emitted with a pronoun; fall back to the noun subject.
                    pron = None

            if pron is not None:
                subj_key = pron.lemma
                subj_plural = _feat(pron).get("number") == "pl"
                subj_tokens = [pron.lemma]
            else:
                subj_key = subj.lemma
                subj_plural = use_plural
                subj_surface = ((plural_form or subj.lemma) if use_plural
                                else subj.lemma)
                # Adjective BEFORE determiner: a/an agrees with the token that
                # follows the article, which is the adjective when there is one.
                adj_lemma = (_rng.choice(adjs).lemma
                             if complexity >= 5 and adjs else None)
                efirst = adj_lemma if adj_lemma else subj_surface
                subj_tokens = [_choose_det(efirst, subj_plural)]
                if adj_lemma:
                    subj_tokens.append(adj_lemma)
                subj_tokens.append(subj_surface)

            if past:
                vform = (getattr(verb, "forms", None) or {}).get(
                    "past") or _finite(verb)
            elif subj_plural:
                vform = verb.lemma
            else:
                vform = _finite(verb)

            sent = list(subj_tokens)
            # Manner adverb, pre-verbal ("the dog quickly chases the cat").
            if complexity >= 4 and manner_advs and _rng.random() < 0.20:
                sent.append(_rng.choice(manner_advs).lemma)
            sent.append(vform)

            # THE PERCEIVED EVENT, built alongside the string rather than
            # recovered from it. `participants` is in CAUSAL order -- actor
            # first -- which is a fact about the world and stays fixed however
            # the sentence orders its words. Keyed on the LEMMA: 'dogs' means
            # what 'dog' means, and the surface form inherits its grounding.
            participants = [self.scene_features(subj_key)]

            # OBJECT NUMBER varies too (E5/E6, #135): objects were the last
            # always-singular NP slot -- a censused-constant axis carrying no
            # information, and the binding constraint on distinct plural
            # forms (the PL image's n-invariant ceiling was built from ~10).
            # Same PLURAL_RATE as subjects; English objects trigger no
            # agreement, so the surface is free. The EVENT stays keyed on
            # the lemma: 'cakes' means what 'cake' means.
            obj_use_plural = False
            if obj is not None:
                obj_plural_form = _form(obj, "plural")
                # Rate gate FIRST: at 0.0 no RNG draw may be consumed, or
                # the "off" corpus is a different REALIZATION than the
                # measured one (the off-switch lesson, in RNG form).
                obj_use_plural = (OBJECT_PLURAL_RATE > 0
                                  and bool(obj_plural_form)
                                  and _rng.random() < OBJECT_PLURAL_RATE)
                obj_lemma = _lemma(obj)
                obj_surface = (
                    obj_plural_form
                    if obj_use_plural and obj_plural_form is not None
                    else obj_lemma
                )
                sent.extend([_choose_det(obj_surface, obj_use_plural),
                             obj_surface])
                participants.append(self.scene_features(obj_lemma))

            # DITRANSITIVE: "the girl gives the ball to the boy". The third
            # participant is the transfer's RECIPIENT, causal slot 2 -> `goal`
            # (#116: these annotations were silently dropped for weeks because
            # a private copy of the role map lacked the key). Recipients are
            # animate by modeling choice; the frame is emitted only if the
            # scene can derive ALL THREE roles, the same bar pronoun frames
            # clear. Three participants also excludes the frame from
            # passivization automatically (the passive gate requires two).
            if (obj is not None and goal_marker is not None
                    and complexity >= 4 and animate
                    and _feat(verb).get("ditransitive")):
                rec_pool = [r for r in animate
                            if r.lemma not in (subj_key, obj_lemma)]
                if rec_pool:
                    rec = _rng.choice(rec_pool)
                    rfeats = self.scene_features(rec.lemma)
                    if rfeats and _derivable(
                            afeats, participants + [rfeats]):
                        sent.extend([goal_marker,
                                     _choose_det(rec.lemma, False),
                                     rec.lemma])
                        participants.append(rfeats)

            # A locative PP needs a STATIC SPATIAL preposition taking a bare NP.
            # Filtering on the lexicon's own features rather than a hand list:
            # `motion` excludes "into"/"out", `goal`/`source` exclude "to"/"from"
            # (fine with a motion verb, wrong with a stative one), and
            # non-spatial excludes "because"/"according", which need their own
            # complement. Without this the generator emitted "plays the paper
            # away the food".
            # The ground of a locative must be a DIFFERENT thing from the
            # participants. "the library sleeps among the library" is not
            # merely odd, it is unanalysable: the same referent occupies a
            # thematic role and the PP at once, so no role assignment of it can
            # be right. Found by comparing scene-derived roles against the
            # positional inducer -- the two disagreed exactly here.
            if complexity >= 6 and locatives and concrete:
                # Excluded by LEMMA, not by surface token: with plural
                # subjects, "her couches hold ... between the couch" slipped
                # the old surface check -- same referent in a thematic role
                # and the PP, the exact unanalysable shape this guard exists
                # for, resurfacing through the inflection.
                used_lemmas = {subj_key} | ({obj.lemma} if obj else set())
                used_lemmas |= {tok for tok in sent}  # incl. any recipient
                loc_pool = [c for c in concrete if c.lemma not in used_lemmas]
                if loc_pool:
                    prep = _rng.choice(locatives)
                    loc = _rng.choice(loc_pool)
                    sent.extend([prep.lemma, det_word, loc.lemma])

            event = SceneEvent(
                action=self.scene_features(verb.lemma),
                participants=participants,
            )
            sentences.append(SentencePlan(sent, event=event))

            # PASSIVE: the SAME event, said in the opposite order.
            #
            # This is the whole reason roles come from the scene. The passive
            # states the identical who-did-what while reversing the surface
            # positions, so it is the one construction where a positional
            # reading is not merely uninformative but INVERTED -- and it is
            # therefore the contrast `_learn_gating_patterns` needs in order to
            # learn that a marker reverses roles, from evidence rather than
            # from spelling.
            #
            # Rate: one in PASSIVE_EVERY eligible clauses. English runs roughly
            # 2-10% passive; this is higher, because the contrastive learner
            # needs both conditions populated and the generated corpus is only
            # ~50 sentences per stage. The number is stated rather than tuned,
            # and it changes token frequencies -- any capacity or frequency
            # result measured before this describes a different corpus.
            # Skipped for pronoun subjects: the by-phrase needs the ACCUSATIVE
            # ("by him"), and the lexicon carries no case forms -- "by he" is
            # not a passive, and training on it would corrupt the marker.
            # Skipped for PLURAL objects: the passive subject is the active
            # object, and the lexicon's aux forms cover is/was only -- "the
            # cats is chased" is not a passive, the same shape as the
            # pronoun-case exclusion below.
            if (complexity >= 4 and len(participants) == 2
                    and pron is None
                    and not obj_use_plural
                    and PASSIVE_EVERY > 0
                    and n_eligible % PASSIVE_EVERY == 0):
                passive = self.passive_of(
                    verb, obj, aux, det_word,
                    by_tokens=[det_word] + subj_tokens[1:],
                    past=past)
                if passive is not None:
                    sentences.append(SentencePlan(passive, event=event))
            if len(participants) == 2:
                n_eligible += 1

        _rng.setstate(_rng_state)
        return sentences

    def passive_of(self, verb, obj, aux, det_word: str,
                    by_tokens: List[str],
                    past: bool = False) -> Optional[List[str]]:
        """`the cat is/was chased by the dog`, or None if the lexicon cannot.

        Returns None rather than approximating: a passive missing its auxiliary
        or its participle is not a passive, and training on a malformed one
        would teach the marker to reverse roles in sentences that are not
        passive at all -- the failure mode `_learn_gating_patterns` guards
        against on the determiner.

        The auxiliary agrees with the PASSIVE subject (the active object,
        singular in the current frames), so tense selects is/was; the by-phrase
        carries the active subject NP's surface, plural and adjective included.
        """
        ppart = (getattr(verb, "forms", None) or {}).get("ppart")
        aux_forms = (getattr(aux, "forms", None) or {}) if aux else {}
        aux_form = aux_forms.get("past") if past else aux_forms.get("3sg")
        if not ppart or not aux_form or self._by_marker is None:
            return None
        return ([det_word, obj.lemma, aux_form, ppart, self._by_marker]
                + list(by_tokens))

    def generate(
        self,
        words: list,
        complexity: int,
        *,
        stage_name: Optional[str] = None,
    ) -> List[SentencePlan]:
        """Generate training sentences; CDS corpora at early stages.

        At ``SENTENCES`` complexity (>=4), CDS telegraphic lines are blended
        with generic full SVO frames so role assignment gets enough exposure.

        CDS and holdout-bridge lines carry NO event, and must not: they are
        real transcribed text, nobody recorded the scene, and inventing one
        would be fabricating the supervision this whole path exists to avoid.
        They keep `roles=None` and fall through to positional induction.
        """
        generic = self.generate_generic(words, complexity)

        if not stage_name:
            return generic

        cds_tokens = self.cds_corpus_sentences(stage_name, words, complexity)
        if not cds_tokens:
            return generic
        cds = [SentencePlan(list(tokens)) for tokens in cds_tokens]

        if complexity >= 4 and stage_name in ("SENTENCES", "COMPLEX_GRAMMAR"):
            seen: set = set()
            merged: List[SentencePlan] = []
            for plan in cds + generic:
                key = tuple(plan.tokens)
                if key in seen:
                    continue
                seen.add(key)
                merged.append(plan)
            if self.holdout_words and stage_name == "SENTENCES":
                from .holdout_bridges import holdout_bridge_token_lists

                for tokens in holdout_bridge_token_lists(self.holdout_words):
                    key = tuple(tokens)
                    if key not in seen:
                        seen.add(key)
                        merged.append(SentencePlan(list(tokens)))
            return merged

        return cds

    def register_surface_forms(self, sentences: List[SentencePlan],
                                stage_words: list) -> set:
        """Register inflected surface forms, sharing the lemma's grounding.

        REQUIRED, not hygiene. Generated sentences now carry finite verb forms
        ("builds", not "build") so they agree with their subject, and
        `compile_corpus` skips any token missing from `stim_map`
        (`if update.word not in self.stim_map: continue`). Without this, every
        inflected verb would be SILENTLY DROPPED from role training and
        ROLE_ACTION would quietly empty out -- the same dead-path shape that
        left the verb outside the role system before.

        The form inherits the LEMMA's grounding rather than being auto-grounded
        from scratch: "builds" means what "build" means, and a form that gets an
        empty `GroundingContext` cannot enter a core lexicon at all.

        KNOWN LIMIT: `word_grounding` holds ONE grounding per surface string,
        so a form claimed by two lemmas ("lives" is life.plural AND live.3sg;
        "thought" is a noun AND think.past) inherits from whichever stage word
        claims it first -- deterministic (stage word order is fixed) but
        arbitrary. Homograph pairs that share a lemma string ("loves" from
        noun-love or verb-love) are unaffected: both routes resolve to the
        same `word_grounding["love"]`. Token-level POS disambiguation would
        need per-context grounding, which the parser does not represent; the
        lexicon INDEX no longer collapses these (see
        `lookup_lexicon_entries`), so readers that know the expected POS can
        recover the hidden reading even though grounding cannot.
        """
        form_to_lemma: Dict[str, str] = {}
        for w in stage_words:
            for form in (getattr(w, "forms", None) or {}).values():
                if isinstance(form, str) and form:
                    form_to_lemma.setdefault(form, w.lemma)

        added: set = set()
        for sent in sentences:
            for tok in sent.tokens:
                if tok in added or tok in self.parser.stim_map:
                    continue
                self.parser.register_word(tok)
                lemma = form_to_lemma.get(tok)
                if lemma is not None:
                    ctx = self.parser.word_grounding.get(lemma)
                    if ctx is not None:
                        self.parser.word_grounding[tok] = ctx
                added.add(tok)
        return added

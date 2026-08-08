"""Curriculum-based developmental training for EmergentParser.

Progressively trains the parser through stages of increasing complexity,
using the lexicon's frequency and age-of-acquisition data to select
stage-appropriate vocabulary.  Plasticity decreases at later stages.

WHY A CURRICULUM RATHER THAN THE WHOLE CORPUS AT ONCE.  Two independent
reasons converge on the same design.

The developmental one: children do not receive their vocabulary uniformly.
The stage names below track the observed sequence -- babble, first words, the
vocabulary spurt, two-word utterances, full sentences -- and vocabulary is
selected by frequency and age of acquisition, so the model meets words in
roughly the order a child does.  If category structure genuinely falls out of
distributional and grounding regularities, it should fall out under that
ordering too, and testing it is part of the claim.

The mechanical one: assemblies are formed by potentiating a connectome that
is already carrying every earlier item, so training order is not neutral.
Presenting complex multi-clause input before the lexicon has stabilised means
role and phrase training write against assemblies that are still moving.
Staging exists so each phase trains on something the previous phase already
made stable.

THE DECREASING BETA IS THE POINT, not a tuning artifact.  ``_STAGE_CONFIG``
runs plasticity from 0.20 at BABBLE down to 0.06 at CONVERSATION.  High beta
early means each exposure moves the connectome a long way, so a handful of
presentations suffices to carve out a word -- fast acquisition, but easily
overwritten.  Low beta later means new material perturbs established
structure only slightly, so the grammar learned earlier survives continued
input.  This is the stability-plasticity trade-off implemented as a schedule,
and it is why a late stage cannot simply be run with an early stage's beta.

Each stage lists its ``phases``: which of the parser's training routines run
at that stage.  Phases are cumulative in practice -- later stages re-run
lexicon and roles rather than assuming earlier work is untouched -- which is
what keeps earlier structure refreshed as beta falls.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from ..core.grounding import GroundingContext
from ..core.scene import SceneEvent
from ..core.sentence import SentencePlan
from .data import GroundedSentence
from ..core.areas import GROUNDING_TO_CORE, DET_CORE
from ..training.perf import (
    PRESET_VOCAB_SKIP_THRESHOLD,
    STAGE_WORD_ORDER_REPS,
    effective_stage_phases,
    stage_distributional_reps,
    stage_training_rounds,
)


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


@dataclass
class StageResult:
    """Metrics from training a single curriculum stage."""
    stage_name: str
    vocab_size: int
    classification_accuracy: float
    beta: float
    sentences_trained: int
    phases_run: List[str] = field(default_factory=list)


# Stage name -> (beta, sentence_complexity, phases)
_STAGE_CONFIG = {
    "BABBLE": {
        "beta": 0.20,
        "complexity": 0,
        "phases": [],
    },
    "FIRST_WORDS": {
        "beta": 0.15,
        "complexity": 1,
        "phases": ["lexicon"],
    },
    "VOCABULARY_SPURT": {
        "beta": 0.12,
        "complexity": 2,
        "phases": ["lexicon", "distributional"],
    },
    "TWO_WORD": {
        "beta": 0.10,
        "complexity": 2,
        "phases": ["lexicon", "distributional", "roles"],
    },
    "SENTENCES": {
        "beta": 0.10,
        "complexity": 4,
        "phases": ["lexicon", "distributional", "roles",
                    "phrases", "word_order", "tense", "mood", "polarity",
                    "prediction"],
    },
    "COMPLEX_GRAMMAR": {
        "beta": 0.08,
        "complexity": 6,
        "phases": ["lexicon", "distributional", "roles",
                    "phrases", "word_order", "tense", "mood", "polarity",
                    "conjunctions"],
    },
    "INSTRUCTIONS": {
        "beta": 0.10,
        "complexity": 3,
        "phases": ["lexicon", "distributional", "roles",
                    "phrases", "mood", "prediction"],
    },
    "DIALOGUE": {
        "beta": 0.08,
        "complexity": 4,
        "phases": ["lexicon", "distributional", "roles",
                    "phrases", "word_order", "mood", "prediction", "dialogue"],
    },
    "CONVERSATION": {
        "beta": 0.06,
        "complexity": 6,
        "phases": ["lexicon", "distributional", "roles",
                    "phrases", "word_order", "tense", "mood", "polarity",
                    "conjunctions", "prediction", "dialogue", "conversation"],
    },
}


class CurriculumTrainer:
    """Wraps EmergentParser training with developmental curriculum stages.

    Progressively trains the parser through stages of increasing complexity,
    using the lexicon's frequency and age-of-acquisition data to select
    stage-appropriate vocabulary.  Plasticity decreases at later stages.

    Uses src/lexicon/curriculum.py's Curriculum class for word selection
    and stage management.
    """

    def __init__(self, parser, *, holdout_words: Optional[Set[str]] = None):
        self.parser = parser
        self.holdout_words: Set[str] = set(holdout_words or ())
        self.parser.lexicon_holdouts = set(self.holdout_words)
        self.stage_results: List[StageResult] = []
        self._lexicon_manager = self._build_lexicon_manager()
        from ..core.corpus_index import TransitionCache
        self._transition_cache = TransitionCache()

    @staticmethod
    def _build_lexicon_manager():
        """Build a LexiconManager from the raw lexicon data files."""
        from neural_assemblies.lexicon.lexicon_manager import (
            LexiconManager, Word, WordCategory, SemanticDomain,
        )
        from neural_assemblies.lexicon.data import (
            NOUNS, VERBS, ADJECTIVES, ADVERBS,
            PREPOSITIONS, PRONOUNS, DETERMINERS, CONJUNCTIONS,
        )

        # Map POS labels -> WordCategory
        _CAT_MAP = {
            "NOUN": WordCategory.NOUN,
            "VERB": WordCategory.VERB,
            "ADJ": WordCategory.ADJECTIVE,
            "ADV": WordCategory.ADVERB,
            "PREP": WordCategory.PREPOSITION,
            "PRON": WordCategory.PRONOUN,
            "DET": WordCategory.DETERMINER,
            "CONJ": WordCategory.CONJUNCTION,
        }

        # Map domain strings -> SemanticDomain (best-effort)
        _DOMAIN_MAP = {}
        for member in SemanticDomain:
            _DOMAIN_MAP[member.name] = member

        lm = LexiconManager()

        categories = [
            (NOUNS, "NOUN"),
            (VERBS, "VERB"),
            (ADJECTIVES, "ADJ"),
            (ADVERBS, "ADV"),
            (PREPOSITIONS, "PREP"),
            (PRONOUNS, "PRON"),
            (DETERMINERS, "DET"),
            (CONJUNCTIONS, "CONJ"),
        ]

        for entries, pos in categories:
            wc = _CAT_MAP[pos]
            for entry in entries:
                domains = []
                for d in entry.get("domains", []):
                    if d in _DOMAIN_MAP:
                        domains.append(_DOMAIN_MAP[d])

                word = Word(
                    lemma=entry["lemma"],
                    category=wc,
                    forms=entry.get("forms", {}),
                    semantic_domains=domains,
                    features=entry.get("features", {}),
                    frequency=entry.get("freq", 0.0),
                    age_of_acquisition=entry.get("aoa", 10.0),
                )
                lm.add_word(word)

        return lm

    def _cds_corpus_sentences(
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

    def _scene_features(self, lemma: str) -> List[str]:
        """The perceptual features of a word, as `roles_from_scene` reads them.

        Deliberately the SAME accessor the derivation uses, so a participant
        bundle cannot be built from one view of a word's grounding and matched
        against another. A word with no grounding yields no features, and
        `SceneEvent.role_of_features` then declines to give it a role.
        """
        from ..core.scene import _word_features

        ctx = self.parser.word_grounding.get(lemma)
        return list(_word_features(ctx)) if ctx is not None else []

    def _generate_sentences_generic(
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
        for frame_i in range(min(50, len(nouns) * len(verbs))):
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
            subj = _rng.choice(subj_pool)

            # SURFACE REALIZATION, each choice licensed by the lexicon:
            #   tense    ~30% past -- forms["past"], present on all 119+ verbs
            #   number   ~30% plural subject -- forms["plural"], 108/116 nouns;
            #            plural PRESENT agreement is the bare lemma (English),
            #            past is number-invariant
            #   pronoun  ~15% subject -- grounded 3rd person, verified below
            past = complexity >= 3 and _rng.random() < 0.30
            plural_form = (getattr(subj, "forms", None) or {}).get("plural")
            use_plural = bool(plural_form) and _rng.random() < 0.30
            pron = (_rng.choice(pron_subjects)
                    if pron_subjects and _rng.random() < 0.15 else None)

            # Object chosen BEFORE the surface is assembled, because the
            # pronoun frame's derivability check needs the object's features.
            # NOT gated on complexity: a transitive verb needs its object to
            # be grammatical at any sentence length.
            obj = None
            if _takes_object(verb):
                obj_pool = [o for o in concrete if o.lemma != subj.lemma]
                if obj_pool:
                    obj = _rng.choice(obj_pool)

            afeats = self._scene_features(verb.lemma)
            if pron is not None:
                pfeats = self._scene_features(pron.lemma)
                parts = [pfeats] + (
                    [self._scene_features(obj.lemma)] if obj else [])
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
            participants = [self._scene_features(subj_key)]

            if obj is not None:
                sent.extend([_choose_det(obj.lemma, False), obj.lemma])
                participants.append(self._scene_features(obj.lemma))

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
                            if r.lemma not in (subj_key, obj.lemma)]
                if rec_pool:
                    rec = _rng.choice(rec_pool)
                    rfeats = self._scene_features(rec.lemma)
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
                action=self._scene_features(verb.lemma),
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
            if (complexity >= 4 and len(participants) == 2
                    and pron is None
                    and PASSIVE_EVERY > 0
                    and n_eligible % PASSIVE_EVERY == 0):
                passive = self._passive_of(
                    verb, obj, aux, det_word,
                    by_tokens=[det_word] + subj_tokens[1:],
                    past=past)
                if passive is not None:
                    sentences.append(SentencePlan(passive, event=event))
            if len(participants) == 2:
                n_eligible += 1

        _rng.setstate(_rng_state)
        return sentences

    def _passive_of(self, verb, obj, aux, det_word: str,
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

    def _generate_sentences(
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
        generic = self._generate_sentences_generic(words, complexity)

        if not stage_name:
            return generic

        cds_tokens = self._cds_corpus_sentences(stage_name, words, complexity)
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

    def _register_surface_forms(self, sentences: List[SentencePlan],
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

    def _set_global_beta(self, beta: float) -> None:
        """Set plasticity (beta) for all area-to-area connections."""
        brain = self.parser.brain
        for area_name in brain.areas:
            area = brain.areas[area_name]
            for src in area.beta_by_area:
                area.beta_by_area[src] = beta
                brain._engine.set_beta(area_name, src, beta)

    def _evaluate_classification(self, words: list) -> float:
        """Quick classification accuracy on a word list.

        Checks whether known-category words are classified correctly
        by the parser.
        """
        from neural_assemblies.lexicon.lexicon_manager import WordCategory

        _CAT_LABEL = {
            WordCategory.NOUN: "NOUN",
            WordCategory.VERB: "VERB",
            WordCategory.ADJECTIVE: "ADJ",
            WordCategory.ADVERB: "ADV",
            WordCategory.PREPOSITION: "PREP",
            WordCategory.PRONOUN: "PRON",
            WordCategory.DETERMINER: "DET",
            WordCategory.CONJUNCTION: "CONJ",
        }

        correct = 0
        total = 0
        for w in words:
            expected = _CAT_LABEL.get(w.category)
            if expected is None:
                continue
            lemma = w.lemma
            if lemma not in self.parser.stim_map:
                continue
            grounding = self.parser.word_grounding.get(lemma)
            cat, _ = self.parser.classify_word_cached(lemma, grounding=grounding)
            if cat == expected:
                correct += 1
            total += 1

        return correct / max(total, 1)

    def _get_stage_words(self, stage_name: str) -> list:
        """Get words appropriate for a curriculum stage.

        Uses AoA and frequency thresholds matching developmental stages.
        """
        _STAGE_THRESHOLDS = {
            "BABBLE": {"max_aoa": 0.0, "min_freq": 0.0, "target": 0},
            "FIRST_WORDS": {"max_aoa": 2.0, "min_freq": 4.0,
                            "target": 50},
            "VOCABULARY_SPURT": {"max_aoa": 2.5, "min_freq": 3.5,
                                 "target": 200},
            "TWO_WORD": {"max_aoa": 3.0, "min_freq": 3.0,
                         "target": 300},
            "SENTENCES": {"max_aoa": 4.0, "min_freq": 2.5,
                          "target": 500},
            "COMPLEX_GRAMMAR": {"max_aoa": 5.0, "min_freq": 2.0,
                                "target": 800},
            "DIALOGUE": {"max_aoa": 4.5, "min_freq": 2.2,
                         "target": 600},
            "CONVERSATION": {"max_aoa": 6.0, "min_freq": 1.8,
                             "target": 1000},
        }

        thresholds = _STAGE_THRESHOLDS.get(stage_name)
        if thresholds is None:
            return []

        max_aoa = thresholds["max_aoa"]
        min_freq = thresholds["min_freq"]
        target = thresholds["target"]

        candidates = self._lexicon_manager.get_by_aoa(max_aoa)
        candidates = [w for w in candidates
                      if w.frequency >= min_freq]

        # Sort by frequency (desc) then AoA (asc)
        candidates.sort(
            key=lambda w: (-w.frequency, w.age_of_acquisition))

        selected = candidates[:target]
        if self.holdout_words:
            selected = [
                w for w in selected if w.lemma not in self.holdout_words
            ]
        return selected

    def _stage_needs_lexicon(self, stage_words: list) -> bool:
        """True if any stage word is missing from core lexicons."""
        for w in stage_words:
            lemma = w.lemma
            ctx = self.parser.word_grounding.get(lemma, GroundingContext())
            core = GROUNDING_TO_CORE.get(ctx.dominant_modality, DET_CORE)
            if lemma not in self.parser.core_lexicons.get(core, {}):
                return True
        return False

    def train_stage(self, stage_name: str) -> StageResult:
        """Train the parser for one curriculum stage.

        Args:
            stage_name: One of the keys in _STAGE_CONFIG.

        Returns:
            StageResult with metrics for this stage.
        """
        config = _STAGE_CONFIG[stage_name]
        beta = config["beta"]
        complexity = config["complexity"]
        phases = effective_stage_phases(
            stage_name, config["phases"], fast=self.parser.fast_training,
        )

        old_rounds = self.parser.rounds
        stage_rounds = stage_training_rounds(
            stage_name, fast=self.parser.fast_training,
        )
        if stage_rounds is not None:
            self.parser.rounds = stage_rounds

        try:
            return self._train_stage_impl(
                stage_name, config, beta, complexity, phases,
            )
        finally:
            self.parser.rounds = old_rounds

    def _train_stage_impl(
        self,
        stage_name: str,
        config: dict,
        beta: float,
        complexity: int,
        phases: list,
    ) -> StageResult:
        stage_words = self._get_stage_words(stage_name)

        if stage_name == "BABBLE":
            from ..train_progress import current_progress
            prog = current_progress()
            n_babble = len(getattr(self.parser, "babble_forms", []))
            prog.info(f"stage BABBLE: {n_babble} forms registered (pre-lexical)")
            result = StageResult(
                stage_name="BABBLE",
                vocab_size=n_babble,
                classification_accuracy=0.0,
                beta=beta,
                sentences_trained=0,
                phases_run=["babble"],
            )
            self.stage_results.append(result)
            return result

        if not stage_words:
            return StageResult(
                stage_name=stage_name,
                vocab_size=0,
                classification_accuracy=0.0,
                beta=beta,
                sentences_trained=0,
                phases_run=[],
            )

        # Set plasticity
        self._set_global_beta(beta)

        # Register words in the parser
        for w in stage_words:
            self.parser.register_word(w.lemma)

        # Generate training sentences
        sentences = self._generate_sentences(
            stage_words, complexity, stage_name=stage_name,
        )
        self._register_surface_forms(sentences, stage_words)

        from ..train_progress import current_progress
        from .data import create_instruction_sentences
        from .holdout_bridges import merge_prediction_corpus
        from ..training.schedule import TrainingScheduleExecutor
        from ..curriculum.conversation import get_conversation_curriculum

        prog = current_progress()
        prog.info(
            f"stage {stage_name}: {len(stage_words)} words, "
            f"{len(sentences)} sentences, beta={beta}",
        )

        if "prediction" in phases:
            if self.holdout_words:
                extra_pred = merge_prediction_corpus(
                    self.parser,
                    self.holdout_words,
                    stage_name=stage_name,
                )
            else:
                extra_pred = create_instruction_sentences()
        else:
            extra_pred = None

        schedule = TrainingScheduleExecutor.build_stage_schedule(
            self.parser, stage_name, sentences, phases,
            extra_prediction=extra_pred,
            conversation_sents=(
                get_conversation_curriculum(self.parser.word_grounding)
                if "conversation" in phases else None
            ),
            transition_cache=self._transition_cache,
        )

        schedule.transition_cache = self._transition_cache

        if "lexicon" in phases and not self._stage_needs_lexicon(stage_words):
            prog.info("skip lexicon (all stage words already trained)")
            schedule.phases = [p for p in schedule.phases if p != "lexicon"]

        if "dialogue" in phases:
            from ..curriculum.dialogue import get_dialogue_pairs
            from ..curriculum.conversation import get_conversation_pairs

            schedule.dialogue_pairs = (
                get_dialogue_pairs()
                + get_conversation_pairs(self.parser.word_grounding)
            )

        if self.holdout_words:
            schedule.holdout_words = self.holdout_words

        executor = TrainingScheduleExecutor(self.parser)
        run_phases = [r.phase for r in executor.run(schedule)]

        # Evaluate (skip in sweep mode — classification pass is for logging only)
        from ..training.perf import sweep_mode_enabled

        if sweep_mode_enabled():
            accuracy = -1.0
        else:
            with prog.phase("evaluate"):
                accuracy = self._evaluate_classification(stage_words)

        result = StageResult(
            stage_name=stage_name,
            vocab_size=len(stage_words),
            classification_accuracy=accuracy,
            beta=beta,
            sentences_trained=len(sentences),
            phases_run=run_phases,
        )
        self.stage_results.append(result)
        return result

    def train_remedial(
        self,
        sentences: List[GroundedSentence],
        *,
        phases: Optional[List[str]] = None,
        label: str = "ADAPTIVE",
        beta: float = 0.10,
        holdout_words: Optional[Set[str]] = None,
    ) -> StageResult:
        """Run a lightweight remedial pass on targeted sentences."""
        if not sentences:
            return StageResult(
                stage_name=label,
                vocab_size=0,
                classification_accuracy=0.0,
                beta=beta,
                sentences_trained=0,
                phases_run=[],
            )

        phases = list(phases or ["distributional"])
        old_rounds = self.parser.rounds
        self.parser.rounds = min(self.parser.rounds, 2)

        try:
            self._set_global_beta(beta)
            from ..train_progress import current_progress
            from ..training.schedule import TrainingScheduleExecutor

            prog = current_progress()
            prog.info(
                f"remedial {label}: {len(sentences)} sentences, phases={phases}",
            )

            schedule = TrainingScheduleExecutor.build_stage_schedule(
                self.parser,
                label,
                # Carry the event across rather than dropping to tokens: a
                # remedial pass re-trains the same sentences, and a sentence
                # that loses its scene here would silently switch to positional
                # roles halfway through training.
                [SentencePlan(list(s.words), event=getattr(s, "event", None),
                              mood=getattr(s, "mood", "declarative"))
                 for s in sentences],
                phases,
                transition_cache=self._transition_cache,
            )
            schedule.transition_cache = self._transition_cache
            if holdout_words:
                schedule.holdout_words = set(holdout_words)

            executor = TrainingScheduleExecutor(self.parser)
            run_phases = [r.phase for r in executor.run(schedule)]

            return StageResult(
                stage_name=label,
                vocab_size=len({w for s in sentences for w in s.words}),
                classification_accuracy=0.0,
                beta=beta,
                sentences_trained=len(sentences),
                phases_run=run_phases,
            )
        finally:
            self.parser.rounds = old_rounds

    def train_curriculum(
        self,
        max_stage: str = "SENTENCES",
        *,
        holdout_words: Optional[Set[str]] = None,
    ) -> List[StageResult]:
        """Train the parser through multiple curriculum stages.

        Runs stages in order from FIRST_WORDS up to (and including)
        max_stage.

        Args:
            max_stage: Last stage name to train (default "SENTENCES").
            holdout_words: Optional lexicon holdouts for generalization probes.

        Returns:
            List of StageResult for each stage trained.
        """
        if holdout_words is not None:
            self.holdout_words = set(holdout_words)
            self.parser.lexicon_holdouts = set(self.holdout_words)
        stage_order = [
            "FIRST_WORDS", "VOCABULARY_SPURT", "TWO_WORD",
            "SENTENCES", "COMPLEX_GRAMMAR",
        ]

        results = []
        from ..train_progress import current_progress
        prog = current_progress()
        for stage_name in stage_order:
            with prog.section(stage_name):
                result = self.train_stage(stage_name)
            results.append(result)
            acc_label = (
                "skipped (sweep)"
                if result.classification_accuracy < 0
                else f"{result.classification_accuracy:.1%}"
            )
            prog.info(
                f"{stage_name} complete: vocab={result.vocab_size} "
                f"acc={acc_label} phases={result.phases_run}",
            )
            if stage_name == max_stage:
                break

        return results

    def train_conversation_path(
        self,
        max_stage: str = "DIALOGUE",
        *,
        skip_early_if_loaded: bool = True,
        holdout_words: Optional[Set[str]] = None,
    ) -> List[StageResult]:
        """Train developmental stages through naturalistic conversation.

        Runs FIRST_WORDS → … → DIALOGUE (or CONVERSATION).

        When ``skip_early_if_loaded`` and vocabulary is already large
        (preset vocab), jumps directly to DIALOGUE/CONVERSATION stages.

        Args:
            max_stage: Last stage; ``DIALOGUE`` or ``CONVERSATION``.
            skip_early_if_loaded: Skip early developmental stages for presets.
            holdout_words: Optional lexicon holdouts for generalization probes.

        Returns:
            List of StageResult per stage trained.
        """
        if holdout_words is not None:
            self.holdout_words = set(holdout_words)
            self.parser.lexicon_holdouts = set(self.holdout_words)
        full_order = [
            "FIRST_WORDS", "VOCABULARY_SPURT", "TWO_WORD",
            "SENTENCES", "COMPLEX_GRAMMAR", "DIALOGUE", "CONVERSATION",
        ]
        if max_stage not in full_order:
            raise ValueError(
                f"max_stage {max_stage!r} not in {full_order}"
            )

        from ..train_progress import current_progress
        from ..training.perf import should_skip_early_curriculum

        prog = current_progress()
        prog.info(f"conversation path target: {max_stage}")

        if should_skip_early_curriculum(len(self.parser.stim_map), max_stage) and skip_early_if_loaded:
            stage_order = full_order[full_order.index("DIALOGUE"):]
            prog.info(
                f"skip early stages (vocab={len(self.parser.stim_map)} "
                f">= {PRESET_VOCAB_SKIP_THRESHOLD})",
            )
        else:
            stage_order = full_order

        results = []
        for stage_name in stage_order:
            with prog.section(stage_name):
                result = self.train_stage(stage_name)
            results.append(result)
            prog.info(
                f"{stage_name}: vocab={result.vocab_size} "
                f"acc={('skipped (sweep)' if result.classification_accuracy < 0 else f'{result.classification_accuracy:.1%}')}",
            )
            if stage_name == max_stage:
                break
        return results

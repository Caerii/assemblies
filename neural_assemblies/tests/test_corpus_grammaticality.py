"""The generated training corpus must be grammatical English.

WHY THIS IS A TEST AND NOT A NOTE. Role induction learns AGENT/PATIENT from noun
position relative to the verb, so the generated corpus IS the supervision. It
previously contained

    the store build the dog

-- no subject-verb agreement, an object forced onto a verb regardless of
transitivity, and subjects drawn uniformly over all nouns. `dog` then received
its ONLY role binding from that sentence, as the object, which made
`nemo2025_curriculum`'s `role_probe_accuracy >= 1.0` unreachable: the probe asks
for dog=AGENT and the training data only ever said PATIENT.

Nothing in the suite noticed, because every downstream metric was measured on
the same broken corpus. This test checks the corpus directly.

WHAT IT CHECKS, all against the lexicon rather than against a judgement of
English: agreement (`forms["3sg"]`), transitivity
(`features.transitive/intransitive/ambitransitive`), and animacy of agents
(`features.animate`, `arguments[0]`).

WHAT IT DOES NOT CHECK: semantic plausibility. "the dog says the cat" passes
here and is nonsense. Stated so the test is not mistaken for a stronger
guarantee than it gives.
"""
from __future__ import annotations

import pytest

from neural_assemblies.assembly_calculus.emergent import EmergentParser
from neural_assemblies.assembly_calculus.emergent.curriculum import (
    CurriculumTrainer,
)
from neural_assemblies.lexicon.lexicon_manager import WordCategory

STAGES = [("TWO_WORD", 3), ("SENTENCES", 4), ("COMPLEX_GRAMMAR", 6)]


def _feat(w) -> dict:
    return getattr(w, "features", None) or {}


@pytest.fixture(scope="module")
def trainer():
    # Small brain: this exercises the SENTENCE GENERATOR, which does not touch
    # the substrate, so there is no reason to pay for a real one.
    return CurriculumTrainer(EmergentParser(n=500, k=20, seed=42))


def _analyse(plans, words):
    sentences = [p.tokens for p in plans]
    by_lemma = {w.lemma: w for w in words}
    form_to_word = {}
    for w in words:
        for f in (getattr(w, "forms", None) or {}).values():
            if isinstance(f, str):
                form_to_word.setdefault(f, w)

    out = []
    for sent in sentences:
        entries = [(t, by_lemma.get(t) or form_to_word.get(t)) for t in sent]
        verbs = [(i, t, w) for i, (t, w) in enumerate(entries)
                 if w is not None and w.category == WordCategory.VERB]
        if not verbs:
            continue
        vi, vtok, vword = verbs[0]
        # Subjects may now be PRONOUNS; objects are still nouns only.
        nouns = [(i, t, w) for i, (t, w) in enumerate(entries)
                 if w is not None and w.category in (WordCategory.NOUN,
                                                     WordCategory.PRONOUN)]
        # A noun after a PREPOSITION is a PP object, not a direct object.
        # Without this the check called "the boy sleeps in the house" an object
        # on an intransitive verb -- 15 false positives when first written.
        prep_at = [i for i, (t, w) in enumerate(entries)
                   if w is not None and w.category == WordCategory.PREPOSITION]
        subj = next((x for x in nouns if x[0] < vi), None)
        obj = next((x for x in nouns
                    if x[0] > vi and x[2].category == WordCategory.NOUN
                    and not any(p < x[0] for p in prep_at)), None)
        out.append((" ".join(sent), vtok, vword, subj, obj))
    return out


def _licensed_verb_forms(vw, subj_tok, subj_w):
    """Verb tokens grammatical for THIS subject.

    English agreement, read off the lexicon: PAST is number-invariant;
    present is 3sg for a singular subject and the bare lemma for a plural
    one. Subject number comes from the surface (token == the noun's plural
    form) or, for pronouns, from the `number` feature.
    """
    forms = getattr(vw, "forms", None) or {}
    if subj_w.category == WordCategory.PRONOUN:
        plural = (getattr(subj_w, "features", None) or {}).get(
            "number") == "pl"
        numbers = [plural]
    else:
        noun_plural = (getattr(subj_w, "forms", None) or {}).get("plural")
        if subj_tok == noun_plural == subj_w.lemma:
            # NUMBER-AMBIGUOUS surface ("fish" is its own plural): either
            # agreement is grammatical. Latent until coverage sampling made
            # rare nouns actually surface as subjects -- the analyzer, not
            # the generator, was wrong on "the fish destroys".
            numbers = [False, True]
        else:
            numbers = [subj_tok == noun_plural]
    lic = {forms.get("past")}
    for plural in numbers:
        lic.add(vw.lemma if plural else forms.get("3sg"))
    return {x for x in lic if x}


@pytest.mark.parametrize("stage,complexity", STAGES)
def test_subject_verb_agreement(trainer, stage, complexity):
    words = trainer._get_stage_words(stage)
    parsed = _analyse(
        trainer.generation.generate_generic(words, complexity), words)
    assert parsed, f"{stage} generated no analysable sentences"
    bad = [
        (s, vtok, sorted(_licensed_verb_forms(vw, subj[1], subj[2])))
        for s, vtok, vw, subj, _obj in parsed
        if subj is not None
        and _licensed_verb_forms(vw, subj[1], subj[2])
        and vtok not in _licensed_verb_forms(vw, subj[1], subj[2])
    ]
    assert not bad, (
        f"{stage}: {len(bad)} sentences break agreement, "
        f"e.g. {bad[0][0]!r} (licensed {bad[0][2]!r})")


@pytest.mark.parametrize("stage,complexity", STAGES)
def test_transitivity_is_respected(trainer, stage, complexity):
    words = trainer._get_stage_words(stage)
    parsed = _analyse(
        trainer.generation.generate_generic(words, complexity), words)
    stranded = [s for s, _t, vw, _su, obj in parsed
                if _feat(vw).get("intransitive") and obj is not None]
    starved = [s for s, _t, vw, _su, obj in parsed
               if (_feat(vw).get("transitive") or _feat(vw).get("ditransitive"))
               and obj is None]
    assert not stranded, (
        f"{stage}: object on an intransitive verb, e.g. {stranded[0]!r}")
    assert not starved, (
        f"{stage}: transitive verb with no object, e.g. {starved[0]!r}")


@pytest.mark.parametrize("stage,complexity", STAGES)
def test_agents_are_animate(trainer, stage, complexity):
    words = trainer._get_stage_words(stage)
    parsed = _analyse(
        trainer.generation.generate_generic(words, complexity), words)
    def _animate(w):
        f = _feat(w)
        if f.get("animate"):
            return True
        # Personal pronouns: he/she by gender, they by number; 'it' is not.
        return bool(f.get("personal")) and (
            f.get("gender") in ("m", "f") or f.get("number") == "pl")

    bad = [s for s, _t, vw, subj, _o in parsed
           if subj is not None
           and (getattr(vw, "arguments", None) or [""])[0]
           in ("agent", "experiencer")
           and not _animate(subj[2])]
    assert not bad, (
        f"{stage}: inanimate subject for an agentive verb, e.g. {bad[0]!r}")


def test_generation_is_deterministic(trainer):
    """The generator reseeds and restores global RNG, so two calls must agree.

    Guards the property that made this corpus auditable at all -- a generator
    that drifted between calls would make every downstream golden unreproducible
    for reasons unrelated to the substrate.
    """
    words = trainer._get_stage_words("SENTENCES")
    first = trainer.generation.generate_generic(words, 4)
    second = trainer.generation.generate_generic(words, 4)
    assert first == second


def test_zipf_arm_is_grammatical_and_concentrates(trainer, monkeypatch):
    """SUBJECT_SAMPLING="zipf" (E13, #142) must stay grammatical AND actually
    redistribute: head-noun subject share strictly above uniform's.

    The concentration check is the arms-must-differ lesson as a unit test --
    a zipf branch that draws like uniform would run E13 as a null and report
    the baseline as a verdict. The default arm is pinned byte-identical
    below (`test_default_sampling_unchanged_by_zipf_branch`).
    """
    from collections import Counter

    from neural_assemblies.assembly_calculus.emergent.curriculum import (
        generation as gen,
    )

    words = trainer._get_stage_words("SENTENCES")

    def subject_counts():
        parsed = _analyse(
            trainer.generation.generate_generic(words, 4), words)
        return Counter(s[2].lemma for _s, _t, _v, s, _o in parsed
                       if s is not None and not _feat(s[2]).get("personal"))

    uni = subject_counts()
    monkeypatch.setattr(gen, "SUBJECT_SAMPLING", "zipf")
    parsed = _analyse(
        trainer.generation.generate_generic(words, 4), words)
    assert parsed, "zipf arm generated no analysable sentences"
    bad = [
        (s, vtok) for s, vtok, vw, subj, _obj in parsed
        if subj is not None
        and _licensed_verb_forms(vw, subj[1], subj[2])
        and vtok not in _licensed_verb_forms(vw, subj[1], subj[2])
    ]
    assert not bad, f"zipf arm breaks agreement, e.g. {bad[0][0]!r}"
    zipf = Counter(s[2].lemma for _s, _t, _v, s, _o in parsed
                   if s is not None and not _feat(s[2]).get("personal"))
    assert zipf and uni
    assert (max(zipf.values()) / sum(zipf.values())
            > max(uni.values()) / sum(uni.values())), (
        f"zipf head share {max(zipf.values())}/{sum(zipf.values())} not "
        f"above uniform's {max(uni.values())}/{sum(uni.values())}")


def test_default_sampling_unchanged_by_zipf_branch(trainer):
    """The default corpus must be byte-identical to the pre-zipf generator.

    The zipf weights are computed outside the frame loop and consume no RNG,
    so the "uniform" realization -- the corpus every E-series measurement
    describes -- cannot have shifted. Pinned as a token checksum rather than
    a golden file: any drift in the default realization is a finding, not
    a formatting change.
    """
    import hashlib

    words = trainer._get_stage_words("SENTENCES")
    plans = trainer.generation.generate_generic(words, 4)
    digest = hashlib.sha256(
        "\n".join(" ".join(p.tokens) for p in plans).encode()).hexdigest()
    assert digest == EXPECTED_DEFAULT_CORPUS_SHA, (
        "default (uniform) corpus realization changed -- if intentional, "
        "re-pin EXPECTED_DEFAULT_CORPUS_SHA and note which measurements "
        "the new corpus invalidates")


#: sha256 of the SENTENCES-stage default corpus (n=500/k=20/seed=42 fixture).
#: Pinned when the zipf branch landed (E13, #142); every E1-E12 measurement
#: describes this realization.
EXPECTED_DEFAULT_CORPUS_SHA = (
    "a148792f3e9a752cf75bece3a98d0b53de9d5703e961502ad5226d6af844186c")

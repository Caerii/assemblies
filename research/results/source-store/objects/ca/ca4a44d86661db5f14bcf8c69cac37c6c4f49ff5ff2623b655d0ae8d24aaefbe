"""How much of category induction is the ANNOTATION? (step 6, grounding half)

`grounded_induction.py` removed role annotations from the loop and recovered
word order 6/6 from perceived scene structure. That closed one circularity. The
other one is still open and this file measures it.

Word categories in this parser are routed by a HAND-AUTHORED table.
`core/grounding.py` maps 45 words to sensory modalities, `dominant_modality`
picks the first non-empty one in a fixed priority order, and
`areas.GROUNDING_TO_CORE` sends that modality to a part-of-speech core area:

    visual -> NOUN_CORE   motor -> VERB_CORE   none -> DET_CORE   ...

So "the parser induces that `dog` is a noun" may be no more than the parser
reading back an annotation someone wrote. That is a strictly stronger worry than
the role one, because categories feed everything downstream: role binding, the
gating parser, the ERP arms.

THE LESION
----------
Train two parsers on IDENTICAL sentences. One keeps the grounding contexts; the
other has every `GroundingContext` replaced by an empty one, so `is_grounded` is
False for every word and `dominant_modality` returns "none" throughout. Nothing
else differs -- same corpus, same order, same seed, same hyperparameters.

Then ask both to classify the vocabulary, and score against the category the
grounding table implies. Note what that scoring means: the intact arm is being
graded against its own input, so its score is an upper bound and a sanity check,
NOT a result. The lesioned arm is the measurement.

THE BASELINE THAT MAKES THE NUMBER MEAN ANYTHING
------------------------------------------------
Accuracy alone is uninterpretable here. The vocabulary is not balanced -- the
grounding docstring records NOUN 10, DET 10, VERB 8, ADJ 5, PREP 5, PRON 4,
ADV 3 -- so a parser that answered "NOUN" every time would already score ~22%.
Majority-class accuracy is therefore printed alongside every arm, and any claim
about the lesioned arm is a claim about the gap between them.

PRE-REGISTERED PREDICTIONS
--------------------------
P1 FUNCTION WORDS SURVIVE. Task #29 established that function words are
   bootstrapped from frames rather than grounding -- they have no modality by
   construction ("none" -> DET_CORE). Predict DET/PREP/CONJ accuracy stays near
   the intact arm.

P2 NOUN/VERB DOES NOT. `classify_distributional` scores NOUN and VERB from
   verb-relative position (`word_as_pre_verb`, `word_as_post_verb`,
   `word_as_action`), which presupposes knowing which word is the verb. If that
   seed comes from grounding, deleting grounding removes the anchor and the
   content-word distinction should collapse toward the majority class.

P3 THE HONEST HEADLINE IS THE GAP. If overall lesioned accuracy lands near the
   majority-class baseline, then category induction in this repo is currently
   annotation-driven, and every downstream claim that depends on categories
   inherits that. Recorded as a prediction so it cannot be softened afterwards.

A large P1/P2 split is the interesting outcome: it would say the FUNCTION-word
route is genuinely distributional and the CONTENT-word route is not, which is a
specific, fixable gap rather than a verdict on the whole pipeline.

RESULT (2026-07-28), 3 seeds
-----------------------------
    arm                                  accuracy
    intact                                  1.000
    lesioned                                0.083
    lesioned + train_distributional         0.278
    majority class                          0.278
    always-answer-DET                       0.083

P2 and P3 HOLD. P1 IS REFUTED, and the way it fails is the point.

The lesioned parser scores 0.083, and 0.083 is exactly 3/36 -- the DET share of
the vocabulary. Per-category it reads DET 1.000 and everything else 0.000. That
is not "function words survive": it is the parser answering DET for EVERY word.
DET is where `dominant_modality` sends anything ungrounded ("none" ->
DET_CORE), so with grounding removed the fallback absorbs the whole vocabulary
and the surviving category scores 1.000 for free. P1 predicted that function
words would be robust because they are bootstrapped from frames; what actually
happened is that they are the default sink. The prediction would have been
scored as confirmed by any measurement that looked at DET alone.

Running `train_distributional` explicitly does not rescue it, and the reason it
had to be tested separately is that `train()` never calls it -- `_dist_categories`
and `_bootstrap_categories` are both EMPTY after a normal training run, so the
first version of this result was at risk of measuring "the path is not wired in"
and reporting "the path cannot do it". With it called by hand, accuracy moves
0.083 -> 0.278 and the predicted distribution becomes NOUN 32, ADJ 4: the sink
moved from DET to NOUN. 0.278 is exactly 10/36, the NOUN share, which is exactly
the majority-class baseline. The distributional route lands ON the baseline, not
above it.

SO: category assignment in this parser is annotation-driven, end to end. Three
independent hand-authored routes feed it, and they are redundant -- cutting any
one or two changes nothing, which is why the first two attempts at this lesion
both returned a clean 1.000:

    1. the corpus's per-word GroundingContexts
    2. `auto_ground` -> `lookup_lexicon_entry`, a BUNDLED POS-TAGGED DICTIONARY
    3. the static `VOCABULARY` table, registered in __init__ and documented as
       authoritative where it disagrees with the corpus

WHAT THIS DOES AND DOES NOT BOUND. It does not touch the composition results:
`universality_composition.py` runs on abstract symbols with no categories at
all, so nothing there depends on this. It does bound every claim that takes
categories as given -- role binding, the gating parser, the ERP arms -- which
should be read as "given these categories" rather than "having induced them".
It also means task #29 ("bootstrap categories from FUNCTION words") is not
established by anything measured here: the mechanism exists, but `train()` does
not invoke it and it does not clear chance when invoked by hand on this corpus.

The obvious confound, stated rather than waved at: this corpus is small and
synthetic (36 words). Distributional induction plausibly needs far more text
before frames are informative, which is exactly what task #30 (CHILDES) is for.
This result says the current pipeline does not induce categories; it does not
say the substrate could not, given a real corpus.
"""

from __future__ import annotations

import copy
import os
import sys
from collections import Counter
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
os.environ.setdefault("TRAIN_PROGRESS", "0")

SEEDS = (42, 7, 123)
N, K = 1000, 50


def truth_map() -> Dict[str, str]:
    """word -> category implied by the hand-authored grounding table."""
    from neural_assemblies.assembly_calculus.emergent.core.areas import (
        CORE_TO_CATEGORY, GROUNDING_TO_CORE,
    )
    from neural_assemblies.assembly_calculus.emergent.curriculum.data import (
        create_training_sentences,
    )

    out: Dict[str, str] = {}
    for sent in create_training_sentences():
        for word, ctx in zip(sent.words, sent.contexts):
            core = GROUNDING_TO_CORE.get(ctx.dominant_modality)
            cat = CORE_TO_CATEGORY.get(core)
            if cat:
                out[word] = cat
    return out


def strip_grounding(sentences):
    """Every GroundingContext replaced by an empty one. Words untouched."""
    from neural_assemblies.assembly_calculus.emergent.core.grounding import (
        GroundingContext,
    )

    out = copy.deepcopy(sentences)
    for sent in out:
        sent.contexts = [GroundingContext() for _ in sent.words]
    return out


def train(sentences, seed: int, *, lesion: bool = False):
    """Train a parser; *lesion* removes EVERY route to a hand-authored category.

    Stripping the sentences' GroundingContexts is NOT SUFFICIENT and the first
    version of this file was wrong because of it. Measured after a
    contexts-only strip: `parser.word_grounding` still held 45 entries with 35
    grounded -- `dog: visual`, `cat: visual` -- and both arms scored 1.000.
    The lesion was a no-op and the perfect score was measuring nothing.

    The reason is `auto_ground`, which does not read the sentence at all: it
    calls `lookup_lexicon_entry(word)` against a BUNDLED POS-TAGGED DICTIONARY
    and rebuilds the grounding from the entry's part of speech. That is a
    stronger annotation than the 45-word modality table, and it re-enters
    through `register_word` no matter what the corpus says.

    So the lesion cuts both: empty contexts AND `auto_ground` returning None.
    The assertion moved onto `parser.word_grounding` -- the object that
    actually decides -- rather than onto the sentences, which is the mistake
    that made the first result look clean.
    """
    from neural_assemblies.assembly_calculus.emergent.core.grounding import (
        VOCABULARY, GroundingContext,
    )
    from neural_assemblies.assembly_calculus.emergent.parser import EmergentParser

    # Route 3: the STATIC VOCABULARY table, registered in __init__ -- not at
    # train time, which is why patching the method after construction did
    # nothing and the assertion below still fired at 35 words. It is a
    # hand-authored word -> GroundingContext map that never consults the
    # corpus, and `_register_corpus_vocabulary` records that it is deliberately
    # authoritative: "the static table is hand-authored and is treated as
    # authoritative where the two disagree".
    #
    # The constructor takes `vocabulary`, so the cut is a parameter rather than
    # a patch. The KEYS are kept and only the contexts emptied: registration
    # also creates the phon stimuli and populates stim_map, and dropping those
    # would measure an untrained parser instead of an ungrounded one.
    vocab = ({w: GroundingContext() for w in VOCABULARY} if lesion else None)
    parser = EmergentParser(n=N, k=K, p=0.05, beta=0.1, seed=seed, rounds=10,
                            vocabulary=vocab)
    if lesion:
        parser.auto_ground = lambda word: None  # type: ignore[method-assign]

    parser.train(sentences)
    if lesion:
        still = {w: c.dominant_modality
                 for w, c in parser.word_grounding.items() if c.is_grounded}
        assert not still, (
            f"{len(still)} words are STILL grounded after the lesion "
            f"(e.g. {dict(list(still.items())[:5])}) -- there is another route "
            f"into word_grounding and the comparison is invalid until it is cut"
        )
    return parser


def score(parser, truth: Dict[str, str]) -> Tuple[float, Dict[str, List[int]]]:
    """Overall accuracy and per-true-category (hits, total)."""
    per: Dict[str, List[int]] = {}
    hits = total = 0
    for word, gold in sorted(truth.items()):
        got, _ = parser.classify_word_cached(word)
        ok = int(got == gold)
        per.setdefault(gold, [0, 0])
        per[gold][0] += ok
        per[gold][1] += 1
        hits += ok
        total += 1
    return (hits / total if total else 0.0), per


def main(seeds: Sequence[int] = SEEDS) -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    from neural_assemblies.assembly_calculus.emergent.curriculum.data import (
        create_training_sentences,
    )

    truth = truth_map()
    counts = Counter(truth.values())
    majority = max(counts.values()) / sum(counts.values())
    print(f"\n  vocabulary {len(truth)} words: "
          f"{dict(sorted(counts.items(), key=lambda kv: -kv[1]))}")
    print(f"  majority-class accuracy = {majority:.3f}  "
          f"(anything at or below this is not induction)")

    base = create_training_sentences()
    lesioned_sents = strip_grounding(base)
    assert all(not c.is_grounded for s in lesioned_sents for c in s.contexts), (
        "grounding survived the strip -- the whole comparison rests on this"
    )
    # The words themselves must be untouched: a lesion that also changed the
    # token stream would confound "no grounding" with "different corpus".
    assert [s.words for s in base] == [s.words for s in lesioned_sents], (
        "the strip altered the token stream"
    )

    arms: Dict[str, List[float]] = {
        "intact": [], "lesioned": [], "lesioned+dist": [],
    }
    per_cat: Dict[str, Dict[str, List[int]]] = {
        "intact": {}, "lesioned": {},
    }
    for seed in seeds:
        for name, sents, les in (("intact", base, False),
                                 ("lesioned", lesioned_sents, True)):
            parser = train(sents, seed, lesion=les)
            acc, per = score(parser, truth)
            arms[name].append(acc)
            for cat, (h, t) in per.items():
                agg = per_cat[name].setdefault(cat, [0, 0])
                agg[0] += h
                agg[1] += t
            if les:
                # `train()` never calls the distributional path --
                # `_dist_categories` and `_bootstrap_categories` are both empty
                # after it. Without this arm the result would be measuring "the
                # path is not wired in" and reporting "the path cannot do it".
                parser.train_distributional([s.words for s in sents])
                parser._category_cache.clear()
                arms["lesioned+dist"].append(score(parser, truth)[0])

    import statistics

    print(f"\n  {'arm':<12} {'accuracy':>10}")
    for name in ("intact", "lesioned", "lesioned+dist"):
        print(f"  {name:<12} {statistics.mean(arms[name]):>10.3f}")
    print(f"  {'majority':<12} {majority:>10.3f}")
    det_share = counts.get("DET", 0) / sum(counts.values())
    print(f"  {'all-DET':<12} {det_share:>10.3f}"
          "   <- the ungrounded fallback class")

    print(f"\n  per category ({len(seeds)} seeds pooled)")
    print(f"  {'category':<10} {'intact':>10} {'lesioned':>10} {'drop':>8}")
    for cat in sorted(per_cat["intact"], key=lambda c: -counts.get(c, 0)):
        ih, it = per_cat["intact"][cat]
        lh, lt = per_cat["lesioned"].get(cat, [0, 0])
        ia = ih / it if it else 0.0
        la = lh / lt if lt else 0.0
        print(f"  {cat:<10} {ia:>10.3f} {la:>10.3f} {la - ia:>8.3f}")

    les = statistics.mean(arms["lesioned"])
    print("\n  verdicts")
    print(f"    P3 headline: lesioned {les:.3f} vs majority {majority:.3f} "
          f"-> {'ANNOTATION-DRIVEN' if les <= majority + 0.05 else 'some induction survives'}")


if __name__ == "__main__":
    main()

"""Is the generated training corpus grammatical English? Check it, don't eyeball it.

WHY THIS EXISTS. Role induction learns AGENT/PATIENT from noun position relative
to the verb, so the generated corpus IS the supervision. It contained strings
like

    the store build the dog

-- no subject-verb agreement, an object forced onto a verb regardless of
transitivity, and subject/object drawn uniformly over all nouns including
abstract ones. `dog` received its ONLY role binding from that one sentence, as
its object, which is why the curriculum golden asks for dog=AGENT and can never
get it (research/notes/the_curriculum_role_golden_never_measured_the_substrate.md).

THREE CHECKS, each mechanical and each derived from the lexicon rather than from
my judgement about English:

  1. AGREEMENT. A `the <singular noun>` subject requires the 3sg finite form.
     `forms["3sg"]` is present for all 136 verbs, so emitting the bare lemma is
     detectably wrong.
  2. TRANSITIVITY. A verb marked `intransitive` must not be given a direct
     object; one marked `transitive`/`ditransitive` must have one. Both are in
     `features`.
  3. SELECTIONAL FIT. A verb whose first `arguments` entry is `agent` or
     `experiencer` needs an animate subject. `features["animate"]` is on 38
     nouns and `features["abstract"]` on 32.

WHAT THIS FILE IS NOT. It is not a full grammaticality judge -- it cannot catch
semantic oddity ("the dog eats the door" passes all three). It checks exactly
the three properties the generator is now responsible for, which is what makes
a regression here attributable.

THE ARM THAT MATTERS: run against the CURRENT generator and report violations by
class. A count of zero on all three is the claim; anything else is the work
that remains.
"""
import collections
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
os.environ.setdefault("TRAIN_PROGRESS", "0")

from neural_assemblies.assembly_calculus.emergent import EmergentParser  # noqa: E402
from neural_assemblies.assembly_calculus.emergent.curriculum import (   # noqa: E402
    CurriculumTrainer,
)
from neural_assemblies.lexicon.lexicon_manager import WordCategory      # noqa: E402

STAGES = [("TWO_WORD", 3), ("SENTENCES", 4), ("COMPLEX_GRAMMAR", 6)]


def _feat(w):
    return getattr(w, "features", None) or {}


def audit(sentences, words):
    by_lemma = {w.lemma: w for w in words}
    form_to_word = {}
    for w in words:
        for f in (getattr(w, "forms", None) or {}).values():
            if isinstance(f, str):
                form_to_word.setdefault(f, w)

    def lookup(tok):
        return by_lemma.get(tok) or form_to_word.get(tok)

    bad = collections.Counter()
    examples = collections.defaultdict(list)

    for sent in sentences:
        entries = [(t, lookup(t)) for t in sent]
        verbs = [(i, t, w) for i, (t, w) in enumerate(entries)
                 if w is not None and w.category == WordCategory.VERB]
        if not verbs:
            continue
        vi, vtok, vword = verbs[0]
        nouns = [(i, t, w) for i, (t, w) in enumerate(entries)
                 if w is not None and w.category in (WordCategory.NOUN,
                                                     WordCategory.PRONOUN)]
        # A noun governed by a PREPOSITION is not a direct object. Without this
        # the check called "the boy sleeps in the house" an object on an
        # intransitive verb -- 15 false positives, an audit bug rather than a
        # generator bug, and exactly the kind that would have been "fixed" in
        # the generator by making the corpus worse.
        prep_at = {i for i, (t, w) in enumerate(entries)
                   if w is not None and w.category == WordCategory.PREPOSITION}
        governed = {i for i in range(len(entries))
                    if any(pi < i for pi in prep_at)}
        subj = next((x for x in nouns if x[0] < vi), None)
        obj = next((x for x in nouns if x[0] > vi and x[0] not in governed
                    and x[2].category == WordCategory.NOUN), None)
        f = _feat(vword)
        s = " ".join(sent)

        # 1. agreement: past is number-invariant; present is 3sg for a
        # singular subject and the bare lemma for a plural one (surface
        # plural for nouns, `number` feature for pronouns).
        vforms = getattr(vword, "forms", None) or {}
        if subj is not None:
            if subj[2].category == WordCategory.PRONOUN:
                plural = (_feat(subj[2]) or {}).get("number") == "pl"
            else:
                plural = subj[1] == (getattr(subj[2], "forms", None)
                                     or {}).get("plural")
            licensed = {vforms.get("past"),
                        vword.lemma if plural else vforms.get("3sg")}
            licensed = {x for x in licensed if x}
            if licensed and vtok not in licensed:
                bad["agreement"] += 1
                examples["agreement"].append(
                    f"{s}   (licensed {sorted(licensed)})")

        # 2. transitivity
        if f.get("intransitive") and obj is not None:
            bad["object_on_intransitive"] += 1
            examples["object_on_intransitive"].append(s)
        if (f.get("transitive") or f.get("ditransitive")) and obj is None:
            bad["missing_object"] += 1
            examples["missing_object"].append(s)

        # 3. selectional fit
        args = getattr(vword, "arguments", None) or []
        def _animate(w):
            f = _feat(w)
            return bool(f.get("animate")) or (
                bool(f.get("personal"))
                and (f.get("gender") in ("m", "f")
                     or f.get("number") == "pl"))

        if args and args[0] in ("agent", "experiencer") and subj is not None:
            if not _animate(subj[2]):
                bad["inanimate_agent"] += 1
                examples["inanimate_agent"].append(s)
    return bad, examples


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = EmergentParser(n=500, k=20, seed=42)
    trainer = CurriculumTrainer(parser)
    print("Generated-corpus grammaticality, by stage")
    print()
    grand = collections.Counter()
    for stage, complexity in STAGES:
        words = trainer._get_stage_words(stage)
        sents = [p.tokens for p in
                 trainer.generation.generate_generic(words, complexity)]
        bad, ex = audit(sents, words)
        total = len(sents)
        grand.update(bad)
        n_bad = sum(bad.values())
        print(f"  {stage:<18} {total:>4} sentences   violations: {n_bad}")
        for kind, count in sorted(bad.items()):
            print(f"       {kind:<24} {count:>4}")
            for e in ex[kind][:3]:
                print(f"          e.g. {e}")
        if total:
            print(f"       sample: {' '.join(sents[0])}")
            print(f"               {' '.join(sents[min(3, total - 1)])}")
        print()

    print("=" * 66)
    if not grand:
        print("NO VIOLATIONS in agreement, transitivity, or selectional fit.")
        print("That is the whole claim -- semantic oddity is NOT checked, so")
        print("'the dog eats the door' would pass here.")
    else:
        print("VIOLATIONS REMAIN:", dict(grand))


if __name__ == "__main__":
    main()

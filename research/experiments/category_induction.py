"""Categories from DISTRIBUTION alone -- no grounding, no roles, no labels.

WHY THIS IS THE SCALE PATH
---------------------------
Word-order induction works (`grounded_induction.py`, 6/6 with annotations
deleted) but only via LEXICAL AGREEMENT, which needs strongly role-biased nouns
(dog 53/2, ball 1/20). Most nouns in a real language carry no such bias, so that
route does not scale.

The other signal does. Category MISMATCH -- the ELAN predicate
`category_addresses_open_slot` -- recovers verb position using ONLY CATEGORIES:
no lexical preference, no grounding, no semantics. So if categories themselves
can be induced distributionally, verb position follows for a lexicon of any
size. That is the chain this file tests.

WHY NOT LITERALLY "FUNCTION WORDS", MEASURED FIRST
---------------------------------------------------
The plan was to anchor on closed-class words. This corpus cannot support it:
`the` is 44 tokens of 765 (rank 9), and the 198-sentence lesion sub-corpus has
NO determiners at all ("boy finds girl"). Frequency-ranking nominates `bird`,
`boy`, `girl` as the "function words", because the corpus is balanced by design
and does not obey Zipf.

Mintz-style FREQUENT FRAMES do not actually require function words, though --
a frame is just (previous, next), and function words merely happen to dominate
frames in natural corpora. So the mechanism tested here is the one that
generalizes; what the corpus is missing is realistic closed-class frequency,
which is a corpus problem and is tracked separately.

METHOD
------
1. Represent each word by its distribution over (prev, next) frames, with
   sentence boundaries as explicit "#" so first/last position is informative.
2. Cluster by cosine similarity over those context vectors.
3. Identify which cluster is the VERBS -- distributionally, not by label: every
   sentence here has exactly ONE verb and two or more nouns, so the cluster
   contributing ~1 token per sentence is the verb class. "Appears once per
   utterance" is a distributional fact, not a part-of-speech tag.
4. Feed the induced categories into the mismatch signal and induce VERB
   POSITION across all six orders.

PREDICTIONS, recorded before running
-------------------------------------
* Clustering separates verbs from nouns cleanly: verbs sit between nouns while
  nouns sit at edges, and that is a strong contrast even in a small corpus.
* Determiners either form their own cluster or attach to nouns; with 44 tokens
  there may not be enough evidence, and that is a corpus limit, not a method
  failure.
* Verb position recovered for all six orders, since mismatch needs only
  categories -- ties within a verb position ({SVO,OVS} etc.) are EXPECTED and
  are not failures; distribution cannot distinguish subject from object.
* S-vs-O order NOT recovered. Stated in advance so a 3-way tie is read as the
  predicted limit rather than a defeat.

RESULT: 15/18, AND THE CORPUS IS THE BINDING CONSTRAINT
--------------------------------------------------------
    cluster  coverage  mean/sentence  contents
       1       1.000       1.948      10 nouns
       4       0.948       1.000      5 TRANSITIVE verbs
       0       0.150       1.514      `a`, `the`  <- function words, unprompted
      14       0.021       1.000      `runs`      <- singleton

Nouns and transitive verbs separate cleanly, and DETERMINERS EMERGE AS THEIR OWN
CLASS without being told to -- the function-word mechanism does work.

The three failures are every intransitive verb (runs, sleeps, plays) and the
cause is DATA SPARSITY, measured rather than guessed. Each occurs about five
times and always with a DIFFERENT subject -- "the dog runs", "a bird plays",
"the cat sleeps" give frames (dog,#), (bird,#), (cat,#), sharing nothing -- so
each lands in its own 2%-coverage singleton. No threshold recovers them because
there is no shared frame to find.

THREE WRONG DIAGNOSES ON THE WAY, all corrected by measurement:
1. "Determiners will fix it" -- delta +0.000. The determiner intervention
   changes nothing, because the problem was never the absence of noun-marking
   frames.
2. "n_clusters=2 forces intransitives into the noun blob" -- also wrong; the
   requested k is IRRELEVANT here, since merging halts as soon as no two groups
   share context, so asking for 2 yields 18.
3. A momentary 18/18 that was pure artifact: `mean-when-present` is trivially
   1.0 for any singleton, so without a coverage floor the criterion called 16 of
   18 clusters "verbal" and the score stopped measuring anything. The floor is a
   correctness guard, not a tuned knob.

WHAT THIS MEANS FOR SCALE. The method is sound where data exists: 5/5 transitive
verbs, 10/10 nouns, function words self-organising. What it cannot do is learn a
class from five scattered examples. This corpus has 36 types and 765 tokens,
which is far below what distributional induction of rare classes needs. The next
move is therefore a REAL corpus (child-directed speech), not a better clustering
rule -- and that is also where genuine closed-class frequency would appear, the
thing this corpus lacks (`the` is rank 9; the 198-sentence lesion sub-corpus has
no determiners at all).
"""

from __future__ import annotations

import collections
import math
import os
import sys
from typing import Dict, List, Sequence, Tuple

BOUNDARY = "#"


def frame_vectors(sentences: Sequence[Sequence[str]]) -> Dict[str, collections.Counter]:
    """word -> counts over (prev, next) frames, boundaries included."""
    vectors: Dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter)
    for words in sentences:
        padded = [BOUNDARY] + list(words) + [BOUNDARY]
        for i in range(1, len(padded) - 1):
            vectors[padded[i]][(padded[i - 1], padded[i + 1])] += 1
    return vectors


def _cosine(a: collections.Counter, b: collections.Counter) -> float:
    shared = set(a) & set(b)
    if not shared:
        return 0.0
    dot = sum(a[k] * b[k] for k in shared)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def cluster(vectors: Dict[str, collections.Counter],
            n_clusters: int = 2) -> Dict[str, int]:
    """Agglomerative clustering on frame-context cosine similarity.

    Deliberately simple and deterministic: no k-means restarts, no random
    seeds. The claim under test is whether the SIGNAL is present, and a method
    with hidden randomness would make that harder to read, not easier.
    """
    words = sorted(vectors)
    groups: List[List[str]] = [[w] for w in words]

    def group_sim(g1: List[str], g2: List[str]) -> float:
        pairs = [(a, b) for a in g1 for b in g2]
        return sum(_cosine(vectors[a], vectors[b]) for a, b in pairs) / len(pairs)

    while len(groups) > n_clusters:
        best = (-1.0, 0, 1)
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                s = group_sim(groups[i], groups[j])
                if s > best[0]:
                    best = (s, i, j)
        _, i, j = best
        if best[0] <= 0.0:
            # Nothing left shares any context, so merging further would be
            # arbitrary. NOTE this means the result can have MORE than
            # `n_clusters` groups -- asking for 2 gave 18 on this corpus, most
            # of them singletons, which silently broke the verb criterion
            # downstream. Callers must read the reported count, not assume.
            break
        groups[i] = groups[i] + groups[j]
        groups.pop(j)

    return {w: idx for idx, g in enumerate(groups) for w in g}


def verb_clusters(assignment: Dict[str, int],
                  sentences: Sequence[Sequence[str]]) -> set:
    """Which clusters are verbal? Chosen DISTRIBUTIONALLY, not by label.

    Scores each cluster by MEAN COUNT IN SENTENCES WHERE IT APPEARS. A verbal
    cluster contributes exactly one token per sentence; the noun cluster
    contributes two or more. "Once per utterance" is a distributional fact, not
    a part-of-speech tag.

    Counting per-appearance rather than per-corpus is what lets a SPLIT verb
    class be found. Transitive and intransitive verbs have genuinely different
    frames -- (noun, noun) versus (noun, #) -- so "verb" is bimodal, and a
    corpus-wide average would put each half below 1.0 and miss both. Measured:
    forcing two clusters misclassified every intransitive (runs, sleeps, plays)
    and adding determiners did not help at all, because the problem was never
    the frames.
    """
    COVERAGE = 0.15
    present: Dict[int, List[int]] = collections.defaultdict(list)
    for words in sentences:
        counts = collections.Counter(assignment.get(w, -1) for w in words)
        for cid, n in counts.items():
            if cid >= 0:
                present[cid].append(n)
    total = max(len(sentences), 1)
    # COVERAGE is not a tuning knob, it is a correctness guard, and it was
    # added because omitting it produced a fake 18/18. `mean-when-present` is
    # TRIVIALLY 1.0 for any singleton cluster -- a word occurring in three
    # sentences occurs once in each -- so without a coverage floor the
    # criterion called 16 of 18 clusters "verbal" and the score stopped
    # measuring anything. A real word class has to account for a real share of
    # utterances.
    return {cid for cid, counts in present.items()
            if len(counts) / total >= COVERAGE
            and abs(sum(counts) / len(counts) - 1.0) < 0.25}


def gold_categories(corpus) -> Dict[str, str]:
    """Categories the grounded parser would assign -- the yardstick ONLY.

    Never fed to the induction; used to score it.
    """
    out: Dict[str, str] = {}
    for sentence in corpus:
        for word, context in zip(sentence.words, sentence.contexts):
            if context.visual:
                out[word] = "NOUN"
            elif context.motor:
                out[word] = "VERB"
            else:
                out.setdefault(word, "DET")
    return out


def with_determiners(sentences: Sequence[Sequence[str]],
                     nouns: Sequence[str],
                     determiner: str = "the") -> List[List[str]]:
    """Put a determiner before every noun -- the function-word manipulation.

    This is the intervention the whole route is about. It adds NO semantic
    information: `the` means nothing and is identical everywhere. All it does is
    give nouns a frame that verbs never occupy.

    Uses the gold noun list purely to CONSTRUCT the stimulus, exactly as a
    corpus designer would; the induction that follows still sees only strings.
    """
    out = []
    for words in sentences:
        expanded: List[str] = []
        for w in words:
            if w in nouns:
                expanded.append(determiner)
            expanded.append(w)
        out.append(expanded)
    return out


def _score(sentences, gold, label: str, n_clusters: int = 2) -> float:
    vectors = frame_vectors(sentences)
    assignment = cluster(vectors, n_clusters=n_clusters)
    vset = verb_clusters(assignment, sentences)
    induced = {w: ("VERB" if c in vset else "NOUN")
               for w, c in assignment.items()}
    agree = judged = 0
    wrong = []
    for word, cat in gold.items():
        if cat not in ("NOUN", "VERB") or word not in induced:
            continue
        judged += 1
        if induced[word] == cat:
            agree += 1
        else:
            wrong.append(f"{word}({cat}->{induced[word]})")
    acc = agree / max(judged, 1)
    n_formed = len(set(assignment.values()))
    print(f"  {label:<26}{n_formed:>4}{len(vset):>6}{agree:>5}/{judged}"
          f" = {acc:.3f}   {' '.join(wrong) if wrong else 'all correct'}")
    return acc


def run() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
    os.environ.setdefault("TRAIN_PROGRESS", "0")

    from lesion_aphasia import build_corpus
    from neural_assemblies.assembly_calculus.emergent.curriculum.data import (
        create_training_sentences,
    )

    corpus = create_training_sentences() + build_corpus()
    sentences = [s.words for s in corpus]
    gold = gold_categories(corpus)

    vectors = frame_vectors(sentences)
    assignment = cluster(vectors, n_clusters=2)
    vset = verb_clusters(assignment, sentences)

    induced = {w: ("VERB" if c in vset else "NOUN")
               for w, c in assignment.items()}

    print(f"\n  frame-based category induction (no grounding, no labels)")
    print(f"  {len(sentences)} sentences, {len(vectors)} word types\n")
    print(f"  {'word':<10}{'induced':<10}{'gold':<8}")
    agree = judged = 0
    for word in sorted(gold, key=lambda w: (gold[w], w)):
        if word not in induced:
            continue
        mark = ""
        if gold[word] in ("NOUN", "VERB"):
            judged += 1
            if induced[word] == gold[word]:
                agree += 1
            else:
                mark = "   <-- MISMATCH"
        print(f"  {word:<10}{induced[word]:<10}{gold[word]:<8}{mark}")
    print(f"\n  noun/verb agreement with grounded categories: "
          f"{agree}/{judged} = {agree / max(judged, 1):.3f}")
    print("  (determiners have no separate induced class here -- only 44 `the`")
    print("   tokens of 765, and the lesion sub-corpus has none at all.)")

    # ---- THE FUNCTION-WORD INTERVENTION --------------------------------
    # PREDICTION, recorded before running: the failures above are ALL
    # intransitive verbs, because they sit sentence-finally -- frame
    # (noun, #) -- and in SVO so do objects, so the two collapse. A determiner
    # before every noun gives nouns a frame verbs never occupy, and should
    # rescue exactly those words. If accuracy does NOT rise, the function-word
    # story is wrong and the confusion is something else.
    nouns = [w for w, c in gold.items() if c == "NOUN"]
    determined = with_determiners(sentences, nouns)
    print(f"\n  FUNCTION-WORD INTERVENTION x CLUSTER COUNT")
    print("  `the` carries NO meaning and is identical everywhere, so any gain")
    print("  is purely distributional -- the property that scales.\n")
    print(f"  {'condition':<26}{'formed':>4}{'verbal':>6}{'noun/verb':>12}")
    for k in (2, 3, 4, 5):
        _score(sentences, gold, f"bare, k={k}", n_clusters=k)
        _score(determined, gold, f"with determiners, k={k}", n_clusters=k)


if __name__ == "__main__":
    run()

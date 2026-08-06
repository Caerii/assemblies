# The ERP calibration frames are made of words the parser never trained

**On the parser every ERP result in this repo is measured on, 9 of the 11
calibration-frame words have no core lexicon entry.** Only `the` and `dog` are
trained. Three of the nine (`bird`, `finds`, `small`) are declared holdouts and
are *supposed* to be untrained. The other six — `cat`, `chases`, `sees`, `she`,
`runs`, `eats` — are simply absent from training, and nothing anywhere raises.

Measured: `research/experiments/erp_frame_vocabulary_audit.py`, output in
`research/experiments/frame_vocab_audit.log`.

```
depth=SENTENCES seed=11
registered stimuli: 517   words with a core assembly: 123
--> 9/9 DEFAULT items contain a word that is neither trained nor a holdout
--> 9/9 AREA_MATCHED items likewise
--> 6/6 SWEEP items likewise
```

## The worst single consequence

`chases` is the MAIN VERB of five of the nine default frames, and the parser
classifies it **NOUN**.

```
grammatical  "the dog chases cat"     -> DET NOUN NOUN NOUN
grammatical  "she chases the cat"     -> NOUN NOUN DET NOUN
violation    "the dog chases finds"   -> DET NOUN NOUN VERB
violation    "she chases the eats"    -> NOUN NOUN DET VERB
```

Those items contain no verb at all. So the contrast they implement is *four
nouns* against *three nouns and a verb* — not "a well-formed transitive
sentence" against "one with a verb in object position". `small`, the whole
point of the attributive-adjective item, classifies **VERB**.

`sees` also never trained, but it does classify VERB, so items 2 of each arm do
have a main verb. **The arms are internally inconsistent**: item 2 parses with a
verb, items 1 and 3 do not.

## This is the `hits` defect, and the fix was verified against the wrong corpus

`frames.py` already carries this lesson, written for exactly one word:

> `hits` is in no curriculum sentence and no holdout — it occurred ONLY in this
> file — so the parser categorised it UNKNOWN and that item's MAIN VERB was
> unrecognised before its critical word was ever reached. A third of the
> category-violation arm was therefore not a category violation.

The replacement was `chases`, justified as "trained (16 curriculum
occurrences)". That count is real — in `create_training_sentences()`, the
**FULL_TRAIN** corpus. Every ERP number here is measured on
`get_parser_cache().fork("SENTENCES")`, which `CurriculumTrainer` builds from
the CDS corpus, where:

| word | occurrences in the SENTENCES corpus |
|---|---|
| `chases` | **0** |
| `sees` | **0** |
| `eats` | **0** |
| `finds` | **0** |
| `small` | **0** |
| `runs` | 1 |
| `cat` | 5 (in the corpus, still never reaches a core lexicon) |
| `dog` | 11 |

Diagnosing a word against one corpus and measuring it on another is the same
mistake as [[backbone-fingerprint-gap]]: the artefact was reproducible and the
attribution was still wrong.

## The guard tests the wrong predicate

`collect_frame_samples` keeps a word when `w in parser.stim_map`. That is
**registered**, not **trained** — 517 stimuli against 123 words with a core
assembly. A registered-but-untrained word passes the filter, receives a phon
stimulus, gets a category from the distributional classifier, and returns a
number.

This is [[same-name-two-meanings]] in its usual costume: `stim_map` membership
means "this word has a stimulus", and the frame code reads it as "the parser
knows this word". Both readings run. Only one is true.

## What this does and does not overturn

It does **not** make the arithmetic wrong. p600_auc 0.7167 is a real, replicated
difference between two stimulus sets.

It does mean **the sets are not the ones the labels describe.** The sentence
"the parser produces a P600 to a category violation in an otherwise grammatical
sentence" is not supported by these items, because in four of six of them the
control is not grammatical and the violation is not a violation of anything the
parser parsed.

## The fix

Frames must be **derived from the parser's trained vocabulary**, not
hand-authored against a remembered corpus, and a guard must fail when they are
not. Editing the frames is the contained fix; adding vocabulary to the corpus
moves the substrate under every result in the repo — the same trade `frames.py`
already reasoned through for `hits`, and the right call then as now.

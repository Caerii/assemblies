# PREREG: a corpus in which history is worth something

Registered 2026-09-09, with the acceptance criterion stated before the
first corpus was generated.

The study-4 corpus behind A3 has an oracle gap of 0.019 MRR
(PREREG_seq_a3_transducer.md, Amendment 3): a state that knew the
generating grammar's phase would beat a bigram by that much, so no state
effect larger than that can be measured on it. Any study of state
induction needs a corpus in which a bigram is wrong for a reason a state
can fix.

## Acceptance criterion

A corpus is accepted when the paired difference between an oracle state
(the generator's own hidden state, estimated from the seed's 200 training
sentences) and a bigram, both scored in the study's units (MRR of the
true next word, random tie-break, 25 test sentences per seed, 20 seeds),
has a lower 95% bound of at least 0.10. The oracle gap is computed by
`research/experiments/study4/ntp_agree.py`.

## Two candidates

**Template with agreement.** DET (ADJ) NOUN [PREP DET (ADJ) NOUN] VERB
DET (ADJ) NOUN TAG, the verb and tag agreeing in number with the subject
noun, the prepositional phrase's noun of independent number. A bigram at
the verb sees the wrong noun half the time when the phrase is present.

**Chain.** AUX NOUN VERB NOUN PRON NOUN TAG, every other word agreeing
with a number drawn once per sentence, the nouns between of independent
number. Four agreement sites behind distractors in seven positions.

The bound on the gap is arithmetic: at a site behind a distractor a
bigram ranks the wrong two-word class first half the time, so the site
gains at most 0.25 MRR, and the corpus gap is at most 0.25 times the
fraction of positions that are such sites. The template has two sites in
about nine positions and cannot meet the bar; the chain has four in seven.

## Result (2026-09-09, 20 seeds)

    corpus      unigram   bigram   phase oracle   phase+number oracle   oracle - bigram (paired)
    template    0.163     0.291    0.319          0.366                 +0.075 +/- 0.009   NOT accepted
    chain       0.138     0.176    0.302          0.428                 +0.253 +/- 0.009   ACCEPTED

The chain is the corpus for PREREG_successor_state.md. It is a probe, not
English: what it isolates is whether an induced state can carry one bit
across a distractor, which is the capacity the template also needs and
cannot measure.

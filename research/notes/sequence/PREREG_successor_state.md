# PREREG: a state that merges prefixes by what they predict

Registered 2026-09-09, before implementing or running.

## The problem

The transducer's induced state is distinct, deterministic and carries
nothing the current word does not (PREREG_seq_a3_transducer.md). The state
area is written by the arc alone, so every prefix gets its own code and
no two prefixes with the same future share one. The assigned-state
machine works because its code is given and shared. Nothing in the
substrate merges contexts by what they predict.

## The construction

Teacher-force the STATE area toward the groundings of the next h words.
During training, at step t, the state's projection receives the arc and,
as additional afferents, the grounding stimuli of words t+1 .. t+h into
the state area (one stacked stimulus per offset). The k-WTA then selects
a state dominated by the successor codes, and arc -> state learns to map
the arc onto it. At test the state is produced by arc -> state alone.
Two prefixes with the same next h words are forced toward the same state
code: that is the merge. The offset h is the only new parameter; h = 0 is
the registered transducer.

Why it can carry a bit across a distractor. On the chain corpus
(PREREG_agreement_corpus.md) the word two ahead of an agreeing word is the
next agreeing word, which carries the subject's number. With h = 2 the
state after an agreeing word is forced toward a code that contains that
number; the arc at the distractor position conjoins the distractor with
that state; OUT is read from the arc; so the number reaches the site. With
h = 1 the state after an agreeing word is forced toward the distractor's
code, which carries no number, and the construction should not help.

## Cells

Hashed transducer, the chain corpus, 20 seeds (42..61), each seed its
own 200/25 corpora, n = 10,000, n_arc = 10,000, k = 200, organ_p = 0.2,
beta = 0.1, arc strength 0.1, three write rounds; h in {0, 1, 2}, and the
state-blind audit at h = 2. Baselines per seed: the bigram and the
phase+number oracle from `ntp_agree.oracle_gap`.

    SR-0  h = 0 reproduces the null: MRR within 0.03 of the seed's bigram
          (paired), and the state-blind delta's interval contains 0.
          Reported as the anchor.
    SR-1  h = 1 does not help: MRR - bigram lower bound < 0.05.
          PREDICTION: PASSES (the forced code carries no number).
    SR-2  h = 2 closes at least 40% of the oracle gap: MRR - bigram
          (paired) lower bound >= 0.10 against a gap of 0.25.
          PREDICTION: PASSES. This is the bar the construction lives on.
    SR-3  the state is informative at h = 2: full minus state-blind, paired,
          lower bound > 0.
    SR-4  the state does not collapse at h = 2: cross-prefix state overlap
          upper bound < 0.5 (H4 restated).

Adoption if SR-2 and SR-3 pass: a register entry stating that a state
teacher-forced toward its successors carries a feature across a
distractor, with h as the horizon, and the mechanism named as merging by
predicted future. FAIL of SR-2 with SR-1 passing: the construction is
sound in principle and the arc -> state map does not learn the copy;
report the site accuracy and the state overlap and stop. FAIL of SR-1
(h = 1 helps): the account of why is wrong; report before anything else.

## Amendment 1 (2026-09-09, before the full run): gap 2, and a forcing gain

A three-seed API smoke on the gap-1 chain (numbers void by rule) showed
two things that change the design, not the question. First, at horizon 0
the induced state already carries the number across one distractor, so
SR-0 as written cannot hold there; the corpus moves to gap 2
(PREREG_agreement_corpus.md, Amendment 1), where the induced state must
carry the bit across two steps. Second, full-strength forcing appeared to
replace the arc's induced content in the state rather than add to it,
so the forcing gets a gain: the successor stimuli's drive scaled by
`successor_gain` (1.0 is a full stimulus; 0.3 leaves the arc's drive in
charge with the successors as a bias).

Cells, gap 2, 20 seeds: horizons {0, 1, 3} at gain 1.0; horizons {0, 3}
at gain 0.3; the state-blind audit at h = 3 and at h = 0.

    SR-0'  h = 0 at gap 2: MRR - bigram and the state-blind delta, REPORTED
           (the induced state's own carry across two steps; no bar).
    SR-1'  h = 1 does not help: MRR(h=1) - MRR(h=0) paired, upper bound
           <= 0.02.  PREDICTION: PASSES.
    SR-2'  h = 3 closes the gap: at one of the two gains, MRR(h=3) - bigram
           lower bound >= 0.085 (40% of 0.214) AND MRR(h=3) - MRR(h=0) paired
           lower bound > 0.  PREDICTION: PASSES at gain 0.3; uncertain at 1.0.
    SR-3'  state informative at h = 3: full minus blind, lower bound > 0.
    SR-4'  no collapse at h = 3: cross-prefix state overlap upper bound < 0.5.

SR-2' is the bar the construction lives on; the h = 0 arm is what it
must beat, since the induced state is not the null on this corpus.

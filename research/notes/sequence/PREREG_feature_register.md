# PREREG: a feature register, the structured slot

Registered 2026-09-09, before implementing or running.

## Why

Two learning-rule constructions failed to carry one bit across two
distractors (PREREG_successor_state.md; PREREG_temporal_memory.md, cells
A pending), and the induced state, which predicts memorized sequences at
unbounded order, carries 0.011 of a 0.214 gap on unseen combinations. The
literature's local models are episodic in the same way. The language-organ
line has always taken the other route: structure supplies what learning
cannot, as roles, categories and grounded features. The intuition to
follow is that agreement is a REGISTER problem: one slot that holds the
subject's number until it is overwritten, written only by words that
carry that feature in the relevant role, and read by the arc.

## The construction

A REGISTER area `REG` (n_reg = n, k) with two feature stimuli, `feat_sg`
and `feat_pl`, and a fiber `REG -> ARC` at the organ's density, learned
Hebbian prev x new like the other organ fibers. A table `feature_of`
maps each word to a feature (sg, pl) or to none. On a tick, if the
current word has a feature, `REG` is projected from that feature's
stimulus (one round) and holds the result; if it has none, `REG` keeps
its winners. The arc is then the conjunction of LEX, STATE and REG. OUT
reads the arc and is teacher-forced as before. Nothing else changes.

The GATE is the table: which words write the register. On the chain
corpus the agreeing classes (AUX, VERB, PRON, TAG) write it and the
distractor nouns do not. This gate is supplied by structure, the way a
parser supplies a role; whether it can be learned is the next question,
not this one.

## Cells and bars

Chain corpus, gap 2 (PREREG_agreement_corpus.md, Amendment 1; oracle -
bigram 0.214), 20 seeds, n = n_arc = n_reg = 10,000, k = 200, the
registered transducer (induced state) plus the register.

    FR-1  GATED REGISTER closes the gap: MRR - bigram (paired) lower
          bound >= 0.15 (70% of 0.214).  PREDICTION: PASSES. The register
          is the oracle's number bit; the phase is in the word and state.
    FR-2  THE REGISTER CARRIES IT, not the state: with the state area held
          empty (state-blind) and the register kept, MRR - bigram lower
          bound >= 0.15.  PREDICTION: PASSES.
    FR-3  REGISTER-BLIND falls back: with the register held empty at test,
          MRR - bigram upper bound <= 0.05.  PREDICTION: PASSES.
    FR-4  THE GATE DOES THE WORK: an UNGATED register, written by every
          word that has a number (nouns included), MRR - bigram upper
          bound <= 0.05 -- the distractor overwrites the subject.
          PREDICTION: PASSES.
    FR-5  reported: agreement-site accuracy (the correct number class
          ranks first at AUX/VERB/PRON/TAG positions), gated vs ungated.

Adoption if FR-1, FR-3 and FR-4 pass: a register entry that a structured
slot -- one feature area, category-gated writes, read by the conjunction
-- carries an agreement feature across distractors on the assembly
substrate where three learning-rule constructions did not, and that the
gate, not the memory, is what the substrate lacks. FR-1 passing with FR-4
failing would mean the arc reads the register's LAST write regardless of
gating, which would be a different mechanism and is reported as such.

## Result (2026-09-09, chain corpus gap 2, seeds 42..61, 20 brains, n = n_arc = n_reg = 10,000, k = 200)

Run: `seq_a3_transducer.py --engine hashed --register --seeds 20`; results
in `research/results/sequence/seq_a3_transducer_results_register_chain_gap2.json`,
log in `research/results/logs/seq_a3_register_chain_gap2.log`; run from a
worktree pinned at 62ac158 (the int8 build).

| Arm | MRR | MRR - bigram (paired) | bar | verdict |
|-----|-----|-----------------------|-----|---------|
| bigram | 0.1221 +/- 0.0043 | | | |
| oracle | 0.3360 +/- 0.0061 | (gap 0.214) | | |
| gated register | 0.2398 +/- 0.0108 | +0.1177 +/- 0.0108, lower bound 0.107 | FR-1 lower bound >= 0.15 | FAIL |
| gated, state held empty | 0.1632 +/- 0.0075 | +0.0410 +/- 0.0082 | FR-2 lower bound >= 0.15 | FAIL |
| gated, register held empty | 0.1333 +/- 0.0049 | +0.0111 +/- 0.0047, upper bound 0.016 | FR-3 upper bound <= 0.05 | PASS |
| ungated register | 0.1220 +/- 0.0040 | -0.0001 +/- 0.0034 | FR-4 upper bound <= 0.05 | PASS |

FR-5 (agreement-site accuracy) was not produced: the harness reports MRR
only. It is owed if this line is reopened.

**Not adopted.** Adoption needed FR-1, FR-3 and FR-4; FR-1 fails. The
gated register carries 55 percent of the oracle gap, not the 70 registered,
and it does not carry it alone: with the state area empty the gain falls
to 0.041, with the register empty to 0.011, so the register and the
induced state are complementary (the register holds the number, the state
the phase), which is the opposite of FR-2's prediction that the register
would suffice. FR-3 and FR-4 hold as predicted: without the register the
transducer is the registered null (+0.011, the A3 result), and an ungated
register that every noun overwrites carries nothing (-0.000). The gate
does the work; the memory without the gate is worthless here.

**TM-10, side by side.** On the same corpus and seeds the local-rule
temporal memory (`SEQ-TEMPORAL-CARRY`, copy state with predicted-win, no
gate supplied by structure) gives +0.148; the hand-gated structured slot
gives +0.118. The learning-rule construction beats the slot that was built
to show what the substrate lacked. The register's remaining case is a
feature the arc chain cannot carry across a long gap; that has not been
measured (the temporal memory's decay law has two points, 72 and 60
percent at gaps 2 and 3).

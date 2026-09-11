# Temporal arc position-pooling audit

Code inspected at f662e09 on 2026-09-10, before any new temporal GPU run.
SEQ-TEMPORAL-CARRY's TM-9 mechanism interpretation is suspended. The independently
collected prediction MRR and high-order task observations are not changed by this
audit. The historical 0.111/0.220 contrasts are not valid distractor-specific data.

## Code path and counterexample

seq_a3_transducer.a3_hashed collects every processed token's arc under its position.
_distractor_overlaps excludes position0 only, then averages all remaining pairs
using subject labels from position0. It never checks noun/distractor identity.
ntp_agree.generate_chain interleaves NOUN distractors with AUX/VERB/PRON/TAG words
whose number is explicitly the subject's number. The processed noninitial positions
therefore include directly informative words, not just distractors. Equal row counts
also stand in for sentence alignment, and unequal counts are silently omitted.

Construct four sentences, two per subject number. Give every distractor the same
arc, so its same-number minus different-number overlap is0. Give agreement-word
arcs disjoint number-specific sets. The historical two-position pool reports contrast
0.5 although the distractor contrast is0. This proves the metric can pass a mechanism
bar without any distractor representation. It does not determine the true corrected
contrast in the recorded brains.

## Required correction

Keep historical arithmetic explicitly named for audit, but refuse the former
_distractor_overlaps path and collect_arcs requests before GPU construction. Do not
silently write corrected values under the old TM-9 protocol. A replacement must
retain sentence identity, token, subject label, position and arc IDs; select actual
distractor tokens; report each position separately; reject dropped/misaligned rows;
and separate readout fitting from held-out evaluation. Register its controls/bars
before measuring. The old serialized mechanism summaries cannot resolve this
selection error into a corrected distractor-only number.

The performance observation still motivates representation-versus-readout research.
The prior statement that g=0 already carries a0.11 distractor contrast does not.
No new scientific outcome is adopted from the constructed software counterexample.

"""The double dissociation again -- with BOTH lesions structural this time.

WHY REDO A RESULT THAT ALREADY HELD
------------------------------------
`lesion_aphasia.py` established a double dissociation over 20 seeds with tight
intervals, but both of its lesions are symbolic, and both docstrings say so:

* `lesion_lexical` zeroes core->ROLE synapses AND clears `role_lexicons`, and
  `decompose()` showed the DICT CLEAR is what produces the effect -- synaptic
  damage alone, even at 99%, cost almost nothing (0.93 -> 0.47 at 100%).
* `lesion_positional` rebinds `word_order_type` to return a corrupted string on
  a fraction of reads: "a model of degraded syntactic processing, not of
  tissue".

So the aphasia model rested on damaging Python attributes. Both routes are now
NEURAL (`nemo_competitive_ab.py`): word order is a gating program and lexical
preference is mutual inhibition arbitrating on learned weight. That makes both
lesions expressible as damage to the machine rather than to bookkeeping.

    LEXICAL lesion   zero the core->ROLE synapses. NOTHING ELSE. No dict is
                     touched, because the competitive parser has no dict in
                     the decision path -- MI compares synaptic drive.
    POSITIONAL lesion remove the verb's slot-advance rules, so the gating never
                     switches AGENT->PATIENT. A structural lesion of the
                     mechanism that encodes word order, not of a stored string.

PREDICTIONS, recorded before running
-------------------------------------
1. Competitive parser + SYNAPSE-ONLY lexical lesion: irreversible collapses
   from 1.000. This is the falsifiable one. In the symbolic parser synapses
   alone were nearly harmless; if they are still nearly harmless here, then MI
   is not really deciding on learned weight and the constructive result is
   weaker than it looks.
2. Gating parser + slot-advance lesion: reversible collapses from 1.000,
   because both nouns now meet an open AGENT slot and the second can no longer
   be forced into PATIENT. Irreversible is already 0.000 for gating, so it has
   nowhere to fall -- that arm is uninformative BY CONSTRUCTION and is reported
   only to make the asymmetry visible.
3. Crossed control: the lexical lesion should NOT hurt gating's reversible
   score. Gating decides by which slot is open and never consults weight, so
   this is the cell that makes it a DOUBLE dissociation rather than "damage
   makes things worse". This one is a real empirical test.

The fourth cell is NOT a control, and saying so matters. "competitive +
positional lesion" must come out IDENTICAL to "competitive intact", but only
because competitive mode calls `competitive_verb_program` directly and never
touches the patched `trans_verb_program` -- the lesion cannot reach it. That is
a mechanistic fact, not evidence. It is reported because it is a useful
LEAK DETECTOR: if that row moves at all, the monkeypatch is escaping its scope
and every other number in the table is suspect.

The deeper reason there is no positional lesion for the competitive route is
itself the dissociation: competitive mode HAS no positional mechanism to
damage. It decides on learned weight alone.

If 1 and 3 hold, the aphasia model is mechanistic end to end.

RESULT: 1 CONFIRMED, 3 REFUTED -- the routes share an OUTPUT PATHWAY
--------------------------------------------------------------------
    condition                          reversible   irreversible
    gating      intact                     1.000        0.000
    gating      + lexical lesion           0.000        0.000
    gating      + positional lesion        0.500        0.000
    competitive intact                     0.667        1.000
    competitive + lexical lesion           0.000        0.000
    competitive + positional (leak check)  0.667        1.000

**PREDICTION 1 CONFIRMED, and it is the headline.** A SYNAPSE-ONLY lesion --
no dict cleared anywhere -- collapses competitive's irreversible score from
1.000 to 0.000. The same synaptic damage in the symbolic parser cost almost
nothing (`decompose()`: even 99% was nearly harmless, and clearing
`role_lexicons` was necessary AND sufficient). So the lexical route really is
carried by learned weights now, which is what the constructive result claimed.

**PREDICTION 3 REFUTED, for a reason worth more than the prediction.** Gating's
reversible score does NOT survive the lexical lesion -- it drops to 0.000, and
the leak check rules out patch escape. Zeroing core->ROLE synapses removes the
ability to bind ANY word into ANY role. Gating does not consult those weights to
MAKE its decision (it decides by which slot is open), but it needs them to
EXPRESS it, and the readout matches formed assemblies against stored ones
through the same synapses.

**So the two routes dissociate in their DECISION MECHANISM but CONVERGE on a
shared EXPRESSION pathway**, and no purely synaptic lesion can be selective. The
positional half IS selective (0.500 for gating, competitive untouched), so what
survives is a SINGLE dissociation plus the synaptic-lexical confirmation, not
the clean double dissociation the symbolic study reported.

Do not "fix" this by inventing a lesion that spares the readout: there isn't
one. Any damage to the core->ROLE weights degrades binding and readout together,
because they are the same synapses. That is an architectural fact about this
model, and arguably the more honest neuropsychology -- a double dissociation
requires separable pathways, and here they converge.
"""

from __future__ import annotations

import copy
import os
import sys
from typing import Sequence

SEEDS = tuple(range(42, 52))


def run(seeds: Sequence[int] = SEEDS) -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
    os.environ.setdefault("TRAIN_PROGRESS", "0")

    from lesion_aphasia import (
        build_corpus, test_items, train_parser, zero_role_synapses,
    )
    from nemo_vs_symbolic import _mean_ci, _score
    from neural_assemblies.assembly_calculus.emergent import nemo_rules
    from neural_assemblies.assembly_calculus.emergent.curriculum.data import (
        create_training_sentences,
    )
    from neural_assemblies.assembly_calculus.emergent.nemo_parse import (
        NemoParser, infer_transitive_verbs,
    )
    from neural_assemblies.assembly_calculus.emergent.nemo_rules import (
        RuleProgram, trans_verb_program,
    )
    from neural_assemblies.core.inhibition import DISINHIBIT, INHIBIT, fiber_rule

    transitive = infer_transitive_verbs(create_training_sentences() + build_corpus())
    kinds = {
        kind: [(list(words), gold)
               for words, gold, k in test_items() if k == kind]
        for kind in ("reversible", "irreversible")
    }

    def no_advance_verb_program(core_area):
        """The verb binds its action but NEVER advances the slot.

        `trans_verb_program`'s POST rules -- DISINHIBIT ROLE_PATIENT and
        INHIBIT ROLE_AGENT -- ARE the word-order mechanism. Dropping exactly
        those two, and keeping everything else, is the structural counterpart
        of corrupting `word_order_type`: the machine still binds verbs, it has
        just lost the ability to move the open slot.
        """
        intact = trans_verb_program(core_area)
        return RuleProgram(
            pre=list(intact.pre),
            post=[fiber_rule(INHIBIT, core_area, nemo_rules.ROLE_ACTION, 0)],
        )

    def parse_with(parser, words, *, competitive, positional_lesion):
        """One parse, optionally with the slot-advance removed.

        The lesion is injected by rebinding `nemo_rules.trans_verb_program`,
        because `program_for_category` looks it up in that module's globals at
        call time. Restored in `finally` -- a leaked patch would silently
        lesion every later condition, which is exactly the contamination class
        that invalidated the ERP harness.
        """
        original = nemo_rules.trans_verb_program
        if positional_lesion:
            nemo_rules.trans_verb_program = no_advance_verb_program
        try:
            return NemoParser(parser, transitive_verbs=transitive,
                              competitive=competitive).parse(list(words))
        finally:
            nemo_rules.trans_verb_program = original

    CONDITIONS = [
        ("gating      intact", False, False, False),
        ("gating      + lexical lesion", False, True, False),
        ("gating      + positional lesion", False, False, True),
        ("competitive intact", True, False, False),
        ("competitive + lexical lesion", True, True, False),
        # Leak detector, not a control -- see the module docstring. Competitive
        # mode never calls the patched `trans_verb_program`, so this MUST equal
        # "competitive intact"; if it does not, the patch is escaping.
        ("competitive + positional (leak check)", True, False, True),
    ]

    print(f"\n  {len(seeds)} seeds, paired.  BOTH lesions structural:")
    print("  lexical = zero core->ROLE synapses (NO dict cleared)")
    print("  positional = verb loses its slot-advance rules\n")
    print(f"  {'condition':<34}{'reversible':>18}{'irreversible':>18}")

    scores = {(name, kind): [] for name, *_ in CONDITIONS for kind in kinds}
    for seed in seeds:
        trained = train_parser(seed)
        for name, competitive, lex_lesion, pos_lesion in CONDITIONS:
            for kind, rows in kinds.items():
                parser = copy.deepcopy(trained)
                if lex_lesion:
                    # SYNAPSES ONLY. `clear_role_lexicons` is deliberately NOT
                    # called -- that dict is what carried the old result, and
                    # the whole question is whether the neural route stands
                    # without it.
                    zero_role_synapses(parser, 1.0, seed)
                ok = total = 0
                for words, gold in rows:
                    pred = parse_with(parser, words, competitive=competitive,
                                      positional_lesion=pos_lesion)
                    a, b = _score(pred, gold)
                    ok += a
                    total += b
                scores[(name, kind)].append(ok / max(total, 1))

    for name, *_ in CONDITIONS:
        cells = "".join(
            f"{m:.3f} +/-{h:.3f}".rjust(18)
            for m, h in (_mean_ci(scores[(name, kind)]) for kind in kinds)
        )
        print(f"  {name:<34}{cells}", flush=True)

    print("\n  DOUBLE DISSOCIATION requires all four:")
    print("    competitive irreversible: intact HIGH, lexical lesion LOW")
    print("    gating      reversible:   intact HIGH, positional lesion LOW")
    print("    competitive irreversible survives the POSITIONAL lesion")
    print("    gating      reversible   survives the LEXICAL lesion")


if __name__ == "__main__":
    run()

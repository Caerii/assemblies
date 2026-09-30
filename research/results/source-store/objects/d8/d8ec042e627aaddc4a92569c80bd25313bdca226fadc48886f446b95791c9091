"""Morph-feature recall: the readout the detect/train pairs never had.

Pinned from task #129 (`research/experiments/what_variation_buys.py`,
`research/notes/language/what_the_variation_bought.md`). What is asserted here is
chosen by MARGIN, per the paper-parity process:

  * WIRING: the "tense"/"number" phases are scheduled ("number" was the
    `phrases` dormant-selector shape -- schedulable, listed by no stage) and
    `detect_number` sees plurality at all (its grounding-feature path read
    "SG" for every token ever generated).
  * READOUT HEALTH: image separation is far from the degenerate 1.0 the
    ablated null reads, and recall answers rather than ties.
  * SEED-42 ACCURACIES with headroom (training is bit-identical across
    processes since the one-seeding-path fix, so these are stable, and
    the floors leave room for environment drift).

FLOORS RE-PINNED at the #149 adoption of split_feature_areas=True (the
n=10 paired gate: balanced tense delta -0.039 +/- 0.058 NS, SG 0.920,
PL 0.415 vs 0.085 shared). Under the split default at seed 42: PAST
0.565 / PRES 0.760 / SG 1.000 / PL 0.231. The old shared-path floors
(PAST 0.826-based) live on in the history of this file; PAST's paired
trend under the split is the named cost (-0.04 balanced, not
significant at n=10) bought for PL leaving the floor.

PL IS NOW PINNED -- weakly, deliberately. Shared-path PL measured
0.0-0.2 (chance; Hebbian mass follows token frequency and the frequent
class swamps the rare one). The split architecture takes seed-42 PL to
0.231 (n=10 mean 0.415): the first configuration where PL reads above
its floor at the DEFAULT corpus. The floor 0.15 asserts exactly
"no longer the always-SG collapse", nothing stronger -- scale claims
live in the E-series experiments.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
os.environ.setdefault("TRAIN_PROGRESS", "0")

from neural_assemblies.assembly_calculus.emergent import EmergentParser
from neural_assemblies.assembly_calculus.emergent.curriculum import (
    CurriculumTrainer,
)
from neural_assemblies.assembly_calculus.emergent.curriculum.trainer import (
    _STAGE_CONFIG,
)


def _attested_forms(parser):
    """Corpus-attested unambiguous verb forms via the canonical builder
    (`evaluation/morph_features.py` -- promoted there after three copies
    of this scan existed within a day)."""
    from neural_assemblies.assembly_calculus.emergent.evaluation \
        .morph_features import attested_morph_sets

    sets_ = attested_morph_sets(parser)
    return sets_["PAST"], sets_["PRESENT"]


def test_number_phase_is_scheduled():
    """"number" was schedulable but listed by NO stage -- the dormant
    selector that made train_number a phase that never ran."""
    for stage in ("SENTENCES", "COMPLEX_GRAMMAR", "CONVERSATION"):
        assert "number" in _STAGE_CONFIG[stage]["phases"], stage
        assert "tense" in _STAGE_CONFIG[stage]["phases"], stage


@pytest.mark.slow
class TestMorphRecall:

    @pytest.fixture(scope="class")
    def trained(self):
        from neural_assemblies.assembly_calculus.emergent.vocabulary_builder \
            import build_vocabulary_preset

        parser = EmergentParser(n=3000, k=30, seed=42,
                                vocabulary=build_vocabulary_preset("core"),
                                fast_training=True)
        trainer = CurriculumTrainer(parser)
        for stage in ("FIRST_WORDS", "VOCABULARY_SPURT", "TWO_WORD",
                      "SENTENCES"):
            trainer.train_stage(stage)
        return parser

    def test_detect_number_sees_plural_forms(self, trained):
        """The teacher, not the substrate: plurality resolves from lexicon
        forms. Before this, plural surfaces inherited the lemma's grounding
        verbatim and the grounding-feature path said SG for everything."""
        assert trained.detect_number("dogs") == "PL"
        assert trained.detect_number("dog") == "SG"

    def test_recall_tense_reads_the_trained_contrast(self, trained):
        """Form->tense association is recallable well above the ablated
        null (which reads ALL ties: identical images, separation 1.0)."""
        past, pres = _attested_forms(trained)
        assert len(past) >= 10 and len(pres) >= 10, (len(past), len(pres))
        seps = []

        def acc(items, label):
            ok = 0
            for w in items:
                got, diag = trained.recall_tense(w)
                if diag["image_separation"] is not None:
                    seps.append(diag["image_separation"])
                ok += got == label
            return ok / len(items)

        past_acc = acc(past, "PAST")
        pres_acc = acc(pres, "PRESENT")
        assert past_acc >= 0.45, (
            f"PAST {past_acc:.3f} (split default measured 0.565 at seed 42; "
            f"shared path measured 0.826 -- the named #149 cost)")
        assert pres_acc >= 0.60, f"PRES {pres_acc:.3f} (measured 0.760)"
        assert seps and max(seps) < 0.5, (
            f"image separation {max(seps) if seps else None} -- images "
            f"merging is the ablated-null signature (1.0)")

    def test_recall_number_sg_reads(self, trained):
        """The frequent class reads cleanly (measured 1.000 at seed 42
        under the split default)."""
        from neural_assemblies.lexicon.data import NOUNS

        sg = [e["lemma"] for e in NOUNS
              if e.get("forms", {}).get("plural")
              and e["lemma"] in trained.stim_map][:15]
        assert len(sg) >= 10
        ok = sum(trained.recall_number(w)[0] == "SG" for w in sg)
        assert ok / len(sg) >= 0.8, f"SG {ok}/{len(sg)}"

    def test_recall_number_pl_off_the_floor(self, trained):
        """PL is no longer the always-SG collapse (#149 gate: split PL
        0.415 +/- n=10, seed 42 = 0.231, vs 0.085 shared). The floor 0.15
        asserts exactly that escape and nothing stronger -- see module
        docstring."""
        from neural_assemblies.lexicon.data import NOUNS

        pl = [e["forms"]["plural"] for e in NOUNS
              if e.get("forms", {}).get("plural")
              and e["forms"]["plural"] in trained.stim_map][:15]
        assert len(pl) >= 10
        ok = sum(trained.recall_number(w)[0] == "PL" for w in pl)
        assert ok / len(pl) >= 0.15, f"PL {ok}/{len(pl)}"

    def test_recall_unknown_word_returns_none(self, trained):
        got, diag = trained.recall_tense("zzz-not-a-word")
        assert got is None
        assert diag["scores"] == {}

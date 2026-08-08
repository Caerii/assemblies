"""Morph-feature recall: the readout the detect/train pairs never had.

Pinned from task #129 (`research/experiments/what_variation_buys.py`,
`research/notes/what_the_variation_bought.md`). What is asserted here is
chosen by MARGIN, per the paper-parity process:

  * WIRING: the "tense"/"number" phases are scheduled ("number" was the
    `phrases` dormant-selector shape -- schedulable, listed by no stage) and
    `detect_number` sees plurality at all (its grounding-feature path read
    "SG" for every token ever generated).
  * READOUT HEALTH: image separation is far from the degenerate 1.0 the
    ablated null reads, and recall answers rather than ties.
  * SEED-42 ACCURACIES with headroom (measured PAST 0.826 / PRES 0.760 /
    SG 1.000 at n=3000; training is bit-identical across processes since
    the one-seeding-path fix, so these are stable, and the floors leave
    room for environment drift).

NOT pinned, deliberately: PL recall. Measured 0.0-0.2 across seeds --
chance -- with HEALTHY image separation: the associations never form
because every sentence trains several singular noun tokens while plural
forms ride ~30% of subjects. Hebbian mass follows token frequency; the
frequent class swamps the rare one (same shape as Zipf-closes-composition
and the starvation results behind PASSIVE_EVERY/DITRANSITIVE_EVERY). That
is an open frequency-imbalance question, not a desired behavior to pin.
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
    """Corpus-attested unambiguous verb forms, split by tense (the
    experiment's test_sets rule, inlined: raw-data scan, homographs and
    zero-derivation pasts excluded)."""
    from neural_assemblies.lexicon.data import NOUNS, VERBS

    noun_surfaces = set()
    for e in NOUNS:
        noun_surfaces.add(e["lemma"])
        pl = e.get("forms", {}).get("plural")
        if pl:
            noun_surfaces.add(pl)
    past, pres = [], []
    for e in VERBS:
        forms = e.get("forms", {})
        for w, bucket in ((forms.get("past"), past),
                          (forms.get("3sg"), pres)):
            if (w and w in parser.stim_map and w != e["lemma"]
                    and w not in noun_surfaces):
                bucket.append(w)
    return past, pres


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
        assert past_acc >= 0.65, f"PAST {past_acc:.3f} (measured 0.826)"
        assert pres_acc >= 0.60, f"PRES {pres_acc:.3f} (measured 0.760)"
        assert seps and max(seps) < 0.5, (
            f"image separation {max(seps) if seps else None} -- images "
            f"merging is the ablated-null signature (1.0)")

    def test_recall_number_sg_reads(self, trained):
        """The frequent class reads cleanly (measured 1.000). PL is at
        chance -- the open frequency-imbalance finding, see module
        docstring -- so only SG is pinned."""
        from neural_assemblies.lexicon.data import NOUNS

        sg = [e["lemma"] for e in NOUNS
              if e.get("forms", {}).get("plural")
              and e["lemma"] in trained.stim_map][:15]
        assert len(sg) >= 10
        ok = sum(trained.recall_number(w)[0] == "SG" for w in sg)
        assert ok / len(sg) >= 0.8, f"SG {ok}/{len(sg)}"

    def test_recall_unknown_word_returns_none(self, trained):
        got, diag = trained.recall_tense("zzz-not-a-word")
        assert got is None
        assert diag["scores"] == {}

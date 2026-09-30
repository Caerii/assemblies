"""Cross-engine projection: explicit sources into sparse targets."""

import numpy as np
import pytest

from neural_assemblies.core.brain import Brain
from neural_assemblies.assembly_calculus import overlap


SEED = 42
P = 0.1


def _has_torch_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


ENGINES = ["numpy_sparse"]
if _has_torch_cuda():
    ENGINES.append("torch_sparse")


def _brain(engine="numpy_sparse", **kwargs):
    defaults = dict(p=P, save_winners=True, seed=SEED, engine=engine)
    defaults.update(kwargs)
    return Brain(**defaults)


@pytest.mark.parametrize("engine", ENGINES)
class TestCrossEngineProjection:
    def test_explicit_lex_projects_into_sparse_area(self, engine):
        b = _brain(engine=engine)
        b.add_explicit_area("LEX", 400, 20, 0.2)
        b.add_area("SUBJ", 5000, 50, 0.2)
        b.areas["LEX"].winners = np.arange(60, 80, dtype=np.uint32)
        b.areas["LEX"].fix_assembly()
        b.project({}, {"LEX": ["SUBJ"]})
        assert len(b.areas["SUBJ"].winners) == 50

    def test_dense_connectome_preserved_after_add_area(self, engine):
        b = _brain(engine=engine)
        b.add_explicit_area("LEX", 400, 20, 0.2)
        b.add_area("SUBJ", 5000, 50, 0.2)
        conn = b.connectomes["LEX"]["SUBJ"]
        assert not conn.sparse
        assert conn.weights.shape == (400, 5000)

    def test_recurrent_stabilization_after_bootstrap(self, engine):
        b = _brain(engine=engine)
        b.add_explicit_area("LEX", 400, 20, 0.2)
        b.add_area("SUBJ", 5000, 50, 0.2)
        b.areas["LEX"].winners = np.arange(60, 80, dtype=np.uint32)
        b.areas["LEX"].fix_assembly()
        b.project({}, {"LEX": ["SUBJ"]})
        from neural_assemblies.assembly_calculus.ops import _snap
        asm1 = _snap(b, "SUBJ")
        for _ in range(8):
            b.project({}, {"LEX": ["SUBJ"], "SUBJ": ["SUBJ"]})
        asm2 = _snap(b, "SUBJ")
        assert len(asm1) == 50
        assert len(asm2) == 50
        assert overlap(asm1, asm2) >= 0.5

    def test_parser_sparse_grammar_with_explicit_lex(self, engine):
        from neural_assemblies.programs import RuleParser

        from neural_assemblies.language import EnglishParserBrain, LEXEME_DICT
        from neural_assemblies.language.language_areas import LEX, SUBJ

        brain = EnglishParserBrain(
            p=0.1, LEX_k=20, non_LEX_n=5000, verbose=False, engine=engine,
        )
        assert not brain.areas[SUBJ].explicit
        assert brain.areas[LEX].explicit
        brain.activateWord(LEX, "cats")
        for rule in LEXEME_DICT["cats"]["PRE_RULES"]:
            brain.applyRule(rule)
        brain.parse_project()
        assert len(brain.area_by_name[SUBJ].winners) > 0

        result = RuleParser(
            language="English", p=0.1, lex_k=20, non_LEX_n=5000, engine=engine,
        ).parse("cats chase mice")
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_sparse_subj_projects_into_explicit_lex(self, engine):
        b = _brain(engine=engine)
        b.add_explicit_area("LEX", 400, 20, 0.2)
        b.add_area("SUBJ", 5000, 50, 0.2)
        b.areas["LEX"].winners = np.arange(60, 80, dtype=np.uint32)
        b.areas["LEX"].fix_assembly()
        b.project({}, {"LEX": ["SUBJ"]})
        for _ in range(6):
            b.project({}, {"LEX": ["SUBJ"], "SUBJ": ["SUBJ"]})
        assert len(b.areas["SUBJ"].winners) == 50
        b.project({}, {"SUBJ": ["LEX"]})
        assert len(b.areas["LEX"].winners) == 20

    def test_readout_sparse_to_lex_word_retrieval(self, engine):
        from neural_assemblies.language import EnglishParserBrain, LEXEME_DICT
        from neural_assemblies.language.language_areas import LEX, SUBJ, VERB

        brain = EnglishParserBrain(
            p=0.1, LEX_k=20, non_LEX_n=5000, verbose=False, engine=engine,
        )
        for word in ["cats", "chase"]:
            brain.activateWord(LEX, word)
            for rule in LEXEME_DICT[word]["PRE_RULES"]:
                brain.applyRule(rule)
            for _ in range(15):
                brain.parse_project()
            for rule in LEXEME_DICT[word]["POST_RULES"]:
                brain.applyRule(rule)
        brain.project({}, {SUBJ: [LEX]})
        word = brain.getWord(LEX)
        assert word is not None

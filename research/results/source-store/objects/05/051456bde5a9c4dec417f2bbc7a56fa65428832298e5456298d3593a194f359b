import pytest

from neural_assemblies.assembly_calculus.emergent.acquisition.continual import (
    replay_corpus_sample,
)


class _FailingParser:
    stim_map = {"known": "stim"}

    def ingest_raw_sentence(self, _sentence):
        pass

    def train_next_token(self, _sentences, *, dedupe_sentences):
        assert dedupe_sentences is False
        raise RuntimeError("training contract failed")


def test_replay_does_not_hide_training_failure():
    with pytest.raises(RuntimeError, match="training contract failed"):
        replay_corpus_sample(_FailingParser(), [["known", "known"]])

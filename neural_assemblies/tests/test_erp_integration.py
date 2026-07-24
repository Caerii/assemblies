"""ERP pipeline integration: calibrate → mine → replay."""

import os

os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
os.environ["TRAIN_PROGRESS"] = "0"

from neural_assemblies.assembly_calculus.emergent import EmergentParser, build_vocabulary_preset
from neural_assemblies.assembly_calculus.emergent.acquisition.wobbly import (
    mine_wobbly_episodes,
    replay_wobbly_episodes,
)
from neural_assemblies.assembly_calculus.emergent.evaluation import (
    calibrate_erp_thresholds,
)


class TestErpIntegration:
    def test_calibrate_mine_replay_pipeline(self, forked_parser):
        # Calibration + mining + replay all mutate, so this takes an isolated
        # fork; the SENTENCES/seed=42 training itself is shared session-wide.
        parser = forked_parser("SENTENCES", seed=42)
        report = calibrate_erp_thresholds(parser)
        assert report.tuned

        mem = mine_wobbly_episodes(
            parser,
            [
                ["the", "dog", "chases", "finds"],
                ["the", "cat", "sees", "runs"],
            ],
            target_words={"finds", "runs"},
        )
        if mem.episodes:
            result = replay_wobbly_episodes(parser, mem)
            assert "assigned" in result
            assert result["episodes"] == len(mem.episodes)

    def test_untrained_parser_mine_empty(self):
        parser = EmergentParser(n=3000, k=30, seed=1, fast_training=True)
        for w in ("the", "dog", "chases", "finds"):
            parser.register_word(w)
        mem = mine_wobbly_episodes(parser, [["the", "dog", "chases", "finds"]])
        assert len(mem.episodes) == 0

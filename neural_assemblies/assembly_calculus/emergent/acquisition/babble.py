"""Pre-lexical babbling stage — noisy phon forms before grounded words.

WHY BABBLE IS A TRAINING STAGE AND NOT A DECORATION.  A child produces
syllable strings for months before producing words, and the standard account
is that this period builds the phonological substrate -- a space of
articulable forms -- that later words are recognised as points in.

The model reproduces the same ordering for a mechanical reason.  Word learning
here pairs a PHON stimulus with a grounding stimulus, and the PHON side is
only useful if similar-sounding forms already drive overlapping assemblies.
Running babble first exposes the phonological areas to many noisy forms and
their bigrams, so the connectome that later word learning writes into already
encodes form similarity.  Without it, every word's phon stimulus is an
isolated, unrelated point, and the model cannot recognise a variant of a known
word as that word.

This is what ``register_early_fuzzy_variants`` exists for: it makes surface
variants of a form resolve to the same lexical entry, which is the pay-off of
having done the babble stage at all.  It runs at BABBLE (the highest-beta
stage in ``_STAGE_CONFIG``, beta = 0.20) precisely because this substrate
should be laid down fast and then be built on rather than rewritten.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, TYPE_CHECKING

from ..acquisition.phonology import generate_babble_forms, generate_babble_utterances

if TYPE_CHECKING:
    from ..parser import EmergentParser


@dataclass
class BabbleReport:
    """Metrics from the babbling stage."""
    n_forms: int = 0
    n_utterances: int = 0
    unique_bigrams: int = 0
    forms: List[str] = field(default_factory=list)


def train_babble_stage(
    parser: "EmergentParser",
    *,
    n_forms: int = 24,
    n_utterances: int = 30,
    repetitions: int = 2,
    seed: int = 0,
) -> BabbleReport:
    """Expose the learner to syllable babble before canonical lexicon training.

    Babbling registers phon stimuli and ingests raw syllable sequences.
    No lexicon projection — only exposure and light incremental parsing.
    """
    from ..train_progress import current_progress

    forms = generate_babble_forms(n_forms, seed=seed)
    utterances = generate_babble_utterances(forms, n=n_utterances, seed=seed + 1)

    parser.babble_forms = list(forms)  # type: ignore[attr-defined]

    prog = current_progress()
    with prog.phase("babble", f"{len(forms)} forms x{repetitions}"):
        for form in forms:
            parser.register_word(form)

        for _rep in range(repetitions):
            for utt in utterances:
                parser.ingest_raw_sentence(utt)

    bigrams: Dict[tuple, int] = {}
    for utt in utterances:
        for i in range(len(utt) - 1):
            pair = (utt[i], utt[i + 1])
            bigrams[pair] = bigrams.get(pair, 0) + 1

    return BabbleReport(
        n_forms=len(forms),
        n_utterances=len(utterances) * repetitions,
        unique_bigrams=len(bigrams),
        forms=forms,
    )


def isolate_babble_forms(parser: "EmergentParser") -> int:
    """Remove babble syllables from active lexicon after the pre-lexical stage.

    Babble registers transient phon forms for distributional exposure; they must
    not compete with grounded lemmas during later curriculum stages.
    """
    forms = list(getattr(parser, "babble_forms", []))
    if not forms:
        return 0

    for form in forms:
        parser.stim_map.pop(form, None)
        parser.word_grounding.pop(form, None)
        parser._category_cache.pop(form, None)
        parser.dist_stats.word_count.pop(form, None)
        parser.dist_stats.position_counts.pop(form, None)
        parser.dist_stats.word_as_pre_verb.pop(form, None)
        parser.dist_stats.word_as_post_verb.pop(form, None)
        parser.dist_stats.word_as_action.pop(form, None)
        cooc = parser.dist_stats.word_cooccurrence.pop(form, None)
        if cooc:
            for other in list(cooc):
                parser.dist_stats.word_cooccurrence.get(other, {}).pop(form, None)

    return len(forms)


def register_early_fuzzy_variants(
    parser: "EmergentParser",
    canonical_words: List[str],
    *,
    seed: int = 0,
) -> Dict[str, List[str]]:
    """Register child-like surface variants linked to canonical lemmas.

    Variant generation is deterministic and has never consumed ``seed``.
    Keep the legacy keyword only for source compatibility, but reject a
    nonzero value so callers cannot mistake it for an experimental factor.
    """
    if seed != 0:
        raise ValueError(
            "register_early_fuzzy_variants does not use seed; omit it or pass 0"
        )
    from ..acquisition.phonology import EARLY_FUZZY_LEMMAS, fuzzy_variants

    if not hasattr(parser, "surface_to_canonical"):
        parser.surface_to_canonical = {}  # type: ignore[attr-defined]

    mapping: Dict[str, List[str]] = {}
    targets = canonical_words or list(EARLY_FUZZY_LEMMAS)

    for _i, canonical in enumerate(targets):
        if canonical not in parser.word_grounding and hasattr(parser, "auto_ground"):
            parser.register_word(canonical)
        variants = fuzzy_variants(canonical, max_variants=3)
        registered: List[str] = []
        for surface in variants:
            parser.register_fuzzy_surface(canonical, surface)  # type: ignore[attr-defined]
            registered.append(surface)
        if registered:
            mapping[canonical] = registered

    return mapping

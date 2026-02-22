"""
Context-free grammar for generating variable-length sentences.

The grammar produces sentences with structural annotations (word roles,
categories, and constituency) that drive training and testing. Experiments
don't need to know sentence structure in advance — the grammar output
tells the training module what prediction pairs and role bindings to create.

Current productions (depth-limited):
  S  -> NP_subj VP
  VP -> V NP_obj
  VP -> V NP_obj PP         (with probability pp_prob)
  PP -> P NP_ppobj

Extension points for future work:
  - Recursive PP:  PP -> P NP PP?    (unbounded depth)
  - Coordination:  NP -> NP "and" NP (parallel structure)
  - Relative clause: NP -> NP "that" VP (center-embedding)
  - Adjectives:    NP -> Adj N       (modifier structure)
"""

from typing import Dict, List, Any, Optional

import numpy as np

from research.experiments.lib.vocabulary import Vocabulary, DEFAULT_VOCAB


class SimpleCFG:
    """Context-free grammar for variable-length sentence generation.

    Each generated sentence is a dict containing:
      words:      list of word strings
      roles:      list of grammatical roles (AGENT, VERB, PATIENT, PREP, PP_OBJ)
      categories: list of category labels (NOUN, VERB, PREP, LOCATION)
      has_pp:     bool indicating whether sentence contains a PP

    The grammar is parameterized by:
      pp_prob:        probability of generating VP -> V NP PP
      novel_obj_prob: probability of using a novel noun in object position
      vocab:          vocabulary specification (word lists and categories)
    """

    def __init__(
        self,
        pp_prob: float = 0.4,
        novel_obj_prob: float = 0.0,
        vocab: Vocabulary = None,
        rng: np.random.Generator = None,
    ):
        self.pp_prob = pp_prob
        self.novel_obj_prob = novel_obj_prob
        self.vocab = vocab or DEFAULT_VOCAB
        self.rng = rng or np.random.default_rng(42)

    def generate(self) -> Dict[str, Any]:
        """Generate one sentence from the grammar.

        Returns:
            Dict with words, roles, categories, has_pp.
        """
        nouns = self.vocab.words_for_category("NOUN")
        verbs = self.vocab.words_for_category("VERB")
        novel = list(self.vocab.novel_words.keys())

        agent = self.rng.choice(nouns)
        verb = self.rng.choice(verbs)

        # Choose patient: sometimes novel noun
        if (self.novel_obj_prob > 0
                and novel
                and self.rng.random() < self.novel_obj_prob):
            patient = self.rng.choice(novel)
            patient_cat = "NOUN"  # novel words are categorized as their parent
        else:
            patient = self.rng.choice([n for n in nouns if n != agent])
            patient_cat = "NOUN"

        words = [agent, verb, patient]
        roles = ["AGENT", "VERB", "PATIENT"]
        categories = ["NOUN", "VERB", patient_cat]

        has_pp = self.rng.random() < self.pp_prob
        if has_pp and "PREP" in self.vocab.categories:
            preps = self.vocab.words_for_category("PREP")
            locs = self.vocab.words_for_category("LOCATION")
            if preps and locs:
                prep = self.rng.choice(preps)
                pp_obj = self.rng.choice(locs)
                words.extend([prep, pp_obj])
                roles.extend(["PREP", "PP_OBJ"])
                categories.extend(["PREP", "LOCATION"])

        return {
            "words": words,
            "roles": roles,
            "categories": categories,
            "has_pp": has_pp,
        }

    def generate_batch(self, n: int) -> List[Dict[str, Any]]:
        """Generate n sentences from the grammar."""
        return [self.generate() for _ in range(n)]


def generate_svo_sentences(
    n_sentences: int,
    vocab: Vocabulary = None,
    rng: np.random.Generator = None,
) -> List[Dict[str, Any]]:
    """Generate plain SVO sentences (no PP).

    Convenience wrapper around SimpleCFG with pp_prob=0.
    """
    v = vocab or DEFAULT_VOCAB
    r = rng or np.random.default_rng(42)
    cfg = SimpleCFG(pp_prob=0.0, vocab=v, rng=r)
    return cfg.generate_batch(n_sentences)


def generate_mixed_sentences(
    n_sentences: int,
    pp_prob: float = 0.4,
    novel_obj_prob: float = 0.0,
    vocab: Vocabulary = None,
    rng: np.random.Generator = None,
) -> List[Dict[str, Any]]:
    """Generate a mix of SVO and SVO+PP sentences.

    Convenience wrapper around SimpleCFG.
    """
    v = vocab or DEFAULT_VOCAB
    r = rng or np.random.default_rng(42)
    cfg = SimpleCFG(pp_prob=pp_prob, novel_obj_prob=novel_obj_prob,
                    vocab=v, rng=r)
    return cfg.generate_batch(n_sentences)

"""
Context-free grammars for generating variable-length sentences.

Grammars produce sentences with structural annotations (word roles, categories)
that drive training and testing. The training module operates generically on
whatever the grammar produces — no sentence-structure knowledge required.

Three grammar classes, each extending the previous:

  SimpleCFG:
    S  -> NP VP
    VP -> V NP (PP)?
    PP -> P NP

  RecursiveCFG (extends SimpleCFG):
    PP -> P NP (PP)?          [recursive PP depth]
    NP -> N ("that" VP)?      [subject-relative clauses]

    Generates sentences like:
      "dog chases cat in garden on hill"      (recursive PP, depth 2)
      "dog that chases cat sees bird"         (center-embedding)
      "dog that sees bird chases cat in park" (relative + PP)

Extension points for future work:
  - Object-relative: NP -> N "that" NP V  (ORC, harder than SRC)
  - Coordination:    NP -> NP "and" NP    (parallel structure)
  - Adjectives:      NP -> Adj N          (modifier structure)
"""

from typing import Dict, List, Any, Optional

import numpy as np

from research.experiments.lib.vocabulary import Vocabulary, DEFAULT_VOCAB


# Default semantic noun groups for semantic covariance.
# When semantic_bias > 0, the grammar prefers objects from the same group
# as the agent, creating statistical regularities that prediction can learn.
DEFAULT_SEMANTIC_GROUPS: Dict[str, List[str]] = {
    "ANIMAL": ["dog", "cat", "bird"],
    "PERSON": ["boy", "girl"],
}


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
      semantic_bias:  probability of picking patient from agent's semantic group
      vocab:          vocabulary specification (word lists and categories)
    """

    def __init__(
        self,
        pp_prob: float = 0.4,
        novel_obj_prob: float = 0.0,
        semantic_bias: float = 0.0,
        semantic_groups: Dict[str, List[str]] = None,
        vocab: Vocabulary = None,
        rng: np.random.Generator = None,
    ):
        self.pp_prob = pp_prob
        self.novel_obj_prob = novel_obj_prob
        self.semantic_bias = semantic_bias
        self.semantic_groups = semantic_groups or DEFAULT_SEMANTIC_GROUPS
        self.vocab = vocab or DEFAULT_VOCAB
        self.rng = rng or np.random.default_rng(42)

        # Build reverse lookup: word -> group name
        self._word_to_group: Dict[str, str] = {}
        for group_name, words in self.semantic_groups.items():
            for w in words:
                self._word_to_group[w] = group_name

    def _pick_patient(self, agent: str, nouns: List[str]) -> str:
        """Pick patient noun, optionally biased toward agent's semantic group."""
        if self.semantic_bias > 0 and self.rng.random() < self.semantic_bias:
            group = self._word_to_group.get(agent)
            if group:
                same_group = [n for n in self.semantic_groups[group]
                              if n != agent and n in nouns]
                if same_group:
                    return self.rng.choice(same_group)
        # Default: random noun != agent
        return self.rng.choice([n for n in nouns if n != agent])

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
            patient = self._pick_patient(agent, nouns)
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


class RecursiveCFG:
    """Context-free grammar with recursive PP and relative clauses.

    Productions:
      S  -> NP VP
      NP -> N
      NP -> N "that" VP        (subject-relative clause, prob rel_prob)
      VP -> V NP
      VP -> V NP PP            (prob pp_prob)
      PP -> P NP
      PP -> P NP PP            (recursive PP, prob recursive_pp_prob)

    Role annotations use depth-indexed names:
      Depth 0 (main clause): AGENT, PATIENT, PP_OBJ, PP_OBJ_1
      Depth 1 (relative):    REL_AGENT, REL_PATIENT

    The grammar is depth-limited by max_pp_depth and max_rel_depth to
    prevent unbounded recursion while still testing recursive structure.
    """

    def __init__(
        self,
        pp_prob: float = 0.4,
        recursive_pp_prob: float = 0.5,
        rel_prob: float = 0.3,
        orc_prob: float = 0.0,
        max_pp_depth: int = 2,
        max_rel_depth: int = 1,
        novel_obj_prob: float = 0.0,
        vocab: Vocabulary = None,
        rng: np.random.Generator = None,
    ):
        self.pp_prob = pp_prob
        self.recursive_pp_prob = recursive_pp_prob
        self.rel_prob = rel_prob
        self.orc_prob = orc_prob
        self.max_pp_depth = max_pp_depth
        self.max_rel_depth = max_rel_depth
        self.novel_obj_prob = novel_obj_prob
        self.vocab = vocab or DEFAULT_VOCAB
        self.rng = rng or np.random.default_rng(42)

    def _pick_noun(self, exclude: List[str] = None) -> str:
        """Pick a noun, optionally excluding some."""
        nouns = self.vocab.words_for_category("NOUN")
        novel = list(self.vocab.novel_words.keys())
        if (self.novel_obj_prob > 0
                and novel
                and self.rng.random() < self.novel_obj_prob):
            return self.rng.choice(novel)
        candidates = [n for n in nouns if n not in (exclude or [])]
        if not candidates:
            candidates = nouns
        return self.rng.choice(candidates)

    def _pick_location(self, exclude: List[str] = None) -> str:
        """Pick a location noun."""
        locs = self.vocab.words_for_category("LOCATION")
        candidates = [l for l in locs if l not in (exclude or [])]
        if not candidates:
            candidates = locs
        return self.rng.choice(candidates)

    def _generate_pp(self, pp_depth: int, used_locs: List[str]
                     ) -> Dict[str, List]:
        """Generate a PP, possibly recursive."""
        preps = self.vocab.words_for_category("PREP")
        prep = self.rng.choice(preps)
        pp_obj = self._pick_location(exclude=used_locs)

        # Role name depends on PP depth
        role = "PP_OBJ" if pp_depth == 0 else f"PP_OBJ_{pp_depth}"

        words = [prep, pp_obj]
        roles = ["PREP", role]
        categories = ["PREP", "LOCATION"]
        productions = []

        # Recursive PP?
        if (pp_depth + 1 < self.max_pp_depth
                and self.rng.random() < self.recursive_pp_prob):
            sub = self._generate_pp(pp_depth + 1, used_locs + [pp_obj])
            words.extend(sub["words"])
            roles.extend(sub["roles"])
            categories.extend(sub["categories"])
            productions.append("PP_P_NP_PP")
            productions.extend(sub.get("productions", []))
        else:
            productions.append("PP_P_NP")

        return {"words": words, "roles": roles, "categories": categories,
                "productions": productions}

    def _generate_vp(self, rel_depth: int, used_nouns: List[str]
                     ) -> Dict[str, List]:
        """Generate VP -> V NP (PP)?"""
        verbs = self.vocab.words_for_category("VERB")
        verb = self.rng.choice(verbs)
        patient = self._pick_noun(exclude=used_nouns)

        # Role prefix for relative clauses
        patient_role = "REL_PATIENT" if rel_depth > 0 else "PATIENT"

        words = [verb, patient]
        roles = ["VERB" if rel_depth == 0 else "REL_VERB", patient_role]
        categories = ["VERB", "NOUN"]
        productions = []

        # Optional PP
        if self.rng.random() < self.pp_prob and "PREP" in self.vocab.categories:
            pp = self._generate_pp(0, [])
            words.extend(pp["words"])
            roles.extend(pp["roles"])
            categories.extend(pp["categories"])
            productions.append("VP_V_NP_PP")
            productions.extend(pp.get("productions", []))
        else:
            productions.append("VP_V_NP")

        return {"words": words, "roles": roles, "categories": categories,
                "productions": productions}

    def _generate_orc(self, head_noun: str, used_nouns: List[str]
                      ) -> Dict[str, List]:
        """Generate an object-relative clause: "that" NP V.

        The head noun is the patient of the embedded verb; the new NP is
        the embedded agent.  Example: head="dog" produces
        "that cat chases" where cat=REL_AGENT, chases=REL_VERB,
        and the head noun "dog" gets REL_PATIENT binding (handled by caller).
        """
        verbs = self.vocab.words_for_category("VERB")
        rel_verb = self.rng.choice(verbs)
        rel_agent = self._pick_noun(exclude=used_nouns + [head_noun])

        words = ["that", rel_agent, rel_verb]
        roles = ["COMP", "REL_AGENT", "REL_VERB"]
        categories = ["COMP", "NOUN", "VERB"]

        return {"words": words, "roles": roles, "categories": categories,
                "productions": ["NP_N_that_NP_V_ORC"]}

    def generate(self) -> Dict[str, Any]:
        """Generate one sentence from the recursive grammar.

        Returns dict with words, roles, categories, has_pp, has_rel,
        rel_type, pp_depth, length.
        """
        nouns = self.vocab.words_for_category("NOUN")
        agent = self.rng.choice(nouns)

        words = [agent]
        roles = ["AGENT"]
        categories = ["NOUN"]
        productions = ["S_NP_VP"]
        has_rel = False
        rel_type = None
        max_pp_depth_seen = 0

        # Relative clause: NP -> N "that" VP (SRC) or N "that" NP V (ORC)
        if (self.rel_prob > 0
                and self.max_rel_depth > 0
                and "COMP" in self.vocab.categories
                and self.rng.random() < self.rel_prob):

            is_orc = self.orc_prob > 0 and self.rng.random() < self.orc_prob

            if is_orc:
                # Object-relative: head noun is patient of embedded verb
                orc = self._generate_orc(agent, [agent])
                words.extend(orc["words"])
                roles.extend(orc["roles"])
                categories.extend(orc["categories"])
                productions.extend(orc.get("productions", []))
                roles[0] = "AGENT+REL_PATIENT"  # main agent + embedded patient
                has_rel = True
                rel_type = "ORC"
            else:
                # Subject-relative: head noun is agent of embedded verb
                comps = self.vocab.words_for_category("COMP")
                comp = self.rng.choice(comps)
                words.append(comp)
                roles.append("COMP")
                categories.append("COMP")
                roles[0] = "AGENT+REL_AGENT"  # dual binding annotation
                productions.append("NP_N_that_VP_SRC")

                rel_vp = self._generate_vp(rel_depth=1, used_nouns=[agent])
                words.extend(rel_vp["words"])
                roles.extend(rel_vp["roles"])
                categories.extend(rel_vp["categories"])
                productions.extend(rel_vp.get("productions", []))
                has_rel = True
                rel_type = "SRC"
        else:
            productions.append("NP_N")

        # Main VP
        main_vp = self._generate_vp(rel_depth=0, used_nouns=[agent])
        words.extend(main_vp["words"])
        roles.extend(main_vp["roles"])
        categories.extend(main_vp["categories"])
        productions.extend(main_vp.get("productions", []))

        # Count PP depth
        pp_roles = [r for r in roles if r.startswith("PP_OBJ")]
        if pp_roles:
            depths = []
            for r in pp_roles:
                if r == "PP_OBJ":
                    depths.append(0)
                else:
                    depths.append(int(r.split("_")[-1]))
            max_pp_depth_seen = max(depths) + 1

        has_pp = any(r.startswith("PP_OBJ") for r in roles)

        return {
            "words": words,
            "roles": roles,
            "categories": categories,
            "has_pp": has_pp,
            "has_rel": has_rel,
            "rel_type": rel_type,
            "pp_depth": max_pp_depth_seen,
            "length": len(words),
            "productions_used": productions,
        }

    def generate_batch(self, n: int) -> List[Dict[str, Any]]:
        """Generate n sentences."""
        return [self.generate() for _ in range(n)]

    @staticmethod
    def production_frequencies(
        batch: List[Dict[str, Any]],
    ) -> Dict[str, int]:
        """Count production rule occurrences across a batch of sentences."""
        counts: Dict[str, int] = {}
        for sent in batch:
            for prod in sent.get("productions_used", []):
                counts[prod] = counts.get(prod, 0) + 1
        return counts


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


class DetCFG:
    """CFG that wraps SimpleCFG and inserts determiners before nouns.

    Produces DET-N-V-DET-N sentences (with optional PP: DET-N-V-DET-N-P-DET-N).
    The DET-N-V pattern creates a 3-position cycle that provides distributional
    evidence for 3-way category discovery in the free-form learner.

    Requires a vocabulary with a "DET" category (e.g. DET_VOCAB).
    """

    def __init__(
        self,
        pp_prob: float = 0.3,
        vocab: Vocabulary = None,
        rng: np.random.Generator = None,
        **kwargs,
    ):
        from research.experiments.lib.vocabulary import DET_VOCAB
        self.vocab = vocab or DET_VOCAB
        self.rng = rng or np.random.default_rng(42)
        self.inner = SimpleCFG(
            pp_prob=pp_prob, vocab=self.vocab, rng=self.rng, **kwargs,
        )

    def generate(self) -> Dict[str, Any]:
        """Generate one sentence, inserting determiners before nouns/locations."""
        sent = self.inner.generate()
        dets = self.vocab.words_for_category("DET")
        new_words: List[str] = []
        new_roles: List[str] = []
        new_cats: List[str] = []
        for w, r, c in zip(sent["words"], sent["roles"], sent["categories"]):
            if c in ("NOUN", "LOCATION"):
                new_words.append(self.rng.choice(dets))
                new_roles.append("DET")
                new_cats.append("DET")
            new_words.append(w)
            new_roles.append(r)
            new_cats.append(c)
        return {
            "words": new_words,
            "roles": new_roles,
            "categories": new_cats,
            "has_pp": sent["has_pp"],
        }

    def generate_batch(self, n: int) -> List[Dict[str, Any]]:
        """Generate n sentences."""
        return [self.generate() for _ in range(n)]


class AdjCFG:
    """CFG that wraps DetCFG and inserts adjectives between DET and NOUN.

    Produces DET-ADJ-N-V-DET-ADJ-N sentences (with optional PP).
    The DET-ADJ-N-V pattern creates a 4-position cycle that provides
    distributional evidence for 4-way category discovery.

    Requires a vocabulary with "DET" and "ADJ" categories (e.g. ADJ_VOCAB).
    """

    def __init__(
        self,
        pp_prob: float = 0.3,
        vocab: Vocabulary = None,
        rng: np.random.Generator = None,
        **kwargs,
    ):
        from research.experiments.lib.vocabulary import ADJ_VOCAB
        self.vocab = vocab or ADJ_VOCAB
        self.rng = rng or np.random.default_rng(42)
        self.inner = DetCFG(
            pp_prob=pp_prob, vocab=self.vocab, rng=self.rng, **kwargs,
        )

    def generate(self) -> Dict[str, Any]:
        """Generate one sentence, inserting adjectives after determiners."""
        sent = self.inner.generate()
        adjs = self.vocab.words_for_category("ADJ")
        new_words: List[str] = []
        new_roles: List[str] = []
        new_cats: List[str] = []
        for w, r, c in zip(sent["words"], sent["roles"], sent["categories"]):
            new_words.append(w)
            new_roles.append(r)
            new_cats.append(c)
            if c == "DET":
                new_words.append(self.rng.choice(adjs))
                new_roles.append("ADJ")
                new_cats.append("ADJ")
        return {
            "words": new_words,
            "roles": new_roles,
            "categories": new_cats,
            "has_pp": sent["has_pp"],
        }

    def generate_batch(self, n: int) -> List[Dict[str, Any]]:
        """Generate n sentences."""
        return [self.generate() for _ in range(n)]

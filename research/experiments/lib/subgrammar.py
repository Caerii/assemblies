"""
Subgrammar decomposition utilities.

Maps production rules to subgrammar groupings and provides per-subgrammar
measurement decomposition.  Inspired by Schulz, Mitropolsky & Poggio (2025)
"Unraveling Syntax: How Language Models Learn Context-Free Grammars", which
shows that CFG loss decomposes recursively over production rules.

Subgrammar definitions correspond to the production rules tracked by
RecursiveCFG.generate() in grammar.py.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any
from collections import defaultdict

import numpy as np


# Named sets of production rules defining each subgrammar type.
# A sentence belongs to the most specific subgrammar whose rules it uses.
SUBGRAMMAR_DEFS: Dict[str, set] = {
    "SVO": {"S_NP_VP", "NP_N", "VP_V_NP"},
    "SVO_PP": {"S_NP_VP", "NP_N", "VP_V_NP_PP", "PP_P_NP"},
    "SVO_PP_recursive": {"S_NP_VP", "NP_N", "VP_V_NP_PP", "PP_P_NP_PP", "PP_P_NP"},
    "SRC": {"S_NP_VP", "NP_N_that_VP_SRC", "VP_V_NP"},
    "ORC": {"S_NP_VP", "NP_N_that_NP_V_ORC", "VP_V_NP"},
    "SRC_PP": {"S_NP_VP", "NP_N_that_VP_SRC", "VP_V_NP_PP", "PP_P_NP"},
}


def classify_sentence(sentence: Dict[str, Any]) -> str:
    """Classify a sentence into its most specific subgrammar.

    Uses the productions_used field from RecursiveCFG.generate().
    Falls back to "SVO" if no productions_used field is present.
    """
    prods = set(sentence.get("productions_used", []))

    if "NP_N_that_NP_V_ORC" in prods:
        return "ORC"
    if "NP_N_that_VP_SRC" in prods:
        if "VP_V_NP_PP" in prods:
            return "SRC_PP"
        return "SRC"
    if "PP_P_NP_PP" in prods:
        return "SVO_PP_recursive"
    if "VP_V_NP_PP" in prods:
        return "SVO_PP"
    return "SVO"


def partition_by_subgrammar(
    batch: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Partition a batch of sentences by subgrammar classification."""
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for sent in batch:
        groups[classify_sentence(sent)].append(sent)
    return dict(groups)


@dataclass
class SubgrammarStats:
    """Per-subgrammar measurement accumulator.

    Records N400 and P600 values keyed by subgrammar name and provides
    summary statistics.
    """

    n400_values: Dict[str, List[float]] = field(
        default_factory=lambda: defaultdict(list))
    p600_values: Dict[str, List[float]] = field(
        default_factory=lambda: defaultdict(list))

    def record(self, subgrammar: str, n400: float, p600: float = 0.0):
        """Record one measurement for a subgrammar."""
        self.n400_values[subgrammar].append(n400)
        self.p600_values[subgrammar].append(p600)

    def means(self) -> Dict[str, Dict[str, float]]:
        """Compute per-subgrammar mean N400 and P600."""
        result = {}
        for sg in set(list(self.n400_values.keys()) +
                       list(self.p600_values.keys())):
            result[sg] = {
                "n400_mean": float(np.mean(self.n400_values[sg]))
                if self.n400_values[sg] else 0.0,
                "p600_mean": float(np.mean(self.p600_values[sg]))
                if self.p600_values[sg] else 0.0,
                "count": len(self.n400_values[sg]),
            }
        return result

    def n400_by_subgrammar(self) -> Dict[str, float]:
        """Return mean N400 per subgrammar."""
        return {sg: float(np.mean(vals))
                for sg, vals in self.n400_values.items() if vals}

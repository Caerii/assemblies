"""
NEMO Emergent Generation Module
===============================

Generates language through neural activation, not templates.

Key Principles:
1. VP component areas (VP_SUBJ, VP_VERB, VP_OBJ) preserve structure
2. Generation works by activating assemblies and reading out words
3. Retrieval uses neural overlap, not string parsing
4. Compatible combinations are found through assembly overlap

Architecture:
- emergent_retriever.py: TRULY emergent retrieval using VP component areas
- neural_decoder.py: Decodes assemblies to words via overlap matching
- vp_decoder.py: VP key-based decoder (hybrid approach)
- activation.py: Spreads activation through learned connections
- hybrid_gen.py: Hybrid emergent response generator

RECOMMENDED: Use EmergentRetriever + EmergentGenerator from emergent_retriever.py
for truly emergent retrieval and generation.

Use HybridEmergentGenerator for faster but less pure generation.
"""

from .vp_decoder import VPDecoder
from .activation import ActivationSpreader
from .generator import EmergentGenerator as LegacyEmergentGenerator
from .neural_decoder import NeuralDecoder, EmergentRetriever as LegacyEmergentRetriever
from .emergent_gen import TrueEmergentGenerator
from .hybrid_gen import HybridEmergentGenerator
from .emergent_retriever import EmergentRetriever, EmergentGenerator

__all__ = [
    # Primary (truly emergent)
    'EmergentRetriever',
    'EmergentGenerator',
    
    # Hybrid approaches
    'HybridEmergentGenerator',
    'VPDecoder',
    
    # Supporting classes
    'ActivationSpreader', 
    'NeuralDecoder',
    
    # Legacy (kept for compatibility)
    'TrueEmergentGenerator',
    'LegacyEmergentGenerator',
    'LegacyEmergentRetriever',
]


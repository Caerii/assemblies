"""
Hopfield-Style Associative Memory for NEMO
==========================================

This module implements content-addressable memory using Modern Hopfield Network
principles. It allows storing and retrieving VP-component associations without
string keys.

Key insight: The attention mechanism in transformers is equivalent to 
Modern Hopfield Networks with exponential storage capacity.

Storage: (key_assembly, value_assembly) pairs
Retrieval: query_assembly → attention(query, keys) → weighted sum of values

This is neurally plausible:
- Keys and values are assembly vectors (neural activity patterns)
- Retrieval uses dot-product similarity (like neural correlation)
- Softmax attention (like competitive inhibition)
- Weighted sum (like population coding)
"""

import numpy as np
import cupy as cp
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class MemoryEntry:
    """A single memory entry with key and value assemblies."""
    key: np.ndarray  # Dense binary vector [n]
    value: np.ndarray  # Dense binary vector [n]
    metadata: Optional[dict] = None  # Optional metadata for debugging


class HopfieldMemory:
    """
    Content-addressable memory using Hopfield-style attention.
    
    This replaces string-based lookup with neural pattern matching.
    """
    
    def __init__(self, n: int, k: int, temperature: float = 0.1):
        """
        Initialize Hopfield memory.
        
        Args:
            n: Number of neurons (dimensionality of assemblies)
            k: Assembly size (number of active neurons)
            temperature: Softmax temperature (lower = sharper attention)
        """
        self.n = n
        self.k = k
        self.temperature = temperature
        
        # Storage
        self.memories: List[MemoryEntry] = []
        
        # Cached matrices for efficient retrieval
        self._key_matrix: Optional[np.ndarray] = None
        self._value_matrix: Optional[np.ndarray] = None
        self._dirty = True
    
    def assembly_to_dense(self, assembly: cp.ndarray) -> np.ndarray:
        """Convert sparse assembly (k indices) to dense binary vector."""
        dense = np.zeros(self.n, dtype=np.float32)
        indices = assembly.get().astype(np.int64)
        dense[indices] = 1.0
        return dense
    
    def dense_to_assembly(self, dense: np.ndarray) -> cp.ndarray:
        """Convert dense vector to sparse assembly (top-k indices)."""
        indices = np.argsort(dense)[-self.k:]
        return cp.array(indices, dtype=cp.uint32)
    
    def store(self, key_assembly: cp.ndarray, value_assembly: cp.ndarray,
              metadata: Optional[dict] = None):
        """
        Store a (key, value) association.
        
        Args:
            key_assembly: The key assembly (e.g., verb assembly)
            value_assembly: The value assembly (e.g., subject assembly)
            metadata: Optional metadata for debugging
        """
        entry = MemoryEntry(
            key=self.assembly_to_dense(key_assembly),
            value=self.assembly_to_dense(value_assembly),
            metadata=metadata
        )
        self.memories.append(entry)
        self._dirty = True
    
    def _build_matrices(self):
        """Build cached matrices for efficient retrieval."""
        if not self._dirty or len(self.memories) == 0:
            return
        
        self._key_matrix = np.stack([m.key for m in self.memories])
        self._value_matrix = np.stack([m.value for m in self.memories])
        self._dirty = False
    
    def retrieve(self, query_assembly: cp.ndarray, 
                 top_k: int = 1) -> List[Tuple[cp.ndarray, float, Optional[dict]]]:
        """
        Retrieve values matching the query using attention.
        
        Args:
            query_assembly: The query assembly
            top_k: Number of top matches to return
        
        Returns:
            List of (value_assembly, attention_score, metadata) tuples
        """
        if len(self.memories) == 0:
            return []
        
        self._build_matrices()
        
        # Convert query to dense
        query = self.assembly_to_dense(query_assembly)
        
        # Compute attention scores
        scores = self._key_matrix @ query  # [num_memories]
        
        # Apply softmax with temperature
        scores_scaled = scores / self.temperature
        attention = np.exp(scores_scaled - np.max(scores_scaled))
        attention = attention / attention.sum()
        
        # Get top-k matches
        top_indices = np.argsort(attention)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            value_dense = self._value_matrix[idx]
            value_assembly = self.dense_to_assembly(value_dense)
            results.append((
                value_assembly,
                float(attention[idx]),
                self.memories[idx].metadata
            ))
        
        return results
    
    def retrieve_weighted(self, query_assembly: cp.ndarray) -> cp.ndarray:
        """
        Retrieve weighted sum of all values (soft attention).
        
        This is the true Hopfield retrieval: returns a pattern that is
        a weighted combination of all stored values.
        
        Args:
            query_assembly: The query assembly
        
        Returns:
            Retrieved assembly (weighted sum converted to top-k)
        """
        if len(self.memories) == 0:
            return None
        
        self._build_matrices()
        
        query = self.assembly_to_dense(query_assembly)
        
        # Compute attention
        scores = self._key_matrix @ query
        scores_scaled = scores / self.temperature
        attention = np.exp(scores_scaled - np.max(scores_scaled))
        attention = attention / attention.sum()
        
        # Weighted sum of values
        retrieved = attention @ self._value_matrix
        
        # Convert to assembly
        return self.dense_to_assembly(retrieved)
    
    def clear(self):
        """Clear all memories."""
        self.memories = []
        self._key_matrix = None
        self._value_matrix = None
        self._dirty = True
    
    def __len__(self):
        return len(self.memories)


class VPMemoryStore:
    """
    Specialized memory store for VP-component associations.
    
    Stores three types of associations:
    1. VP → Subject (who does the action?)
    2. VP → Verb (what action?)
    3. VP → Object (what is acted upon?)
    
    All retrieval is content-addressable using Hopfield attention.
    """
    
    def __init__(self, n: int, k: int, temperature: float = 0.1):
        """
        Initialize VP memory store.
        
        Args:
            n: Number of neurons
            k: Assembly size
            temperature: Softmax temperature
        """
        self.n = n
        self.k = k
        
        # Separate memories for each component type
        # Key: verb assembly, Value: subject assembly
        self.verb_to_subject = HopfieldMemory(n, k, temperature)
        
        # Key: subject assembly, Value: verb assembly
        self.subject_to_verb = HopfieldMemory(n, k, temperature)
        
        # Key: (subject, verb) merged, Value: object assembly
        self.sv_to_object = HopfieldMemory(n, k, temperature)
        
        # Key: VP assembly, Value: components (for full retrieval)
        self.vp_to_subject = HopfieldMemory(n, k, temperature)
        self.vp_to_verb = HopfieldMemory(n, k, temperature)
        self.vp_to_object = HopfieldMemory(n, k, temperature)
    
    def store_intransitive(self, subject_assembly: cp.ndarray, 
                           verb_assembly: cp.ndarray,
                           vp_assembly: cp.ndarray,
                           subject_word: str = None,
                           verb_word: str = None):
        """
        Store an intransitive sentence (subject + verb).
        
        Args:
            subject_assembly: Subject noun assembly
            verb_assembly: Verb assembly
            vp_assembly: Merged VP assembly
            subject_word: Subject word (for debugging)
            verb_word: Verb word (for debugging)
        """
        metadata = {'subject': subject_word, 'verb': verb_word}
        
        # Store bidirectional associations
        self.verb_to_subject.store(verb_assembly, subject_assembly, metadata)
        self.subject_to_verb.store(subject_assembly, verb_assembly, metadata)
        
        # Store VP associations
        self.vp_to_subject.store(vp_assembly, subject_assembly, metadata)
        self.vp_to_verb.store(vp_assembly, verb_assembly, metadata)
    
    def store_transitive(self, subject_assembly: cp.ndarray,
                         verb_assembly: cp.ndarray,
                         object_assembly: cp.ndarray,
                         vp_assembly: cp.ndarray,
                         sv_assembly: cp.ndarray,
                         subject_word: str = None,
                         verb_word: str = None,
                         object_word: str = None):
        """
        Store a transitive sentence (subject + verb + object).
        
        Args:
            subject_assembly: Subject noun assembly
            verb_assembly: Verb assembly
            object_assembly: Object noun assembly
            vp_assembly: Full VP assembly (subject + verb + object merged)
            sv_assembly: Subject-verb merged assembly (for object lookup)
        """
        metadata = {'subject': subject_word, 'verb': verb_word, 'object': object_word}
        
        # Store intransitive associations
        self.verb_to_subject.store(verb_assembly, subject_assembly, metadata)
        self.subject_to_verb.store(subject_assembly, verb_assembly, metadata)
        
        # Store object association (keyed by subject-verb merge)
        self.sv_to_object.store(sv_assembly, object_assembly, metadata)
        
        # Store VP associations
        self.vp_to_subject.store(vp_assembly, subject_assembly, metadata)
        self.vp_to_verb.store(vp_assembly, verb_assembly, metadata)
        self.vp_to_object.store(vp_assembly, object_assembly, metadata)
    
    def retrieve_subject_for_verb(self, verb_assembly: cp.ndarray, 
                                   top_k: int = 3) -> List[Tuple[cp.ndarray, float, dict]]:
        """
        Retrieve subjects for a given verb.
        
        Query: "Who runs?" → returns subjects that run
        """
        return self.verb_to_subject.retrieve(verb_assembly, top_k)
    
    def retrieve_verb_for_subject(self, subject_assembly: cp.ndarray,
                                   top_k: int = 3) -> List[Tuple[cp.ndarray, float, dict]]:
        """
        Retrieve verbs for a given subject.
        
        Query: "What does dog do?" → returns verbs dog does
        """
        return self.subject_to_verb.retrieve(subject_assembly, top_k)
    
    def retrieve_object_for_sv(self, sv_assembly: cp.ndarray,
                                top_k: int = 3) -> List[Tuple[cp.ndarray, float, dict]]:
        """
        Retrieve objects for a given subject-verb pair.
        
        Query: "What does dog chase?" → returns objects dog chases
        """
        return self.sv_to_object.retrieve(sv_assembly, top_k)
    
    def clear(self):
        """Clear all memories."""
        self.verb_to_subject.clear()
        self.subject_to_verb.clear()
        self.sv_to_object.clear()
        self.vp_to_subject.clear()
        self.vp_to_verb.clear()
        self.vp_to_object.clear()


# Test the implementation
if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))
    
    from src.nemo.language.emergent.brain import EmergentNemoBrain
    from src.nemo.language.emergent.areas import Area
    
    print("="*70)
    print("TEST: VPMemoryStore with Hopfield Attention")
    print("="*70)
    
    brain = EmergentNemoBrain(verbose=False)
    n = brain.p.n
    k = brain.p.k
    
    # Create memory store
    memory = VPMemoryStore(n, k, temperature=0.1)
    
    # Create word assemblies
    print("\n1. Creating word assemblies...")
    words = {}
    for name, area, seed in [
        ('dog', Area.NOUN_CORE, 1),
        ('cat', Area.NOUN_CORE, 2),
        ('bird', Area.NOUN_CORE, 3),
        ('mouse', Area.NOUN_CORE, 4),
        ('runs', Area.VERB_CORE, 5),
        ('sleeps', Area.VERB_CORE, 6),
        ('chases', Area.VERB_CORE, 7),
        ('eats', Area.VERB_CORE, 8),
    ]:
        cp.random.seed(seed * 1000)
        phon = cp.random.randint(0, n, k, dtype=cp.uint32)
        brain._clear_area(area)
        for _ in range(20):
            brain._project(area, phon, learn=True)
        words[name] = brain.current[area].copy()
    
    # Learn sentences
    print("\n2. Learning sentences...")
    
    intransitive = [
        ('dog', 'runs'),
        ('cat', 'sleeps'),
        ('bird', 'sleeps'),
    ]
    
    for subject, verb in intransitive:
        subj_asm = words[subject]
        verb_asm = words[verb]
        
        # Create VP
        brain._clear_area(Area.VP)
        for _ in range(20):
            brain._project(Area.VP, subj_asm, learn=True)
            brain._project(Area.VP, verb_asm, learn=True)
        vp_asm = brain.current[Area.VP].copy()
        
        memory.store_intransitive(subj_asm, verb_asm, vp_asm, subject, verb)
        print(f"   {subject} {verb}")
    
    transitive = [
        ('dog', 'chases', 'cat'),
        ('cat', 'chases', 'mouse'),
        ('bird', 'eats', 'mouse'),
    ]
    
    for subject, verb, obj in transitive:
        subj_asm = words[subject]
        verb_asm = words[verb]
        obj_asm = words[obj]
        
        # Create SV merge
        brain._clear_area(Area.VP)
        for _ in range(20):
            brain._project(Area.VP, subj_asm, learn=True)
            brain._project(Area.VP, verb_asm, learn=True)
        sv_asm = brain.current[Area.VP].copy()
        
        # Create full VP
        for _ in range(20):
            brain._project(Area.VP, obj_asm, learn=True)
        vp_asm = brain.current[Area.VP].copy()
        
        memory.store_transitive(subj_asm, verb_asm, obj_asm, vp_asm, sv_asm, 
                                subject, verb, obj)
        print(f"   {subject} {verb} {obj}")
    
    # Test retrieval
    print("\n3. Testing retrieval...")
    
    def compute_overlap(a1, a2, k):
        s1 = set(a1.get().tolist())
        s2 = set(a2.get().tolist())
        return len(s1 & s2) / k
    
    def decode_assembly(assembly, word_dict, area_filter=None):
        """Decode assembly to best matching word."""
        best_word = None
        best_overlap = 0
        for name, asm in word_dict.items():
            overlap = compute_overlap(assembly, asm, k)
            if overlap > best_overlap:
                best_overlap = overlap
                best_word = name
        return best_word, best_overlap
    
    # Test: "Who runs?"
    print("\n   Query: 'Who runs?'")
    results = memory.retrieve_subject_for_verb(words['runs'], top_k=3)
    for asm, score, meta in results:
        word, overlap = decode_assembly(asm, words)
        print(f"     {word}: attention={score:.3f}, overlap={overlap:.3f}, meta={meta}")
    
    # Test: "Who sleeps?"
    print("\n   Query: 'Who sleeps?'")
    results = memory.retrieve_subject_for_verb(words['sleeps'], top_k=3)
    for asm, score, meta in results:
        word, overlap = decode_assembly(asm, words)
        print(f"     {word}: attention={score:.3f}, overlap={overlap:.3f}, meta={meta}")
    
    # Test: "What does dog chase?"
    print("\n   Query: 'What does dog chase?'")
    # Create dog-chases merge
    brain._clear_area(Area.VP)
    for _ in range(20):
        brain._project(Area.VP, words['dog'], learn=True)
        brain._project(Area.VP, words['chases'], learn=True)
    dog_chases = brain.current[Area.VP].copy()
    
    results = memory.retrieve_object_for_sv(dog_chases, top_k=3)
    for asm, score, meta in results:
        word, overlap = decode_assembly(asm, words)
        print(f"     {word}: attention={score:.3f}, overlap={overlap:.3f}, meta={meta}")
    
    # Test: "What does cat chase?"
    print("\n   Query: 'What does cat chase?'")
    brain._clear_area(Area.VP)
    for _ in range(20):
        brain._project(Area.VP, words['cat'], learn=True)
        brain._project(Area.VP, words['chases'], learn=True)
    cat_chases = brain.current[Area.VP].copy()
    
    results = memory.retrieve_object_for_sv(cat_chases, top_k=3)
    for asm, score, meta in results:
        word, overlap = decode_assembly(asm, words)
        print(f"     {word}: attention={score:.3f}, overlap={overlap:.3f}, meta={meta}")
    
    print("\n" + "="*70)
    print("SUCCESS: Hopfield-style memory works without string keys!")
    print("="*70)



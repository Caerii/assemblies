# hyperdimensional_assembly.py

"""
Integration of Assembly Calculus with Hyperdimensional Computing.

This module combines the biological plausibility of neural assemblies
with the mathematical elegance of hyperdimensional computing to create
a powerful computational framework.

Key Concepts:
- Assembly Calculus: Sparse neural assemblies with projection, association, merge
- HDC: High-dimensional vectors with binding, superposition, permutation
- Integration: Assembly operations implemented using HDC primitives
"""

import numpy as np
from typing import List, Tuple, Optional, Union
from .neural_computation import NeuralComputationEngine
from .statistics import StatisticalEngine

class HyperdimensionalAssembly:
    """
    Integration of Assembly Calculus with Hyperdimensional Computing.
    
    This class implements assembly operations using hyperdimensional
    computing primitives, combining biological plausibility with
    mathematical rigor.
    """
    
    def __init__(self, brain, dimension: int = 10000, rng: np.random.Generator = None):
        """
        Initialize hyperdimensional assembly system.
        
        Args:
            brain: Neural assembly brain instance
            dimension: Dimension of hypervectors (typically 10,000)
            rng: Random number generator for reproducibility
        """
        self.brain = brain
        self.dimension = dimension
        self.rng = rng or np.random.default_rng()
        
        # Initialize HDC primitives
        self._init_hdc_primitives()
        
        # Map assembly operations to HDC operations
        self.assembly_to_hdc = {
            'projection': self._assembly_projection_to_hdc,
            'association': self._assembly_association_to_hdc,
            'merge': self._assembly_merge_to_hdc
        }
    
    def _init_hdc_primitives(self):
        """Initialize HDC primitive vectors and operations."""
        # Create random hypervectors for binding operations
        self.binding_vectors = self._create_random_hypervectors(1000)
        
        # Create permutation matrices for sequence operations
        self.permutation_matrices = self._create_permutation_matrices(10)
        
        # Initialize similarity threshold
        self.similarity_threshold = 0.7
    
    def _create_random_hypervectors(self, count: int) -> np.ndarray:
        """Create random hypervectors for HDC operations."""
        # Generate random hypervectors with balanced +1/-1 components
        vectors = self.rng.choice([-1, 1], size=(count, self.dimension))
        return vectors
    
    def _create_permutation_matrices(self, count: int) -> List[np.ndarray]:
        """Create permutation matrices for sequence operations."""
        matrices = []
        for i in range(count):
            # Create circular shift permutation
            shift = i + 1
            matrix = np.roll(np.eye(self.dimension), shift, axis=1)
            matrices.append(matrix)
        return matrices
    
    def encode_assembly_as_hypervector(self, assembly: np.ndarray) -> np.ndarray:
        """
        Convert neural assembly to hyperdimensional vector.
        
        Args:
            assembly: Sparse neural assembly (indices of active neurons)
            
        Returns:
            Hyperdimensional vector representing the assembly
        """
        # Initialize zero hypervector
        hypervector = np.zeros(self.dimension)
        
        # Map assembly neurons to hypervector components
        for neuron_idx in assembly:
            # Use modular arithmetic to map to hypervector dimensions
            hdv_idx = neuron_idx % self.dimension
            hypervector[hdv_idx] = 1.0
        
        # Normalize to maintain similarity properties
        hypervector = hypervector / np.linalg.norm(hypervector)
        
        return hypervector
    
    def decode_hypervector_to_assembly(self, hypervector: np.ndarray, k: int) -> np.ndarray:
        """
        Convert hyperdimensional vector back to neural assembly.
        
        Args:
            hypervector: Hyperdimensional vector
            k: Number of neurons in target assembly
            
        Returns:
            Sparse neural assembly
        """
        # Find top-k components
        top_indices = np.argsort(-np.abs(hypervector))[:k]
        
        # Convert back to neuron indices
        assembly = top_indices
        
        return assembly
    
    def _assembly_projection_to_hdc(self, source_assembly: np.ndarray, 
                                  target_assembly: np.ndarray) -> np.ndarray:
        """
        Implement assembly projection using HDC binding.
        
        Assembly projection: A → B
        HDC equivalent: B = A ⊗ R (where R is a random binding vector)
        """
        # Convert assemblies to hypervectors
        source_hv = self.encode_assembly_as_hypervector(source_assembly)
        
        # Select random binding vector
        binding_vector = self.rng.choice(self.binding_vectors)
        
        # Perform binding operation (element-wise multiplication)
        projected_hv = source_hv * binding_vector
        
        # Convert back to assembly
        projected_assembly = self.decode_hypervector_to_assembly(projected_hv, len(target_assembly))
        
        return projected_assembly
    
    def _assembly_association_to_hdc(self, assembly_a: np.ndarray, 
                                   assembly_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Implement assembly association using HDC superposition.
        
        Assembly association: A + B → A' + B' (increased overlap)
        HDC equivalent: A' = A + α*B, B' = B + α*A (weighted superposition)
        """
        # Convert assemblies to hypervectors
        hv_a = self.encode_assembly_as_hypervector(assembly_a)
        hv_b = self.encode_assembly_as_hypervector(assembly_b)
        
        # Compute association strength based on current overlap
        overlap = np.dot(hv_a, hv_b)
        alpha = 0.1 * (1 - overlap)  # Stronger association for less similar assemblies
        
        # Perform weighted superposition
        hv_a_new = hv_a + alpha * hv_b
        hv_b_new = hv_b + alpha * hv_a
        
        # Normalize to maintain hypervector properties
        hv_a_new = hv_a_new / np.linalg.norm(hv_a_new)
        hv_b_new = hv_b_new / np.linalg.norm(hv_b_new)
        
        # Convert back to assemblies
        assembly_a_new = self.decode_hypervector_to_assembly(hv_a_new, len(assembly_a))
        assembly_b_new = self.decode_hypervector_to_assembly(hv_b_new, len(assembly_b))
        
        return assembly_a_new, assembly_b_new
    
    def _assembly_merge_to_hdc(self, assembly_a: np.ndarray, 
                             assembly_b: np.ndarray) -> np.ndarray:
        """
        Implement assembly merge using HDC superposition.
        
        Assembly merge: A + B → C
        HDC equivalent: C = A + B (superposition)
        """
        # Convert assemblies to hypervectors
        hv_a = self.encode_assembly_as_hypervector(assembly_a)
        hv_b = self.encode_assembly_as_hypervector(assembly_b)
        
        # Perform superposition
        merged_hv = hv_a + hv_b
        
        # Normalize
        merged_hv = merged_hv / np.linalg.norm(merged_hv)
        
        # Convert back to assembly
        merged_assembly = self.decode_hypervector_to_assembly(merged_hv, len(assembly_a))
        
        return merged_assembly
    
    def compute_similarity(self, assembly_a: np.ndarray, assembly_b: np.ndarray) -> float:
        """
        Compute similarity between two assemblies using HDC cosine similarity.
        
        Args:
            assembly_a: First assembly
            assembly_b: Second assembly
            
        Returns:
            Similarity score between 0 and 1
        """
        # Convert to hypervectors
        hv_a = self.encode_assembly_as_hypervector(assembly_a)
        hv_b = self.encode_assembly_as_hypervector(assembly_b)
        
        # Compute cosine similarity
        similarity = np.dot(hv_a, hv_b) / (np.linalg.norm(hv_a) * np.linalg.norm(hv_b))
        
        return max(0, similarity)  # Ensure non-negative
    
    def sequence_encoding(self, sequence_assemblies: List[np.ndarray]) -> np.ndarray:
        """
        Encode a sequence of assemblies using HDC permutation.
        
        Args:
            sequence_assemblies: List of assemblies in sequence
            
        Returns:
            Single assembly representing the sequence
        """
        if not sequence_assemblies:
            return np.array([])
        
        # Convert first assembly to hypervector
        sequence_hv = self.encode_assembly_as_hypervector(sequence_assemblies[0])
        
        # Apply permutations for subsequent assemblies
        for i, assembly in enumerate(sequence_assemblies[1:], 1):
            # Convert to hypervector
            hv = self.encode_assembly_as_hypervector(assembly)
            
            # Apply permutation based on position
            perm_matrix = self.permutation_matrices[i % len(self.permutation_matrices)]
            permuted_hv = perm_matrix @ hv
            
            # Superpose with sequence
            sequence_hv = sequence_hv + permuted_hv
        
        # Normalize
        sequence_hv = sequence_hv / np.linalg.norm(sequence_hv)
        
        # Convert back to assembly
        sequence_assembly = self.decode_hypervector_to_assembly(sequence_hv, len(sequence_assemblies[0]))
        
        return sequence_assembly
    
    def sequence_decoding(self, sequence_assembly: np.ndarray, 
                         expected_length: int) -> List[np.ndarray]:
        """
        Decode a sequence assembly back to individual assemblies.
        
        Args:
            sequence_assembly: Assembly representing the sequence
            expected_length: Expected length of the sequence
            
        Returns:
            List of individual assemblies
        """
        # Convert to hypervector
        sequence_hv = self.encode_assembly_as_hypervector(sequence_assembly)
        
        decoded_assemblies = []
        
        for i in range(expected_length):
            # Apply inverse permutation
            perm_matrix = self.permutation_matrices[i % len(self.permutation_matrices)]
            inverse_perm = perm_matrix.T  # Transpose for inverse
            
            # Extract assembly at position i
            extracted_hv = inverse_perm @ sequence_hv
            
            # Convert back to assembly
            assembly = self.decode_hypervector_to_assembly(extracted_hv, len(sequence_assembly))
            decoded_assemblies.append(assembly)
        
        return decoded_assemblies
    
    def hierarchical_encoding(self, hierarchy: dict) -> np.ndarray:
        """
        Encode hierarchical structures using HDC binding and superposition.
        
        Args:
            hierarchy: Nested dictionary representing hierarchy
            
        Returns:
            Assembly representing the hierarchy
        """
        def encode_node(node, level=0):
            if isinstance(node, dict):
                # Internal node - bind children
                children_hvs = [encode_node(child, level + 1) for child in node.values()]
                if children_hvs:
                    # Bind all children
                    result_hv = children_hvs[0]
                    for child_hv in children_hvs[1:]:
                        result_hv = result_hv * child_hv
                    return result_hv
                else:
                    return np.zeros(self.dimension)
            else:
                # Leaf node - encode as assembly
                if hasattr(node, '__iter__') and not isinstance(node, str):
                    # It's already an assembly
                    return self.encode_assembly_as_hypervector(node)
                else:
                    # Convert to assembly first
                    assembly = np.array([hash(str(node)) % self.dimension])
                    return self.encode_assembly_as_hypervector(assembly)
        
        # Encode the hierarchy
        hierarchy_hv = encode_node(hierarchy)
        
        # Convert back to assembly
        hierarchy_assembly = self.decode_hypervector_to_assembly(hierarchy_hv, 100)
        
        return hierarchy_assembly
    
    def analogical_reasoning(self, source_assemblies: List[np.ndarray], 
                           target_assemblies: List[np.ndarray]) -> np.ndarray:
        """
        Perform analogical reasoning using HDC operations.
        
        Args:
            source_assemblies: Source domain assemblies
            target_assemblies: Target domain assemblies
            
        Returns:
            Assembly representing the analogy
        """
        if len(source_assemblies) != len(target_assemblies):
            raise ValueError("Source and target must have same number of assemblies")
        
        # Convert to hypervectors
        source_hvs = [self.encode_assembly_as_hypervector(a) for a in source_assemblies]
        target_hvs = [self.encode_assembly_as_hypervector(a) for a in target_assemblies]
        
        # Compute transformation from source to target
        # T = Σ(target_i ⊗ source_i) / |source_i|²
        transformation = np.zeros((self.dimension, self.dimension))
        
        for source_hv, target_hv in zip(source_hvs, target_hvs):
            # Outer product for transformation
            transformation += np.outer(target_hv, source_hv)
        
        # Normalize transformation
        transformation = transformation / np.linalg.norm(transformation)
        
        # Apply transformation to create analogy
        analogy_hv = transformation @ source_hvs[0]  # Apply to first source
        
        # Convert back to assembly
        analogy_assembly = self.decode_hypervector_to_assembly(analogy_hv, len(source_assemblies[0]))
        
        return analogy_assembly

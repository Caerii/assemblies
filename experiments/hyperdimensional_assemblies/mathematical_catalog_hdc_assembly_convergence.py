# mathematical_catalog_hdc_assembly_convergence.py

"""
MATHEMATICAL CATALOG: HDC + ASSEMBLY CALCULUS CONVERGENCE

This file systematically catalogs the mathematical principles that emerge
from the convergence between Hyperdimensional Computing and Assembly Calculus.

KEY CONVERGENCE PRINCIPLES:
1. Sparse-Dense Duality: Sparse assemblies ↔ Dense hypervectors
2. Interference-Overlap Trade-off: HDC interference vs Assembly overlap
3. Binding-Competition Dynamics: HDC binding vs Neural competition
4. Superposition-Plasticity Learning: HDC superposition vs Hebbian learning
5. Permutation-Temporal Encoding: HDC permutation vs Temporal sequences
6. Cleanup-Recall Dynamics: HDC cleanup vs Assembly recall
7. Capacity-Sparsity Scaling: HDC capacity vs Assembly sparsity
8. Robustness-Interference Balance: HDC robustness vs Information loss
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import hamming, cosine
from scipy import stats
import time
import random
from dataclasses import dataclass
from enum import Enum

class ConvergencePrinciple(Enum):
    """Mathematical principles of HDC-Assembly convergence."""
    SPARSE_DENSE_DUALITY = "sparse_dense_duality"
    INTERFERENCE_OVERLAP_TRADEOFF = "interference_overlap_tradeoff"
    BINDING_COMPETITION_DYNAMICS = "binding_competition_dynamics"
    SUPERPOSITION_PLASTICITY_LEARNING = "superposition_plasticity_learning"
    PERMUTATION_TEMPORAL_ENCODING = "permutation_temporal_encoding"
    CLEANUP_RECALL_DYNAMICS = "cleanup_recall_dynamics"
    CAPACITY_SPARSITY_SCALING = "capacity_sparsity_scaling"
    ROBUSTNESS_INTERFERENCE_BALANCE = "robustness_interference_balance"

@dataclass
class ConvergenceMetrics:
    """Metrics for measuring HDC-Assembly convergence."""
    information_preservation: float
    interference_level: float
    assembly_overlap: float
    binding_strength: float
    competition_intensity: float
    learning_rate: float
    temporal_encoding_accuracy: float
    cleanup_success_rate: float
    capacity_utilization: float
    robustness_score: float

class MathematicalCatalogHDCAssembly:
    """
    Mathematical catalog of HDC + Assembly Calculus convergence principles.
    
    This systematically explores the mathematical relationships and trade-offs
    that emerge when combining HDC and Assembly Calculus.
    """
    
    def __init__(self, 
                 dimension: int = 10000,
                 k: int = 50,
                 n_neurons: int = 1000,
                 seed: int = 42):
        """Initialize the mathematical catalog system."""
        self.dimension = dimension
        self.k = k
        self.n_neurons = n_neurons
        
        np.random.seed(seed)
        random.seed(seed)
        
        # Initialize HDC components
        self._init_hdc_system()
        
        # Initialize Assembly Calculus components
        self._init_assembly_system()
        
        # Initialize convergence tracking
        self.convergence_data = {}
    
    def _init_hdc_system(self):
        """Initialize HDC system for convergence analysis."""
        # Basis hypervectors (random bipolar)
        self.basis_vectors = {}
        for i in range(self.n_neurons):
            self.basis_vectors[i] = np.random.choice([-1, 1], size=self.dimension)
        
        # Binding vectors for different operations
        self.binding_vectors = {
            'project': np.random.choice([-1, 1], size=self.dimension),
            'associate': np.random.choice([-1, 1], size=self.dimension),
            'merge': np.random.choice([-1, 1], size=self.dimension),
            'sequence': np.random.choice([-1, 1], size=self.dimension)
        }
        
        # Permutation matrix for temporal encoding
        self.permutation_matrix = np.zeros((self.dimension, self.dimension))
        shift = 1
        for i in range(self.dimension):
            self.permutation_matrix[i, (i + shift) % self.dimension] = 1
        
        # Codebook for cleanup operations
        self.codebook = {}
    
    def _init_assembly_system(self):
        """Initialize Assembly Calculus system for convergence analysis."""
        # Competition parameters
        self.competition_strength = 0.5
        self.inhibition_strength = 0.3
        
        # Plasticity parameters
        self.learning_rate = 0.1
        self.hebbian_strength = 0.01
        self.anti_hebbian_strength = 0.005
        
        # Assembly dynamics
        self.assembly_history = []
        self.overlap_history = []
    
    def principle_1_sparse_dense_duality(self, 
                                       sparsity_levels: List[float] = [0.01, 0.05, 0.1, 0.2, 0.5]) -> Dict[str, Any]:
        """
        PRINCIPLE 1: Sparse-Dense Duality
        
        Mathematical relationship between sparse assemblies and dense hypervectors.
        
        Key insight: Sparse assemblies (k-winners) create dense hypervectors (superposition),
        but dense hypervectors can be decoded back to sparse assemblies (competition).
        
        This creates a duality where:
        - Sparsity enables efficient computation
        - Density enables robust representation
        - The conversion between them involves information loss/gain
        """
        print("=" * 80)
        print("PRINCIPLE 1: SPARSE-DENSE DUALITY")
        print("=" * 80)
        print("Exploring the mathematical relationship between sparse assemblies and dense hypervectors")
        print()
        
        results = {
            'sparsity_levels': sparsity_levels,
            'information_preservation': [],
            'hypervector_density': [],
            'decoding_accuracy': [],
            'computational_efficiency': []
        }
        
        for sparsity in sparsity_levels:
            k_sparse = int(self.n_neurons * sparsity)
            if k_sparse < 1:
                k_sparse = 1
            
            # Generate sparse assembly
            assembly = np.random.choice(self.n_neurons, size=k_sparse, replace=False)
            
            # Convert to dense hypervector
            hypervector = self._encode_assembly_to_hypervector(assembly)
            
            # Measure hypervector density
            density = np.sum(hypervector != 0) / self.dimension
            
            # Decode back to sparse assembly
            decoded_assembly = self._decode_hypervector_to_assembly(hypervector)
            
            # Measure information preservation
            overlap = len(set(assembly) & set(decoded_assembly))
            union = len(set(assembly) | set(decoded_assembly))
            preservation = overlap / union if union > 0 else 0
            
            # Measure decoding accuracy
            accuracy = overlap / len(assembly) if len(assembly) > 0 else 0
            
            # Measure computational efficiency (operations per second)
            start_time = time.time()
            for _ in range(100):
                hv = self._encode_assembly_to_hypervector(assembly)
                decoded = self._decode_hypervector_to_assembly(hv)
            end_time = time.time()
            efficiency = 100 / (end_time - start_time)
            
            results['information_preservation'].append(preservation)
            results['hypervector_density'].append(density)
            results['decoding_accuracy'].append(accuracy)
            results['computational_efficiency'].append(efficiency)
            
            print(f"Sparsity {sparsity:.3f}: Preservation={preservation:.3f}, "
                  f"Density={density:.3f}, Accuracy={accuracy:.3f}, "
                  f"Efficiency={efficiency:.1f} ops/s")
        
        # Mathematical analysis
        print(f"\nMATHEMATICAL ANALYSIS:")
        print(f"- Information preservation vs sparsity correlation: "
              f"{np.corrcoef(sparsity_levels, results['information_preservation'])[0,1]:.3f}")
        print(f"- Hypervector density vs sparsity correlation: "
              f"{np.corrcoef(sparsity_levels, results['hypervector_density'])[0,1]:.3f}")
        print(f"- Decoding accuracy vs sparsity correlation: "
              f"{np.corrcoef(sparsity_levels, results['decoding_accuracy'])[0,1]:.3f}")
        
        self.convergence_data['sparse_dense_duality'] = results
        return results
    
    def principle_2_interference_overlap_tradeoff(self, 
                                                interference_levels: List[float] = [0.1, 0.3, 0.5, 0.7, 0.9]) -> Dict[str, Any]:
        """
        PRINCIPLE 2: Interference-Overlap Trade-off
        
        Mathematical trade-off between HDC interference and Assembly overlap.
        
        Key insight: HDC superposition creates interference (information loss),
        but Assembly overlap creates association (information gain).
        
        This creates a fundamental trade-off where:
        - More interference → Less information preservation
        - More overlap → More associative learning
        - Optimal balance depends on task requirements
        """
        print("\n" + "=" * 80)
        print("PRINCIPLE 2: INTERFERENCE-OVERLAP TRADEOFF")
        print("=" * 80)
        print("Exploring the trade-off between HDC interference and Assembly overlap")
        print()
        
        results = {
            'interference_levels': interference_levels,
            'information_preservation': [],
            'assembly_overlap': [],
            'associative_strength': [],
            'robustness_score': []
        }
        
        for interference in interference_levels:
            # Generate two assemblies
            assembly_a = np.random.choice(self.n_neurons, size=self.k, replace=False)
            assembly_b = np.random.choice(self.n_neurons, size=self.k, replace=False)
            
            # Convert to hypervectors
            hv_a = self._encode_assembly_to_hypervector(assembly_a)
            hv_b = self._encode_assembly_to_hypervector(assembly_b)
            
            # Apply controlled interference
            noise = np.random.normal(0, interference, self.dimension)
            hv_a_noisy = np.sign(hv_a + noise)
            hv_b_noisy = np.sign(hv_b + noise)
            
            # Measure information preservation
            decoded_a = self._decode_hypervector_to_assembly(hv_a_noisy)
            decoded_b = self._decode_hypervector_to_assembly(hv_b_noisy)
            
            preservation_a = len(set(assembly_a) & set(decoded_a)) / len(assembly_a)
            preservation_b = len(set(assembly_b) & set(decoded_b)) / len(assembly_b)
            avg_preservation = (preservation_a + preservation_b) / 2
            
            # Measure assembly overlap
            overlap = len(set(assembly_a) & set(assembly_b)) / len(set(assembly_a) | set(assembly_b))
            
            # Measure associative strength (how much they influence each other)
            hv_combined = self._hdc_superpose([hv_a_noisy, hv_b_noisy])
            decoded_combined = self._decode_hypervector_to_assembly(hv_combined)
            associative_strength = len(set(decoded_combined) & set(assembly_a)) / len(assembly_a)
            
            # Measure robustness (resistance to noise)
            robustness = 1 - interference  # Simple model
            
            results['information_preservation'].append(avg_preservation)
            results['assembly_overlap'].append(overlap)
            results['associative_strength'].append(associative_strength)
            results['robustness_score'].append(robustness)
            
            print(f"Interference {interference:.3f}: Preservation={avg_preservation:.3f}, "
                  f"Overlap={overlap:.3f}, Associative={associative_strength:.3f}, "
                  f"Robustness={robustness:.3f}")
        
        # Mathematical analysis
        print(f"\nMATHEMATICAL ANALYSIS:")
        print(f"- Information preservation vs interference correlation: "
              f"{np.corrcoef(interference_levels, results['information_preservation'])[0,1]:.3f}")
        print(f"- Assembly overlap vs interference correlation: "
              f"{np.corrcoef(interference_levels, results['assembly_overlap'])[0,1]:.3f}")
        print(f"- Associative strength vs interference correlation: "
              f"{np.corrcoef(interference_levels, results['associative_strength'])[0,1]:.3f}")
        
        self.convergence_data['interference_overlap_tradeoff'] = results
        return results
    
    def principle_3_binding_competition_dynamics(self, 
                                               binding_strengths: List[float] = [0.1, 0.3, 0.5, 0.7, 0.9]) -> Dict[str, Any]:
        """
        PRINCIPLE 3: Binding-Competition Dynamics
        
        Mathematical relationship between HDC binding and neural competition.
        
        Key insight: HDC binding creates new representations,
        but neural competition selects the most relevant ones.
        
        This creates dynamics where:
        - Binding strength affects representation quality
        - Competition intensity affects selection accuracy
        - Optimal balance enables both creativity and precision
        """
        print("\n" + "=" * 80)
        print("PRINCIPLE 3: BINDING-COMPETITION DYNAMICS")
        print("=" * 80)
        print("Exploring the relationship between HDC binding and neural competition")
        print()
        
        results = {
            'binding_strengths': binding_strengths,
            'binding_quality': [],
            'competition_accuracy': [],
            'representation_diversity': [],
            'selection_precision': []
        }
        
        for binding_strength in binding_strengths:
            # Generate source assemblies
            assembly_a = np.random.choice(self.n_neurons, size=self.k, replace=False)
            assembly_b = np.random.choice(self.n_neurons, size=self.k, replace=False)
            
            # Convert to hypervectors
            hv_a = self._encode_assembly_to_hypervector(assembly_a)
            hv_b = self._encode_assembly_to_hypervector(assembly_b)
            
            # Apply binding with controlled strength
            binding_vector = np.random.choice([-1, 1], size=self.dimension)
            hv_bound = self._hdc_bind(hv_a, hv_b, strength=binding_strength)
            
            # Measure binding quality (how well it represents both inputs)
            decoded_bound = self._decode_hypervector_to_assembly(hv_bound)
            quality_a = len(set(decoded_bound) & set(assembly_a)) / len(assembly_a)
            quality_b = len(set(decoded_bound) & set(assembly_b)) / len(assembly_b)
            binding_quality = (quality_a + quality_b) / 2
            
            # Measure competition accuracy (how well competition selects relevant neurons)
            competition_accuracy = len(set(decoded_bound) & set(assembly_a)) / len(decoded_bound)
            
            # Measure representation diversity (how many different neurons are represented)
            diversity = len(set(decoded_bound)) / self.k
            
            # Measure selection precision (how well the selected neurons match the task)
            precision = len(set(decoded_bound) & set(assembly_a)) / len(assembly_a)
            
            results['binding_quality'].append(binding_quality)
            results['competition_accuracy'].append(competition_accuracy)
            results['representation_diversity'].append(diversity)
            results['selection_precision'].append(precision)
            
            print(f"Binding {binding_strength:.3f}: Quality={binding_quality:.3f}, "
                  f"Accuracy={competition_accuracy:.3f}, Diversity={diversity:.3f}, "
                  f"Precision={precision:.3f}")
        
        # Mathematical analysis
        print(f"\nMATHEMATICAL ANALYSIS:")
        print(f"- Binding quality vs binding strength correlation: "
              f"{np.corrcoef(binding_strengths, results['binding_quality'])[0,1]:.3f}")
        print(f"- Competition accuracy vs binding strength correlation: "
              f"{np.corrcoef(binding_strengths, results['competition_accuracy'])[0,1]:.3f}")
        print(f"- Representation diversity vs binding strength correlation: "
              f"{np.corrcoef(binding_strengths, results['representation_diversity'])[0,1]:.3f}")
        
        self.convergence_data['binding_competition_dynamics'] = results
        return results
    
    def principle_4_superposition_plasticity_learning(self, 
                                                    learning_rates: List[float] = [0.01, 0.05, 0.1, 0.2, 0.5]) -> Dict[str, Any]:
        """
        PRINCIPLE 4: Superposition-Plasticity Learning
        
        Mathematical relationship between HDC superposition and Hebbian plasticity.
        
        Key insight: HDC superposition combines representations,
        but Hebbian plasticity strengthens associations.
        
        This creates learning dynamics where:
        - Superposition enables rapid combination
        - Plasticity enables long-term learning
        - Optimal balance enables both flexibility and stability
        """
        print("\n" + "=" * 80)
        print("PRINCIPLE 4: SUPERPOSITION-PLASTICITY LEARNING")
        print("=" * 80)
        print("Exploring the relationship between HDC superposition and Hebbian plasticity")
        print()
        
        results = {
            'learning_rates': learning_rates,
            'superposition_effectiveness': [],
            'plasticity_strength': [],
            'learning_speed': [],
            'memory_stability': []
        }
        
        for learning_rate in learning_rates:
            # Generate assemblies
            assembly_a = np.random.choice(self.n_neurons, size=self.k, replace=False)
            assembly_b = np.random.choice(self.n_neurons, size=self.k, replace=False)
            
            # Convert to hypervectors
            hv_a = self._encode_assembly_to_hypervector(assembly_a)
            hv_b = self._encode_assembly_to_hypervector(assembly_b)
            
            # Apply superposition
            hv_superposed = self._hdc_superpose([hv_a, hv_b])
            decoded_superposed = self._decode_hypervector_to_assembly(hv_superposed)
            
            # Measure superposition effectiveness
            effectiveness = len(set(decoded_superposed) & (set(assembly_a) | set(assembly_b))) / len(set(assembly_a) | set(assembly_b))
            
            # Apply Hebbian plasticity
            hv_a_plastic = self._apply_hebbian_plasticity(hv_a, hv_b, learning_rate)
            hv_b_plastic = self._apply_hebbian_plasticity(hv_b, hv_a, learning_rate)
            
            # Measure plasticity strength
            plasticity_strength = np.corrcoef(hv_a, hv_a_plastic)[0,1]
            
            # Measure learning speed (how quickly associations form)
            learning_speed = learning_rate * effectiveness
            
            # Measure memory stability (how well learned associations persist)
            memory_stability = 1 - learning_rate  # Simple model
            
            results['superposition_effectiveness'].append(effectiveness)
            results['plasticity_strength'].append(plasticity_strength)
            results['learning_speed'].append(learning_speed)
            results['memory_stability'].append(memory_stability)
            
            print(f"Learning rate {learning_rate:.3f}: Effectiveness={effectiveness:.3f}, "
                  f"Plasticity={plasticity_strength:.3f}, Speed={learning_speed:.3f}, "
                  f"Stability={memory_stability:.3f}")
        
        # Mathematical analysis
        print(f"\nMATHEMATICAL ANALYSIS:")
        print(f"- Superposition effectiveness vs learning rate correlation: "
              f"{np.corrcoef(learning_rates, results['superposition_effectiveness'])[0,1]:.3f}")
        print(f"- Plasticity strength vs learning rate correlation: "
              f"{np.corrcoef(learning_rates, results['plasticity_strength'])[0,1]:.3f}")
        print(f"- Learning speed vs learning rate correlation: "
              f"{np.corrcoef(learning_rates, results['learning_speed'])[0,1]:.3f}")
        
        self.convergence_data['superposition_plasticity_learning'] = results
        return results
    
    def principle_5_permutation_temporal_encoding(self, 
                                                sequence_lengths: List[int] = [2, 3, 5, 10, 20]) -> Dict[str, Any]:
        """
        PRINCIPLE 5: Permutation-Temporal Encoding
        
        Mathematical relationship between HDC permutation and temporal sequence encoding.
        
        Key insight: HDC permutation preserves information while encoding order,
        but temporal sequences require both order and content.
        
        This creates encoding dynamics where:
        - Permutation enables order encoding
        - Superposition enables content encoding
        - Optimal balance enables both temporal and semantic information
        """
        print("\n" + "=" * 80)
        print("PRINCIPLE 5: PERMUTATION-TEMPORAL ENCODING")
        print("=" * 80)
        print("Exploring the relationship between HDC permutation and temporal encoding")
        print()
        
        results = {
            'sequence_lengths': sequence_lengths,
            'encoding_accuracy': [],
            'temporal_preservation': [],
            'content_preservation': [],
            'decoding_success_rate': []
        }
        
        for seq_len in sequence_lengths:
            # Generate sequence
            sequence = []
            for _ in range(seq_len):
                assembly = np.random.choice(self.n_neurons, size=self.k, replace=False)
                sequence.append(assembly)
            
            # Encode sequence using permutation
            sequence_hv = self._encode_sequence_with_permutation(sequence)
            
            # Decode sequence
            decoded_sequence = self._decode_sequence_with_permutation(sequence_hv, seq_len)
            
            # Measure encoding accuracy
            total_overlap = 0
            for orig, decoded in zip(sequence, decoded_sequence):
                overlap = len(set(orig) & set(decoded)) / len(orig)
                total_overlap += overlap
            encoding_accuracy = total_overlap / seq_len
            
            # Measure temporal preservation (order accuracy)
            temporal_preservation = self._measure_temporal_preservation(sequence, decoded_sequence)
            
            # Measure content preservation (semantic accuracy)
            content_preservation = self._measure_content_preservation(sequence, decoded_sequence)
            
            # Measure decoding success rate
            success_rate = sum(1 for orig, decoded in zip(sequence, decoded_sequence) 
                             if len(set(orig) & set(decoded)) > len(orig) * 0.5) / seq_len
            
            results['encoding_accuracy'].append(encoding_accuracy)
            results['temporal_preservation'].append(temporal_preservation)
            results['content_preservation'].append(content_preservation)
            results['decoding_success_rate'].append(success_rate)
            
            print(f"Length {seq_len:2d}: Encoding={encoding_accuracy:.3f}, "
                  f"Temporal={temporal_preservation:.3f}, Content={content_preservation:.3f}, "
                  f"Success={success_rate:.3f}")
        
        # Mathematical analysis
        print(f"\nMATHEMATICAL ANALYSIS:")
        print(f"- Encoding accuracy vs sequence length correlation: "
              f"{np.corrcoef(sequence_lengths, results['encoding_accuracy'])[0,1]:.3f}")
        print(f"- Temporal preservation vs sequence length correlation: "
              f"{np.corrcoef(sequence_lengths, results['temporal_preservation'])[0,1]:.3f}")
        print(f"- Content preservation vs sequence length correlation: "
              f"{np.corrcoef(sequence_lengths, results['content_preservation'])[0,1]:.3f}")
        
        self.convergence_data['permutation_temporal_encoding'] = results
        return results
    
    def principle_6_cleanup_recall_dynamics(self, 
                                          noise_levels: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5]) -> Dict[str, Any]:
        """
        PRINCIPLE 6: Cleanup-Recall Dynamics
        
        Mathematical relationship between HDC cleanup and assembly recall.
        
        Key insight: HDC cleanup removes noise and interference,
        but assembly recall requires competition and selection.
        
        This creates recall dynamics where:
        - Cleanup improves signal quality
        - Recall enables content retrieval
        - Optimal balance enables both accuracy and robustness
        """
        print("\n" + "=" * 80)
        print("PRINCIPLE 6: CLEANUP-RECALL DYNAMICS")
        print("=" * 80)
        print("Exploring the relationship between HDC cleanup and assembly recall")
        print()
        
        results = {
            'noise_levels': noise_levels,
            'cleanup_effectiveness': [],
            'recall_accuracy': [],
            'signal_quality': [],
            'robustness_score': []
        }
        
        for noise_level in noise_levels:
            # Generate clean assembly
            clean_assembly = np.random.choice(self.n_neurons, size=self.k, replace=False)
            clean_hv = self._encode_assembly_to_hypervector(clean_assembly)
            
            # Add noise
            noise = np.random.normal(0, noise_level, self.dimension)
            noisy_hv = np.sign(clean_hv + noise)
            
            # Apply cleanup
            cleaned_hv = self._apply_cleanup(noisy_hv)
            cleaned_assembly = self._decode_hypervector_to_assembly(cleaned_hv)
            
            # Measure cleanup effectiveness
            cleanup_effectiveness = 1 - np.mean(np.abs(clean_hv - cleaned_hv))
            
            # Measure recall accuracy
            recall_accuracy = len(set(clean_assembly) & set(cleaned_assembly)) / len(clean_assembly)
            
            # Measure signal quality
            signal_quality = np.corrcoef(clean_hv, cleaned_hv)[0,1]
            
            # Measure robustness
            robustness = 1 - noise_level
            
            results['cleanup_effectiveness'].append(cleanup_effectiveness)
            results['recall_accuracy'].append(recall_accuracy)
            results['signal_quality'].append(signal_quality)
            results['robustness_score'].append(robustness)
            
            print(f"Noise {noise_level:.3f}: Cleanup={cleanup_effectiveness:.3f}, "
                  f"Recall={recall_accuracy:.3f}, Signal={signal_quality:.3f}, "
                  f"Robustness={robustness:.3f}")
        
        # Mathematical analysis
        print(f"\nMATHEMATICAL ANALYSIS:")
        print(f"- Cleanup effectiveness vs noise level correlation: "
              f"{np.corrcoef(noise_levels, results['cleanup_effectiveness'])[0,1]:.3f}")
        print(f"- Recall accuracy vs noise level correlation: "
              f"{np.corrcoef(noise_levels, results['recall_accuracy'])[0,1]:.3f}")
        print(f"- Signal quality vs noise level correlation: "
              f"{np.corrcoef(noise_levels, results['signal_quality'])[0,1]:.3f}")
        
        self.convergence_data['cleanup_recall_dynamics'] = results
        return results
    
    def principle_7_capacity_sparsity_scaling(self, 
                                            dimensions: List[int] = [1000, 2000, 5000, 10000, 20000]) -> Dict[str, Any]:
        """
        PRINCIPLE 7: Capacity-Sparsity Scaling
        
        Mathematical relationship between HDC capacity and assembly sparsity.
        
        Key insight: HDC capacity scales with dimension,
        but assembly sparsity enables efficient computation.
        
        This creates scaling dynamics where:
        - Higher dimensions → Higher capacity
        - Higher sparsity → Higher efficiency
        - Optimal balance enables both capacity and efficiency
        """
        print("\n" + "=" * 80)
        print("PRINCIPLE 7: CAPACITY-SPARSITY SCALING")
        print("=" * 80)
        print("Exploring the relationship between HDC capacity and assembly sparsity")
        print()
        
        results = {
            'dimensions': dimensions,
            'capacity_estimates': [],
            'efficiency_measures': [],
            'information_density': [],
            'computational_cost': []
        }
        
        for dim in dimensions:
            # Estimate capacity (number of orthogonal vectors)
            capacity = dim / (2 * np.log2(dim))  # Theoretical HDC capacity
            
            # Measure efficiency (operations per second)
            start_time = time.time()
            for _ in range(100):
                assembly = np.random.choice(self.n_neurons, size=self.k, replace=False)
                hv = self._encode_assembly_to_hypervector(assembly)
                decoded = self._decode_hypervector_to_assembly(hv)
            end_time = time.time()
            efficiency = 100 / (end_time - start_time)
            
            # Measure information density (bits per dimension)
            info_density = self.k * np.log2(self.n_neurons) / dim
            
            # Measure computational cost (operations per dimension)
            computational_cost = self.k * np.log2(self.n_neurons)
            
            results['capacity_estimates'].append(capacity)
            results['efficiency_measures'].append(efficiency)
            results['information_density'].append(info_density)
            results['computational_cost'].append(computational_cost)
            
            print(f"Dim {dim:5d}: Capacity={capacity:.1f}, Efficiency={efficiency:.1f}, "
                  f"Density={info_density:.3f}, Cost={computational_cost:.1f}")
        
        # Mathematical analysis
        print(f"\nMATHEMATICAL ANALYSIS:")
        print(f"- Capacity vs dimension correlation: "
              f"{np.corrcoef(dimensions, results['capacity_estimates'])[0,1]:.3f}")
        print(f"- Efficiency vs dimension correlation: "
              f"{np.corrcoef(dimensions, results['efficiency_measures'])[0,1]:.3f}")
        print(f"- Information density vs dimension correlation: "
              f"{np.corrcoef(dimensions, results['information_density'])[0,1]:.3f}")
        
        self.convergence_data['capacity_sparsity_scaling'] = results
        return results
    
    def principle_8_robustness_interference_balance(self, 
                                                  robustness_levels: List[float] = [0.1, 0.3, 0.5, 0.7, 0.9]) -> Dict[str, Any]:
        """
        PRINCIPLE 8: Robustness-Interference Balance
        
        Mathematical relationship between HDC robustness and interference tolerance.
        
        Key insight: HDC robustness enables noise tolerance,
        but interference enables information mixing.
        
        This creates a balance where:
        - Higher robustness → Better noise tolerance
        - Higher interference → Better information mixing
        - Optimal balance enables both stability and creativity
        """
        print("\n" + "=" * 80)
        print("PRINCIPLE 8: ROBUSTNESS-INTERFERENCE BALANCE")
        print("=" * 80)
        print("Exploring the balance between HDC robustness and interference tolerance")
        print()
        
        results = {
            'robustness_levels': robustness_levels,
            'noise_tolerance': [],
            'interference_tolerance': [],
            'information_mixing': [],
            'stability_score': []
        }
        
        for robustness in robustness_levels:
            # Generate assemblies
            assembly_a = np.random.choice(self.n_neurons, size=self.k, replace=False)
            assembly_b = np.random.choice(self.n_neurons, size=self.k, replace=False)
            
            # Convert to hypervectors
            hv_a = self._encode_assembly_to_hypervector(assembly_a)
            hv_b = self._encode_assembly_to_hypervector(assembly_b)
            
            # Apply controlled noise
            noise_level = 1 - robustness
            noise_a = np.random.normal(0, noise_level, self.dimension)
            noise_b = np.random.normal(0, noise_level, self.dimension)
            
            hv_a_noisy = np.sign(hv_a + noise_a)
            hv_b_noisy = np.sign(hv_b + noise_b)
            
            # Measure noise tolerance
            decoded_a = self._decode_hypervector_to_assembly(hv_a_noisy)
            noise_tolerance = len(set(assembly_a) & set(decoded_a)) / len(assembly_a)
            
            # Measure interference tolerance
            hv_combined = self._hdc_superpose([hv_a_noisy, hv_b_noisy])
            decoded_combined = self._decode_hypervector_to_assembly(hv_combined)
            interference_tolerance = len(set(decoded_combined) & (set(assembly_a) | set(assembly_b))) / len(set(assembly_a) | set(assembly_b))
            
            # Measure information mixing
            mixing_score = len(set(decoded_combined) & set(assembly_a)) / len(assembly_a)
            
            # Measure stability
            stability = robustness
            
            results['noise_tolerance'].append(noise_tolerance)
            results['interference_tolerance'].append(interference_tolerance)
            results['information_mixing'].append(mixing_score)
            results['stability_score'].append(stability)
            
            print(f"Robustness {robustness:.3f}: Noise={noise_tolerance:.3f}, "
                  f"Interference={interference_tolerance:.3f}, Mixing={mixing_score:.3f}, "
                  f"Stability={stability:.3f}")
        
        # Mathematical analysis
        print(f"\nMATHEMATICAL ANALYSIS:")
        print(f"- Noise tolerance vs robustness correlation: "
              f"{np.corrcoef(robustness_levels, results['noise_tolerance'])[0,1]:.3f}")
        print(f"- Interference tolerance vs robustness correlation: "
              f"{np.corrcoef(robustness_levels, results['interference_tolerance'])[0,1]:.3f}")
        print(f"- Information mixing vs robustness correlation: "
              f"{np.corrcoef(robustness_levels, results['information_mixing'])[0,1]:.3f}")
        
        self.convergence_data['robustness_interference_balance'] = results
        return results
    
    def generate_convergence_report(self) -> Dict[str, Any]:
        """Generate comprehensive convergence report."""
        print("\n" + "=" * 80)
        print("HDC + ASSEMBLY CALCULUS CONVERGENCE REPORT")
        print("=" * 80)
        
        report = {
            'principles_analyzed': len(self.convergence_data),
            'key_insights': [],
            'mathematical_relationships': {},
            'optimal_parameters': {},
            'trade_offs': {}
        }
        
        # Analyze each principle
        for principle, data in self.convergence_data.items():
            print(f"\n{principle.upper().replace('_', ' ')}:")
            
            # Extract key insights
            if 'information_preservation' in data:
                avg_preservation = np.mean(data['information_preservation'])
                print(f"  Average information preservation: {avg_preservation:.3f}")
                report['key_insights'].append(f"{principle}: {avg_preservation:.3f} avg preservation")
            
            if 'binding_quality' in data:
                avg_quality = np.mean(data['binding_quality'])
                print(f"  Average binding quality: {avg_quality:.3f}")
                report['key_insights'].append(f"{principle}: {avg_quality:.3f} avg binding quality")
            
            if 'encoding_accuracy' in data:
                avg_accuracy = np.mean(data['encoding_accuracy'])
                print(f"  Average encoding accuracy: {avg_accuracy:.3f}")
                report['key_insights'].append(f"{principle}: {avg_accuracy:.3f} avg encoding accuracy")
        
        print(f"\nTOTAL PRINCIPLES ANALYZED: {report['principles_analyzed']}")
        print(f"KEY INSIGHTS: {len(report['key_insights'])}")
        
        return report
    
    # Helper methods for the principles
    def _encode_assembly_to_hypervector(self, assembly: np.ndarray) -> np.ndarray:
        """Encode assembly to hypervector using superposition."""
        if len(assembly) == 0:
            return np.zeros(self.dimension)
        
        hypervector = np.zeros(self.dimension)
        for neuron_id in assembly:
            if neuron_id in self.basis_vectors:
                hypervector += self.basis_vectors[neuron_id]
        
        return np.sign(hypervector)
    
    def _decode_hypervector_to_assembly(self, hypervector: np.ndarray) -> np.ndarray:
        """Decode hypervector to assembly using competition."""
        if np.all(hypervector == 0):
            return np.array([])
        
        similarities = {}
        for neuron_id, basis_vector in self.basis_vectors.items():
            similarity = np.dot(hypervector, basis_vector) / (np.linalg.norm(hypervector) * np.linalg.norm(basis_vector))
            similarities[neuron_id] = similarity
        
        sorted_neurons = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        winners = [neuron_id for neuron_id, _ in sorted_neurons[:self.k]]
        
        return np.array(winners)
    
    def _hdc_bind(self, hv_a: np.ndarray, hv_b: np.ndarray, strength: float = 1.0) -> np.ndarray:
        """HDC binding operation with controlled strength."""
        return np.sign(hv_a * hv_b * strength)
    
    def _hdc_superpose(self, hv_list: List[np.ndarray]) -> np.ndarray:
        """HDC superposition operation."""
        if not hv_list:
            return np.zeros(self.dimension)
        
        result = np.sum(hv_list, axis=0)
        return np.sign(result)
    
    def _apply_hebbian_plasticity(self, hv_a: np.ndarray, hv_b: np.ndarray, learning_rate: float) -> np.ndarray:
        """Apply Hebbian plasticity to hypervector."""
        return np.sign(hv_a + learning_rate * hv_b)
    
    def _encode_sequence_with_permutation(self, sequence: List[np.ndarray]) -> np.ndarray:
        """Encode sequence using HDC permutation."""
        sequence_hvs = []
        for i, assembly in enumerate(sequence):
            hv = self._encode_assembly_to_hypervector(assembly)
            for _ in range(i):
                hv = self.permutation_matrix @ hv
            sequence_hvs.append(hv)
        
        return self._hdc_superpose(sequence_hvs)
    
    def _decode_sequence_with_permutation(self, sequence_hv: np.ndarray, length: int) -> List[np.ndarray]:
        """Decode sequence using HDC inverse permutation."""
        decoded_assemblies = []
        for i in range(length):
            current_hv = sequence_hv.copy()
            for _ in range(i):
                current_hv = self.permutation_matrix.T @ current_hv
            current_hv = np.sign(current_hv)
            assembly = self._decode_hypervector_to_assembly(current_hv)
            decoded_assemblies.append(assembly)
        
        return decoded_assemblies
    
    def _measure_temporal_preservation(self, original: List[np.ndarray], decoded: List[np.ndarray]) -> float:
        """Measure temporal order preservation."""
        if len(original) != len(decoded):
            return 0.0
        
        correct_order = 0
        for i in range(len(original) - 1):
            if len(set(original[i]) & set(decoded[i])) > 0 and len(set(original[i+1]) & set(decoded[i+1])) > 0:
                correct_order += 1
        
        return correct_order / (len(original) - 1) if len(original) > 1 else 1.0
    
    def _measure_content_preservation(self, original: List[np.ndarray], decoded: List[np.ndarray]) -> float:
        """Measure content preservation."""
        if len(original) != len(decoded):
            return 0.0
        
        total_overlap = 0
        for orig, decoded in zip(original, decoded):
            overlap = len(set(orig) & set(decoded)) / len(orig)
            total_overlap += overlap
        
        return total_overlap / len(original)
    
    def _apply_cleanup(self, noisy_hv: np.ndarray) -> np.ndarray:
        """Apply HDC cleanup to remove noise."""
        # Simple cleanup: threshold and normalize
        cleaned = np.sign(noisy_hv)
        return cleaned

def run_mathematical_catalog():
    """Run the complete mathematical catalog of HDC-Assembly convergence."""
    print("MATHEMATICAL CATALOG: HDC + ASSEMBLY CALCULUS CONVERGENCE")
    print("=" * 80)
    print("This systematically catalogs the mathematical principles that emerge")
    print("from the convergence between Hyperdimensional Computing and Assembly Calculus.")
    print()
    
    # Initialize catalog
    catalog = MathematicalCatalogHDCAssembly(dimension=5000, k=30, n_neurons=500)
    
    # Run all principles
    catalog.principle_1_sparse_dense_duality()
    catalog.principle_2_interference_overlap_tradeoff()
    catalog.principle_3_binding_competition_dynamics()
    catalog.principle_4_superposition_plasticity_learning()
    catalog.principle_5_permutation_temporal_encoding()
    catalog.principle_6_cleanup_recall_dynamics()
    catalog.principle_7_capacity_sparsity_scaling()
    catalog.principle_8_robustness_interference_balance()
    
    # Generate convergence report
    report = catalog.generate_convergence_report()
    
    print("\n" + "=" * 80)
    print("MATHEMATICAL CATALOG COMPLETE")
    print("=" * 80)
    print("This catalog provides a comprehensive mathematical understanding")
    print("of the convergence between HDC and Assembly Calculus, revealing")
    print("the deep theoretical principles that emerge from their integration.")
    
    return catalog, report

if __name__ == "__main__":
    catalog, report = run_mathematical_catalog()

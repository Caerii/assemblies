#!/usr/bin/env python3
"""
CUDA Function Signatures for Universal Brain Simulator
=====================================================

This module contains the configuration for CUDA function signatures,
eliminating repetitive ctypes setup code.
"""

import ctypes

# =============================================================================
# CUDA FUNCTION SIGNATURE CONFIGURATIONS
# =============================================================================

CUDA_SIGNATURES = {
    'optimized_brain': {
        'cuda_create_optimized_brain': {
            'argtypes': [
                ctypes.c_uint32,  # uint32_t n_neurons
                ctypes.c_uint32,  # uint32_t n_areas
                ctypes.c_uint32,  # uint32_t k_active
                ctypes.c_uint32   # uint32_t seed
            ],
            'restype': ctypes.c_void_p
        },
        'cuda_simulate_step_optimized': {
            'argtypes': [
                ctypes.c_void_p   # void* brain_ptr
            ],
            'restype': None
        },
        'cuda_destroy_optimized_brain': {
            'argtypes': [
                ctypes.c_void_p   # void* brain_ptr
            ],
            'restype': None
        }
    },
    
    'individual_optimized': {
        'cuda_generate_candidates_optimized': {
            'argtypes': [
                ctypes.c_void_p,  # curandState* states
                ctypes.c_void_p,  # float* candidates
                ctypes.c_uint32,  # uint32_t num_candidates
                ctypes.c_float,   # float mean
                ctypes.c_float,   # float stddev
                ctypes.c_float    # float cutoff
            ],
            'restype': None
        },
        'cuda_top_k_selection_radix': {
            'argtypes': [
                ctypes.c_void_p,  # const float* activations
                ctypes.c_void_p,  # uint32_t* top_k_indices
                ctypes.c_uint32,  # uint32_t total_neurons
                ctypes.c_uint32   # uint32_t k
            ],
            'restype': None
        },
        'cuda_accumulate_weights_shared_memory': {
            'argtypes': [
                ctypes.c_void_p,  # const uint32_t* activated_neurons
                ctypes.c_void_p,  # const float* synapse_weights
                ctypes.c_void_p,  # const uint32_t* synapse_indices
                ctypes.c_void_p,  # const uint32_t* synapse_offsets
                ctypes.c_void_p,  # float* activations
                ctypes.c_uint32,  # uint32_t num_activated
                ctypes.c_uint32   # uint32_t target_size
            ],
            'restype': None
        },
        'cuda_initialize_curand': {
            'argtypes': [
                ctypes.c_void_p,  # curandState* states
                ctypes.c_uint32,  # uint32_t n
                ctypes.c_uint32   # uint32_t seed
            ],
            'restype': None
        }
    },
    
    'original_kernels': {
        'cuda_generate_candidates': {
            'argtypes': [
                ctypes.c_void_p,  # curandState* states
                ctypes.c_void_p,  # float* candidates
                ctypes.c_uint32,  # uint32_t num_candidates
                ctypes.c_float,   # float mean
                ctypes.c_float,   # float stddev
                ctypes.c_float    # float cutoff
            ],
            'restype': None
        },
        'cuda_top_k_selection': {
            'argtypes': [
                ctypes.c_void_p,  # const float* activations
                ctypes.c_void_p,  # uint32_t* top_k_indices
                ctypes.c_uint32,  # uint32_t total_neurons
                ctypes.c_uint32   # uint32_t k
            ],
            'restype': None
        },
        'cuda_accumulate_weights': {
            'argtypes': [
                ctypes.c_void_p,  # const uint32_t* activated_neurons
                ctypes.c_void_p,  # const float* synapse_weights
                ctypes.c_void_p,  # const uint32_t* synapse_indices
                ctypes.c_void_p,  # const uint32_t* synapse_offsets
                ctypes.c_void_p,  # float* activations
                ctypes.c_uint32,  # uint32_t num_activated
                ctypes.c_uint32   # uint32_t target_size
            ],
            'restype': None
        },
        'cuda_initialize_curand': {
            'argtypes': [
                ctypes.c_void_p,  # curandState* states
                ctypes.c_uint32,  # uint32_t n
                ctypes.c_uint32   # uint32_t seed
            ],
            'restype': None
        }
    }
}

# =============================================================================
# SIGNATURE SETUP FUNCTIONS
# =============================================================================

def setup_signatures(dll, signature_type: str) -> bool:
    """
    Setup function signatures for a DLL from configuration
    
    Args:
        dll: The ctypes CDLL object
        signature_type: Type of signatures to setup ('optimized_brain', 'individual_optimized', 'original_kernels')
        
    Returns:
        bool: True if setup was successful, False otherwise
    """
    if not dll:
        return False
    
    signatures = CUDA_SIGNATURES.get(signature_type, {})
    if not signatures:
        print(f"⚠️  Unknown signature type: {signature_type}")
        return False
    
    try:
        for func_name, config in signatures.items():
            if hasattr(dll, func_name):
                func = getattr(dll, func_name)
                func.argtypes = config['argtypes']
                func.restype = config['restype']
            else:
                print(f"⚠️  Function {func_name} not found in DLL")
        
        print(f"   ✅ {signature_type} signatures configured")
        return True
        
    except Exception as e:
        print(f"⚠️  Failed to setup {signature_type} signatures: {e}")
        return False

def setup_optimized_brain_signatures(dll) -> bool:
    """
    Setup signatures for optimized brain simulator
    
    Args:
        dll: The ctypes CDLL object
        
    Returns:
        bool: True if setup was successful, False otherwise
    """
    return setup_signatures(dll, 'optimized_brain')

def setup_individual_optimized_signatures(dll) -> bool:
    """
    Setup signatures for individual optimized kernels
    
    Args:
        dll: The ctypes CDLL object
        
    Returns:
        bool: True if setup was successful, False otherwise
    """
    return setup_signatures(dll, 'individual_optimized')

def setup_original_kernel_signatures(dll) -> bool:
    """
    Setup signatures for original kernels
    
    Args:
        dll: The ctypes CDLL object
        
    Returns:
        bool: True if setup was successful, False otherwise
    """
    return setup_signatures(dll, 'original_kernels')

def get_available_functions(dll, signature_type: str) -> list:
    """
    Get list of available functions for a signature type
    
    Args:
        dll: The ctypes CDLL object
        signature_type: Type of signatures to check
        
    Returns:
        list: List of available function names
    """
    if not dll:
        return []
    
    signatures = CUDA_SIGNATURES.get(signature_type, {})
    available = []
    
    for func_name in signatures.keys():
        if hasattr(dll, func_name):
            available.append(func_name)
    
    return available

def validate_dll_interface(dll, signature_type: str) -> bool:
    """
    Validate that a DLL has the required interface for a signature type
    
    Args:
        dll: The ctypes CDLL object
        signature_type: Type of signatures to validate
        
    Returns:
        bool: True if DLL has required interface, False otherwise
    """
    if not dll:
        return False
    
    signatures = CUDA_SIGNATURES.get(signature_type, {})
    if not signatures:
        return False
    
    # Check for required functions
    required_functions = list(signatures.keys())
    available_functions = get_available_functions(dll, signature_type)
    
    missing_functions = set(required_functions) - set(available_functions)
    if missing_functions:
        print(f"⚠️  Missing functions for {signature_type}: {missing_functions}")
        return False
    
    return True

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def print_signature_info(signature_type: str):
    """
    Print information about a signature type
    
    Args:
        signature_type: Type of signatures to print info for
    """
    signatures = CUDA_SIGNATURES.get(signature_type, {})
    if not signatures:
        print(f"Unknown signature type: {signature_type}")
        return
    
    print(f"\n📋 {signature_type.upper()} SIGNATURES:")
    print("=" * 50)
    
    for func_name, config in signatures.items():
        print(f"  {func_name}:")
        print(f"    argtypes: {len(config['argtypes'])} parameters")
        print(f"    restype: {config['restype']}")

def get_signature_count(signature_type: str) -> int:
    """
    Get the number of functions in a signature type
    
    Args:
        signature_type: Type of signatures to count
        
    Returns:
        int: Number of functions in the signature type
    """
    signatures = CUDA_SIGNATURES.get(signature_type, {})
    return len(signatures)

def list_signature_types() -> list:
    """
    Get list of available signature types
    
    Returns:
        list: List of available signature types
    """
    return list(CUDA_SIGNATURES.keys())

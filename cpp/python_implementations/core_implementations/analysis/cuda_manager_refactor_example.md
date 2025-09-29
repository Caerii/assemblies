# CUDAManager Refactoring Example

## Current CUDAManager Issues

The current `CUDAManager` has **5 different responsibilities**:

1. **DLL Loading** - Loading and managing DLL files
2. **Function Signature Setup** - Setting up ctypes signatures  
3. **Brain Instance Management** - Creating/destroying brain instances
4. **Kernel Execution** - Running simulation steps
5. **Error Handling** - Managing CUDA errors and fallbacks

## Proposed Refactored Structure

### 1. **DLL Loader** (`cuda_manager/dll_loader.py`)

```python
class DLLLoader:
    """Handles DLL loading and management"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self._loaded_dlls = {}
    
    def load_optimized_dll(self) -> Optional[ctypes.CDLL]:
        """Load optimized CUDA DLL"""
        try:
            dll_path = get_dll_path('assemblies_cuda_brain_optimized.dll')
            if os.path.exists(dll_path):
                dll = ctypes.CDLL(dll_path)
                self._loaded_dlls['optimized'] = dll
                return dll
        except Exception as e:
            print(f"⚠️  Optimized DLL loading failed: {e}")
        return None
    
    def load_original_dll(self) -> Optional[ctypes.CDLL]:
        """Load original CUDA DLL"""
        try:
            dll_path = get_dll_path('assemblies_cuda_kernels.dll')
            if os.path.exists(dll_path):
                dll = ctypes.CDLL(dll_path)
                self._loaded_dlls['original'] = dll
                return dll
        except Exception as e:
            print(f"⚠️  Original DLL loading failed: {e}")
        return None
    
    def get_dll(self, dll_type: str) -> Optional[ctypes.CDLL]:
        """Get loaded DLL by type"""
        return self._loaded_dlls.get(dll_type)
    
    def cleanup(self):
        """Cleanup loaded DLLs"""
        self._loaded_dlls.clear()
```

### 2. **Signature Manager** (`cuda_manager/signature_manager.py`)

```python
class SignatureManager:
    """Handles function signature setup for different DLL types"""
    
    def __init__(self, dll_loader: DLLLoader):
        self.dll_loader = dll_loader
    
    def setup_optimized_signatures(self) -> bool:
        """Set up signatures for optimized brain simulator"""
        dll = self.dll_loader.get_dll('optimized')
        if not dll:
            return False
        
        try:
            if hasattr(dll, 'cuda_create_optimized_brain'):
                # Optimized brain simulator interface
                dll.cuda_create_optimized_brain.argtypes = [
                    ctypes.c_uint32,  # n_neurons
                    ctypes.c_uint32,  # n_areas
                    ctypes.c_uint32,  # k_active
                    ctypes.c_uint32   # seed
                ]
                dll.cuda_create_optimized_brain.restype = ctypes.c_void_p
                
                dll.cuda_simulate_step_optimized.argtypes = [
                    ctypes.c_void_p   # brain_ptr
                ]
                dll.cuda_simulate_step_optimized.restype = None
                
                dll.cuda_destroy_optimized_brain.argtypes = [
                    ctypes.c_void_p   # brain_ptr
                ]
                dll.cuda_destroy_optimized_brain.restype = None
                
                return True
        except Exception as e:
            print(f"⚠️  Optimized signature setup failed: {e}")
        return False
    
    def setup_original_signatures(self) -> bool:
        """Set up signatures for original kernels"""
        dll = self.dll_loader.get_dll('original')
        if not dll:
            return False
        
        try:
            # Individual kernel signatures
            if hasattr(dll, 'cuda_generate_candidates'):
                dll.cuda_generate_candidates.argtypes = [
                    ctypes.c_void_p,  # states
                    ctypes.c_void_p,  # candidates
                    ctypes.c_uint32,  # num_candidates
                    ctypes.c_float,   # mean
                    ctypes.c_float,   # stddev
                    ctypes.c_float    # cutoff
                ]
                dll.cuda_generate_candidates.restype = None
            
            # ... more signatures
            
            return True
        except Exception as e:
            print(f"⚠️  Original signature setup failed: {e}")
        return False
```

### 3. **Brain Instance Manager** (`cuda_manager/brain_instance.py`)

```python
class BrainInstanceManager:
    """Manages brain instance lifecycle"""
    
    def __init__(self, dll_loader: DLLLoader, signature_manager: SignatureManager):
        self.dll_loader = dll_loader
        self.signature_manager = signature_manager
        self._brain_ptr = None
        self._instance_id = id(self)
        self._cleanup_called = False
    
    def create_optimized_brain(self, n_neurons: int, n_areas: int, k_active: int, seed: int) -> bool:
        """Create optimized brain instance"""
        dll = self.dll_loader.get_dll('optimized')
        if not dll:
            return False
        
        try:
            self._brain_ptr = dll.cuda_create_optimized_brain(
                ctypes.c_uint32(n_neurons),
                ctypes.c_uint32(n_areas),
                ctypes.c_uint32(k_active),
                ctypes.c_uint32(seed)
            )
            print(f"🧠 Optimized brain instance {self._instance_id} created: {self._brain_ptr}")
            return True
        except Exception as e:
            print(f"⚠️  Failed to create optimized brain: {e}")
            return False
    
    def destroy_brain(self):
        """Destroy brain instance"""
        if self._cleanup_called:
            print(f"⚠️  Brain cleanup already called for instance {self._instance_id}")
            return
        
        self._cleanup_called = True
        
        if self._brain_ptr and self._brain_ptr != 0:
            dll = self.dll_loader.get_dll('optimized')
            if dll:
                try:
                    dll.cuda_destroy_optimized_brain(ctypes.c_void_p(self._brain_ptr))
                    print(f"🧠 Brain instance {self._instance_id} destroyed")
                except Exception as e:
                    print(f"⚠️  Brain destruction error: {e}")
        
        self._brain_ptr = None
    
    @property
    def brain_ptr(self):
        """Get brain pointer"""
        return self._brain_ptr
    
    @property
    def is_created(self) -> bool:
        """Check if brain instance is created"""
        return self._brain_ptr is not None and self._brain_ptr != 0
```

### 4. **Kernel Executor** (`cuda_manager/kernel_executor.py`)

```python
class KernelExecutor:
    """Handles kernel execution"""
    
    def __init__(self, dll_loader: DLLLoader, brain_instance_manager: BrainInstanceManager):
        self.dll_loader = dll_loader
        self.brain_instance_manager = brain_instance_manager
    
    def execute_optimized_step(self) -> bool:
        """Execute optimized simulation step"""
        if not self.brain_instance_manager.is_created:
            return False
        
        dll = self.dll_loader.get_dll('optimized')
        if not dll:
            return False
        
        try:
            dll.cuda_simulate_step_optimized(
                ctypes.c_void_p(self.brain_instance_manager.brain_ptr)
            )
            return True
        except Exception as e:
            print(f"⚠️  Optimized step execution failed: {e}")
            return False
    
    def execute_individual_kernels(self, kernel_type: str, **kwargs) -> bool:
        """Execute individual kernels"""
        dll = self.dll_loader.get_dll(kernel_type)
        if not dll:
            return False
        
        try:
            if kernel_type == 'original':
                # Execute original kernels
                if 'candidates' in kwargs:
                    dll.cuda_generate_candidates(
                        ctypes.c_void_p(kwargs['states']),
                        ctypes.c_void_p(kwargs['candidates']),
                        ctypes.c_uint32(kwargs['num_candidates']),
                        ctypes.c_float(kwargs['mean']),
                        ctypes.c_float(kwargs['stddev']),
                        ctypes.c_float(kwargs['cutoff'])
                    )
                # ... more kernel executions
            return True
        except Exception as e:
            print(f"⚠️  Individual kernel execution failed: {e}")
            return False
```

### 5. **Error Handler** (`cuda_manager/error_handler.py`)

```python
class CUDAErrorHandler:
    """Handles CUDA errors and fallbacks"""
    
    def __init__(self):
        self._error_count = 0
        self._max_errors = 10
    
    def handle_dll_load_error(self, dll_type: str, error: Exception) -> bool:
        """Handle DLL loading errors"""
        self._error_count += 1
        print(f"⚠️  DLL load error ({dll_type}): {error}")
        
        if self._error_count >= self._max_errors:
            print("❌ Too many DLL load errors, disabling CUDA")
            return False
        return True
    
    def handle_kernel_execution_error(self, kernel_type: str, error: Exception) -> bool:
        """Handle kernel execution errors"""
        self._error_count += 1
        print(f"⚠️  Kernel execution error ({kernel_type}): {error}")
        
        if self._error_count >= self._max_errors:
            print("❌ Too many kernel execution errors, falling back to CuPy")
            return False
        return True
    
    def handle_brain_creation_error(self, error: Exception) -> bool:
        """Handle brain creation errors"""
        self._error_count += 1
        print(f"⚠️  Brain creation error: {error}")
        
        if self._error_count >= self._max_errors:
            print("❌ Too many brain creation errors, falling back to CuPy")
            return False
        return True
    
    def reset_error_count(self):
        """Reset error count"""
        self._error_count = 0
    
    @property
    def error_count(self) -> int:
        """Get current error count"""
        return self._error_count
```

### 6. **Refactored CUDAManager** (`cuda_manager/__init__.py`)

```python
class CUDAManager:
    """Refactored CUDA manager with separated concerns"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        
        # Initialize components
        self.dll_loader = DLLLoader(config)
        self.signature_manager = SignatureManager(self.dll_loader)
        self.brain_instance_manager = BrainInstanceManager(self.dll_loader, self.signature_manager)
        self.kernel_executor = KernelExecutor(self.dll_loader, self.brain_instance_manager)
        self.error_handler = CUDAErrorHandler()
        
        # State tracking
        self._using_optimized_kernels = False
        self._initialized = False
    
    def load_kernels(self) -> bool:
        """Load CUDA kernels with fallback logic"""
        if not self.config.use_cuda_kernels:
            return False
        
        # Try optimized kernels first
        if self.config.use_optimized_kernels:
            if self._try_load_optimized_kernels():
                return True
            else:
                print("⚠️  Optimized CUDA kernels not available, trying original...")
        
        # Fallback to original kernels
        return self._try_load_original_kernels()
    
    def _try_load_optimized_kernels(self) -> bool:
        """Try to load optimized kernels"""
        dll = self.dll_loader.load_optimized_dll()
        if dll and self.signature_manager.setup_optimized_signatures():
            self._using_optimized_kernels = True
            print("✅ Optimized CUDA kernels loaded successfully!")
            return True
        return False
    
    def _try_load_original_kernels(self) -> bool:
        """Try to load original kernels"""
        dll = self.dll_loader.load_original_dll()
        if dll and self.signature_manager.setup_original_signatures():
            self._using_optimized_kernels = False
            print("✅ Original CUDA kernels loaded successfully!")
            return True
        return False
    
    def create_optimized_brain(self, n_neurons: int, n_areas: int, k_active: int, seed: int) -> bool:
        """Create optimized brain instance"""
        return self.brain_instance_manager.create_optimized_brain(n_neurons, n_areas, k_active, seed)
    
    def simulate_step_optimized(self) -> bool:
        """Simulate one step using optimized brain"""
        return self.kernel_executor.execute_optimized_step()
    
    def cleanup(self):
        """Cleanup all CUDA resources"""
        self.brain_instance_manager.destroy_brain()
        self.dll_loader.cleanup()
        self.error_handler.reset_error_count()
    
    # Properties
    @property
    def is_loaded(self) -> bool:
        return self.dll_loader.get_dll('optimized') is not None or self.dll_loader.get_dll('original') is not None
    
    @property
    def using_optimized_kernels(self) -> bool:
        return self._using_optimized_kernels
    
    @property
    def optimized_brain_ptr(self):
        return self.brain_instance_manager.brain_ptr
```

## Benefits of This Refactoring

### **1. Single Responsibility Principle**
- Each class has one clear responsibility
- Easy to understand and modify
- Reduced coupling between components

### **2. Testability**
- Each component can be unit tested independently
- Mock dependencies easily
- Test different scenarios in isolation

### **3. Extensibility**
- Easy to add new DLL types
- Easy to add new signature types
- Easy to add new kernel execution strategies

### **4. Maintainability**
- Changes to one component don't affect others
- Easier to debug issues
- Clear separation of concerns

### **5. Reusability**
- Components can be reused in different contexts
- Can create different combinations of components
- Can implement different CUDA management strategies

## Implementation Steps

1. **Create new directory structure**:
   ```
   cuda_manager/
   ├── __init__.py
   ├── dll_loader.py
   ├── signature_manager.py
   ├── brain_instance.py
   ├── kernel_executor.py
   └── error_handler.py
   ```

2. **Implement each component**:
   - Start with `DLLLoader` (simplest)
   - Then `SignatureManager` (depends on DLL loader)
   - Then `BrainInstanceManager` (depends on both)
   - Then `KernelExecutor` (depends on all above)
   - Finally `ErrorHandler` (independent)

3. **Update the main CUDAManager**:
   - Use composition instead of inheritance
   - Delegate to appropriate components
   - Maintain the same public interface

4. **Update tests**:
   - Test each component independently
   - Test integration between components
   - Test error handling scenarios

5. **Update documentation**:
   - Document each component's responsibility
   - Document the new architecture
   - Update usage examples

This refactoring will make the CUDA management much more maintainable, testable, and extensible while preserving the same public interface.

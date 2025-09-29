#!/usr/bin/env python3
"""
Build script for the C++ Brain extension.

This script handles building the C++ extension with proper dependencies
and error handling.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def check_dependencies():
    """Check if required dependencies are available."""
    print("Checking dependencies...")
    
    # Check Python version
    if sys.version_info < (3, 6):
        print("Error: Python 3.6+ required")
        return False
    
    # Check for C++ compiler
    try:
        if platform.system() == "Windows":
            # Try to find MSVC
            result = subprocess.run(["cl"], capture_output=True, text=True)
            if result.returncode != 0:
                print("Warning: MSVC compiler not found. Install Visual Studio Build Tools.")
        else:
            # Try to find GCC or Clang
            for compiler in ["g++", "clang++"]:
                result = subprocess.run([compiler, "--version"], capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"Found compiler: {compiler}")
                    break
            else:
                print("Error: No C++ compiler found. Install GCC or Clang.")
                return False
    except FileNotFoundError:
        print("Error: No C++ compiler found.")
        return False
    
    # Check for pybind11
    try:
        import pybind11
        print(f"Found pybind11: {pybind11.__version__}")
    except ImportError:
        print("Installing pybind11...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pybind11"], check=True)
    
    # Check for numpy
    try:
        import numpy
        print(f"Found numpy: {numpy.__version__}")
    except ImportError:
        print("Installing numpy...")
        subprocess.run([sys.executable, "-m", "pip", "install", "numpy"], check=True)
    
    return True


def build_extension():
    """Build the C++ extension."""
    print("Building C++ extension...")
    
    try:
        # Change to cpp directory
        cpp_dir = Path(__file__).parent
        os.chdir(cpp_dir)
        
        # Build the extension
        subprocess.run([
            sys.executable, "setup.py", "build_ext", "--inplace"
        ], check=True)
        
        print("Build successful!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Build failed: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False


def test_extension():
    """Test the built extension."""
    print("Testing extension...")
    
    try:
        # Test import
        import brain_cpp
        print("✓ Extension imports successfully")
        
        # Test basic functionality
        brain = brain_cpp.Brain(0.05, 0.1, 10000.0, 7777)
        brain.add_area("A", 1000, 50)
        brain.add_stimulus("stimA", 50)
        brain.add_fiber("stimA", "A")
        
        print("✓ Basic functionality works")
        return True
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


def main():
    """Main build process."""
    print("C++ Brain Extension Builder")
    print("=" * 40)
    
    # Check dependencies
    if not check_dependencies():
        print("Dependency check failed. Please install required dependencies.")
        sys.exit(1)
    
    # Build extension
    if not build_extension():
        print("Build failed. Check error messages above.")
        sys.exit(1)
    
    # Test extension
    if not test_extension():
        print("Test failed. Extension may not work correctly.")
        sys.exit(1)
    
    print("\n✓ Build completed successfully!")
    print("You can now use the high-performance C++ Brain implementation.")
    
    # Show usage example
    print("\nUsage example:")
    print("```python")
    print("from src.core.brain_cpp import BrainCPP")
    print("brain = BrainCPP(p=0.05, beta=0.1)")
    print("brain.add_area('A', 1000, 50)")
    print("```")


if __name__ == "__main__":
    main()

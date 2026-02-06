#!/usr/bin/env python3
"""
Test Visual Studio + CUDA environment setup
"""

import subprocess
import os
import tempfile

def test_vs_cuda_environment():
    print("🔧 TESTING VISUAL STUDIO + CUDA ENVIRONMENT")
    print("=" * 60)
    
    # Find Visual Studio script
    vs_scripts = [
        r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
    ]
    
    vs_script = None
    for script in vs_scripts:
        if os.path.exists(script):
            vs_script = script
            print(f"✓ Found Visual Studio script: {script}")
            break
    
    if not vs_script:
        print("❌ No Visual Studio environment script found")
        return False
    
    # CUDA paths
    cuda_home = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"
    cuda_bin = os.path.join(cuda_home, "bin")
    cuda_lib = os.path.join(cuda_home, "lib", "x64")
    
    print("✓ CUDA paths configured")
    print()
    
    # Create a batch file to test the environment
    batch_content = f'''@echo off
call "{vs_script}"
set PATH={cuda_bin};{cuda_lib};%PATH%
set CUDA_HOME={cuda_home}

echo Testing cl.exe...
cl 2>&1 | findstr "Microsoft"
if errorlevel 1 (
    echo ❌ cl.exe failed
    exit /b 1
)

echo Testing nvcc...
nvcc --version
if errorlevel 1 (
    echo ❌ nvcc failed
    exit /b 1
)

echo ✅ Both Visual Studio and CUDA are working!
'''
    
    # Write and execute the batch file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.bat', delete=False) as f:
        f.write(batch_content)
        batch_file = f.name
    
    try:
        print("Running environment test...")
        result = subprocess.run([batch_file], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ SUCCESS! Environment is ready!")
            print()
            print("Output:")
            print(result.stdout)
            return True
        else:
            print("❌ Environment test failed:")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Test timed out")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    finally:
        # Clean up
        try:
            os.unlink(batch_file)
        except:
            pass

def main():
    success = test_vs_cuda_environment()
    
    if success:
        print()
        print("🚀 NEXT STEP: COMPILE CUDA KERNELS!")
        print("=" * 60)
        print("Your environment is ready for CUDA compilation.")
        print("We can now build the GPU-accelerated neural simulation!")
    else:
        print()
        print("🔧 TROUBLESHOOTING NEEDED")
        print("=" * 60)
        print("Environment setup needs attention.")
        print("But your C++ version is already incredibly fast!")

if __name__ == "__main__":
    main()


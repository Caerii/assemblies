#!/usr/bin/env python3
"""
Simple test to verify CUDA + Visual Studio environment
"""

import subprocess
import os

def main():
    print("🔧 SIMPLE CUDA + VISUAL STUDIO TEST")
    print("=" * 50)
    
    # Create simple batch script
    batch_content = '''@echo off
call "C:\\Program Files (x86)\\Microsoft Visual Studio\\2022\\BuildTools\\VC\\Auxiliary\\Build\\vcvars64.bat"
set PATH=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\bin;%PATH%

echo Testing compilers...
cl 2>&1 | findstr "Microsoft"
nvcc --version
echo Environment test complete!
'''
    
    # Write batch file
    with open('test_env.bat', 'w') as f:
        f.write(batch_content)
    
    print("Running environment test...")
    try:
        result = subprocess.run(['test_env.bat'], capture_output=True, text=True, timeout=30)
        
        print("Return code:", result.returncode)
        print("STDOUT:")
        print(result.stdout)
        print("STDERR:")
        print(result.stderr)
        
        if result.returncode == 0 and "nvcc" in result.stdout and "Microsoft" in result.stdout:
            print("\n✅ SUCCESS! Both Visual Studio and CUDA working!")
            return True
        else:
            print("\n❌ Environment test had issues")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        return False
    finally:
        # Clean up
        try:
            os.remove('test_env.bat')
        except:
            pass

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🚀 READY TO COMPILE CUDA KERNELS!")
    else:
        print("\n🔧 Environment needs setup")


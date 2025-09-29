#include <iostream>
#include <cuda_runtime.h>
#include <curand_kernel.h>

__global__ void simple_test_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = idx * 2.0f;
    }
}

int main() {
    const int n = 1000;
    float* h_data = new float[n];
    float* d_data;
    
    std::cout << "🚀 Testing CUDA with RTX 4090..." << std::endl;
    
    // Allocate GPU memory
    cudaError_t err = cudaMalloc(&d_data, n * sizeof(float));
    if (err != cudaSuccess) {
        std::cout << "❌ GPU memory allocation failed: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    
    // Launch kernel
    simple_test_kernel<<<(n+255)/256, 256>>>(d_data, n);
    
    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "❌ Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    
    // Wait for completion
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cout << "❌ Kernel execution failed: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    
    // Copy back
    err = cudaMemcpy(h_data, d_data, n * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cout << "❌ Memory copy failed: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    
    // Check first few values
    std::cout << "✓ CUDA Test Results:" << std::endl;
    for (int i = 0; i < 5; i++) {
        std::cout << "  data[" << i << "] = " << h_data[i] << " (expected: " << i*2 << ")" << std::endl;
    }
    
    // Cleanup
    cudaFree(d_data);
    delete[] h_data;
    
    std::cout << "🎉 CUDA test completed successfully on RTX 4090!" << std::endl;
    return 0;
}
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

// Simple test kernel
__global__ void test_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f + 1.0f;
    }
}

// C interface
extern "C" {
    int test_cuda() {
        std::cout << "🧪 Testing CUDA..." << std::endl;
        
        // Allocate device memory
        float* d_data;
        cudaMalloc(&d_data, 1024 * sizeof(float));
        
        // Initialize host data
        std::vector<float> h_data(1024, 1.0f);
        
        // Copy to device
        cudaMemcpy(d_data, h_data.data(), 1024 * sizeof(float), cudaMemcpyHostToDevice);
        
        // Launch kernel
        dim3 block(256);
        dim3 grid((1024 + block.x - 1) / block.x);
        test_kernel<<<grid, block>>>(d_data, 1024);
        
        // Wait for completion
        cudaDeviceSynchronize();
        
        // Copy back
        cudaMemcpy(h_data.data(), d_data, 1024 * sizeof(float), cudaMemcpyDeviceToHost);
        
        // Verify result
        if (h_data[0] == 3.0f) {
            std::cout << "✅ CUDA test passed!" << std::endl;
            cudaFree(d_data);
            return 1;
        } else {
            std::cout << "❌ CUDA test failed!" << std::endl;
            cudaFree(d_data);
            return 0;
        }
    }
}

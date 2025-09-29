#include <iostream>
#include <cuda_runtime.h>

__global__ void hello_cuda() {
    printf("Hello from GPU thread %d!\n", threadIdx.x);
}

int main() {
    printf("Testing CUDA...\n");
    hello_cuda<<<1, 5>>>();
    cudaDeviceSynchronize();
    printf("CUDA test completed!\n");
    return 0;
}

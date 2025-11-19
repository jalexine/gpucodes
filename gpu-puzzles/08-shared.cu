#include <cstdio>
#include <cuda_runtime.h>

const int SIZE = 8;
const int TPB = 4;

__global__ void shared(const float* a, float* out, int size) {
    __shared__ float shared[TPB];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int local_i = threadIdx.x;

    if (i < size) {
	shared[local_i] = a[i];
	__syncthreads();
        out[i] = shared[local_i] + 10.0f;
    }
}

int main() {
    float h_a[SIZE];
    float h_out[SIZE];
  
    for (int i = 0; i < SIZE; ++i) {
        h_a[i] = 1.0f;  
    }

    float* d_a = nullptr;
    float* d_out = nullptr;

    cudaMalloc(&d_a,  SIZE * sizeof(float));
    cudaMalloc(&d_out, SIZE * sizeof(float));

    cudaMemcpy(d_a, h_a, SIZE * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(TPB);
    dim3 numBlocks(2);

    shared<<<numBlocks, threadsPerBlock>>>(d_a, d_out, SIZE);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < SIZE; ++i) {
             std::printf("h_out[%d] = %f\n", i, h_out[i]);
    }


    cudaFree(d_a);
    cudaFree(d_out);
    return 0;
}


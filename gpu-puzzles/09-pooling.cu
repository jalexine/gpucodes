#include <cstdio>
#include <cuda_runtime.h>

const int SIZE = 8;
const int TPB  = 8;

__global__ void pooling(const float* a, float* out, int size) {
    __shared__ float shared[TPB];

    int i       = blockIdx.x * blockDim.x + threadIdx.x;
    int local_i = threadIdx.x;

    if (i < size) {
        shared[local_i] = a[i];
        __syncthreads();
        if (i == 0) {
            out[i] = shared[local_i];
        } else if (i == 1) {
            out[i] = shared[local_i] + shared[local_i - 1];
        } else {
            out[i] = shared[local_i] + shared[local_i - 1] + shared[local_i - 2];
        }
    }
}

int main() {
    float h_a[SIZE];
    float h_out[SIZE];
    for (int i = 0; i < SIZE; ++i) {
        h_a[i] = static_cast<float>(i);
    }

    float* d_a = nullptr;
    float* d_out = nullptr;

    cudaMalloc(&d_a, SIZE * sizeof(float));
    cudaMalloc(&d_out, SIZE * sizeof(float));

    cudaMemcpy(d_a, h_a, SIZE * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(TPB);
    dim3 numBlocks((SIZE + TPB - 1) / TPB);

    pooling<<<numBlocks, threadsPerBlock>>>(d_a, d_out, SIZE);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < SIZE; ++i) {
        std::printf("out[%d] = %f\n", i, h_out[i]);
    }

    cudaFree(d_a);
    cudaFree(d_out);
    return 0;
}



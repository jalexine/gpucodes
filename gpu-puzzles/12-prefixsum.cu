#include <cstdio>
#include <cuda_runtime.h>

const int TPB = 8;

__global__ void prefixsum(const float* a, float* out, int size) {
    __shared__ float cache[TPB];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int local_i = threadIdx.x;

    if (i < size)
        cache[local_i] = a[i];
    __syncthreads();

    for (int k = 0; k < 3; ++k) {
        int p = 1 << k;
        if (local_i % (p * 2) == 0)
            cache[local_i] += cache[local_i + p];
        __syncthreads();
    }

    if (local_i == 0)
        out[blockIdx.x] = cache[0];
}

int main() {
    const int SIZE = 8;
    const int OUT_SIZE = (SIZE + TPB - 1) / TPB;

    float h_a[SIZE], h_out[OUT_SIZE];
    for (int i = 0; i < SIZE; ++i)
        h_a[i] = (float)i;

    float *d_a, *d_out;
    cudaMalloc(&d_a, SIZE * sizeof(float));
    cudaMalloc(&d_out, OUT_SIZE * sizeof(float));

    cudaMemcpy(d_a, h_a, SIZE * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threads(TPB);
    dim3 blocks((SIZE + TPB - 1) / TPB);
    prefixsum<<<blocks, threads>>>(d_a, d_out, SIZE);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, OUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < OUT_SIZE; ++i)
        printf("%.1f ", h_out[i]);
    printf("\n");

    cudaFree(d_a);
    cudaFree(d_out);
    return 0;
}


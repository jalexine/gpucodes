#include <cstdio>
#include <cuda_runtime.h>

const int TPB = 8;
const int BATCH = 4;

__global__ void axis_sum(const float* a, float* out, int size) {
    __shared__ float cache[TPB];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int local_i = threadIdx.x;
    int batch = blockIdx.y;

    if (i < size)
        cache[local_i] = a[batch * size + i];
    __syncthreads();

    for (int k = 0; k < 3; ++k) {
        int p = 1 << k;
        if (local_i % (p * 2) == 0) {
            if (i + p < size)
                cache[local_i] += cache[local_i + p];
        }
        __syncthreads();
    }

    if (local_i == 0)
        out[batch] = cache[0];
}

int main() {
    const int SIZE = 6;
    const int TOTAL = SIZE * BATCH;

    float h_a[TOTAL], h_out[BATCH];
    for (int b = 0; b < BATCH; ++b)
        for (int i = 0; i < SIZE; ++i)
            h_a[b * SIZE + i] = (float)(b * SIZE + i);

    float *d_a, *d_out;
    cudaMalloc(&d_a, TOTAL * sizeof(float));
    cudaMalloc(&d_out, BATCH * sizeof(float));
    cudaMemcpy(d_a, h_a, TOTAL * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threads(TPB);
    dim3 blocks((SIZE + TPB - 1) / TPB, BATCH);
    axis_sum<<<blocks, threads>>>(d_a, d_out, SIZE);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, BATCH * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < BATCH; ++i)
        printf("%.1f ", h_out[i]);
    printf("\n");

    cudaFree(d_a);
    cudaFree(d_out);
    return 0;
}


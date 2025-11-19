#include <cstdio>
#include <cuda_runtime.h>

const int TPB = 3;

__global__ void matmul(float* out, const float* a, const float* b, int size) {
    __shared__ float a_shared[TPB][TPB];
    __shared__ float b_shared[TPB][TPB];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int local_i = threadIdx.x;
    int local_j = threadIdx.y;

    float acc = 0.0f;

    for (int k = 0; k < size; k += TPB) {
        if (i < size && k + local_j < size)
            a_shared[local_i][local_j] = a[i * size + (k + local_j)];
        if (j < size && k + local_i < size)
            b_shared[local_i][local_j] = b[(k + local_i) * size + j];
        __syncthreads();

        for (int local_k = 0; local_k < min(TPB, size - k); ++local_k)
            acc += a_shared[local_i][local_k] * b_shared[local_k][local_j];

        __syncthreads();
    }

    if (i < size && j < size)
        out[i * size + j] = acc;
}

int main() {
    const int SIZE = 2;
    const int N = SIZE * SIZE;

    float h_a[N], h_b[N], h_out[N];
    for (int i = 0; i < SIZE; ++i)
        for (int j = 0; j < SIZE; ++j)
            h_a[i * SIZE + j] = static_cast<float>(i * SIZE + j),
            h_b[j * SIZE + i] = static_cast<float>(i * SIZE + j);

    float *d_a, *d_b, *d_out;
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));

    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threads(TPB, TPB);
    dim3 blocks(1, 1);
    matmul<<<blocks, threads>>>(d_out, d_a, d_b, SIZE);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < SIZE; ++i) {
        for (int j = 0; j < SIZE; ++j)
            printf("%.1f ", h_out[i * SIZE + j]);
        printf("\n");
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
    return 0;
}


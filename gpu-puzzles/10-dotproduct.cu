#include <cstdio>
#include <cuda_runtime.h>

const int SIZE = 8;
const int TPB = 8;

__global__ void dot(const float* a, const float* b, float* out, int size) {
    __shared__ float shared[TPB];
    
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int local_i = threadIdx.x;
    if (i < size) {
        shared[local_i] = a[i] * b[i];
    } else {
        shared[local_i] = 0.0f;
    }

    __syncthreads();
    if (local_i == 0) {
        float s = 0.0f;
        for (int k = 0; k < size; ++k) {
            s += shared[k];
        }
        out[0] = s;
    }
}

int main() {
    float h_a[SIZE], h_b[SIZE], h_out[1];

    for (int i = 0; i < SIZE; ++i) {
        h_a[i] = (float)i;
        h_b[i] = (float)i;
    }

    float *d_a = nullptr, *d_b = nullptr, *d_out = nullptr;

    cudaMalloc(&d_a, SIZE * sizeof(float));
    cudaMalloc(&d_b, SIZE * sizeof(float));
    cudaMalloc(&d_out, sizeof(float));

    cudaMemcpy(d_a, h_a, SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, SIZE * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(TPB);
    dim3 numBlocks(1);

    dot<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_out, SIZE);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    printf("dot product = %f\n", h_out[0]);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    return 0;
}


#include <cstdio>
#include <cuda_runtime.h>
 
const int N = 9;

__global__ void blocks(const float* a, float* out, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        out[i] = a[i] + 10.0f;
    }
}

int main() {
    float h_a[N];
    float h_out[N];

    for (int k = 0; k < N; ++k) {
        h_a[k] = (float)k;
    }

    float* d_a  = nullptr;
    float* d_out = nullptr;

    cudaMalloc(&d_a,  N * sizeof(float));
    cudaMalloc(&d_out, N    * sizeof(float));

    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(4, 1);
    dim3 numBlocks(3, 1);

    blocks<<<numBlocks, threadsPerBlock>>>(d_a, d_out, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        printf("out[%d] = %.1f\n", i, h_out[i]);
    }

    cudaFree(d_a);
    cudaFree(d_out);

    return 0;
}


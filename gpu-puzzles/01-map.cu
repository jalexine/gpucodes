#include <cstdio>
#include <cuda_runtime.h>

const int N = 4;

__global__ void map_kernel(const float* a, float* out) {
    int i = threadIdx.x;
    out[i] = a[i] + 10.0f;
}

int main() {
    float h_a[N];
    float h_out[N];

    for (int i = 0; i < N; i++)
        h_a[i] = (float)i;

    float *d_a = nullptr;
    float *d_out = nullptr;
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));

    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);

    map_kernel<<<1, N>>>(d_a, d_out);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        std::printf("h_out[%d] = %f\n", i, h_out[i]);
    }

    cudaFree(d_a);
    cudaFree(d_out);
    return 0;
}


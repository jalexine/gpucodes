#include <cstdio>
#include <cuda_runtime.h>

const int N = 4;

__global__ void zip(const float* a, const float* b, float* out) {
    int i = threadIdx.x;
    out[i] = a[i] + b[i];
}

int main() {
    float h_a[N];
    float h_b[N];
    float h_out[N];

    for (int i = 0; i < N; i++){ 
    h_a[i] = (float)i;
    h_b[i] = 10.0f * (float)i;
    }
    float *d_a = nullptr;
    float *d_b = nullptr;
    float *d_out = nullptr;
    cudaMalloc(&d_a,  N * sizeof(float));
    cudaMalloc(&d_b,  N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));

    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);

    zip<<<1, N>>>(d_a, d_b, d_out);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        std::printf("h_out[%d] = %f\n", i, h_out[i]);
    }

    cudaFree(d_a);
    cudaFree(d_out);
    return 0;
}

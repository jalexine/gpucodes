#include <cstdio>
#include <cuda_runtime.h>

const int MAX_CONV = 4;
const int TPB = 8;
const int TPB_MAX_CONV = TPB + MAX_CONV;

__global__ void conv1d(const float* a, const float* b, float* out,
                       int a_size, int b_size) {
    __shared__ float shared_a[TPB_MAX_CONV];
    __shared__ float shared_b[MAX_CONV];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int local_i = threadIdx.x;

    // reading into shared memory
    if (i < a_size)
        shared_a[local_i] = a[i];
    else
        shared_a[local_i] = 0.0f;

    if (local_i < b_size)
        shared_b[local_i] = b[local_i];

    if (local_i < MAX_CONV) {
        int next = i + TPB;
        shared_a[TPB + local_i] =
            (next < a_size) ? a[next] : 0.0f;
    }
    __syncthreads();

    // calcul conv
    if (i < a_size) {
        float acc = 0.0f;
        for (int k = 0; k < b_size; ++k) {
            if (i + k < a_size)
                acc += shared_a[local_i + k] * shared_b[k];
        }
        out[i] = acc;
    }
}

int main() {
    const int SIZE = 6;
    const int CONV = 3;

    float h_a[SIZE];
    float h_b[CONV];
    float h_out[SIZE];

    for (int i = 0; i < SIZE; ++i) h_a[i] = (float)i;
    for (int i = 0; i < CONV; ++i) h_b[i] = (float)i;

    float *d_a, *d_b, *d_out;
    cudaMalloc(&d_a, SIZE * sizeof(float));
    cudaMalloc(&d_b, CONV * sizeof(float));
    cudaMalloc(&d_out, SIZE * sizeof(float));

    cudaMemcpy(d_a, h_a, SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, CONV * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threads(TPB);
    dim3 blocks((SIZE + TPB - 1) / TPB);

    conv1d<<<blocks, threads>>>(d_a, d_b, d_out, SIZE, CONV);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, SIZE * sizeof(float), cudaMemcpyDeviceToHost);

   for (int i = 0; i < SIZE; ++i)
        printf("%.1f ", h_out[i]);
    printf("\n");


    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    return 0;
}


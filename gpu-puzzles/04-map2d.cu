#include <cstdio>
#include <cuda_runtime.h>

const int SIZE = 2;
const int N = SIZE * SIZE;

__global__ void map2d(const float* a, float* out, int size) {
    int i = threadIdx.x;
    int j = threadIdx.y;
    if (i<size && j<size){
    int idx = i * size + j;
    out[idx] = a[idx] + 10.0f;
    }
}

int main() {
    float h_a[N];
    float h_out[N];

    for (int k = 0; k < N; k++)
        h_a[k] = (float)k;

    float *d_a = nullptr;
    float *d_out = nullptr;
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));

    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(3, 3);
    dim3 numBlocks(1, 1);


    map2d<<<numBlocks, threadsPerBlock>>>(d_a, d_out, SIZE);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < SIZE; i++){
	for(int j=0; j < SIZE; j++){
	   int idx = i * SIZE + j;
           std::printf("h_out[%d,%d] = %f\n", i, j, h_out[idx]);
	   }
    }

    cudaFree(d_a);
    cudaFree(d_out);
    return 0;
}


#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>

constexpr int Br = 16;
constexpr int Bc = 32;

// warp reduce sum
__device__ __forceinline__ float warpReduceSum(float v) {
    for (int o = 16; o > 0; o >>= 1)
        v += __shfl_down_sync(0xffffffff, v, o);
    return v;
}

//my reimplemetation of flashattn
// 1 warp per query, K/V streamed in tiles, online softmax
__global__ void attn_flash_tile2d_kernel(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int N,
    int D
) {
    int warp = threadIdx.x / 32;
    int lane = threadIdx.x % 32;

    int q_row = blockIdx.x * Br + warp;
    if (warp >= Br || q_row >= N) return;

    extern __shared__ float smem[];
    float* sQ = smem;
    float* sK = sQ + Br * D;
    float* sV = sK + Bc * D;

    float m = -1e20f;
    float l = 0.0f;

    // we start output at 0 as we accumulate tile bytile
    for (int d = lane; d < D; d += 32)
        O[q_row * D + d] = 0.0f;

    // load Q once into shared bc we reuse it for every KV tile
    for (int idx = threadIdx.x; idx < Br * D; idx += blockDim.x) {
        int r = idx / D;
        int d = idx % D;
        int gi = blockIdx.x * Br + r;
        sQ[idx] = (gi < N) ? Q[gi * D + d] : 0.0f;
    }
    __syncthreads();

    for (int tile = 0; tile < N; tile += Bc) {
        int cols = (N - tile < Bc) ? (N - tile) : Bc;

        // load current K/V tile into shared so dot products hit fast memory
        for (int idx = threadIdx.x; idx < Bc * D; idx += blockDim.x) {
            int tj = idx / D;
            int d  = idx % D;
            int gj = tile + tj;
            if (tj < cols && gj < N) {
                sK[idx] = K[gj * D + d];
                sV[idx] = V[gj * D + d];
            } else {
                sK[idx] = 0.0f;
                sV[idx] = 0.0f;
            }
        }
        __syncthreads();

        // pass 1: we need the max score in this tile so softmax doesn't overflow
        float lane0_max = -1e20f;
        for (int tj = 0; tj < cols; tj++) {
            float acc = 0.0f;
            for (int d = lane; d < D; d += 32)
                acc += sQ[warp * D + d] * sK[tj * D + d];

            float s = warpReduceSum(acc);
            s = __shfl_sync(0xffffffff, s, 0); // broadcast within warp

            if (lane == 0) lane0_max = fmaxf(lane0_max, s);
        }
        float tile_max = __shfl_sync(0xffffffff, lane0_max, 0);

        // online softmax: rescale previous partial sums into the new max frame
        //keep running (m,l) state to avoid storing scores
        float m_new = fmaxf(m, tile_max);
        float alpha = expf(m - m_new);

        if (lane == 0) l *= alpha;
        l = __shfl_sync(0xffffffff, l, 0);

        for (int d = lane; d < D; d += 32)
            O[q_row * D + d] *= alpha;

        // pass 2: accumulate exp(score - m_new) * V & update l
        float l_tile = 0.0f;

        for (int tj = 0; tj < cols; tj++) {
            float acc = 0.0f;
            for (int d = lane; d < D; d += 32)
                acc += sQ[warp * D + d] * sK[tj * D + d];

            float s = warpReduceSum(acc);
            s = __shfl_sync(0xffffffff, s, 0);

            float w = expf(s - m_new);

            if (lane == 0) l_tile += w;

            for (int d = lane; d < D; d += 32)
                O[q_row * D + d] += w * sV[tj * D + d];
        }

        l_tile = __shfl_sync(0xffffffff, l_tile, 0);
        if (lane == 0) l += l_tile;
        l = __shfl_sync(0xffffffff, l, 0);

        m = m_new;
        __syncthreads(); // next tile overwrites sK/sV
    }
    // final normaaa
    float inv_l = 1.0f / __shfl_sync(0xffffffff, l, 0);
    for (int d = lane; d < D; d += 32)
        O[q_row * D + d] *= inv_l;

    //TODO:shared memory optimizations 
    //TODO: multihead/batch support
}

int main() {
    int N = 512;
    int D = 64;
    size_t bytes = (size_t)N * D * sizeof(float);

    float *hQ = (float*)malloc(bytes);
    float *hK = (float*)malloc(bytes);
    float *hV = (float*)malloc(bytes);

    for (int i = 0; i < N * D; i++) {
        hQ[i] = 2.f * rand() / (float)RAND_MAX - 1.f;
        hK[i] = 2.f * rand() / (float)RAND_MAX - 1.f;
        hV[i] = 2.f * rand() / (float)RAND_MAX - 1.f;
    }

    float *dQ, *dK, *dV, *dO;
    cudaMalloc(&dQ, bytes);
    cudaMalloc(&dK, bytes);
    cudaMalloc(&dV, bytes);
    cudaMalloc(&dO, bytes);

    cudaMemcpy(dQ, hQ, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dK, hK, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dV, hV, bytes, cudaMemcpyHostToDevice);

    dim3 block(Br * 32);
    dim3 grid((N + Br - 1) / Br);
    size_t shmem = (size_t)(Br * D + 2 * Bc * D) * sizeof(float);

    attn_flash_tile2d_kernel<<<grid, block, shmem>>>(dQ, dK, dV, dO, N, D);
    cudaDeviceSynchronize();

    cudaFree(dQ);
    cudaFree(dK);
    cudaFree(dV);
    cudaFree(dO);

    free(hQ);
    free(hK);
    free(hV);

    return 0;
}


// flashattnv2.cu
// Dan's (xinpw8) reimplementation of Flash Attention V2
//   - Bc=64 tiling for better memory locality
//   - float4 vectorized loads for coalesced global memory access
//   - Score caching in shared memory to avoid redundant computation
//   - Register accumulators for output to reduce shared memory pressure
//   - Online softmax to compute attention in a single pass
//   - Warp shuffle reductions for fast intra-warp communication

#include <cstdio>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>

#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA %s (%d) at %s:%d\n", cudaGetErrorString(e), e, __FILE__, __LINE__); exit(1);} } while(0)

constexpr int N  = 512;
constexpr int D  = 64;
constexpr int Br = 16;
constexpr int Bc = 64;

__device__ __forceinline__ float warpReduceSum(float v) {
    for (int o = 16; o > 0; o >>= 1)
        v += __shfl_down_sync(0xffffffff, v, o);
    return v;
}

// naive baseline
__global__ void attn_naive_kernel(const float* Q, const float* K, const float* V, float* O) {
    int q = blockIdx.x;
    int d = threadIdx.x;
    if (q >= N || d >= D) return;

    float scale = rsqrtf((float)D);
    float m = -1e20f;
    for (int j = 0; j < N; j++) {
        float s = 0.0f;
        for (int k = 0; k < D; k++)
            s += Q[q * D + k] * K[j * D + k];
        m = fmaxf(m, s * scale);
    }

    float l = 0.0f;
    for (int j = 0; j < N; j++) {
        float s = 0.0f;
        for (int k = 0; k < D; k++)
            s += Q[q * D + k] * K[j * D + k];
        l += expf(s * scale - m);
    }

    float out = 0.0f;
    for (int j = 0; j < N; j++) {
        float s = 0.0f;
        for (int k = 0; k < D; k++)
            s += Q[q * D + k] * K[j * D + k];
        out += expf(s * scale - m) / l * V[j * D + d];
    }
    O[q * D + d] = out;
}

// flash attention v2: Bc=64 tiles, float4 loads, register accumulators, cached scores
__global__ void flashattn_v2_kernel(const float* Q, const float* K, const float* V, float* O) {
    int warp = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    int q_row = blockIdx.x * Br + warp;
    if (warp >= Br || q_row >= N) return;

    extern __shared__ float smem[];
    float* sQ = smem;
    float* sK = sQ + Br * D;
    float* sV = sK + Bc * D;
    float* sS = sV + Bc * D;

    float scale = rsqrtf((float)D);
    float o0 = 0.0f, o1 = 0.0f;  // Register accumulators: keep output in registers, no shared memory writes until end
    float m = -1e20f, l = 0.0f;

    for (int idx = threadIdx.x; idx < Br * D; idx += blockDim.x) {
        int r = idx / D, c = idx % D;
        int gi = blockIdx.x * Br + r;
        sQ[idx] = (gi < N) ? Q[gi * D + c] : 0.0f;
    }
    __syncthreads();

    for (int tile = 0; tile < N; tile += Bc) {
        int cols = min(Bc, N - tile);

        // float4 loads: 4x fewer memory transactions, better bandwidth utilization
        for (int iv = threadIdx.x; iv < (Bc * D) / 4; iv += blockDim.x) {
            int idx = iv * 4;
            int tj = idx / D, c = idx % D, gj = tile + tj;
            if (tj < cols && gj < N) {
                *((float4*)(sK + idx)) = *((float4*)(K + gj * D + c));
                *((float4*)(sV + idx)) = *((float4*)(V + gj * D + c));
            } else {
                *((float4*)(sK + idx)) = make_float4(0, 0, 0, 0);
                *((float4*)(sV + idx)) = make_float4(0, 0, 0, 0);
            }
        }
        __syncthreads();

        // Score caching: store QK^T scores in shared memory to avoid recomputation in softmax pass
        float lane0_max = -1e20f;
        for (int tj = 0; tj < cols; tj++) {
            float acc = sQ[warp * D + lane] * sK[tj * D + lane]
                      + sQ[warp * D + lane + 32] * sK[tj * D + lane + 32];
            float s = __shfl_sync(0xffffffff, warpReduceSum(acc), 0) * scale;
            if (lane == 0) {
                sS[warp * Bc + tj] = s;
                lane0_max = fmaxf(lane0_max, s);
            }
        }
        float tile_max = __shfl_sync(0xffffffff, lane0_max, 0);

        // Online softmax: rescale running sum when max changes, avoids storing full score matrix
        float m_new = fmaxf(m, tile_max);
        float alpha = expf(m - m_new);
        if (lane == 0) l *= alpha;
        l = __shfl_sync(0xffffffff, l, 0);
        o0 *= alpha;
        o1 *= alpha;

        float l_tile = 0.0f;
        for (int tj = 0; tj < cols; tj++) {
            float s = __shfl_sync(0xffffffff, (lane == 0) ? sS[warp * Bc + tj] : 0.0f, 0);
            float w = expf(s - m_new);
            if (lane == 0) l_tile += w;
            o0 += w * sV[tj * D + lane];
            o1 += w * sV[tj * D + lane + 32];
        }

        l = __shfl_sync(0xffffffff, l + ((lane == 0) ? l_tile : 0.0f), 0);
        m = m_new;
        __syncthreads();
    }

    float inv_l = 1.0f / l;
    O[q_row * D + lane] = o0 * inv_l;
    O[q_row * D + lane + 32] = o1 * inv_l;
}

int main() {
    srand(0);
    size_t bytes = N * D * sizeof(float);
    float *hQ = (float*)malloc(bytes);
    float *hK = (float*)malloc(bytes);
    float *hV = (float*)malloc(bytes);
    float *hO_naive = (float*)malloc(bytes);
    float *hO_flash = (float*)malloc(bytes);

    for (int i = 0; i < N * D; i++) {
        hQ[i] = 2.f * rand() / RAND_MAX - 1.f;
        hK[i] = 2.f * rand() / RAND_MAX - 1.f;
        hV[i] = 2.f * rand() / RAND_MAX - 1.f;
    }

    float *dQ, *dK, *dV, *dO_naive, *dO_flash;
    CUDA_CHECK(cudaMalloc(&dQ, bytes));
    CUDA_CHECK(cudaMalloc(&dK, bytes));
    CUDA_CHECK(cudaMalloc(&dV, bytes));
    CUDA_CHECK(cudaMalloc(&dO_naive, bytes));
    CUDA_CHECK(cudaMalloc(&dO_flash, bytes));
    CUDA_CHECK(cudaMemcpy(dQ, hQ, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK, hK, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV, hV, bytes, cudaMemcpyHostToDevice));

    size_t shmem = (Br * D + 2 * Bc * D + Br * Bc) * sizeof(float);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int WARMUP = 10, ITERS = 50;
    for (int i = 0; i < WARMUP; i++) {
        attn_naive_kernel<<<N, D>>>(dQ, dK, dV, dO_naive);
        flashattn_v2_kernel<<<(N + Br - 1) / Br, Br * 32, shmem>>>(dQ, dK, dV, dO_flash);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < ITERS; i++)
        attn_naive_kernel<<<N, D>>>(dQ, dK, dV, dO_naive);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms_naive;
    CUDA_CHECK(cudaEventElapsedTime(&ms_naive, start, stop));
    ms_naive /= ITERS;

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < ITERS; i++)
        flashattn_v2_kernel<<<(N + Br - 1) / Br, Br * 32, shmem>>>(dQ, dK, dV, dO_flash);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms_flash;
    CUDA_CHECK(cudaEventElapsedTime(&ms_flash, start, stop));
    ms_flash /= ITERS;

    CUDA_CHECK(cudaMemcpy(hO_naive, dO_naive, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hO_flash, dO_flash, bytes, cudaMemcpyDeviceToHost));

    float max_diff = 0.0f;
    for (int i = 0; i < N * D; i++)
        max_diff = fmaxf(max_diff, fabsf(hO_flash[i] - hO_naive[i]));

    printf("naive  : %.3f ms / launch\n", ms_naive);
    printf("flash  : %.3f ms / launch\n", ms_flash);
    printf("speedup: %.2fx\n", ms_naive / ms_flash);
    printf("max |diff| = %.3e\n", max_diff);

    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO_naive); cudaFree(dO_flash);
    free(hQ); free(hK); free(hV); free(hO_naive); free(hO_flash);
    return 0;
}

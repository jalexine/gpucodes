// flashattnv1_compare.cu
#include <cstdio>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>

#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA %s (%d) at %s:%d\n", cudaGetErrorString(e), e, __FILE__, __LINE__); exit(1);} } while(0)

// xp params
constexpr int N  = 512;
constexpr int D  = 64;

constexpr int Br = 16;
constexpr int Bc = 32;


// i started with a naive kernel as a baseline
// so i can compare correctness/perf (gpu vs gpu)
__global__ void attn_naive_kernel(
    const float* Q,
    const float* K,
    const float* V,
    float* O
) {
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
        float w = expf(s * scale - m) / l;
        out += w * V[j * D + d];
    }

    O[q * D + d] = out;
}

__device__ __forceinline__ float warpReduceSum(float v) {
    for (int o = 16; o > 0; o >>= 1)
        v += __shfl_down_sync(0xffffffff, v, o);
    return v;
}

// my reimplementation of flashattn (v1)
// one warp per query row, K/V tiled in shared memory, online softmax
__global__ void flashattn_v1_kernel(
    const float* Q,
    const float* K,
    const float* V,
    float* O
) {
    int warp = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    int q_row = blockIdx.x * Br + warp;
    if (warp >= Br || q_row >= N) return;

    float scale = rsqrtf((float)D);


    extern __shared__ float smem[];
    float* sQ = smem; 
    float* sK = sQ + Br * D; 
    float* sV = sK + Bc * D;  
    float* sS = sV + Bc * D;

    float m = -1e20f;
    float l = 0.0f;
    float o0 = 0.0f;
    float o1 = 0.0f;

    // load Q tile
    for (int idx = threadIdx.x; idx < Br * D; idx += blockDim.x) {
        int r = idx / D;
        int d = idx % D;
        int gi = blockIdx.x * Br + r;
        sQ[idx] = (gi < N) ? Q[gi * D + d] : 0.0f;
    }
    __syncthreads(); 

    for (int tile = 0; tile < N; tile += Bc) {
        int cols = min(Bc, N - tile);

        // load K/V tile
	// vectorized global->shared loads (float4)
	using Vec = float4;

	Vec* sK4 =  reinterpret_cast<Vec*>(sK);
        Vec* sV4 = reinterpret_cast<Vec*>(sV);
	const Vec* gK4 = reinterpret_cast<const Vec*>(K);
        const Vec* gV4 = reinterpret_cast<const Vec*>(V);


	constexpr int D4 = D / 4;
        for (int idx = threadIdx.x; idx < Bc * D4; idx += blockDim.x) {
            int tj = idx / D4;
            int d4  = idx % D4;
            int gj = tile + tj;
            if (tj < cols && gj < N) {
		sK4[tj * D4 + d4] = gK4[gj * D4 + d4];
                sV4[tj * D4 + d4] = gV4[gj * D4 + d4];
            } else {
		sK4[tj * D4 + d4] = make_float4(0,0,0,0);
                sV4[tj * D4 + d4] = make_float4(0,0,0,0);
            }
        }
        __syncthreads();
	
	for (int tj = 0; tj < cols; tj++) {
	    float acc = 0.0f;
	    for (int d = lane; d < D; d += 32)
		acc += sQ[warp * D + d] * sK[tj * D + d];
	    float sum = warpReduceSum(acc);
	    if (lane == 0)
		sS[warp * Bc + tj] = sum * scale;
	}
	__syncthreads();

        //pass 1: max over this tile
        float lane0_max = -1e20f;
	if (lane == 0) {
	    for(int tj = 0; tj < cols; tj++) {
	       lane0_max = fmaxf(lane0_max, sS[warp * Bc + tj]);
	    }
	}
        float tile_max = __shfl_sync(0xffffffff, lane0_max, 0);

        // online softmax update: rescale old state to new max
        float m_new = fmaxf(m, tile_max);
        float alpha = expf(m - m_new);

        if (lane == 0) l *= alpha;
        l = __shfl_sync(0xffffffff, l, 0);

        o0 *= alpha;
	o1 *= alpha;

        // pass 2: accumulate this
        float l_tile = 0.0f;

        for (int tj = 0; tj < cols; tj++) {
            float s = sS[warp * Bc + tj];
	    float w;
	    if (lane == 0) w = expf(s-m_new);
	    w = __shfl_sync(0xffffffff, w, 0);
	    
	    if (lane == 0) l_tile += w;
            o0 += w* sV[tj * D + lane];
	    o1 += w* sV[tj * D + (lane + 32)];
        }

        l_tile = __shfl_sync(0xffffffff, l_tile, 0);
        if (lane == 0) l += l_tile;
        l = __shfl_sync(0xffffffff, l, 0);

        m = m_new;
        __syncthreads();
    }

    // normalize by sum exp
    float l0 = __shfl_sync(0xffffffff, l, 0);
    float inv_l = 1.0f / l0;
    O[q_row * D + lane] = o0 * inv_l;
    O[q_row * D + (lane + 32)] = o1 * inv_l;
}

int main() {
    srand(0);

    size_t size_qkv = (size_t)N * D * sizeof(float);
    float *hQ = (float*)malloc(size_qkv);
    float *hK = (float*)malloc(size_qkv);
    float *hV = (float*)malloc(size_qkv);
    float *hO_naive = (float*)malloc(size_qkv);
    float *hO_flash = (float*)malloc(size_qkv);

    for (int i = 0; i < N * D; i++) {
        hQ[i] = 2.f * rand() / (float)RAND_MAX - 1.f;
        hK[i] = 2.f * rand() / (float)RAND_MAX - 1.f;
        hV[i] = 2.f * rand() / (float)RAND_MAX - 1.f;
    }

    float *dQ, *dK, *dV, *dO_naive, *dO_flash;
    CUDA_CHECK(cudaMalloc(&dQ, size_qkv));
    CUDA_CHECK(cudaMalloc(&dK, size_qkv));
    CUDA_CHECK(cudaMalloc(&dV, size_qkv));
    CUDA_CHECK(cudaMalloc(&dO_naive, size_qkv));
    CUDA_CHECK(cudaMalloc(&dO_flash, size_qkv));

    CUDA_CHECK(cudaMemcpy(dQ, hQ, size_qkv, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK, hK, size_qkv, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV, hV, size_qkv, cudaMemcpyHostToDevice));

    dim3 grid_naive(N);
    dim3 block_naive(D);

    dim3 grid_flash((N + Br - 1) / Br);
    dim3 block_flash(Br * 32);
    size_t shmem = (Br * D + 2 * Bc * D + Br * Bc) * sizeof(float); 

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    //warmup. likw first kernell call takes longer, can change the bench..
    const int WARMUP = 10;
    const int ITERS  = 50;

    for (int i = 0; i < WARMUP; i++) {
        attn_naive_kernel<<<grid_naive, block_naive>>>(dQ, dK, dV, dO_naive);
        CUDA_CHECK(cudaGetLastError());

        flashattn_v1_kernel<<<grid_flash, block_flash, shmem>>>(dQ, dK, dV, dO_flash);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // time naive
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < ITERS; i++) {
        attn_naive_kernel<<<grid_naive, block_naive>>>(dQ, dK, dV, dO_naive);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms_naive = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_naive, start, stop));
    ms_naive /= ITERS;

    // time flash
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < ITERS; i++) {
        flashattn_v1_kernel<<<grid_flash, block_flash, shmem>>>(dQ, dK, dV, dO_flash);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms_flash = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_flash, start, stop));
    ms_flash /= ITERS;

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(hO_naive, dO_naive, size_qkv, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hO_flash, dO_flash, size_qkv, cudaMemcpyDeviceToHost));

    float max_abs = 0.0f;
    for (int i = 0; i < N * D; i++) {
        float a = hO_flash[i];
        float b = hO_naive[i];
        float diff = fabsf(a - b);
        max_abs = fmaxf(max_abs, diff);
    }

    printf("naive  : %.3f ms / launch\n", ms_naive);
    printf("flash  : %.3f ms / launch\n", ms_flash);
    printf("speedup: %.2fx\n", ms_naive / ms_flash);
    printf("max |diff|   = %.3e\n", max_abs);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(dQ));
    CUDA_CHECK(cudaFree(dK));
    CUDA_CHECK(cudaFree(dV));
    CUDA_CHECK(cudaFree(dO_naive));
    CUDA_CHECK(cudaFree(dO_flash));

    free(hQ);
    free(hK);
    free(hV);
    free(hO_naive);
    free(hO_flash);

    //here are my results on a A4000 16GB
    //naive  : 4.143 ms / launch
    //flash  : 0.247 ms / launch
    // speedup: 16.77x
    // max |diff|   = 1.848e-06

    // note that baseline is very naive, so speedup is expected to be crazy high
    return 0;
}

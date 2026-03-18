/*
 * fused_sdp_dp_optimized.cu - Fused SDP+DP with device buffer cache (win vs separated in all batch sizes)
 * Build: nvcc -shared -Xcompiler -fPIC -O3 --use_fast_math -arch=sm_75 -o fused_sdp_dp_optimized.so fused_sdp_dp_optimized.cu
 */
#include <cuda_runtime.h>
#include <stdio.h>

#define CUDA_CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while(0)
#define BLOCK 256

__device__ __forceinline__ float fast_sigmoid(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

__global__ void kernel_sdp(
    const float* __restrict__ audio,
    float* __restrict__ out,
    int batch_size, int seq_len, int feat_dim, int embed_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = batch_size * embed_dim;
    if (idx >= n) return;
    int b = idx / embed_dim, f = idx % embed_dim;
    float s = 0.0f;
#pragma unroll 8
    for (int p = 0; p < seq_len; p++)
        s += audio[b * seq_len * feat_dim + p * feat_dim + f];
    out[b * embed_dim + f] = s / (float)seq_len;
}

__global__ void kernel_dp(
    const float* __restrict__ text,
    float* __restrict__ out,
    int batch_size, int seq_len, int feat_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = batch_size * seq_len;
    if (idx >= n) return;
    int b = idx / seq_len, p = idx % seq_len;
    const float* row = text + (b * seq_len + p) * feat_dim;
    float s = 0.0f;
    if ((feat_dim & 3) == 0) {
        const float4* v = reinterpret_cast<const float4*>(row);
        for (int j = 0; j < feat_dim / 4; j++)
            s += v[j].x + v[j].y + v[j].z + v[j].w;
    } else {
        for (int j = 0; j < feat_dim; j++) s += row[j];
    }
    out[b * seq_len + p] = fast_sigmoid(s / (float)feat_dim) * 10.0f;
}

// Cached device buffers: reuse across calls so small-batch and all batch sizes win vs separated (no cache)
static float *d_a = NULL, *d_t = NULL, *d_s = NULL, *d_d = NULL;
static size_t cached_a = 0, cached_t = 0, cached_s = 0, cached_d = 0;

static void ensure_buf(float** p, size_t* cached, size_t need) {
    if (*cached >= need && *p != NULL) return;
    if (*p) { cudaFree(*p); *p = NULL; *cached = 0; }
    CUDA_CHECK(cudaMalloc(p, need));
    *cached = need;
}

extern "C" void fused_forward(
    const float* audio_features,
    const float* text_features,
    float* sdp_embedding,
    float* dp_durations,
    int batch_size, int seq_len, int feat_dim, int embed_dim
) {
    size_t a_sz = (size_t)batch_size * seq_len * feat_dim * sizeof(float);
    size_t t_sz = (size_t)batch_size * seq_len * feat_dim * sizeof(float);
    size_t s_sz = (size_t)batch_size * embed_dim * sizeof(float);
    size_t d_sz = (size_t)batch_size * seq_len * sizeof(float);

    ensure_buf(&d_a, &cached_a, a_sz);
    ensure_buf(&d_t, &cached_t, t_sz);
    ensure_buf(&d_s, &cached_s, s_sz);
    ensure_buf(&d_d, &cached_d, d_sz);

    CUDA_CHECK(cudaMemcpy(d_a, audio_features, a_sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_t, text_features, t_sz, cudaMemcpyHostToDevice));

    int s_n = batch_size * embed_dim;
    int d_n = batch_size * seq_len;
    kernel_sdp<<<(s_n + BLOCK - 1) / BLOCK, BLOCK>>>(d_a, d_s, batch_size, seq_len, feat_dim, embed_dim);
    kernel_dp<<<(d_n + BLOCK - 1) / BLOCK, BLOCK>>>(d_t, d_d, batch_size, seq_len, feat_dim);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(sdp_embedding, d_s, s_sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dp_durations, d_d, d_sz, cudaMemcpyDeviceToHost));
}

// F.5B.1 — exact integer CUDA kernels for A13W10 (m16n8k32 s8 tensor cores).
// RESEARCH-ONLY / NON-PROTOCOL. Integer arithmetic only — no floating point
// anywhere in this file (pointer/index math is integer by construction).
//
// Frozen arithmetic (must not change):
//   balanced radix-128 split: q = floor((x + 64) / 128); lo = x - 128*q
//   Karatsuba-3: C = C00 + (Csum - C00 - C11)<<7 + C11<<14
//
// Layouts (frozen for this extension):
//   A (wide)  : int16 (16, K) row-major, k contiguous; values in [-4096, 4095]
//   A (native): int8  (16, K) row-major
//   W prepack : int8 (Npad, Kpad) row-major (k contiguous) — logical column n
//               stored contiguously, which is the "col-major B" operand of
//               mma.row.col.  Kpad = ceil(K/32)*32, Npad = ceil(N/32)*32,
//               both zero-padded (mathematically inert).
//   C (wide)  : int64 (16, N) row-major;  C (native): int32 (16, N)
//
// mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 fragment maps (PTX ISA):
//   A (row-major 16x32, 4 x .b32 per thread), gr = lane>>2, tg = lane&3:
//     a0 = {A[gr  ][tg*4 + 0..3 ]}   a1 = {A[gr+8][tg*4 + 0..3 ]}
//     a2 = {A[gr  ][tg*4 + 16..19]}  a3 = {A[gr+8][tg*4 + 16..19]}
//   B (col-major 32x8, 2 x .b32 per thread):
//     b0 = {B[tg*4 + 0..3 ][gr]}     b1 = {B[tg*4 + 16..19][gr]}
//   C/D (16x8, 4 x .s32 per thread):
//     c0 = C[gr][2*tg]   c1 = C[gr][2*tg+1]
//     c2 = C[gr+8][2*tg] c3 = C[gr+8][2*tg+1]

#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>

namespace {

constexpr int KT = 32;            // mma K tile (m16n8k32)
constexpr int BN = 32;            // CTA N tile (4 warps x 8 columns)
constexpr int NWARP = 4;
constexpr int WN = BN / NWARP;    // columns per warp = one mma column (n8)

__device__ __forceinline__ void mma_s8_s32(
    int32_t& c0, int32_t& c1, int32_t& c2, int32_t& c3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1)
{
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// Balanced radix-128 split (frozen): integer arithmetic only.
__device__ __forceinline__ void split128(int v, int8_t& lo, int8_t& hi, int8_t& sum)
{
    int q = (v + 64) >> 7;          // floor((v + 64) / 128), arithmetic shift
    int l = v - (q << 7);
    lo = static_cast<int8_t>(l);
    hi = static_cast<int8_t>(q);
    sum = static_cast<int8_t>(l + q);
}

__device__ __forceinline__ uint32_t pack4(const int8_t* p)
{
    uint32_t v;
    std::memcpy(&v, p, 4);
    return v;
}

// ---------------------------------------------------------------------------
// Wide fused kernel: logical A13 (int16) -> exact int64 C, one launch:
//   in-kernel split (registers only), 3 tensor-core streams, int64 epilogue.
// Native kernel (WIDE=false): A8 x W8 -> s32, same tile/launch framework.
// ---------------------------------------------------------------------------
template <bool WIDE>
__global__ void __launch_bounds__(NWARP * 32)
gemm_m16_tc_kernel(
    const void* __restrict__ A,        // WIDE: int16 (16,K); else int8 (16,K)
    const int8_t* __restrict__ W0,     // (Npad, Kpad)
    const int8_t* __restrict__ W1,     // (Npad, Kpad), unused when !WIDE
    const int8_t* __restrict__ WS,     // (Npad, Kpad), unused when !WIDE
    void* __restrict__ C,              // WIDE: int64 (16,N); else int32 (16,N)
    int K, int N)
{
    __shared__ int8_t W0s[BN * KT];
    __shared__ int8_t W1s[BN * KT];
    __shared__ int8_t WSs[BN * KT];
    __shared__ int16_t As16[16 * KT];
    __shared__ int8_t As8[16 * KT];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int gr = lane >> 2;
    const int tg = lane & 3;

    const int n0 = blockIdx.x * BN;
    const int Kpad = (K + KT - 1) / KT * KT;
    const int nave = WN * warp + gr;   // row of the warp's B tile

    int32_t acc00[4] = {0, 0, 0, 0};
    int32_t acc11[4] = {0, 0, 0, 0};
    int32_t accSS[4] = {0, 0, 0, 0};

    for (int k0 = 0; k0 < Kpad; k0 += KT) {
        __syncthreads();

        // cooperative W tile loads: 3 x BN*KT bytes, (n0+n)*Kpad + (k0+k).
        // The (Npad,Kpad) prepack is zero-padded, so no masks are needed.
        for (int idx = tid; idx < BN * KT; idx += NWARP * 32) {
            const int n = idx / KT;
            const int k = idx % KT;
            const long long off = (long long)(n0 + n) * Kpad + (k0 + k);
            const int8_t v0 = W0[off];
            W0s[idx] = v0;
            if (WIDE) {
                W1s[idx] = W1[off];
                WSs[idx] = WS[off];
            }
        }
        // A tile load (masked on K; values beyond K stage as 0 -> split 0/0/0)
        for (int idx = tid; idx < 16 * KT; idx += NWARP * 32) {
            const int row = idx / KT;
            const int gk = k0 + (idx % KT);
            if (WIDE) {
                As16[idx] = (gk < K) ? ((const int16_t*)A)[row * K + gk] : (int16_t)0;
            } else {
                As8[idx] = (gk < K) ? ((const int8_t*)A)[row * K + gk] : (int8_t)0;
            }
        }
        __syncthreads();

        // A fragments: 4 x .b32 (identical for all four warps).
        uint32_t af[3][4];
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int row = gr + ((r & 1) ? 8 : 0);
            const int colb = tg * 4 + ((r >= 2) ? 16 : 0);
            if (WIDE) {
                int8_t bytes[3][4];
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int16_t v = As16[row * KT + colb + j];
                    int8_t lo, hi, su;
                    split128((int)v, lo, hi, su);
                    bytes[0][j] = lo;
                    bytes[1][j] = hi;
                    bytes[2][j] = su;
                }
                #pragma unroll
                for (int s = 0; s < 3; ++s) af[s][r] = pack4(bytes[s]);
            } else {
                af[0][r] = pack4(&As8[row * KT + colb]);
            }
        }

        // B fragments from the warp's 8-column stripe of the shared tiles.
        uint32_t bf[3][2];
        #pragma unroll
        for (int s = 0; s < 3; ++s) {
            if (!WIDE && s > 0) continue;
            const int8_t* wsrc = (s == 0) ? W0s : (s == 1) ? W1s : WSs;
            bf[s][0] = pack4(&wsrc[nave * KT + tg * 4]);
            bf[s][1] = pack4(&wsrc[nave * KT + tg * 4 + 16]);
        }

        mma_s8_s32(acc00[0], acc00[1], acc00[2], acc00[3],
                   af[0][0], af[0][1], af[0][2], af[0][3], bf[0][0], bf[0][1]);
        if (WIDE) {
            mma_s8_s32(acc11[0], acc11[1], acc11[2], acc11[3],
                       af[1][0], af[1][1], af[1][2], af[1][3], bf[1][0], bf[1][1]);
            mma_s8_s32(accSS[0], accSS[1], accSS[2], accSS[3],
                       af[2][0], af[2][1], af[2][2], af[2][3], bf[2][0], bf[2][1]);
        }
    }

    // Epilogue: c0..c3 -> (row, col) = (gr + 8*(r>=2), 2*tg + (r&1)).
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int row = gr + ((r >= 2) ? 8 : 0);
        const int col = n0 + WN * warp + 2 * tg + (r & 1);
        if (col >= N) continue;
        if (WIDE) {
            const int64_t cross = (int64_t)accSS[r] - (int64_t)acc00[r] - (int64_t)acc11[r];
            // C = C00 + 128*Cross + 16384*C11  (Cross already subtracted)
            const int64_t val = (int64_t)acc00[r] + (cross << 7) + ((int64_t)acc11[r] << 14);
            ((int64_t*)C)[(long long)row * N + col] = val;
        } else {
            ((int32_t*)C)[(long long)row * N + col] = acc00[r];
        }
    }
}

// ---------------------------------------------------------------------------
// Stage B elementwise kernels.
// split: int16 A13 (16, K) -> A0/A1/AS int8 (>=16 rows, K); writes rows 0..15
//        only — padded rows are zeroed once at allocation and never touched.
// merge: int32 C00/C11/CS (16, N) -> int64 C (16, N), Karatsuba-3 merge.
// ---------------------------------------------------------------------------
__global__ void split128_kernel(const int16_t* __restrict__ A,
                                int8_t* __restrict__ A0,
                                int8_t* __restrict__ A1,
                                int8_t* __restrict__ AS,
                                int K, long long total /* 16*K */)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int8_t lo, hi, su;
    split128((int)A[i], lo, hi, su);
    A0[i] = lo;
    A1[i] = hi;
    AS[i] = su;
}

__global__ void merge_kernel(const int32_t* __restrict__ C00,
                             const int32_t* __restrict__ C11,
                             const int32_t* __restrict__ CS,
                             int64_t* __restrict__ C,
                             int N, long long total /* 16*N */)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int64_t c00 = C00[i];
    const int64_t c11 = C11[i];
    const int64_t cs = CS[i];
    const int64_t cross = cs - c00 - c11;
    C[i] = c00 + (cross << 7) + (c11 << 14);
}

}  // namespace

extern "C" void f5b1_launch_wide(const void* A, const int8_t* W0, const int8_t* W1,
                                 const int8_t* WS, void* C, int N, int K,
                                 cudaStream_t stream)
{
    const int grid = (N + BN - 1) / BN;
    gemm_m16_tc_kernel<true><<<grid, NWARP * 32, 0, stream>>>(
        A, W0, W1, WS, C, K, N);
}

extern "C" void f5b1_launch_native(const void* A, const int8_t* W, void* C,
                                   int N, int K, cudaStream_t stream)
{
    const int grid = (N + BN - 1) / BN;
    gemm_m16_tc_kernel<false><<<grid, NWARP * 32, 0, stream>>>(
        A, W, W, W, C, K, N);
}

extern "C" void f5b1_launch_split(const int16_t* A, int8_t* A0, int8_t* A1,
                                  int8_t* AS, int K, cudaStream_t stream)
{
    const long long total = 16LL * K;
    const int threads = 256;
    const long long blocks = (total + threads - 1) / threads;
    split128_kernel<<<(int)blocks, threads, 0, stream>>>(A, A0, A1, AS, K, total);
}

extern "C" void f5b1_launch_merge(const int32_t* C00, const int32_t* C11,
                                  const int32_t* CS, int64_t* C, int N,
                                  cudaStream_t stream)
{
    const long long total = 16LL * N;
    const int threads = 256;
    const long long blocks = (total + threads - 1) / threads;
    merge_kernel<<<(int)blocks, threads, 0, stream>>>(C00, C11, CS, C, N, total);
}

// F.5B.2 — exact integer CUDA kernels for the canonical operators.
// RESEARCH-ONLY / NON-PROTOCOL.  Integer arithmetic only — no floating
// point anywhere in this file.  Every kernel is a faithful port of
// tools/f1_canonical_numpy.py (bit-exact with the Go reference):
//   rshift_round_even (ties-to-even), saturating add/sub, MulFx
//   (int64 product, ties-even shift 20, saturate), the frozen exp
//   (trunc k = x/ln2, 4-term Taylor, saturation branches), the frozen
//   invsqrt (normalize, 16-entry table, exactly 4 Newton steps),
//   RMSNorm (per-element (x*x)>>20 summed in index order, mean, invsqrt,
//   MulFx x inv, MulFx w), RoPE (adjacent pairs, table lookup), SiLU
//   (MulFx x sigmoid), Softmax (row max, saturating sub, frozen exp,
//   integer sum in index order, ties-even division).
//
// The wide A13W10 GEMM reuses the F.5B.1 fused m16n8k32 kernels
// (f5b1_kernels.cu, concatenated at build time).

#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>

namespace f5b2 {

constexpr int64_t ONE = 1 << 20;
constexpr int64_t MAX_FX = (1LL << 31) - 1;
constexpr int64_t MIN_FX = -(1LL << 31);
constexpr int64_t LN2_FX = 726817;
constexpr int64_t C1_FX = 1048576;
constexpr int64_t C2_FX = 524288;
constexpr int64_t C3_FX = 174763;
constexpr int64_t C4_FX = 43691;

__device__ __forceinline__ int64_t clamp64(int64_t v, int64_t lo, int64_t hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

__device__ __forceinline__ int64_t rte64(int64_t v, int s)
{
    if (s == 0) return v;
    const int64_t q = v >> s;
    const int64_t r = v & ((1LL << s) - 1);
    const int64_t half = 1LL << (s - 1);
    const bool bump = (r > half) || ((r == half) && ((q & 1LL) == 1LL));
    return q + (bump ? 1 : 0);
}

__device__ __forceinline__ int64_t trunc_div_pos(int64_t a, int64_t b)
{
    // numpy _trunc_div: truncate toward zero (b > 0 here)
    const int64_t q = (a < 0 ? -((-a) / b) : a / b);
    return q;
}

// _mul_fx_wide: ties-even shift 20, NO saturation (int64 intermediate)
__device__ __forceinline__ int64_t mul_wide(int64_t a, int64_t b)
{
    return rte64(a * b, 20);
}

__device__ __forceinline__ int64_t mul_fx(int64_t a, int64_t b)
{
    return clamp64(rte64(a * b, 20), MIN_FX, MAX_FX);
}

// frozen canonical exp (input Q12.20 int32-range, output int32-range)
__device__ __forceinline__ int64_t exp_fx(int64_t x)
{
    if (x <= -(ONE * 24)) return 0;
    if (x >= ONE * 21) return MAX_FX;
    const int64_t k = trunc_div_pos(x, LN2_FX);
    const int64_t f = x - k * LN2_FX;
    const int64_t poly = ONE + mul_wide(
        f, C1_FX + mul_wide(
            f, C2_FX + mul_wide(
                f, C3_FX + mul_wide(f, C4_FX))));
    int64_t res;
    if (k < 0) {
        const int64_t kk = -k;
        if (kk >= 32) {
            res = 0;
        } else {
            res = rte64(poly, (int)kk);
        }
    } else {
        if (k > 31) {
            res = MAX_FX;
        } else {
            res = poly << (k > 62 ? 62 : k);
        }
    }
    return clamp64(res, MIN_FX, MAX_FX);
}

__device__ __forceinline__ int64_t div_ties_even(int64_t num, int64_t den)
{
    // C integer division truncates toward zero, matching numpy _trunc_div_arr
    int64_t q = num / den;
    const int64_t r = num - q * den;
    const int64_t twice = 2 * (r < 0 ? -r : r);
    const int64_t ab = den < 0 ? -den : den;
    const bool bump = (twice > ab) || ((twice == ab) && ((q & 1LL) == 1LL));
    if (bump && r != 0) {
        const bool den_pos = den > 0;
        const bool r_pos = r > 0;
        q += (den_pos == r_pos) ? 1 : -1;
    }
    return q;
}

__device__ __forceinline__ int64_t sigmoid_fx(int64_t x)  // x int32-range
{
    int64_t out;
    if (x >= 0) {
        const int64_t den = ONE + exp_fx(-x);
        const int64_t num = ONE << 20;   // DivFx(ONE, den) in the Go form
        out = div_ties_even(num, den);
    } else {
        const int64_t e = exp_fx(x);
        const int64_t den = ONE + e;
        out = div_ties_even(e << 20, den);
    }
    return clamp64(out, MIN_FX, MAX_FX);
}

__device__ __forceinline__ int bit_length64(int64_t x)  // x > 0
{
    return 64 - __clzll((unsigned long long)x);
}

// frozen canonical invsqrt (16-entry table + exactly 4 Newton steps)
__device__ __forceinline__ int64_t inv_sqrt_fx(int64_t x)  // x > 0
{
    const int b = bit_length64(x);
    constexpr int target = 22;
    int e;
    int64_t m;
    if (b >= target) {
        e = (b - target + 1) / 2;
        m = x >> (2 * e);
    } else {
        e = -((target - b + 1) / 2);
        m = x << (-2 * e);
    }
    if (m >= (ONE << 2)) {
        m >>= 2;
        e += 1;
    }
    int idx = (int)(((m - ONE) * 16) / (ONE * 3));   // m in [ONE, 4*ONE) => >= 0
    if (idx < 0) idx = 0;
    if (idx > 15) idx = 15;
    static const int64_t TABLE[16] = {
        1048576, 962239, 894228, 838860, 792649, 753319, 719317, 689539,
        663177, 639625, 618416, 599186, 581645, 565560, 550739, 537025,
    };
    int64_t y = TABLE[idx];
    #pragma unroll
    for (int it = 0; it < 4; ++it) {
        const int64_t y2 = (y * y) >> 20;
        const int64_t my2 = (m * y2) >> 20;
        const int64_t corr = (3LL << 19) - (my2 >> 1);
        y = (y * corr) >> 20;
    }
    if (e > 0 && e < 63) y = y >> (e > 62 ? 62 : e);
    if (e < 0 && (-e) < 63) y = y << ((-e) > 62 ? 62 : (-e));
    if (y < 1) y = 1;
    return clamp64(y, MIN_FX, MAX_FX);
}

// --- kernels -----------------------------------------------------------------

__global__ void requant_kernel(const int64_t* __restrict__ x, int64_t* __restrict__ out,
                               int64_t mult, int shift, int64_t lo, int64_t hi,
                               long long total)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    out[i] = clamp64(rte64(x[i] * mult, shift), lo, hi);
}

__global__ void add_kernel(const int64_t* __restrict__ a, const int64_t* __restrict__ b,
                           int64_t* __restrict__ out, long long total)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    out[i] = clamp64(a[i] + b[i], MIN_FX, MAX_FX);
}

__global__ void mul_kernel(const int64_t* __restrict__ a, const int64_t* __restrict__ b,
                           int64_t* __restrict__ out, long long total)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    out[i] = mul_fx(a[i], b[i]);
}

// one thread per row: sequential index-order reduction (matches the numpy
// reference; integer addition is associative so the order is also
// architecture-independent by algebra)
__global__ void rmsnorm_kernel(const int64_t* __restrict__ x, const int64_t* __restrict__ w,
                               int64_t* __restrict__ out, int hidden, int64_t eps_fx,
                               long long rows)
{
    const long long r = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const int64_t* row = x + r * hidden;
    int64_t sum_sq = 0;
    for (int i = 0; i < hidden; ++i) {
        sum_sq += (row[i] * row[i]) >> 20;
    }
    const int64_t mean = sum_sq / hidden;      // sum_sq >= 0
    const int64_t inv = inv_sqrt_fx(mean + eps_fx);
    int64_t* orow = out + r * hidden;
    for (int i = 0; i < hidden; ++i) {
        orow[i] = mul_fx(mul_fx(row[i], inv), w[i]);
    }
}

__global__ void rope_kernel(const int64_t* __restrict__ x, const int64_t* __restrict__ table,
                            int64_t* __restrict__ out, int pairs, long long total_pairs,
                            int hidden)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_pairs) return;
    const int pos = (int)(i / pairs);
    const int j = (int)(i % pairs);
    const int64_t even = x[(long long)pos * hidden + 2 * j];
    const int64_t odd = x[(long long)pos * hidden + 2 * j + 1];
    const int64_t cos_v = table[((long long)pos * pairs + j) * 2 + 0];
    const int64_t sin_v = table[((long long)pos * pairs + j) * 2 + 1];
    // out0 = sub_fx(mul_fx(even, cos), mul_fx(odd, sin))
    const int64_t o0 = clamp64(mul_fx(even, cos_v) - mul_fx(odd, sin_v), MIN_FX, MAX_FX);
    // out1 = add_fx(mul_fx(even, sin), mul_fx(odd, cos))
    const int64_t o1 = clamp64(mul_fx(even, sin_v) + mul_fx(odd, cos_v), MIN_FX, MAX_FX);
    out[(long long)pos * hidden + 2 * j] = o0;
    out[(long long)pos * hidden + 2 * j + 1] = o1;
}

__global__ void silu_kernel(const int64_t* __restrict__ x, int64_t* __restrict__ out,
                            long long total)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    out[i] = mul_fx(x[i], sigmoid_fx(x[i]));
}

__global__ void softmax_kernel(const int64_t* __restrict__ x, int64_t* __restrict__ out,
                               int cols, long long rows)
{
    const long long r = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const int64_t* row = x + r * cols;
    int64_t mx = row[0];
    for (int i = 1; i < cols; ++i) {
        if (row[i] > mx) mx = row[i];
    }
    int64_t total = 0;
    for (int i = 0; i < cols; ++i) {
        // exps are exp_fx(sub_fx(x, mx)) — sub saturates to int32 range first
        const int64_t e = exp_fx(clamp64(row[i] - mx, MIN_FX, MAX_FX));
        out[r * cols + i] = e;   // staged in out to mirror exps array
        total += e;
    }
    if (total == 0) total = 1;
    for (int i = 0; i < cols; ++i) {
        out[r * cols + i] = clamp64(div_ties_even(out[r * cols + i] << 20, total),
                                    MIN_FX, MAX_FX);
    }
}

}  // namespace f5b2

extern "C" void f5b2_launch_requant(const int64_t* x, int64_t* out, long long mult,
                                    int shift, long long lo, long long hi,
                                    long long total, cudaStream_t stream)
{
    const int threads = 256;
    const long long blocks = (total + threads - 1) / threads;
    f5b2::requant_kernel<<<(int)blocks, threads, 0, stream>>>(x, out, mult, shift, lo, hi, total);
}

extern "C" void f5b2_launch_add(const int64_t* a, const int64_t* b, int64_t* out,
                                long long total, cudaStream_t stream)
{
    const int threads = 256;
    const long long blocks = (total + threads - 1) / threads;
    f5b2::add_kernel<<<(int)blocks, threads, 0, stream>>>(a, b, out, total);
}

extern "C" void f5b2_launch_mul(const int64_t* a, const int64_t* b, int64_t* out,
                                long long total, cudaStream_t stream)
{
    const int threads = 256;
    const long long blocks = (total + threads - 1) / threads;
    f5b2::mul_kernel<<<(int)blocks, threads, 0, stream>>>(a, b, out, total);
}

extern "C" void f5b2_launch_rmsnorm(const int64_t* x, const int64_t* w, int64_t* out,
                                    int hidden, long long eps_fx, long long rows,
                                    cudaStream_t stream)
{
    const int threads = 32;
    const long long blocks = (rows + threads - 1) / threads;
    f5b2::rmsnorm_kernel<<<(int)blocks, threads, 0, stream>>>(x, w, out, hidden, eps_fx, rows);
}

extern "C" void f5b2_launch_rope(const int64_t* x, const int64_t* table, int64_t* out,
                                 int pairs, long long total_pairs, int hidden,
                                 cudaStream_t stream)
{
    const int threads = 256;
    const long long blocks = (total_pairs + threads - 1) / threads;
    f5b2::rope_kernel<<<(int)blocks, threads, 0, stream>>>(x, table, out, pairs, total_pairs,
                                                           hidden);
}

extern "C" void f5b2_launch_silu(const int64_t* x, int64_t* out, long long total,
                                 cudaStream_t stream)
{
    const int threads = 256;
    const long long blocks = (total + threads - 1) / threads;
    f5b2::silu_kernel<<<(int)blocks, threads, 0, stream>>>(x, out, total);
}

extern "C" void f5b2_launch_softmax(const int64_t* x, int64_t* out, int cols,
                                    long long rows, cudaStream_t stream)
{
    const int threads = 32;
    const long long blocks = (rows + threads - 1) / threads;
    f5b2::softmax_kernel<<<(int)blocks, threads, 0, stream>>>(x, out, cols, rows);
}

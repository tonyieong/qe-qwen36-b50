// decode_bench: time one decoded token of Qwen3.6-35B-A3B on the Arc Pro B50 with hand-written SYCL kernels.
// Random weights; shapes, formats and op order follow llama.cpp's qwen35moe graph. Not numerically the model.
//   formats: Q8 = int8 + fp16 scale / 32 (8.5 bpw), Q6 = 4-bit low plane + 2-bit high plane + fp16 scale / 32 (6.5 bpw),
//            Q4 = nibbles + fp16 scale / 32 (4.5 bpw) for routed experts, router weights fp32.
// usage: decode_bench <ctx_len> <expert_mode: all|none> [tokens]
// Build: icpx -fsycl -O3 -fsycl-targets=spir64_gen -Xs "-device bmg-g21" decode_bench.cpp -o decode_bench
#include <sycl/sycl.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#include <atomic>
#include <thread>
#include <immintrin.h>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>

namespace syclex = sycl::ext::oneapi::experimental;

constexpr int SG = 16;
constexpr int D = 2048;          // hidden
constexpr int N_LAYER = 40;
constexpr int N_EXPERT = 256, K_USED = 8, FF_EXP = 512, FF_SH = 512;
constexpr int VOCAB = 248320;
// gated delta net
constexpr int DK = 128, NK = 16, NV = 32;             // head dim, k heads, v heads
constexpr int QKV_DIM = DK * NK * 2 + DK * NV;        // 8192
constexpr int Z_DIM = DK * NV;                        // 4096
constexpr int CONV_K = 4;
// full attention
constexpr int HD = 256, NQ = 16, NKV = 2, ROT = 64;
constexpr int Q_ROWS = HD * NQ * 2;                   // q + gate = 8192
constexpr int ATT_CHUNK = 256;

enum Fmt { F_Q8, F_Q6, F_Q4, F_F32 };
// per row: [quant data][fp16 scale per 32 weights]; Q6 quant data = 4-bit low plane (cols/2) + 2-bit high plane (cols/4)
static inline size_t qbytes(Fmt f, int cols) {
    return f == F_Q8 ? (size_t) cols : f == F_Q6 ? (size_t) cols / 2 + cols / 4 : (size_t) cols / 2;
}
static size_t row_bytes(Fmt f, int cols) {
    return f == F_F32 ? (size_t) cols * 4 : qbytes(f, cols) + (size_t) (cols / 32) * 2;
}

// dot of block b (32 weights) of a row with x[b*32 .. b*32+31]; all loads are 16/8-byte vectors
static inline float row_blk_dot(Fmt f, const uint8_t * row, int cols, int b, const float * x) {
    const sycl::float4 * xv = (const sycl::float4 *) (x + b * 32);
    float xs[32];
#pragma unroll
    for (int i = 0; i < 8; ++i) { const sycl::float4 t = xv[i]; xs[4 * i] = t.x(); xs[4 * i + 1] = t.y(); xs[4 * i + 2] = t.z(); xs[4 * i + 3] = t.w(); }
    float acc = 0.f;
    if (f == F_Q8) {
        const sycl::uint4 * qv = (const sycl::uint4 *) (row + (size_t) b * 32);
        const sycl::uint4 a = qv[0], c = qv[1];
        const uint32_t w[8] = {a.x(), a.y(), a.z(), a.w(), c.x(), c.y(), c.z(), c.w()};
#pragma unroll
        for (int i = 0; i < 8; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j) acc += (float) (int8_t) ((w[i] >> (8 * j)) & 0xff) * xs[4 * i + j];
        return acc * (float) ((const sycl::half *) (row + cols))[b];
    }
    if (f == F_Q6) {
        const sycl::uint4 lo = *(const sycl::uint4 *) (row + (size_t) b * 16);
        const sycl::uint2 hi = *(const sycl::uint2 *) (row + (size_t) cols / 2 + (size_t) b * 8);
        const uint32_t w[4] = {lo.x(), lo.y(), lo.z(), lo.w()};
        const uint64_t hb = ((uint64_t) hi.y() << 32) | hi.x();
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int k = i * 8 + j;
                const int v = (int) ((w[i] >> (4 * j)) & 0xf) | (int) (((hb >> (2 * k)) & 0x3) << 4);
                acc += (float) (v - 32) * xs[k];
            }
        return acc * (float) ((const sycl::half *) (row + (size_t) cols / 2 + cols / 4))[b];
    }
    // Q4
    const sycl::uint4 q = *(const sycl::uint4 *) (row + (size_t) b * 16);
    const uint32_t w[4] = {q.x(), q.y(), q.z(), q.w()};
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j) acc += (float) ((int) ((w[i] >> (4 * j)) & 0xf) - 8) * xs[i * 8 + j];
    return acc * (float) ((const sycl::half *) (row + (size_t) cols / 2))[b];
}

struct Mat {
    uint8_t * w = nullptr;
    Fmt f;
    int rows, cols;
    size_t rb;
};

struct Layer {
    bool recr;
    // linear attention
    Mat qkvz, ab, out;            // qkvz: 8192+4096 rows, ab: 64 rows f32 (alpha, beta), out: 2048 x 4096
    float * conv_w, * conv_state, * ssm_state, * ssm_norm, * a_log, * dt_bias;
    // full attention
    Mat qkv, wo;                  // qkv: 8192 + 512 + 512 rows
    float * qn, * kn;
    sycl::half * kcache, * vcache;
    // ffn
    Mat router_sh;                // rows: 256 router + 1 shared gate (f32)
    Mat sh_gu, sh_down;           // shared expert
    float * attn_norm, * post_norm;
};

static std::mt19937 rng(7);

// per-kernel timing (only while g_prof is set)
static bool g_prof = false;
static std::vector<std::pair<const char *, sycl::event>> g_ev;
static inline void P(const char * name, sycl::event e) { if (g_prof) g_ev.push_back({name, e}); }
static inline sycl::event P2(const char * name, sycl::event e) { P(name, e); return e; }

static uint8_t * dev_random(sycl::queue & q, size_t bytes, bool small_scales) {
    auto * p = sycl::malloc_device<uint8_t>(bytes, q);
    std::vector<uint8_t> h(std::min<size_t>(bytes, 32 << 20));
    for (auto & c : h) c = rng() & 0xff;
    for (size_t off = 0; off < bytes; off += h.size()) q.memcpy(p + off, h.data(), std::min(h.size(), bytes - off));
    q.wait();
    (void) small_scales;
    return p;
}
static float * dev_fill(sycl::queue & q, size_t n, float v) {
    auto * p = sycl::malloc_device<float>(n, q);
    q.fill(p, v, n).wait();
    return p;
}
static Mat make_mat(sycl::queue & q, Fmt f, int rows, int cols, size_t & total) {
    Mat m{nullptr, f, rows, cols, row_bytes(f, cols)};
    const size_t bytes = m.rb * rows;
    if (f == F_F32) {
        m.w = (uint8_t *) dev_fill(q, (size_t) rows * cols, 0.001f);
    } else {
        m.w = dev_random(q, bytes, true);
        // fix every scale to a small finite value so the math stays finite
        const int nb = cols / 32;
        const size_t rb = m.rb, qb = qbytes(f, cols);
        uint8_t * w = m.w;
        q.parallel_for(sycl::range<1>((size_t) rows * nb), [=](sycl::id<1> i) {
            ((sycl::half *) (w + (i[0] / nb) * rb + qb))[i[0] % nb] = sycl::half(0.0005f);
        }).wait();
    }
    total += bytes;
    return m;
}

// ---- kernels ----
// xn = rmsnorm(x) * nw   (one work-group)
static float * g_xn = nullptr;
static const float * norm(sycl::queue & q, const float * x, const float * nw, int n = D) {
    float * xn = g_xn;
    P("rmsnorm", q.parallel_for(sycl::nd_range<1>(512, 512), [=](sycl::nd_item<1> it) {
        float ss = 0.f;
        for (int i = it.get_local_id(0); i < n; i += 512) ss += x[i] * x[i];
        ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
        const float inv = sycl::rsqrt(ss / n + 1e-6f);
        for (int i = it.get_local_id(0); i < n; i += 512) xn[i] = x[i] * inv * nw[i];
    }));
    return xn;
}

// y[r] = W[r] . x (+ res[r]); one sub-group per row, x read from global (L1/L2 resident)
static void gemv(sycl::queue & q, const Mat & m, const float * x, float * y, const float * res = nullptr,
                 const char * name = "gemv") {
    const int nrows = m.rows, cols = m.cols;
    const Fmt f = m.f;
    const uint8_t * W = m.w;
    const size_t rb = m.rb;
    constexpr int WG = 128;
    const size_t g = (size_t) ((nrows + WG / SG - 1) / (WG / SG)) * WG;
    P(name, q.parallel_for(sycl::nd_range<1>(g, WG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        const auto sg = it.get_sub_group();
        const int r = it.get_global_id(0) / SG;
        if (r >= nrows) return;
        const int lane = sg.get_local_id()[0];
        const uint8_t * row = W + (size_t) r * rb;
        float acc = 0.f;
        if (f == F_F32) {
            const sycl::float4 * wr = (const sycl::float4 *) row;
            const sycl::float4 * xv = (const sycl::float4 *) x;
            for (int i = lane; i < cols / 4; i += SG) {
                const sycl::float4 a = wr[i], b = xv[i];
                acc += a.x() * b.x() + a.y() * b.y() + a.z() * b.z() + a.w() * b.w();
            }
        } else {
            const int nb = cols / 32;
            for (int b = lane; b < nb; b += SG) acc += row_blk_dot(f, row, cols, b, x);
        }
        acc = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
        if (lane == 0) y[r] = acc + (res ? res[r] : 0.f);
    }));
}


// ================= CPU expert path (AVX2) =================
// expert layout in host RAM is identical to the GPU arena: 2*FF_EXP gate/up rows of D cols, then D down rows of FF_EXP cols,
// each row = nibbles (cols/2 bytes, element 2k in the low nibble of byte k) followed by fp16 scales (one per 32)
static inline float hsum256(__m256 v) {
    __m128 a = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    a = _mm_add_ps(a, _mm_movehl_ps(a, a));
    a = _mm_add_ss(a, _mm_movehdup_ps(a));
    return _mm_cvtss_f32(a);
}
// x quantized per 32 to int8 with one float scale
static void quant_x(const float * x, int n, int8_t * xq, float * xs) {
    const __m256 sign = _mm256_set1_ps(-0.f);
    for (int b = 0; b < n / 32; ++b) {
        __m256 v[4];
        __m256 mx = _mm256_setzero_ps();
        for (int i = 0; i < 4; ++i) { v[i] = _mm256_loadu_ps(x + b * 32 + i * 8); mx = _mm256_max_ps(mx, _mm256_andnot_ps(sign, v[i])); }
        const float amax = hsum256(_mm256_setzero_ps()) + [&] { __m128 m = _mm_max_ps(_mm256_castps256_ps128(mx), _mm256_extractf128_ps(mx, 1));
            m = _mm_max_ps(m, _mm_movehl_ps(m, m)); m = _mm_max_ss(m, _mm_movehdup_ps(m)); return _mm_cvtss_f32(m); }();
        const float d = amax / 127.f, id = d > 0 ? 1.f / d : 0.f;
        xs[b] = d;
        const __m256 mul = _mm256_set1_ps(id);
        __m256i i32[4];
        for (int i = 0; i < 4; ++i) i32[i] = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(v[i], mul), _MM_ROUND_NEAREST | _MM_FROUND_NO_EXC));
        __m256i p01 = _mm256_packs_epi32(i32[0], i32[1]), p23 = _mm256_packs_epi32(i32[2], i32[3]);
        __m256i p = _mm256_packs_epi16(p01, p23);
        p = _mm256_permutevar8x32_epi32(p, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
        _mm256_storeu_si256((__m256i *) (xq + b * 32), p);
    }
}
static inline float cpu_row_dot_q4(const uint8_t * row, int cols, const int8_t * xq, const float * xs) {
    const int nb = cols / 32;
    const uint16_t * sc = (const uint16_t *) (row + cols / 2);
    const __m128i m4 = _mm_set1_epi8(0x0f);
    const __m256i off = _mm256_set1_epi8(8);
    const __m256i ones = _mm256_set1_epi16(1);
    __m256 acc = _mm256_setzero_ps();
    for (int b = 0; b < nb; ++b) {
        const __m128i raw = _mm_loadu_si128((const __m128i *) (row + b * 16));
        const __m128i lo = _mm_and_si128(raw, m4), hi = _mm_and_si128(_mm_srli_epi16(raw, 4), m4);
        __m256i w = _mm256_set_m128i(_mm_unpackhi_epi8(lo, hi), _mm_unpacklo_epi8(lo, hi));
        w = _mm256_sub_epi8(w, off);
        const __m256i xv = _mm256_loadu_si256((const __m256i *) (xq + b * 32));
        const __m256i p = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_sign_epi8(w, w), _mm256_sign_epi8(xv, w)), ones);
        const float d = _cvtsh_ss(sc[b]) * xs[b];
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(p), acc);
    }
    return hsum256(acc);
}

struct CpuPool {
    int nt;
    std::vector<std::thread> th;
    std::atomic<int> gen{0}, done{0};
    std::atomic<bool> stop{false};
    // job
    int n_miss = 0;
    const uint8_t * exp_ptr[K_USED];
    float exp_w[K_USED];
    alignas(64) int8_t xq[D];
    float xs[D / 32];
    alignas(64) float hbuf[K_USED * FF_EXP];
    alignas(64) int8_t hq[K_USED * FF_EXP];
    float hs[K_USED * FF_EXP / 32];
    float * out = nullptr;   // D floats (host USM, read by the GPU)
    std::atomic<int> phase_done{0};

    void work(int tid, int phase) {
        if (phase == 0) {          // gate+up rows of every missed expert
            const int total = n_miss * FF_EXP;
            const int per = (total + nt - 1) / nt;
            for (int r = tid * per; r < std::min(total, (tid + 1) * per); ++r) {
                const int e = r / FF_EXP, rr = r % FF_EXP;
                const uint8_t * base = exp_ptr[e];
                const size_t rb = row_bytes(F_Q4, D);
                const float g = cpu_row_dot_q4(base + (size_t) rr * rb, D, xq, xs);
                const float u = cpu_row_dot_q4(base + (size_t) (FF_EXP + rr) * rb, D, xq, xs);
                hbuf[e * FF_EXP + rr] = g / (1.f + std::exp(-g)) * u;
            }
        } else {                   // down rows, weighted sum over missed experts
            const int per = (D + nt - 1) / nt;
            for (int r = tid * per; r < std::min(D, (tid + 1) * per); ++r) {
                float acc = 0.f;
                for (int e = 0; e < n_miss; ++e) {
                    const uint8_t * row = exp_ptr[e] + (size_t) 2 * FF_EXP * row_bytes(F_Q4, D) + (size_t) r * row_bytes(F_Q4, FF_EXP);
                    acc += exp_w[e] * cpu_row_dot_q4(row, FF_EXP, hq + e * FF_EXP, hs + e * (FF_EXP / 32));
                }
                out[r] = acc;
            }
        }
    }
    void start(int n) {
        nt = n;
        for (int t = 1; t < nt; ++t) {
            th.emplace_back([this, t]() {
                cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(t, &cs);
                pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs);
                int seen = 0;
                while (!stop.load(std::memory_order_relaxed)) {
                    const int g = gen.load(std::memory_order_acquire);
                    if (g == seen) { _mm_pause(); continue; }
                    seen = g;
                    work(t, g & 1 ? 0 : 1);
                    done.fetch_add(1, std::memory_order_acq_rel);
                }
            });
        }
        cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(0, &cs);
        pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs);
    }
    // run one phase on all threads (main thread takes tid 0); gen odd = phase 0, even = phase 1
    void run_phase(int phase) {
        done.store(0, std::memory_order_relaxed);
        const int g = gen.load() + 1;
        // keep parity = phase
        const int gg = ((g & 1) == (phase == 0 ? 1 : 0)) ? g : g + 1;
        gen.store(gg, std::memory_order_release);
        work(0, phase);
        while (done.load(std::memory_order_acquire) < nt - 1) _mm_pause();
    }
    // full expert computation for the current job
    void compute(const float * xf) {
        quant_x(xf, D, xq, xs);
        run_phase(0);
        for (int e = 0; e < n_miss; ++e) quant_x(hbuf + e * FF_EXP, FF_EXP, hq + e * FF_EXP, hs + e * (FF_EXP / 32));
        run_phase(1);
    }
    ~CpuPool() { stop = true; for (auto & t : th) t.join(); }
};

static inline bool is_miss(int layer, int id, int tokn, int pct) {
    uint32_t h = (uint32_t) (layer * 7919 + id * 104729 + tokn * 1299709);
    h ^= h >> 13; h *= 0x5bd1e995; h ^= h >> 15;
    return (int) (h % 100) < pct;
}

int main(int argc, char ** argv) {
    const int ctx_len = argc > 1 ? atoi(argv[1]) : 4096;
    const std::string emode = argc > 2 ? argv[2] : "all";
    const int n_tok = argc > 3 ? atoi(argv[3]) : 100;
    const int arena_n = 2000;
    const size_t EXP_GU_RB = row_bytes(F_Q4, D), EXP_D_RB = row_bytes(F_Q4, FF_EXP);
    const size_t EXP_BYTES = 2 * FF_EXP * EXP_GU_RB + D * EXP_D_RB;

    sycl::queue q{sycl::gpu_selector_v, sycl::property_list{sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{}}};
    printf("device: %s, ctx %d, experts %s\n", q.get_device().get_info<sycl::info::device::name>().c_str(), ctx_len, emode.c_str());

    size_t wbytes = 0;
    std::vector<Layer> L(N_LAYER);
    for (int il = 0; il < N_LAYER; ++il) {
        Layer & l = L[il];
        l.recr = (il + 1) % 4 != 0;
        l.attn_norm = dev_fill(q, D, 1.f);
        l.post_norm = dev_fill(q, D, 1.f);
        if (l.recr) {
            l.qkvz = make_mat(q, F_Q6, QKV_DIM + Z_DIM, D, wbytes);
            l.ab = make_mat(q, F_F32, 2 * NV, D, wbytes);
            l.out = make_mat(q, F_Q6, D, Z_DIM, wbytes);
            l.conv_w = dev_fill(q, (size_t) QKV_DIM * CONV_K, 0.25f);
            l.conv_state = dev_fill(q, (size_t) QKV_DIM * (CONV_K - 1), 0.f);
            l.ssm_state = dev_fill(q, (size_t) NV * DK * DK, 0.f);
            l.ssm_norm = dev_fill(q, DK, 1.f);
            l.a_log = dev_fill(q, NV, -1.f);
            l.dt_bias = dev_fill(q, NV, 0.f);
        } else {
            l.qkv = make_mat(q, F_Q8, Q_ROWS + 2 * HD * NKV, D, wbytes);
            l.wo = make_mat(q, F_Q8, D, HD * NQ, wbytes);
            l.qn = dev_fill(q, HD, 1.f);
            l.kn = dev_fill(q, HD, 1.f);
            const size_t kv = (size_t) ctx_len * NKV * HD;
            l.kcache = sycl::malloc_device<sycl::half>(kv, q);
            l.vcache = sycl::malloc_device<sycl::half>(kv, q);
            q.fill(l.kcache, sycl::half(0.01f), kv);
            q.fill(l.vcache, sycl::half(0.01f), kv).wait();
        }
        l.router_sh = make_mat(q, F_F32, N_EXPERT + 1, D, wbytes);
        l.sh_gu = make_mat(q, F_Q8, 2 * FF_SH, D, wbytes);
        l.sh_down = make_mat(q, F_Q8, D, FF_SH, wbytes);
    }
    Mat lm = make_mat(q, F_Q6, VOCAB, D, wbytes);
    float * out_norm = dev_fill(q, D, 1.f);
    uint8_t * arena = dev_random(q, arena_n * EXP_BYTES, true);
    {
        // per expert: 2*FF_EXP gate/up rows (D cols), then D down rows (FF_EXP cols); set every fp16 scale
        const size_t n_gu = (size_t) arena_n * 2 * FF_EXP * (D / 32), n_dn = (size_t) arena_n * D * (FF_EXP / 32);
        q.parallel_for(sycl::range<1>(n_gu), [=](sycl::id<1> i) {
            const size_t row = i[0] / (D / 32), b = i[0] % (D / 32);
            const size_t e = row / (2 * FF_EXP), r = row % (2 * FF_EXP);
            ((sycl::half *) (arena + e * EXP_BYTES + r * EXP_GU_RB + D / 2))[b] = sycl::half(0.0005f);
        });
        q.parallel_for(sycl::range<1>(n_dn), [=](sycl::id<1> i) {
            const size_t row = i[0] / (FF_EXP / 32), b = i[0] % (FF_EXP / 32);
            const size_t e = row / D, r = row % D;
            ((sycl::half *) (arena + e * EXP_BYTES + 2 * FF_EXP * EXP_GU_RB + r * EXP_D_RB + FF_EXP / 2))[b] = sycl::half(0.0005f);
        }).wait();
    }
    printf("dense weights read per token: %.3f GB (+ KV cache %.1f MB at this ctx)\n", wbytes / 1e9,
           10.0 * ctx_len * NKV * HD * 2 * 2 / 1e6);

    // activations
    g_xn = dev_fill(q, D, 0.f);
    float * x = dev_fill(q, D, 0.1f);        // residual stream
    float * x2 = dev_fill(q, D, 0.f);
    float * qkvz = dev_fill(q, QKV_DIM + Z_DIM, 0.f);
    float * ab = dev_fill(q, 2 * NV, 0.f);
    float * attn_o = dev_fill(q, Z_DIM, 0.f);
    float * qkv_a = dev_fill(q, Q_ROWS + 2 * HD * NKV, 0.f);
    float * part = dev_fill(q, (size_t) NQ * (ctx_len / ATT_CHUNK + 1) * (HD + 2), 0.f);
    float * att = dev_fill(q, HD * NQ, 0.f);
    float * rlog = dev_fill(q, N_EXPERT + 1, 0.f);
    float * shgu = dev_fill(q, 2 * FF_SH, 0.f);
    float * h = dev_fill(q, K_USED * FF_EXP + FF_SH, 0.f);
    int32_t * ids = sycl::malloc_device<int32_t>(K_USED, q);
    float * wts = dev_fill(q, K_USED + 1, 0.f);
    float * logits = dev_fill(q, VOCAB, 0.f);
    int32_t * tok = sycl::malloc_device<int32_t>(1, q);
    int32_t * pos_d = sycl::malloc_device<int32_t>(1, q);
    const int pos = ctx_len - 1;
    q.memcpy(pos_d, &pos, 4).wait();
    const bool hybrid = emode == "hybrid";
    const bool experts_on = emode == "all" || hybrid;
    const int miss_pct = argc > 4 ? atoi(argv[4]) : 15;
    struct HostIO { int32_t ids[K_USED]; float wts[K_USED + 1]; float xf[D]; float cpu_out[D]; };
    HostIO * hio = sycl::malloc_host<HostIO>(1, q);
    memset(hio, 0, sizeof(HostIO));
    int tokn = 0;
    // host RAM expert arena (same layout as the GPU arena)
    const int cpu_n = 3000;
    uint8_t * harena = nullptr;
    CpuPool pool;
    double cpu_us_sum = 0.0; long cpu_calls = 0, miss_total = 0;
    if (hybrid) {
        harena = (uint8_t *) std::aligned_alloc(1 << 21, ((size_t) cpu_n * EXP_BYTES + (1 << 21)) & ~(size_t) ((1 << 21) - 1));
        madvise(harena, (size_t) cpu_n * EXP_BYTES, MADV_HUGEPAGE);
        uint8_t * dtmp = sycl::malloc_device<uint8_t>(EXP_BYTES * 16, q);
        // fill host arena from a device-side copy of 16 GPU experts (already random with fixed scales)
        q.memcpy(dtmp, arena, EXP_BYTES * 16).wait();
        for (int e = 0; e < cpu_n; e += 16) q.memcpy(harena + (size_t) e * EXP_BYTES, dtmp, EXP_BYTES * std::min(16, cpu_n - e)).wait();
        sycl::free(dtmp, q);
        pool.start(16);
        pool.out = hio->cpu_out;
        // raw RAM read bandwidth with 16 threads over the arena (ceiling for the CPU path)
        {
            const size_t bytes = (size_t) cpu_n * EXP_BYTES;
            for (int rep = 0; rep < 2; ++rep) {
                std::vector<std::thread> ts;
                std::atomic<uint64_t> sink{0};
                auto c0 = std::chrono::high_resolution_clock::now();
                for (int t = 0; t < 16; ++t) ts.emplace_back([&, t] {
                    const size_t per = bytes / 16;
                    const __m256i * p = (const __m256i *) (harena + t * per);
                    __m256i acc = _mm256_setzero_si256();
                    for (size_t i = 0; i < per / 32; i += 2) acc = _mm256_xor_si256(acc, _mm256_xor_si256(_mm256_load_si256(p + i), _mm256_load_si256(p + i + 1)));
                    sink += (uint64_t) _mm256_extract_epi64(acc, 0);
                });
                for (auto & t : ts) t.join();
                const double sec = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - c0).count();
                if (rep) printf("CPU RAM read bandwidth (16 threads): %.1f GB/s\n", bytes / sec / 1e9);
            }
        }
        // standalone CPU timing: n missed experts per layer
        std::vector<float> xt(D, 0.05f);
        for (int nm = 1; nm <= 4; ++nm) {
            pool.n_miss = nm;
            for (int e = 0; e < nm; ++e) { pool.exp_ptr[e] = harena + (size_t) ((e * 977) % cpu_n) * EXP_BYTES; pool.exp_w[e] = 0.1f; }
            pool.compute(xt.data());
            const int reps = 200;
            auto c0 = std::chrono::high_resolution_clock::now();
            for (int r = 0; r < reps; ++r) {
                for (int e = 0; e < nm; ++e) pool.exp_ptr[e] = harena + (size_t) (((r * 31 + e) * 977) % cpu_n) * EXP_BYTES;
                pool.compute(xt.data());
            }
            const double us = std::chrono::duration<double, std::micro>(std::chrono::high_resolution_clock::now() - c0).count() / reps;
            printf("CPU: %d missed expert(s) in one layer: %.1f us (%.1f GB/s from RAM)\n", nm, us, nm * EXP_BYTES / us / 1e3);
        }
    }

    auto token = [&]() {
        for (int il = 0; il < N_LAYER; ++il) {
            Layer & l = L[il];
            if (l.recr) {
                // 1. qkv + z projections, fused rmsnorm
                { const float * xn = norm(q, x, l.attn_norm);
                  gemv(q, l.qkvz, xn, qkvz, nullptr, "dn_qkvz");
                  gemv(q, l.ab, xn, ab, nullptr, "dn_ab"); }
                // 2. conv1d (state update) + silu, l2 norm of q and k heads, gate/beta
                {
                    float * cs = l.conv_state, * cw = l.conv_w;
                    P("dn_conv", q.parallel_for(sycl::nd_range<1>(QKV_DIM, 128), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                        const int c = it.get_global_id(0);
                        const float in = qkvz[c];
                        float acc = cw[c * CONV_K + CONV_K - 1] * in;
                        for (int k = 0; k < CONV_K - 1; ++k) acc += cw[c * CONV_K + k] * cs[c * (CONV_K - 1) + k];
                        for (int k = 0; k < CONV_K - 2; ++k) cs[c * (CONV_K - 1) + k] = cs[c * (CONV_K - 1) + k + 1];
                        cs[c * (CONV_K - 1) + CONV_K - 2] = in;
                        float v = acc / (1.f + sycl::exp(-acc));
                        // l2 norm for q/k heads (128 lanes of the work-group = one head)
                        if (c < 2 * DK * NK) {
                            const float ss = sycl::reduce_over_group(it.get_group(), v * v, sycl::plus<float>());
                            v *= sycl::rsqrt(ss + 1e-6f);
                        }
                        qkvz[c] = v;
                    }));
                }
                // 3. gated delta rule, one work-group per v head; thread (col j = lid % 128, part p = lid / 128) holds S[p*32 .. p*32+31][j]
                {
                    float * S = l.ssm_state, * nw = l.ssm_norm, * al = l.a_log, * dtb = l.dt_bias;
                    P("dn_delta", q.submit([&](sycl::handler & hd) {
                        sycl::local_accessor<float, 1> red(4 * DK, hd);
                        hd.parallel_for(sycl::nd_range<1>(NV * 512, 512), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                            const int hv = it.get_group(0), lid = it.get_local_id(0);
                            const int j = lid % DK, p = lid / DK;
                            const int hk = hv / (NV / NK);
                            const float * qv = qkvz + hk * DK;
                            const float * kv = qkvz + DK * NK + hk * DK;
                            const float vj = qkvz[2 * DK * NK + hv * DK + j];
                            const float a = ab[hv] + dtb[hv];
                            const float g = sycl::exp(-sycl::exp(al[hv]) * (a > 20.f ? a : sycl::log1p(sycl::exp(a))));
                            const float beta = 1.f / (1.f + sycl::exp(-ab[NV + hv]));
                            float * Sh = S + (size_t) hv * DK * DK;
                            float s[32];
                            float part = 0.f;
#pragma unroll
                            for (int i = 0; i < 32; ++i) {
                                s[i] = Sh[(size_t) (p * 32 + i) * DK + j] * g;
                                part += s[i] * kv[p * 32 + i];
                            }
                            red[p * DK + j] = part;
                            it.barrier(sycl::access::fence_space::local_space);
                            const float kvj = red[j] + red[DK + j] + red[2 * DK + j] + red[3 * DK + j];
                            it.barrier(sycl::access::fence_space::local_space);
                            const float delta = beta * (vj - kvj);
                            part = 0.f;
#pragma unroll
                            for (int i = 0; i < 32; ++i) {
                                s[i] += kv[p * 32 + i] * delta;
                                part += s[i] * qv[p * 32 + i];
                                Sh[(size_t) (p * 32 + i) * DK + j] = s[i];
                            }
                            red[p * DK + j] = part;
                            it.barrier(sycl::access::fence_space::local_space);
                            const float oj = (red[j] + red[DK + j] + red[2 * DK + j] + red[3 * DK + j]) * sycl::rsqrt((float) DK);
                            const float ss = sycl::reduce_over_group(it.get_group(), p == 0 ? oj * oj : 0.f, sycl::plus<float>());
                            const float z = qkvz[QKV_DIM + hv * DK + j];
                            if (p == 0) attn_o[hv * DK + j] = oj * sycl::rsqrt(ss / DK + 1e-6f) * nw[j] * (z / (1.f + sycl::exp(-z)));
                        });
                    }));
                }
                // 4. output projection + residual
                gemv(q, l.out, attn_o, x2, x, "dn_out");
            } else {
                // full attention: q(+gate), k, v projections
                gemv(q, l.qkv, norm(q, x, l.attn_norm), qkv_a, nullptr, "fa_qkv");
                // q/k rms norm per head, partial rope on the first ROT dims, append k/v at pos
                {
                    float * qn = l.qn, * kn = l.kn;
                    sycl::half * kc = l.kcache, * vc = l.vcache;
                    P("fa_rope", q.parallel_for(sycl::nd_range<1>((NQ + NKV) * HD, HD), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                        const int hh = it.get_group(0), d = it.get_local_id(0);
                        const int ps = *pos_d;
                        const bool isq = hh < NQ;
                        float * v = isq ? qkv_a + hh * 2 * HD : qkv_a + Q_ROWS + (hh - NQ) * HD;
                        const float val = v[d];
                        const float ss = sycl::reduce_over_group(it.get_group(), val * val, sycl::plus<float>());
                        float y = val * sycl::rsqrt(ss / HD + 1e-6f) * (isq ? qn[d] : kn[d]);
                        it.barrier();
                        v[d] = y;
                        it.barrier();
                        if (d < ROT) {
                            const int pair = d < ROT / 2 ? d + ROT / 2 : d - ROT / 2;
                            const float th = ps * sycl::pow(10000000.f, -2.f * (d % (ROT / 2)) / ROT);
                            const float other = v[pair];
                            y = d < ROT / 2 ? y * sycl::cos(th) - other * sycl::sin(th) : y * sycl::cos(th) + other * sycl::sin(th);
                        }
                        it.barrier();
                        if (isq) {
                            v[d] = y;
                        } else {
                            const int kh = hh - NQ;
                            kc[((size_t) ps * NKV + kh) * HD + d] = sycl::half(y);
                            vc[((size_t) ps * NKV + kh) * HD + d] = sycl::half(qkv_a[Q_ROWS + NKV * HD + kh * HD + d]);
                        }
                    }));
                }
                // split-K decode attention: work-group (8 sub-groups) per (q head, chunk); sub-group s takes positions t0+s, t0+s+8, ...
                const int n_chunks = (ctx_len + ATT_CHUNK - 1) / ATT_CHUNK;
                {
                    sycl::half * kc = l.kcache, * vc = l.vcache;
                    const int clen = ctx_len;
                    P("fa_attn", q.submit([&](sycl::handler & hd) {
                        sycl::local_accessor<float, 1> sm(8 * (HD + 2), hd);
                        hd.parallel_for(sycl::nd_range<2>({(size_t) NQ, (size_t) n_chunks * 128}, {1, 128}),
                                        [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG)]] {
                            const int hq = it.get_global_id(0), ch = it.get_group(1);
                            const auto sg = it.get_sub_group();
                            const int s_id = sg.get_group_id()[0], lane = sg.get_local_id()[0];
                            const int kh = hq / (NQ / NKV);
                            float qd[16], acc[16];
                            for (int i = 0; i < 16; ++i) { qd[i] = qkv_a[hq * 2 * HD + lane * 16 + i] * 0.0625f; acc[i] = 0.f; }
                            float m = -1e30f, sum = 0.f;
                            const int t0 = ch * ATT_CHUNK, t1 = sycl::min(clen, t0 + ATT_CHUNK);
                            for (int t = t0 + s_id; t < t1; t += 8) {
                                const sycl::vec<sycl::half, 16> kk = *(const sycl::vec<sycl::half, 16> *) (kc + ((size_t) t * NKV + kh) * HD + lane * 16);
                                float d = 0.f;
#pragma unroll
                                for (int i = 0; i < 16; ++i) d += qd[i] * (float) kk[i];
                                d = sycl::reduce_over_group(sg, d, sycl::plus<float>());
                                const float mn = sycl::max(m, d);
                                const float c = sycl::exp(m - mn), e = sycl::exp(d - mn);
                                sum = sum * c + e;
                                const sycl::vec<sycl::half, 16> vv = *(const sycl::vec<sycl::half, 16> *) (vc + ((size_t) t * NKV + kh) * HD + lane * 16);
#pragma unroll
                                for (int i = 0; i < 16; ++i) acc[i] = acc[i] * c + e * (float) vv[i];
                                m = mn;
                            }
                            float * my = &sm[s_id * (HD + 2)];
                            for (int i = 0; i < 16; ++i) my[lane * 16 + i] = acc[i];
                            if (lane == 0) { my[HD] = m; my[HD + 1] = sum; }
                            it.barrier(sycl::access::fence_space::local_space);
                            const int lid = it.get_local_id(1);
                            float M = -1e30f;
                            for (int k = 0; k < 8; ++k) M = sycl::max(M, sm[k * (HD + 2) + HD]);
                            float * pp = part + ((size_t) hq * n_chunks + ch) * (HD + 2);
                            for (int dd = lid; dd < HD; dd += 128) {
                                float a = 0.f;
                                for (int k = 0; k < 8; ++k) a += sm[k * (HD + 2) + dd] * sycl::exp(sm[k * (HD + 2) + HD] - M);
                                pp[dd] = a;
                            }
                            if (lid == 0) {
                                float S = 0.f;
                                for (int k = 0; k < 8; ++k) S += sm[k * (HD + 2) + HD + 1] * sycl::exp(sm[k * (HD + 2) + HD] - M);
                                pp[HD] = M; pp[HD + 1] = S;
                            }
                        });
                    }));
                }
                // combine chunks, apply sigmoid output gate
                P("fa_combine", q.parallel_for(sycl::nd_range<1>(NQ * HD, HD), [=](sycl::nd_item<1> it) {
                    const int hq = it.get_group(0), d = it.get_local_id(0);
                    float m = -1e30f;
                    for (int c = 0; c < n_chunks; ++c) m = sycl::max(m, part[((size_t) hq * n_chunks + c) * (HD + 2) + HD]);
                    float s = 0.f, a = 0.f;
                    for (int c = 0; c < n_chunks; ++c) {
                        const float * pp = part + ((size_t) hq * n_chunks + c) * (HD + 2);
                        const float w = sycl::exp(pp[HD] - m);
                        s += pp[HD + 1] * w;
                        a += pp[d] * w;
                    }
                    const float gt = qkv_a[hq * 2 * HD + HD + d];
                    att[hq * HD + d] = a / s / (1.f + sycl::exp(-gt));
                }));
                gemv(q, l.wo, att, x2, x, "fa_wo");
            }
            // ---- MoE ffn on x2 ----
            const float * xf = norm(q, x2, l.post_norm);
            gemv(q, l.router_sh, xf, rlog, nullptr, "router");
            gemv(q, l.sh_gu, xf, shgu, nullptr, "shexp_gu");
            // top-8 softmax over 256 router logits (one sub-group), shared gate sigmoid
            const bool hyb = hybrid;
            sycl::event e_topk = P2("topk", q.parallel_for(sycl::nd_range<1>(SG, SG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                const auto sg = it.get_sub_group();
                const int lane = sg.get_local_id()[0];
                float v[N_EXPERT / SG];
                for (int i = 0; i < N_EXPERT / SG; ++i) v[i] = rlog[i * SG + lane];
                float sel[K_USED];
                for (int k = 0; k < K_USED; ++k) {
                    float best = -1e30f; int bi = 0;
                    for (int i = 0; i < N_EXPERT / SG; ++i) if (v[i] > best) { best = v[i]; bi = i; }
                    const float mx = sycl::reduce_over_group(sg, best, sycl::maximum<float>());
                    const int who = sycl::reduce_over_group(sg, best == mx ? lane : SG, sycl::minimum<int>());
                    const int bidx = sycl::group_broadcast(sg, bi, who);
                    if (lane == who) v[bi] = -1e30f;
                    sel[k] = mx;
                    if (lane == 0) ids[k] = bidx * SG + who;
                }
                if (lane == 0) {
                    float s = 0.f;
                    for (int k = 0; k < K_USED; ++k) s += sycl::exp(sel[k] - sel[0]);
                    for (int k = 0; k < K_USED; ++k) wts[k] = sycl::exp(sel[k] - sel[0]) / s;
                    wts[K_USED] = 1.f / (1.f + sycl::exp(-rlog[N_EXPERT]));
                    if (hyb) { for (int k = 0; k < K_USED; ++k) { hio->ids[k] = ids[k]; hio->wts[k] = wts[k]; } }
                }
                if (hyb) for (int i = lane; i < D; i += SG) hio->xf[i] = xf[i];
            }));
            // routed gate/up (+ shared expert activation in the same launch)
            {
                const int nrow = (experts_on ? K_USED * FF_EXP : 0) + FF_SH;
                const int layer = il;
                const int tk = tokn, mp = miss_pct;
                P("moe_gate_up", q.parallel_for(sycl::nd_range<1>((size_t) nrow * SG, 128), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                    const auto sg = it.get_sub_group();
                    const int r = it.get_global_id(0) / SG, lane = sg.get_local_id()[0];
                    if (r >= nrow - FF_SH) {
                        if (lane == 0) {
                            const int rr = r - (nrow - FF_SH);
                            const float g = shgu[rr], u = shgu[FF_SH + rr];
                            h[K_USED * FF_EXP + rr] = g / (1.f + sycl::exp(-g)) * u;
                        }
                        return;
                    }
                    const int e = r / FF_EXP, rr = r % FF_EXP;
                    if (hyb && is_miss(layer, ids[e], tk, mp)) return;
                    const uint8_t * base = arena + (size_t) ((layer * N_EXPERT + ids[e]) % arena_n) * EXP_BYTES;
                    float acc[2];
                    for (int m = 0; m < 2; ++m) {
                        const uint8_t * row = base + ((size_t) m * FF_EXP + rr) * EXP_GU_RB;
                        float a = 0.f;
                        for (int b = lane; b < D / 32; b += SG) a += row_blk_dot(F_Q4, row, D, b, xf);
                        acc[m] = sycl::reduce_over_group(sg, a, sycl::plus<float>());
                    }
                    if (lane == 0) h[e * FF_EXP + rr] = acc[0] / (1.f + sycl::exp(-acc[0])) * acc[1];
                }));
            }
            // hybrid: the host reads the routing, computes missed experts on the CPU while the GPU runs the hits
            bool use_cpu = false;
            if (hybrid) {
                e_topk.wait();
                int nm = 0;
                for (int k = 0; k < K_USED; ++k) {
                    if (is_miss(il, hio->ids[k], tokn, miss_pct)) {
                        pool.exp_ptr[nm] = harena + (size_t) ((il * N_EXPERT + hio->ids[k]) % cpu_n) * EXP_BYTES;
                        pool.exp_w[nm] = hio->wts[k];
                        ++nm;
                    }
                }
                if (nm) {
                    auto c0 = std::chrono::high_resolution_clock::now();
                    pool.n_miss = nm;
                    pool.compute(hio->xf);
                    cpu_us_sum += std::chrono::duration<double, std::micro>(std::chrono::high_resolution_clock::now() - c0).count();
                    ++cpu_calls; miss_total += nm;
                    use_cpu = true;
                }
            }
            // routed down + shared down, weighted sum, residual -> x
            {
                const bool uc = use_cpu;
                const int tk = tokn, mp = miss_pct;
                const float * cpo = hio->cpu_out;
                const int layer = il;
                const uint8_t * shd = l.sh_down.w;
                const size_t shrb = l.sh_down.rb;
                const int ne = experts_on ? K_USED : 0;
                P("moe_down", q.parallel_for(sycl::nd_range<1>((size_t) D * SG, 128), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                    const auto sg = it.get_sub_group();
                    const int r = it.get_global_id(0) / SG, lane = sg.get_local_id()[0];
                    float acc = 0.f;
                    for (int e = 0; e < ne; ++e) {
                        if (hyb && is_miss(layer, ids[e], tk, mp)) continue;
                        const uint8_t * row = arena + (size_t) ((layer * N_EXPERT + ids[e]) % arena_n) * EXP_BYTES +
                                              (size_t) 2 * FF_EXP * EXP_GU_RB + (size_t) r * EXP_D_RB;
                        acc += wts[e] * row_blk_dot(F_Q4, row, FF_EXP, lane, h + e * FF_EXP);
                    }
                    acc += wts[K_USED] * row_blk_dot(F_Q8, shd + (size_t) r * shrb, FF_SH, lane, h + K_USED * FF_EXP);
                    acc = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
                    if (lane == 0) x[r] = x2[r] + acc + (uc ? cpo[r] : 0.f);
                }));
            }
        }
        ++tokn;
        // final norm + lm head + argmax
        gemv(q, lm, norm(q, x, out_norm), logits, nullptr, "lm_head");
        P("argmax", q.parallel_for(sycl::nd_range<1>(1024, 1024), [=](sycl::nd_item<1> it) {
            float best = -1e30f; int bi = 0;
            for (int i = it.get_local_id(0); i < VOCAB; i += 1024) if (logits[i] > best) { best = logits[i]; bi = i; }
            const float mx = sycl::reduce_over_group(it.get_group(), best, sycl::maximum<float>());
            const int who = sycl::reduce_over_group(it.get_group(), best == mx ? bi : VOCAB, sycl::minimum<int>());
            if (it.get_local_id(0) == 0) tok[0] = who;
        }));
    };

    // per-kernel profile of one token
    token(); q.wait();
    g_prof = true; token(); q.wait(); g_prof = false;
    {
        std::vector<std::pair<std::string, std::pair<double, int>>> agg;
        double tot = 0.0;
        for (auto & [name, ev] : g_ev) {
            const double us = (ev.get_profiling_info<sycl::info::event_profiling::command_end>() -
                               ev.get_profiling_info<sycl::info::event_profiling::command_start>()) / 1e3;
            tot += us;
            bool found = false;
            for (auto & a : agg) if (a.first == name) { a.second.first += us; a.second.second++; found = true; }
            if (!found) agg.push_back({name, {us, 1}});
        }
        printf("kernel time per token: %.3f ms over %zu launches\n", tot / 1e3, g_ev.size());
        for (auto & a : agg) printf("  %-12s %4d x  %8.1f us total  %6.1f us avg\n", a.first.c_str(), a.second.second, a.second.first, a.second.first / a.second.second);
    }
    // eager submission
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_tok; ++i) token();
    q.wait();
    double ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count() / n_tok;
    printf("eager:       %.3f ms/token = %.1f tok/s\n", ms, 1000.0 / ms);

    // host-side submit cost alone (no wait between tokens is already measured; measure enqueue only)
    t0 = std::chrono::high_resolution_clock::now();
    token();
    const double enq = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();
    q.wait();
    printf("enqueue only: %.3f ms/token (host cost of submitting all kernels)\n", enq);

    if (hybrid) {
        printf("hybrid: miss rate %d%%, %.1f missed experts per token, CPU %.1f us per layer with misses (%.2f ms per token)\n",
               miss_pct, (double) miss_total / tokn, cpu_us_sum / std::max(1L, cpu_calls), cpu_us_sum / tokn / 1e3);
        return 0;
    }
    // SYCL graph: record one token, replay
    try {
        syclex::command_graph g{q.get_context(), q.get_device()};
        g.begin_recording(q);
        token();
        g.end_recording();
        auto ge = g.finalize();
        q.ext_oneapi_graph(ge); q.wait();
        t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < n_tok; ++i) q.ext_oneapi_graph(ge);
        q.wait();
        ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count() / n_tok;
        printf("sycl graph:  %.3f ms/token = %.1f tok/s\n", ms, 1000.0 / ms);
    } catch (const sycl::exception & e) {
        printf("sycl graph failed: %s\n", e.what());
    }
    int tk = 0;
    q.memcpy(&tk, tok, 4).wait();
    printf("(argmax token %d)\n", tk);
    return 0;
}

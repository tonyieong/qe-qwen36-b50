// qe: Qwen3.6-35B-A3B inference engine for an Intel Arc GPU (SYCL) with hot experts in VRAM and cold experts on the CPU (AVX2)
// modes:
//   qe check <pack> <ref.bin> [opts]            compare logits with a llama.cpp reference dump (ref_dump)
//   qe gen   <pack> <vocab.gguf> <prompt> [opts] greedy generation with timing
// opts: --resident N (experts kept in VRAM, default 7000)  --ctx N (default 32768)  --stats a.bin,b.bin (routing counts)
#include <sycl/sycl.hpp>
#include <oneapi/mkl/blas.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include "pack.h"
#include "llama.h"
#include "httplib.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "json.hpp"
#include <mutex>
#include <random>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <immintrin.h>
#include <map>
#include <memory>
#include <deque>
#include <condition_variable>
#include <fstream>
#include <functional>
#include <pthread.h>
#include <sstream>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

using namespace qe;
constexpr int SG = 16;
constexpr int ATT_CHUNK = 256;

static double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// ============================ pack file ============================
struct Pack {
    uint8_t * base = nullptr;
    size_t size = 0;
    std::map<std::string, PackEntry> e;
    explicit Pack(const char * path) {
        const int fd = open(path, O_RDONLY);
        if (fd < 0) { perror(path); exit(1); }
        struct stat st; fstat(fd, &st); size = st.st_size;
        base = (uint8_t *) mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
        close(fd);
        const PackHeader * h = (const PackHeader *) base;
        if (memcmp(h->magic, "QE36PK01", 8) != 0) { fprintf(stderr, "bad pack\n"); exit(1); }
        const PackEntry * t = (const PackEntry *) (base + h->table_off);
        for (uint32_t i = 0; i < h->n_entries; ++i) e[t[i].name] = t[i];
    }
    const PackEntry & get(const std::string & n) const {
        auto it = e.find(n);
        if (it == e.end()) { fprintf(stderr, "pack: missing %s\n", n.c_str()); exit(1); }
        return it->second;
    }
    const uint8_t * ptr(const std::string & n) const { return base + get(n).off; }
};

// ============================ device helpers ============================
struct DMat { uint8_t * w = nullptr; Fmt f = F_F32; int rows = 0, cols = 0; size_t rb = 0; };

// IQ4_NL value for a 4-bit index without a memory table: the 16 int8 values packed in two 64-bit constants
static inline float nl4_val(uint32_t n) {
    constexpr uint64_t LO = 0xF6EADDCFBFAD9881ull;   // -127 -104 -83 -65 -49 -35 -22 -10
    constexpr uint64_t HI = 0x7159453526190D01ull;   //    1   13  25  38  53  69  89 113
    const uint64_t w = (n & 8) ? HI : LO;
    return (float) (int8_t) ((w >> ((n & 7) * 8)) & 0xff);
}

// dot of 32-weight block b of a row with x[b*32 ..]; 16/8-byte vector loads
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
    // NL4: byte j -> elements j (low nibble) and j+16 (high nibble)
    const sycl::uint4 q = *(const sycl::uint4 *) (row + (size_t) b * 16);
    const uint32_t w[4] = {q.x(), q.y(), q.z(), q.w()};
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint32_t by = (w[i] >> (8 * j)) & 0xff;
            acc += nl4_val(by & 0xf) * xs[i * 4 + j] + nl4_val(by >> 4) * xs[16 + i * 4 + j];
        }
    return acc * (float) ((const sycl::half *) (row + (size_t) cols / 2))[b];
}

// y[r] = W[r] . x (+ res[r])
static void gemv(sycl::queue & q, const DMat & m, const float * x, float * y, const float * res = nullptr) {
    const int nrows = m.rows, cols = m.cols;
    const Fmt f = m.f;
    const uint8_t * W = m.w;
    const size_t rb = m.rb;
    constexpr int WG = 128;
    const size_t g = (size_t) ((nrows + WG / SG - 1) / (WG / SG)) * WG;
    q.parallel_for(sycl::nd_range<1>(g, WG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
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
            for (int b = lane; b < cols / 32; b += SG) acc += row_blk_dot(f, row, cols, b, x);
        }
        acc = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
        if (lane == 0) y[r] = acc + (res ? res[r] : 0.f);
    });
}

// xn = rmsnorm(x) * w
static void rmsnorm(sycl::queue & q, const float * x, const float * w, float * xn, int n = D) {
    q.parallel_for(sycl::nd_range<1>(512, 512), [=](sycl::nd_item<1> it) {
        float ss = 0.f;
        for (int i = it.get_local_id(0); i < n; i += 512) ss += x[i] * x[i];
        ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
        const float inv = sycl::rsqrt(ss / n + RMS_EPS);
        for (int i = it.get_local_id(0); i < n; i += 512) xn[i] = x[i] * inv * w[i];
    });
}

// ============================ batched (prefill) kernels ============================
constexpr int T_MAX = 1024;  // tokens per prefill chunk
constexpr int TG = 8;        // tokens per register tile

// dequantize block b of a quantized row into w[32] (scale applied)
static inline void deq_block(Fmt f, const uint8_t * row, int cols, int b, float * w) {
    if (f == F_Q8) {
        const sycl::uint4 * qv = (const sycl::uint4 *) (row + (size_t) b * 32);
        const sycl::uint4 a = qv[0], c = qv[1];
        const uint32_t u[8] = {a.x(), a.y(), a.z(), a.w(), c.x(), c.y(), c.z(), c.w()};
        const float d = (float) ((const sycl::half *) (row + cols))[b];
#pragma unroll
        for (int i = 0; i < 8; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j) w[4 * i + j] = (float) (int8_t) ((u[i] >> (8 * j)) & 0xff) * d;
    } else if (f == F_Q6) {
        const sycl::uint4 lo = *(const sycl::uint4 *) (row + (size_t) b * 16);
        const sycl::uint2 hi = *(const sycl::uint2 *) (row + (size_t) cols / 2 + (size_t) b * 8);
        const uint32_t u[4] = {lo.x(), lo.y(), lo.z(), lo.w()};
        const uint64_t hb = ((uint64_t) hi.y() << 32) | hi.x();
        const float d = (float) ((const sycl::half *) (row + (size_t) cols / 2 + cols / 4))[b];
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int k = i * 8 + j;
                w[k] = (float) (((int) ((u[i] >> (4 * j)) & 0xf) | (int) (((hb >> (2 * k)) & 0x3) << 4)) - 32) * d;
            }
    } else {
        const sycl::uint4 qq = *(const sycl::uint4 *) (row + (size_t) b * 16);
        const uint32_t u[4] = {qq.x(), qq.y(), qq.z(), qq.w()};
        const float d = (float) ((const sycl::half *) (row + (size_t) cols / 2))[b];
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const uint32_t by = (u[i] >> (8 * j)) & 0xff;
                w[i * 4 + j] = nl4_val(by & 0xf) * d;
                w[16 + i * 4 + j] = nl4_val(by >> 4) * d;
            }
    }
}

// Y[t*ldy + yoff + r] = W[r] . X[t*cols ..] (+ R[t*ldy + yoff + r]) for t < T; sub-group per row, TG tokens per pass
static void gemm(sycl::queue & q, const DMat & m, const float * X, int T, float * Y, int ldy, int yoff, const float * R = nullptr) {
    const int nrows = m.rows, cols = m.cols;
    const Fmt f = m.f;
    const uint8_t * W = m.w;
    const size_t rb = m.rb;
    constexpr int WG = 128;
    const size_t g = (size_t) ((nrows + WG / SG - 1) / (WG / SG)) * WG;
    q.parallel_for(sycl::nd_range<1>(g, WG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        const auto sg = it.get_sub_group();
        const int r = it.get_global_id(0) / SG;
        if (r >= nrows) return;
        const int lane = sg.get_local_id()[0];
        const uint8_t * row = W + (size_t) r * rb;
        for (int t0 = 0; t0 < T; t0 += TG) {
            const int tn = sycl::min(TG, T - t0);
            float acc[TG];
#pragma unroll
            for (int k = 0; k < TG; ++k) acc[k] = 0.f;
            if (f == F_F32) {
                const float * wr = (const float *) row;
                for (int i = lane; i < cols; i += SG) {
                    const float wv = wr[i];
#pragma unroll
                    for (int k = 0; k < TG; ++k) if (k < tn) acc[k] += wv * X[(size_t) (t0 + k) * cols + i];
                }
            } else {
                float w[32];
                for (int b = lane; b < cols / 32; b += SG) {
                    deq_block(f, row, cols, b, w);
#pragma unroll
                    for (int k = 0; k < TG; ++k) {
                        if (k >= tn) break;
                        const sycl::float4 * xv = (const sycl::float4 *) (X + (size_t) (t0 + k) * cols + b * 32);
                        float a = 0.f;
#pragma unroll
                        for (int i = 0; i < 8; ++i) { const sycl::float4 xx = xv[i]; a += w[4 * i] * xx.x() + w[4 * i + 1] * xx.y() + w[4 * i + 2] * xx.z() + w[4 * i + 3] * xx.w(); }
                        acc[k] += a;
                    }
                }
            }
#pragma unroll
            for (int k = 0; k < TG; ++k) {
                const float v = sycl::reduce_over_group(sg, acc[k], sycl::plus<float>());
                if (lane == 0 && k < tn) {
                    const size_t o = (size_t) (t0 + k) * ldy + yoff + r;
                    Y[o] = v + (R ? R[o] : 0.f);
                }
            }
        }
    });
}

// W (any format) -> f16 row-major [rows][cols]
static void dequant_f16(sycl::queue & q, const DMat & m, sycl::half * out) {
    const int cols = m.cols, nb = cols / 32;
    const Fmt f = m.f;
    const uint8_t * W = m.w;
    const size_t rb = m.rb;
    q.parallel_for(sycl::range<1>((size_t) m.rows * nb), [=](sycl::id<1> i) {
        const size_t r = i[0] / nb; const int b = (int) (i[0] % nb);
        const uint8_t * row = W + r * rb;
        sycl::half * o = out + r * cols + b * 32;
        if (f == F_F32) { for (int k = 0; k < 32; ++k) o[k] = sycl::half(((const float *) row)[b * 32 + k]); return; }
        float w[32];
        deq_block(f, row, cols, b, w);
        for (int k = 0; k < 32; ++k) o[k] = sycl::half(w[k]);
    });
}

// Y[t*ldy + yoff + r] = W[r] . X[t] (+ R) via oneMKL f16 GEMM (XMX); scratch: wf (rows*cols halves), xf (T*cols halves)
static void gemm_mkl(sycl::queue & q, const DMat & m, const float * X, int T, float * Y, int ldy, int yoff, const float * R,
                     sycl::half * wf, sycl::half * xf) {
    const int cols = m.cols, rows = m.rows;
    q.parallel_for(sycl::range<1>((size_t) T * cols), [=](sycl::id<1> i) { xf[i[0]] = sycl::half(X[i[0]]); });
    dequant_f16(q, m, wf);
    if (R) {
        q.parallel_for(sycl::range<2>(T, rows), [=](sycl::id<2> i) { const size_t o = i[0] * ldy + yoff + i[1]; Y[o] = R[o]; });
    }
    oneapi::mkl::blas::row_major::gemm(q, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::trans, T, rows, cols, 1.0f,
                                       xf, cols, wf, cols, R ? 1.0f : 0.0f, Y + yoff, ldy);
}

// Y[t*ldy + r] = W[r] . X[t*cols ..] (+ R[t*ldy + r]) for t < NT (NT small, compile-time)
template <int NT>
static void gemv_nt(sycl::queue & q, const DMat & m, const float * X, float * Y, int ldy, const float * R = nullptr, const sycl::half * Xh = nullptr) {
    const int nrows = m.rows, cols = m.cols;
    const Fmt f = m.f;
    const uint8_t * W = m.w;
    const size_t rb = m.rb;
    constexpr int WG = 128;
    const size_t g = (size_t) ((nrows + WG / SG - 1) / (WG / SG)) * WG;
    q.parallel_for(sycl::nd_range<1>(g, WG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        const auto sg = it.get_sub_group();
        const int r = it.get_global_id(0) / SG;
        if (r >= nrows) return;
        const int lane = sg.get_local_id()[0];
        const uint8_t * row = W + (size_t) r * rb;
        float acc[NT];
#pragma unroll
        for (int k = 0; k < NT; ++k) acc[k] = 0.f;
        if (f == F_F32) {
            const sycl::float4 * wr = (const sycl::float4 *) row;
            for (int i = lane; i < cols / 4; i += SG) {
                const sycl::float4 a = wr[i];
#pragma unroll
                for (int k = 0; k < NT; ++k) {
                    const sycl::float4 b = ((const sycl::float4 *) (X + (size_t) k * cols))[i];
                    acc[k] += a.x() * b.x() + a.y() * b.y() + a.z() * b.z() + a.w() * b.w();
                }
            }
        } else {
            float w[32];
            for (int b = lane; b < cols / 32; b += SG) {
                deq_block(f, row, cols, b, w);
#pragma unroll
                for (int k = 0; k < NT; ++k) {
                    float a = 0.f;
                    if (Xh) {
                        const sycl::vec<sycl::half, 8> * xv = (const sycl::vec<sycl::half, 8> *) (Xh + (size_t) k * cols + b * 32);
#pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            const sycl::vec<sycl::half, 8> xx = xv[i];
#pragma unroll
                            for (int j = 0; j < 8; ++j) a += w[8 * i + j] * (float) xx[j];
                        }
                    } else {
                        const sycl::float4 * xv = (const sycl::float4 *) (X + (size_t) k * cols + b * 32);
#pragma unroll
                        for (int i = 0; i < 8; ++i) { const sycl::float4 xx = xv[i]; a += w[4 * i] * xx.x() + w[4 * i + 1] * xx.y() + w[4 * i + 2] * xx.z() + w[4 * i + 3] * xx.w(); }
                    }
                    acc[k] += a;
                }
            }
        }
#pragma unroll
        for (int k = 0; k < NT; ++k) {
            const float v = sycl::reduce_over_group(sg, acc[k], sycl::plus<float>());
            if (lane == 0) Y[(size_t) k * ldy + r] = v + (R ? R[(size_t) k * ldy + r] : 0.f);
        }
    });
}

// row-wise rmsnorm of T rows: Yn[t] = rmsnorm(X[t]) * w
static void rmsnorm_rows(sycl::queue & q, const float * X, const float * w, float * Yn, int T, int n = D) {
    q.parallel_for(sycl::nd_range<1>((size_t) T * 256, 256), [=](sycl::nd_item<1> it) {
        const int t = it.get_group(0);
        const float * x = X + (size_t) t * n;
        float ss = 0.f;
        for (int i = it.get_local_id(0); i < n; i += 256) ss += x[i] * x[i];
        ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
        const float inv = sycl::rsqrt(ss / n + RMS_EPS);
        for (int i = it.get_local_id(0); i < n; i += 256) Yn[(size_t) t * n + i] = x[i] * inv * w[i];
    });
}

// ============================ XMX expert GEMM (prefill) ============================
// out[row][n] = sum_k X[row][k] * W_e[n][k] for the padded rows of every resident expert e
//   X: f16 [rows][K] (rows of expert e = pad_off[e] .. pad_off[e+1], multiple of 8), W_e: NL4 rows at
//   arena + slot*EB + wrow0 + n*wrb (N rows of K), out f32 [rows][N]
// work-group = (expert, block of 128 output columns), 8 sub-groups x 16 columns; weights dequantized to f16 in SLM (VNNI)
namespace sxm = sycl::ext::oneapi::experimental::matrix;
static void moe_xmx(sycl::queue & q, const sycl::half * X, int K, int N, const uint8_t * arena, size_t EB, size_t wrow0, size_t wrb,
                    const int32_t * bslot, const int32_t * pad_off, float * out, const uint8_t * stage = nullptr) {
    constexpr int TM = 8, TN = 16, TK = 16, MT = 4, NB = 128;
    q.submit([&](sycl::handler & h) {
        sycl::local_accessor<sycl::half, 1> Bs((32 / 2) * NB * 2, h);
        h.parallel_for(sycl::nd_range<2>({(size_t) N_EXPERT, (size_t) (N / NB) * 128}, {1, 128}), [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
            const int e = it.get_global_id(0), nb = it.get_group(1);
            const int slot = bslot[e];
            const int r0 = pad_off[e], r1 = pad_off[e + 1];
            if (slot < 0 || r0 == r1) return;
            const auto sg = it.get_sub_group();
            const int sgi = sg.get_group_id()[0], lid = it.get_local_id(1);
            const uint8_t * wbase = (slot >= 1000000 ? stage + (size_t) (slot - 1000000) * EB : arena + (size_t) slot * EB) + wrow0;
            const uint8_t * myrow = wbase + (size_t) (nb * NB + lid) * wrb;      // this work-item dequantizes one weight row
            auto pX = sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(X);
            auto pO = sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(out);
            auto pB = Bs.template get_multi_ptr<sycl::access::decorated::no>();
            for (int m0 = r0; m0 < r1; m0 += TM * MT) {
                const int nt = sycl::min(MT, (r1 - m0) / TM);
                sxm::joint_matrix<sycl::sub_group, float, sxm::use::accumulator, TM, TN> c[MT];
#pragma unroll
                for (int t = 0; t < MT; ++t) sxm::joint_matrix_fill(sg, c[t], 0.f);
                for (int k0 = 0; k0 < K; k0 += 32) {
                    {   // NL4 block k0/32 of my row -> 32 f16 values, stored VNNI-packed: Bs[(k/2)][col*2 + k%2]
                        const int b = k0 / 32;
                        const sycl::uint4 qq = *(const sycl::uint4 *) (myrow + (size_t) b * 16);
                        const float d = (float) ((const sycl::half *) (myrow + K / 2))[b];
                        const uint32_t u[4] = {qq.x(), qq.y(), qq.z(), qq.w()};
#pragma unroll
                        for (int i = 0; i < 4; ++i)
#pragma unroll
                            for (int j = 0; j < 4; ++j) {
                                const uint32_t by = (u[i] >> (8 * j)) & 0xff;
                                const int k_lo = i * 4 + j, k_hi = 16 + i * 4 + j;
                                Bs[(k_lo / 2) * NB * 2 + lid * 2 + (k_lo & 1)] = sycl::half(nl4_val(by & 0xf) * d);
                                Bs[(k_hi / 2) * NB * 2 + lid * 2 + (k_hi & 1)] = sycl::half(nl4_val(by >> 4) * d);
                            }
                    }
                    it.barrier(sycl::access::fence_space::local_space);
#pragma unroll
                    for (int ks = 0; ks < 32; ks += TK) {
                        sxm::joint_matrix<sycl::sub_group, sycl::half, sxm::use::b, TK, TN, sxm::layout::ext_intel_packed> bm;
                        sxm::joint_matrix_load(sg, bm, pB + (ks / 2) * NB * 2 + sgi * TN * 2, NB * 2);
#pragma unroll
                        for (int t = 0; t < MT; ++t) {
                            if (t >= nt) break;
                            sxm::joint_matrix<sycl::sub_group, sycl::half, sxm::use::a, TM, TK, sxm::layout::row_major> am;
                            sxm::joint_matrix_load(sg, am, pX + (size_t) (m0 + t * TM) * K + k0 + ks, K);
                            sxm::joint_matrix_mad(sg, c[t], am, bm, c[t]);
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                }
#pragma unroll
                for (int t = 0; t < MT; ++t) {
                    if (t >= nt) break;
                    sxm::joint_matrix_store(sg, c[t], pO + (size_t) (m0 + t * TM) * N + nb * NB + sgi * TN, N, sxm::layout::row_major);
                }
            }
        });
    });
}

// ============================ prefill flash attention (XMX) ============================
// work-group = (kv head, tile of 8 query tokens); its 8 sub-groups are the 8 q heads sharing that kv head.
// KV (int8, paged) is dequantized 16 positions at a time into SLM as f16; S = Q K^T and O = P V run on joint_matrix.
// Two passes over the keys (row max, then exp / sum / P V) so the accumulators never need rescaling.
//   QH: f16 [T][NQ][HD] queries pre-scaled by 1/sqrt(HD); O: f32 [T][NQ*HD] unnormalized; L: f32 [T][NQ] row sums
static void flash_prefill(sycl::queue & q, const sycl::half * QH, int T, int pos0, const int8_t * kc, const int8_t * vc,
                          const sycl::half * ks, const sycl::half * vs, const int32_t * bt, float * O, float * Lsum) {
    constexpr int TQ = 8, KB = 16, NC = HD / 16;   // query rows per tile, keys per block, 16-wide dim chunks
    const int n_tiles = (T + TQ - 1) / TQ;
    namespace sx = sycl::ext::oneapi::experimental;
    q.submit([&](sycl::handler & h) {
        sycl::local_accessor<sycl::half, 1> Kt((HD / 2) * KB * 2, h);    // K^T tile, VNNI: (d, p) at [d/2][p*2 + d%2]
        sycl::local_accessor<sycl::half, 1> Vt((KB / 2) * HD * 2, h);    // V tile,   VNNI: (p, d) at [p/2][d*2 + p%2]
        sycl::local_accessor<float, 1> Ss(8 * TQ * KB, h);               // scores per sub-group
        sycl::local_accessor<sycl::half, 1> Ps(8 * TQ * KB, h);          // probabilities per sub-group (A operand)
        sycl::local_accessor<float, 1> Ms(8 * TQ * 2, h);                // row max / row sum per sub-group
        h.parallel_for(sycl::nd_range<2>({(size_t) NKV, (size_t) n_tiles * 128}, {1, 128}),
                       sx::properties{sycl::ext::intel::experimental::grf_size<256>},
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
            const int kh = it.get_global_id(0), tile = it.get_group(1);
            const auto sg = it.get_sub_group();
            const int s_id = sg.get_group_id()[0], lane = sg.get_local_id()[0], lid = it.get_local_id(1);
            const int hq = kh * (NQ / NKV) + s_id;
            const int t0 = tile * TQ;
            const int kmax = pos0 + sycl::min(t0 + TQ - 1, T - 1);            // last key any row of this tile may see
            auto pQ = sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(QH);
            auto pO = sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(O);
            auto pK = Kt.template get_multi_ptr<sycl::access::decorated::no>();
            auto pV = Vt.template get_multi_ptr<sycl::access::decorated::no>();
            auto pS = Ss.template get_multi_ptr<sycl::access::decorated::no>() + s_id * TQ * KB;
            auto pP = Ps.template get_multi_ptr<sycl::access::decorated::no>() + s_id * TQ * KB;
            float * mrow = &Ms[s_id * TQ * 2];
            if (lane < TQ) { mrow[lane] = -1e30f; mrow[TQ + lane] = 0.f; }
            // dequantize keys [p0, p0+KB) (and values when withV) of this kv head into SLM; masked / missing positions -> 0
            auto load_tile = [&](int p0, bool withV) {
                for (int idx = lid; idx < KB * (HD / 16); idx += 128) {       // one 16-value group per item
                    const int pp = idx / (HD / 16), g = idx % (HD / 16);
                    const int p = p0 + pp;
                    float kd = 0.f, vd = 0.f;
                    sycl::vec<int8_t, 16> kk(0), vv(0);
                    if (p <= kmax) {
                        const size_t o = ((size_t) (bt[p >> 8] * 256 + (p & 255)) * NKV + kh) * HD + g * 16;
                        kk = *(const sycl::vec<int8_t, 16> *) (kc + o);
                        kd = (float) ks[o / 16];
                        if (withV) { vv = *(const sycl::vec<int8_t, 16> *) (vc + o); vd = (float) vs[o / 16]; }
                    }
#pragma unroll
                    for (int i = 0; i < 16; ++i) {
                        const int d = g * 16 + i;
                        Kt[(d / 2) * KB * 2 + pp * 2 + (d & 1)] = sycl::half(kd * (float) kk[i]);
                        if (withV) Vt[(pp / 2) * HD * 2 + d * 2 + (pp & 1)] = sycl::half(vd * (float) vv[i]);
                    }
                }
            };
            // S = Q (8 x 256) K^T (256 x 16) -> pS
            auto scores = [&]() {
                sx::matrix::joint_matrix<sycl::sub_group, float, sx::matrix::use::accumulator, TQ, KB> c;
                sx::matrix::joint_matrix_fill(sg, c, 0.f);
#pragma unroll
                for (int dc = 0; dc < NC; ++dc) {
                    sx::matrix::joint_matrix<sycl::sub_group, sycl::half, sx::matrix::use::a, TQ, 16, sx::matrix::layout::row_major> a;
                    sx::matrix::joint_matrix<sycl::sub_group, sycl::half, sx::matrix::use::b, 16, KB, sx::matrix::layout::ext_intel_packed> b;
                    sx::matrix::joint_matrix_load(sg, a, pQ + ((size_t) t0 * NQ + hq) * HD + dc * 16, NQ * HD);
                    sx::matrix::joint_matrix_load(sg, b, pK + (dc * 8) * KB * 2, KB * 2);
                    sx::matrix::joint_matrix_mad(sg, c, a, b, c);
                }
                sx::matrix::joint_matrix_store(sg, c, pS, KB, sx::matrix::layout::row_major);
            };
            // pass 1: row max
            for (int p0 = 0; p0 <= kmax; p0 += KB) {
                load_tile(p0, false);
                it.barrier(sycl::access::fence_space::local_space);
                scores();
                sycl::group_barrier(sg);
                if (lane < TQ && t0 + lane < T) {
                    const int qp = pos0 + t0 + lane;
                    float m = mrow[lane];
                    for (int j = 0; j < KB; ++j) if (p0 + j <= qp) m = sycl::max(m, pS[lane * KB + j]);
                    mrow[lane] = m;
                }
                it.barrier(sycl::access::fence_space::local_space);
            }
            // pass 2: P = exp(S - m), row sums, O += P V
            sx::matrix::joint_matrix<sycl::sub_group, float, sx::matrix::use::accumulator, TQ, 16> acc[NC];
#pragma unroll
            for (int n = 0; n < NC; ++n) sx::matrix::joint_matrix_fill(sg, acc[n], 0.f);
            for (int p0 = 0; p0 <= kmax; p0 += KB) {
                load_tile(p0, true);
                it.barrier(sycl::access::fence_space::local_space);
                scores();
                sycl::group_barrier(sg);
                if (lane < TQ) {
                    const int qp = pos0 + t0 + lane;
                    const float m = mrow[lane];
                    float sum = 0.f;
                    for (int j = 0; j < KB; ++j) {
                        const float pv = (t0 + lane < T && p0 + j <= qp) ? sycl::exp(pS[lane * KB + j] - m) : 0.f;
                        pP[lane * KB + j] = sycl::half(pv);
                        sum += pv;
                    }
                    mrow[TQ + lane] += sum;
                }
                sycl::group_barrier(sg);
                sx::matrix::joint_matrix<sycl::sub_group, sycl::half, sx::matrix::use::a, TQ, KB, sx::matrix::layout::row_major> pa;
                sx::matrix::joint_matrix_load(sg, pa, pP, KB);
#pragma unroll
                for (int n = 0; n < NC; ++n) {
                    sx::matrix::joint_matrix<sycl::sub_group, sycl::half, sx::matrix::use::b, KB, 16, sx::matrix::layout::ext_intel_packed> vb;
                    sx::matrix::joint_matrix_load(sg, vb, pV + n * 16 * 2, HD * 2);
                    sx::matrix::joint_matrix_mad(sg, acc[n], pa, vb, acc[n]);
                }
                it.barrier(sycl::access::fence_space::local_space);
            }
#pragma unroll
            for (int n = 0; n < NC; ++n)
                sx::matrix::joint_matrix_store(sg, acc[n], pO + (size_t) t0 * NQ * HD + hq * HD + n * 16, NQ * HD, sx::matrix::layout::row_major);
            if (lane < TQ && t0 + lane < T) Lsum[(size_t) (t0 + lane) * NQ + hq] = mrow[TQ + lane];
        });
    });
}

// single-pass variant: online softmax, the O accumulators are rescaled per row with Intel's coordinate-aware joint_matrix_apply
static void flash_prefill2(sycl::queue & q, const sycl::half * QH, int T, int pos0, const int8_t * kc, const int8_t * vc,
                           const sycl::half * ks, const sycl::half * vs, const int32_t * bt, float * O, float * Lsum) {
    constexpr int TQ = 8, KB = 32, NC = HD / 16, NCH = NC / 2;   // NCH: output chunks per sub-group (half the head)
    const int n_tiles = (T + TQ - 1) / TQ;
    namespace sx = sycl::ext::oneapi::experimental;
    namespace sm = sycl::ext::oneapi::experimental::matrix;
    q.submit([&](sycl::handler & h) {
        sycl::local_accessor<sycl::half, 1> Kt((HD / 2) * KB * 2, h);    // K^T tile, VNNI: (d, p) at [d/2][p*2 + d%2]
        sycl::local_accessor<sycl::half, 1> Vt((KB / 2) * HD * 2, h);    // V tile,   VNNI: (p, d) at [p/2][d*2 + p%2]
        sycl::local_accessor<float, 1> Ss(16 * TQ * KB, h);
        sycl::local_accessor<sycl::half, 1> Ps(16 * TQ * KB, h);
        sycl::local_accessor<float, 1> Ms(16 * TQ * 3, h);                // per sub-group: row max, row sum, rescale factor
        h.parallel_for(sycl::nd_range<2>({(size_t) NKV, (size_t) n_tiles * 256}, {1, 256}),
                       sx::properties{sycl::ext::intel::experimental::grf_size<256>},
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
            const int kh = it.get_global_id(0), tile = it.get_group(1);
            const auto sg = it.get_sub_group();
            const int s_id = sg.get_group_id()[0], lane = sg.get_local_id()[0], lid = it.get_local_id(1);
            const int hq = kh * (NQ / NKV) + s_id / 2, dh = s_id & 1;   // q head, which half of the head dims
            const int t0 = tile * TQ;
            const int kmax = pos0 + sycl::min(t0 + TQ - 1, T - 1);
            auto pQ = sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(QH);
            auto pO = sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(O);
            auto pK = Kt.template get_multi_ptr<sycl::access::decorated::no>();
            auto pV = Vt.template get_multi_ptr<sycl::access::decorated::no>();
            auto pS = Ss.template get_multi_ptr<sycl::access::decorated::no>() + (s_id / 2) * TQ * KB;
            auto pP = Ps.template get_multi_ptr<sycl::access::decorated::no>() + (s_id / 2) * TQ * KB;
            float * mrow = &Ms[(s_id / 2) * TQ * 3];
            if (dh == 0 && lane < TQ) { mrow[lane] = -1e30f; mrow[TQ + lane] = 0.f; mrow[2 * TQ + lane] = 1.f; }
            sm::joint_matrix<sycl::sub_group, float, sm::use::accumulator, TQ, 16> acc[NCH];
#pragma unroll
            for (int n = 0; n < NCH; ++n) sm::joint_matrix_fill(sg, acc[n], 0.f);
            for (int p0 = 0; p0 <= kmax; p0 += KB) {
                // K and V for keys [p0, p0+KB) of this kv head -> SLM (f16); masked / missing positions -> 0
                for (int idx = lid; idx < KB * (HD / 16); idx += 256) {
                    const int pp = idx / (HD / 16), g = idx % (HD / 16);
                    const int p = p0 + pp;
                    float kd = 0.f, vd = 0.f;
                    sycl::vec<int8_t, 16> kk(0), vv(0);
                    if (p <= kmax) {
                        const size_t o = ((size_t) (bt[p >> 8] * 256 + (p & 255)) * NKV + kh) * HD + g * 16;
                        kk = *(const sycl::vec<int8_t, 16> *) (kc + o); kd = (float) ks[o / 16];
                        vv = *(const sycl::vec<int8_t, 16> *) (vc + o); vd = (float) vs[o / 16];
                    }
#pragma unroll
                    for (int i = 0; i < 16; ++i) {
                        const int d = g * 16 + i;
                        Kt[(d / 2) * KB * 2 + pp * 2 + (d & 1)] = sycl::half(kd * (float) kk[i]);
                        Vt[(pp / 2) * HD * 2 + d * 2 + (pp & 1)] = sycl::half(vd * (float) vv[i]);
                    }
                }
                it.barrier(sycl::access::fence_space::local_space);
                // S = Q K^T: this sub-group computes key columns [dh*16, dh*16+16) of its head
                {
                    sm::joint_matrix<sycl::sub_group, float, sm::use::accumulator, TQ, 16> c0;
                    sm::joint_matrix_fill(sg, c0, 0.f);
#pragma unroll
                    for (int dc = 0; dc < NC; ++dc) {
                        sm::joint_matrix<sycl::sub_group, sycl::half, sm::use::a, TQ, 16, sm::layout::row_major> a;
                        sm::joint_matrix<sycl::sub_group, sycl::half, sm::use::b, 16, 16, sm::layout::ext_intel_packed> b0;
                        sm::joint_matrix_load(sg, a, pQ + ((size_t) t0 * NQ + hq) * HD + dc * 16, NQ * HD);
                        sm::joint_matrix_load(sg, b0, pK + (dc * 8) * KB * 2 + dh * 32, KB * 2);
                        sm::joint_matrix_mad(sg, c0, a, b0, c0);
                    }
                    sm::joint_matrix_store(sg, c0, pS + dh * 16, KB, sm::layout::row_major);
                }
                it.barrier(sycl::access::fence_space::local_space);
                // online softmax (first sub-group of each head): 2 lanes per row, 16 keys each
                if (dh == 0) {
                    const int row = lane >> 1, half_ = lane & 1;
                    const int qp = pos0 + t0 + row;
                    const bool live = t0 + row < T;
                    float mx = -1e30f;
                    for (int j = half_ * 16; j < half_ * 16 + 16; ++j) if (live && p0 + j <= qp) mx = sycl::max(mx, pS[row * KB + j]);
                    mx = sycl::max(mx, sycl::permute_group_by_xor(sg, mx, 1));
                    const float m_old = mrow[row];
                    const float m_new = sycl::max(m_old, mx);
                    float sum = 0.f;
                    for (int j = half_ * 16; j < half_ * 16 + 16; ++j) {
                        const float pv = (live && p0 + j <= qp) ? sycl::exp(pS[row * KB + j] - m_new) : 0.f;
                        pP[row * KB + j] = sycl::half(pv);
                        sum += pv;
                    }
                    sum += sycl::permute_group_by_xor(sg, sum, 1);
                    if (half_ == 0) {
                        const float alpha = sycl::exp(m_old - m_new);
                        mrow[TQ + row] = mrow[TQ + row] * alpha + sum;
                        mrow[2 * TQ + row] = alpha;
                        mrow[row] = m_new;
                    }
                }
                it.barrier(sycl::access::fence_space::local_space);
                // rescale O rows, then O += P V
                {
                    const float * al = mrow + 2 * TQ;
#pragma unroll
                    for (int n = 0; n < NCH; ++n)
                        sycl::ext::intel::experimental::matrix::joint_matrix_apply(sg, acc[n], [=](float & x, size_t r, size_t) { x *= al[r]; });
#pragma unroll
                    for (int ksub = 0; ksub < KB; ksub += 16) {
                        sm::joint_matrix<sycl::sub_group, sycl::half, sm::use::a, TQ, 16, sm::layout::row_major> pa;
                        sm::joint_matrix_load(sg, pa, pP + ksub, KB);
#pragma unroll
                        for (int n = 0; n < NCH; ++n) {
                            sm::joint_matrix<sycl::sub_group, sycl::half, sm::use::b, 16, 16, sm::layout::ext_intel_packed> vb;
                            sm::joint_matrix_load(sg, vb, pV + (ksub / 2) * HD * 2 + (dh * NCH + n) * 16 * 2, HD * 2);
                            sm::joint_matrix_mad(sg, acc[n], pa, vb, acc[n]);
                        }
                    }
                }
                it.barrier(sycl::access::fence_space::local_space);
            }
#pragma unroll
            for (int n = 0; n < NCH; ++n)
                sm::joint_matrix_store(sg, acc[n], pO + (size_t) t0 * NQ * HD + hq * HD + (dh * NCH + n) * 16, NQ * HD, sm::layout::row_major);
            if (dh == 0 && lane < TQ && t0 + lane < T) Lsum[(size_t) (t0 + lane) * NQ + hq] = mrow[TQ + lane];
        });
    });
}

// ============================ CPU expert path ============================
static inline float hsum256(__m256 v) {
    __m128 a = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    a = _mm_add_ps(a, _mm_movehl_ps(a, a));
    a = _mm_add_ss(a, _mm_movehdup_ps(a));
    return _mm_cvtss_f32(a);
}
static void quant_x(const float * x, int n, int8_t * xq, float * xs) {
    const __m256 sign = _mm256_set1_ps(-0.f);
    for (int b = 0; b < n / 32; ++b) {
        __m256 v[4], mx = _mm256_setzero_ps();
        for (int i = 0; i < 4; ++i) { v[i] = _mm256_loadu_ps(x + b * 32 + i * 8); mx = _mm256_max_ps(mx, _mm256_andnot_ps(sign, v[i])); }
        __m128 m = _mm_max_ps(_mm256_castps256_ps128(mx), _mm256_extractf128_ps(mx, 1));
        m = _mm_max_ps(m, _mm_movehl_ps(m, m)); m = _mm_max_ss(m, _mm_movehdup_ps(m));
        const float d = _mm_cvtss_f32(m) / 127.f, id = d > 0 ? 1.f / d : 0.f;
        xs[b] = d;
        const __m256 mul = _mm256_set1_ps(id);
        __m256i i32[4];
        for (int i = 0; i < 4; ++i) i32[i] = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(v[i], mul), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        __m256i p = _mm256_packs_epi16(_mm256_packs_epi32(i32[0], i32[1]), _mm256_packs_epi32(i32[2], i32[3]));
        p = _mm256_permutevar8x32_epi32(p, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
        _mm256_storeu_si256((__m256i *) (xq + b * 32), p);
    }
}
static inline float cpu_dot_nl4(const uint8_t * row, int cols, const int8_t * xq, const float * xs) {
    const uint16_t * sc = (const uint16_t *) (row + cols / 2);
    const __m128i tbl = _mm_loadu_si128((const __m128i *) KVALUES_NL4);
    const __m128i m4 = _mm_set1_epi8(0x0f);
    const __m256i ones = _mm256_set1_epi16(1);
    __m256 acc = _mm256_setzero_ps();
    for (int b = 0; b < cols / 32; ++b) {
        const __m128i raw = _mm_loadu_si128((const __m128i *) (row + b * 16));
        const __m128i lo = _mm_shuffle_epi8(tbl, _mm_and_si128(raw, m4));
        const __m128i hi = _mm_shuffle_epi8(tbl, _mm_and_si128(_mm_srli_epi16(raw, 4), m4));
        const __m256i w = _mm256_set_m128i(hi, lo);
        const __m256i xv = _mm256_loadu_si256((const __m256i *) (xq + b * 32));
        const __m256i p = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_sign_epi8(w, w), _mm256_sign_epi8(xv, w)), ones);
        acc = _mm256_fmadd_ps(_mm256_set1_ps(_cvtsh_ss(sc[b]) * xs[b]), _mm256_cvtepi32_ps(p), acc);
    }
    return hsum256(acc);
}

// out[t] = row . x_t for n <= 4 activation rows (weights decoded once per block)
static inline void cpu_dot_nl4_x4(const uint8_t * row, int cols, const int8_t * const * xq, const float * const * xs, int n, float * out) {
    const uint16_t * sc = (const uint16_t *) (row + cols / 2);
    const __m128i tbl = _mm_loadu_si128((const __m128i *) KVALUES_NL4);
    const __m128i m4 = _mm_set1_epi8(0x0f);
    const __m256i ones = _mm256_set1_epi16(1);
    __m256 acc[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};
    for (int b = 0; b < cols / 32; ++b) {
        const __m128i raw = _mm_loadu_si128((const __m128i *) (row + b * 16));
        const __m256i w = _mm256_set_m128i(_mm_shuffle_epi8(tbl, _mm_and_si128(_mm_srli_epi16(raw, 4), m4)), _mm_shuffle_epi8(tbl, _mm_and_si128(raw, m4)));
        const __m256i aw = _mm256_sign_epi8(w, w);
        const float dw = _cvtsh_ss(sc[b]);
        for (int t = 0; t < n; ++t) {
            const __m256i xv = _mm256_loadu_si256((const __m256i *) (xq[t] + b * 32));
            const __m256i p = _mm256_madd_epi16(_mm256_maddubs_epi16(aw, _mm256_sign_epi8(xv, w)), ones);
            acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(dw * xs[t][b]), _mm256_cvtepi32_ps(p), acc[t]);
        }
    }
    for (int t = 0; t < n; ++t) out[t] = hsum256(acc[t]);
}

struct CpuPool {
    int nt = 1;
    std::vector<std::thread> th;
    std::atomic<int> gen{0}, done{0};
    std::atomic<bool> stop{false};
    std::mutex wmu;
    std::condition_variable wcv;       // idle workers sleep here after spinning for a while
    static constexpr int MAXT = 8;     // tokens per multi-token CPU pass
    static constexpr int MAXM = MAXT * K_USED;
    int n_miss = 0;
    const uint8_t * exp_ptr[MAXM];
    float exp_w[MAXM];
    int miss_tok[MAXM];
    int n_tok_m = 0;                   // > 0: multi-token mode
    alignas(64) int8_t xqm[MAXT][D];
    float xsm[MAXT][D / 32];
    float * outs[MAXT];
    alignas(64) int8_t xq[D];
    float xs[D / 32];
    alignas(64) float hbuf[MAXM * FF_EXP];
    alignas(64) int8_t hq[MAXM * FF_EXP];
    float hs[MAXM * FF_EXP / 32];
    float * out = nullptr;
    std::function<void(int)> job;      // arbitrary per-thread job (run_job)
    int job_phase = 0;

    void work(int tid, int phase) {
        if (job_phase) { job(tid); return; }
        if (phase == 0) {
            const int total = n_miss * FF_EXP, per = (total + nt - 1) / nt;
            for (int r = tid * per; r < std::min(total, (tid + 1) * per); ++r) {
                const int e = r / FF_EXP, rr = r % FF_EXP;
                const int8_t * xqp = n_tok_m ? xqm[miss_tok[e]] : xq;
                const float * xsp = n_tok_m ? xsm[miss_tok[e]] : xs;
                const float g = cpu_dot_nl4(exp_ptr[e] + (size_t) rr * exp_gu_rb(), D, xqp, xsp);
                const float u = cpu_dot_nl4(exp_ptr[e] + (size_t) (FF_EXP + rr) * exp_gu_rb(), D, xqp, xsp);
                hbuf[e * FF_EXP + rr] = g / (1.f + std::exp(-g)) * u;
            }
        } else if (n_tok_m) {
            const int per = (D + nt - 1) / nt;
            for (int r = tid * per; r < std::min(D, (tid + 1) * per); ++r) {
                float acc[MAXT] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
                for (int e = 0; e < n_miss; ++e)
                    acc[miss_tok[e]] += exp_w[e] * cpu_dot_nl4(exp_ptr[e] + 2 * (size_t) FF_EXP * exp_gu_rb() + (size_t) r * exp_d_rb(), FF_EXP,
                                                               hq + e * FF_EXP, hs + e * (FF_EXP / 32));
                for (int t = 0; t < n_tok_m; ++t) outs[t][r] = acc[t];
            }
        } else {
            const int per = (D + nt - 1) / nt;
            for (int r = tid * per; r < std::min(D, (tid + 1) * per); ++r) {
                float acc = 0.f;
                for (int e = 0; e < n_miss; ++e)
                    acc += exp_w[e] * cpu_dot_nl4(exp_ptr[e] + 2 * (size_t) FF_EXP * exp_gu_rb() + (size_t) r * exp_d_rb(), FF_EXP,
                                                  hq + e * FF_EXP, hs + e * (FF_EXP / 32));
                out[r] = acc;
            }
        }
    }
    void start(int n) {
        nt = n;
        for (int t = 1; t < nt; ++t) th.emplace_back([this, t] {
            cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(t, &cs);
            pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs);
            int seen = 0, idle = 0;
            while (!stop.load(std::memory_order_relaxed)) {
                const int g = gen.load(std::memory_order_acquire);
                if (g == seen) {
                    if (++idle < 200000) { _mm_pause(); continue; }   // ~8 ms of spinning, then sleep
                    std::unique_lock<std::mutex> lk(wmu);
                    wcv.wait_for(lk, std::chrono::milliseconds(50), [&] { return gen.load(std::memory_order_acquire) != seen || stop.load(); });
                    continue;
                }
                idle = 0;
                seen = g;
                work(t, g & 1 ? 0 : 1);
                done.fetch_add(1, std::memory_order_acq_rel);
            }
        });
    }
    void run_phase(int phase) {
        done.store(0, std::memory_order_relaxed);
        int g = gen.load() + 1;
        if ((g & 1) != (phase == 0 ? 1 : 0)) ++g;
        { std::lock_guard<std::mutex> lk(wmu); gen.store(g, std::memory_order_release); }
        wcv.notify_all();
        work(0, phase);
        for (int spins = 0; done.load(std::memory_order_acquire) < nt - 1; ++spins) { if (spins < 20000) _mm_pause(); else sched_yield(); }
    }
    void run_job(const std::function<void(int)> & f) {
        job = f; job_phase = 1;
        run_phase(0);
        job_phase = 0;
    }
    // misses of several tokens in one pass: x rows xf[t*D], results to outs[t]
    void compute_multi(const float * xf, int nt_tok) {
        for (int t = 0; t < nt_tok; ++t) quant_x(xf + (size_t) t * D, D, xqm[t], xsm[t]);
        n_tok_m = nt_tok;
        run_phase(0);
        for (int e = 0; e < n_miss; ++e) quant_x(hbuf + e * FF_EXP, FF_EXP, hq + e * FF_EXP, hs + e * (FF_EXP / 32));
        run_phase(1);
        n_tok_m = 0;
    }
    void compute(const float * xf) {
        quant_x(xf, D, xq, xs);
        run_phase(0);
        for (int e = 0; e < n_miss; ++e) quant_x(hbuf + e * FF_EXP, FF_EXP, hq + e * FF_EXP, hs + e * (FF_EXP / 32));
        run_phase(1);
    }
    ~CpuPool() { stop = true; wcv.notify_all(); for (auto & t : th) t.join(); }
};

// ============================ engine ============================
struct Layer {
    bool recr;
    float * attn_norm, * post_norm;
    DMat router;                       // 256 router rows + 1 shared-expert gate row (f32)
    DMat sh_gu, sh_down;               // shared expert: gate|up (1024 rows), down
    // gated delta net
    DMat qkv, z, ab, ssm_out;          // ab = alpha rows | beta rows (f32)
    float * conv_w, * dt_bias, * ssm_a, * ssm_norm;
    float * conv_state, * ssm_state;
    float * conv_bak = nullptr, * ssm_bak = nullptr;   // state after token `snap` of the last forward()
    // full attention
    DMat wqkv, wo;                     // q|gate (8192), k (512), v (512)
    float * q_norm, * k_norm;
    int8_t * kc, * vc;                 // [pos][kv head][HD] int8
    sycl::half * ks, * vs;             // [pos][kv head][HD/16] scales
};

struct SeqGroup { int seq, pos0, n, snap; };   // consecutive tokens of one slot in a multi-sequence batch

struct HostIO { int32_t ids[K_USED]; float wts[K_USED + 1]; float xf[D]; float cpu_out[D]; };

struct Engine {
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
    sycl::queue qc{q.get_context(), q.get_device(), sycl::property::queue::in_order{}};   // staging copies, overlap with compute
    Pack & pk;
    int max_ctx;                       // context per slot
    int n_slots = 1;
    int sq = 0;                        // current slot for the single-sequence paths
    int gqa_min_ctx = getenv("QE_GQA_MIN") ? atoi(getenv("QE_GQA_MIN")) : 8192;   // decode attention switches to the kv-head-shared kernel
    std::vector<int> rope_delta;       // per slot: KV index - rope position for text tokens (images advance rope by max(nx, ny))
    int32_t * RPOS = nullptr;          // optional [T][3] (t, h, w) rope positions for the next forward (images)
    bool use_rpos = false;
    static constexpr size_t CONV_SZ = (size_t) QKV_DIM * (CONV_K - 1), SSM_SZ = (size_t) NV * DK * DK;
    // paged KV: a pool of `pool_ctx` positions in blocks of KV_BLK, each slot maps its positions through a block table
    static constexpr int KV_BLK = ATT_CHUNK;          // 256: one decode attention chunk = one block
    int pool_ctx = 0, max_blocks = 0;                 // pool size (positions), blocks per slot table (max_ctx / KV_BLK)
    int32_t * BT = nullptr;                           // device [n_slots][max_blocks]
    std::vector<int32_t> bt_h;
    std::vector<std::vector<int>> slot_blocks;
    std::vector<int> free_blocks;
    std::function<bool(int)> on_pool_full;            // asked to free blocks of other slots; true if it freed something
    size_t kv_sz() const { return (size_t) pool_ctx * NKV * HD; }
    size_t kvs_sz() const { return (size_t) pool_ctx * NKV * (HD / 16); }
    // make sure positions [0, n_pos) of `slot` have blocks; throws when the pool is exhausted
    void ensure_kv(int slot, int n_pos) {
        const int need = (n_pos + KV_BLK - 1) / KV_BLK;
        if (need > max_blocks) throw std::runtime_error("context longer than the per-slot limit");
        auto & bl = slot_blocks[slot];
        const int had = (int) bl.size();
        while ((int) bl.size() < need) {
            if (free_blocks.empty() && !(on_pool_full && on_pool_full(slot))) {
                if ((int) bl.size() > had) q.memcpy(BT + (size_t) slot * max_blocks + had, bt_h.data() + (size_t) slot * max_blocks + had, (bl.size() - had) * 4).wait();
                throw std::runtime_error("KV cache pool exhausted");
            }
            if (free_blocks.empty()) continue;
            const int b = free_blocks.back(); free_blocks.pop_back();
            bt_h[(size_t) slot * max_blocks + bl.size()] = b;
            bl.push_back(b);
        }
        if ((int) bl.size() > had) q.memcpy(BT + (size_t) slot * max_blocks + had, bt_h.data() + (size_t) slot * max_blocks + had, (bl.size() - had) * 4).wait();
    }
    // keep blocks for positions [0, keep_pos), release the rest
    void release_kv(int slot, int keep_pos) {
        const int keep = (keep_pos + KV_BLK - 1) / KV_BLK;
        auto & bl = slot_blocks[slot];
        while ((int) bl.size() > keep) { free_blocks.push_back(bl.back()); bl.pop_back(); }
    }
    int kv_used_blocks() const { return pool_ctx / KV_BLK - (int) free_blocks.size(); }
    std::vector<Layer> L;
    DMat lm;
    float * out_norm;
    const uint8_t * embd;              // host, Q8 rows
    uint8_t * arena = nullptr;         // device, resident experts
    int32_t * slot_d = nullptr;        // device [layer*256+e] -> arena slot or -1
    std::vector<int32_t> slot_h;
    std::vector<const uint8_t *> host_exp;  // [layer*256+e] -> host copy (non-resident) or nullptr
    uint8_t * host_arena = nullptr;
    // activations
    float * x, * x2, * xn, * qkvz, * ab, * attn_o, * qkv_a, * part, * att, * rlog, * shgu, * h, * wts, * logits;
    int32_t * ids, * tok;
    HostIO * hio;
    float * x_host;
    CpuPool pool;
    // prefill buffers
    float * X, * X2, * XN, * QZ, * AB, * AO, * QA, * ATT, * RL, * SHGU, * SHH, * SD, * HP, * YP, * WB, * SHG;
    int32_t * IDB, * PAIRS, * OFF, * BSLOT;
    float * X_host;
    float * HB, * LOGB;                // final-normed hidden rows, logits rows (small T)
    // MTP (nextn) draft block
    bool has_mtp = false;
    Layer mtpL;
    DMat eh_proj;
    float * enorm = nullptr, * hnorm = nullptr, * head_norm = nullptr;
    uint8_t * mtp_arena = nullptr;
    std::vector<int32_t> mtp_slots;
    std::vector<const uint8_t *> mtp_hexp;
    float * HPREV = nullptr, * HLAST = nullptr;
    float * HPEND = nullptr;           // per slot: main hidden rows for the pending MTP inputs (up to 2)
    int32_t * TOKB;
    int32_t * ids_host;
    float * xn_host, * wb_host, * ypc_h, * xs_b, * hcold, * hs_b;
    uint8_t * pcold_h;
    int8_t * xq_b, * hq_b;
    static constexpr int STAGE = 96;
    sycl::half * WF = nullptr, * XF = nullptr;
    void bgemm(const DMat & m, const float * Xin, int T, float * Y, int ldy, int yoff, const float * R = nullptr) {
        if (T >= 16) gemm_mkl(q, m, Xin, T, Y, ldy, yoff, R, WF, XF);
        else gemm(q, m, Xin, T, Y, ldy, yoff, R);
    }
    uint8_t * stage = nullptr;
    size_t dev_bytes = 0;
    // stats
    long n_tok = 0, n_miss = 0;
    double cpu_ms = 0;

    uint8_t * dmalloc(size_t n) { uint8_t * p = sycl::malloc_device<uint8_t>(n, q); if (!p) { fprintf(stderr, "device alloc %zu failed\n", n); exit(1); } dev_bytes += n; return p; }
    float * dfill(size_t n, float v) { float * p = (float *) dmalloc(n * 4); q.fill(p, v, n).wait(); return p; }
    DMat upload(const std::vector<std::string> & names) {
        DMat m;
        const PackEntry & e0 = pk.get(names[0]);
        m.f = (Fmt) e0.fmt; m.cols = e0.cols; m.rb = row_bytes(m.f, m.cols);
        size_t total = 0;
        for (auto & n : names) { const PackEntry & e = pk.get(n); m.rows += e.rows; total += e.bytes; }
        m.w = dmalloc(total);
        size_t off = 0;
        for (auto & n : names) { const PackEntry & e = pk.get(n); q.memcpy(m.w + off, pk.base + e.off, e.bytes).wait(); off += e.bytes; }
        return m;
    }
    float * uploadf(const std::string & n) { const PackEntry & e = pk.get(n); float * p = (float *) dmalloc(e.bytes); q.memcpy(p, pk.base + e.off, e.bytes).wait(); return p; }

    // ctx = per-slot context limit, pool = total KV positions shared by the slots (0: ctx per slot, no sharing)
    Engine(Pack & p, int ctx, int resident, const std::vector<std::string> & stats, Pack * mp = nullptr, int slots = 1, int kv_pool = 0)
        : pk(p), max_ctx(ctx), n_slots(std::max(1, slots)) {
        rope_delta.assign(n_slots, 0);
        pool_ctx = (kv_pool > 0 ? kv_pool : max_ctx * n_slots) / KV_BLK * KV_BLK;
        max_blocks = (max_ctx + KV_BLK - 1) / KV_BLK;
        bt_h.assign((size_t) n_slots * max_blocks, 0);
        slot_blocks.assign(n_slots, {});
        for (int b = pool_ctx / KV_BLK - 1; b >= 0; --b) free_blocks.push_back(b);
        printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
        const double t0 = now_ms();
        L.resize(N_LAYER);
        for (int il = 0; il < N_LAYER; ++il) {
            Layer & l = L[il];
            const std::string pre = "blk." + std::to_string(il) + ".";
            l.recr = is_recurrent(il);
            l.attn_norm = uploadf(pre + "attn_norm");
            l.post_norm = uploadf(pre + "post_norm");
            l.router = upload({pre + "router", pre + "sh_gate"});
            l.sh_gu = upload({pre + "sh_gate_proj", pre + "sh_up"});
            l.sh_down = upload({pre + "sh_down"});
            if (l.recr) {
                l.qkv = upload({pre + "qkv"});
                l.z = upload({pre + "z"});
                l.ab = upload({pre + "alpha", pre + "beta"});
                l.ssm_out = upload({pre + "ssm_out"});
                l.conv_w = uploadf(pre + "conv");
                l.dt_bias = uploadf(pre + "dt_bias");
                l.ssm_a = uploadf(pre + "ssm_a");
                l.ssm_norm = uploadf(pre + "ssm_norm");
                l.conv_state = dfill(CONV_SZ * n_slots, 0.f);
                l.ssm_state = dfill(SSM_SZ * n_slots, 0.f);
                l.conv_bak = dfill(CONV_SZ * n_slots, 0.f);
                l.ssm_bak = dfill(SSM_SZ * n_slots, 0.f);
            } else {
                l.wqkv = upload({pre + "wq", pre + "wk", pre + "wv"});
                l.wo = upload({pre + "wo"});
                l.q_norm = uploadf(pre + "q_norm");
                l.k_norm = uploadf(pre + "k_norm");
                l.kc = (int8_t *) dmalloc(kv_sz());
                l.vc = (int8_t *) dmalloc(kv_sz());
                l.ks = (sycl::half *) dmalloc(kvs_sz() * 2);
                l.vs = (sycl::half *) dmalloc(kvs_sz() * 2);
            }
        }
        lm = upload({"output"});
        out_norm = uploadf("output_norm");
        embd = pk.ptr("token_embd");
        printf("dense weights + KV on device: %.2f GB (%.1f s)\n", dev_bytes / 1e9, (now_ms() - t0) / 1e3);

        // activations
        x = dfill(D, 0); x2 = dfill(D, 0); xn = dfill(D, 0);
        qkvz = dfill(QKV_DIM + Z_DIM, 0); ab = dfill(2 * NV, 0); attn_o = dfill(Z_DIM, 0);
        qkv_a = dfill(Q_ROWS + 2 * HD * NKV, 0);
        part = dfill((size_t) NQ * (max_ctx / ATT_CHUNK + 1) * (HD + 2), 0);
        att = dfill(HD * NQ, 0); rlog = dfill(N_EXPERT + 1, 0); shgu = dfill(2 * FF_SH, 0);
        h = dfill(K_USED * FF_EXP + FF_SH, 0); wts = dfill(K_USED + 1, 0); logits = dfill(VOCAB, 0);
        ids = (int32_t *) dmalloc(K_USED * 4); tok = (int32_t *) dmalloc(4);
        hio = sycl::malloc_host<HostIO>(1, q);
        x_host = sycl::malloc_host<float>(D, q);

        if (mp) {   // MTP block: one full-attention layer + MoE (all 256 experts resident)
            Pack & m = *mp;
            auto up = [&](const std::vector<std::string> & names) { Pack * keep = &pk; (void) keep; DMat d; const PackEntry & e0 = m.get(names[0]);
                d.f = (Fmt) e0.fmt; d.cols = e0.cols; d.rb = row_bytes(d.f, d.cols); size_t total = 0;
                for (auto & n : names) { const PackEntry & e = m.get(n); d.rows += e.rows; total += e.bytes; }
                d.w = dmalloc(total); size_t off = 0;
                for (auto & n : names) { const PackEntry & e = m.get(n); q.memcpy(d.w + off, m.base + e.off, e.bytes).wait(); off += e.bytes; }
                return d; };
            auto upf = [&](const std::string & n) { const PackEntry & e = m.get(n); float * pp = (float *) dmalloc(e.bytes); q.memcpy(pp, m.base + e.off, e.bytes).wait(); return pp; };
            mtpL.recr = false;
            mtpL.attn_norm = upf("mtp.attn_norm"); mtpL.post_norm = upf("mtp.post_norm");
            mtpL.router = up({"mtp.router", "mtp.sh_gate"});
            mtpL.sh_gu = up({"mtp.sh_gate_proj", "mtp.sh_up"});
            mtpL.sh_down = up({"mtp.sh_down"});
            mtpL.wqkv = up({"mtp.wq", "mtp.wk", "mtp.wv"});
            mtpL.wo = up({"mtp.wo"});
            mtpL.q_norm = upf("mtp.q_norm"); mtpL.k_norm = upf("mtp.k_norm");
            mtpL.kc = (int8_t *) dmalloc(kv_sz());
            mtpL.vc = (int8_t *) dmalloc(kv_sz());
            mtpL.ks = (sycl::half *) dmalloc(kvs_sz() * 2);
            mtpL.vs = (sycl::half *) dmalloc(kvs_sz() * 2);
            eh_proj = up({"mtp.eh_proj"});
            enorm = upf("mtp.enorm"); hnorm = upf("mtp.hnorm"); head_norm = upf("mtp.head_norm");
            const PackEntry & ee = m.get("mtp.experts");
            mtp_arena = dmalloc(ee.bytes);
            q.memcpy(mtp_arena, m.base + ee.off, ee.bytes).wait();
            mtp_slots.resize(N_EXPERT); for (int e = 0; e < N_EXPERT; ++e) mtp_slots[e] = e;
            mtp_hexp.assign(N_EXPERT, nullptr);
            HPREV = dfill((size_t) T_MAX * D, 0); HLAST = dfill((size_t) D * n_slots, 0);
            HPEND = dfill((size_t) 2 * D * n_slots, 0);
            has_mtp = true;
            printf("MTP block loaded; device total %.2f GB\n", dev_bytes / 1e9);
        }
        // expert residency: rank (layer, expert) by routing counts, the top `resident` go to VRAM
        std::vector<int64_t> cnt(N_LAYER * N_EXPERT, 0);
        for (auto & sp : stats) {
            FILE * f = fopen(sp.c_str(), "rb");
            if (!f) { fprintf(stderr, "stats %s missing, uniform ranking\n", sp.c_str()); continue; }
            int16_t il; int32_t nt;
            std::vector<uint8_t> buf;
            while (fread(&il, 2, 1, f) == 1 && fread(&nt, 4, 1, f) == 1) {
                buf.resize((size_t) nt * K_USED);
                if (fread(buf.data(), 1, buf.size(), f) != buf.size()) break;
                for (uint8_t e : buf) cnt[il * N_EXPERT + e]++;
            }
            fclose(f);
        }
        std::vector<int> order(N_LAYER * N_EXPERT);
        for (size_t i = 0; i < order.size(); ++i) order[i] = (int) i;
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return cnt[a] > cnt[b]; });
        resident = std::min(resident, N_LAYER * N_EXPERT);
        slot_h.assign(N_LAYER * N_EXPERT, -1);
        host_exp.assign(N_LAYER * N_EXPERT, nullptr);
        arena = dmalloc((size_t) resident * exp_bytes());
        for (int s = 0; s < resident; ++s) {
            const int k = order[s];
            slot_h[k] = s;
            const uint8_t * src = pk.ptr("blk." + std::to_string(k / N_EXPERT) + ".experts") + (size_t) (k % N_EXPERT) * exp_bytes();
            q.memcpy(arena + (size_t) s * exp_bytes(), src, exp_bytes());
            if (s % 256 == 255) q.wait();
        }
        q.wait();
        const int n_host = N_LAYER * N_EXPERT - resident;
        const size_t hb = ((size_t) n_host * exp_bytes() + (2 << 20)) & ~(size_t) ((2 << 20) - 1);
        host_arena = (uint8_t *) std::aligned_alloc(2 << 20, std::max<size_t>(hb, 2 << 20));
        madvise(host_arena, hb, MADV_HUGEPAGE);
        for (int s = resident, j = 0; s < N_LAYER * N_EXPERT; ++s, ++j) {
            const int k = order[s];
            const uint8_t * src = pk.ptr("blk." + std::to_string(k / N_EXPERT) + ".experts") + (size_t) (k % N_EXPERT) * exp_bytes();
            uint8_t * dst = host_arena + (size_t) j * exp_bytes();
            memcpy(dst, src, exp_bytes());
            host_exp[k] = dst;
        }
        slot_d = (int32_t *) dmalloc(slot_h.size() * 4);
        q.memcpy(slot_d, slot_h.data(), slot_h.size() * 4).wait();
        printf("experts: %d in VRAM (%.2f GB), %d in RAM (%.2f GB); device total %.2f GB; load %.1f s\n", resident,
               resident * exp_bytes() / 1e9, n_host, n_host * exp_bytes() / 1e9, dev_bytes / 1e9, (now_ms() - t0) / 1e3);
        pool.start(getenv("QE_THREADS") ? atoi(getenv("QE_THREADS")) : 12);
        pool.out = hio->cpu_out;
        X = dfill((size_t) T_MAX * D, 0); X2 = dfill((size_t) T_MAX * D, 0); XN = dfill((size_t) T_MAX * D, 0);
        QZ = dfill((size_t) T_MAX * (QKV_DIM + Z_DIM), 0); AB = dfill((size_t) T_MAX * 2 * NV, 0); AO = dfill((size_t) T_MAX * Z_DIM, 0);
        QA = dfill((size_t) T_MAX * (Q_ROWS + 2 * HD * NKV), 0); ATT = dfill((size_t) T_MAX * HD * NQ, 0);
        RL = dfill((size_t) T_MAX * (N_EXPERT + 1), 0); SHGU = dfill((size_t) T_MAX * 2 * FF_SH, 0); SHH = dfill((size_t) T_MAX * FF_SH, 0);
        SD = dfill((size_t) T_MAX * D, 0); HP = dfill((size_t) T_MAX * K_USED * FF_EXP, 0); YP = dfill((size_t) T_MAX * K_USED * D, 0);
        WB = dfill((size_t) T_MAX * K_USED, 0); SHG = dfill(T_MAX, 0);
        IDB = (int32_t *) dmalloc((size_t) T_MAX * K_USED * 4); PAIRS = (int32_t *) dmalloc((size_t) T_MAX * K_USED * 4);
        OFF = (int32_t *) dmalloc((N_EXPERT + 1) * 4); BSLOT = (int32_t *) dmalloc(N_EXPERT * 4); BSLOT2 = (int32_t *) dmalloc(N_EXPERT * 4);
        stage = dmalloc((size_t) STAGE * exp_bytes());
        WF = (sycl::half *) dmalloc((size_t) (Q_ROWS + 2 * HD * NKV) * D * 2);
        XF = (sycl::half *) dmalloc((size_t) T_MAX * Z_DIM * 2);
        X_host = sycl::malloc_host<float>((size_t) T_MAX * D, q);
        HB = dfill((size_t) T_MAX * D, 0); LOGB = dfill((size_t) 8 * VOCAB, 0);
        partm = dfill((size_t) NQ * 8 * (max_ctx / ATT_CHUNK + 2) * (HD + 2), 0);
        cpu_outm = sycl::malloc_host<float>((size_t) 8 * D, q);
        XH = (sycl::half *) dmalloc((size_t) 8 * Z_DIM * 2);
        QHb = (sycl::half *) dmalloc((size_t) T_MAX * NQ * HD * 2);
        q.memset(QHb, 0, (size_t) T_MAX * NQ * HD * 2).wait();
        LSUM = dfill((size_t) T_MAX * NQ, 1.f);
        XGp = (sycl::half *) dmalloc((size_t) PAD_ROWS * D * 2);
        HHp = (sycl::half *) dmalloc((size_t) PAD_ROWS * FF_EXP * 2);
        HGUp = (float *) dmalloc((size_t) PAD_ROWS * 2 * FF_EXP * 4);
        YDp = (float *) dmalloc((size_t) PAD_ROWS * D * 4);
        PADOFF = (int32_t *) dmalloc((N_EXPERT + 1) * 4);
        PADMAP = (int32_t *) dmalloc((size_t) PAD_ROWS * 4);
        printf("device total with XMX MoE buffers: %.2f GB\n", dev_bytes / 1e9);
        TOKB = (int32_t *) dmalloc(4 * T_MAX);
        BT = (int32_t *) dmalloc((size_t) n_slots * max_blocks * 4);
        q.memcpy(BT, bt_h.data(), bt_h.size() * 4).wait();
        RPOS = (int32_t *) dmalloc((size_t) 3 * T_MAX * 4);
        ids_host = sycl::malloc_host<int32_t>((size_t) T_MAX * K_USED, q);
        xn_host = sycl::malloc_host<float>((size_t) T_MAX * D, q);
        wb_host = sycl::malloc_host<float>((size_t) T_MAX * K_USED, q);
        ypc_h = sycl::malloc_host<float>((size_t) T_MAX * K_USED * D, q);
        pcold_h = sycl::malloc_host<uint8_t>((size_t) T_MAX * K_USED, q);
        xq_b = (int8_t *) std::aligned_alloc(64, (size_t) T_MAX * D);
        xs_b = (float *) std::aligned_alloc(64, (size_t) T_MAX * (D / 32) * 4);
        hcold = (float *) std::aligned_alloc(64, (size_t) T_MAX * K_USED * FF_EXP * 4);
        hq_b = (int8_t *) std::aligned_alloc(64, (size_t) T_MAX * K_USED * FF_EXP);
        hs_b = (float *) std::aligned_alloc(64, (size_t) T_MAX * K_USED * (FF_EXP / 32) * 4);
        printf("device total with prefill buffers: %.2f GB\n", dev_bytes / 1e9);
    }

    // optional phase timing for prefill (QE_PROF=1): waits on the queue at each phase boundary
    bool prof = getenv("QE_PROF") != nullptr;
    std::map<std::string, double> pt;
    double pt_last = 0;
    void tick(const char * name) { if (!prof) return; q.wait(); const double t = now_ms(); pt[name] += t - pt_last; pt_last = t; }
    void prof_report() { if (!prof) return; for (auto & [k, v] : pt) printf("  %-14s %8.1f ms\n", k.c_str(), v); pt.clear(); }

    // ---------------- batched forward: T tokens at positions pos0 .. pos0+T-1 ----------------
    // snap >= 0: keep the recurrent states after token `snap` (restore_states() rolls back to it)
    // all_logits: logits + greedy token for every row (LOGB / TOKB), else only the last row (logits / tok); HB = final-normed hidden rows
    void prefill(const int * toks, int T, int pos0) { forward(toks, T, pos0, -1, false); }
    void restore_states() {
        for (auto & l : L) if (l.recr) {
            q.memcpy(l.conv_state + sq * CONV_SZ, l.conv_bak + sq * CONV_SZ, CONV_SZ * 4);
            q.memcpy(l.ssm_state + sq * SSM_SZ, l.ssm_bak + sq * SSM_SZ, SSM_SZ * 4);
        }
    }
    const float * embd_in = nullptr;   // when set, the next embed_rows() copies these host rows instead of looking up tokens
    void embed_rows(const int * toks, int T) {
        if (embd_in) { q.memcpy(X, embd_in, (size_t) T * D * 4); return; }
        for (int t = 0; t < T; ++t) {
            const uint8_t * row = embd + (size_t) toks[t] * row_bytes(F_Q8, D);
            const sycl::half * sc = (const sycl::half *) (row + D);
            for (int i = 0; i < D; ++i) X_host[(size_t) t * D + i] = (float) ((const int8_t *) row)[i] * (float) sc[i / 32];
        }
        q.memcpy(X, X_host, (size_t) T * D * 4);
    }
    void forward(const int * toks, int T, int pos0, int snap, bool all_logits) {
        ensure_kv(sq, pos0 + T);
        if (prof) { q.wait(); pt_last = now_ms(); }
        embed_rows(toks, T);
        tick("embed");
        for (int il = 0; il < N_LAYER; ++il) {
            Layer & l = L[il];
            rmsnorm_rows(q, X, l.attn_norm, XN, T);
            if (l.recr) {
                bgemm(l.qkv, XN, T, QZ, QKV_DIM + Z_DIM, 0);
                bgemm(l.z, XN, T, QZ, QKV_DIM + Z_DIM, QKV_DIM);
                bgemm(l.ab, XN, T, AB, 2 * NV, 0);
                tick("dn_proj");
                dn_recurrent(l, T, snap);
                tick("dn_conv_delta");
                bgemm(l.ssm_out, AO, T, X2, D, 0, X);
                tick("dn_out");
            } else {
                attn_block(l, T, pos0);
            }
            moe_batched(l, il * N_EXPERT, arena, slot_h.data(), host_exp.data(), T);
        }
        rmsnorm_rows(q, X, out_norm, HB, T);
        if (all_logits) {
            gemm(q, lm, HB, T, LOGB, VOCAB, 0);
            float * lg = LOGB;
            int32_t * tb = TOKB;
            q.parallel_for(sycl::nd_range<1>((size_t) T * 1024, 1024), [=](sycl::nd_item<1> it) {
                const int t = it.get_group(0);
                const float * row = lg + (size_t) t * VOCAB;
                float best = -1e30f; int bi = 0;
                for (int i = it.get_local_id(0); i < VOCAB; i += 1024) if (row[i] > best) { best = row[i]; bi = i; }
                const float mx = sycl::reduce_over_group(it.get_group(), best, sycl::maximum<float>());
                const int who = sycl::reduce_over_group(it.get_group(), best == mx ? bi : VOCAB, sycl::minimum<int>());
                if (it.get_local_id(0) == 0) tb[t] = who;
            });
            q.wait();
            return;
        }
        // logits for the last token only
        gemv(q, lm, HB + (size_t) (T - 1) * D, logits);
        {
            float * lg = logits;
            int32_t * tk = tok;
            q.parallel_for(sycl::nd_range<1>(1024, 1024), [=](sycl::nd_item<1> it) {
                float best = -1e30f; int bi = 0;
                for (int i = it.get_local_id(0); i < VOCAB; i += 1024) if (lg[i] > best) { best = lg[i]; bi = i; }
                const float mx = sycl::reduce_over_group(it.get_group(), best, sycl::maximum<float>());
                const int who = sycl::reduce_over_group(it.get_group(), best == mx ? bi : VOCAB, sycl::minimum<int>());
                if (it.get_local_id(0) == 0) tk[0] = who;
            });
        }
        q.wait();
    }

    void dn_recurrent(Layer & l, int T, int snap, int row0 = 0) {
                {   // causal conv over the chunk, per channel, carrying the conv state
                    float * cs = l.conv_state + sq * CONV_SZ, * cw = l.conv_w, * qz = QZ + (size_t) row0 * (QKV_DIM + Z_DIM);
                    float * cb = l.conv_bak + sq * CONV_SZ;
                    q.parallel_for(sycl::range<1>(QKV_DIM), [=](sycl::id<1> idc) {
                        const int c = idc[0];
                        float * st = cs + c * (CONV_K - 1);
                        float s0 = st[0], s1 = st[1], s2 = st[2];
                        const float w0 = cw[c * CONV_K], w1 = cw[c * CONV_K + 1], w2 = cw[c * CONV_K + 2], w3 = cw[c * CONV_K + 3];
                        for (int t = 0; t < T; ++t) {
                            float & io = qz[(size_t) t * (QKV_DIM + Z_DIM) + c];
                            const float in = io;
                            const float acc = w0 * s0 + w1 * s1 + w2 * s2 + w3 * in;
                            s0 = s1; s1 = s2; s2 = in;
                            io = acc / (1.f + sycl::exp(-acc));
                            if (t == snap) { cb[c * 3] = s0; cb[c * 3 + 1] = s1; cb[c * 3 + 2] = s2; }
                        }
                        st[0] = s0; st[1] = s1; st[2] = s2;
                    });
                    q.parallel_for(sycl::nd_range<1>((size_t) T * 2 * NK * DK, DK), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                        const int g = it.get_group(0), t = g / (2 * NK), hh = g % (2 * NK);
                        float & v = qz[(size_t) t * (QKV_DIM + Z_DIM) + hh * DK + it.get_local_id(0)];
                        const float ss = sycl::reduce_over_group(it.get_group(), v * v, sycl::plus<float>());
                        v *= sycl::rsqrt(ss + RMS_EPS);
                    });
                }
                {   // gated delta rule over the chunk; the head state lives in registers
                    float * S = l.ssm_state + sq * SSM_SZ, * nw = l.ssm_norm, * sa = l.ssm_a, * dtb = l.dt_bias;
                    float * qz = QZ + (size_t) row0 * (QKV_DIM + Z_DIM), * abp = AB + (size_t) row0 * 2 * NV, * ao = AO + (size_t) row0 * Z_DIM;
                    float * SB = l.ssm_bak + sq * SSM_SZ;
                    q.submit([&](sycl::handler & hd) {
                        sycl::local_accessor<float, 1> red(4 * DK, hd);
                        hd.parallel_for(sycl::nd_range<1>(NV * 4 * DK, 4 * DK), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                            const int hv = it.get_group(0), lid = it.get_local_id(0);
                            const int j = lid % DK, p = lid / DK;
                            const int hk = hv % NK;
                            float * Sh = S + (size_t) hv * DK * DK;
                            float s[32];
#pragma unroll
                            for (int i = 0; i < 32; ++i) s[i] = Sh[(size_t) (p * 32 + i) * DK + j];
                            for (int t = 0; t < T; ++t) {
                                const float * row = qz + (size_t) t * (QKV_DIM + Z_DIM);
                                const float * qv = row + hk * DK;
                                const float * kv = row + DK * NK + hk * DK;
                                const float vj = row[2 * DK * NK + hv * DK + j];
                                const float a = abp[t * 2 * NV + hv] + dtb[hv];
                                const float sp = a > 20.f ? a : sycl::log1p(sycl::exp(a));
                                const float g = sycl::exp(sp * sa[hv]);
                                const float beta = 1.f / (1.f + sycl::exp(-abp[t * 2 * NV + NV + hv]));
                                float part = 0.f;
#pragma unroll
                                for (int i = 0; i < 32; ++i) { s[i] *= g; part += s[i] * kv[p * 32 + i]; }
                                red[p * DK + j] = part;
                                it.barrier(sycl::access::fence_space::local_space);
                                const float kvj = red[j] + red[DK + j] + red[2 * DK + j] + red[3 * DK + j];
                                it.barrier(sycl::access::fence_space::local_space);
                                const float delta = beta * (vj - kvj);
                                part = 0.f;
#pragma unroll
                                for (int i = 0; i < 32; ++i) { s[i] += kv[p * 32 + i] * delta; part += s[i] * qv[p * 32 + i]; }
                                red[p * DK + j] = part;
                                it.barrier(sycl::access::fence_space::local_space);
                                const float oj = (red[j] + red[DK + j] + red[2 * DK + j] + red[3 * DK + j]) * sycl::rsqrt((float) DK);
                                const float ss = sycl::reduce_over_group(it.get_group(), p == 0 ? oj * oj : 0.f, sycl::plus<float>());
                                const float zz = row[QKV_DIM + hv * DK + j];
                                if (p == 0) ao[(size_t) t * Z_DIM + hv * DK + j] = oj * sycl::rsqrt(ss / DK + RMS_EPS) * nw[j] * (zz / (1.f + sycl::exp(-zz)));
                                if (t == snap) {
#pragma unroll
                                    for (int i = 0; i < 32; ++i) SB[(size_t) hv * DK * DK + (size_t) (p * 32 + i) * DK + j] = s[i];
                                }
                                it.barrier(sycl::access::fence_space::local_space);
                            }
#pragma unroll
                            for (int i = 0; i < 32; ++i) Sh[(size_t) (p * 32 + i) * DK + j] = s[i];
                        });
                    });
                }
    }
    void rope_kv(Layer & l, int T, int pos0, int row0 = 0) {
        constexpr int QL = Q_ROWS + 2 * HD * NKV;
        const int rdelta = rope_delta[sq];
        const int32_t * rp = use_rpos ? RPOS : nullptr;

                {
                    float * qa = QA + (size_t) row0 * QL, * qn = l.q_norm, * kn = l.k_norm;
                    int8_t * kc = l.kc, * vc = l.vc;
                    const int32_t * bt = BT + (size_t) sq * max_blocks;
                    sycl::half * ks = l.ks, * vs = l.vs;
                    q.parallel_for(sycl::nd_range<1>((size_t) T * (NQ + NKV) * HD, HD), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                        const int g = it.get_group(0), t = g / (NQ + NKV), hh = g % (NQ + NKV), d = it.get_local_id(0);
                        const int pos = pos0 + t;
                        int pt = pos - rdelta, ph = pt, pw = pt;
                        if (rp) { pt = rp[(row0 + t) * 3]; ph = rp[(row0 + t) * 3 + 1]; pw = rp[(row0 + t) * 3 + 2]; }
                        const bool isq = hh < NQ;
                        float * base = qa + (size_t) t * QL;
                        float * v = isq ? base + hh * 2 * HD : base + Q_ROWS + (hh - NQ) * HD;
                        const float val = v[d];
                        const float ss = sycl::reduce_over_group(it.get_group(), val * val, sycl::plus<float>());
                        float y = val * sycl::rsqrt(ss / HD + RMS_EPS) * (isq ? qn[d] : kn[d]);
                        it.barrier();
                        v[d] = y;
                        it.barrier();
                        if (d < N_ROT) {
                            const int i = d % (N_ROT / 2);
                            const float th = (float) (i % 3 == 1 ? ph : i % 3 == 2 ? pw : pt) * sycl::pow(ROPE_BASE, -2.f * i / N_ROT);
                            const float cs = sycl::cos(th), sn = sycl::sin(th);
                            y = d < N_ROT / 2 ? v[d] * cs - v[d + N_ROT / 2] * sn : v[d] * cs + v[d - N_ROT / 2] * sn;
                        }
                        it.barrier();
                        if (isq) {
                            v[d] = y;
                        } else {
                            const int kh = hh - NQ;
                            const auto sg = it.get_sub_group();
                            const float vv = base[Q_ROWS + NKV * HD + kh * HD + d];
                            const float ka = sycl::reduce_over_group(sg, sycl::fabs(y), sycl::maximum<float>());
                            const float va = sycl::reduce_over_group(sg, sycl::fabs(vv), sycl::maximum<float>());
                            const float kd = ka / 127.f, vd = va / 127.f;
                            const size_t o = ((size_t) (bt[(pos) >> 8] * 256 + ((pos) & 255)) * NKV + kh) * HD + d;
                            kc[o] = (int8_t) sycl::rint(kd > 0 ? y / kd : 0.f);
                            vc[o] = (int8_t) sycl::rint(vd > 0 ? vv / vd : 0.f);
                            if (d % 16 == 0) { ks[o / 16] = sycl::half(kd); vs[o / 16] = sycl::half(vd); }
                        }
                    });
                }
    }
    void topk_rows(int T) {
        {   // top-8 per token (sub-group per token), shared gate
            float * rl = RL, * wb = WB, * shg = SHG;
            int32_t * idb = IDB;
            q.parallel_for(sycl::nd_range<1>((size_t) T * SG, SG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                const auto sg = it.get_sub_group();
                const int t = it.get_group(0), lane = sg.get_local_id()[0];
                const float * r = rl + (size_t) t * (N_EXPERT + 1);
                float v[N_EXPERT / SG];
                for (int i = 0; i < N_EXPERT / SG; ++i) v[i] = r[i * SG + lane];
                float sel[K_USED];
                for (int k = 0; k < K_USED; ++k) {
                    float best = -1e30f; int bi = 0;
                    for (int i = 0; i < N_EXPERT / SG; ++i) if (v[i] > best) { best = v[i]; bi = i; }
                    const float mx = sycl::reduce_over_group(sg, best, sycl::maximum<float>());
                    const int who = sycl::reduce_over_group(sg, best == mx ? lane : SG, sycl::minimum<int>());
                    const int bidx = sycl::group_broadcast(sg, bi, who);
                    if (lane == who) v[bi] = -1e30f;
                    sel[k] = mx;
                    if (lane == 0) idb[t * K_USED + k] = bidx * SG + who;
                }
                if (lane == 0) {
                    float s = 0.f;
                    for (int k = 0; k < K_USED; ++k) s += sycl::exp(sel[k] - sel[0]);
                    for (int k = 0; k < K_USED; ++k) wb[t * K_USED + k] = sycl::exp(sel[k] - sel[0]) / s;
                    shg[t] = 1.f / (1.f + sycl::exp(-r[N_EXPERT]));
                }
            });
        }
    }
    void shared_act(int T) {
        {   // shared expert activation
            float * sgu = SHGU, * shh = SHH;
            q.parallel_for(sycl::range<1>((size_t) T * FF_SH), [=](sycl::id<1> i) {
                const int t = i[0] / FF_SH, r = i[0] % FF_SH;
                const float g = sgu[(size_t) t * 2 * FF_SH + r], u = sgu[(size_t) t * 2 * FF_SH + FF_SH + r];
                shh[i[0]] = g / (1.f + sycl::exp(-g)) * u;
            });
        }
    }

    // ---------------- NT tokens through the decode-style kernels (speculative verification) ----------------
    float * partm = nullptr, * cpu_outm = nullptr;
    // XMX MoE (prefill): padded per-expert rows
    sycl::half * XGp = nullptr, * HHp = nullptr;
    float * HGUp = nullptr, * YDp = nullptr;
    int32_t * PADOFF = nullptr, * PADMAP = nullptr;
    static constexpr int PAD_ROWS = T_MAX * K_USED + 8 * N_EXPERT;
    sycl::half * XH = nullptr;
    sycl::half * QHb = nullptr;        // flash attention: f16 queries
    float * LSUM = nullptr;
    const sycl::half * to_h(const float * src, size_t n) {
        sycl::half * dst = XH;
        q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) { dst[i[0]] = sycl::half(src[i[0]]); });
        return dst;
    }
    template <int NT>
    void decode_multi(const int * toks, const SeqGroup * grp, int ng) {
        const int sq_saved = sq;
        for (int g = 0; g < ng; ++g) ensure_kv(grp[g].seq, grp[g].pos0 + grp[g].n);
        constexpr int QL = Q_ROWS + 2 * HD * NKV;
        embed_rows(toks, NT);
        for (int il = 0; il < N_LAYER; ++il) {
            Layer & l = L[il];
            rmsnorm_rows(q, X, l.attn_norm, XN, NT);
            if (l.recr) {
                const sycl::half * xh = to_h(XN, (size_t) NT * D);
                gemv_nt<NT>(q, l.qkv, XN, QZ, QKV_DIM + Z_DIM, nullptr, xh);
                gemv_nt<NT>(q, l.z, XN, QZ + QKV_DIM, QKV_DIM + Z_DIM, nullptr, xh);
                gemv_nt<NT>(q, l.ab, XN, AB, 2 * NV);
                tick("dm_dn_proj");
                for (int g = 0, r0 = 0; g < ng; r0 += grp[g].n, ++g) { sq = grp[g].seq; dn_recurrent(l, grp[g].n, grp[g].snap, r0); }
                tick("dm_dn_rec");
                gemv_nt<NT>(q, l.ssm_out, AO, X2, D, X, to_h(AO, (size_t) NT * Z_DIM));
                tick("dm_dn_out");
            } else {
                gemv_nt<NT>(q, l.wqkv, XN, QA, QL, nullptr, to_h(XN, (size_t) NT * D));
                for (int g = 0, row0 = 0; g < ng; row0 += grp[g].n, ++g) {
                sq = grp[g].seq;
                const int GN = grp[g].n, pos0 = grp[g].pos0;
                rope_kv(l, GN, pos0, row0);
                const int n_chunks = (pos0 + GN + ATT_CHUNK - 1) / ATT_CHUNK;
                if (pos0 + GN >= gqa_min_ctx) {
                    const int8_t * kc = l.kc, * vc = l.vc;
                    const int32_t * bt = BT + (size_t) sq * max_blocks;
                    const sycl::half * ks = l.ks, * vs = l.vs;
                    float * qa = QA + (size_t) row0 * QL, * pp0 = partm;
                    q.parallel_for(sycl::nd_range<2>({(size_t) NKV, (size_t) GN * n_chunks * 128}, {1, 128}), [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG)]] {
                        const int kh = it.get_global_id(0), g1 = it.get_group(1);
                        const int t = g1 / n_chunks, ch = g1 % n_chunks;
                        const auto sg = it.get_sub_group();
                        const int s_id = sg.get_group_id()[0], lane = sg.get_local_id()[0];
                        const int hq = kh * (NQ / NKV) + s_id;
                        const float * qrow = qa + (size_t) t * QL + hq * 2 * HD;
                        float qd[16], acc[16];
                        for (int i = 0; i < 16; ++i) { qd[i] = qrow[lane * 16 + i] * 0.0625f; acc[i] = 0.f; }
                        float m = -1e30f, sum = 0.f;
                        const int n_vis = pos0 + t + 1;
                        const int t0 = ch * ATT_CHUNK, t1 = sycl::min(n_vis, t0 + ATT_CHUNK);
                        const size_t base = (size_t) bt[ch] * 256;
                        for (int p = t0; p < t1; ++p) {
                            const size_t ko = ((base + (p - t0)) * NKV + kh) * HD + lane * 16;
                            const sycl::vec<int8_t, 16> kk = *(const sycl::vec<int8_t, 16> *) (kc + ko);
                            float dd = 0.f;
#pragma unroll
                            for (int i = 0; i < 16; ++i) dd += qd[i] * (float) kk[i];
                            dd = sycl::reduce_over_group(sg, dd * (float) ks[ko / 16], sycl::plus<float>());
                            const float mn = sycl::max(m, dd);
                            const float c = sycl::exp(m - mn), e = sycl::exp(dd - mn);
                            sum = sum * c + e;
                            const sycl::vec<int8_t, 16> vv = *(const sycl::vec<int8_t, 16> *) (vc + ko);
                            const float vdq = (float) vs[ko / 16];
#pragma unroll
                            for (int i = 0; i < 16; ++i) acc[i] = acc[i] * c + e * vdq * (float) vv[i];
                            m = mn;
                        }
                        float * pp = pp0 + (((size_t) hq * GN + t) * n_chunks + ch) * (HD + 2);
                        for (int i = 0; i < 16; ++i) pp[lane * 16 + i] = acc[i];
                        if (lane == 0) { pp[HD] = m; pp[HD + 1] = sum; }
                    });
                } else {
                    const int8_t * kc = l.kc, * vc = l.vc;
                    const int32_t * bt = BT + (size_t) sq * max_blocks;
                    const sycl::half * ks = l.ks, * vs = l.vs;
                    float * qa = QA + (size_t) row0 * QL, * pp0 = partm;
                    q.submit([&](sycl::handler & hd) {
                        sycl::local_accessor<float, 1> sm(8 * (HD + 2), hd);
                        hd.parallel_for(sycl::nd_range<2>({(size_t) NQ, (size_t) GN * n_chunks * 128}, {1, 128}),
                                        [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG)]] {
                            const int hq = it.get_global_id(0), g1 = it.get_group(1);
                            const int t = g1 / n_chunks, ch = g1 % n_chunks;
                            const auto sg = it.get_sub_group();
                            const int s_id = sg.get_group_id()[0], lane = sg.get_local_id()[0];
                            const int kh = hq / (NQ / NKV);
                            const float * qrow = qa + (size_t) t * QL + hq * 2 * HD;
                            float qd[16], acc[16];
                            for (int i = 0; i < 16; ++i) { qd[i] = qrow[lane * 16 + i] * 0.0625f; acc[i] = 0.f; }
                            float m = -1e30f, sum = 0.f;
                            const int n_vis = pos0 + t + 1;
                            const int t0 = ch * ATT_CHUNK, t1 = sycl::min(n_vis, t0 + ATT_CHUNK);
                            for (int p = t0 + s_id; p < t1; p += 8) {
                                const size_t ko = ((size_t) (bt[(p) >> 8] * 256 + ((p) & 255)) * NKV + kh) * HD + lane * 16;
                                const sycl::vec<int8_t, 16> kk = *(const sycl::vec<int8_t, 16> *) (kc + ko);
                                float dd = 0.f;
#pragma unroll
                                for (int i = 0; i < 16; ++i) dd += qd[i] * (float) kk[i];
                                dd = sycl::reduce_over_group(sg, dd * (float) ks[ko / 16], sycl::plus<float>());
                                const float mn = sycl::max(m, dd);
                                const float c = sycl::exp(m - mn), e = sycl::exp(dd - mn);
                                sum = sum * c + e;
                                const sycl::vec<int8_t, 16> vv = *(const sycl::vec<int8_t, 16> *) (vc + ko);
                                const float vdq = (float) vs[ko / 16];
#pragma unroll
                                for (int i = 0; i < 16; ++i) acc[i] = acc[i] * c + e * vdq * (float) vv[i];
                                m = mn;
                            }
                            float * my = &sm[s_id * (HD + 2)];
                            for (int i = 0; i < 16; ++i) my[lane * 16 + i] = acc[i];
                            if (lane == 0) { my[HD] = m; my[HD + 1] = sum; }
                            it.barrier(sycl::access::fence_space::local_space);
                            const int lid = it.get_local_id(1);
                            float M = -1e30f;
                            for (int k = 0; k < 8; ++k) M = sycl::max(M, sm[k * (HD + 2) + HD]);
                            float * pp = pp0 + (((size_t) hq * GN + t) * n_chunks + ch) * (HD + 2);
                            for (int dd = lid; dd < HD; dd += 128) {
                                float a = 0.f;
                                for (int k = 0; k < 8; ++k) a += sm[k * (HD + 2) + dd] * sycl::exp(sm[k * (HD + 2) + HD] - M);
                                pp[dd] = a;
                            }
                            if (lid == 0) {
                                float Ssum = 0.f;
                                for (int k = 0; k < 8; ++k) Ssum += sm[k * (HD + 2) + HD + 1] * sycl::exp(sm[k * (HD + 2) + HD] - M);
                                pp[HD] = M; pp[HD + 1] = Ssum;
                            }
                        });
                    });
                }
                {
                    float * qa = QA + (size_t) row0 * QL, * pp0 = partm, * at = ATT + (size_t) row0 * HD * NQ;
                    q.parallel_for(sycl::nd_range<1>((size_t) NQ * GN * HD, HD), [=](sycl::nd_item<1> it) {
                        const int g = it.get_group(0), hq = g / GN, t = g % GN, d = it.get_local_id(0);
                        const float * base = pp0 + ((size_t) hq * GN + t) * n_chunks * (HD + 2);
                        float m = -1e30f;
                        for (int c = 0; c < n_chunks; ++c) m = sycl::max(m, base[(size_t) c * (HD + 2) + HD]);
                        float s2 = 0.f, a = 0.f;
                        for (int c = 0; c < n_chunks; ++c) {
                            const float * pp = base + (size_t) c * (HD + 2);
                            const float w = sycl::exp(pp[HD] - m);
                            s2 += pp[HD + 1] * w;
                            a += pp[d] * w;
                        }
                        const float gt = qa[(size_t) t * QL + hq * 2 * HD + HD + d];
                        at[(size_t) t * HD * NQ + hq * HD + d] = a / s2 / (1.f + sycl::exp(-gt));
                    });
                }
                }
                gemv_nt<NT>(q, l.wo, ATT, X2, D, X, to_h(ATT, (size_t) NT * HD * NQ));
                tick("dm_attn");
            }
            // ---- MoE for NT tokens ----
            rmsnorm_rows(q, X2, l.post_norm, XN, NT);
            gemv_nt<NT>(q, l.router, XN, RL, N_EXPERT + 1);
            gemv_nt<NT>(q, l.sh_gu, XN, SHGU, 2 * FF_SH, nullptr, to_h(XN, (size_t) NT * D));
            topk_rows(NT);
            q.memcpy(ids_host, IDB, NT * K_USED * 4);
            q.memcpy(wb_host, WB, NT * K_USED * 4);
            sycl::event e_r = q.memcpy(xn_host, XN, (size_t) NT * D * 4);
            shared_act(NT);
            tick("dm_router_topk");
            {
                const int layer = il;
                const float * xf = XN;
                float * hp = HP;
                const int32_t * idb = IDB, * sl = slot_d;
                const uint8_t * ar = arena;
                const size_t EB = exp_bytes(), GRB = exp_gu_rb();
                q.parallel_for(sycl::nd_range<1>((size_t) NT * K_USED * FF_EXP * SG, 128), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                    const auto sg = it.get_sub_group();
                    const int gid = it.get_global_id(0) / SG, lane = sg.get_local_id()[0];
                    const int p = gid / FF_EXP, rr = gid % FF_EXP;
                    const int slot = sl[layer * N_EXPERT + idb[p]];
                    if (slot < 0) return;
                    const uint8_t * base = ar + (size_t) slot * EB;
                    const float * xr = xf + (size_t) (p / K_USED) * D;
                    float acc[2];
                    for (int m = 0; m < 2; ++m) {
                        const uint8_t * row = base + ((size_t) m * FF_EXP + rr) * GRB;
                        float a = 0.f;
                        for (int b = lane; b < D / 32; b += SG) a += row_blk_dot(F_NL4, row, D, b, xr);
                        acc[m] = sycl::reduce_over_group(sg, a, sycl::plus<float>());
                    }
                    if (lane == 0) hp[(size_t) p * FF_EXP + rr] = acc[0] / (1.f + sycl::exp(-acc[0])) * acc[1];
                });
            }
            tick("dm_gate_up");
            e_r.wait();
            int miss[NT];
            {
                int nm = 0;
                for (int t = 0; t < NT; ++t) {
                    miss[t] = 0;
                    for (int k = 0; k < K_USED; ++k) {
                        const int key = il * N_EXPERT + ids_host[t * K_USED + k];
                        if (slot_h[key] < 0) { pool.exp_ptr[nm] = host_exp[key]; pool.exp_w[nm] = wb_host[t * K_USED + k]; pool.miss_tok[nm] = t; ++nm; ++miss[t]; }
                    }
                    pool.outs[t] = cpu_outm + (size_t) t * D;
                }
                if (nm) {
                    for (int t = 0; t < NT; ++t) if (!miss[t]) miss[t] = 1;   // results are written for every token; mark all as valid
                    pool.n_miss = nm;
                    pool.compute_multi(xn_host, NT);
                    n_miss += nm;
                }
            }
            tick("dm_cpu");
            {
                const int layer = il;
                uint32_t mmask = 0;
                for (int t = 0; t < NT; ++t) if (miss[t]) mmask |= 1u << t;
                const float * cpo = cpu_outm, * hp = HP, * wb = WB, * xr = X2, * shh = SHH, * shg = SHG;
                float * xo = X;
                const int32_t * idb = IDB, * sl = slot_d;
                const uint8_t * ar = arena, * shd = l.sh_down.w;
                const size_t EB = exp_bytes(), GRB = exp_gu_rb(), DRB = exp_d_rb(), shrb = l.sh_down.rb;
                const Fmt shf = l.sh_down.f;
                q.parallel_for(sycl::nd_range<1>((size_t) NT * D * SG, 128), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                    const auto sg = it.get_sub_group();
                    const int gid = it.get_global_id(0) / SG, lane = sg.get_local_id()[0];
                    const int t = gid / D, r = gid % D;
                    float acc = 0.f;
                    for (int k = 0; k < K_USED; ++k) {
                        const int p = t * K_USED + k;
                        const int slot = sl[layer * N_EXPERT + idb[p]];
                        if (slot < 0) continue;
                        const uint8_t * row = ar + (size_t) slot * EB + 2 * (size_t) FF_EXP * GRB + (size_t) r * DRB;
                        acc += wb[p] * row_blk_dot(F_NL4, row, FF_EXP, lane, hp + (size_t) p * FF_EXP);
                    }
                    acc += shg[t] * row_blk_dot(shf, shd + (size_t) r * shrb, FF_SH, lane, shh + (size_t) t * FF_SH);
                    acc = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
                    if (lane == 0) xo[(size_t) t * D + r] = xr[(size_t) t * D + r] + acc + ((mmask >> t) & 1 ? cpo[(size_t) t * D + r] : 0.f);
                });
            }
            tick("dm_down");
        }
        rmsnorm_rows(q, X, out_norm, HB, NT);
        gemv_nt<NT>(q, lm, HB, LOGB, VOCAB, nullptr, to_h(HB, (size_t) NT * D));
        {
            float * lg = LOGB;
            int32_t * tb = TOKB;
            q.parallel_for(sycl::nd_range<1>((size_t) NT * 1024, 1024), [=](sycl::nd_item<1> it) {
                const int t = it.get_group(0);
                const float * row = lg + (size_t) t * VOCAB;
                float best = -1e30f; int bi = 0;
                for (int i = it.get_local_id(0); i < VOCAB; i += 1024) if (row[i] > best) { best = row[i]; bi = i; }
                const float mx = sycl::reduce_over_group(it.get_group(), best, sycl::maximum<float>());
                const int who = sycl::reduce_over_group(it.get_group(), best == mx ? bi : VOCAB, sycl::minimum<int>());
                if (it.get_local_id(0) == 0) tb[t] = who;
            });
        }
        n_tok += NT;
        sq = sq_saved;
    }
    template <int NT>
    void decode_multi(const int * toks, int pos0, int snap) {
        const SeqGroup g{sq, pos0, NT, snap};
        decode_multi<NT>(toks, &g, 1);
    }


    // ---- prompt checkpoints: recurrent state of a slot right after its prompt, kept in host RAM ----
    struct Ckpt { std::vector<float> conv, ssm; std::vector<float> hlast; int rope_delta = 0; size_t len = 0; };
    std::vector<Ckpt> ckpt;
    void ckpt_save(int slot, size_t len) {
        if (ckpt.size() < (size_t) n_slots) ckpt.resize(n_slots);
        Ckpt & c = ckpt[slot];
        int nrec = 0; for (auto & l : L) nrec += l.recr;
        c.conv.resize((size_t) nrec * CONV_SZ); c.ssm.resize((size_t) nrec * SSM_SZ); c.hlast.resize(D);
        int k = 0;
        for (auto & l : L) if (l.recr) {
            q.memcpy(c.conv.data() + (size_t) k * CONV_SZ, l.conv_state + slot * CONV_SZ, CONV_SZ * 4);
            q.memcpy(c.ssm.data() + (size_t) k * SSM_SZ, l.ssm_state + slot * SSM_SZ, SSM_SZ * 4);
            ++k;
        }
        if (HLAST) q.memcpy(c.hlast.data(), HLAST + (size_t) slot * D, D * 4);
        q.wait();
        c.rope_delta = rope_delta[slot];
        c.len = len;
    }
    void ckpt_restore(int slot) {
        Ckpt & c = ckpt[slot];
        int k = 0;
        for (auto & l : L) if (l.recr) {
            q.memcpy(l.conv_state + slot * CONV_SZ, c.conv.data() + (size_t) k * CONV_SZ, CONV_SZ * 4);
            q.memcpy(l.ssm_state + slot * SSM_SZ, c.ssm.data() + (size_t) k * SSM_SZ, SSM_SZ * 4);
            ++k;
        }
        if (HLAST) q.memcpy(HLAST + (size_t) slot * D, c.hlast.data(), D * 4);
        q.wait();
        rope_delta[slot] = c.rope_delta;
    }

    // ---- multi-slot speculative step helpers ----
    // after a slot's prompt: its pending MTP input is the first sampled token with the last prompt hidden row
    void spec_begin(int slot) { q.memcpy(HPEND + (size_t) slot * 2 * D, HLAST + (size_t) slot * D, D * 4).wait(); }
    // draft for slot: MTP over its pending tokens (positions p-n+1 .. p)
    int mtp_draft(int slot, const std::vector<int> & pend, int p) {
        sq = slot;
        q.memcpy(HPREV, HPEND + (size_t) slot * 2 * D, pend.size() * D * 4);
        mtp_forward(pend.data(), HPREV, (int) pend.size(), p - (int) pend.size() + 1);
        return argmax();
    }
    // verify all slots' [y, d] pairs in one pass (tokens grouped by slot)
    void verify_batch(const std::vector<int> & toks, const std::vector<SeqGroup> & groups) {
        const int n = (int) toks.size();
        switch (n) {
            case 1: decode_multi<1>(toks.data(), groups.data(), (int) groups.size()); break;
            case 2: decode_multi<2>(toks.data(), groups.data(), (int) groups.size()); break;
            case 3: decode_multi<3>(toks.data(), groups.data(), (int) groups.size()); break;
            case 4: decode_multi<4>(toks.data(), groups.data(), (int) groups.size()); break;
            case 5: decode_multi<5>(toks.data(), groups.data(), (int) groups.size()); break;
            case 6: decode_multi<6>(toks.data(), groups.data(), (int) groups.size()); break;
            case 7: decode_multi<7>(toks.data(), groups.data(), (int) groups.size()); break;
            case 8: decode_multi<8>(toks.data(), groups.data(), (int) groups.size()); break;
            default: fprintf(stderr, "verify_batch: %d tokens unsupported\n", n); exit(1);
        }
    }
    // after sampling: accepted -> both rows kept (pending = rows r0, r0+1); rejected -> roll the slot back to after row r0
    void accept_update(int slot, int r0, bool accepted) {
        if (!accepted) { const int keep = sq; sq = slot; restore_states(); sq = keep; }
        const int nrows = accepted ? 2 : 1;
        q.memcpy(HPEND + (size_t) slot * 2 * D, HB + (size_t) r0 * D, (size_t) nrows * D * 4);
        q.memcpy(HLAST + (size_t) slot * D, HB + (size_t) (r0 + nrows - 1) * D, D * 4);
    }

    // MTP block for T tokens at pos0..: input token i with the main model's final-normed hidden of position i-1 (rows of hprev)
    // leaves the draft for the position after the last row in `tok` (device) / `logits`
    void mtp_forward(const int * toks, const float * hprev, int T, int pos0) {
        ensure_kv(sq, pos0 + T);
        embed_rows(toks, T);
        rmsnorm_rows(q, X, enorm, XN, T);
        rmsnorm_rows(q, hprev, hnorm, X2, T);
        {
            float * cat = AO, * a = XN, * b = X2;
            q.parallel_for(sycl::range<1>((size_t) T * 2 * D), [=](sycl::id<1> i) {
                const int t = i[0] / (2 * D), c = i[0] % (2 * D);
                cat[i[0]] = c < D ? a[(size_t) t * D + c] : b[(size_t) t * D + c - D];
            });
        }
        gemm(q, eh_proj, AO, T, X, D, 0);
        rmsnorm_rows(q, X, mtpL.attn_norm, XN, T);
        attn_block(mtpL, T, pos0);
        moe_batched(mtpL, 0, mtp_arena, mtp_slots.data(), mtp_hexp.data(), T);
        rmsnorm(q, X + (size_t) (T - 1) * D, head_norm, xn);
        gemv(q, lm, xn, logits);
        float * lg = logits;
        int32_t * tk = tok;
        q.parallel_for(sycl::nd_range<1>(1024, 1024), [=](sycl::nd_item<1> it) {
            float best = -1e30f; int bi = 0;
            for (int i = it.get_local_id(0); i < VOCAB; i += 1024) if (lg[i] > best) { best = lg[i]; bi = i; }
            const float mx = sycl::reduce_over_group(it.get_group(), best, sycl::maximum<float>());
            const int who = sycl::reduce_over_group(it.get_group(), best == mx ? bi : VOCAB, sycl::minimum<int>());
            if (it.get_local_id(0) == 0) tk[0] = who;
        });
    }
    // prompt chunk processed by the main model (HB holds its hidden rows): feed the MTP block so its KV covers the chunk
    void mtp_fill_chunk(const int * toks, int T, int pos0) {
        if (!has_mtp) return;
        if (pos0 == 0) {
            if (T > 1) { q.memcpy(HPREV, HB, (size_t) (T - 1) * D * 4); mtp_forward(toks + 1, HPREV, T - 1, 1); }
        } else {
            q.memcpy(HPREV, HLAST + (size_t) sq * D, D * 4);
            if (T > 1) q.memcpy(HPREV + D, HB, (size_t) (T - 1) * D * 4);
            mtp_forward(toks, HPREV, T, pos0);
        }
        q.memcpy(HLAST + (size_t) sq * D, HB + (size_t) (T - 1) * D, D * 4).wait();
    }

    // full-attention block on XN (normed) for T tokens at pos0..: X2 = X + attn(XN)
    void attn_block(Layer & l, int T, int pos0) {
                constexpr int QL = Q_ROWS + 2 * HD * NKV;
                bgemm(l.wqkv, XN, T, QA, QL, 0);
                tick("fa_qkv");
                rope_kv(l, T, pos0);
                if (T >= 16 && getenv("QE_NO_FLASH") == nullptr) {   // XMX flash attention
                    const int8_t * kc = l.kc, * vc = l.vc;
                    const int32_t * bt = BT + (size_t) sq * max_blocks;
                    const sycl::half * ks = l.ks, * vs = l.vs;
                    const float * qa = QA;
                    sycl::half * qh = QHb;
                    q.parallel_for(sycl::range<1>((size_t) T * NQ * HD), [=](sycl::id<1> i) {
                        const size_t t = i[0] / (NQ * HD), r = i[0] % (NQ * HD), hq = r / HD, d = r % HD;
                        qh[i[0]] = sycl::half(qa[t * QL + hq * 2 * HD + d] * 0.0625f);
                    });
                    if (getenv("QE_FLASH1")) flash_prefill(q, QHb, T, pos0, kc, vc, ks, vs, bt, ATT, LSUM);
                    else flash_prefill2(q, QHb, T, pos0, kc, vc, ks, vs, bt, ATT, LSUM);
                    float * at = ATT;
                    const float * ls = LSUM;
                    q.parallel_for(sycl::range<1>((size_t) T * NQ * HD), [=](sycl::id<1> i) {
                        const size_t t = i[0] / (NQ * HD), r = i[0] % (NQ * HD), hq = r / HD, d = r % HD;
                        const float gt = qa[t * QL + hq * 2 * HD + HD + d];
                        at[i[0]] = at[i[0]] / ls[t * NQ + hq] / (1.f + sycl::exp(-gt));
                    });
                } else
                {   // causal attention: work-group per (q head, query token), 8 sub-groups over the visible positions
                    const int8_t * kc = l.kc, * vc = l.vc;
                    const int32_t * bt = BT + (size_t) sq * max_blocks;
                    const sycl::half * ks = l.ks, * vs = l.vs;
                    float * qa = QA, * at = ATT;
                    q.submit([&](sycl::handler & hd) {
                        sycl::local_accessor<float, 1> sm(8 * (HD + 2), hd);
                        hd.parallel_for(sycl::nd_range<2>({(size_t) NQ, (size_t) T * 128}, {1, 128}), [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG)]] {
                            const int hq = it.get_global_id(0), t = it.get_group(1);
                            const auto sg = it.get_sub_group();
                            const int s_id = sg.get_group_id()[0], lane = sg.get_local_id()[0];
                            const int kh = hq / (NQ / NKV);
                            const float * qrow = qa + (size_t) t * QL + hq * 2 * HD;
                            float qd[16], acc[16];
                            for (int i = 0; i < 16; ++i) { qd[i] = qrow[lane * 16 + i] * 0.0625f; acc[i] = 0.f; }
                            float m = -1e30f, sum = 0.f;
                            const int n_vis = pos0 + t + 1;
                            for (int p = s_id; p < n_vis; p += 8) {
                                const size_t ko = ((size_t) (bt[(p) >> 8] * 256 + ((p) & 255)) * NKV + kh) * HD + lane * 16;
                                const sycl::vec<int8_t, 16> kk = *(const sycl::vec<int8_t, 16> *) (kc + ko);
                                const float kd = (float) ks[ko / 16];
                                float dd = 0.f;
#pragma unroll
                                for (int i = 0; i < 16; ++i) dd += qd[i] * (float) kk[i];
                                dd *= kd;
                                dd = sycl::reduce_over_group(sg, dd, sycl::plus<float>());
                                const float mn = sycl::max(m, dd);
                                const float c = sycl::exp(m - mn), e = sycl::exp(dd - mn);
                                sum = sum * c + e;
                                const sycl::vec<int8_t, 16> vv = *(const sycl::vec<int8_t, 16> *) (vc + ko);
                                const float vdq = (float) vs[ko / 16];
#pragma unroll
                                for (int i = 0; i < 16; ++i) acc[i] = acc[i] * c + e * vdq * (float) vv[i];
                                m = mn;
                            }
                            float * my = &sm[s_id * (HD + 2)];
                            for (int i = 0; i < 16; ++i) my[lane * 16 + i] = acc[i];
                            if (lane == 0) { my[HD] = m; my[HD + 1] = sum; }
                            it.barrier(sycl::access::fence_space::local_space);
                            const int lid = it.get_local_id(1);
                            float M = -1e30f;
                            for (int k = 0; k < 8; ++k) M = sycl::max(M, sm[k * (HD + 2) + HD]);
                            float Ssum = 0.f;
                            for (int k = 0; k < 8; ++k) Ssum += sm[k * (HD + 2) + HD + 1] * sycl::exp(sm[k * (HD + 2) + HD] - M);
                            for (int dd = lid; dd < HD; dd += 128) {
                                float a = 0.f;
                                for (int k = 0; k < 8; ++k) a += sm[k * (HD + 2) + dd] * sycl::exp(sm[k * (HD + 2) + HD] - M);
                                const float gt = qrow[HD + dd];
                                at[(size_t) t * HD * NQ + hq * HD + dd] = a / Ssum / (1.f + sycl::exp(-gt));
                            }
                        });
                    });
                }
                tick("fa_rope_attn");
                bgemm(l.wo, ATT, T, X2, D, 0, X);
                tick("fa_out");
    }

    // MoE for T tokens: X2 (post-attention residual) -> X; experts of this layer are keys key0 .. key0+255 in the tables
    void moe_batched(Layer & l, int key0, const uint8_t * ar_base, const int32_t * slots, const uint8_t * const * hexp, int T) {
        rmsnorm_rows(q, X2, l.post_norm, XN, T);
        bgemm(l.router, XN, T, RL, N_EXPERT + 1, 0);
        bgemm(l.sh_gu, XN, T, SHGU, 2 * FF_SH, 0);
        topk_rows(T);
        shared_act(T);
        bgemm(l.sh_down, SHH, T, SD, D, 0);
        tick("moe_router_sh");
        // group (token, k) pairs by expert on the host
        q.memcpy(ids_host, IDB, (size_t) T * K_USED * 4);
        q.memcpy(wb_host, WB, (size_t) T * K_USED * 4).wait();
        std::vector<int> cnt(N_EXPERT + 1, 0);
        std::vector<int> & pairs = h_pairs, & off = h_off;     // copied to the device asynchronously: members, not locals
        pairs.assign((size_t) T * K_USED, 0); off.assign(N_EXPERT + 1, 0);
        for (int p = 0; p < T * K_USED; ++p) cnt[ids_host[p]]++;
        for (int e = 0; e < N_EXPERT; ++e) off[e + 1] = off[e] + cnt[e];
        {
            std::vector<int> fill(off.begin(), off.end() - 1);
            for (int p = 0; p < T * K_USED; ++p) pairs[fill[ids_host[p]]++] = p;
        }
        q.memcpy(PAIRS, pairs.data(), pairs.size() * 4);
        q.memcpy(OFF, off.data(), off.size() * 4);
        tick("moe_group");
        // resident experts on the GPU, cold experts on the CPU at the same time
        std::vector<int> cold;
        std::vector<int> bslot(N_EXPERT, -1);
        for (int e = 0; e < N_EXPERT; ++e) {
            if (!cnt[e]) continue;
            const int s = slots[key0 + e];
            if (s >= 0) bslot[e] = s; else cold.push_back(e);
        }
        sycl::event e_xn;
        if (!cold.empty()) e_xn = q.memcpy(xn_host, XN, (size_t) T * D * 4);   // before any staging copy so the CPU can start early
        if (T >= 16 && !cold.empty() && hexp[key0 + cold[0]]) {
            // busiest cold experts go to the GPU through the staging buffer, the rest stay on the CPU
            std::sort(cold.begin(), cold.end(), [&](int a, int b) { return cnt[a] > cnt[b]; });
            const int min_tok = getenv("QE_STAGE_MIN") ? atoi(getenv("QE_STAGE_MIN")) : 14;
            int ns = 0;
            while (ns < (int) cold.size() && ns < STAGE && cnt[cold[ns]] >= min_tok) ++ns;
            for (int k = 0; k < ns; ++k) {
                stage_ev.push_back(qc.memcpy(stage + (size_t) k * exp_bytes(), hexp[key0 + cold[k]], exp_bytes()));
                bslot[cold[k]] = 1000000 + k;
            }
            cold.erase(cold.begin(), cold.begin() + ns);
            if (prof) pt["staged_experts(count)"] += ns;
        }
        std::fill(pcold_h, pcold_h + T * K_USED, (uint8_t) 0);
        if (T >= 16) {
            // resident experts first (overlaps the staging copies on the copy queue), then the staged ones
            // host sources of async copies must outlive the copies: keep them in members
            std::vector<int> & b_res = h_bres, & b_stg = h_bstg;
            b_res = bslot; b_stg.assign(N_EXPERT, -1);
            bool any_stg = false;
            for (int e = 0; e < N_EXPERT; ++e) if (bslot[e] >= 1000000) { b_stg[e] = bslot[e]; b_res[e] = -1; any_stg = true; }
            q.memcpy(BSLOT, b_res.data(), N_EXPERT * 4);
            moe_round_xmx(T, ar_base, cnt, off, pairs, b_res, BSLOT);
            if (any_stg) {
                q.memcpy(BSLOT2, b_stg.data(), N_EXPERT * 4);
                q.ext_oneapi_submit_barrier(stage_ev);
                moe_round_xmx(T, ar_base, cnt, off, pairs, b_stg, BSLOT2);
            }
            stage_ev.clear();
        } else {
            h_bres = bslot;
            q.memcpy(BSLOT, h_bres.data(), N_EXPERT * 4);
            moe_round(T, ar_base);          // in-order queue: runs after the copy, overlaps with the CPU work below
        }
        if (prof && getenv("QE_PROF_SPLIT")) tick("moe_gpu_only");
        if (!cold.empty()) { e_xn.wait(); cpu_cold_experts(hexp + key0, T, cold, off, pairs); }
        if (prof && getenv("QE_PROF_SPLIT")) { const double t = now_ms(); pt["moe_cpu_only"] += t - pt_last; pt_last = t; }
        tick("moe_experts");
        if (prof) pt["cold_experts(count)"] += cold.size();
        {   // combine: X = X2 + sum_k (GPU or CPU expert result) + shg[t] * SD[t]
            float * xo = X, * x2 = X2, * yp = YP, * sd = SD, * shg = SHG, * ypc = ypc_h;
            const uint8_t * pc = pcold_h;
            q.parallel_for(sycl::range<1>((size_t) T * D), [=](sycl::id<1> i) {
                const int t = i[0] / D, r = i[0] % D;
                float acc = x2[i[0]] + shg[t] * sd[i[0]];
                for (int k = 0; k < K_USED; ++k) {
                    const size_t p = (size_t) t * K_USED + k;
                    acc += pc[p] ? ypc[p * D + r] : yp[p * D + r];
                }
                xo[i[0]] = acc;
            });
        }
        tick("moe_combine");
    }

    // cold experts of one layer for a prefill chunk: rows streamed once per expert, all its tokens per row
    void cpu_cold_experts(const uint8_t * const * hexp, int T, const std::vector<int> & cold, const std::vector<int> & off, const std::vector<int> & pairs) {
        std::vector<uint8_t> need(T, 0);
        for (int e : cold) for (int k = off[e]; k < off[e + 1]; ++k) { need[pairs[k] / K_USED] = 1; pcold_h[pairs[k]] = 1; }
        pool.run_job([&](int tid) {
            for (int t = tid; t < T; t += pool.nt) if (need[t]) quant_x(xn_host + (size_t) t * D, D, xq_b + (size_t) t * D, xs_b + (size_t) t * (D / 32));
        });
        const int RB = 32, n_tasks_gu = (int) cold.size() * (FF_EXP / RB);
        std::atomic<int> next{0};
        pool.run_job([&](int) {
            for (int task; (task = next.fetch_add(1)) < n_tasks_gu;) {
                const int e = cold[task / (FF_EXP / RB)], r0 = (task % (FF_EXP / RB)) * RB;
                const uint8_t * base = hexp[e];
                for (int r = r0; r < r0 + RB; ++r) {
                    const uint8_t * rg = base + (size_t) r * exp_gu_rb(), * ru = base + (size_t) (FF_EXP + r) * exp_gu_rb();
                    for (int k = off[e]; k < off[e + 1]; k += 4) {
                        const int n = std::min(4, off[e + 1] - k);
                        const int8_t * xqp[4]; const float * xsp[4];
                        for (int i = 0; i < n; ++i) { const int t = pairs[k + i] / K_USED; xqp[i] = xq_b + (size_t) t * D; xsp[i] = xs_b + (size_t) t * (D / 32); }
                        float g[4], u[4];
                        cpu_dot_nl4_x4(rg, D, xqp, xsp, n, g);
                        cpu_dot_nl4_x4(ru, D, xqp, xsp, n, u);
                        for (int i = 0; i < n; ++i) hcold[(size_t) pairs[k + i] * FF_EXP + r] = g[i] / (1.f + std::exp(-g[i])) * u[i];
                    }
                }
            }
        });
        std::vector<int> cp;
        for (int e : cold) for (int k = off[e]; k < off[e + 1]; ++k) cp.push_back(pairs[k]);
        pool.run_job([&](int tid) {
            for (size_t i = tid; i < cp.size(); i += pool.nt)
                quant_x(hcold + (size_t) cp[i] * FF_EXP, FF_EXP, hq_b + (size_t) cp[i] * FF_EXP, hs_b + (size_t) cp[i] * (FF_EXP / 32));
        });
        const int RD = 128, n_tasks_d = (int) cold.size() * (D / RD);
        next = 0;
        pool.run_job([&](int) {
            for (int task; (task = next.fetch_add(1)) < n_tasks_d;) {
                const int e = cold[task / (D / RD)], r0 = (task % (D / RD)) * RD;
                const uint8_t * base = hexp[e] + 2 * (size_t) FF_EXP * exp_gu_rb();
                for (int r = r0; r < r0 + RD; ++r) {
                    const uint8_t * rd = base + (size_t) r * exp_d_rb();
                    for (int k = off[e]; k < off[e + 1]; k += 4) {
                        const int n = std::min(4, off[e + 1] - k);
                        const int8_t * hqp[4]; const float * hsp[4];
                        for (int i = 0; i < n; ++i) { const int p = pairs[k + i]; hqp[i] = hq_b + (size_t) p * FF_EXP; hsp[i] = hs_b + (size_t) p * (FF_EXP / 32); }
                        float y[4];
                        cpu_dot_nl4_x4(rd, FF_EXP, hqp, hsp, n, y);
                        for (int i = 0; i < n; ++i) ypc_h[(size_t) pairs[k + i] * D + r] = wb_host[pairs[k + i]] * y[i];
                    }
                }
            }
        });
    }

    // GPU expert pass for a big chunk with XMX: gather token rows per resident expert (padded to 8), two GEMMs, scatter
    std::vector<sycl::event> stage_ev;
    int32_t * BSLOT2 = nullptr;
    std::vector<int> h_bres, h_bstg, h_padoff[2], h_padmap[2];   // host sources of async copies (must outlive them)
    std::vector<int> h_pairs, h_off;
    int xmx_call = 0;
    void moe_round_xmx(int T, const uint8_t * ar, const std::vector<int> & cnt, const std::vector<int> & off,
                       const std::vector<int> & pairs, const std::vector<int> & bslot, const int32_t * bslot_d) {
        const int cix = (xmx_call++) & 1;
        std::vector<int> & pad_off = h_padoff[cix];
        pad_off.assign(N_EXPERT + 1, 0);
        for (int e = 0; e < N_EXPERT; ++e) pad_off[e + 1] = pad_off[e] + (bslot[e] >= 0 ? (cnt[e] + 7) / 8 * 8 : 0);
        const int rows = pad_off[N_EXPERT];
        if (rows == 0) return;
        std::vector<int> & padmap = h_padmap[cix];
        padmap.assign(rows, -1);
        for (int e = 0; e < N_EXPERT; ++e)
            if (bslot[e] >= 0) for (int k = 0; k < cnt[e]; ++k) padmap[pad_off[e] + k] = pairs[off[e] + k];
        q.memcpy(PADOFF, pad_off.data(), (N_EXPERT + 1) * 4);
        q.memcpy(PADMAP, padmap.data(), (size_t) rows * 4);
        {   // gather normalized token rows as f16
            const float * xn = XN;
            const int32_t * pm = PADMAP;
            sycl::half * xg = XGp;
            q.parallel_for(sycl::range<1>((size_t) rows * D), [=](sycl::id<1> i) {
                const int r = i[0] / D, c = i[0] % D;
                const int p = pm[r];
                xg[i[0]] = p < 0 ? sycl::half(0.f) : sycl::half(xn[(size_t) (p / K_USED) * D + c]);
            });
        }
        moe_xmx(q, XGp, D, 2 * FF_EXP, ar, exp_bytes(), 0, exp_gu_rb(), bslot_d, PADOFF, HGUp, stage);
        {
            const float * hgu = HGUp;
            sycl::half * hh = HHp;
            q.parallel_for(sycl::range<1>((size_t) rows * FF_EXP), [=](sycl::id<1> i) {
                const int r = i[0] / FF_EXP, j = i[0] % FF_EXP;
                const float g = hgu[(size_t) r * 2 * FF_EXP + j], u = hgu[(size_t) r * 2 * FF_EXP + FF_EXP + j];
                hh[i[0]] = sycl::half(g / (1.f + sycl::exp(-g)) * u);
            });
        }
        moe_xmx(q, HHp, FF_EXP, D, ar, exp_bytes(), 2 * (size_t) FF_EXP * exp_gu_rb(), exp_d_rb(), bslot_d, PADOFF, YDp, stage);
        {   // scatter weighted results to the per-pair buffer
            const float * yd = YDp, * wb = WB;
            const int32_t * pm = PADMAP;
            float * yp = YP;
            q.parallel_for(sycl::range<1>((size_t) rows * D), [=](sycl::id<1> i) {
                const int r = i[0] / D, c = i[0] % D;
                const int p = pm[r];
                if (p >= 0) yp[(size_t) p * D + c] = wb[p] * yd[i[0]];
            });
        }
    }

    void moe_round(int T, const uint8_t * ar) {
        const int32_t * prs = PAIRS, * of = OFF, * bs = BSLOT;
        const uint8_t * st = stage;
        const float * xn = XN, * wb = WB;
        float * hp = HP, * yp = YP;
        const size_t EB = exp_bytes(), GRB = exp_gu_rb(), DRB = exp_d_rb();
        auto base_of = [=](int s) { return s >= 1000000 ? st + (size_t) (s - 1000000) * EB : ar + (size_t) s * EB; };
        // gate/up: sub-group per (expert, row), its tokens in groups of TG
        q.parallel_for(sycl::nd_range<1>((size_t) N_EXPERT * FF_EXP * SG, 128), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            const auto sg = it.get_sub_group();
            const int gid = it.get_global_id(0) / SG, lane = sg.get_local_id()[0];
            const int e = gid / FF_EXP, rr = gid % FF_EXP;
            const int s = bs[e];
            if (s < 0) return;
            const int p0 = of[e], p1 = of[e + 1];
            const uint8_t * base = base_of(s);
            const uint8_t * rg = base + (size_t) rr * GRB, * ru = base + (size_t) (FF_EXP + rr) * GRB;
            for (int pp = p0; pp < p1; pp += TG) {
                const int n = sycl::min(TG, p1 - pp);
                float ag[TG], au[TG];
#pragma unroll
                for (int k = 0; k < TG; ++k) { ag[k] = 0.f; au[k] = 0.f; }
                float wg[32], wu[32];
                for (int b = lane; b < D / 32; b += SG) {
                    deq_block(F_NL4, rg, D, b, wg);
                    deq_block(F_NL4, ru, D, b, wu);
#pragma unroll
                    for (int k = 0; k < TG; ++k) {
                        if (k >= n) break;
                        const float * xr = xn + (size_t) (prs[pp + k] / K_USED) * D + b * 32;
                        float a = 0.f, c = 0.f;
#pragma unroll
                        for (int i = 0; i < 32; ++i) { a += wg[i] * xr[i]; c += wu[i] * xr[i]; }
                        ag[k] += a; au[k] += c;
                    }
                }
#pragma unroll
                for (int k = 0; k < TG; ++k) {
                    const float g = sycl::reduce_over_group(sg, ag[k], sycl::plus<float>());
                    const float u = sycl::reduce_over_group(sg, au[k], sycl::plus<float>());
                    if (lane == 0 && k < n) hp[(size_t) prs[pp + k] * FF_EXP + rr] = g / (1.f + sycl::exp(-g)) * u;
                }
            }
        });
        // down: sub-group per (expert, output row)
        q.parallel_for(sycl::nd_range<1>((size_t) N_EXPERT * D * SG, 128), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            const auto sg = it.get_sub_group();
            const int gid = it.get_global_id(0) / SG, lane = sg.get_local_id()[0];
            const int e = gid / D, r = gid % D;
            const int s = bs[e];
            if (s < 0) return;
            const int p0 = of[e], p1 = of[e + 1];
            const uint8_t * row = base_of(s) + 2 * (size_t) FF_EXP * GRB + (size_t) r * DRB;
            float w[32];
            deq_block(F_NL4, row, FF_EXP, lane, w);
            for (int pp = p0; pp < p1; ++pp) {
                const int p = prs[pp];
                const float * hr = hp + (size_t) p * FF_EXP + lane * 32;
                float a = 0.f;
#pragma unroll
                for (int i = 0; i < 32; ++i) a += w[i] * hr[i];
                a = sycl::reduce_over_group(sg, a, sycl::plus<float>());
                if (lane == 0) yp[(size_t) p * D + r] = a * wb[p];
            }
        });
    }

    void reset() {
        rope_delta[sq] = 0;
        release_kv(sq, 0);
        for (auto & l : L) if (l.recr) { q.fill(l.conv_state + sq * CONV_SZ, 0.f, CONV_SZ); q.fill(l.ssm_state + sq * SSM_SZ, 0.f, SSM_SZ); }
        q.wait();
    }

    // one token at position pos; logits left on the device, argmax in `tok`
    void decode(int token, int pos) {
        ensure_kv(sq, pos + 1);
        // embedding row (Q8) dequantized on the host
        {
            const uint8_t * row = embd + (size_t) token * row_bytes(F_Q8, D);
            const sycl::half * sc = (const sycl::half *) (row + D);
            for (int i = 0; i < D; ++i) x_host[i] = (float) ((const int8_t *) row)[i] * (float) sc[i / 32];
            q.memcpy(x, x_host, D * 4);
        }
        const int n_ctx = pos + 1;
        for (int il = 0; il < N_LAYER; ++il) {
            Layer & l = L[il];
            rmsnorm(q, x, l.attn_norm, xn);
            if (l.recr) {
                gemv(q, l.qkv, xn, qkvz);
                gemv(q, l.z, xn, qkvz + QKV_DIM);
                gemv(q, l.ab, xn, ab);
                // causal conv1d over [state(3), x] + silu; l2 norm of every q / k head (work-group = head)
                {
                    float * cs = l.conv_state + sq * CONV_SZ, * cw = l.conv_w, * qz = qkvz;
                    q.parallel_for(sycl::nd_range<1>(QKV_DIM, DK), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                        const int c = it.get_global_id(0);
                        const float in = qz[c];
                        float * st = cs + c * (CONV_K - 1);
                        float acc = cw[c * CONV_K + CONV_K - 1] * in;
                        for (int k = 0; k < CONV_K - 1; ++k) acc += cw[c * CONV_K + k] * st[k];
                        for (int k = 0; k < CONV_K - 2; ++k) st[k] = st[k + 1];
                        st[CONV_K - 2] = in;
                        float v = acc / (1.f + sycl::exp(-acc));
                        if (c < 2 * DK * NK) {
                            const float ss = sycl::reduce_over_group(it.get_group(), v * v, sycl::plus<float>());
                            v *= sycl::rsqrt(ss + RMS_EPS);
                        }
                        qz[c] = v;
                    });
                }
                // gated delta rule (v head hv uses k head hv % NK); S stored [hv][i (k dim)][j (v dim)]
                {
                    float * S = l.ssm_state + sq * SSM_SZ, * nw = l.ssm_norm, * sa = l.ssm_a, * dtb = l.dt_bias, * qz = qkvz, * abp = ab, * ao = attn_o;
                    q.submit([&](sycl::handler & hd) {
                        sycl::local_accessor<float, 1> red(4 * DK, hd);
                        hd.parallel_for(sycl::nd_range<1>(NV * 4 * DK, 4 * DK), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                            const int hv = it.get_group(0), lid = it.get_local_id(0);
                            const int j = lid % DK, p = lid / DK;
                            const int hk = hv % NK;
                            const float * qv = qz + hk * DK;
                            const float * kv = qz + DK * NK + hk * DK;
                            const float vj = qz[2 * DK * NK + hv * DK + j];
                            const float a = abp[hv] + dtb[hv];
                            const float sp = a > 20.f ? a : sycl::log1p(sycl::exp(a));
                            const float g = sycl::exp(sp * sa[hv]);
                            const float beta = 1.f / (1.f + sycl::exp(-abp[NV + hv]));
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
                            const float zz = qz[QKV_DIM + hv * DK + j];
                            if (p == 0) ao[hv * DK + j] = oj * sycl::rsqrt(ss / DK + RMS_EPS) * nw[j] * (zz / (1.f + sycl::exp(-zz)));
                        });
                    });
                }
                gemv(q, l.ssm_out, attn_o, x2, x);
            } else {
                gemv(q, l.wqkv, xn, qkv_a);
                // q/k rms norm, neox rope on the first N_ROT dims, write k/v to the cache
                {
                    float * qa = qkv_a, * qn = l.q_norm, * kn = l.k_norm;
                    const int rdelta = rope_delta[sq];
                    int8_t * kc = l.kc, * vc = l.vc;
                    const int32_t * bt = BT + (size_t) sq * max_blocks;
                    sycl::half * ks = l.ks, * vs = l.vs;
                    q.parallel_for(sycl::nd_range<1>((NQ + NKV) * HD, HD), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                        const int hh = it.get_group(0), d = it.get_local_id(0);
                        const bool isq = hh < NQ;
                        float * v = isq ? qa + hh * 2 * HD : qa + Q_ROWS + (hh - NQ) * HD;
                        const float val = v[d];
                        const float ss = sycl::reduce_over_group(it.get_group(), val * val, sycl::plus<float>());
                        float y = val * sycl::rsqrt(ss / HD + RMS_EPS) * (isq ? qn[d] : kn[d]);
                        it.barrier();
                        v[d] = y;
                        it.barrier();
                        if (d < N_ROT) {
                            const int i = d % (N_ROT / 2);
                            const float th = (float) (pos - rdelta) * sycl::pow(ROPE_BASE, -2.f * i / N_ROT);
                            const float cs = sycl::cos(th), sn = sycl::sin(th);
                            y = d < N_ROT / 2 ? v[d] * cs - v[d + N_ROT / 2] * sn : v[d] * cs + v[d - N_ROT / 2] * sn;
                        }
                        it.barrier();
                        if (isq) {
                            v[d] = y;
                        } else {
                            const int kh = hh - NQ;
                            const auto sg = it.get_sub_group();
                            const float vv = qa[Q_ROWS + NKV * HD + kh * HD + d];
                            const float ka = sycl::reduce_over_group(sg, sycl::fabs(y), sycl::maximum<float>());
                            const float va = sycl::reduce_over_group(sg, sycl::fabs(vv), sycl::maximum<float>());
                            const float kd = ka / 127.f, vd = va / 127.f;
                            const size_t o = ((size_t) (bt[(pos) >> 8] * 256 + ((pos) & 255)) * NKV + kh) * HD + d;
                            kc[o] = (int8_t) sycl::rint(kd > 0 ? y / kd : 0.f);
                            vc[o] = (int8_t) sycl::rint(vd > 0 ? vv / vd : 0.f);
                            if (d % 16 == 0) { ks[o / 16] = sycl::half(kd); vs[o / 16] = sycl::half(vd); }
                        }
                    });
                }
                const int n_chunks = (n_ctx + ATT_CHUNK - 1) / ATT_CHUNK;
                if (n_ctx >= gqa_min_ctx) {   // long context: one work-group per (kv head, chunk), its 8 sub-groups = the 8 q heads -> K/V fetched once
                    const int8_t * kc = l.kc, * vc = l.vc;
                    const int32_t * bt = BT + (size_t) sq * max_blocks;
                    const sycl::half * ks = l.ks, * vs = l.vs;
                    float * qa = qkv_a, * pp0 = part;
                    q.parallel_for(sycl::nd_range<2>({(size_t) NKV, (size_t) n_chunks * 128}, {1, 128}), [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG)]] {
                        const int kh = it.get_global_id(0), ch = it.get_group(1);
                        const auto sg = it.get_sub_group();
                        const int s_id = sg.get_group_id()[0], lane = sg.get_local_id()[0];
                        const int hq = kh * (NQ / NKV) + s_id;
                        float qd[16], acc[16];
                        for (int i = 0; i < 16; ++i) { qd[i] = qa[hq * 2 * HD + lane * 16 + i] * 0.0625f; acc[i] = 0.f; }
                        float m = -1e30f, sum = 0.f;
                        const int t0 = ch * ATT_CHUNK, t1 = sycl::min(n_ctx, t0 + ATT_CHUNK);
                        const size_t base = (size_t) bt[ch] * 256;
                        for (int t = t0; t < t1; ++t) {
                            const size_t ko = ((base + (t - t0)) * NKV + kh) * HD + lane * 16;
                            const sycl::vec<int8_t, 16> kk = *(const sycl::vec<int8_t, 16> *) (kc + ko);
                            float dd = 0.f;
#pragma unroll
                            for (int i = 0; i < 16; ++i) dd += qd[i] * (float) kk[i];
                            dd = sycl::reduce_over_group(sg, dd * (float) ks[ko / 16], sycl::plus<float>());
                            const float mn = sycl::max(m, dd);
                            const float c = sycl::exp(m - mn), e = sycl::exp(dd - mn);
                            sum = sum * c + e;
                            const sycl::vec<int8_t, 16> vv = *(const sycl::vec<int8_t, 16> *) (vc + ko);
                            const float vdq = (float) vs[ko / 16];
#pragma unroll
                            for (int i = 0; i < 16; ++i) acc[i] = acc[i] * c + e * vdq * (float) vv[i];
                            m = mn;
                        }
                        float * pp = pp0 + ((size_t) hq * n_chunks + ch) * (HD + 2);
                        for (int i = 0; i < 16; ++i) pp[lane * 16 + i] = acc[i];
                        if (lane == 0) { pp[HD] = m; pp[HD + 1] = sum; }
                    });
                } else {
                    const int8_t * kc = l.kc, * vc = l.vc;
                    const int32_t * bt = BT + (size_t) sq * max_blocks;
                    const sycl::half * ks = l.ks, * vs = l.vs;
                    float * qa = qkv_a, * pp0 = part;
                    q.submit([&](sycl::handler & hd) {
                        sycl::local_accessor<float, 1> sm(8 * (HD + 2), hd);
                        hd.parallel_for(sycl::nd_range<2>({(size_t) NQ, (size_t) n_chunks * 128}, {1, 128}),
                                        [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG)]] {
                            const int hq = it.get_global_id(0), ch = it.get_group(1);
                            const auto sg = it.get_sub_group();
                            const int s_id = sg.get_group_id()[0], lane = sg.get_local_id()[0];
                            const int kh = hq / (NQ / NKV);
                            float qd[16], acc[16];
                            for (int i = 0; i < 16; ++i) { qd[i] = qa[hq * 2 * HD + lane * 16 + i] * 0.0625f; acc[i] = 0.f; }
                            float m = -1e30f, sum = 0.f;
                            const int t0 = ch * ATT_CHUNK, t1 = sycl::min(n_ctx, t0 + ATT_CHUNK);
                            for (int t = t0 + s_id; t < t1; t += 8) {
                                const size_t ko = ((size_t) (bt[(t) >> 8] * 256 + ((t) & 255)) * NKV + kh) * HD + lane * 16;
                                const sycl::vec<int8_t, 16> kk = *(const sycl::vec<int8_t, 16> *) (kc + ko);
                                const float kd = (float) ks[ko / 16];
                                float dd = 0.f;
#pragma unroll
                                for (int i = 0; i < 16; ++i) dd += qd[i] * (float) kk[i];
                                dd *= kd;
                                dd = sycl::reduce_over_group(sg, dd, sycl::plus<float>());
                                const float mn = sycl::max(m, dd);
                                const float c = sycl::exp(m - mn), e = sycl::exp(dd - mn);
                                sum = sum * c + e;
                                const sycl::vec<int8_t, 16> vv = *(const sycl::vec<int8_t, 16> *) (vc + ko);
                                const float vdq = (float) vs[ko / 16];
#pragma unroll
                                for (int i = 0; i < 16; ++i) acc[i] = acc[i] * c + e * vdq * (float) vv[i];
                                m = mn;
                            }
                            float * my = &sm[s_id * (HD + 2)];
                            for (int i = 0; i < 16; ++i) my[lane * 16 + i] = acc[i];
                            if (lane == 0) { my[HD] = m; my[HD + 1] = sum; }
                            it.barrier(sycl::access::fence_space::local_space);
                            const int lid = it.get_local_id(1);
                            float M = -1e30f;
                            for (int k = 0; k < 8; ++k) M = sycl::max(M, sm[k * (HD + 2) + HD]);
                            float * pp = pp0 + ((size_t) hq * n_chunks + ch) * (HD + 2);
                            for (int dd = lid; dd < HD; dd += 128) {
                                float a = 0.f;
                                for (int k = 0; k < 8; ++k) a += sm[k * (HD + 2) + dd] * sycl::exp(sm[k * (HD + 2) + HD] - M);
                                pp[dd] = a;
                            }
                            if (lid == 0) {
                                float Ssum = 0.f;
                                for (int k = 0; k < 8; ++k) Ssum += sm[k * (HD + 2) + HD + 1] * sycl::exp(sm[k * (HD + 2) + HD] - M);
                                pp[HD] = M; pp[HD + 1] = Ssum;
                            }
                        });
                    });
                }
                {
                    float * qa = qkv_a, * pp0 = part, * at = att;
                    q.parallel_for(sycl::nd_range<1>(NQ * HD, HD), [=](sycl::nd_item<1> it) {
                        const int hq = it.get_group(0), d = it.get_local_id(0);
                        float m = -1e30f;
                        for (int c = 0; c < n_chunks; ++c) m = sycl::max(m, pp0[((size_t) hq * n_chunks + c) * (HD + 2) + HD]);
                        float s = 0.f, a = 0.f;
                        for (int c = 0; c < n_chunks; ++c) {
                            const float * pp = pp0 + ((size_t) hq * n_chunks + c) * (HD + 2);
                            const float w = sycl::exp(pp[HD] - m);
                            s += pp[HD + 1] * w;
                            a += pp[d] * w;
                        }
                        const float gt = qa[hq * 2 * HD + HD + d];
                        at[hq * HD + d] = a / s / (1.f + sycl::exp(-gt));
                    });
                }
                gemv(q, l.wo, att, x2, x);
            }
            // ---------------- MoE ----------------
            rmsnorm(q, x2, l.post_norm, xn);
            gemv(q, l.router, xn, rlog);
            gemv(q, l.sh_gu, xn, shgu);
            sycl::event e_topk;
            {
                float * rl = rlog, * wt = wts, * xf = xn;
                int32_t * id = ids;
                HostIO * io = hio;
                e_topk = q.parallel_for(sycl::nd_range<1>(SG, SG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                    const auto sg = it.get_sub_group();
                    const int lane = sg.get_local_id()[0];
                    float v[N_EXPERT / SG];
                    for (int i = 0; i < N_EXPERT / SG; ++i) v[i] = rl[i * SG + lane];
                    float sel[K_USED];
                    for (int k = 0; k < K_USED; ++k) {
                        float best = -1e30f; int bi = 0;
                        for (int i = 0; i < N_EXPERT / SG; ++i) if (v[i] > best) { best = v[i]; bi = i; }
                        const float mx = sycl::reduce_over_group(sg, best, sycl::maximum<float>());
                        const int who = sycl::reduce_over_group(sg, best == mx ? lane : SG, sycl::minimum<int>());
                        const int bidx = sycl::group_broadcast(sg, bi, who);
                        if (lane == who) v[bi] = -1e30f;
                        sel[k] = mx;
                        if (lane == 0) id[k] = bidx * SG + who;
                    }
                    if (lane == 0) {
                        float s = 0.f;
                        for (int k = 0; k < K_USED; ++k) s += sycl::exp(sel[k] - sel[0]);
                        for (int k = 0; k < K_USED; ++k) { wt[k] = sycl::exp(sel[k] - sel[0]) / s; io->ids[k] = id[k]; io->wts[k] = wt[k]; }
                        wt[K_USED] = 1.f / (1.f + sycl::exp(-rl[N_EXPERT]));
                    }
                    for (int i = lane; i < D; i += SG) io->xf[i] = xf[i];
                });
            }
            // routed gate/up for resident experts + shared expert activation
            {
                const int nrow = K_USED * FF_EXP + FF_SH, layer = il;
                const float * xf = xn, * sg_ = shgu;
                float * hh = h;
                const int32_t * id = ids, * sl = slot_d;
                const uint8_t * ar = arena;
                const size_t EB = exp_bytes(), GRB = exp_gu_rb();
                q.parallel_for(sycl::nd_range<1>((size_t) nrow * SG, 128), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                    const auto sg = it.get_sub_group();
                    const int r = it.get_global_id(0) / SG, lane = sg.get_local_id()[0];
                    if (r >= K_USED * FF_EXP) {
                        if (lane == 0) {
                            const int rr = r - K_USED * FF_EXP;
                            const float g = sg_[rr], u = sg_[FF_SH + rr];
                            hh[K_USED * FF_EXP + rr] = g / (1.f + sycl::exp(-g)) * u;
                        }
                        return;
                    }
                    const int e = r / FF_EXP, rr = r % FF_EXP;
                    const int slot = sl[layer * N_EXPERT + id[e]];
                    if (slot < 0) return;
                    const uint8_t * base = ar + (size_t) slot * EB;
                    float acc[2];
                    for (int m = 0; m < 2; ++m) {
                        const uint8_t * row = base + ((size_t) m * FF_EXP + rr) * GRB;
                        float a = 0.f;
                        for (int b = lane; b < D / 32; b += SG) a += row_blk_dot(F_NL4, row, D, b, xf);
                        acc[m] = sycl::reduce_over_group(sg, a, sycl::plus<float>());
                    }
                    if (lane == 0) hh[e * FF_EXP + rr] = acc[0] / (1.f + sycl::exp(-acc[0])) * acc[1];
                });
            }
            // host: missed experts on the CPU while the GPU runs the resident ones
            e_topk.wait();
            int nm = 0;
            for (int k = 0; k < K_USED; ++k) {
                const int key = il * N_EXPERT + hio->ids[k];
                if (slot_h[key] < 0) { pool.exp_ptr[nm] = host_exp[key]; pool.exp_w[nm] = hio->wts[k]; ++nm; }
            }
            if (nm) {
                const double c0 = now_ms();
                pool.n_miss = nm;
                pool.compute(hio->xf);
                cpu_ms += now_ms() - c0;
                n_miss += nm;
            }
            // routed down (resident) + shared down + CPU result, residual -> x
            {
                const int layer = il;
                const bool uc = nm > 0;
                const float * cpo = hio->cpu_out, * hh = h, * wt = wts, * xr = x2;
                float * xo = x;
                const int32_t * id = ids, * sl = slot_d;
                const uint8_t * ar = arena, * shd = l.sh_down.w;
                const size_t EB = exp_bytes(), GRB = exp_gu_rb(), DRB = exp_d_rb(), shrb = l.sh_down.rb;
                const Fmt shf = l.sh_down.f;
                q.parallel_for(sycl::nd_range<1>((size_t) D * SG, 128), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                    const auto sg = it.get_sub_group();
                    const int r = it.get_global_id(0) / SG, lane = sg.get_local_id()[0];
                    float acc = 0.f;
                    for (int e = 0; e < K_USED; ++e) {
                        const int slot = sl[layer * N_EXPERT + id[e]];
                        if (slot < 0) continue;
                        const uint8_t * row = ar + (size_t) slot * EB + 2 * (size_t) FF_EXP * GRB + (size_t) r * DRB;
                        acc += wt[e] * row_blk_dot(F_NL4, row, FF_EXP, lane, hh + e * FF_EXP);
                    }
                    acc += wt[K_USED] * row_blk_dot(shf, shd + (size_t) r * shrb, FF_SH, lane, hh + K_USED * FF_EXP);
                    acc = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
                    if (lane == 0) xo[r] = xr[r] + acc + (uc ? cpo[r] : 0.f);
                });
            }
        }
        rmsnorm(q, x, out_norm, xn);
        gemv(q, lm, xn, logits);
        {
            float * lg = logits;
            int32_t * tk = tok;
            q.parallel_for(sycl::nd_range<1>(1024, 1024), [=](sycl::nd_item<1> it) {
                float best = -1e30f; int bi = 0;
                for (int i = it.get_local_id(0); i < VOCAB; i += 1024) if (lg[i] > best) { best = lg[i]; bi = i; }
                const float mx = sycl::reduce_over_group(it.get_group(), best, sycl::maximum<float>());
                const int who = sycl::reduce_over_group(it.get_group(), best == mx ? bi : VOCAB, sycl::minimum<int>());
                if (it.get_local_id(0) == 0) tk[0] = who;
            });
        }
        ++n_tok;
    }
    int argmax() { int t; q.memcpy(&t, tok, 4).wait(); return t; }
};

// ============================ chat server ============================
using json = nlohmann::ordered_json;

// length of the longest prefix of s that ends on a complete UTF-8 character
static size_t utf8_complete_len(const std::string & s) {
    const size_t n = s.size();
    for (size_t back = 1; back <= 4 && back <= n; ++back) {
        const unsigned char c = (unsigned char) s[n - back];
        if ((c & 0xC0) == 0x80) continue;                 // continuation byte, keep looking for the lead byte
        const size_t need = (c & 0x80) == 0 ? 1 : (c & 0xE0) == 0xC0 ? 2 : (c & 0xF0) == 0xE0 ? 3 : (c & 0xF8) == 0xF0 ? 4 : 1;
        return need > back ? n - back : n;
    }
    return n;
}
static std::string jdump(const json & j) { return j.dump(-1, ' ', false, json::error_handler_t::replace); }

static std::vector<std::string> * g_images = nullptr;   // when set, image parts become media markers and their URLs are collected
static std::string content_text(const json & c, bool & has_image) {
    if (c.is_string()) return c.get<std::string>();
    std::string out;
    if (c.is_array())
        for (auto & p : c) {
            const std::string t = p.value("type", "");
            if (t == "text" || (t.empty() && p.contains("text"))) out += p.value("text", "");
            else if (t == "image_url" || t == "input_image" || t == "image" || p.contains("image_url") || p.contains("image")) {
                has_image = true;
                if (g_images) {
                    std::string url;
                    if (p.contains("image_url")) url = p["image_url"].is_string() ? p["image_url"].get<std::string>() : p["image_url"].value("url", "");
                    else if (p.contains("image")) url = p["image"].is_string() ? p["image"].get<std::string>() : "";
                    g_images->push_back(url);
                    out += mtmd_default_marker();
                }
            }
        }
    return out;
}
static std::string b64_decode(const std::string & in) {
    static int T[256];
    static bool init = false;
    if (!init) { for (int i = 0; i < 256; ++i) T[i] = -1; const char * a = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"; for (int i = 0; i < 64; ++i) T[(unsigned char) a[i]] = i; T['-'] = 62; T['_'] = 63; init = true; }
    std::string out; int val = 0, bits = -8;
    for (unsigned char c : in) { if (T[c] < 0) continue; val = (val << 6) + T[c]; bits += 6; if (bits >= 0) { out.push_back((char) ((val >> bits) & 0xFF)); bits -= 8; } }
    return out;
}
static std::string trim(const std::string & s) {
    size_t a = 0, b = s.size();
    while (a < b && isspace((unsigned char) s[a])) ++a;
    while (b > a && isspace((unsigned char) s[b - 1])) --b;
    return s.substr(a, b - a);
}
// jinja/Python json.dumps style: ", " and ": " separators, non-ASCII kept as is, key order preserved
static void pydump(const json & j, std::string & o) {
    if (j.is_object()) {
        o += '{'; bool first = true;
        for (auto it = j.begin(); it != j.end(); ++it) { if (!first) o += ", "; first = false; pydump(it.key(), o); o += ": "; pydump(it.value(), o); }
        o += '}';
    } else if (j.is_array()) {
        o += '['; bool first = true;
        for (auto & v : j) { if (!first) o += ", "; first = false; pydump(v, o); }
        o += ']';
    } else if (j.is_string()) {
        o += '"';
        for (unsigned char c : j.get<std::string>()) {
            switch (c) {
                case '"': o += "\\\""; break; case '\\': o += "\\\\"; break; case '\n': o += "\\n"; break;
                case '\r': o += "\\r"; break; case '\t': o += "\\t"; break; case '\b': o += "\\b"; break; case '\f': o += "\\f"; break;
                default: if (c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); o += b; } else o += (char) c;
            }
        }
        o += '"';
    } else if (j.is_boolean()) {
        o += j.get<bool>() ? "true" : "false";
    } else if (j.is_null()) {
        o += "null";
    } else {
        o += j.dump();
    }
}
static std::string pydumps(const json & j) { std::string o; pydump(j, o); return o; }
// jinja `x | string` for tool-call argument values that are not mappings / sequences
static std::string pystr(const json & v) {
    if (v.is_string()) return v.get<std::string>();
    if (v.is_boolean()) return v.get<bool>() ? "True" : "False";
    if (v.is_null()) return "None";
    return v.dump();
}

// Qwen3.6 chat template (text + tools); images flagged in has_image
static std::string render_chat(const json & msgs, const json & tools, bool thinking, bool & has_image) {
    std::string out;
    size_t num_sys = 0;
    std::string sys;
    if (!msgs.empty() && (msgs[0].value("role", "") == "system" || msgs[0].value("role", "") == "developer")) {
        sys = trim(content_text(msgs[0].value("content", json("")), has_image)); num_sys = 1;
        if (msgs.size() > 1 && (msgs[1].value("role", "") == "system" || msgs[1].value("role", "") == "developer")) {
            sys += "\n" + trim(content_text(msgs[1].value("content", json("")), has_image)); num_sys = 2;
        }
    }
    if (tools.is_array() && !tools.empty()) {
        out += "<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n<tools>";
        for (auto & t : tools) out += "\n" + pydumps(t);
        out += "\n</tools>";
        out += "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>";
        if (!sys.empty()) out += "\n\n" + sys;
        out += "<|im_end|>\n";
    } else if (!sys.empty()) {
        out += "<|im_start|>system\n" + sys + "<|im_end|>\n";
    }
    // last real user query (tool responses wrapped as user messages do not count)
    size_t last_query = msgs.size() - 1;
    for (size_t i = msgs.size(); i-- > 0;) {
        if (msgs[i].value("role", "") != "user") continue;
        bool dummy = false;
        std::vector<std::string> * keep = g_images;
        g_images = nullptr;                    // only a peek: do not collect the images twice
        const std::string c = trim(content_text(msgs[i].value("content", json("")), dummy));
        g_images = keep;
        if (!(c.rfind("<tool_response>", 0) == 0 && c.size() >= 16 && c.compare(c.size() - 16, 16, "</tool_response>") == 0)) { last_query = i; break; }
    }
    for (size_t i = num_sys; i < msgs.size(); ++i) {
        const json & m = msgs[i];
        const std::string role = m.value("role", "user");
        if (role == "system" || role == "developer") continue;
        std::string c = trim(m.contains("content") && !m["content"].is_null() ? content_text(m["content"], has_image) : "");
        if (role == "user") {
            out += "<|im_start|>user\n" + c + "<|im_end|>\n";
        } else if (role == "assistant") {
            std::string reasoning;
            if (m.contains("reasoning_content") && m["reasoning_content"].is_string()) {
                reasoning = m["reasoning_content"].get<std::string>();
            } else if (c.find("</think>") != std::string::npos) {
                std::string r = c.substr(0, c.find("</think>"));
                while (!r.empty() && r.back() == '\n') r.pop_back();
                const size_t b = r.rfind("<think>");
                if (b != std::string::npos) r = r.substr(b + 7);
                while (!r.empty() && r.front() == '\n') r.erase(0, 1);
                reasoning = r;
                c = c.substr(c.rfind("</think>") + 8);
                while (!c.empty() && c.front() == '\n') c.erase(0, 1);
            }
            reasoning = trim(reasoning);
            if (i > last_query) out += "<|im_start|>assistant\n<think>\n" + reasoning + "\n</think>\n\n" + c;
            else out += "<|im_start|>assistant\n" + c;
            if (m.contains("tool_calls") && m["tool_calls"].is_array()) {
                bool first = true;
                for (auto tc : m["tool_calls"]) {
                    if (tc.contains("function")) tc = tc["function"];
                    const std::string name = tc.value("name", "");
                    if (first) out += !trim(c).empty() ? "\n\n<tool_call>\n<function=" + name + ">\n" : "<tool_call>\n<function=" + name + ">\n";
                    else out += "\n<tool_call>\n<function=" + name + ">\n";
                    first = false;
                    json args = tc.contains("arguments") ? tc["arguments"] : json::object();
                    if (args.is_string()) { try { args = json::parse(args.get<std::string>()); } catch (...) { args = json::object(); } }
                    if (args.is_object())
                        for (auto it = args.begin(); it != args.end(); ++it) {
                            out += "<parameter=" + it.key() + ">\n";
                            out += (it.value().is_object() || it.value().is_array()) ? pydumps(it.value()) : pystr(it.value());
                            out += "\n</parameter>\n";
                        }
                    out += "</function>\n</tool_call>";
                }
            }
            out += "<|im_end|>\n";
        } else if (role == "tool") {
            if (i == 0 || msgs[i - 1].value("role", "") != "tool") out += "<|im_start|>user";
            out += "\n<tool_response>\n" + c + "\n</tool_response>";
            if (i + 1 == msgs.size() || msgs[i + 1].value("role", "") != "tool") out += "<|im_end|>\n";
        }
    }
    out += "<|im_start|>assistant\n";
    out += thinking ? "<think>\n" : "<think>\n\n</think>\n\n";
    return out;
}

// parse <tool_call><function=NAME><parameter=P>\nV\n</parameter>...</function></tool_call> blocks out of the answer text
static json parse_tool_calls(const std::string & text, const json & tools, std::string & content_out) {
    json calls = json::array();
    const size_t first = text.find("<tool_call>");
    content_out = trim(first == std::string::npos ? text : text.substr(0, first));
    size_t pos = first;
    int n = 0;
    while (pos != std::string::npos) {
        const size_t end = text.find("</tool_call>", pos);
        const std::string blk = text.substr(pos + 11, (end == std::string::npos ? text.size() : end) - pos - 11);
        const size_t f = blk.find("<function=");
        if (f != std::string::npos) {
            const size_t fe = blk.find('>', f);
            const std::string name = trim(blk.substr(f + 10, fe - f - 10));
            json schema_props = json::object();
            if (tools.is_array())
                for (auto & t : tools) {
                    const json & fn = t.contains("function") ? t["function"] : t;
                    if (fn.value("name", "") == name && fn.contains("parameters") && fn["parameters"].contains("properties")) schema_props = fn["parameters"]["properties"];
                }
            json args = json::object();
            size_t p = fe;
            while ((p = blk.find("<parameter=", p)) != std::string::npos) {
                const size_t pe = blk.find('>', p);
                const std::string pname = trim(blk.substr(p + 11, pe - p - 11));
                size_t vend = blk.find("</parameter>", pe);
                if (vend == std::string::npos) vend = blk.find("<parameter=", pe + 1);
                if (vend == std::string::npos) vend = blk.find("</function>", pe);
                if (vend == std::string::npos) vend = blk.size();
                std::string v = blk.substr(pe + 1, vend - pe - 1);
                if (!v.empty() && v.front() == '\n') v.erase(0, 1);
                if (!v.empty() && v.back() == '\n') v.pop_back();
                std::string type = schema_props.contains(pname) ? schema_props[pname].value("type", "") : "";
                if (type == "string") args[pname] = v;
                else { try { args[pname] = json::parse(v); } catch (...) { args[pname] = v; } }
                p = vend;
            }
            calls.push_back({{"id", "call_" + std::to_string((long long) now_ms()) + "_" + std::to_string(n++)}, {"type", "function"},
                             {"function", {{"name", name}, {"arguments", jdump(args)}}}});
        }
        pos = end == std::string::npos ? std::string::npos : text.find("<tool_call>", end);
    }
    return calls;
}

struct Sampler {
    float temp = 1.0f, top_p = 0.95f, min_p = 0.0f, presence = 0.0f;
    int top_k = 20;
    std::mt19937_64 rng{std::random_device{}()};
    std::vector<std::pair<float, int>> cand;
    int sample(const float * lg, const std::vector<int> & seen_counts_ids) {
        // presence penalty applied to candidate logits only (cheap)
        const int k = top_k > 0 ? std::min(top_k, 256) : 256;
        cand.clear();
        cand.reserve(k + 1);
        float thr = -1e30f;
        for (int v = 0; v < VOCAB; ++v) {
            if (lg[v] <= thr && (int) cand.size() >= k) continue;
            cand.push_back({lg[v], v});
            if ((int) cand.size() > 2 * k) {
                std::nth_element(cand.begin(), cand.begin() + k, cand.end(), [](auto & a, auto & b) { return a.first > b.first; });
                cand.resize(k);
                thr = std::min_element(cand.begin(), cand.end())->first;
            }
        }
        std::sort(cand.begin(), cand.end(), [](auto & a, auto & b) { return a.first > b.first; });
        if ((int) cand.size() > k) cand.resize(k);
        if (presence != 0.f)
            for (auto & c : cand) if (std::binary_search(seen_counts_ids.begin(), seen_counts_ids.end(), c.second)) c.first -= presence;
        std::sort(cand.begin(), cand.end(), [](auto & a, auto & b) { return a.first > b.first; });
        if (temp <= 0.f) return cand[0].second;
        std::vector<double> p(cand.size());
        double z = 0;
        for (size_t i = 0; i < cand.size(); ++i) { p[i] = std::exp((cand[i].first - cand[0].first) / temp); z += p[i]; }
        size_t n = cand.size();
        double cum = 0;
        for (size_t i = 0; i < cand.size(); ++i) {
            if (p[i] / z < min_p * (p[0] / z)) { n = i; break; }
            cum += p[i] / z;
            if (cum >= top_p) { n = i + 1; break; }
        }
        n = std::max<size_t>(n, 1);
        double zz = 0;
        for (size_t i = 0; i < n; ++i) zz += p[i];
        double r = std::uniform_real_distribution<double>(0, zz)(rng);
        for (size_t i = 0; i < n; ++i) { r -= p[i]; if (r <= 0) return cand[i].second; }
        return cand[n - 1].second;
    }
};

// ---------------- multi-slot scheduler ----------------
// one engine thread owns the GPU; HTTP threads submit requests and read events from a per-request channel
struct Event { int kind; std::string r, c; std::string fin; json timings; int n_prompt = 0, n_gen = 0; std::string err; };   // 0 text, 1 done
struct ImageSeg { size_t start; int n, nx, ny; std::vector<float> embd; std::shared_ptr<mtmd::input_chunks> chunks; size_t chunk_idx; };   // prompt[start .. start+n) are this image's rows
struct Request {
    std::vector<int> prompt;           // image rows carry negative pseudo ids (hash of the image) for prefix matching
    std::vector<ImageSeg> images;
    Sampler smp;
    int max_tokens = 0;
    std::vector<std::string> stops;
    bool thinking = true;
    std::mutex mu;
    std::condition_variable cv;
    std::deque<Event> ev;
    std::atomic<bool> cancelled{false};
    void push(Event e) { { std::lock_guard<std::mutex> lk(mu); ev.push_back(std::move(e)); } cv.notify_all(); }
    bool pop(Event & e, int timeout_ms) {
        std::unique_lock<std::mutex> lk(mu);
        if (!cv.wait_for(lk, std::chrono::milliseconds(timeout_ms), [&] { return !ev.empty(); })) return false;
        e = std::move(ev.front()); ev.pop_front();
        return true;
    }
};

struct Slot {
    std::shared_ptr<Request> req;
    std::vector<int> cached;           // tokens whose effect is in this slot's state
    int phase = 0;                     // 0 idle, 1 prompt, 2 generating
    size_t prompt_pos = 0;             // next prompt token to process
    size_t ckpt_pos = 0;               // prompt position where the recurrent state is checkpointed (before the generation prompt)
    int y = -1, p = 0;
    std::vector<int> pend, seen;
    bool in_think = false;
    std::string text_all;
    int n_gen = 0, n_prompt_eval = 0;
    int kv_reserve = 0;                // KV blocks promised to this request at admission
    double t_start = 0, t_gen0 = 0, t_prompt = 0;
};

struct Server {
    Engine & eng;
    const llama_vocab * vocab;
    std::string model_name = "qwen3.6-35b-a3b";
    std::vector<Slot> slots;
    std::mutex qmu;
    std::condition_variable qcv;
    std::deque<std::shared_ptr<Request>> waiting;
    float * lg_host;
    int rr = 0;                        // round-robin pointer for prompt chunks
    mtmd_context * mctx = nullptr;     // vision encoder (CPU), optional
    int im_start_id = -1;
    std::mutex mtmd_mu;

    Server(Engine & e, const llama_vocab * v) : eng(e), vocab(v), slots(e.n_slots) {
        lg_host = sycl::malloc_host<float>((size_t) 8 * VOCAB, eng.q);
        llama_token t[4];
        if (llama_tokenize(vocab, "<|im_start|>", 12, t, 4, false, true) == 1) im_start_id = t[0];
        // KV pool full: drop the cache of an idle slot (least recently used first)
        eng.on_pool_full = [this](int need) {
            int victim = -1;
            for (int i = 0; i < (int) slots.size(); ++i)
                if (i != need && slots[i].phase == 0 && !eng.slot_blocks[i].empty() && (victim < 0 || slots[i].t_start < slots[victim].t_start)) victim = i;
            if (victim < 0) return false;
            eng.release_kv(victim, 0);
            slots[victim].cached.clear();
            if (victim < (int) eng.ckpt.size()) eng.ckpt[victim].len = 0;
            fprintf(stderr, "kv pool: evicted idle slot %d\n", victim);
            return true;
        };
    }

    std::string piece(int t) {
        char buf[512];
        const int l = llama_token_to_piece(vocab, t, buf, sizeof(buf), 0, true);
        return std::string(buf, std::max(0, l));
    }

    void submit(const std::shared_ptr<Request> & r) { { std::lock_guard<std::mutex> lk(qmu); waiting.push_back(r); } qcv.notify_all(); }

    // sample from device logits row (greedy uses the device argmax)
    int pick(Slot & s, const float * dev_logits, const int32_t * dev_tok) {
        Sampler & smp = s.req->smp;
        if (smp.temp <= 0.f && smp.presence == 0.f) { int t; eng.q.memcpy(&t, dev_tok, 4).wait(); return t; }
        eng.q.memcpy(lg_host, dev_logits, VOCAB * 4).wait();
        return smp.sample(lg_host, s.seen);
    }

    void finish(Slot & s, const std::string & reason, const std::string & err = "") {
        Event e; e.kind = 1; e.fin = reason; e.err = err; e.n_prompt = (int) s.req->prompt.size(); e.n_gen = s.n_gen;
        const double tg = now_ms() - s.t_gen0;
        e.timings = json{{"prompt_n", s.n_prompt_eval}, {"prompt_ms", s.t_prompt}, {"prompt_per_second", s.t_prompt > 0 ? s.n_prompt_eval / (s.t_prompt / 1e3) : 0},
                         {"predicted_n", s.n_gen}, {"predicted_ms", tg}, {"predicted_per_second", tg > 0 ? s.n_gen / (tg / 1e3) : 0}};
        s.req->push(std::move(e));
        s.req.reset();
        s.phase = 0;
    }

    // one generated token for slot s; false when the request is complete (finish already sent)
    bool emit(Slot & s, int tk) {
        Request & r = *s.req;
        ++s.n_gen;
        if (r.cancelled) { finish(s, "stop"); return false; }
        if (llama_vocab_is_eog(vocab, tk)) { finish(s, "stop"); return false; }
        s.seen.insert(std::upper_bound(s.seen.begin(), s.seen.end(), tk), tk);
        const std::string p = piece(tk);
        if (s.in_think && p == "</think>") { s.in_think = false; }
        else {
            s.text_all += p;
            for (auto & st : r.stops)
                if (!st.empty() && !s.in_think && s.text_all.size() >= st.size() && s.text_all.compare(s.text_all.size() - st.size(), st.size(), st) == 0) { finish(s, "stop"); return false; }
            Event e; e.kind = 0;
            if (s.in_think) e.r = p; else e.c = p;
            r.push(std::move(e));
        }
        if (s.n_gen >= r.max_tokens) { finish(s, "length"); return false; }
        if (s.p + 2 >= eng.max_ctx) { finish(s, "length"); return false; }
        return true;
    }

    void assign_waiting() {
        std::lock_guard<std::mutex> lk(qmu);
        while (!waiting.empty()) {
            auto r = waiting.front();
            // free slot with the longest cached prefix of this prompt
            // free slot with the longest reusable prefix: its whole cached sequence, or its prompt checkpoint
            int best = -1; size_t best_len = 0; bool best_ck = false;
            for (int i = 0; i < (int) slots.size(); ++i) {
                if (slots[i].phase != 0) continue;
                const auto & c = slots[i].cached;
                size_t m = 0; bool ck = false;
                if (c.size() < r->prompt.size() && std::equal(c.begin(), c.end(), r->prompt.begin())) m = c.size();
                else if (i < (int) eng.ckpt.size() && eng.ckpt[i].len > 0 && eng.ckpt[i].len < r->prompt.size() && eng.ckpt[i].len <= c.size() &&
                         std::equal(c.begin(), c.begin() + eng.ckpt[i].len, r->prompt.begin())) { m = eng.ckpt[i].len; ck = true; }
                if (best < 0 || m > best_len) { best = i; best_len = m; best_ck = ck; }
            }
            if (best < 0) return;
            // admission: KV blocks this request needs beyond what the slot keeps, vs free + evictable (other idle slots)
            {
                const int B = Engine::KV_BLK;
                const size_t want_pos = r->prompt.size() + (size_t) std::min(r->max_tokens, 4096);
                const int need = (int) ((want_pos + B - 1) / B) - (int) ((best_len + B - 1) / B);
                int avail = (int) eng.free_blocks.size();
                bool busy = false;
                for (int i = 0; i < (int) slots.size(); ++i) {
                    if (slots[i].phase != 0) {     // running: blocks it was promised but has not allocated yet are taken
                        busy = true;
                        avail -= std::max(0, slots[i].kv_reserve - (int) eng.slot_blocks[i].size());
                    } else if (i != best) avail += (int) eng.slot_blocks[i].size();
                }
                if (best_len == 0) avail += (int) eng.slot_blocks[best].size();   // the slot's own old cache is dropped
                if (need > avail) {
                    if (busy) return;                      // wait until running requests finish and their caches can be evicted
                    waiting.pop_front();
                    Event e; e.kind = 1; e.fin = "error"; e.err = "prompt does not fit in the KV cache pool";
                    r->push(std::move(e));
                    continue;
                }
            }
            waiting.pop_front();
            Slot & s = slots[best];
            s.kv_reserve = (int) ((r->prompt.size() + (size_t) std::min(r->max_tokens, 4096) + Engine::KV_BLK - 1) / Engine::KV_BLK);
            s.req = r;
            if (best_len > 0 && best_ck) { eng.ckpt_restore(best); s.cached.resize(best_len); eng.release_kv(best, (int) best_len); }
            if (best_len == 0) { eng.sq = best; eng.reset(); s.cached.clear(); }
            s.prompt_pos = best_len;
            s.ckpt_pos = 0;
            for (size_t k = r->prompt.size(); k-- > 0;) if (r->prompt[k] == im_start_id) { s.ckpt_pos = k; break; }
            s.n_prompt_eval = (int) (r->prompt.size() - best_len);
            s.phase = 1; s.n_gen = 0; s.seen.clear(); s.text_all.clear(); s.in_think = r->thinking;
            s.t_start = now_ms();
        }
    }

    void run_prompt_chunk() {
        for (int k = 0; k < (int) slots.size(); ++k) {
            const int i = (rr + k) % (int) slots.size();
            Slot & s = slots[i];
            if (s.phase != 1) continue;
            rr = i + 1;
            if (s.req->cancelled) { finish(s, "stop"); return; }
            const auto & pr = s.req->prompt;
            eng.sq = i;
            int n = (int) std::min<size_t>(T_MAX, pr.size() - s.prompt_pos);
            if (s.prompt_pos < s.ckpt_pos) n = (int) std::min<size_t>(n, s.ckpt_pos - s.prompt_pos);
            const ImageSeg * img = nullptr;
            for (auto & im : s.req->images) {
                if (s.prompt_pos >= im.start && s.prompt_pos < im.start + im.n) { img = &im; n = (int) std::min<size_t>(n, im.start + im.n - s.prompt_pos); break; }
                if (im.start > s.prompt_pos) n = (int) std::min<size_t>(n, im.start - s.prompt_pos);
            }
            if (img && img->embd.empty()) {
                ImageSeg & im = const_cast<ImageSeg &>(*img);
                std::lock_guard<std::mutex> lk(mtmd_mu);
                const double t0 = now_ms();
                set_affinity(-1);     // the encoder's worker threads inherit this thread's CPU mask
                const int rc = mtmd_encode_chunk(mctx, (*im.chunks)[im.chunk_idx]);
                set_affinity(0);
                if (rc != 0) { fprintf(stderr, "image encoding failed\n"); finish(s, "stop"); return; }
                const float * e = mtmd_get_output_embd(mctx);
                im.embd.assign(e, e + (size_t) im.n * D);
                fprintf(stderr, "image encoded: %d tokens in %.0f ms\n", im.n, now_ms() - t0);
            }
            if (img) {   // image rows: embeddings in, rope (t, h, w) = (p0, p0 + row, p0 + col) with p0 the rope position at the image start
                const int p0 = (int) img->start - eng.rope_delta[i];
                std::vector<int32_t> rp((size_t) n * 3);
                for (int k = 0; k < n; ++k) {
                    const int j = (int) (s.prompt_pos - img->start) + k;
                    rp[k * 3] = p0; rp[k * 3 + 1] = p0 + j / img->nx; rp[k * 3 + 2] = p0 + j % img->nx;
                }
                eng.q.memcpy(eng.RPOS, rp.data(), rp.size() * 4).wait();
                eng.embd_in = img->embd.data() + (s.prompt_pos - img->start) * D;
                eng.use_rpos = true;
            }
            try { eng.ensure_kv(i, (int) (s.prompt_pos + n)); }
            catch (const std::exception & ex) {
                eng.embd_in = nullptr; eng.use_rpos = false;
                finish(s, "error", ex.what());
                return;
            }
            eng.prefill(pr.data() + s.prompt_pos, n, (int) s.prompt_pos);
            const bool last = s.prompt_pos + n == pr.size();
            int first = -1;
            if (last) first = pick(s, eng.logits, eng.tok);
            eng.mtp_fill_chunk(pr.data() + s.prompt_pos, n, (int) s.prompt_pos);
            eng.q.wait();
            eng.embd_in = nullptr;
            eng.use_rpos = false;
            s.cached.insert(s.cached.end(), pr.begin() + s.prompt_pos, pr.begin() + s.prompt_pos + n);
            s.prompt_pos += n;
            if (img && s.prompt_pos == img->start + img->n) eng.rope_delta[i] += img->n - std::max(img->nx, img->ny);
            if (s.prompt_pos == s.ckpt_pos && s.ckpt_pos > 0) eng.ckpt_save(i, s.ckpt_pos);
            if (last) {
                s.t_prompt = now_ms() - s.t_start;
                s.t_gen0 = now_ms();
                s.phase = 2;
                s.p = (int) s.cached.size();
                s.y = first;
                s.pend = {first};
                if (eng.has_mtp) eng.spec_begin(i);
                emit(s, first);
            }
            return;
        }
    }

    void run_gen_step() {
        std::vector<int> act;
        for (int i = 0; i < (int) slots.size(); ++i) if (slots[i].phase == 2) act.push_back(i);
        if (act.empty()) return;
        if (!eng.has_mtp) {   // plain decode, one slot after another
            for (int i : act) {
                Slot & s = slots[i];
                eng.sq = i;
                eng.decode(s.y, s.p);
                s.cached.push_back(s.y); s.p++;
                s.y = pick(s, eng.logits, eng.tok);
                emit(s, s.y);
            }
            return;
        }
        {
            std::vector<int> ok;
            for (int i : act) {
                try { eng.ensure_kv(i, slots[i].p + 2); ok.push_back(i); }
                catch (const std::exception & ex) { finish(slots[i], "error", ex.what()); }
            }
            act.swap(ok);
            if (act.empty()) return;
        }
        std::vector<int> bt, drafts;
        std::vector<SeqGroup> gr;
        for (int i : act) {
            Slot & s = slots[i];
            const int d = eng.mtp_draft(i, s.pend, s.p);
            drafts.push_back(d);
            gr.push_back({i, s.p, 2, 0});
            bt.push_back(s.y); bt.push_back(d);
        }
        eng.verify_batch(bt, gr);
        std::vector<int> tb(bt.size());
        eng.q.memcpy(tb.data(), eng.TOKB, tb.size() * 4).wait();
        for (size_t k = 0; k < act.size(); ++k) {
            Slot & s = slots[act[k]];
            const int d = drafts[k];
            const bool greedy = s.req->smp.temp <= 0.f && s.req->smp.presence == 0.f;
            const int z1 = greedy ? tb[2 * k] : pick(s, eng.LOGB + 2 * k * VOCAB, eng.TOKB + 2 * k);
            if (z1 == d) {
                const int z2 = greedy ? tb[2 * k + 1] : pick(s, eng.LOGB + (2 * k + 1) * VOCAB, eng.TOKB + 2 * k + 1);
                eng.accept_update(act[k], (int) (2 * k), true);
                s.cached.push_back(s.y); s.cached.push_back(d);
                s.pend = {d, z2}; s.y = z2; s.p += 2;
                if (emit(s, d)) emit(s, z2);
            } else {
                eng.accept_update(act[k], (int) (2 * k), false);
                s.cached.push_back(s.y);
                s.pend = {z1}; s.y = z1; s.p += 1;
                emit(s, z1);
            }
        }
    }

    static void set_affinity(int cpu) {   // cpu < 0: all CPUs
        cpu_set_t cs; CPU_ZERO(&cs);
        if (cpu >= 0) CPU_SET(cpu, &cs); else for (int c = 0; c < (int) std::thread::hardware_concurrency(); ++c) CPU_SET(c, &cs);
        pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs);
    }
    void loop() {
        set_affinity(0);
        while (true) {
            assign_waiting();
            bool any_prompt = false, any_gen = false;
            for (auto & s : slots) { any_prompt |= s.phase == 1; any_gen |= s.phase == 2; }
            if (!any_prompt && !any_gen) {
                std::unique_lock<std::mutex> lk(qmu);
                qcv.wait_for(lk, std::chrono::milliseconds(200), [&] { return !waiting.empty(); });
                continue;
            }
            if (any_prompt) run_prompt_chunk();
            if (any_gen) run_gen_step();
            eng.q.wait();
        }
    }

    void handle_chat(const httplib::Request & hreq, httplib::Response & res) {
        json body;
        try { body = json::parse(hreq.body); } catch (...) { res.status = 400; res.set_content(R"({"error":{"message":"invalid JSON"}})", "application/json"); return; }
        bool thinking = true;
        if (body.contains("chat_template_kwargs") && body["chat_template_kwargs"].contains("enable_thinking"))
            thinking = body["chat_template_kwargs"]["enable_thinking"].get<bool>();
        if (body.value("reasoning_effort", "") == "none") thinking = false;
        bool has_image = false;
        const json tools = body.contains("tools") && body["tools"].is_array() && body.value("tool_choice", json("auto")) != json("none") ? body["tools"] : json::array();
        std::vector<std::string> image_urls;
        std::string text;
        {
            std::lock_guard<std::mutex> lk(mtmd_mu);   // g_images is shared
            g_images = mctx ? &image_urls : nullptr;
            text = render_chat(body.value("messages", json::array()), tools, thinking, has_image);
            g_images = nullptr;
        }
        auto err = [&](const char * m) { res.status = 400; res.set_content(jdump(json{{"error", {{"message", m}}}}), "application/json"); };
        if (has_image && !mctx) { err("images need the server started with --mmproj"); return; }
        auto r = std::make_shared<Request>();
        if (image_urls.empty()) {
            r->prompt.resize(text.size() + 16);
            r->prompt.resize(llama_tokenize(vocab, text.c_str(), text.size(), r->prompt.data(), r->prompt.size(), false, true));
        } else {
            std::vector<mtmd::bitmap> bmps;
            std::vector<std::string> ids;
            for (auto & u : image_urls) {
                const size_t comma = u.find(',');
                if (u.rfind("data:", 0) != 0 || comma == std::string::npos) { err("only base64 data: URLs are supported for images"); return; }
                const std::string bytes = b64_decode(u.substr(comma + 1));
                const mtmd_helper_bitmap_wrapper bw = mtmd_helper_bitmap_init_from_buf(mctx, (const unsigned char *) bytes.data(), bytes.size(), false, mtmd_helper_init_opt_default());
                mtmd::bitmap bm(bw.bitmap);
                if (!bm.ptr) { err("could not decode image"); return; }
                uint64_t h = 1469598103934665603ULL;
                for (unsigned char c : bytes) { h ^= c; h *= 1099511628211ULL; }
                bm.set_id(std::to_string(h).c_str());
                ids.push_back(std::to_string(h));
                bmps.push_back(std::move(bm));
            }
            std::lock_guard<std::mutex> lk(mtmd_mu);
            auto shared_chunks = std::make_shared<mtmd::input_chunks>(mtmd_input_chunks_init());
            mtmd::input_chunks & chunks = *shared_chunks;
            mtmd_input_text txt{};
            txt.text = text.c_str();
            txt.text_len = text.size();
            txt.add_special = false;
            txt.parse_special = true;
            std::vector<const mtmd_bitmap *> bp;
            for (auto & b : bmps) bp.push_back(b.ptr.get());
            if (mtmd_tokenize(mctx, chunks.ptr.get(), &txt, bp.data(), bp.size()) != 0) {
                fprintf(stderr, "tokenize failed, marker '%s', prompt: %.300s\n", mtmd_default_marker(), text.c_str());
                err("image tokenization failed"); return;
            }
            int img_i = 0;
            for (size_t ci = 0; ci < chunks.size(); ++ci) {
                const mtmd_input_chunk * ch = chunks[ci];
                if (mtmd_input_chunk_get_type(ch) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
                    size_t nt = 0;
                    const llama_token * tk = mtmd_input_chunk_get_tokens_text(ch, &nt);
                    r->prompt.insert(r->prompt.end(), tk, tk + nt);
                } else if (mtmd_input_chunk_get_type(ch) == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
                    const mtmd_image_tokens * it = mtmd_input_chunk_get_tokens_image(ch);
                    ImageSeg seg;
                    seg.start = r->prompt.size();
                    seg.n = (int) mtmd_input_chunk_get_n_tokens(ch);
                    seg.nx = (int) mtmd_image_tokens_get_nx(it);
                    seg.ny = (int) mtmd_image_tokens_get_ny(it);
                    seg.chunks = shared_chunks;
                    seg.chunk_idx = ci;     // encoded later on the engine thread, only if this part is not reused from a cache
                    const uint64_t hid = std::stoull(ids[std::min<size_t>(img_i, ids.size() - 1)]);
                    for (int k = 0; k < seg.n; ++k) r->prompt.push_back(-1 - (int) ((hid + (uint64_t) k * 2654435761ULL) % 1000000000ULL));
                    r->images.push_back(std::move(seg));
                    fprintf(stderr, "image %d: %d tokens (%dx%d)\n", img_i, r->images.back().n, r->images.back().nx, r->images.back().ny);
                    ++img_i;
                }
            }
        }
        if ((int) r->prompt.size() >= eng.max_ctx - 2) { res.status = 400; res.set_content(R"({"error":{"message":"prompt exceeds the context size"}})", "application/json"); return; }
        r->smp.temp = body.value("temperature", 0.7f);
        r->smp.top_p = body.value("top_p", 0.8f);
        r->smp.top_k = body.value("top_k", 20);
        r->smp.min_p = body.value("min_p", 0.0f);
        r->smp.presence = body.value("presence_penalty", 0.0f);
        if (body.contains("seed") && body["seed"].is_number_integer()) r->smp.rng.seed(body["seed"].get<int64_t>());
        int max_tokens = body.value("max_tokens", body.value("max_completion_tokens", 32768));
        r->max_tokens = std::max(1, std::min(max_tokens, eng.max_ctx - (int) r->prompt.size() - 2));
        if (body.contains("stop")) { if (body["stop"].is_string()) r->stops.push_back(body["stop"]); else if (body["stop"].is_array()) for (auto & s2 : body["stop"]) r->stops.push_back(s2); }
        r->thinking = thinking;
        const bool stream = body.value("stream", false);
        const bool usage_in_stream = body.contains("stream_options") && body["stream_options"].value("include_usage", false);
        const std::string id = "chatcmpl-" + std::to_string((long long) now_ms());
        const long created = (long) time(nullptr);
        const int n_prompt = (int) r->prompt.size();
        submit(r);

        if (!stream) {
            std::string reasoning, content;
            Event e;
            while (true) {
                if (!r->pop(e, 1000)) continue;
                if (e.kind == 0) { reasoning += e.r; content += e.c; continue; }
                break;
            }
            if (!e.err.empty()) { res.status = 503; res.set_content(jdump(json{{"error", {{"message", e.err}, {"type", "server_error"}}}}), "application/json"); return; }
            while (!content.empty() && content.front() == '\n' && thinking) content.erase(0, 1);
            json msg = {{"role", "assistant"}, {"content", content}};
            std::string fin2 = e.fin;
            if (!tools.empty() && content.find("<tool_call>") != std::string::npos) {
                std::string text_part;
                const json calls = parse_tool_calls(content, tools, text_part);
                msg["content"] = text_part.empty() ? json(nullptr) : json(text_part);
                if (!calls.empty()) { msg["tool_calls"] = calls; fin2 = "tool_calls"; }
            }
            if (thinking) msg["reasoning_content"] = reasoning;
            json out = {{"id", id}, {"object", "chat.completion"}, {"created", created}, {"model", model_name},
                        {"choices", json::array({json{{"index", 0}, {"message", msg}, {"finish_reason", fin2}}})},
                        {"usage", {{"prompt_tokens", n_prompt}, {"completion_tokens", e.n_gen}, {"total_tokens", n_prompt + e.n_gen}}},
                        {"timings", e.timings}};
            res.set_content(jdump(out), "application/json");
            return;
        }
        res.set_chunked_content_provider("text/event-stream", [=, this](size_t, httplib::DataSink & sink) mutable {
            auto send = [&](const json & j) { const std::string s2 = "data: " + jdump(j) + "\n\n"; return sink.write(s2.data(), s2.size()); };
            auto chunk = [&](const json & delta, const json & fin) {
                return json{{"id", id}, {"object", "chat.completion.chunk"}, {"created", created}, {"model", model_name},
                            {"choices", json::array({json{{"index", 0}, {"delta", delta}, {"finish_reason", fin}}})}};
            };
            if (!send(chunk({{"role", "assistant"}, {"content", ""}}, nullptr))) { r->cancelled = true; return false; }
            bool content_started = false;
            std::string pend_r, pend_c, all_c;
            size_t sent_c = 0;
            const bool has_tools = !tools.empty();
            static const std::string TC = "<tool_call>";
            Event e;
            while (true) {
                if (!r->pop(e, 1000)) { if (!sink.is_writable()) { r->cancelled = true; return false; } continue; }
                if (e.kind == 1) break;
                bool ok = true;
                if (!e.r.empty()) {
                    pend_r += e.r;
                    const size_t k = utf8_complete_len(pend_r);
                    if (k) { ok = send(chunk({{"reasoning_content", pend_r.substr(0, k)}}, nullptr)); pend_r.erase(0, k); }
                } else if (!e.c.empty()) {
                    pend_c += e.c;
                    const size_t k = utf8_complete_len(pend_c);
                    if (k) {
                        std::string cc = pend_c.substr(0, k);
                        pend_c.erase(0, k);
                        if (has_tools) {
                            all_c += cc;
                            size_t limit = all_c.find(TC);
                            if (limit == std::string::npos) {
                                limit = all_c.size();
                                for (size_t L2 = std::min(TC.size() - 1, all_c.size()); L2 > 0; --L2)
                                    if (all_c.compare(all_c.size() - L2, L2, TC, 0, L2) == 0) { limit = all_c.size() - L2; break; }
                            }
                            limit = std::max(limit, sent_c);
                            cc = all_c.substr(sent_c, limit - sent_c);
                            sent_c = limit;
                        }
                        if (!content_started && thinking) { while (!cc.empty() && cc.front() == '\n') cc.erase(0, 1); }
                        if (!cc.empty()) { content_started = true; ok = send(chunk({{"content", cc}}, nullptr)); }
                    }
                }
                if (!ok) { r->cancelled = true; return false; }
            }
            if (!e.err.empty()) {
                send(json{{"error", {{"message", e.err}, {"type", "server_error"}}}});
                const std::string done = "data: [DONE]\n\n";
                sink.write(done.data(), done.size());
                sink.done();
                return true;
            }
            std::string fin2 = e.fin;
            if (has_tools && all_c.find(TC) != std::string::npos) {
                std::string text_part;
                json calls = parse_tool_calls(all_c, tools, text_part);
                if (!calls.empty()) {
                    for (size_t i = 0; i < calls.size(); ++i) calls[i]["index"] = (int) i;
                    send(chunk({{"tool_calls", calls}}, nullptr));
                    fin2 = "tool_calls";
                }
            } else if (has_tools && sent_c < all_c.size()) {
                send(chunk({{"content", all_c.substr(sent_c)}}, nullptr));
            }
            json last = chunk(json::object(), fin2);
            last["timings"] = e.timings;
            send(last);
            if (usage_in_stream) {
                json u = {{"id", id}, {"object", "chat.completion.chunk"}, {"created", created}, {"model", model_name}, {"choices", json::array()},
                          {"usage", {{"prompt_tokens", n_prompt}, {"completion_tokens", e.n_gen}, {"total_tokens", n_prompt + e.n_gen}}}};
                send(u);
            }
            const std::string done = "data: [DONE]\n\n";
            sink.write(done.data(), done.size());
            sink.done();
            return true;
        });
    }

    void run(const std::string & host, int port) {
        std::thread worker([this] { loop(); });
        worker.detach();
        httplib::Server svr;
        svr.new_task_queue = [] { return new httplib::ThreadPool(16); };
        svr.set_read_timeout(600);
        svr.set_write_timeout(600);
        svr.Get("/health", [](const httplib::Request &, httplib::Response & res) { res.set_content(R"({"status":"ok"})", "application/json"); });
        svr.Get("/v1/models", [this](const httplib::Request &, httplib::Response & res) {
            res.set_content(json{{"object", "list"}, {"data", json::array({json{{"id", model_name}, {"object", "model"}, {"owned_by", "qe"}}})}}.dump(), "application/json");
        });
        svr.Get("/props", [this](const httplib::Request &, httplib::Response & res) {
            res.set_content(json{{"total_slots", (int) slots.size()}, {"n_ctx_per_slot", eng.max_ctx}, {"kv_pool", eng.pool_ctx},
                                 {"kv_pool_used", eng.kv_used_blocks() * Engine::KV_BLK}, {"mtp", eng.has_mtp}}.dump(), "application/json");
        });
        svr.Post("/v1/chat/completions", [this](const httplib::Request & req, httplib::Response & res) { handle_chat(req, res); });
        svr.Post("/chat/completions", [this](const httplib::Request & req, httplib::Response & res) { handle_chat(req, res); });
        printf("listening on http://%s:%d (%d slots, up to %d ctx each, shared KV pool %d)\n", host.c_str(), port, (int) slots.size(), eng.max_ctx, eng.pool_ctx);
        fflush(stdout);
        svr.listen(host, port);
    }
};

// ============================ main ============================
int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: qe render <request.json> | qe check ...\n"); return 1; }
    if (argc < 4 && std::string(argv[1]) != "render") { fprintf(stderr, "usage: qe check <pack> <ref.bin> | qe gen <pack> <vocab.gguf> <prompt> [-n N] | qe serve <pack> <vocab.gguf> [--port P] [--host H]\n"); return 1; }
    const std::string mode = argv[1];
    if (mode == "render") {   // qe render <request.json> : print the prompt the server would build (template check)
        std::ifstream f(argv[2]); std::stringstream ss; ss << f.rdbuf();
        const json body = json::parse(ss.str());
        bool thinking = true;
        if (body.contains("chat_template_kwargs") && body["chat_template_kwargs"].contains("enable_thinking")) thinking = body["chat_template_kwargs"]["enable_thinking"].get<bool>();
        bool has_image = false;
        fputs(render_chat(body["messages"], body.value("tools", json::array()), thinking, has_image).c_str(), stdout);
        return 0;
    }
    int resident = 7000, ctx = 32768, n_gen = 128, port = 8090;
    std::string host = "0.0.0.0";
    bool use_prefill = false;
    std::string mtp_path;
    int slots = 1, kv_pool = 0;
    std::string mmproj_path;
    std::vector<std::string> stats;
    for (int i = 4; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--resident") resident = atoi(argv[++i]);
        else if (a == "--ctx") ctx = atoi(argv[++i]);
        else if (a == "-n") n_gen = atoi(argv[++i]);
        else if (a == "--port") port = atoi(argv[++i]);
        else if (a == "--host") host = argv[++i];
        else if (a == "--prefill") use_prefill = true;
        else if (a == "--mtp") mtp_path = argv[++i];
        else if (a == "--slots") slots = atoi(argv[++i]);
        else if (a == "--kv-pool") kv_pool = atoi(argv[++i]);
        else if (a == "--mmproj") mmproj_path = argv[++i];
        else if (a == "--stats") { std::stringstream ss(argv[++i]); std::string s; while (std::getline(ss, s, ',')) stats.push_back(s); }
        else if (mode == "gen" && i == 4) continue;
    }
    Pack pk(argv[2]);
    Pack * mpk = mtp_path.empty() ? nullptr : new Pack(mtp_path.c_str());
    Engine eng(pk, ctx, resident, stats, mpk, slots, kv_pool);

    if (mode == "check") {
        FILE * f = fopen(argv[3], "rb");
        int n; fread(&n, 4, 1, f);
        std::vector<int> toks(n); fread(toks.data(), 4, n, f);
        std::vector<float> ref((size_t) n * VOCAB);
        fread(ref.data(), 4, ref.size(), f);
        int ng; fread(&ng, 4, 1, f);
        std::vector<int> rgen(ng); fread(rgen.data(), 4, ng, f);
        fclose(f);
        std::vector<float> lg(VOCAB);
        double kld_sum = 0; int top1 = 0;
        if (use_prefill) {
            const double t0 = now_ms();
            for (int c = 0; c < n; c += T_MAX) eng.prefill(toks.data() + c, std::min(T_MAX, n - c), c);
            printf("prefill %d tokens: %.1f ms\n", n, now_ms() - t0);
            eng.q.memcpy(lg.data(), eng.logits, VOCAB * 4).wait();
            const float * r = ref.data() + (size_t) (n - 1) * VOCAB;
            double mr = -1e30, me = -1e30; int ar = 0, ae = 0;
            for (int v = 0; v < VOCAB; ++v) { if (r[v] > mr) { mr = r[v]; ar = v; } if (lg[v] > me) { me = lg[v]; ae = v; } }
            double zr = 0, ze = 0;
            for (int v = 0; v < VOCAB; ++v) { zr += std::exp(r[v] - mr); ze += std::exp(lg[v] - me); }
            double kl = 0;
            for (int v = 0; v < VOCAB; ++v) { const double pr = std::exp(r[v] - mr) / zr; if (pr > 1e-10) kl += pr * ((r[v] - mr - std::log(zr)) - (lg[v] - me - std::log(ze))); }
            printf("last position after prefill: ref top %d engine top %d KL %.4f\n", ar, ae, kl);
        }
        for (int i = 0; i < n && !use_prefill; ++i) {
            eng.decode(toks[i], i);
            eng.q.memcpy(lg.data(), eng.logits, VOCAB * 4).wait();
            const float * r = ref.data() + (size_t) i * VOCAB;
            // KL(ref || engine) over the full vocabulary
            double mr = -1e30, me = -1e30;
            int ar = 0, ae = 0;
            for (int v = 0; v < VOCAB; ++v) { if (r[v] > mr) { mr = r[v]; ar = v; } if (lg[v] > me) { me = lg[v]; ae = v; } }
            double zr = 0, ze = 0;
            for (int v = 0; v < VOCAB; ++v) { zr += std::exp(r[v] - mr); ze += std::exp(lg[v] - me); }
            double kl = 0;
            for (int v = 0; v < VOCAB; ++v) {
                const double pr = std::exp(r[v] - mr) / zr;
                if (pr > 1e-10) kl += pr * ((r[v] - mr - std::log(zr)) - (lg[v] - me - std::log(ze)));
            }
            kld_sum += kl;
            top1 += ar == ae;
            if (i < 4 || i == n - 1 || kl > 0.5) printf("pos %3d  ref top %6d  engine top %6d  KL %.4f\n", i, ar, ae, kl);
        }
        if (!use_prefill) printf("prompt %d tokens: mean KL %.4f, top-1 agreement %.1f%%\n", n, kld_sum / n, 100.0 * top1 / n);
        int same = 0, cur = eng.argmax();
        std::vector<int> g;
        for (int k = 0; k < ng; ++k) { g.push_back(cur); eng.decode(cur, n + k); cur = eng.argmax(); }
        for (int k = 0; k < ng && g[k] == rgen[k]; ++k) same++;
        printf("greedy continuation matches the reference for the first %d of %d tokens\n", same, ng);
        return 0;
    }

    if (mode == "multi") {   // qe multi <pack> <vocab.gguf> <prompts-file, one per line> [-n N] --slots S ...
        auto mp2 = llama_model_default_params();
        mp2.vocab_only = true;
        llama_model * vm = llama_model_load_from_file(argv[3], mp2);
        const llama_vocab * vocab = llama_model_get_vocab(vm);
        std::ifstream pf(argv[4]);
        std::vector<std::string> prompts; std::string line;
        while (std::getline(pf, line)) if (!line.empty()) prompts.push_back(line);
        const int S2 = std::min((int) prompts.size(), eng.n_slots);
        struct Run { std::vector<int> toks, out, pend; int y, p; bool done = false; };
        std::vector<Run> runs(S2);
        const double t0 = now_ms();
        for (int i = 0; i < S2; ++i) {
            const std::string pr = "<|im_start|>user\n" + prompts[i] + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
            runs[i].toks.resize(pr.size() + 16);
            runs[i].toks.resize(llama_tokenize(vocab, pr.c_str(), pr.size(), runs[i].toks.data(), runs[i].toks.size(), false, true));
            eng.sq = i; eng.reset();
            for (size_t c = 0; c < runs[i].toks.size(); c += T_MAX) {
                const int n = (int) std::min<size_t>(T_MAX, runs[i].toks.size() - c);
                eng.prefill(runs[i].toks.data() + c, n, (int) c);
                if (c + n == runs[i].toks.size()) runs[i].y = eng.argmax();
                eng.mtp_fill_chunk(runs[i].toks.data() + c, n, (int) c);
            }
            eng.spec_begin(i);
            runs[i].p = (int) runs[i].toks.size();
            runs[i].pend = {runs[i].y};
            runs[i].out.push_back(runs[i].y);
        }
        const double t1 = now_ms();
        int rounds = 0, acc = 0, prop = 0;
        while (true) {
            std::vector<int> act;
            for (int i = 0; i < S2; ++i) if (!runs[i].done) act.push_back(i);
            if (act.empty()) break;
            std::vector<int> bt; std::vector<SeqGroup> gr; std::vector<int> drafts;
            for (int i : act) {
                const int d = eng.mtp_draft(i, runs[i].pend, runs[i].p);
                drafts.push_back(d);
                gr.push_back({i, runs[i].p, 2, 0});
                bt.push_back(runs[i].y); bt.push_back(d);
            }
            eng.verify_batch(bt, gr);
            std::vector<int> tb(bt.size());
            eng.q.memcpy(tb.data(), eng.TOKB, tb.size() * 4).wait();
            ++rounds;
            for (size_t k = 0; k < act.size(); ++k) {
                Run & r = runs[act[k]];
                const int z1 = tb[2 * k], d = drafts[k];
                ++prop;
                if (z1 == d) {
                    ++acc;
                    eng.accept_update(act[k], (int) (2 * k), true);
                    r.out.push_back(d); r.out.push_back(tb[2 * k + 1]);
                    r.pend = {d, tb[2 * k + 1]}; r.y = tb[2 * k + 1]; r.p += 2;
                } else {
                    eng.accept_update(act[k], (int) (2 * k), false);
                    r.out.push_back(z1);
                    r.pend = {z1}; r.y = z1; r.p += 1;
                }
                if ((int) r.out.size() >= n_gen) r.done = true;
                for (int t : r.out) if (llama_vocab_is_eog(vocab, t)) r.done = true;
            }
        }
        eng.q.wait();
        const double t2 = now_ms();
        eng.prof_report();
        long total = 0;
        for (int i = 0; i < S2; ++i) {
            std::string o;
            for (int t : runs[i].out) { char buf[256]; const int l = llama_token_to_piece(vocab, t, buf, sizeof(buf), 0, false); o.append(buf, std::max(0, l)); }
            total += runs[i].out.size();
            printf("=== slot %d (%zu tokens): %s\n", i, runs[i].out.size(), o.substr(0, 300).c_str());
        }
        printf("--- %d sequences: prompts %.0f ms; %ld tokens in %.0f ms = %.1f tok/s aggregate, %.1f tok/s per sequence; %d rounds, %.1f%% drafts accepted\n",
               S2, t1 - t0, total, t2 - t1, total / ((t2 - t1) / 1e3), total / ((t2 - t1) / 1e3) / S2, rounds, 100.0 * acc / std::max(1, prop));
        return 0;
    }
    if (mode == "serve") {
        auto mp = llama_model_default_params();
        mp.vocab_only = true;
        llama_model * vm = llama_model_load_from_file(argv[3], mp);
        Server srv(eng, llama_model_get_vocab(vm));
        if (!mmproj_path.empty()) {
            mtmd_context_params mpp = mtmd_context_params_default();
            mpp.use_gpu = false;
            mpp.n_threads = 12;
            mpp.print_timings = false;
            mpp.warmup = false;
            srv.mctx = mtmd_init_from_file(mmproj_path.c_str(), vm, mpp);
            if (!srv.mctx) { fprintf(stderr, "failed to load mmproj %s\n", mmproj_path.c_str()); return 1; }
            printf("vision encoder loaded (CPU)\n");
        }
        srv.run(host, port);
        return 0;
    }
    // gen
    auto mp = llama_model_default_params();
    mp.vocab_only = true;
    llama_model * vm = llama_model_load_from_file(argv[3], mp);
    const llama_vocab * vocab = llama_model_get_vocab(vm);
    std::string user_text = argv[4];
    if (!user_text.empty() && user_text[0] == '@') {   // @file: read the prompt from a file (long prompts exceed the argv limit)
        std::ifstream pf(user_text.substr(1)); std::stringstream ss; ss << pf.rdbuf(); user_text = ss.str();
    }
    const std::string prompt = "<|im_start|>user\n" + user_text + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    std::vector<llama_token> toks(prompt.size() + 16);
    toks.resize(llama_tokenize(vocab, prompt.c_str(), prompt.size(), toks.data(), toks.size(), false, true));
    const double t0 = now_ms();
    for (size_t c = 0; c < toks.size(); c += T_MAX) {
        const int n = (int) std::min<size_t>(T_MAX, toks.size() - c);
        eng.prefill(toks.data() + c, n, (int) c);
        const int keep = eng.argmax();
        eng.mtp_fill_chunk(toks.data() + c, n, (int) c);
        eng.q.memcpy(eng.tok, &keep, 4).wait();
    }
    int cur = eng.argmax();
    eng.prof_report();
    if (eng.has_mtp) {   // greedy speculative decoding with the MTP draft
        const double t1 = now_ms();
        std::string out;
        int produced = 0, rounds = 0, accepted = 0;
        int p = (int) toks.size();              // position of `cur`
        std::vector<int> pend{cur};             // MTP inputs not yet in the MTP KV: tokens at positions p-len+1 .. p
        eng.q.memcpy(eng.HPREV, eng.HLAST, D * 4).wait();
        std::vector<int> emitted{cur};
        bool stop = llama_vocab_is_eog(vocab, cur);
        while (!stop && produced < n_gen) {
            // draft: MTP over pending tokens, hidden rows = main hidden of the previous positions (HPREV filled below)
            if (eng.prof) { eng.q.wait(); eng.pt_last = now_ms(); }
            eng.mtp_forward(pend.data(), eng.HPREV, (int) pend.size(), p - (int) pend.size() + 1);
            const int d = eng.argmax();
            eng.tick("ROUND_mtp_total");
            const int pair[2] = {cur, d};
            eng.decode_multi<2>(pair, p, 0);
            eng.tick("verify_lm_head");
            int tb[2];
            eng.q.memcpy(tb, eng.TOKB, 8).wait();
            ++rounds;
            if (tb[0] == d) {
                ++accepted;
                emitted.push_back(d); emitted.push_back(tb[1]);
                // MTP pending: (d at p+1, hidden p), (tb1 at p+2, hidden p+1)
                eng.q.memcpy(eng.HPREV, eng.HB, 2 * D * 4);
                pend = {d, tb[1]};
                cur = tb[1]; p += 2;
            } else {
                eng.restore_states();
                emitted.push_back(tb[0]);
                eng.q.memcpy(eng.HPREV, eng.HB, D * 4);
                pend = {tb[0]};
                cur = tb[0]; p += 1;
            }
            produced = (int) emitted.size();
            for (size_t k = emitted.size() - (tb[0] == d ? 2 : 1); k < emitted.size(); ++k) if (llama_vocab_is_eog(vocab, emitted[k])) stop = true;
        }
        eng.q.wait();
        const double t2 = now_ms();
        eng.prof_report();
        for (int t : emitted) { char buf[256]; const int l = llama_token_to_piece(vocab, t, buf, sizeof(buf), 0, false); out.append(buf, std::max(0, l)); }
        printf("%s\n---\nprompt %zu tokens in %.0f ms; generated %d tokens at %.1f tok/s (MTP: %d rounds, %.1f%% drafts accepted, %.2f tokens/round)\n",
               out.c_str(), toks.size(), t1 - t0, produced, produced / ((t2 - t1) / 1e3), rounds, 100.0 * accepted / std::max(1, rounds),
               (double) produced / std::max(1, rounds));
        return 0;
    }
    const double t1 = now_ms();
    const long miss0 = eng.n_miss; const double cpu0 = eng.cpu_ms;
    std::string out;
    int produced = 0;
    for (int k = 0; k < n_gen; ++k) {
        char buf[256];
        const int l = llama_token_to_piece(vocab, cur, buf, sizeof(buf), 0, false);
        out.append(buf, std::max(0, l));
        ++produced;
        if (llama_vocab_is_eog(vocab, cur)) break;
        eng.decode(cur, (int) toks.size() + k);
        cur = eng.argmax();
    }
    const double t2 = now_ms();
    printf("%s\n---\nprompt %zu tokens in %.0f ms; generated %d tokens at %.1f tok/s; CPU experts %.1f per token, %.2f ms/token\n",
           out.c_str(), toks.size(), t1 - t0, produced, produced / ((t2 - t1) / 1e3),
           (double) (eng.n_miss - miss0) / produced, (eng.cpu_ms - cpu0) / produced);
    return 0;
}

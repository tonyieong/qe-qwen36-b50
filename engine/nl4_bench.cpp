// nl4_bench: GEMV bandwidth for different IQ4_NL decode strategies on the Arc B50 (rows of 2048, SoA rows like qe)
// build: icpx -fsycl -O3 -fsycl-targets=spir64_gen -Xs "-device bmg-g21" nl4_bench.cpp -o nl4_bench
#include <sycl/sycl.hpp>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

constexpr int SG = 16, COLS = 2048, NB = COLS / 32;
constexpr int ROWS = 65536;                       // 65536 x 2048 NL4 = 75 MB (well above the 18 MB L2)
constexpr size_t RB = COLS / 2 + NB * 2;          // nibbles + fp16 scales
static const int8_t KV[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

template <typename F>
double run(sycl::queue & q, const char * name, F kernel_launch, size_t bytes, float * y, const std::vector<float> & ref) {
    kernel_launch(); q.wait();
    const int it = 20;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < it; ++i) kernel_launch();
    q.wait();
    const double s = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count() / it;
    std::vector<float> out(ref.size());
    q.memcpy(out.data(), y, out.size() * 4).wait();
    double err = 0;
    for (size_t i = 0; i < ref.size(); ++i) err = std::max(err, (double) std::fabs(out[i] - ref[i]) / (std::fabs(ref[i]) + 1e-3));
    printf("%-28s %7.1f GB/s  (%.3f ms)  max rel err %.2e\n", name, bytes / s / 1e9, s * 1e3, err);
    return s;
}

int main() {
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
    printf("%s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    std::mt19937 rng(1);
    std::vector<uint8_t> W((size_t) ROWS * RB);
    for (auto & b : W) b = rng() & 0xff;
    for (int r = 0; r < ROWS; ++r)
        for (int b = 0; b < NB; ++b) { const sycl::half h(0.001f * (1 + (rng() % 7))); memcpy(&W[(size_t) r * RB + COLS / 2 + b * 2], &h, 2); }
    std::vector<float> X(COLS);
    for (auto & v : X) v = ((int) (rng() % 2001) - 1000) / 1000.f;
    // reference on the host
    std::vector<float> ref(ROWS);
    for (int r = 0; r < ROWS; ++r) {
        const uint8_t * row = &W[(size_t) r * RB];
        double acc = 0;
        for (int b = 0; b < NB; ++b) {
            sycl::half h; memcpy(&h, row + COLS / 2 + b * 2, 2);
            for (int j = 0; j < 16; ++j) {
                const uint8_t by = row[b * 16 + j];
                acc += (double) (float) h * (KV[by & 15] * X[b * 32 + j] + KV[by >> 4] * X[b * 32 + 16 + j]);
            }
        }
        ref[r] = (float) acc;
    }
    uint8_t * dW = sycl::malloc_device<uint8_t>(W.size(), q);
    float * dX = sycl::malloc_device<float>(COLS, q), * dY = sycl::malloc_device<float>(ROWS, q);
    int8_t * dXq = sycl::malloc_device<int8_t>(COLS, q);
    float * dXs = sycl::malloc_device<float>(NB, q);
    q.memcpy(dW, W.data(), W.size()); q.memcpy(dX, X.data(), COLS * 4).wait();
    // int8 activations for the dp4a variant (one scale per 32)
    {
        std::vector<int8_t> xq(COLS); std::vector<float> xs(NB);
        for (int b = 0; b < NB; ++b) {
            float am = 0; for (int j = 0; j < 32; ++j) am = std::max(am, std::fabs(X[b * 32 + j]));
            xs[b] = am / 127.f;
            for (int j = 0; j < 32; ++j) xq[b * 32 + j] = (int8_t) std::lrint(am > 0 ? X[b * 32 + j] / xs[b] : 0);
        }
        q.memcpy(dXq, xq.data(), COLS); q.memcpy(dXs, xs.data(), NB * 4).wait();
    }
    const size_t bytes = W.size();
    const sycl::nd_range<1> nr((size_t) ROWS * SG, 128);

    // 1. current: local int8 table, dynamic index
    run(q, "1 local table", [&] {
        q.parallel_for(nr, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            const auto sg = it.get_sub_group();
            const int r = it.get_global_id(0) / SG, lane = sg.get_local_id()[0];
            const uint8_t * row = dW + (size_t) r * RB;
            const int8_t kv[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
            float acc = 0.f;
            for (int b = lane; b < NB; b += SG) {
                const sycl::uint4 qq = *(const sycl::uint4 *) (row + b * 16);
                const uint32_t u[4] = {qq.x(), qq.y(), qq.z(), qq.w()};
                const float * x = dX + b * 32;
                float a = 0.f;
#pragma unroll
                for (int i = 0; i < 4; ++i)
#pragma unroll
                    for (int j = 0; j < 4; ++j) { const uint32_t by = (u[i] >> (8 * j)) & 0xff; a += kv[by & 0xf] * x[i * 4 + j] + kv[by >> 4] * x[16 + i * 4 + j]; }
                acc += a * (float) ((const sycl::half *) (row + COLS / 2))[b];
            }
            acc = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
            if (lane == 0) dY[r] = acc;
        });
    }, bytes, dY, ref);

    // 2. table packed in two 64-bit constants, values extracted with shifts
    run(q, "2 shift-packed table", [&] {
        q.parallel_for(nr, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            const auto sg = it.get_sub_group();
            const int r = it.get_global_id(0) / SG, lane = sg.get_local_id()[0];
            const uint8_t * row = dW + (size_t) r * RB;
            constexpr uint64_t LO = 0xF6EADDCFBFAD9881ull;   // bytes of KV[0..7]  (-127 .. -10)
            constexpr uint64_t HI = 0x7159453526190D01ull;   // bytes of KV[8..15] (1 .. 113)
            auto lut = [](uint32_t n) { const uint64_t w = n & 8 ? HI : LO; return (float) (int8_t) ((w >> ((n & 7) * 8)) & 0xff); };
            float acc = 0.f;
            for (int b = lane; b < NB; b += SG) {
                const sycl::uint4 qq = *(const sycl::uint4 *) (row + b * 16);
                const uint32_t u[4] = {qq.x(), qq.y(), qq.z(), qq.w()};
                const float * x = dX + b * 32;
                float a = 0.f;
#pragma unroll
                for (int i = 0; i < 4; ++i)
#pragma unroll
                    for (int j = 0; j < 4; ++j) { const uint32_t by = (u[i] >> (8 * j)) & 0xff; a += lut(by & 0xf) * x[i * 4 + j] + lut(by >> 4) * x[16 + i * 4 + j]; }
                acc += a * (float) ((const sycl::half *) (row + COLS / 2))[b];
            }
            acc = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
            if (lane == 0) dY[r] = acc;
        });
    }, bytes, dY, ref);

    // 3. 256-entry byte table in SLM: one lookup gives both nibbles as packed int8 pair
    run(q, "3 SLM byte-pair table", [&] {
        q.submit([&](sycl::handler & h) {
            sycl::local_accessor<int16_t, 1> t2(256, h);
            h.parallel_for(nr, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                const int8_t kv[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
                for (int i = it.get_local_id(0); i < 256; i += 128) t2[i] = (int16_t) ((uint8_t) kv[i & 15] | ((int) (uint8_t) kv[i >> 4] << 8));
                it.barrier(sycl::access::fence_space::local_space);
                const auto sg = it.get_sub_group();
                const int r = it.get_global_id(0) / SG, lane = sg.get_local_id()[0];
                const uint8_t * row = dW + (size_t) r * RB;
                float acc = 0.f;
                for (int b = lane; b < NB; b += SG) {
                    const sycl::uint4 qq = *(const sycl::uint4 *) (row + b * 16);
                    const uint32_t u[4] = {qq.x(), qq.y(), qq.z(), qq.w()};
                    const float * x = dX + b * 32;
                    float a = 0.f;
#pragma unroll
                    for (int i = 0; i < 4; ++i)
#pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            const int16_t pr = t2[(u[i] >> (8 * j)) & 0xff];
                            a += (float) (int8_t) (pr & 0xff) * x[i * 4 + j] + (float) (int8_t) (pr >> 8) * x[16 + i * 4 + j];
                        }
                    acc += a * (float) ((const sycl::half *) (row + COLS / 2))[b];
                }
                acc = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
                if (lane == 0) dY[r] = acc;
            });
        });
    }, bytes, dY, ref);

    // 4. int8 x int8 dot products (dp4a-style), weights decoded with the shift-packed table into int8 words
    run(q, "4 int8 dot (dp4a pattern)", [&] {
        q.parallel_for(nr, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            const auto sg = it.get_sub_group();
            const int r = it.get_global_id(0) / SG, lane = sg.get_local_id()[0];
            const uint8_t * row = dW + (size_t) r * RB;
            constexpr uint64_t LO = 0xF6EADDCFBFAD9881ull, HI = 0x7159453526190D01ull;
            auto lutb = [](uint32_t n) -> int32_t { const uint64_t w = n & 8 ? HI : LO; return (int32_t) (int8_t) ((w >> ((n & 7) * 8)) & 0xff); };
            float acc = 0.f;
            for (int b = lane; b < NB; b += SG) {
                const sycl::uint4 qq = *(const sycl::uint4 *) (row + b * 16);
                const uint32_t u[4] = {qq.x(), qq.y(), qq.z(), qq.w()};
                const sycl::int4 * xv = (const sycl::int4 *) (dXq + b * 32);
                const sycl::int4 x0 = xv[0], x1 = xv[1];
                const int32_t xw[8] = {x0.x(), x0.y(), x0.z(), x0.w(), x1.x(), x1.y(), x1.z(), x1.w()};
                int32_t s = 0;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int32_t xl = xw[i], xh = xw[4 + i];
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const uint32_t by = (u[i] >> (8 * j)) & 0xff;
                        s += lutb(by & 0xf) * (int32_t) (int8_t) (xl >> (8 * j)) + lutb(by >> 4) * (int32_t) (int8_t) (xh >> (8 * j));
                    }
                }
                acc += (float) s * (float) ((const sycl::half *) (row + COLS / 2))[b] * dXs[b];
            }
            acc = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
            if (lane == 0) dY[r] = acc;
        });
    }, bytes, dY, ref);

    // 5. x held in registers (each lane owns 4 blocks = 128 values), each sub-group walks RPS rows reusing them
    for (int RPS : {4, 8, 16}) {
        char name[64]; snprintf(name, sizeof name, "5 x in regs, %d rows/sg", RPS);
        run(q, name, [&] {
            q.parallel_for(sycl::nd_range<1>((size_t) ROWS / RPS * SG, 128), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                const auto sg = it.get_sub_group();
                const int r0 = (int) (it.get_global_id(0) / SG) * RPS, lane = sg.get_local_id()[0];
                constexpr uint64_t LO = 0xF6EADDCFBFAD9881ull, HI = 0x7159453526190D01ull;
                auto lut = [](uint32_t n) { const uint64_t w = n & 8 ? HI : LO; return (float) (int8_t) ((w >> ((n & 7) * 8)) & 0xff); };
                float xr[4][32];
#pragma unroll
                for (int k = 0; k < 4; ++k)
#pragma unroll
                    for (int j = 0; j < 32; ++j) xr[k][j] = dX[(lane + 16 * k) * 32 + j];
                for (int rr = 0; rr < RPS; ++rr) {
                    const uint8_t * row = dW + (size_t) (r0 + rr) * RB;
                    float acc = 0.f;
#pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        const int b = lane + 16 * k;
                        const sycl::uint4 qq = *(const sycl::uint4 *) (row + b * 16);
                        const uint32_t u[4] = {qq.x(), qq.y(), qq.z(), qq.w()};
                        float a = 0.f;
#pragma unroll
                        for (int i = 0; i < 4; ++i)
#pragma unroll
                            for (int j = 0; j < 4; ++j) { const uint32_t by = (u[i] >> (8 * j)) & 0xff; a += lut(by & 0xf) * xr[k][i * 4 + j] + lut(by >> 4) * xr[k][16 + i * 4 + j]; }
                        acc += a * (float) ((const sycl::half *) (row + COLS / 2))[b];
                    }
                    acc = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
                    if (lane == 0) dY[r0 + rr] = acc;
                }
            });
        }, bytes, dY, ref);
    }

    // reference points: same row count with Q8 (int8 + scale) bytes, to show the bandwidth ceiling of a cheap decode
    {
        const size_t RB8 = COLS + NB * 2;
        uint8_t * d8 = sycl::malloc_device<uint8_t>((size_t) ROWS * RB8, q);
        q.memset(d8, 1, (size_t) ROWS * RB8).wait();
        std::vector<float> dummy;
        run(q, "ref: Q8 rows", [&] {
            q.parallel_for(nr, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                const auto sg = it.get_sub_group();
                const int r = it.get_global_id(0) / SG, lane = sg.get_local_id()[0];
                const uint8_t * row = d8 + (size_t) r * RB8;
                float acc = 0.f;
                for (int b = lane; b < NB; b += SG) {
                    const sycl::uint4 * qv = (const sycl::uint4 *) (row + b * 32);
                    const sycl::uint4 a = qv[0], c = qv[1];
                    const uint32_t w[8] = {a.x(), a.y(), a.z(), a.w(), c.x(), c.y(), c.z(), c.w()};
                    const float * x = dX + b * 32;
                    float s = 0.f;
#pragma unroll
                    for (int i = 0; i < 8; ++i)
#pragma unroll
                        for (int j = 0; j < 4; ++j) s += (float) (int8_t) ((w[i] >> (8 * j)) & 0xff) * x[4 * i + j];
                    acc += s * (float) ((const sycl::half *) (row + COLS))[b];
                }
                acc = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
                if (lane == 0) dY[r] = acc;
            });
        }, (size_t) ROWS * RB8, dY, dummy);
    }
    return 0;
}

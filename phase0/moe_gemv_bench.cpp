// moe_gemv_bench: how fast can the Arc Pro B50 run Qwen3.6-35B-A3B's routed-expert work per decoded token?
// Shapes: hidden 2048, expert intermediate 512, 8 routed experts per layer, 40 layers.
// Weights: 4-bit nibbles + one fp16 scale per 32 weights (4.5 bits per weight), rows stored contiguously.
// Build: icpx -fsycl -O3 -fsycl-targets=spir64_gen -Xs "-device bmg-g21" moe_gemv_bench.cpp -o moe_gemv_bench
#include <sycl/sycl.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

constexpr int H = 2048;      // hidden size
constexpr int I = 512;       // expert intermediate size
constexpr int K_USED = 8;    // routed experts per token
constexpr int N_LAYER = 40;
constexpr int SG = 16;       // sub-group size

// per-matrix layout: [rows][cols/2] bytes of nibbles, then [rows][cols/32] halves of scales
constexpr size_t mat_q_bytes(int rows, int cols) { return (size_t) rows * cols / 2; }
constexpr size_t mat_s_bytes(int rows, int cols) { return (size_t) rows * (cols / 32) * 2; }
constexpr size_t GATE_BYTES = mat_q_bytes(I, H) + mat_s_bytes(I, H);
constexpr size_t DOWN_BYTES = mat_q_bytes(H, I) + mat_s_bytes(H, I);
constexpr size_t EXPERT_BYTES = 2 * GATE_BYTES + DOWN_BYTES;  // gate, up, down

static inline float dot_block32(const sycl::uint4 & q, float s, const float * x) {
    // q holds 32 nibbles: low nibble = even index, high nibble = odd index; centered at 8
    float acc = 0.f;
    const uint32_t w[4] = {q.x(), q.y(), q.z(), q.w()};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint32_t b = (w[i] >> (8 * j)) & 0xff;
            const int k = i * 8 + j * 2;
            acc += (float) ((int) (b & 0xf) - 8) * x[k] + (float) ((int) (b >> 4) - 8) * x[k + 1];
        }
    }
    return acc * s;
}

int main(int argc, char ** argv) {
    const size_t arena_experts = argc > 1 ? atoi(argv[1]) : 6000;
    const int n_tokens = argc > 2 ? atoi(argv[2]) : 200;

    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
    printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

    // ---- 1. raw read bandwidth ceiling ----
    {
        const size_t n = (size_t) 2 << 30;  // 2 GiB
        auto * buf = sycl::malloc_device<sycl::uint4>(n / 16, q);
        auto * out = sycl::malloc_device<uint32_t>(1, q);
        q.memset(buf, 1, n).wait();
        const size_t n_vec = n / 16;
        const size_t gsize = 160 * 1024 * SG;
        for (int rep = 0; rep < 2; ++rep) {
            auto t0 = std::chrono::high_resolution_clock::now();
            const int iters = 10;
            for (int it = 0; it < iters; ++it) {
                q.parallel_for(sycl::nd_range<1>(gsize, 256), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                    uint32_t acc = 0;
                    for (size_t i = it.get_global_id(0); i < n_vec; i += gsize) {
                        const sycl::uint4 v = buf[i];
                        acc ^= v.x() ^ v.y() ^ v.z() ^ v.w();
                    }
                    if (acc == 0x12345678) {
                        out[0] = acc;
                    }
                });
            }
            q.wait();
            const double s = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
            if (rep) {
                printf("raw read bandwidth: %.1f GB/s\n", (double) n * iters / s / 1e9);
            }
        }
        sycl::free(buf, q);
        sycl::free(out, q);
    }

    // ---- 2. MoE expert GEMV per token ----
    const size_t arena_bytes = arena_experts * EXPERT_BYTES;
    printf("expert: %.3f MB, arena: %zu experts = %.2f GB\n", EXPERT_BYTES / 1e6, arena_experts, arena_bytes / 1e9);
    auto * arena = sycl::malloc_device<uint8_t>(arena_bytes, q);
    {
        // fill with pseudo-random nibbles and small scales (content does not matter for speed)
        std::vector<uint8_t> chunk(64 << 20);
        std::mt19937 rng(1);
        for (auto & c : chunk) c = rng() & 0xff;
        for (size_t off = 0; off < arena_bytes; off += chunk.size()) {
            q.memcpy(arena + off, chunk.data(), std::min(chunk.size(), arena_bytes - off)).wait();
        }
    }
    auto * x = sycl::malloc_device<float>(H, q);
    auto * h = sycl::malloc_device<float>(K_USED * I, q);
    auto * y = sycl::malloc_device<float>(H, q);
    auto * ids = sycl::malloc_device<int32_t>(K_USED, q);
    auto * wts = sycl::malloc_device<float>(K_USED, q);
    {
        std::vector<float> hx(H, 0.01f), hw(K_USED, 0.125f);
        q.memcpy(x, hx.data(), H * 4).wait();
        q.memcpy(wts, hw.data(), K_USED * 4).wait();
    }

    std::mt19937 rng(42);
    std::vector<int32_t> sel(K_USED);
    auto run_token = [&](bool sync_each_layer) {
        for (int l = 0; l < N_LAYER; ++l) {
            for (int k = 0; k < K_USED; ++k) sel[k] = rng() % arena_experts;
            q.memcpy(ids, sel.data(), K_USED * 4);
            // gate+up fused: one sub-group per (expert, row); row = 2048 weights = 64 blocks of 32 -> 4 per lane
            q.parallel_for(sycl::nd_range<1>(K_USED * I * SG, 128), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                const auto sg = it.get_sub_group();
                const int gid = it.get_global_id(0) / SG;
                const int lane = sg.get_local_id()[0];
                const int e = gid / I, r = gid % I;
                const uint8_t * base = arena + (size_t) ids[e] * EXPERT_BYTES;
                float accg = 0.f, accu = 0.f;
#pragma unroll
                for (int m = 0; m < 2; ++m) {
                    const uint8_t * W = base + m * GATE_BYTES;
                    const sycl::uint4 * qrow = (const sycl::uint4 *) (W + (size_t) r * (H / 2));
                    const sycl::half * srow = (const sycl::half *) (W + mat_q_bytes(I, H)) + (size_t) r * (H / 32);
                    float acc = 0.f;
#pragma unroll
                    for (int b = 0; b < 4; ++b) {
                        const int blk = b * SG + lane;
                        acc += dot_block32(qrow[blk], (float) srow[blk], x + blk * 32);
                    }
                    acc = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
                    if (m == 0) accg = acc; else accu = acc;
                }
                if (lane == 0) {
                    h[e * I + r] = accg / (1.f + sycl::exp(-accg)) * accu;
                }
            });
            // down: one sub-group per output row, loops over the 8 experts; row = 512 weights = 16 blocks -> 1 per lane
            q.parallel_for(sycl::nd_range<1>(H * SG, 128), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                const auto sg = it.get_sub_group();
                const int r = it.get_global_id(0) / SG;
                const int lane = sg.get_local_id()[0];
                float acc = 0.f;
#pragma unroll
                for (int e = 0; e < K_USED; ++e) {
                    const uint8_t * W = arena + (size_t) ids[e] * EXPERT_BYTES + 2 * GATE_BYTES;
                    const sycl::uint4 * qrow = (const sycl::uint4 *) (W + (size_t) r * (I / 2));
                    const sycl::half * srow = (const sycl::half *) (W + mat_q_bytes(H, I)) + (size_t) r * (I / 32);
                    acc += wts[e] * dot_block32(qrow[lane], (float) srow[lane], h + e * I + lane * 32);
                }
                acc = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
                if (lane == 0) {
                    y[r] = acc;
                }
            });
            if (sync_each_layer) q.wait();
        }
        q.wait();
    };

    for (int pass = 0; pass < 2; ++pass) {
        const bool sync = pass == 1;
        run_token(sync);  // warm-up
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int t = 0; t < n_tokens; ++t) run_token(sync);
        const double s = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count() / n_tokens;
        const double bytes = (double) N_LAYER * K_USED * EXPERT_BYTES;
        printf("routed experts per token (%s): %.3f ms, %.1f GB/s effective (%.0f MB/token)\n",
               sync ? "host sync after every layer" : "no host sync", s * 1e3, bytes / s / 1e9, bytes / 1e6);
    }
    return 0;
}

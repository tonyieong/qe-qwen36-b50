// xmx_probe: list joint_matrix combinations on this GPU and time a simple f16 GEMM tile kernel
#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdio>
#include <vector>
namespace sx = sycl::ext::oneapi::experimental::matrix;
namespace sxi = sycl::ext::oneapi::experimental;

int main() {
    sycl::queue q{sycl::gpu_selector_v};
    auto dev = q.get_device();
    printf("%s\n", dev.get_info<sycl::info::device::name>().c_str());
    auto combos = dev.get_info<sxi::info::device::matrix_combinations>();
    for (auto & c : combos) {
        auto tn = [](sx::matrix_type t) {
            switch (t) { case sx::matrix_type::fp16: return "fp16"; case sx::matrix_type::bf16: return "bf16"; case sx::matrix_type::fp32: return "fp32";
                         case sx::matrix_type::sint8: return "s8"; case sx::matrix_type::uint8: return "u8"; case sx::matrix_type::sint32: return "s32"; default: return "?"; }
        };
        printf("A %s B %s C %s D %s  M %zu N %zu K %zu (max M %zu N %zu K %zu)\n", tn(c.atype), tn(c.btype), tn(c.ctype), tn(c.dtype),
               c.msize, c.nsize, c.ksize, c.max_msize, c.max_nsize, c.max_ksize);
    }
    // f16 GEMM C[M x N] = A[M x K] * B[K x N], A row-major, B packed (VNNI) — tiles 8x16x16, one sub-group per 8x16 C tile
    constexpr int TM = 8, TN = 16, TK = 16;
    const int M = 2048, N = 2048, K = 2048;
    std::vector<sycl::half> A(M * K, 0.01f), B(K * N, 0.02f);
    sycl::half * dA = sycl::malloc_device<sycl::half>(M * K, q), * dB = sycl::malloc_device<sycl::half>(K * N, q);
    float * dC = sycl::malloc_device<float>(M * N, q);
    q.memcpy(dA, A.data(), M * K * 2); q.memcpy(dB, B.data(), K * N * 2).wait();
    auto run = [&]() {
        return q.parallel_for(sycl::nd_range<2>({(size_t) M / TM, (size_t) N / TN * 16}, {1, 16 * 8}), [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
            const auto sg = it.get_sub_group();
            const int mt = it.get_global_id(0), nt = (int) (it.get_global_id(1) / 16);
            sx::joint_matrix<sycl::sub_group, sycl::half, sx::use::a, TM, TK, sx::layout::row_major> a;
            sx::joint_matrix<sycl::sub_group, sycl::half, sx::use::b, TK, TN, sx::layout::ext_intel_packed> b;
            sx::joint_matrix<sycl::sub_group, float, sx::use::accumulator, TM, TN> c;
            sx::joint_matrix_fill(sg, c, 0.f);
            auto pA = sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(dA);
            auto pB = sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(dB);
            auto pC = sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(dC);
            for (int k = 0; k < K; k += TK) {
                sx::joint_matrix_load(sg, a, pA + (size_t) mt * TM * K + k, K);
                sx::joint_matrix_load(sg, b, pB + (size_t) (k / 2) * N * 2 + nt * TN * 2, N * 2);
                sx::joint_matrix_mad(sg, c, a, b, c);
            }
            sx::joint_matrix_store(sg, c, pC + (size_t) mt * TM * N + nt * TN, N, sx::layout::row_major);
        });
    };
    run().wait();
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; ++i) run();
    q.wait();
    const double s = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count() / 10;
    float c0; q.memcpy(&c0, dC, 4).wait();
    printf("naive joint_matrix f16 GEMM %dx%dx%d: %.2f ms, %.1f TFLOPS, C[0]=%.4f (expect %.4f)\n", M, N, K, s * 1e3, 2.0 * M * N * K / s / 1e12, c0, 0.01f * 0.02f * K);
    return 0;
}

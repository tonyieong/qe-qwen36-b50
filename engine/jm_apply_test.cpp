// does joint_matrix_apply accept a (value, row, col) lambda on this toolchain?
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <cstdio>
namespace sx = sycl::ext::oneapi::experimental::matrix;
int main() {
    sycl::queue q{sycl::gpu_selector_v};
    float * out = sycl::malloc_shared<float>(8 * 16, q);
    float * rs = sycl::malloc_shared<float>(8, q);
    for (int i = 0; i < 8; ++i) rs[i] = (float) (i + 1);
    q.parallel_for(sycl::nd_range<1>(16, 16), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
        auto sg = it.get_sub_group();
        sx::joint_matrix<sycl::sub_group, float, sx::use::accumulator, 8, 16> c;
        sx::joint_matrix_fill(sg, c, 1.f);
        sycl::ext::intel::experimental::matrix::joint_matrix_apply(sg, c, [=](float & x, size_t row, size_t col) { x *= rs[row] + 0.01f * col; });
        auto p = sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(out);
        sx::joint_matrix_store(sg, c, p, 16, sx::layout::row_major);
    }).wait();
    printf("row0 col0 %.2f  row3 col5 %.2f  row7 col15 %.2f (expect 1.00 4.05 8.15)\n", out[0], out[3 * 16 + 5], out[7 * 16 + 15]);
    return 0;
}

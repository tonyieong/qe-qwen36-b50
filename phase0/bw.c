// RAM read bandwidth vs thread count (AVX2 streaming reads over a 4 GiB buffer)
#include <immintrin.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
int main(void) {
    const size_t n = (size_t) 4 << 30;
    char * p = aligned_alloc(1 << 21, n);
    madvise(p, n, MADV_HUGEPAGE);
    memset(p, 1, n);
    int th[] = {1, 2, 4, 8, 12, 16};
    for (int k = 0; k < 6; ++k) {
        omp_set_num_threads(th[k]);
        long long sink = 0;
        double t0 = omp_get_wtime();
        for (int rep = 0; rep < 3; ++rep) {
#pragma omp parallel reduction(^:sink)
            {
                int t = omp_get_thread_num(), nt = omp_get_num_threads();
                size_t per = n / nt;
                const __m256i * q = (const __m256i *) (p + t * per);
                __m256i a0 = _mm256_setzero_si256(), a1 = a0, a2 = a0, a3 = a0;
                for (size_t i = 0; i < per / 32; i += 4) {
                    a0 = _mm256_add_epi64(a0, _mm256_load_si256(q + i));
                    a1 = _mm256_add_epi64(a1, _mm256_load_si256(q + i + 1));
                    a2 = _mm256_add_epi64(a2, _mm256_load_si256(q + i + 2));
                    a3 = _mm256_add_epi64(a3, _mm256_load_si256(q + i + 3));
                }
                a0 = _mm256_add_epi64(_mm256_add_epi64(a0, a1), _mm256_add_epi64(a2, a3));
                sink ^= _mm256_extract_epi64(a0, 0);
            }
        }
        double s = omp_get_wtime() - t0;
        printf("%2d threads: %.1f GB/s (%lld)\n", th[k], 3.0 * n / s / 1e9, sink & 1);
    }
    return 0;
}

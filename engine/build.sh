#!/bin/bash
# build qe on the guest; exits non-zero on compile errors
cd ~/src/engine
L=~/src/llama.cpp
set +u; source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1
icpx -fsycl -O3 -march=native -std=c++17 -fsycl-targets=spir64_gen -Xs "-device bmg-g21" -I $L/include -I $L/ggml/include -I $L/vendor/cpp-httplib -I $L/vendor/nlohmann -I $L/tools/mtmd qe.cpp httplib.o -L $L/build-cpu/bin -lllama -lggml -lggml-base -lmtmd -Wl,-rpath,$L/build-cpu/bin -qmkl=sequential -lmkl_sycl_blas -lpthread -o qe.new 2> /tmp/qe-build.log
rc=$?
grep -E "error" -A4 /tmp/qe-build.log | head -30
[ $rc -eq 0 ] && mv qe.new qe && echo BUILD_OK
exit $rc

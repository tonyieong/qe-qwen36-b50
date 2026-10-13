#!/bin/bash
# usage: run-sycl.sh <model.gguf> <log> [extra llama-server args...]
set +u; source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1
export ONEAPI_DEVICE_SELECTOR=level_zero:0 UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1 SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 ZES_ENABLE_SYSMAN=1
M=$1; L=$2; shift 2
exec ~/src/llama.cpp/build-sycl/bin/llama-server -m "$M" -ngl 99 -fa on -t 8 --host 0.0.0.0 --port 8080 "$@" > "$L" 2>&1

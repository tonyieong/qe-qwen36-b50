#!/bin/bash
# start the qe engine server (OpenAI-compatible) for Qwen3.6-35B-A3B
# QE_PACK / QE_VOCAB / QE_MMPROJ pick the model; QE_PROFILE is where the adaptive expert cache keeps what it learned
set +u; source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1
export ONEAPI_DEVICE_SELECTOR=level_zero:0 SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1
M=/home/tony/models
S=/home/tony/src/phase0
PACK=${QE_PACK:-$M/hauhau-qe.pack}
export QE_PROFILE=${QE_PROFILE:-${PACK%.pack}.profile}
exec /home/tony/src/engine/qe serve $PACK ${QE_VOCAB:-$M/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ4_XS.gguf} --ctx ${QE_CTX:-131072} --kv-pool ${QE_KV_POOL:-131072} --resident 5900 --slots ${QE_SLOTS:-4} --mtp $M/qwen36-qe-mtp.pack --mmproj ${QE_MMPROJ:-$M/mmproj-Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-f16.gguf} --stats $S/statsA.bin,$S/statsB.bin --port ${QE_PORT:-8090} "$@"

#!/bin/bash
# start the qe engine server (OpenAI-compatible) for Qwen3.6-35B-A3B
set +u; source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1
export ONEAPI_DEVICE_SELECTOR=level_zero:0 SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1
S=/home/tony/src/phase0
exec /home/tony/src/engine/qe serve /home/tony/models/qwen36-qe.pack /home/tony/models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --ctx ${QE_CTX:-131072} --kv-pool ${QE_KV_POOL:-131072} --resident 5900 --slots ${QE_SLOTS:-4} --mtp /home/tony/models/qwen36-qe-mtp.pack --mmproj /home/tony/models/mmproj-Qwen3.6-35B-A3B-Q8_0.gguf --stats $S/statsA.bin,$S/statsB.bin --port ${QE_PORT:-8090} "$@"

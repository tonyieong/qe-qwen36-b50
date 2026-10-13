#!/bin/bash
set +u; source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1
export ONEAPI_DEVICE_SELECTOR=level_zero:0 SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1
S=/home/tony/src/phase0; M=/home/tony/models
QE_PROF=1 timeout 3000 /home/tony/src/engine/qe gen $M/qwen36-qe.pack $M/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf @$1 -n 40 --ctx 131072 --resident 5900 --mtp $M/qwen36-qe-mtp.pack --stats $S/statsA.bin,$S/statsB.bin 2>&1 | grep -E "^  (fa_|dn_conv|moe_exp|dn_proj)|ALPHA|prompt|generated"

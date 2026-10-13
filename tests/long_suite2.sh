#!/bin/bash
set +u; source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1
export ONEAPI_DEVICE_SELECTOR=level_zero:0 SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1
S=/home/tony/src/phase0; M=/home/tony/models
for f in needle_sum needle60k_sum needle110k_sum; do
  echo "=== $f"
  timeout 3600 /home/tony/src/engine/qe gen $M/qwen36-qe.pack $M/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf @/tmp/$f.txt -n 300 --ctx 131072 --resident 5900 --mtp $M/qwen36-qe-mtp.pack --stats $S/statsA.bin,$S/statsB.bin > /tmp/$f.out 2>&1
  grep -oE "7391-ALPHA-紅豆" /tmp/$f.out | head -1; grep -E "prompt [0-9]+ tokens" /tmp/$f.out
done
echo SUITE2_DONE

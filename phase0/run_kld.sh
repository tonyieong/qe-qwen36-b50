#!/bin/bash
# KL-divergence test: Q8_0 reference vs UD-IQ4_XS (DeltaNet Q8) vs UD-IQ4_XS-dn6 (DeltaNet Q6_K)
# waits for the Q8_0 download, stops the API service while the GPU is busy, restarts it at the end
M=~/models
OUT=~/src/phase0/kld
mkdir -p $OUT
while pgrep -f "curl.*Q8_0.gguf" >/dev/null; do sleep 20; done
ls -la $M/Qwen3.6-35B-A3B-Q8_0.gguf

set +u; source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1
export ONEAPI_DEVICE_SELECTOR=level_zero:0 UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1 SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
PPL=~/src/llama.cpp/build-sycl/bin/llama-perplexity
TXT=~/src/phase0/kld-text.txt
COMMON="-f $TXT -c 2048 -b 2048 -ub 512 -ngl 99 -fa on -t 16"

sudo systemctl stop llama-qwen36 2>/dev/null
sleep 3
echo "== base Q8_0"
$PPL -m $M/Qwen3.6-35B-A3B-Q8_0.gguf $COMMON -ncmoe 32 --kl-divergence-base $OUT/base.kld > $OUT/base.log 2>&1
tail -3 $OUT/base.log
for v in UD-IQ4_XS UD-IQ4_XS-dn6; do
    echo "== $v"
    $PPL -m $M/Qwen3.6-35B-A3B-$v.gguf $COMMON -ncmoe 10 --kl-divergence-base $OUT/base.kld --kl-divergence > $OUT/$v.log 2>&1
    sed -n '/====== Perplexity statistics/,$p' $OUT/$v.log | grep -vE "^\s*$" | head -40
done
sudo systemctl start llama-qwen36 2>/dev/null
echo KLD_DONE

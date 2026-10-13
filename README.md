# qe — Qwen3.6-35B-A3B inference engine for Intel Arc Pro B50

A purpose-built SYCL + AVX2 inference engine and OpenAI-compatible server for **Qwen3.6-35B-A3B** (qwen35moe: 30 Gated DeltaNet + 10 full-attention layers, 256-expert MoE, MTP head), tuned for one machine:

- Proxmox VM: Intel Arc Pro B50 16 GB (PCIe passthrough), Xeon D-1581 (16 vCPU), 66 GB RAM, Ubuntu 26.04
- Model does not fit in VRAM: hot experts stay in a VRAM arena, cold experts run on the CPU (AVX2 IQ4_NL kernel)

## Results (same hardware)

| | llama.cpp SYCL (UD-IQ4_XS) | qe |
|---|---|---|
| decode, 1 request | 26.6 tok/s | 52–66 tok/s (MTP speculative decoding) |
| decode, 4 concurrent | 8.6 tok/s each | ~17 tok/s each (64 aggregate) |
| prefill, short prompt | — | ~860 tok/s |
| 24K / 60K / 110K context | — | prefill 47 s / 193 s / 518 s, decode 43 / 28 / 21 tok/s |

Accuracy vs Q8_0 reference (713-token text): mean KL 0.014, top-1 agreement 95.8 %.

## Features

- Weight formats: experts IQ4_NL (4.5 bpw), DeltaNet + lm_head Q6, attention + shared experts Q8, int8 KV cache (fp16 scale per 16 values)
- MTP speculative decoding with DeltaNet state snapshot/restore
- XMX (joint_matrix) MoE prefill and flash attention; oneMKL for dense prefill
- Multi-slot continuous batching: 4 slots × up to 128K context sharing a paged 128K-position KV pool, admission control (queues requests that do not fit), prefix cache + prompt checkpoints for multi-turn reuse
- OpenAI API: `/v1/chat/completions` with streaming, `reasoning_content`, tool calls (Qwen template rendered byte-identical to jinja2), images via llama.cpp libmtmd (CPU vision encoder)

## Layout

- `engine/qe.cpp` — engine + server (`qe check | gen | multi | serve | render`)
- `engine/qconv.cpp` — converts the Q8_0 GGUF into the qe pack (`--mtp` builds the MTP pack)
- `engine/build.sh`, `engine/run-qe.sh` — build and launch scripts
- `engine/ref_dump.cpp`, `engine/tt/` — reference logits dumper, chat-template tests
- `phase0/` — early benchmarks (bandwidth, MoE GEMV, expert routing statistics); `statsA.bin`/`statsB.bin` are expert routing stats used to choose resident experts
- `tests/` — concurrency, long-context needle and vision tests
- `systemd/` — service units (qe and the llama.cpp baseline)

## Build / run

Requires the Intel oneAPI toolkit (icpx, oneMKL) and a llama.cpp checkout at `~/src/llama.cpp` built for CPU in `build-cpu/` (used for the tokenizer and libmtmd), plus `httplib.o` compiled from llama.cpp's `vendor/cpp-httplib`.

```sh
./engine/build.sh
qconv Qwen3.6-35B-A3B-Q8_0.gguf qwen36-qe.pack            # main pack
qconv --mtp <mtp-Q8_0.gguf> qwen36-qe-mtp.pack             # MTP pack
./engine/run-qe.sh                                          # serves on :8090 (QE_PORT)
```

Paths in the scripts are specific to the author's VM; adjust them. Model weights are not included.

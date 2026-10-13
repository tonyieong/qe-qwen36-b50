#!/usr/bin/env python3
# analyze expert_stats output: how much of the routed-expert traffic a VRAM cache of the hottest experts would serve
# usage: analyze_routing.py <rank_stats.bin> [<test_stats.bin>]
#   the cache is chosen from rank_stats; the hit rate is measured on test_stats (held-out text) when given
import sys
import numpy as np

N_LAYER, N_EXPERT, N_USED = 40, 256, 8


def load(path):
    raw = open(path, "rb").read()
    counts = np.zeros((N_LAYER, N_EXPERT), dtype=np.int64)
    off, n_tok = 0, 0
    while off < len(raw):
        il = int(np.frombuffer(raw, np.int16, 1, off)[0]); off += 2
        nt = int(np.frombuffer(raw, np.int32, 1, off)[0]); off += 4
        ids = np.frombuffer(raw, np.uint8, nt * N_USED, off); off += nt * N_USED
        np.add.at(counts[il], ids, 1)
        if il == 0:
            n_tok += nt
    return counts, n_tok


rank, n_rank = load(sys.argv[1])
test, n_test = (load(sys.argv[2]) if len(sys.argv) > 2 else (rank, n_rank))
print(f"rank set: {n_rank} tokens, test set: {n_test} tokens ({'held-out' if len(sys.argv) > 2 else 'same text'})")

order = np.argsort(-rank.flatten())
test_flat = test.flatten()
total = test_flat.sum()
n_all = N_LAYER * N_EXPERT
print("\ncache the N hottest (layer, expert) pairs, hit rate on the test set:")
for frac in (0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9):
    n = int(frac * n_all)
    hit = test_flat[order[:n]].sum() / total
    print(f"  {frac * 100:3.0f}% ({n:5d} experts) -> hit {hit * 100:5.1f}%, "
          f"misses per token {(1 - hit) * N_USED * N_LAYER:5.1f} expert calls")

print("\nper-layer skew on the test set (share of activations taken by the layer's top 25% / 50% experts):")
for il in range(0, N_LAYER, 5):
    c = np.sort(test[il])[::-1]
    s = c.sum()
    print(f"  layer {il:2d}: top25% {c[: N_EXPERT // 4].sum() / s * 100:5.1f}%  top50% {c[: N_EXPERT // 2].sum() / s * 100:5.1f}%  "
          f"never used {int((c == 0).sum())}")

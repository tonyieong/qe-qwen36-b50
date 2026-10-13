#!/usr/bin/env python3
# concurrency check: fire N identical-length chat requests at once, report per-request and aggregate decode speed
import json, sys, time, threading, urllib.request

URL = "http://localhost:8090/v1/chat/completions"
PROMPTS = ["用繁體中文詳細解釋TCP三次握手同四次揮手。", "Explain how a B-tree index works in PostgreSQL.",
           "寫一個Python script統計nginx log入面最多request嘅IP。", "Describe the causes of World War I in detail.",
           "解釋Kubernetes入面Deployment同StatefulSet嘅分別。", "Write a bash script that backs up /etc daily with rotation.",
           "用廣東話解釋量子糾纏係咩。", "Compare TCP BBR and CUBIC congestion control."]

def one(i, out):
    body = {"messages": [{"role": "user", "content": PROMPTS[i % len(PROMPTS)]}], "max_tokens": 300,
            "temperature": 0.7, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    d = json.load(urllib.request.urlopen(req, timeout=900))
    out[i] = (d["timings"]["predicted_n"], d["timings"]["predicted_per_second"], time.time() - t0)

for n in map(int, sys.argv[1:] or ["1", "2", "4"]):
    out = {}
    t0 = time.time()
    th = [threading.Thread(target=one, args=(i, out)) for i in range(n)]
    [t.start() for t in th]; [t.join() for t in th]
    wall = time.time() - t0
    toks = sum(v[0] for v in out.values())
    per = sorted(round(v[1], 1) for v in out.values())
    print(f"concurrency {n}: per-request decode {per} tok/s, aggregate {toks / wall:.1f} tok/s (incl. prefill), wall {wall:.0f}s")

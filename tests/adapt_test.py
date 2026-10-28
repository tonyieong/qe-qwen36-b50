#!/usr/bin/env python3
# sequential requests of one kind (default: coding), prints decode speed per request and the server's expert hit rate
import json, sys, urllib.request
URL = "http://localhost:8090"
tasks = ["Write a Python asyncio web crawler with rate limiting, retries and a SQLite cache. Full code.",
         "Write a C++17 thread pool with a work-stealing queue and futures. Full code with comments.",
         "Write a Rust CLI that parses large CSV files in parallel and prints column statistics. Full code.",
         "Write a TypeScript Express REST API with JWT auth, input validation and tests. Full code.",
         "Write a Go program implementing a Raft leader election simulation. Full code.",
         "Write a Python script that trains a small transformer on character-level text with PyTorch. Full code.",
         "Write a Bash script that backs up Proxmox VMs to S3 with rotation and logging. Full code.",
         "Write a SQL schema and queries for an inventory system with triggers and window functions."]
n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
p0 = json.load(urllib.request.urlopen(URL + "/props"))
for i in range(n):
    body = {"messages": [{"role": "user", "content": tasks[i % len(tasks)]}], "max_tokens": 500, "temperature": 0.6,
            "chat_template_kwargs": {"enable_thinking": False}}
    d = json.load(urllib.request.urlopen(urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600))
    p = json.load(urllib.request.urlopen(URL + "/props"))
    hit = 1 - (p["expert_misses"] - p0["expert_misses"]) / max(1, p["expert_lookups"] - p0["expert_lookups"])
    p0 = p
    print(f"req {i}: {d['timings']['predicted_n']} tok at {d['timings']['predicted_per_second']:.1f} tok/s | hit {hit*100:.1f}% swaps {p.get('expert_swaps')}", flush=True)

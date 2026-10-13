#!/usr/bin/env python3
# concurrent long-context requests against the qe server: N "agents", each with its own ~K-token document
# usage: agents_test.py <n_agents> <approx_tokens_each> [max_tokens]
import json
import sys
import threading
import time
import urllib.error
import urllib.request

URL = "http://localhost:8090/v1/chat/completions"
n = int(sys.argv[1])
tok_each = int(sys.argv[2])
max_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 250

text = open("/home/tony/src/phase0/kld-text.txt").read().replace("<|im_start|>", "").replace("<|im_end|>", "")
chars_per_tok = len(text) / 24000.0          # the corpus is ~24K tokens
need = int(tok_each * chars_per_tok)
big = text * (need // len(text) + 2)
questions = ["總結文件入面提到嘅網絡協議同埋佢哋嘅用途，大約150字。", "List the programming tasks mentioned in the document and summarize them in about 120 words.",
             "文件入面有冇提到廣東話或者香港相關內容？簡短總結。", "Which infrastructure tools (Docker, Kubernetes, Proxmox...) appear in the document? Summarize briefly."]
results = {}


def agent(i):
    start = (i * 7919) % len(text)
    doc = f"[agent {i} document]\n" + big[start:start + need]
    body = {"messages": [{"role": "user", "content": doc + "\n\n" + questions[i % len(questions)]}],
            "max_tokens": max_tokens, "temperature": 0.7, "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.time()
    try:
        d = json.load(urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=7200))
        tm = d["timings"]
        results[i] = (time.time() - t0, tm["prompt_n"], tm["prompt_ms"] / 1000, tm["predicted_n"], tm["predicted_per_second"],
                      d["choices"][0]["message"]["content"][:80].replace("\n", " "))
    except urllib.error.HTTPError as e:
        results[i] = (time.time() - t0, "HTTP %d: %s" % (e.code, e.read().decode()[:160]))


t0 = time.time()
th = [threading.Thread(target=agent, args=(i,)) for i in range(n)]
for t in th:
    t.start()
    time.sleep(0.2)
for t in th:
    t.join()
for i in range(n):
    r = results[i]
    if len(r) == 2:
        print(f"agent {i}: {r[0]:.0f}s  ERROR {r[1]}")
    else:
        print(f"agent {i}: total {r[0]:.0f}s | prompt {r[1]} tok in {r[2]:.0f}s | decode {r[3]} tok at {r[4]:.1f} tok/s | {r[5]}")
print(f"wall {time.time() - t0:.0f}s")

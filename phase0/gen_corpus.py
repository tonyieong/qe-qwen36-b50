#!/usr/bin/env python3
# build a routing-analysis corpus: real prompts + the model's own answers, saved as chat-formatted text
import json, os, urllib.request

OUT = os.path.expanduser("~/src/phase0/corpus")
os.makedirs(OUT, exist_ok=True)
P = [
    "用繁體中文解釋BGP路由選擇嘅過程同屬性優先次序。", "用廣東話講下點樣煲一個好飲嘅老火湯。", "寫一篇300字嘅短文，主題係香港嘅夜景。",
    "用繁體中文比較Docker同虛擬機嘅分別。", "解釋Proxmox VE入面PCI passthrough嘅原理同設定步驟。", "用廣東話解釋點樣計按揭供款。",
    "總結第二次世界大戰嘅主要原因，用繁體中文。", "解釋Linux入面cgroups同namespaces嘅作用。", "用繁體中文寫一份簡單嘅辭職信範本。",
    "Write a Python asyncio web scraper with rate limiting and retries.", "Implement a thread-safe LRU cache in C++17 with tests.",
    "Write a Bash script to monitor disk usage and send an alert when over 90%.", "Explain the transformer attention mechanism with equations.",
    "Write a SQL query to find the top 3 customers by revenue per month, and explain it.", "Write a Go HTTP server with middleware for logging and auth.",
    "Explain how TLS 1.3 handshake works step by step.", "Solve: if 3x + 7 = 2x - 5, find x, then compute x^2 - 4x. Show steps.",
    "A train leaves at 9:00 at 80 km/h; another leaves the same station at 10:30 at 120 km/h. When does the second catch up? Show reasoning.",
    "Write a React component for a sortable, filterable table with TypeScript.", "Explain the difference between OSPF and IS-IS in detail.",
    "Write a Rust function to parse a CSV line handling quoted fields, with unit tests.", "Summarize the plot of Romeo and Juliet in 200 words.",
    "用繁體中文寫一個Python程式，讀取Excel檔案並計算每月銷售總額，附註解。", "解釋H3C交換機上面點樣設定VLAN同trunk，畀出命令例子。",
    "用繁體中文解釋量子計算嘅基本概念同佢對密碼學嘅影響。",
    "Write a Kubernetes deployment YAML with HPA, liveness probes and resource limits, and explain each part.",
]

for i, p in enumerate(P):
    for think in (False, True):
        if think and i % 3:
            continue
        body = {"messages": [{"role": "user", "content": p}], "max_tokens": 700, "temperature": 0.7,
                "chat_template_kwargs": {"enable_thinking": think}}
        req = urllib.request.Request("http://localhost:8080/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        r = json.load(urllib.request.urlopen(req, timeout=900))
        m = r["choices"][0]["message"]
        ans = ""
        if m.get("reasoning_content"):
            ans += "<think>\n" + m["reasoning_content"] + "\n</think>\n\n"
        ans += m.get("content") or ""
        text = "<|im_start|>user\n" + p + "<|im_end|>\n<|im_start|>assistant\n" + ans + "<|im_end|>\n"
        suffix = "t" if think else ""
        with open(f"{OUT}/{i:02d}{suffix}.txt", "w") as f:
            f.write(text)
        print(i, think, r["usage"]["completion_tokens"], flush=True)

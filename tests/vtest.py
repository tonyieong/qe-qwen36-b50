#!/usr/bin/env python3
# image understanding check against the qe server: describe, follow-up on the same image, OCR
import base64
import json
import sys
import time
import urllib.error
import urllib.request

URL = "http://localhost:8090/v1/chat/completions"


def post(body):
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        return json.load(urllib.request.urlopen(req, timeout=900))
    except urllib.error.HTTPError as e:
        print("HTTP", e.code, e.read().decode())
        return None


def image_part(path):
    mime = "image/png" if path.endswith(".png") else "image/jpeg"
    b64 = base64.b64encode(open(path, "rb").read()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}


def ask(msgs):
    t = time.time()
    d = post({"messages": msgs, "max_tokens": 300, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})
    if not d:
        return ""
    tm = d["timings"]
    c = d["choices"][0]["message"]["content"]
    print(f"[{time.time() - t:.1f}s total, prompt eval {tm['prompt_n']} tok in {tm['prompt_ms']:.0f} ms, decode {tm['predicted_per_second']:.1f} tok/s]")
    print(c[:600])
    print()
    return c


img = sys.argv[1]
first = [{"role": "user", "content": [image_part(img), {"type": "text", "text": sys.argv[2] if len(sys.argv) > 2 else "用繁體中文描述呢張圖。"}]}]
ans = ask(first)
if len(sys.argv) <= 3:
    ask(first + [{"role": "assistant", "content": ans}, {"role": "user", "content": "用一句講出圖入面最主要嘅顏色。"}])

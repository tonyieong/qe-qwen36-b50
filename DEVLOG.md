# qe 開發記錄

按時間順序記錄 qe 的開發過程：每一步做了什麼、量到什麼數字、試過但失敗的方案，以及踩過的坑。設計上的取捨整理在 [DESIGN.md](DESIGN.md)。

## 2026-10-10 至 10-11：GPU 直通與 llama.cpp 基準

**GPU 直通到 VM**
- Proxmox host 已經認到 B50，只是舊的 `pci.ids` 把它顯示成「Battlemage G21」，容易誤以為認不到。
- VM 加入 GPU 後 noVNC 黑畫面，原因有兩個：
  1. 改成 q35 時沒有加 `pcie=1`，GPU 掛在傳統 PCI bus 上，xe driver 連載入都失敗。
  2. 真正令畫面變黑的原因：guest 的 xe driver 初始化顯示部分時經 vgaarb 取得 VGA 資源，順手關掉了虛擬顯示卡（bochs）的 memory decode。VM 其實一直在正常運行，只是 console 停在最後一格。
- 修正：`options xe probe_display=0`（B50 只用來運算，不需要顯示輸出），另加 udev 規則讓 GNOME 不使用 B50。

**llama.cpp 基準**

| 設定 | Decode |
|---|---|
| 純 CPU | 5–6 tok/s |
| Vulkan，IQ4_XS，6 層 expert 在 CPU | 13.7 tok/s |
| SYCL，IQ4_XS，5 層 expert 在 CPU | 31–32 tok/s |
| SYCL，IQ3_XXS，全部在 VRAM | 41.7 tok/s |
| SYCL，128K context + 圖片，10 層 expert 在 CPU | 26.6 tok/s |

- MTP、DFlash、DFlash2、ngram speculative decoding 全部試過，**沒有一個有提升**（17–32 tok/s）。原因是 draft 要佔 VRAM，驗證多個 token 又會用到更多放在 CPU 的 expert。
- VM 由 12 核加到 16 核並 pin 到實體核心，速度不變。這證明 llama.cpp 的瓶頸在 GPU kernel，不在 CPU。
- 當時的結論是「4-bit 下 B50 上限約 32 tok/s」。後來證明這個結論錯了：上限是 llama.cpp 的上限，不是硬件的上限。

**Strata 對比**：裝了 Strata 跑 Qwen3.8-Flash-Next IQ2_XS（2-bit），只有 7.4–12.2 tok/s，廣東話有錯字，啟動要 10 分鐘，而且要 54 GB RAM。社群在 B50 上有 25–27 tok/s，差異來自這台機的舊 CPU、PCIe 3.0 和沒有 NVMe。放棄。

## 2026-10-11：階段 0，可行性驗證

目標：用實測數字判斷自製 engine 能否達到約 60 tok/s，做不到就停手。

1. **GPU 4-bit GEMV**：手寫 SYCL kernel 達 150.9 GB/s（理論頻寬的 67%），llama.cpp 約 25%。
2. **GPU/CPU 每層同步**：約 12 µs，分工可行。
3. **Expert 命中率**：自己生成 35 條中英文及程式碼對話做語料，記錄每層 expert 的使用次數，一半選熱門 expert、另一半驗證。60% 放 VRAM → 83% 命中。
4. **DeltaNet 降 Q6 的精度**：以 Q8_0 為基準量 KL divergence。Q8 0.01630 ± 0.00049，Q6 0.01649 ± 0.00045，差異在誤差內。
5. **Guest RAM 頻寬**：約 32 GB/s。

估算 45–60 tok/s，決定繼續。

## 2026-10-11：階段 1，原型

- **Dense 部分 kernel**：用隨機權重量時間。AoS 佈局只有 36–55 GB/s；**改成對齊的 SoA 佈局後**，lm_head（Q6，413 MB）1.87 ms ≈ 221 GB/s，DeltaNet 投影約 200 GB/s，expert 約 163 GB/s。
- 全部 expert 在 GPU 時：4K context 63.5 tok/s，32K 48.1 tok/s（attention 未優化）。
- 加上 12% expert 交 CPU（AVX2）：4K 55.4 tok/s，32K 42.9 tok/s。
- 原型用的是 Q4_0 式格式（線性），精度不如 IQ4_XS，正式版改用 IQ4_NL。

## 2026-10-12：v1.0 至 v1.5

**v1.0：正確的 engine**
- `qconv` 把 Q8_0 GGUF 轉成 pack，`qe` 實作完整前向計算。
- 對比 llama.cpp Q8_0：平均 KL 0.004，top-1 一致 100%，greedy 續寫 32/32 一致。
- Decode 53.2–53.5 tok/s，但 prompt 處理只有約 53 tok/s（逐 token 跑）。瓶頸是每層 host 同步與太多細 kernel。

**v1.1：batched prefill 與 128K**
- Dense 部分用 oneMKL f16 GEMM，冷門 expert 在 CPU 以 batch 方式計算 → 短 prompt 約 312 tok/s，24K 約 220 tok/s。
- KV cache 改 int8（每 16 個值一個 scale），128K context 放得下。
- OpenAI 相容 server 完成，支援串流與 `reasoning_content`。
- 失敗嘗試：MoE prefill 用「每條 lane 處理一行」的 kernel，因為記憶體讀取不合併，反而更慢。結論是要用 XMX。

**v1.2：MTP**
- 實作 MTP speculative decoding：`decode_multi<2>` 驗證、DeltaNet 狀態快照/還原、冷門 expert 以 batch 計算。
- **60–66 tok/s**（命令行 greedy 64–66，API 含 sampling 60–62），接受率 81–95%。**達成原定 60 tok/s 目標。**
- 坑：串流時中文字的 UTF-8 bytes 被切成兩個 chunk，nlohmann json 拋例外令連線中斷。改為緩存不完整的 UTF-8 序列。

**v1.3：tool calling 與 XMX prefill**
- Chat template 用 C++ 重寫，與 jinja2 參考輸出逐 byte 一致。
- MoE prefill 改用 XMX `joint_matrix`（f16 VNNI 在 SLM），chunk 上限 1024；最忙的冷門 expert 臨時搬上 GPU（`QE_STAGE_MIN` 最佳約 14）；CPU kernel 改為分塊。
- Prefill：2.7K token 8.6 → 3.1 秒（約 860 tok/s，llama.cpp 約 500），24K 108 → 62 秒。
- **坑：CPU thread 太多會塌。** 14–16 條忙等 thread 跑在 16 個 vCPU 上，與 GDM 登入畫面及 QEMU 自身 thread 搶 CPU，速度跌到約 5 tok/s。上限定為 12。

**v1.4：多 slot**
- Continuous batching：每個 slot 有自己的 DeltaNet 狀態、KV、MTP KV；scheduler thread 負責分配。
- 1 / 2 / 4 個同時請求：每個 59.8 / 36 / 20 tok/s（llama.cpp 為 26.7 / 13.3 / 8.6）。
- 坑：CpuPool 的工作區上限 4 個 slot 時不夠，導致 segfault，`MAXT` 改為 8。

**v1.5：圖片**
- 用 llama.cpp libmtmd，視覺編碼器在 CPU。圖片 token 用負數假 id 參與前綴比對，實作交錯式 M-RoPE 與每個 slot 的 `rope_delta`。
- 加入 prompt checkpoint：多輪對話追問時不必重新處理整段 prompt，同一張圖追問只需 1.4 秒。
- 坑 1：CpuPool 建立時把建立它的 thread 也 pin 到某粒 CPU，之後所有新 thread 都繼承這個 affinity，結果圖片編碼只用 1 粒 CPU，要 160 秒。移除 pin 後回到 4.5 秒。
- 坑 2：`mtmd_input_text` 有 `text_len` 欄位，用 aggregate 初始化時被留成 0。
- 坑 3：檢查最後一條訊息時把圖片收集了兩次。

## 2026-10-12 至 10-13：精度與解碼實驗（兩個都沒有帶來提升）

**Dense 降到 Q4**：DeltaNet、lm_head、attention 全部改 Q4，KL 由 0.0142 升到 0.0320–0.0457，速度沒有變快。放棄並刪除測試 pack。

**IQ4_NL 解碼優化**：獨立 benchmark 中位元運算查表 121 GB/s（陣列查表 76–97 GB/s，SLM 查表 108–112 GB/s，輸入放 register 反而只有 72–78 GB/s）。套入 engine 後整體速度沒有可量度的變化（56.8 / 58.8 vs 58.4 / 60.7 tok/s，在誤差內）。量度每輪時間分佈後發現沒有單一瓶頸（見 DESIGN.md 第 9 節）。保留位元運算版本，因為沒有副作用。

## 2026-10-13：v1.6，長 context 與 4 slot × 128K

需求：4 個 research agent 同時使用，每個都可能用到很長的 context。

**先量現況**：只測過 24K。估計 120K prefill 要 10 分鐘以上，decode 會明顯變慢。

**改動**
1. **Paged KV pool**：256 個位置一塊，每個 slot 一張 block table，4 個 slot 共用 128K。
2. **GQA decode attention**：同一個 KV head 的 8 個 query head 共用一次 KV 讀取（context ≥ 8K 時啟用）。
3. **XMX flash attention**：
   - v1（兩趟）比預期慢，原因是每個 sub-group 16 個累加器造成 register spill。
   - v2 把 head 維度拆給兩個 sub-group（每個 work-group 16 個 sub-group），兩邊共享 score；縮放用 Intel 擴充的 `joint_matrix_apply`（標準版不提供 row/col）。
4. **Staging 與計算重疊**：冷門 expert 搬運放到第二條 queue，常駐 expert 先算。

**坑：DEVICE_LOST 與 GPU page fault。** 加入重疊後，GPU 偶發 DEVICE_LOST，kernel log 出現 xe「Fault response: Unsuccessful」。原因是非同步 `q.memcpy` 的來源是函數內的局部 vector，函數返回時 vector 已被釋放，複製還沒開始。改為 Engine 的成員 vector（雙緩衝）後連續三次測試通過。

**坑：長 prompt 超出命令列長度限制。** 221 KB 的 prompt 出現「Argument list too long」，`qe gen` 改為支援 `@file`。

**結果（單一請求）**

| Context | Prefill | Decode | 優化前 |
|---|---|---|---|
| 24K | 47 秒 | 43 tok/s | 78 秒 / 37 tok/s |
| 60K | 193 秒 | 28 tok/s | 415 秒 / 20 tok/s |
| 110K | 518 秒 | 21 tok/s | 約 640 秒 / 13 tok/s |

三個長度的「藏密碼」測試全部答對。精度不變：KL 0.0142，top-1 95.8%。

**併發測試**
- 4 個 agent 各約 28K 同時送入：每個 prefill 約 240 秒，decode 每個 9–12 tok/s，257 秒內全部完成，pool 用了 113K。
- 2 個約 70K 同時送入（總量超過 pool）：第一版其中一個做了 444 秒才回 503「KV cache pool exhausted」。原因是兩個請求入隊時 pool 都是空的，block 要到 prefill 時才分配。加入 admission control（入隊時先承諾 block，放不下就排隊）後，第二個請求會等第一個完成再執行，兩個都成功（242 秒 + 480 秒）。

## 最終數字

| | llama.cpp SYCL | qe |
|---|---|---|
| 單人 decode | 26.6 tok/s | 52–66 tok/s |
| 4 人同時 | 每人 8.6 tok/s | 每人約 17 tok/s |
| 短 prompt prefill | 約 500 tok/s | 約 860 tok/s |
| 與 Q8_0 的 KL* | 0.0163 | 0.0142 |
| Context | 128K 共用 | 4 slot，每個最多 128K，共用 128K pool |

\* 兩個 KL 用的測試文本不同（llama.cpp 為約 2 萬 token 的對話語料，qe 為 713 token 的參考文本），只能作大致比較。

## 心得

1. **先量再寫。** 階段 0 只花幾個小時，卻決定了整個項目值不值得做。「llama.cpp 只用到 25% 頻寬」這一個數字，推翻了「硬件上限 32 tok/s」的結論。
2. **記憶體佈局比運算重要。** 同一個 GEMV，AoS 改成 SoA 就由 50 GB/s 變成 200 GB/s。
3. **Speculative decoding 能否見效，取決於驗證成本。** 同一個 MTP，在 llama.cpp 上是負優化，在 qe 上是 +20%，差別只在冷門 expert 能否以 batch 計算。
4. **虛擬機裏的 CPU 不是你的。** vCPU 會被搶佔，thread 數要留餘裕。
5. **非同步 API 的生命週期 bug 很難查。** 症狀是 GPU page fault，原因卻在 host 端的 vector 釋放時機。
6. **到最後沒有單一瓶頸。** 每部分約佔四分之一時間，局部優化再快也看不到整體效果，這通常代表已經接近該硬件的實際上限。

# qe 設計思路

本文件說明 qe 為什麼這樣設計：每個決定背後的量測數據、取捨，以及被否決的方案。開發過程按時間順序記錄在 [DEVLOG.md](DEVLOG.md)。

## 1. 起點：硬件限制

| 項目 | 規格 | 對設計的影響 |
|---|---|---|
| GPU | Intel Arc Pro B50，16 GB，理論頻寬 224 GB/s，有 XMX 矩陣單元 | 放不下整個模型 |
| CPU | Xeon D-1581（2015 年 Broadwell，16 核，AVX2，無 AVX-512） | 冷門 expert 只能靠 AVX2 |
| RAM | 66 GB，雙通道 DDR4-2133，實測約 32 GB/s | CPU 側頻寬約為 GPU 的七分之一 |
| PCIe | Gen3 x8（平台上限） | GPU 與 CPU 之間不能大量搬權重 |
| 環境 | Proxmox VM，GPU 直通 | vCPU 會被 host 搶佔 |

模型 Qwen3.6-35B-A3B 結構：40 層 = 30 層 Gated DeltaNet + 10 層 full attention；每層 MoE 有 256 個 expert，每個 token 選 top-8，另加一個 shared expert；附帶一層 MTP（next-token prediction）。

4-bit 下 expert 約 18 GB，加上 dense 部分約 20 GB，**超出 VRAM**。所以設計的核心問題是：**哪些權重放 GPU、哪些放 CPU，以及怎樣令兩邊都跑在頻寬上限附近。**

## 2. 為什麼不用 llama.cpp

llama.cpp SYCL 在這台機上實測 26–32 tok/s。分析後發現兩個結構性問題：

1. **Kernel 頻寬利用率低**：IQ3_XXS 全部放 VRAM 時 41.7 tok/s，每 token 讀約 1.4 GB，即實際只用了約 58 GB/s，大約是理論頻寬的 25%。
2. **CPU offload 粒度太粗**：`-ncmoe` 以「整層」為單位把 expert 放到 CPU。但 expert 使用率極不平均，按層切等於把很多熱門 expert 也放到 CPU。

階段 0 的驗證（見 DEVLOG）證明：手寫 SoA 佈局 GEMV 可達 150–220 GB/s；而按 expert 熱度切分，60% expert 放 VRAM 已有 82–83% 命中率。這兩個數字令自製 engine 的理論速度約為 llama.cpp 的兩倍，因此決定自己寫。

另外也評估過 Strata（Qwen3.8-Flash-Next 專用的引擎）。它只支援 Flash-Next；在這台機上 2-bit Flash-Next 實測只有約 10 tok/s，而且廣東話質素較差，所以換模型不是出路。

## 3. 權重格式

| 部分 | 格式 | bpw | 理由 |
|---|---|---|---|
| Routed experts | IQ4_NL（非線性 4-bit，每 32 個權重一個 fp16 scale） | 4.5 | 佔模型約 90%，必須 4-bit；非線性碼表比 Q4_0 準，大小相同 |
| DeltaNet 投影、lm_head | Q6 | 6.5 | 每個 token 都要全讀；實測 Q8→Q6 的 KL 差異落在誤差範圍內（0.01630 vs 0.01649），但少讀約 0.25 GB |
| Attention、shared expert | Q8 | 8.5 | 讀取量小，attention 對量化敏感 |
| Router | F32 | 32 | 很小，選錯 expert 的代價大 |
| KV cache | int8，每 16 個值一個 fp16 scale | 約 9 | 128K context 下 KV 約 1.4 GB，fp16 放不下 |

**被否決：dense 部分全部降到 Q4。** 實測 KL 由 0.0142 升到 0.0320–0.0457（差 2–3 倍），速度反而沒有提升（每個 token 的時間分散在多個部分，見第 9 節），所以不採用。

**為什麼比 Unsloth UD-IQ4_XS 更準：** UD-IQ4_XS 的 expert 大部分其實是 IQ3_S（約 3.4–3.8 bpw）。qe 全部 expert 用真正的 4.5 bpw，同 Q8_0 比較的平均 KL 為 0.014（UD-IQ4_XS 為 0.016；兩者測試文本不同，只能作大致比較）。

所有權重由 Q8_0 GGUF 經 `qconv` 重新量化成 pack 檔，並改成 **SoA（structure of arrays）佈局**：scale 與量化值分開連續存放，令 GPU 可以用對齊的向量讀取。這一點是頻寬由 36–55 GB/s 升到 160–220 GB/s 的關鍵；AoS 或逐 byte 讀取只有前者的水平。

## 4. 熱 / 冷 expert 分工

- **選擇方式**：用一批中英文及程式碼對話記錄每層每個 expert 的使用次數（`phase0/statsA.bin`、`statsB.bin`），按次數排序，最熱門的 N 個放入 VRAM arena（`--resident`，目前 5900 個，約佔全部 10240 個的 58%，數量由剩餘 VRAM 決定）。用一半語料選、另一半驗證，兩個方向結果一致，避免過擬合。
- **命中率**（held-out）：50% → 75%，60% → 83%，70% → 89%，80% → 94%。每層最熱門 25% expert 承擔 39–62% 的運算。
- **冷門 expert 在 CPU 計**：AVX2 IQ4_NL kernel，以 thread pool 執行，與 GPU 同時進行。每層 GPU 與 CPU 同步一次只需約 12 µs，所以分工的同步成本很低。
- **Thread 數上限 12**：VM 有 16 個 vCPU，但 14–16 條忙等 thread 會與 QEMU 及桌面程式搶 CPU，被搶佔時速度會塌到約 5 tok/s。12 條是穩定的上限。
- **Prefill 時把最忙的冷門 expert 臨時搬上 GPU**（staging）：一個 chunk 內被大量 token 選中的冷門 expert（≥ `QE_STAGE_MIN`，預設 14 個 token），經 PCIe 搬上 GPU 用 XMX 計，比在 CPU 計快。搬運在第二條 queue 上進行，與常駐 expert 的計算重疊。

## 5. Decode 路徑

- **GEMV kernel**：SoA 向量讀取，每個 sub-group 處理一行，IQ4_NL 解碼用位元運算查表（比陣列查表快 20–60%，雖然對整體速度沒有可量度的影響）。
- **MTP speculative decoding**：用模型自帶的 MTP 層預測下一個 token，主模型一次驗證 2 個 token（`decode_multi<NT>`）。DeltaNet 是遞迴狀態，驗證失敗時要還原，所以每次驗證前先做狀態快照。接受方式是「從主模型分佈抽樣，與草稿相同即接受」，因此輸出分佈與不開 MTP 完全一樣。接受率 81–95%，每輪約出 1.9 個 token，速度由約 53 升到 60–66 tok/s。
  - 在 llama.cpp 上 MTP 是負優化，原因是驗證多個 token 會觸發更多放在 CPU 的 expert。qe 的 CPU 側以 batch 方式處理多個 token 的冷門 expert，所以驗證成本低得多，MTP 才有正收益。
  - **被否決：DFlash / DFlash2。** 在 llama.cpp 上實測反而更慢（17–30 vs 32 tok/s），原因是 draft model 佔 VRAM（要把更多 expert 移到 CPU），每次驗證 4–8 個 token 又令 CPU 側工作量倍增，而中文長答的接受率只有 36–56%。DFlash2 適合整個模型都放得入 VRAM 的大卡，這裡條件不成立。
- **長 context 的 decode attention（GQA 共用）**：每個 KV head 對應 8 個 query head。原本每個 query head 各自讀一次 KV，128K 時 KV 約 147 MB，遠超 L2 cache，等於讀 8 倍。context ≥ 8K（`QE_GQA_MIN`）時改用「每個 work-group 一個 (KV head, chunk)，sub-group 對應 query head」的 kernel，KV 只讀一次。

## 6. Prefill 路徑

- **Dense 部分**：權重解量化成 f16，用 oneMKL GEMM。
- **MoE**：XMX `joint_matrix` kernel，權重解量化成 f16 VNNI 格式放入 SLM。Chunk 上限 1024 token，令每個 expert 收到足夠多 token 填滿矩陣單元。
- **Flash attention（v2）**：以 XMX 計 Q·Kᵀ 和 P·V，online softmax。
  - Work-group 對應 (KV head, 8 個 token)，16 個 sub-group = 8 個 query head × 2 個維度半邊。KV 每塊 32 個位置，解量化進 SLM，8 個 head 共用。
  - 第一版（v1，兩趟）每個 sub-group 持有 16 個累加器，register spill 嚴重，比預期慢。v2 把 head 維度拆成兩半交給兩個 sub-group，兩邊共享 score 計算，並用 Intel 擴充的 `joint_matrix_apply`（帶 row/col）直接在累加器上做縮放。
  - 效果：24K prompt 78 → 47 秒，60K 415 → 193 秒，110K 約 640 → 518 秒。

## 7. 多 slot 與 paged KV

- **Continuous batching**：多個 slot 各自有 DeltaNet 狀態、KV、MTP KV。Scheduler thread 把各 slot 要 decode 的 token 組成 `SeqGroup` 列表一次過送進 `decode_multi`，權重只讀一次，多人共用。Prompt 以 chunk 輪流處理，夾在 decode 步之間，避免長 prompt 令其他人停頓。
- **Paged KV pool**：KV 以 256 個位置為一塊（等於 attention chunk 大小），每個 slot 一張 block table。`--ctx` 是每個 slot 上限（128K），`--kv-pool` 是總量（128K）。用多少佔多少，所以 4 個 slot 都「可以」去到 128K，只是加起來不能超過 pool。
  - **為什麼不給每個 slot 各自預留 128K**：要多約 5 GB VRAM，常駐 expert 由 5900 減到約 3100，命中率由約 90% 跌到約 55%，單人速度會跌到 25–35 tok/s。共用 pool 不影響單人速度。
- **Admission control**：請求入隊時先計算需要多少 block（prompt + min(max_tokens, 4096)），扣除其他執行中請求已承諾但未分配的 block，再加上可以清走的閒置 slot cache。放不下就排隊，等前面完成；只有單一請求本身大於 pool 才立即回 503。第一版沒有「承諾」的概念，兩個 70K 請求同時進來，都以為有位，結果其中一個做了 444 秒才失敗。
- **Prefix cache 與 checkpoint**：新請求會分配到前綴重疊最長的 slot。DeltaNet 狀態不能「截斷」回某個位置，所以在 prompt 最後一個 `<|im_start|>` 處把遞迴狀態存到 host RAM，多輪對話追問時從 checkpoint 繼續，只需處理新增部分。

## 8. 伺服器功能

- **OpenAI 相容 API**：httplib + nlohmann json，支援串流、`reasoning_content`（思考內容分開輸出）、tool calls。串流時要確保不在 UTF-8 字元中間切斷，否則 json 序列化會拋例外。
- **Chat template**：用 C++ 重新實作 Qwen 的 jinja template，與 jinja2 參考輸出逐 byte 比對（`qe render`、`engine/tt/`）。
- **圖片**：借用 llama.cpp 的 libmtmd，視覺編碼器在 CPU 跑（不佔 VRAM）。圖片 token 用負數假 id 參與前綴比對；位置編碼是交錯式 M-RoPE（按 `i % 3` 分配 t/h/w 三段），每個 slot 有自己的 `rope_delta`。

## 9. 時間都花在哪裏（decode，每輪 MTP 約 30 ms）

| 部分 | 時間 |
|---|---|
| Expert gate/up + down（GPU） | 約 11.8 ms |
| CPU 冷門 expert | 約 7.3 ms |
| DeltaNet 投影 + 輸出 | 約 7.2 ms |
| Router/top-k、attention、lm_head、MTP 等 | 約 9 ms |

沒有單一瓶頸，每部分約佔四分之一。這是為什麼後期的局部優化（如 IQ4_NL 解碼快 20–60%、dense 降 Q4）對整體速度幾乎沒有影響。要再大幅提升，需要同時改善多個部分，或更換硬件（例如更高頻寬的 GPU）。

## 10. 已知弱點

- 多個長 prompt 同時進來時，prefill 是輪流做的；4 個 28K prompt 需要約 4 分鐘才一齊出第一個字。可以改為把多個 slot 的 prompt 合併成一個 batch 計算。
- 長 context 下 attention 效率仍有提升空間（110K decode 只有 21 tok/s）。
- 引擎只支援 Qwen3.6-35B-A3B 這一個模型，而且針對這台機調校（resident 數量、thread 數等）。

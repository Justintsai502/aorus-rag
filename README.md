# AORUS MASTER 16 AM6H — 規格問答 RAG

在 **4 GB VRAM** 內回答 [GIGABYTE AORUS MASTER 16 AM6H](https://www.gigabyte.com/tw/Laptop/AORUS-MASTER-16-AM6H/sp)
規格問題的繁中／英文問答系統。RAG 核心（chunking、檢索、融合、prompt、streaming）
全部手寫，無 LangChain / LlamaIndex；推論引擎為 llama.cpp，環境由 uv 管理。

```bash
git clone https://github.com/Justintsai502/aorus-rag && cd aorus-rag
uv sync
uv run aorus-rag search "螢幕更新率是多少"     # 立刻可跑，零下載
```

語料與向量索引已 commit 進 repo，**不需要 build**。沒有模型時自動改用 BM25 檢索。

| | Recall@3 | 關鍵字正確率 | 拒答率 | 數字接地 | 答案 TTFT | TPS |
|---|---|---|---|---|---|---|
| **本系統（top-5）** | **0.971** | 96.8% | 100% | 100% | 0.113 s | 61.7 |
| 同模型無 RAG | — | 29.0% | **0%** | **0%** | — | — |

VRAM ≈ 1.6 GB。MacBook Pro M2 / 8 GB / Metal。Qwen3-1.7B Q4_K_M + bge-m3。
檢索指標為 40 題全集，生成指標為其中 36 題核心集。

---

## 1. 題目要求對照

| 要求 | 實作 |
|---|---|
| No Frameworks | chunking、BM25、向量索引、RRF、prompt、streaming 全手寫。依賴僅 `numpy` + `httpx` + `llama-cpp-python` |
| uv | `pyproject.toml` + `uv.lock` + `.python-version`，全部 commit |
| llama.cpp | in-process（預設，數據皆出自此）與 llama-server 兩條路徑 |
| 4 GB VRAM | 約 1.6 GB，帳本由 GGUF metadata 計算 |
| 繁中 + 英文混合 | 雙語 key 錨點 + bge-m3 跨語檢索 + 語言偵測 |
| Key-Value 解析 | stdlib `html.parser`，含結構驗證斷言 |
| Streaming | 逐 token yield，含 TTFT / TPS 打點 |
| TTFT / TPS | §5 |
| 定性評測 | 40 題自建集，含拒答與跨欄位推理 |

## 2. 安裝與執行

```bash
# 零下載：BM25 檢索
uv sync && uv run aorus-rag search "Thunderbolt 5 在哪一側"

# 完整功能
CMAKE_ARGS="-DGGML_METAL=on" uv sync --extra llama   # macOS，CUDA 用 -DGGML_CUDA=on
bash scripts/download_models.sh                      # qwen3-1.7b + bge-m3，1.74 GB
uv run aorus-rag ask "這台的電池容量是多少？"
uv run aorus-rag eval-retrieval                      # 檢索評測，不需生成模型
uv run aorus-rag bench --repeats 3 --top-k-sweep 3 5 --no-rag-control
```

系統依環境自動選擇可用的路徑：

| 手上有什麼 | 檢索 | 生成 |
|---|---|---|
| 只有 repo | BM25 | ✗ |
| + bge-m3（0.63 GB） | hybrid | ✗ |
| + 生成模型（1.11 GB） | hybrid | ✓ |

> macOS 若出現 `ModuleNotFoundError: aorus_rag`，執行 `chflags -R nohidden .venv`。

## 3. 架構

```
BUILD（離線，已完成並 commit）
  fetch      httpx + 瀏覽器 header + HTTP/2
  parse      stdlib HTMLParser → 17 組雙語 Key-Value
  normalize  規則式抽取 56 條原子事實
  chunk      Key-anchored 三層切分 → 240 chunks
  embed      bge-m3（CPU）→ index.npz（240 × 1024）

ASK（線上）
  偵測語言 → dense(numpy cosine) ⊕ BM25 → RRF → key boost → 去重
           → 組 prompt → llama.cpp streaming
```

Embedding 模型跑在 CPU，VRAM 帳本只需為生成模型負責。

## 4. 設計決策

### 4.1 記憶體帳本

| 項目 | 佔用 |
|---|---|
| Qwen3-1.7B Q4_K_M | 1.11 GB |
| KV cache（n_ctx=4096, q8_0） | 0.232 GB |
| compute buffer | ~0.30 GB |
| **VRAM** | **≈ 1.6 GB / 4 GB** |
| bge-m3 Q8_0（CPU） | 0.63 GB，不計入 VRAM |

KV cache 由 GGUF metadata 算出：`2(K,V) × 28 層 × 8 KV head × 128 dim × 1.06 B ≈ 112 KB/token`。

### 4.2 模型選擇

同一組題目、同一個 retriever 下比較兩顆模型：

| 模型 | top-5 關鍵字 | TPS | 答案 TTFT | 權重+KV |
|---|---|---|---|---|
| **Qwen3-1.7B** | 96.8% | **61.7** | **0.113 s** | **1.34 GB** |
| Qwen2.5-3B | 96.8% | 36.0 | 0.167 s | 2.01 GB |

品質相同，1.7B 快 1.7 倍、少 0.67 GB，預設採用 1.7B。
RAG 是抽取式任務 —— 答案已在 context 中，模型只需讀懂並引用 —— 對參數量不敏感。
兩者答錯的題目不同（rs02 vs rs01，皆為接孔推理），屬於同一類問題的共同上限。

1.7B 需要較深的檢索：top-3 時關鍵字 93.5%、誤拒率 6.5%（3B 為 96.8%、3.2%），
因此預設 `top_k = 5`。

### 4.3 為什麼 llama.cpp 而非 vLLM

vLLM 的核心優化（PagedAttention、continuous batching）針對**多使用者調度**。
本題是單機、單使用者、batch = 1，這些優化沒有收益，卻需要 CUDA、全模型進 VRAM
與 torch 依賴。vLLM 解決的是「記憶體充足但使用效率不足」，本題的限制是
「記憶體不足」。llama.cpp 的原生低 bit 量化、KV cache 量化、CPU/GPU 分層卸載
與本題限制逐項對應。

### 4.4 Key-anchored chunking

按規格表自身結構切成三種粒度，每塊都貼上**雙語鍵前綴**：

```
L0 spec_row   顯示器 / Display: 16" 16:10; OLED WQXGA (2560×1600) 240Hz; …
L1 spec_line  顯示器 / Display: VESA DisplayHDR True Black 500
L2 fact       螢幕更新率 / Display refresh rate: 240Hz
```

規格表的「值」在中英文頁面逐字節相同，只有「鍵」被翻譯，
中文問題透過雙語鍵前綴來匹配。

| kind | 數量 | |
|---|---|---|
| `feature` | 121 | 特色頁敘述（散熱、GiMATE…） |
| `fact` | 56 | 原子事實 |
| `spec_line` | 36 | 規格表單行 |
| `spec_row` | 17 | 規格表整列 |
| `footnote` | 6 | 註腳 |
| `sku` | 4 | 三個型號的差異（§4.6） |

### 4.5 Hybrid 檢索

| retriever | Recall@1 | Recall@3 | MRR | 延遲 |
|---|---|---|---|---|
| dense (bge-m3) | 0.914 | 0.971 | 0.950 | 17.2 ms |
| bm25 | 0.886 | 0.914 | 0.907 | **0.17 ms** |
| hybrid (RRF) | 0.914 | 0.971 | 0.944 | 17.0 ms |

在此語料規模（240 chunks）下，hybrid 與 dense 表現相同。BM25 的作用是保底：
不需任何模型、0.17 ms 完成，並作為可切換的 ablation 組（`--mode dense|bm25|hybrid`）。

dense 的 17 ms 中 16.9 ms 為 query embedding，向量搜尋僅 0.031 ms，因此不需要 ANN 索引。

### 4.6 SKU 型號

BZH / BYH / BXH 是 AM6H 的三個銷售型號（三者規格頁皆導向 AM6H），彼此只差顯示晶片。
規格頁比較表的值與型號名稱位於不同 DOM 區塊，系統依欄位順序配對後產生帶型號標記的 chunk：

```
AORUS MASTER 16 BZH 的顯示晶片: RTX 5090; 24GB GDDR7; 175W
AORUS MASTER 16 BYH 的顯示晶片: RTX 5080; 16GB GDDR7; 175W
AORUS MASTER 16 BXH 的顯示晶片: RTX 5070 Ti; 12GB GDDR7; 140W
```

每個值都綁定型號代號：「顯示卡是什麼」對應 AM6H 規格表，「哪個型號是 RTX 5080」對應 BYH。
有差異的欄位由程式自動偵測（目前為顯示晶片 1 欄）。

### 4.7 Prompt

- 依參考資料回答，需要時組合多筆資料
- 資料中沒有時回答「提供的規格資料中沒有這項資訊」，不標來源
- 規格數字逐字引用，不換算單位
- 先寫完整答案，再標來源編號
- 附兩個範例（一個可答、一個應拒答）

## 5. 評測

### 5.1 方法

40 題自建集（`data/eval/qa.jsonl`）：繁中 10、英文 10、中英混合 5、
**拒答 5**（5G、售價、保固、指紋辨識、續航）、跨欄位推理 6、SKU 4。
檢索評測使用全部 40 題，生成評測使用其中 36 題核心集（不含 SKU）。

```
recall@k          任一 gold document 出現在 top-k
keyword accuracy  must_include 字串全部出現（支援「擇一組」語法）
refusal rate      negative 題正確拒答的比例
false refusal     可回答的題目卻拒答的比例
number grounding  答案中的數字都能在 context 中找到的比例
TTFT              送出 → 第一個 token
ttft_answer       送出 → 第一個答案 token（排除 Qwen3 的 think 區塊）
TPS               (n_tokens - 1) / decode_s，decode-only
```

每題 3 次取中位數，warmup 不計，測量時關閉其他應用程式。

### 5.2 結果

**(a) top-k 掃描**（Qwen2.5-3B，36 題核心集）

| top-k | prompt tokens | TTFT | TPS | 關鍵字 | 誤拒 |
|---|---|---|---|---|---|
| 1 | 269 | **0.102 s** | 36.0 | 80.7% | 16.1% |
| 3 | 338 | 0.165 s | 35.9 | 96.8% | 3.2% |
| 5 | 480 | 0.167 s | 36.0 | 96.8% | **0.0%** |
| 8 | 650 | 0.296 s | 35.1 | 96.8% | 0.0% |

prompt 從 269 → 650 tokens，**TTFT 增加 2.9 倍，TPS 幾乎不變（−2.5%）** ——
prefill 是整段 prompt 的 GEMM，decode 是每 token 一次 GEMV。

逐題來看：top-1 的 6 題錯誤中有 5 題是誤拒（答案未被檢索到）；
top-8 時多出的 context 包含特色頁的模糊敘述（「Ultra 200HX 系列」），
與規格表的「Ultra 9 275HX」競爭。top-k 曲線先上升、飽和、再下降，預設取 5。

**(b) RAG vs no-RAG**（同模型，唯一差別是有沒有 context）

| 條件 | 關鍵字正確率 | negative 拒答率 | 數字接地 |
|---|---|---|---|
| **RAG (top-5)** | **96.8%** | **100%** | **100%** |
| no-RAG（同模型、無 context） | 29.0% | **0%** | **0%** |

AM6H 是 2025 年新品，模型預訓練資料不包含它：

```
Q: 這台筆電的電池容量是多少？
   RAG    : 電池容量是 99Wh [1]。
   no-RAG : ...電池容量為 95Wh。

Q: 顯示卡是哪一張？
   RAG    : NVIDIA® GeForce RTX™ 5090 Laptop GPU [3]。
   no-RAG : ...是 NVIDIA GeForce RTX 3080。

Q: 這台筆電支援 5G 行動網路嗎？
   RAG    : 提供的規格資料中沒有這項資訊。
   no-RAG : 不支援，因為它是一款筆電，通常不具備外接 5G 設備的插槽或連接埠。
```

no-RAG 的 negative 拒答率為 0%；RAG 讓模型在資料不足時明確回答「沒有這項資訊」。

**(c) TPS 對照理論上限**

decode 為 memory-bandwidth bound：`M2 頻寬 100 GB/s ÷ 模型 1.11 GB ≈ 90 tok/s`，
實測 61.7 tok/s，達理論值 69%。

## 6. 已知限制

1. 模型對照為兩顆（1.7B、3B）；4B 與其他量化等級（Q5_K_M / Q8_0）未測。
2. 關鍵字命中無法判斷答案是否完全正確（例：rs06 答出色域 `DCIP-3 100%` 而非 HDR 等級，仍計為命中）。
3. 未採用 LLM-as-judge，以可重現的自動指標為主。
4. SKU 題的正解部分排在第 3–5 名，依賴 `top_k = 5`。
5. 評測集為自行撰寫。
6. 生成評測為 36 題核心集，不含 SKU 題。

## 7. 專案結構

```
pyproject.toml / uv.lock / .python-version   uv 環境（lock 已 commit）
scripts/download_models.sh                   GGUF 下載（純 curl，可續傳）
data/raw/*.html + manifest.json              快取網頁與出處紀錄
data/corpus.jsonl                            240 chunks
data/index.npz                               240 × 1024（bge-m3）
data/eval/qa.jsonl                           40 題評測集
results/                                     評測輸出
src/aorus_rag/
  config      路徑、模型規格、runtime 參數      fetch     下載與快取
  parse       HTML 解析 + 結構驗證              normalize 56 條原子事實
  chunk       Key-anchored 三層切分             embed     llama.cpp embedding
  index       BM25 / VectorIndex / RRF          retrieve  融合、boost、去重
  prompt      語言偵測、context packing          llm       streaming + 計時
  pipeline    build / ask 兩階段                bench     評測指標
  cli         7 個指令
tests/                                       37 個測試，不需模型
```

MIT

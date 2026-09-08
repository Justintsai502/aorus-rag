# AORUS MASTER 16 AM6H — 規格問答 RAG（4 GB VRAM 手寫實作）

在 **4 GB VRAM 預算**內，從零手寫一套繁中／英文混合的規格問答系統，回答
[GIGABYTE AORUS MASTER 16 AM6H](https://www.gigabyte.com/tw/Laptop/AORUS-MASTER-16-AM6H/sp)
的產品規格問題。

**不使用任何 RAG 框架**：Chunking、Embedding、Vector Index、BM25、Hybrid 融合、
Prompt 組裝、Streaming 解析全部為純 Python 實作，共約 2,700 行。
推論引擎為 **llama.cpp**，環境由 **uv** 管理。

```bash
uv sync
uv run aorus-rag build --embed-model hashing     # 不需下載模型即可跑通
uv run aorus-rag search "螢幕更新率是多少"         # 純檢索，零模型
```

---

## 目錄

1. [快速開始](#1-快速開始)
2. [系統架構](#2-系統架構)
3. [4 GB VRAM 記憶體帳本與模型選擇](#3-4-gb-vram-記憶體帳本與模型選擇)
4. [No-Framework 對照表](#4-no-framework-對照表)
5. [資料解析：規格表的結構化處理](#5-資料解析規格表的結構化處理)
6. [RAG 設計決策與取捨](#6-rag-設計決策與取捨)
7. [評測方法與結果](#7-評測方法與結果)
8. [已知限制與後續改進](#8-已知限制與後續改進)

---

## 1. 快速開始

### 環境

```bash
git clone <this repo> && cd aorus-rag
uv sync                       # 建立環境（uv 會自動取得 Python 3.11）
uv run pytest -q              # 23 個測試，不需要任何模型
```

### 安裝推論引擎（需編譯，約 5–10 分鐘）

`llama-cpp-python` 必須針對硬體編譯，因此獨立成 optional dependency：

```bash
# macOS（Apple GPU / Metal）
CMAKE_ARGS="-DGGML_METAL=on" uv sync --extra llama

# NVIDIA GPU（Colab / Kaggle）
CMAKE_ARGS="-DGGML_CUDA=on"  uv sync --extra llama
```

> ⚠️ 忘記帶 `CMAKE_ARGS` 會編出純 CPU 版本，能跑但 TPS 會低數倍。
> 載入模型時 log 中出現 `ggml_metal_init` 或 `ggml_cuda_init` 才代表編對了。

### 下載模型

```bash
bash scripts/download_models.sh          # 預設組合 qwen2.5-3b + e5-small，約 2.06 GB
bash scripts/download_models.sh all      # 全部五個模型，約 5.7 GB（做對照實驗用）
```

### 執行

```bash
uv run aorus-rag build --embed-model e5-small        # 建語料 + 向量索引
uv run aorus-rag ask "這台的電池容量是多少？"           # 串流回答
uv run aorus-rag ask "How many Type-C ports?" --show-context
uv run aorus-rag eval-retrieval                      # 檢索評測（不需 LLM）
uv run aorus-rag bench --repeats 3 --top-k-sweep 1 3 5 8 --no-rag-control
```

### 在 Kaggle / Colab 上重現

```python
!git clone <this repo> && cd aorus-rag
!pip install uv
!cd aorus-rag && CMAKE_ARGS="-DGGML_CUDA=on" uv sync --extra llama
!cd aorus-rag && bash scripts/download_models.sh
!cd aorus-rag && uv run aorus-rag bench --repeats 3
!nvidia-smi --query-gpu=memory.used --format=csv    # 驗證 VRAM 峰值
```

uv 會在專案內建立獨立的 `.venv`，與 notebook 預裝的數百個套件完全隔離。

> **macOS 疑難排解**：若出現 `ModuleNotFoundError: No module named 'aorus_rag'`，
> 是 uv 在 venv 內的檔案被標記為 macOS hidden flag，而 Python 3.11+ 的 `site`
> 模組會跳過 hidden 的 `.pth` 檔。修正：`chflags -R nohidden .venv`。

---

## 2. 系統架構

```
              ┌──────────── build 階段（只載 embedding 模型，峰值 ~0.3 GB）────────────┐
              │                                                                      │
  gigabyte.com│  fetch.py      httpx + 完整瀏覽器 header（繞過 Akamai 403）             │
      │       │      ↓         data/raw/*.html（已 commit，可離線重現）                │
      ▼       │  parse.py      stdlib HTMLParser → 17 組雙語 Key-Value + 特色頁內文     │
   4 個頁面    │      ↓                                                                │
  (zh / en)   │  normalize.py  規則式抽取 56 條原子事實（240Hz、99Wh、TB5 在左側…）      │
              │      ↓                                                                │
              │  chunk.py      Key-anchored 三層切分 → 236 chunks                      │
              │      ↓                                                                │
              │  embed.py      llama.cpp embedding（CPU）→ index.py → data/index.npz   │
              └──────────────────────────────────────────────────────────────────────┘

              ┌──────────── ask 階段（只載生成模型，VRAM ~2.1 GB）────────────────────┐
              │                                                                      │
   使用者提問  │  prompt.py     語言偵測（CJK 比例）                                    │
      │       │      ↓                                                                │
      ▼       │  retrieve.py   dense(numpy cosine) ⊕ BM25 → RRF → key boost → 去重     │
              │      ↓                                                                │
              │  prompt.py     Key-anchored context + 引用編號 + 抗幻覺指令             │
              │      ↓                                                                │
              │  llm.py        llama.cpp streaming，逐 token 打點（TTFT / TPS）         │
              └──────────────────────────────────────────────────────────────────────┘
```

**兩階段分離是刻意的**：embedding 模型與生成模型從不同時常駐，
所以 VRAM 帳本只需要為生成模型負責。

---

## 3. 4 GB VRAM 記憶體帳本與模型選擇

### 3.1 帳本（預設配置）

| 項目 | 配置 | 佔用 |
|---|---|---|
| 生成模型權重 | Qwen2.5-3B-Instruct **Q4_K_M** | **1.93 GB** |
| KV cache | `n_ctx=4096`、`type_k/type_v=q8_0` | **~0.15 GB** |
| compute buffer / overhead | — | ~0.30 GB |
| Embedding 模型 | multilingual-e5-small q8_0，**掛 CPU** | 0.13 GB（不計入 VRAM） |
| **VRAM 合計** | | **≈ 2.4 GB / 4 GB** ✅ |

KV cache 計算（Qwen2.5-3B：36 層、2 個 KV head × 128 dim）：

```
每 token = 2 (K,V) × 36 層 × 256 dim × 1 byte (q8_0) ≈ 18 KB
4096 tokens ≈ 0.15 GB          （f16 則為 0.29 GB）
```

### 3.2 三組可選配置

| 配置 | 生成模型 | Embedding | VRAM 小計 | 適用 |
|---|---|---|---|---|
| A 保守 | Qwen3-1.7B Q4_K_M `1.11 GB` | e5-small `0.13` | **~1.6 GB** | 4 GB 以下、極限環境 |
| **B 預設** | **Qwen2.5-3B Q4_K_M `1.93 GB`** | **e5-small `0.13`** | **~2.4 GB** | **本專案預設** |
| C 進取 | Qwen3-4B-Instruct-2507 Q4_K_M `2.50 GB` | bge-m3 Q8_0 `0.63` | ~3.3 GB | 4 GB 上限、追求品質 |

### 3.3 為什麼是 Qwen2.5-3B + Q4_K_M

**任務性質決定了模型規模。** RAG 把知識負擔外包給檢索，模型只需要
「讀懂 context 並抽取／改寫」，而非「記得規格」：

```
Context: 電池 / Battery: Li-ion 99Wh
Question: 這台電池多大？
Answer:  99Wh [1]
```

這是**抽取式**任務，對指令遵循敏感、對參數量不敏感。3B 級距在此已足夠，
而多出的 VRAM 拿去換更長的 `n_ctx`（能塞更多檢索結果）比換更大的模型划算。

**為什麼不是 Llama-3.2-3B / Gemma-3-4B**：兩者在繁體中文都會出現簡繁混寫，
對台灣使用者是明顯缺陷。Qwen 系列的中文訓練資料比例最高。

**為什麼是 Q4_K_M**：社群長期共識是 Q4_K_M 以上掉點極小、Q3 以下明顯退步。
本專案不引用他人結論，`bench` 指令支援直接量測 Q4_K_M / Q5_K_M / Q8_0 對照
（見 [§7](#7-評測方法與結果)）。

### 3.4 為什麼是 llama.cpp 而不是 vLLM

vLLM 的核心創新 —— **PagedAttention**（KV cache 分頁）與
**continuous batching**（token 級排程）—— 都是為了解決**多使用者調度**問題：
讓 batch 維持又大又滿，把權重搬運成本攤提到更多序列上。

```
本題情境：單機、單使用者、batch = 1
  ├─ PagedAttention      沒有多份 KV cache 要調度        → 收益 0
  ├─ continuous batching 沒有等待佇列可以補位            → 收益 0
  ├─ prefix sharing      只有一個使用者                  → 收益 0
  └─ 大 batch 攤提       batch 恆為 1                    → 收益 0

同時要付出的成本：
  ├─ CUDA-only（M2 無法執行）
  ├─ 要求整個模型放得進 VRAM（無 n_gpu_layers 分層卸載退路）
  └─ 依賴整套 torch（CUDA 環境下磁碟 ~2.5 GB、RSS 數百 MB）
```

更根本的錯配：**vLLM 解的是「記憶體很多但用得不夠有效率」，本題的問題是
「記憶體根本不夠」** —— 方向相反。

llama.cpp 的優化方向與本題限制逐項對應：

| 本題限制 | llama.cpp 對應能力 |
|---|---|
| 4 GB VRAM | 原生低 bit 量化（K-quant / I-quant），量化是一等公民 |
| KV cache 佔用 | `type_k/type_v` 量化，KV 記憶體直接砍半 |
| 記憶體不足退路 | `n_gpu_layers` CPU/GPU 分層卸載 |
| 消費級筆電（Apple Silicon） | Metal / CUDA / ROCm / Vulkan / CPU 全支援 |
| 要量測單人 TTFT | in-process 呼叫，無 HTTP 往返，無框架 overhead |

---

## 4. No-Framework 對照表

### 本專案手寫的部分

| 元件 | 實作 | 檔案 |
|---|---|---|
| HTML 解析 | stdlib `html.parser` 狀態機（不用 BeautifulSoup / lxml） | `parse.py` |
| 結構化抽取 | 規則式原子事實抽取 + I/O 側邊結構解析 | `normalize.py` |
| Chunking | Key-anchored 三層切分 + 句界滑動視窗 | `chunk.py` |
| 中英混合斷詞 | 英數 token + 中文 char unigram/bigram（不用 jieba） | `index.py` |
| BM25 | Okapi BM25（k1=1.5, b=0.75）含 IDF 平滑 | `index.py` |
| Vector Index | numpy L2 正規化 + 內積 = cosine（不用 FAISS / Chroma） | `index.py` |
| Hybrid 融合 | Reciprocal Rank Fusion（k=60） | `index.py` |
| 檢索後處理 | Key 精確命中 boost、doc 級多樣性去重 | `retrieve.py` |
| Prompt 組裝 | 雙語 system prompt、context packing、預算裁切 | `prompt.py` |
| Streaming | 逐 token yield + TTFT/TPS 打點 | `llm.py` |
| 評測 | Recall@k / MRR / 關鍵字命中 / 拒答率 / 數字接地 | `bench.py` |

### 使用的依賴（皆非 RAG 框架）

| 套件 | 用途 | 為什麼不算框架 |
|---|---|---|
| `numpy` | 矩陣乘法 | 純數學工具；不知道什麼是 chunk 或檢索，所有檢索決策由本專案控制 |
| `llama-cpp-python` | 推論引擎 binding | 題目指定；不介入 RAG 邏輯 |
| `httpx` | HTTP 客戶端 | 抓網頁 |

### 明確排除

`LangChain`、`LlamaIndex`、`Haystack`、`ChromaDB`、`FAISS`、
`sentence-transformers`、`torch`、`transformers`。

後三者嚴格說是 library 而非 framework，但**它們取代掉的正是題目要求手寫的環節**
（embedding 封裝、向量索引），且 `sentence-transformers → torch` 的依賴鏈會直接
吃掉 4 GB 預算的一大塊。整個專案**不需要 torch**。

---

## 5. 資料解析：規格表的結構化處理

### 5.1 反爬蟲

規格頁位於 Akamai Bot Manager 後方，需要**兩個**條件才會放行：

```
① 裸請求
   → 403 Access Denied

② 完整瀏覽器 header 集（UA / Accept / Accept-Language / sec-ch-ua* /
                       Sec-Fetch-* / Upgrade-Insecure-Requests）+ HTTP/1.1
   → 仍然 403  ← 這一步卡了很久

③ 同樣的 header + HTTP/2
   → 200 OK
```

**第二個條件是 HTTP/2。** 真實 Chrome 一定協商 h2，所以「Chrome UA 卻走
HTTP/1.1」本身就是機器人特徵。這也是 `httpx[http2]` 出現在依賴清單、
而 `fetch.py` 裡 `http2=True` 是必要而非優化的原因。

頁面是 **server-side rendered**，所以不需要 Playwright 這類 headless browser。
`fetch.py` 帶完整 header，抓下的 HTML 存入 `data/raw/` 並 **commit 進 repo**，
讓評測結果在網站改版或離線環境下仍可重現。

### 5.2 一個會產生「看似正確的錯誤答案」的陷阱

規格頁同時包含 **AM6H 本身**與**三台姊妹機（BZH / BYH / BXH）的比較欄位**：

```html
<!-- AM6H：有 title 有 value -->
<ul class="spec-item-list">
  <li class="spec-title"><div>中央處理器</div></li>
  <li class="spec-desc"><div>Intel® Core™ Ultra 9 Processor 275HX ...</div></li>
</ul>

<!-- 桌機比較欄位：只有 value，共 51 個（3 台 × 17 列）-->
<div class="spec-item-list" data-spec-row="1"><span>...</span></div>
```

若用「抓所有 `.spec-item-list`」的直覺寫法，會把**另外三台筆電的規格混進語料**——
而且產生的錯誤答案看起來完全合理。

解法：解析器鎖定 `li.spec-title` / `li.spec-desc` 的 `ul` 變體（恰好 17 列），
並在 `validate_spec_items()` 中斷言列數與姊妹機型號未洩漏。這是測試套件裡最嚴格的一條。

### 5.3 一個影響設計的發現

**中英文頁面的規格「值」完全相同（17/17 逐字節一致），只有「鍵」被翻譯。**

```
zh: 中央處理器      → Intel® Core™ Ultra 9 Processor 275HX (36MB cache, ...)
en: CPU            → Intel® Core™ Ultra 9 Processor 275HX (36MB cache, ...)
                     └────────────── 完全相同 ──────────────┘
```

**推論：雙語問題完全發生在「鍵」這一側。** 一個中文問題（「螢幕更新率多少」）
在值裡面沒有任何中文可以匹配。這直接決定了 chunking 策略必須把
**雙語鍵當作錨點前綴**（見 §6.1）。這條發現寫成了測試
（`test_keys_are_translated_but_values_are_not`）。

### 5.4 原子事實抽取

從 17 列自由文字再抽出 **56 條可直接回答的原子事實**：

```
display.refresh_rate   螢幕更新率 / Display refresh rate = 240Hz
display.contrast       對比度 / Display contrast ratio   = 1,000,000:1
battery.capacity       電池容量 / Battery capacity        = 99Wh
io.count.usb_c         Type-C 連接埠數量                  = 2
io.side.thunderbolt5   Thunderbolt 5 位置                = 左側 / Left side
```

I/O 列另有專屬處理：原始文字以 `Left Side:` / `Right Side:` 分段，
解析為側邊標記的 port 清單，再衍生出**數量聚合**與**位置查詢**兩類事實 ——
「有幾個 Type-C？」不該要求 3B 模型自己數清單。

> 實作細節：`display.contrast` 的正規表示式最初寫成 `[\d,]+:1`，
> 會從 `16:10`（螢幕比例）誤抓出 `16:1`。已改為要求千分位分隔格式，
> 並寫成回歸測試。

---

## 6. RAG 設計決策與取捨

### 6.1 Key-anchored chunking（最重要的一個決定）

一般切分器的預設值（`chunk_size=1000, overlap=200`）套在這份資料上會這樣：

```
❌ 通用切分：整份規格表 ~3,000 字 → 3 塊大雜燴
   「作業系統 Windows 11 Pro… 中央處理器 Ultra 9 275HX (36MB cache, 5.4 GHz,
     24 cores)… 顯示晶片 RTX 5090 24GB GDDR7 175W… 螢幕 2560×1600 240Hz
     500nits… 記憶體 64GB DDR5 5600MHz…」

   問「電池多大？」→ 檢索回這一塊 → 裡面有 36MB / 5.4 / 24 / 175 / 64 / 5600…
   3B 模型很容易挑錯一個數字
```

```
✅ 本專案：三層粒度，每一塊都貼上雙語鍵錨點
   L0 spec_row   顯示器 / Display: 16" 16:10; OLED WQXGA (2560×1600) 240Hz; …
   L1 spec_line  顯示器 / Display: OLED WQXGA (2560×1600) 240Hz, 1ms, …
   L2 fact       螢幕更新率 / Display refresh rate: 240Hz
```

**錨點前綴不是裝飾。** 沒有它，`1 x HDMI 2.1` 這種單行的 embedding
幾乎不帶「它回答什麼問題」的訊號；而依照 §5.3 的發現，中文問題在值裡
沒有任何中文可匹配。

語料組成（236 chunks）：

| kind | 數量 | 來源 |
|---|---|---|
| `fact` | 56 | 原子事實（最高精度） |
| `spec_line` | 36 | 規格表單行 |
| `spec_row` | 17 | 規格表整列 |
| `footnote` | 6 | 註腳（標記為註記，避免被當成規格引用） |
| `feature` | 121 | 特色頁敘述文（散熱、GiMATE 等「how/why」問題） |

### 6.2 為什麼一定要 Hybrid（Dense + BM25）

規格表充滿**精確字串**：`Thunderbolt 5`、`5600MHz`、`RTX 5090`、`802.11be`。
這正好是小型多語 embedding 模型最弱的地方（英數 token 的語意訊號稀薄），
卻是 BM25 最強的地方。

```
Dense  → 覆蓋語意改寫（「螢幕多亮」↔「brightness」）
BM25   → 覆蓋精確識別碼（「TB5」「99Wh」「Q4_K_M」）
RRF    → 用「排名」而非「分數」融合，
          不必讓 [0,1] 的 cosine 與無上界的 BM25 分數可比較
```

中文斷詞採 **char bigram 而非詞典斷詞**：不需附帶詞典、沒有產品術語的
OOV 問題，且「更新率」與「螢幕更新率」仍能部分重疊。

### 6.3 檢索後處理

- **Key 精確命中 boost**（+35%）：問題字面包含某 chunk 的鍵時直接加權。
  「螢幕更新率是多少」含有「螢幕更新率」——這比任何相似度分數都強，且成本是一次子字串比對。
- **Doc 級多樣性**：同一規格列最多取 2 個 chunk，避免 context 是同一列的五種切法。
- **精度優先的 tie-break**：分數相同時 `fact` > `spec_line` > `spec_row` > `feature`。

### 6.4 Prompt 設計

四條硬性規則（`prompt.py`）：

1. 只根據 context 回答；沒有就明說「沒有這項資訊」
2. **規格數字逐字照抄，不得換算單位**（3B 模型很樂意把 99Wh 換算成「約 26,000mAh」）
3. 每個事實標來源編號 `[1]`
4. 使用者用什麼語言就用什麼語言回答；中文用台灣用語

### 6.5 核心取捨：top-k 與 TTFT

```
TTFT ≈ 檢索時間 + prefill 時間
                  └─ 與 prompt token 數成正比

top-k ↑  → context ↑ → prefill 矩陣列數 ↑ → TTFT 變差
                                          → TPS 幾乎不變（decode 每步只生 1 token）
```

`bench --top-k-sweep 1 3 5 8` 直接量測這條曲線（見 §7.3）。

---

## 7. 評測方法與結果

### 7.1 評測集

自建 36 題（`data/eval/qa.jsonl`）：

| 類型 | 題數 | 目的 |
|---|---|---|
| 繁中事實題 | 10 | 基本正確率 |
| 英文事實題 | 10 | 跨語檢索 |
| 中英混合題 | 5 | 語系混用（`What's the 電池 capacity?`） |
| **Negative（規格表沒有的）** | 5 | 拒答能力（5G、售價、保固年限、指紋辨識、續航時數） |
| 跨欄位推理題 | 6 | 需組合多列（外接雙 4K、自行升級記憶體與 SSD…） |

每題標註 `gold_docs`（正解來源列）與 `must_include`（答案必須包含的字串／數字）。

### 7.2 指標定義

```
── 檢索（不需要 LLM，CPU 上毫秒級）────────────────────────
recall@k  任一 gold document 出現在 top-k 的比例
mrr       第一個 gold document 的倒數排名平均

── 生成 ──────────────────────────────────────────────
keyword accuracy  must_include 字串全部出現在答案中
refusal rate      negative 題是否正確拒答
false refusal     可回答的題目卻拒答（拒答不能靠一律說不知道來刷分）
number grounding  答案中每個數字都能在 context 中找到
                  → 找不到即為幻覺，零成本且不需 judge 模型

── 延遲（定義寫死在 llm.StreamStats）──────────────────
TTFT      送出請求 → 第一個非空 token
decode_s  第一個 token → 最後一個 token
tps       (n_tokens - 1) / decode_s      decode-only，排除 prefill
e2e_tps   n_tokens / total_s             端到端，包含 prefill
```

分開報 decode-only 與 end-to-end 是必要的：prefill 是整段 prompt 的 GEMM、
decode 是每 token 一次 GEMV，混在一起會掩蓋 §6.5 那條取捨。

每題跑 3 次取中位數，第一次 warmup 不計。

### 7.3 結果

#### 檢索：dense vs BM25 vs hybrid

以下為**已實測**數據（36 題中 31 題有 gold document；`hashing` embedder）：

| retriever | Recall@1 | Recall@3 | Recall@5 | MRR | 中位延遲 |
|---|---|---|---|---|---|
| dense (hashing) | 0.742 | 0.806 | 0.903 | 0.786 | 0.06 ms |
| **bm25** | **0.903** | **0.935** | **1.000** | **0.927** | 0.15 ms |
| hybrid | 0.871 | 0.935 | 0.935 | 0.898 | 0.21 ms |

> **這組數字的正確讀法**：`hashing` 是零依賴的 fallback embedder
> （token 雜湊 + 符號投影），**只捕捉字面重疊、完全沒有語意能力**——
> 它無法把「螢幕多亮」對到 `brightness`。在這個條件下 dense 只是一個較差的
> 詞彙檢索器，BM25 勝出是預期內的結果，hybrid 被拖累也是。
>
> 這組數字的價值在於**確立 lexical-only 的下界**：即使完全沒有語意檢索，
> Key-anchored chunking + BM25 已經能達到 Recall@5 = 1.000。
> 真正的 dense / hybrid 對照需要換上 `e5-small`：
> `uv run aorus-rag build --embed-model e5-small && uv run aorus-rag eval-retrieval`

#### 生成品質與延遲

> **狀態：尚未執行。** 這一節需要下載 GGUF 權重（約 2.06 GB）並編譯
> `llama-cpp-python`。指令與表格結構如下，執行後 `results/bench.json`
> 會產生所有欄位。

```bash
uv run aorus-rag bench --repeats 3 --top-k-sweep 1 3 5 8 --no-rag-control
```

**(a) 模型對照**（同一組 36 題、同一 retriever）

| 模型 | 量化 | VRAM | TTFT (s) | TPS | 關鍵字正確率 | 拒答率 | 數字接地 |
|---|---|---|---|---|---|---|---|
| Qwen3-1.7B | Q4_K_M | 1.6 GB | — | — | — | — | — |
| Qwen2.5-3B | Q4_K_M | 2.4 GB | — | — | — | — | — |
| Qwen3-4B | Q4_K_M | 3.3 GB | — | — | — | — | — |

**(b) top-k 對延遲的影響**（§6.5 的核心取捨）

| top-k | prompt tokens | TTFT (s) | TPS | 關鍵字正確率 |
|---|---|---|---|---|
| 1 | — | — | — | — |
| 3 | — | — | — | — |
| 5 | — | — | — | — |
| 8 | — | — | — | — |

**(c) RAG vs no-RAG 對照組**

AM6H 為 2025 年新品，模型預訓練資料不可能包含其規格，
因此 no-RAG 組的任何「正確」答案都應視為巧合。

| 條件 | 關鍵字正確率 | 數字接地 |
|---|---|---|
| RAG (top-k=4) | — | — |
| no-RAG（同模型、無 context） | — | — |

**理論參考值**：decode 為 memory-bandwidth bound，TPS 上界約為
`記憶體頻寬 ÷ 模型大小`。M2 頻寬約 100 GB/s、模型 1.93 GB →
理論上限約 52 tok/s，實測應落在其下。README 定稿時會將實測值與此理論值並列。

---

## 8. 已知限制與後續改進

### 限制

1. **生成端數據尚未量測**（見 §7.3）；檢索端數據為實測。
2. **`hashing` embedder 沒有語意能力**，只作為零依賴的開發／測試 fallback，
   不應視為正式檢索器。
3. **LLM-as-judge 未採用**。用同一顆 3B 模型評自己的答案不可靠，
   本專案改用 `must_include` 關鍵字命中與 number grounding 這類
   可驗證、可重現的自動指標，並誠實承認其覆蓋面較窄
   （例如無法評價流暢度與語氣）。
4. **特色頁的區塊切分偏粗**：該頁 DOM 巢狀鬆散，多數段落落在一個 catch-all
   區塊中，因此改用產品名稱作為統一錨點而非該區塊標題（避免誤標）。
5. **評測集為本人撰寫**，存在與系統設計同源的偏差風險。

### 後續改進

- **Prefix caching**：system prompt 每次相同（約 200 tokens），其 KV 可重用。
  llama.cpp 支援，預期直接改善 TTFT，且可做成 A/B 對照實驗。
- **量化掃描**：Q4_K_M / Q5_K_M / Q8_0 的品質-記憶體曲線實測。
- **雙環境對照**：M2 MacBook（Metal，即題目所指的「消費級筆電」）
  與 Kaggle T4（CUDA，可用 `nvidia-smi` 提供 4 GB 上限的可驗證證據）。
- **Reranker**：目前靠 RRF + key boost，可加一顆極小的 cross-encoder，
  但需重新核算 VRAM 帳本。

---

## 專案結構

```
pyproject.toml / uv.lock / .python-version    uv 環境定義（lock 已 commit）
scripts/download_models.sh                    GGUF 下載（純 curl，可續傳）
data/raw/*.html                               快取網頁（可離線重現）
data/corpus.jsonl                             236 個 chunk
data/eval/qa.jsonl                            36 題評測集
results/                                      評測輸出
src/aorus_rag/
  config.py      路徑、來源 URL、模型規格、runtime 參數
  fetch.py       帶瀏覽器 header 的下載與快取
  parse.py       stdlib HTMLParser 解析 + 汙染防護斷言
  normalize.py   56 條原子事實抽取
  chunk.py       Key-anchored 三層切分
  embed.py       llama.cpp embedding + hashing fallback
  index.py       BM25 / VectorIndex / RRF / 混合斷詞
  retrieve.py    融合、key boost、多樣性去重
  prompt.py      語言偵測、context packing、雙語 prompt
  llm.py         llama.cpp streaming + TTFT/TPS 打點
  pipeline.py    build / ask 兩階段編排
  bench.py       檢索與生成評測
  cli.py         指令列介面
tests/                                        23 個測試，不需模型
```

## 授權

MIT

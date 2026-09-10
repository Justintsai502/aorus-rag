# AORUS MASTER 16 AM6H — 規格問答 RAG（4 GB VRAM 手寫實作）

在 **4 GB VRAM 預算**內，從零手寫一套繁中／英文混合的規格問答系統，回答
[GIGABYTE AORUS MASTER 16 AM6H](https://www.gigabyte.com/tw/Laptop/AORUS-MASTER-16-AM6H/sp)
的產品規格問題。

**不使用任何 RAG 框架**：Chunking、Embedding、Vector Index、BM25、Hybrid 融合、
Prompt 組裝、Streaming 解析全部為純 Python 實作，共約 2,700 行。
推論引擎為 **llama.cpp**，環境由 **uv** 管理。

```bash
uv sync
uv run aorus-rag build --embed-model hashing   # 零下載，僅供跑通管線（會印出警告）
uv run aorus-rag search "螢幕更新率是多少"       # 純檢索，不需要生成模型
```

**實測結果**（MacBook Pro M2 8GB、Metal、Qwen2.5-3B Q4_K_M + bge-m3）：

| | 檢索 Recall@3 | 關鍵字正確率 | negative 拒答率 | 數字接地 | TTFT | TPS |
|---|---|---|---|---|---|---|
| **本系統（RAG, top-3）** | **1.000** | **96.8%** | **100%** | **100%** | 0.165 s | 35.9 |
| 同模型、無 RAG 對照 | — | 25.8% | 0% | 0% | 0.115 s | 36.0 |

峰值 VRAM ≈ 2.3 GB / 4 GB。詳見 [§7 評測](#7-評測方法與結果)。

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
uv run pytest -q              # 29 個測試，不需要任何模型
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
bash scripts/download_models.sh          # 預設組合 qwen2.5-3b + bge-m3，約 2.56 GB
bash scripts/download_models.sh all      # 全部五個模型，約 6.3 GB（做對照實驗用）
```

### 執行

```bash
uv run aorus-rag build                               # 建語料 + 向量索引（預設 bge-m3）
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
| KV cache | `n_ctx=4096`、`type_k/type_v=q8_0` | **0.075 GB** |
| compute buffer / overhead | — | ~0.30 GB |
| **VRAM 合計** | | **≈ 2.3 GB / 4 GB** ✅ |
| Embedding 模型 | bge-m3 Q8_0，**掛 CPU**（`n_gpu_layers=0`） | 0.63 GB（不計入 VRAM） |

KV cache 由 GGUF metadata 直接算出，不是估的：

```
從檔案讀到：block_count=36, embedding_length=2048,
            head_count=16, head_count_kv=2
head_dim  = 2048 / 16 = 128
KV 維度   = head_count_kv × head_dim = 2 × 128 = 256

每 token = 2 (K,V) × 36 層 × 256 × 1.0625 B (q8_0 含 scale) = 19.1 KB
4096 tokens = 0.075 GB          （f16 則為 0.141 GB）
```

> **一個反直覺的發現**：參數量只有一半的 Qwen3-1.7B，KV cache 反而大 3 倍 ——
>
> | 模型 | 層數 | KV heads | KV/token | @4096 q8_0 |
> |---|---|---|---|---|
> | Qwen2.5-3B | 36 | **2** | 36 KB | **0.075 GB** |
> | Qwen3-1.7B | 28 | **8** | 112 KB | **0.232 GB** |
>
> 原因是 GQA 的壓縮程度不同：Qwen2.5-3B 把 KV head 從 16 個減到 2 個，
> Qwen3-1.7B 只減到 8 個。**決定 KV cache 大小的是 KV head 數，不是參數量。**
> 在 context 拉長時這個差距會放大，所以「換小模型」不必然省記憶體 ——
> 要看你省的是權重還是 KV。

> **關於 Apple Silicon 的統一記憶體**：M2 沒有獨立 VRAM，CPU 與 GPU 共用同一塊
> 實體記憶體，因此「embedding 掛 CPU 所以不計入 VRAM」這個說法只在**有獨立
> 顯示記憶體的環境**（如 Colab / Kaggle 的 T4）成立。在 Mac 上，總記憶體佔用
> 是 1.93 + 0.63 + 0.075 + overhead ≈ 3.0 GB，仍在 4 GB 內，但兩個數字的意義不同：
> 前者是「4 GB VRAM 限制」的達成證明，後者是本機實際佔用。README §7 的兩組
> 硬體數據分別對應這兩種情況。

### 3.2 三組可選配置

| 配置 | 生成模型 | 權重 | KV@4096 q8_0 | VRAM 小計 | 適用 |
|---|---|---|---|---|---|
| A 保守 | Qwen3-1.7B Q4_K_M | 1.11 GB | 0.232 GB | **~1.6 GB** | 極限環境 |
| **B 預設** | **Qwen2.5-3B Q4_K_M** | **1.93 GB** | **0.075 GB** | **~2.3 GB** | **本專案預設** |
| C 進取 | Qwen3-4B-Instruct-2507 Q4_K_M | 2.50 GB | — | ~3.1 GB | 4 GB 上限、追品質 |

Embedding 一律用 bge-m3 Q8_0（0.63 GB，掛 CPU）。

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

### 3.4 兩種 llama.cpp 使用方式

題目允許「llama.cpp（Python binding / Server）」，兩條路徑都實作了：

| 後端 | 呼叫方式 | 用途 |
|---|---|---|
| **`--backend in-process`**（預設） | `llama-cpp-python` 直接載入 GGUF | TTFT 最低，無 HTTP 往返、無框架 overhead。**README 所有評測數據都出自這條路徑** |
| `--backend server` | HTTP + SSE 打 `llama-server` 的 `/v1/chat/completions` | 展示可部署形態 |

啟動 server（需額外依賴）：

```bash
uv sync --extra server
uv run python -m llama_cpp.server --model models/Qwen2.5-3B-Instruct-Q4_K_M.gguf \
    --n_gpu_layers -1 --n_ctx 4096 --port 8080
uv run aorus-rag ask "電池多大？" --backend server
```

> 誠實標註：**server 路徑僅提供程式碼，未納入實測數據。** 量測 TTFT 時多一層
> HTTP 與 SSE 解析會混入非模型的延遲，這正是預設走 in-process 的原因。

### 3.5 為什麼是 llama.cpp 而不是 vLLM

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

規格頁位於 Akamai Bot Manager 後方。實測六種組合，**只有一格通過**：

| headers | 協定 | 結果 |
|---|---|---|
| Chrome 128 完整 header | HTTP/1.1 | 403 |
| **Chrome 128 完整 header** | **HTTP/2** | **200** ✅ |
| 誠實的 bot UA | HTTP/1.1 | 403 |
| 誠實的 bot UA | HTTP/2 | 403 |
| Chrome 28（HTTP/2 出現前的瀏覽器） | HTTP/1.1 | 403 |
| 無 header | HTTP/2 | 403 |

**兩個條件是 AND，缺一不可：**

```
① 完整的現代瀏覽器 header
   （UA / Accept / Accept-Language / sec-ch-ua* / Sec-Fetch-* /
     Upgrade-Insecure-Requests）
② HTTP/2
```

值得注意的是，這不是「宣稱 Chrome 卻走 HTTP/1.1」的矛盾偵測 ——
Chrome 28 這種 HTTP/2 出現前的 UA 走 HTTP/1.1 並不矛盾，但一樣被擋。
兩個條件各自獨立生效。

這就是 `httpx[http2]` 出現在依賴清單、而 `fetch.py` 裡 `http2=True`
是必要條件而非效能優化的原因。（附帶一提：`httpx` 預設走 HTTP/1.1，
所以這個問題會在「用 curl 驗證成功之後、改寫成 Python」時才浮現。）

頁面是 **server-side rendered**，所以不需要 Playwright 這類 headless browser。
`fetch.py` 帶完整 header，抓下的 HTML 存入 `data/raw/` 並 **commit 進 repo**，
讓評測結果在網站改版或離線環境下仍可重現。

### 5.2 一個會產生「看似正確的錯誤答案」的陷阱

規格頁同時包含 **AM6H 本身**與**三個 SKU（BZH / BYH / BXH）的比較欄位**：

```html
<!-- AM6H：有 title 有 value -->
<ul class="spec-item-list">
  <li class="spec-title"><div>中央處理器</div></li>
  <li class="spec-desc"><div>Intel® Core™ Ultra 9 Processor 275HX ...</div></li>
</ul>

<!-- 桌機比較欄位：只有 value，共 51 個（3 台 × 17 列）-->
<div class="spec-item-list" data-spec-row="1"><span>...</span></div>
```

瀏覽器實測（375px vs 1600px）確認兩份表格由 CSS 互斥切換：

| | 手機 375px | 桌機 1600px |
|---|---|---|
| `.mobile-spec-content` | 顯示 | 隱藏 |
| `.desktop-spec-content` | 隱藏 | 顯示 |
| 畫面上的 GPU 數量 | **1**（RTX 5090） | **3**（5090 / 5080 / 5070 Ti） |

桌機比較表的三欄標題為 **BZH / BYH / BXH**，AM6H 本身不在其中。三欄之間
**17 列裡有 16 列完全相同**，只有「顯示晶片」不同：

```
BZH  RTX 5090 Laptop GPU    24GB GDDR7   175W   AI Boost 1797 MHz
BYH  RTX 5080 Laptop GPU    16GB GDDR7   175W   AI Boost 1902 MHz
BXH  RTX 5070 Ti Laptop GPU 12GB GDDR7   140W   AI Boost 1962 MHz
```

這三個代號**不是別的機型，是同一台機器的三個 SKU**。實測三個代號的規格頁
URL 全部 302 重導向回 AM6H 頁面：

```
/tw/Laptop/AORUS-MASTER-16-{BZH,BYH,BXH}/sp  ──302──►  /tw/Laptop/AORUS-MASTER-16-AM6H
/tw/Laptop/AORUS-MASTER-16-AM6H/sp           ──200──►  自己的規格表（17 列）
```

所以這是**同一個實體的多個版本**，而不是多個實體 —— 混進同一份語料會產生
「同一個問題的三個互相矛盾的答案」，比混進不相干的資料更難被模型化解。

若用「抓所有 `.spec-item-list`」的直覺寫法，會把**另外三台筆電的規格混進語料**。
危險之處在於 16/17 的資料是對的 —— 抽檢「電池多大」「CPU 是什麼」「重量多少」
全部會通過，只有顯示卡這一題會拿到三個互相矛盾的答案，而那正是電競筆電的頭號規格。

值和機型名稱還分存在不同 DOM 區塊（值在 `div.spec-item-list`，機型名稱在
sticky header），所以抓值時不會自動帶到「這是 BXH 的」這個上下文。

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

### 6.2 Hybrid（Dense + BM25）—— 以及它為什麼沒有贏

原始假設是「規格表充滿精確字串（`Thunderbolt 5`、`5600MHz`、`RTX 5090`、
`802.11be`），這是小型 embedding 模型最弱、BM25 最強的地方，所以 hybrid 必勝」：

```
Dense  → 覆蓋語意改寫（「外接 4K 螢幕」↔「HDMI 2.1」）
BM25   → 覆蓋精確識別碼（「TB5」「99Wh」）
RRF    → 用「排名」而非「分數」融合，
          不必讓 [0,1] 的 cosine 與無上界的 BM25 分數可比較
```

**實測結果不支持這個假設**（詳見 §7.3）：

| retriever | Recall@1 | Recall@3 | MRR |
|---|---|---|---|
| dense (bge-m3) | 0.968 | **1.000** | 0.984 |
| bm25 | 0.903 | 0.935 | 0.927 |
| hybrid (RRF) | 0.968 | **1.000** | 0.984 |

**hybrid 與 dense 完全相同。** 原因是 dense 在 Recall@3 已經飽和 —— 沒有空間
可以再改善。BM25 找得到而 dense 找不到的題目，一題也沒有。

保留 hybrid 的理由因此改變了，而且要說清楚：

1. **它是保底而非增益。** BM25 單獨也有 Recall@5 = 1.000，且延遲只有
   0.14 ms（dense 的 1/160）。若 embedding 模型載入失敗或語料換成
   embedder 不熟的領域，BM25 這條路仍在。
2. **它是可切換的 ablation 組。** `--retriever dense|bm25|hybrid` 讓這個
   結論是被量出來的，而不是被假設的。
3. **在更大的語料上結論可能反轉。** 236 chunks 太小，dense 很容易飽和；
   數千 chunk 時 hybrid 通常才會顯出價值。

> 這一節保留原始假設與否定它的數據，是刻意的。**把「我猜 hybrid 會贏」改寫成
> 「hybrid 果然贏了」很容易，但那會讓這份報告失去它唯一真正的價值 ——
> 數字是量出來的。**

中文斷詞採 **char bigram 而非詞典斷詞**：不需附帶詞典、沒有產品術語的
OOV 問題，且「更新率」與「螢幕更新率」仍能部分重疊。

### 6.3 Embedding 必須與索引同源

向量檢索有一個會**默默算出垃圾**的失效模式：用 A 模型建索引、用 B 模型查詢。
兩者的向量活在不同的空間，cosine 相似度照樣算得出數字，只是毫無意義 ——
沒有例外、沒有警告、結果看起來完全正常。

因此 embedding 的每一項模型專屬設定都跟著模型走，而不是寫死：

| 模型專屬項目 | e5-small (BERT 系) | bge-m3 | 實作位置 |
|---|---|---|---|
| 非對稱前綴 | `query: ` / `passage: ` | 無 | `embed.PREFIXES`，查詢與語料分別套用 |
| Pooling | mean | CLS | `ModelSpec.pooling` → llama.cpp `pooling_type` |
| 向量維度 | 384 | 1024 | 建索引時實測，寫入 `index.npz` |
| 最大長度 | 512 | 1024 | `ModelSpec.n_ctx` |

`index.npz` 記錄建索引時用的模型名稱，查詢時若不一致直接拒絕執行：

```
$ uv run aorus-rag search "..." （索引是 hashing 建的，但要求 bge-m3）
error: index was built with 'hashing' but 'bge-m3' was requested;
       rebuild with `uv run aorus-rag build --embed-model bge-m3`
```

`hashing` fallback 只用於零下載的管線驗證，使用時 build / search /
eval-retrieval 都會在 stderr 印出醒目警告，避免它的數字被誤當成正式結果。

### 6.4 檢索後處理

- **Key 精確命中 boost**（+35%）：問題字面包含某 chunk 的鍵時直接加權。
  「螢幕更新率是多少」含有「螢幕更新率」——這比任何相似度分數都強，且成本是一次子字串比對。
- **Doc 級多樣性**：同一規格列最多取 2 個 chunk，避免 context 是同一列的五種切法。
- **精度優先的 tie-break**：分數相同時 `fact` > `spec_line` > `spec_row` > `feature`。

### 6.5 Prompt 設計

四條硬性規則（`prompt.py`）：

1. 只根據 context 回答；沒有就明說「沒有這項資訊」
2. **規格數字逐字照抄，不得換算單位**（3B 模型很樂意把 99Wh 換算成「約 26,000mAh」）
3. 每個事實標來源編號 `[1]`
4. 使用者用什麼語言就用什麼語言回答；中文用台灣用語

### 6.6 核心取捨：top-k 與 TTFT

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
decode 是每 token 一次 GEMV，混在一起會掩蓋 §6.6 那條取捨。

每題跑 3 次取中位數，第一次 warmup 不計。

### 7.3 結果

#### 檢索：dense vs BM25 vs hybrid

實測數據（36 題中 31 題有 gold document，embedder = bge-m3 Q8_0）：

| retriever | Recall@1 | Recall@3 | Recall@5 | MRR | 中位延遲 |
|---|---|---|---|---|---|
| dense (bge-m3) | **0.968** | **1.000** | 1.000 | **0.984** | 23.7 ms |
| bm25 | 0.903 | 0.935 | 1.000 | 0.927 | **0.14 ms** |
| hybrid (RRF) | 0.968 | 1.000 | 1.000 | 0.984 | 22.8 ms |

**三個值得說明的結果：**

**① Hybrid 沒有贏過 dense。** 在這個語料規模（236 chunks）配上一顆強的多語
embedding 模型，dense 的 Recall@3 已經是 1.000 —— 沒有空間可以再改善。
Hybrid 在這裡的價值不是提升上限，而是**在 dense 失效時提供保底**
（BM25 單獨就有 Recall@5 = 1.000）。若語料擴大到數千 chunk、或換上較弱的
embedder，兩者的差距才會出現。誠實地說：**就這份資料而言，hybrid 是保險，不是增益。**

**② BM25 輸的正好是「換句話說」的題目。** 逐題比對後，BM25 排名落後而 dense
排第一的只有兩題，都是跨欄位推理題：

```
rs01  我想外接兩台 4K 螢幕，這台有哪些影像輸出接孔可以用？   bm25 第 5 名 → dense 第 1 名
rs02  出門只想帶一條線，同時充電又外接螢幕，該插哪個孔？      bm25 第 5 名 → dense 第 1 名
```

規格表的原文是 `1 x HDMI 2.1` 和 `Type-C with Thunderbolt™5 (support USB4,
DisplayPort™ 2.1 and Power Delivery 3.0)` —— 字面上完全沒有「4K」「外接」
「充電」。**這就是語意檢索存在的理由，也是純 BM25 方案的天花板。**

dense 唯一沒排第一的是 `顯示卡是哪一張？`（第 2 名）—— 口語的「顯示卡」
對上規格表的「顯示晶片」。

**③ 那 23 ms 幾乎全部是 embedding，不是搜尋。**

```
把問題轉成向量（bge-m3 跑 CPU）    20.78 ms   ← 佔 100%
236×1024 矩陣乘法找最相似          0.031 ms   ← 千分之一
```

**向量搜尋本身是免費的**（31 微秒），這也證實了不引入 FAISS/HNSW 的判斷 ——
近似最近鄰要優化的那 0.031 ms 根本不是瓶頸。

真正的取捨是：**dense 用 21 ms 的 TTFT 換 +6.5 個百分點的 Recall@1**。
在單人、延遲敏感的情境下，這個交換是否值得取決於延遲預算；若要壓 TTFT，
把 embedder 換小或移上 GPU 比優化搜尋演算法有效得多。

#### 生成品質與延遲

**硬體**：MacBook Pro M2、8 GB 統一記憶體、Metal 後端
**模型**：Qwen2.5-3B-Instruct Q4_K_M、`n_ctx=4096`、KV cache `q8_0`
**檢索**：hybrid（bge-m3 + BM25 + RRF）
**方法**：36 題 × 每題 3 次取中位數，第一次 warmup 不計；測量時關閉其他應用程式

##### (a) top-k 掃描 —— RAG 的核心取捨

| top-k | prompt tokens | TTFT (s) | TPS (decode) | e2e tok/s | 關鍵字正確率 | 拒答率 | 誤拒率 | 數字接地 |
|---|---|---|---|---|---|---|---|---|
| 1 | 269 | **0.102** | 36.0 | 33.2 | 80.7% | 100% | 16.1% | 97.0% |
| **3** | 338 | 0.165 | 35.9 | 32.9 | **96.8%** | 100% | 3.2% | 100% |
| **5** | 480 | 0.167 | 36.0 | 33.7 | **96.8%** | 100% | **0.0%** | 98.4% |
| 8 | 650 | 0.296 | 35.1 | 32.0 | 96.8% | 100% | 0.0% | 100% |

**預測被完全證實**：prompt 從 269 → 650 tokens，**TTFT 惡化 2.9 倍（0.102 → 0.296 s）**，
而 **TPS 幾乎不動（36.0 → 35.1，−2.5%）**。原因就是 §6.6 那條 ——
prefill 是整段 prompt 的 GEMM（隨長度線性成長），decode 是每 token 一次 GEMV
（與 prompt 長度幾乎無關）。

**逐題失敗分析比平均值更有資訊量：**

```
top-1  6 題錯 —— 5 題是「誤拒」：只給 1 段 context，答案根本沒被撈進來
top-3  1 題錯 —— rs06「螢幕支援 HDR 嗎？」答案在第 4 名的 chunk
top-5  1 題錯 —— rs01「外接兩台 4K 螢幕」只答了 Type-C，漏掉 HDMI
top-8  1 題錯 —— zh04「用的是什麼處理器？」答「Ultra 200HX 系列」
```

**top-8 的錯誤特別值得注意**：正確答案是規格表的 `Ultra 9 Processor 275HX`，
但多撈的 5 段裡有一段特色頁文案寫著「最新 Intel® Core™ Ultra 200HX 系列處理器」，
模型挑了那個較模糊的來源。

> **所以 top-k 不是「越大越準、只是變慢」。** 超過某個點之後，多出來的 context
> 會引入**競爭性的干擾來源**，準確率不升反降。這條曲線是先上升、飽和、再劣化。

**結論**：預設 `top_k=4`。top-3 已經飽和，top-5 把誤拒率壓到 0，
兩者 TTFT 幾乎相同（0.165 vs 0.167 s），而 top-8 開始引入干擾。

##### (b) RAG vs no-RAG —— 這是整份報告最有力的一組數字

同一顆模型、同樣 36 題，唯一差別是有沒有給檢索到的 context：

| 條件 | 關鍵字正確率 | negative 題拒答率 | 數字接地 | prompt tokens |
|---|---|---|---|---|
| **RAG (top-3)** | **96.8%** | **100%** | **100%** | 338 |
| no-RAG（同模型、無 context） | 25.8% | **0%** | **0%** | 40 |

AM6H 是 2025 年新品，模型預訓練資料不可能包含它。實際輸出：

```
Q: 這台筆電的電池容量是多少？
   RAG    : 這台筆電的電池容量是 99Wh [1]。
   no-RAG : GIGABYTE AORUS MASTER 16 AM6H 的電池容量為 95Wh。        ← 編了一個很像的數字

Q: 顯示卡是哪一張？
   RAG    : 顯示卡是 NVIDIA® GeForce RTX™ 5090 Laptop GPU [3]。
   no-RAG : ...的顯示卡是 NVIDIA GeForce RTX 3080。                   ← 差兩個世代

Q: How much video memory does the GPU have?
   RAG    : The GPU has 24GB GDDR7 [1].
   no-RAG : ...comes with a NVIDIA GeForce RTX 3090 GPU, which has 24 GB...  ← 型號錯、容量猜對
```

**最能說明問題的是 negative 題**（規格表中確實沒有的資訊）：

```
Q: 這台筆電支援 5G 行動網路嗎？
   RAG    : 提供的規格資料中沒有這項資訊。
   no-RAG : GIGABYTE AORUS MASTER 16 AM6H 不支援 5G 行動網路，
            因為它是一款筆電，通常不具備外接 5G 設備的插槽或連接埠。   ← 連理由都編了
```

**no-RAG 的 negative 拒答率是 0% —— 它從不說「不知道」。** 這正是 RAG 要解決的問題：
不是讓模型變聰明，而是讓它**知道自己不知道**。

`數字接地 0%` 這個數字也值得解釋：no-RAG 條件下沒有 context 可以比對，
所以答案中的每一個數字依定義都是未接地的 —— 而它確實產生了大量數字（95Wh、
RTX 3080…），也就是說它**用同樣自信的語氣輸出了完全捏造的規格**。

##### (c) top-3 的分類別正確率

| 題型 | 正確 / 總數 |
|---|---|
| 事實題（繁中 10 + 英文 10 + 混合 5） | **25 / 25** |
| Negative（該拒答） | **5 / 5** |
| 跨欄位推理 | 5 / 6 |

唯一失手的 rs06「螢幕支援 HDR 嗎？是哪一個等級？」在 top-5 與 top-8 都答對
（`VESA DisplayHDR True Black 500`），是檢索深度而非模型能力的問題。

##### (d) TPS 對照理論上限

decode 是 memory-bandwidth bound，理論上限 ≈ `記憶體頻寬 ÷ 模型大小`：

```
M2 記憶體頻寬 ≈ 100 GB/s
模型大小       = 1.93 GB
理論上限       ≈ 100 / 1.93 ≈ 52 tok/s
實測           = 36 tok/s   → 達到理論值的 69%
```

差距來自 KV cache 讀取、attention、sampling 與非權重的記憶體流量。
**69% 是量化模型在 Metal 上的合理區間** —— 這個對照的用意是證明數字有物理依據，
而不是報一個孤立的測量值。

##### (e) 尚未完成的部分

| 項目 | 狀態 |
|---|---|
| Qwen3-1.7B 對照組 | ⛔ 未完成 |
| Qwen3-4B 對照組 | ⛔ 未下載 |
| Colab / Kaggle T4 的 `nvidia-smi` VRAM 佐證 | ⛔ 未執行 |

原因誠實記錄：本機為 8 GB 統一記憶體，跑完 3B 的完整掃描（約 470 次生成）後
機器發熱明顯，且同時常駐生成模型（1.93 GB）與 embedding 模型（0.63 GB）時
swap 已達 1.6 GB。多模型對照在此機器上會影響量測品質，**正確的做法是移到
Kaggle T4 執行**，而不是在受污染的環境下硬跑出數字。

重現指令：

```bash
uv run aorus-rag bench --model qwen3-1.7b --repeats 3 --top-k-sweep 3 5 \
    --no-rag-control --output bench_qwen3-1.7b.json
```

## 8. 已知限制與後續改進

### 限制

1. **只有單一模型的實測數據**。Qwen2.5-3B 的完整掃描已完成，但 1.7B / 4B 的
   對照組尚未執行（見 §7.3(e)），因此「為什麼選 3B」目前是基於任務性質的論證
   加上單點實測，缺少跨模型的邊際效益曲線。
2. **`hashing` embedder 沒有語意能力**，只作為零依賴的開發／測試 fallback，
   不應視為正式檢索器；正式數據一律以 bge-m3 為準。
   另外，`cstr/multilingual-e5-small-GGUF` 這份轉檔在目前的 llama.cpp 無法載入
   （缺 `bert.token_type_count`），設定檔中保留該選項但標註了此限制。
3. **LLM-as-judge 未採用**。用同一顆 3B 模型評自己的答案不可靠，
   本專案改用 `must_include` 關鍵字命中與 number grounding 這類
   可驗證、可重現的自動指標，並誠實承認其覆蓋面較窄
   （例如無法評價流暢度與語氣）。
4. **特色頁的區塊切分偏粗**：該頁 DOM 巢狀鬆散，多數段落落在一個 catch-all
   區塊中，因此改用產品名稱作為統一錨點而非該區塊標題（避免誤標）。
5. **評測集為本人撰寫**，存在與系統設計同源的偏差風險。
6. **單一 SKU 範圍**：AM6H 是機型頁，BZH / BYH / BXH 是它的三個 SKU
   （已由 302 重導向確認，見 §5.2），彼此只差顯示卡：5090 / 5080 / 5070 Ti。
   本系統只涵蓋 AM6H 規格表本身所載的內容，因此「顯示卡是什麼」會答 RTX 5090
   （即該頁的正式規格，對應最高階的 BZH 配置）。跨 SKU 查詢不在範圍內 ——
   正確的支援方式是把 SKU 代號做成 chunk 的 metadata 欄位並讓檢索能依 SKU 過濾。
   這是 RAG 常見的一類問題：同一實體有多個版本／時間點／地區的資料時，
   解法是加維度（metadata + 過濾），而不是全部塞進 context 讓模型自己判斷。

### 後續改進

- **Prefix caching**：system prompt 每次相同（約 200 tokens），其 KV 可重用。
  llama.cpp 支援，預期直接改善 TTFT，且可做成 A/B 對照實驗。
- **模型與量化掃描**：1.7B / 3B / 4B 以及 Q4_K_M / Q5_K_M / Q8_0 的
  品質-記憶體-延遲曲線，應在 Kaggle T4 上執行以避免本機記憶體壓力污染量測。
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
tests/                                        29 個測試，不需模型
```

## 授權

MIT

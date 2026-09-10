# AORUS MASTER 16 AM6H — 規格問答 RAG

在 **4 GB VRAM** 內回答 [GIGABYTE AORUS MASTER 16 AM6H](https://www.gigabyte.com/tw/Laptop/AORUS-MASTER-16-AM6H/sp)
規格問題的繁中／英文問答系統。RAG 核心（chunking、檢索、融合、prompt、streaming）
全部手寫，無 LangChain / LlamaIndex；推論引擎為 llama.cpp，環境由 uv 管理。

```bash
git clone https://github.com/Justintsai502/aorus-rag && cd aorus-rag
uv sync
uv run aorus-rag search "螢幕更新率是多少"     # 立刻可跑，零下載
```

語料與向量索引已 commit 進 repo，**不需要 build**。沒有模型時自動降級為 BM25 檢索。

| | Recall@3 | 關鍵字正確率 | 拒答率 | 數字接地 | 答案 TTFT | TPS |
|---|---|---|---|---|---|---|
| **本系統（top-5）** | **0.971** | 96.8% | 100% | 100% | 0.113 s | 61.7 |
| 同模型無 RAG | — | 29.0% | **0%** | **0%** | — | — |

峰值 VRAM ≈ 1.6 GB。MacBook Pro M2 / 8 GB / Metal。Qwen3-1.7B Q4_K_M + bge-m3。
檢索指標為 40 題全集，生成指標為其中 36 題核心集（見 §5.1）。

---

## 1. 題目要求對照

| 要求 | 實作 |
|---|---|
| No Frameworks | chunking、BM25、向量索引、RRF、prompt、streaming 全手寫。依賴僅 `numpy` + `httpx` + `llama-cpp-python` |
| uv | `pyproject.toml` + `uv.lock` + `.python-version`，全部 commit |
| llama.cpp | in-process（預設，數據皆出自此）與 llama-server 兩條路徑 |
| 4 GB VRAM | 實測 1.6 GB，帳本由 GGUF metadata 精算 |
| 繁中 + 英文混合 | 雙語 key 錨點 + bge-m3 跨語檢索 + 語言偵測 |
| Key-Value 解析 | stdlib `html.parser`，含汙染防護斷言 |
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

系統依環境自動選擇能跑的最好路徑：

| 手上有什麼 | 檢索 | 生成 |
|---|---|---|
| 只有 repo | BM25 自動接手 | ✗ |
| + bge-m3（0.63 GB） | hybrid | ✗ |
| + 生成模型（1.11 GB） | hybrid | ✓ |

> macOS 若出現 `ModuleNotFoundError: aorus_rag`：uv 在 venv 內的檔案被標為 hidden，
> 而 Python 3.11+ 會跳過 hidden 的 `.pth`。修正：`chflags -R nohidden .venv`

## 3. 架構

```
BUILD（離線，已完成並 commit）
  fetch      httpx + 瀏覽器 header + HTTP/2（Akamai 兩者缺一不可）
  parse      stdlib HTMLParser → 17 組雙語 Key-Value
  normalize  規則式抽取 56 條原子事實
  chunk      Key-anchored 三層切分 → 240 chunks
  embed      bge-m3（CPU）→ index.npz（240 × 1024）

ASK（線上）
  偵測語言 → dense(numpy cosine) ⊕ BM25 → RRF → key boost → 去重
           → 組 prompt → llama.cpp streaming
```

**兩階段分離**：embedding 與生成模型從不同時常駐，VRAM 帳本只需為生成模型負責。

## 4. 設計決策

### 4.1 記憶體帳本

| 項目 | 佔用 |
|---|---|
| Qwen3-1.7B Q4_K_M | 1.11 GB |
| KV cache（n_ctx=4096, q8_0） | 0.232 GB |
| compute buffer | ~0.30 GB |
| **VRAM** | **≈ 1.6 GB / 4 GB** |
| bge-m3 Q8_0（掛 CPU） | 0.63 GB，不計入 VRAM |

KV cache 由 GGUF metadata 算出：`2(K,V) × 28 層 × 8 KV head × 128 dim × 1.06 B ≈ 112 KB/token`。

> **反直覺**：Qwen3-1.7B 的 KV cache 比 Qwen2.5-3B **大 3 倍**（0.232 vs 0.075 GB），
> 因為它保留 8 個 KV head、3B 只留 2 個。決定 KV 大小的是 KV head 數，不是參數量。
> context 拉到 8k 以上時，1.7B 省下的權重會被 KV 吃回去。

> **Apple Silicon 但書**：M2 無獨立 VRAM，「embedding 掛 CPU 不計入 VRAM」只在
> 有獨立顯存的環境（Colab/Kaggle T4）成立。Mac 上實際總佔用 ≈ 3.0 GB。

### 4.2 模型選擇（被自己的評測推翻）

原始判斷是「RAG 是抽取式任務，3B 足夠」，選了 Qwen2.5-3B。實測後改為 1.7B：

| 模型 | top-5 關鍵字 | TPS | 答案 TTFT | 權重+KV |
|---|---|---|---|---|
| **Qwen3-1.7B** | 96.8% | **61.7** | **0.113 s** | **1.34 GB** |
| Qwen2.5-3B | 96.8% | 36.0 | 0.167 s | 2.01 GB |

品質打平、快 1.7 倍、省 0.67 GB。兩者甚至答錯**不同**的題（rs02 vs rs01，都是接孔推理），
代表這不是誰比較強，而是兩者撞到同一個天花板。原始論證只對了一半 ——
「抽取式任務對參數量不敏感」對到連 3B 都是多餘的。

**但書**：1.7B 需要更深的檢索。top-3 時掉到 93.5%、誤拒率翻倍到 6.5%。
延遲預算緊到只能 top-3 時，3B 反而穩。

### 4.3 為什麼 llama.cpp 而非 vLLM

vLLM 的核心創新（PagedAttention、continuous batching）都在解決**多使用者調度**。
本題是單機、單使用者、batch = 1 —— 這些收益數學上為零，卻要付 CUDA-only、
全模型須進 VRAM、torch 依賴三項成本。更根本的錯配：vLLM 解的是「記憶體多但用得沒效率」，
本題是「記憶體根本不夠」。llama.cpp 的優化方向（原生低 bit 量化、KV cache 量化、
CPU/GPU 分層卸載）與本題限制逐項對應。

### 4.4 Key-anchored chunking

通用切分器的預設值（`chunk_size=1000`）會把整份 3000 字規格表切成 3 塊大雜燴，
問「電池多大」撈回的那塊裡有 `36MB`、`5.4GHz`、`175W`、`64GB`、`5600MHz` —— 小模型很容易挑錯。

本專案按規格表自身結構切成三種粒度，每塊都貼上**雙語鍵前綴**：

```
L0 spec_row   顯示器 / Display: 16" 16:10; OLED WQXGA (2560×1600) 240Hz; …
L1 spec_line  顯示器 / Display: VESA DisplayHDR True Black 500
L2 fact       螢幕更新率 / Display refresh rate: 240Hz
```

**前綴不是裝飾**：規格表的「值」在中英文頁面**逐字節相同**，只有「鍵」被翻譯 ——
中文問題在值裡沒有任何中文可以匹配。這條發現寫成了測試。

| kind | 數量 | |
|---|---|---|
| `feature` | 121 | 特色頁敘述（散熱、GiMATE…） |
| `fact` | 56 | 原子事實（最高精度） |
| `spec_line` | 36 | 規格表單行 |
| `spec_row` | 17 | 規格表整列 |
| `footnote` | 6 | 註腳，標記為註記避免被當規格引用 |
| `sku` | 4 | 三個型號的差異（§4.6） |

### 4.5 Hybrid —— 以及它為什麼沒有贏

原始假設是「規格表充滿精確字串，BM25 必能補 dense 的不足」。實測不支持：

| retriever | Recall@1 | Recall@3 | MRR | 延遲 |
|---|---|---|---|---|
| dense (bge-m3) | 0.914 | 0.971 | 0.950 | 17.2 ms |
| bm25 | 0.886 | 0.914 | 0.907 | **0.17 ms** |
| hybrid (RRF) | 0.914 | 0.971 | 0.944 | 17.0 ms |

**hybrid 與 dense 幾乎相同**。BM25 找得到而 dense 找不到的題目，一題也沒有。
保留 hybrid 的理由因此改變：它是**保底**（BM25 零模型、0.17 ms）與**可切換的
ablation 組**，不是增益。在更大的語料上結論可能反轉。

BM25 唯一勝出的場景是換句話說的推理題（`rs01`「外接兩台 4K 螢幕」對上規格表的
`1 x HDMI 2.1`，字面零重疊）—— 那正是 dense 存在的理由，方向相反。

**延遲的真相**：dense 的 17 ms 裡有 **16.9 ms 是 embedding，矩陣搜尋只要 0.031 ms**。
向量搜尋本身是免費的，這是不引入 FAISS/HNSW 的實據 —— 要優化的那 0.031 ms 不是瓶頸。

**RRF 的一個修正**：RRF 評的是共識，所以只有單邊找到的強訊號會被雙重懲罰。
實測案例：`rs06`「螢幕支援 HDR 嗎」的正解是 dense 第 1 名、BM25 完全沒找到，
RRF 把它壓到第 8 名，答案變成拒答。`_rescue_top_hits()` 保證每個檢索器的第 1 名
留在最終結果裡。第一版把它們**提到最前面**，代價是 Recall@1 掉 2.8 個百分點；
改成**附加在尾端**後指標完全不變，而失敗案例修好了。

### 4.6 SKU 變體：把「排除」改成「標記」

BZH / BYH / BXH 是 AM6H 實際販售的三個型號（三者的規格頁 URL 皆 302 導回 AM6H），
彼此**只差顯示晶片**。規格頁的桌機比較表帶著這三欄，但值與型號名稱**存在不同 DOM 區塊**。

主解析器刻意排除比較表 —— 無標記地混入會讓「顯示卡是什麼」出現三個矛盾答案。
但那資訊本身有價值，所以另外解析並**綁定**：

```
AORUS MASTER 16 BZH 的顯示晶片: RTX 5090; 24GB GDDR7; 175W
AORUS MASTER 16 BYH 的顯示晶片: RTX 5080; 16GB GDDR7; 175W
AORUS MASTER 16 BXH 的顯示晶片: RTX 5070 Ti; 12GB GDDR7; 140W
```

同樣三個值，無標記時是「一個問題的三個矛盾答案」，綁定後是「三個問題的三個答案」。
「哪些欄位有差異」由程式自動偵測（實測只有 1 欄），不是寫死。

### 4.7 Prompt

四條硬性規則，每一條都是被實測逼出來的：

- **可以組合多筆資料回答** —— 少了這句，模型會把需要跨欄位的題目一律拒答
- **不可以只輸出來源編號** —— 第一版讓 3B 回答「電池多大」時只吐出 `[1]`
- **規格數字逐字照抄，不得換算單位** —— 否則會把 99Wh 換算成「約 26,000mAh」
- **附兩個範例**（一個可答、一個該拒答）—— 小模型對範例的服從度遠高於敘述

## 5. 評測

### 5.1 方法

40 題自建集（`data/eval/qa.jsonl`）：繁中 10、英文 10、中英混合 5、
**拒答 5**（5G、售價、保固、指紋辨識、續航）、跨欄位推理 6、SKU 4。

```
recall@k          任一 gold document 出現在 top-k
keyword accuracy  must_include 字串全部出現（支援「擇一組」語法）
refusal rate      negative 題是否正確拒答
false refusal     可回答的題目卻拒答（防止靠一律拒答刷分）
number grounding  答案中每個數字都能在 context 中找到 → 找不到即幻覺
TTFT              送出 → 第一個 token
ttft_answer       送出 → 第一個**答案** token（差額即 thinking 成本）
TPS               (n_tokens - 1) / decode_s，decode-only
```

每題 3 次取中位數，warmup 不計，測量時關閉其他應用程式。

> **評測集範圍**：**檢索**評測跑完整 40 題。**生成**評測的數字測於 36 題核心集 ——
> 4 題 SKU 是後來新增的，重跑生成評測時監督腳本因 swap 衝到 4.2 GB 自動中止
> （見 §5.3(d)），未在受污染的環境下補測。SKU 題以檢索評測（Recall@5 = 1.000）
> 加上逐題生成驗證（4/4 正確，如「BZH 的顯示記憶體是 24GB GDDR7 [5]」）覆蓋。

### 5.2 一個差點讓對照組失效的陷阱

第一次跑 Qwen3-1.7B 時數據好得可疑：關鍵字 100%、TTFT 快 3.7 倍。
檢查原始輸出才發現 **36 個答案全部包著 `<think>` 區塊**（平均 569 字元的簡體中文獨白，
3B 只有 41 字元）—— Qwen3 是混合推理模型，thinking mode 預設開啟。

三個指標同時失真：關鍵字出現在獨白裡、TTFT 量到的是 `<think>` 的第一個 token、
未接地的數字全來自獨白。

處理：`ModelSpec.thinking_switch` 自動附加 `/no_think`（token 92 → 15）、
`strip_thinking()` 計分前剝除、新增 `ttft_answer_s` 量「第一個答案 token」。
非推理模型兩個 TTFT 相同，指標因此跨模型可比。

> **這是自建評測最大的風險**：指標會給你一個數字，但不會告訴你它量錯了東西。
> 只看彙總表格的話，這份報告會宣稱「1.7B 全面勝過 3B」。

### 5.3 結果

**(a) top-k 掃描**（Qwen2.5-3B，36 題核心集）

| top-k | prompt tokens | TTFT | TPS | 關鍵字 | 誤拒 |
|---|---|---|---|---|---|
| 1 | 269 | **0.102 s** | 36.0 | 80.7% | 16.1% |
| 3 | 338 | 0.165 s | 35.9 | 96.8% | 3.2% |
| 5 | 480 | 0.167 s | 36.0 | 96.8% | **0.0%** |
| 8 | 650 | 0.296 s | 35.1 | 96.8% | 0.0% |

prompt 從 269 → 650 tokens，**TTFT 惡化 2.9 倍，TPS 幾乎不動（−2.5%）** ——
prefill 是整段 prompt 的 GEMM，decode 是每 token 一次 GEMV。

**逐題分析比平均值更有資訊量**：top-1 錯 6 題，其中 5 題是誤拒（答案根本沒被撈進來）。
top-8 錯了一題 top-3 答對的 —— 問 CPU 時答「Ultra 200HX 系列」（來自特色頁文案），
正解是規格表的「Ultra 9 275HX」。**多出來的 context 引入了競爭性的模糊來源**。
所以 top-k 曲線是先上升、飽和、再劣化，預設取 5。

**(b) RAG vs no-RAG**（同模型，唯一差別是有沒有 context）

| 條件 | 關鍵字正確率 | negative 拒答率 | 數字接地 |
|---|---|---|---|
| **RAG (top-5)** | **96.8%** | **100%** | **100%** |
| no-RAG（同模型、無 context） | 29.0% | **0%** | **0%** |

AM6H 是 2025 年新品，模型預訓練資料不可能包含它：

```
Q: 這台筆電的電池容量是多少？
   RAG    : 電池容量是 99Wh [1]。
   no-RAG : ...電池容量為 95Wh。                      ← 編了一個很像的數字

Q: 顯示卡是哪一張？
   RAG    : NVIDIA® GeForce RTX™ 5090 Laptop GPU [3]。
   no-RAG : ...是 NVIDIA GeForce RTX 3080。           ← 差兩個世代

Q: 這台筆電支援 5G 行動網路嗎？
   RAG    : 提供的規格資料中沒有這項資訊。
   no-RAG : 不支援，因為它是一款筆電，通常不具備外接 5G 設備的插槽或連接埠。  ← 連理由都編了
```

**no-RAG 的 negative 拒答率是 0% —— 它從不說「不知道」。**
RAG 在這裡的價值不是讓模型變聰明，而是讓它**知道自己不知道**。

**(c) TPS 對照理論上限**

decode 是 memory-bandwidth bound：`M2 頻寬 100 GB/s ÷ 模型 1.11 GB ≈ 90 tok/s` 為理論值，
實測 61.7 tok/s（69%）。差距來自 KV cache 讀取、attention 與非權重的記憶體流量。
這個對照的用意是證明數字有物理依據，而不是報一個孤立的測量值。

**(d) 未完成**

| 項目 | 狀態 |
|---|---|
| Qwen3-4B 對照 | 未執行 |
| 量化掃描 Q4_K_M / Q5_K_M / Q8_0 | 未執行 |
| Kaggle T4 的 `nvidia-smi` VRAM 佐證 | 未執行 |

本機為 8 GB 統一記憶體：3B 的完整掃描會把 swap 推到 1.6 GB 並使機器明顯發熱，
1.7B 則全程 swap ≤ 884 MB、無散熱警告。再加測應移到 Kaggle T4，
**而不是在受污染的環境下硬跑出數字**。兩次評測皆在監督腳本下執行，
設三道自動中止防線（過熱降頻、swap > 1700 MB、可用記憶體 < 0.8 GB），均未觸發。

## 6. 資料取得

規格頁在 Akamai Bot Manager 後方。實測六種組合，**只有一格通過**：

| headers | 協定 | 結果 |
|---|---|---|
| Chrome 128 完整 header | HTTP/1.1 | 403 |
| **Chrome 128 完整 header** | **HTTP/2** | **200** |
| bot UA | HTTP/1.1 或 HTTP/2 | 403 |
| Chrome 28（HTTP/2 前的瀏覽器） | HTTP/1.1 | 403 |

**兩個條件是 AND。** 這不是「宣稱 Chrome 卻走 HTTP/1.1」的矛盾偵測 ——
HTTP/2 出現前的 UA 走 HTTP/1.1 並不矛盾，一樣被擋。頁面是 server-side rendered，
不需要 headless browser。`httpx` 預設走 HTTP/1.1，所以這個問題會在
「用 curl 驗證成功之後、改寫成 Python」時才浮現。

**一個會產生「看似正確的錯誤答案」的陷阱**：規格頁同時包含 AM6H 本身與
三個 SKU 的比較欄位（`div.spec-item-list`，51 個，只有值沒有標題）。
用「抓所有 `.spec-item-list`」的直覺寫法會把另外三台的規格混進語料 ——
而 **17 列裡有 16 列完全相同**，抽檢「電池」「CPU」「重量」全會通過，
只有顯示卡這一題會拿到三個矛盾答案，而那正是電競筆電的頭號規格。

解法是**鎖定結構**（只認 `li.spec-title` + `li.spec-desc` 的 `ul` 變體）
而非過濾內容，再加三道斷言：恰好 17 列、每列有鍵有值、不得出現 SKU 代號。
四種污染形狀都有測試覆蓋。

## 7. 已知限制

1. **只有兩個模型的對照**。1.7B 與 3B 已測並據此改了預設，但 4B 未測 ——
   只證明了「1.7B 足夠」，沒有證明「更大沒用」。量化維度完全未測，
   Q4_K_M 的選擇仍是引用社群共識。
2. **關鍵字命中會放過部分正確的答案**。實測案例：`rs06` 問「HDR 是哪一個等級」，
   答案給了 `DCIP-3 100%`（色域，非 HDR 等級）卻因含「HDR」而通過。
   自動指標可重現但覆蓋面窄。
3. **未採用 LLM-as-judge**。用同一顆 1.7B 評自己的答案不可靠，
   改用可驗證的自動指標並承認其侷限。
4. **SKU 查詢是最弱的一環**。加入 4 題 SKU 後 Recall@1 從 0.968 降到 0.914，
   靠 `top_k=5` 才答得對。跨 SKU 的深入查詢需要把型號做成檢索過濾條件。
5. **評測集為本人撰寫**，存在與系統設計同源的偏差風險。
6. **`hashing` fallback 無語意能力**，僅供零依賴驗證，使用時會印出警告。
7. **生成與檢索的評測集規模不同**（36 vs 40）。加入 SKU 題後重跑生成評測時，
   監督腳本因記憶體壓力自動中止。與其提高門檻硬跑出受污染的數字，
   選擇據實標明範圍。

## 8. 專案結構

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
  parse       HTML 解析 + 汙染防護              normalize 56 條原子事實
  chunk       Key-anchored 三層切分             embed     llama.cpp embedding
  index       BM25 / VectorIndex / RRF          retrieve  融合、boost、rescue、去重
  prompt      語言偵測、context packing          llm       streaming + 計時 + thinking 處理
  pipeline    build / ask 兩階段                bench     評測指標
  cli         7 個指令
tests/                                       37 個測試，不需模型
```

MIT

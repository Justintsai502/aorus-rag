"""Language detection, context packing and prompt assembly.

Written out by hand rather than pulled from a template library, because every
line here is doing a specific job for this dataset: forbidding unit conversion
(a 3B model will happily turn 99Wh into "about 26,000mAh"), forcing citations,
and making refusal an explicit, rewarded behaviour rather than an accident.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .retrieve import Hit

_CJK_RE = re.compile(r"[㐀-䶿一-鿿豈-﫿]")


def detect_language(text: str) -> str:
    """Return "zh" or "en" from the script mix.

    Mixed questions ("What's the 電池 capacity?") are common for this product
    category, so the rule is a ratio rather than "contains any CJK": a couple
    of Chinese nouns dropped into an English sentence should still get an
    English answer.
    """
    cjk = len(_CJK_RE.findall(text))
    letters = len(re.findall(r"[A-Za-z]", text))
    if cjk == 0:
        return "en"
    if letters == 0:
        return "zh"
    return "zh" if cjk * 2 >= letters else "en"


# These prompts were tuned against the 3B model rather than written once and
# hoped for. The first draft ("每個事實後面標上來源編號" + "簡潔直接") made the
# model answer a battery question with the single token "[1]" -- it merged the
# citation rule and the brevity rule into "just emit the citation". A second
# draft fixed that but refused cross-field questions whose answer *was* in the
# context, because "只根據參考資料回答" reads to a small model as "do not
# combine facts". Hence the explicit permission to combine, the explicit ban on
# citation-only replies, and the two worked examples: one answerable, one not.

SYSTEM_ZH = """你是 GIGABYTE AORUS MASTER 16 AM6H 的規格查詢助理。

作答規則：
- 依據「參考資料」回答。需要組合多筆資料才能回答時，就組合起來回答。
- 只有在參考資料完全沒有相關資訊時，才回答「提供的規格資料中沒有這項資訊」，且此時不要標來源編號。
- 規格數字逐字照抄，不得換算單位或四捨五入。
- 有答案時，先寫出完整答案，再於句尾加上來源編號；不可以只輸出編號。
- 用繁體中文（台灣用語），簡潔直接。

範例一 ——
參考資料：[1] 變壓器功率 / Adapter power: 330W
問題：變壓器幾瓦？
變壓器是 330W [1]。

範例二 ——
參考資料：[1] 電池容量 / Battery capacity: 99Wh
問題：這台有幾種顏色？
提供的規格資料中沒有這項資訊。"""

SYSTEM_EN = """You are a spec assistant for the GIGABYTE AORUS MASTER 16 AM6H laptop.

Rules:
- Answer from the Reference section. When a question needs several entries
  combined, combine them.
- Only when the Reference holds nothing relevant, answer "That information is
  not in the provided specifications" -- and add no citation in that case.
- Quote spec values verbatim. Never convert units or round.
- When you do have an answer, write the full answer first and put the source
  number at the end of the sentence. Never reply with a citation marker alone.
- Answer in English, concise and direct.

Example 1 --
Reference: [1] Adapter power: 330W
Question: How many watts is the adapter?
The adapter is 330W [1].

Example 2 --
Reference: [1] Battery capacity: 99Wh
Question: How many colours does it come in?
That information is not in the provided specifications."""

REFERENCE_HEADER = {"zh": "參考資料：", "en": "Reference:"}
QUESTION_HEADER = {"zh": "問題：", "en": "Question:"}


@dataclass
class BuiltPrompt:
    system: str
    user: str
    used_hits: list[Hit]
    context_chars: int


def render_context(hits: list[Hit]) -> str:
    return "\n".join(f"[{i + 1}] {h.chunk.context_line()}" for i, h in enumerate(hits))


def build_prompt(
    question: str,
    hits: list[Hit],
    lang: str | None = None,
    max_context_chars: int = 2400,
) -> BuiltPrompt:
    """Assemble the prompt, dropping the weakest hits if the budget is exceeded.

    Context is trimmed from the tail: hits arrive ranked, so the lowest-ranked
    chunk is the cheapest thing to lose.
    """
    lang = lang or detect_language(question)
    system = SYSTEM_ZH if lang == "zh" else SYSTEM_EN

    used = list(hits)
    while used and len(render_context(used)) > max_context_chars:
        used.pop()

    context = render_context(used)
    user = f"{REFERENCE_HEADER[lang]}\n{context}\n\n{QUESTION_HEADER[lang]}{question}"
    return BuiltPrompt(system=system, user=user, used_hits=used, context_chars=len(context))


def to_messages(prompt: BuiltPrompt) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": prompt.system},
        {"role": "user", "content": prompt.user},
    ]


# The no-RAG control condition for the benchmark: same model, same question,
# no retrieved context. The AM6H launched in 2025, so any correct answer here
# would have to come from pretraining -- which is exactly what we want to test.
NO_RAG_SYSTEM = {
    "zh": "你是筆記型電腦規格助理。請直接回答使用者關於 GIGABYTE AORUS MASTER 16 AM6H 的問題。",
    "en": "You are a laptop spec assistant. Answer the user's question about the "
    "GIGABYTE AORUS MASTER 16 AM6H directly.",
}


def build_no_rag_messages(question: str, lang: str | None = None) -> list[dict[str, str]]:
    lang = lang or detect_language(question)
    return [
        {"role": "system", "content": NO_RAG_SYSTEM[lang]},
        {"role": "user", "content": question},
    ]

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


SYSTEM_ZH = """你是 GIGABYTE AORUS MASTER 16 AM6H 筆記型電腦的規格查詢助理。

規則：
1. 只根據下方「參考資料」回答。參考資料沒有提到的，直接說「提供的規格資料中沒有這項資訊」，不要猜測、不要用一般常識補完。
2. 規格數字必須逐字照抄參考資料，不得改寫、換算單位或四捨五入。
3. 每個事實後面標上來源編號，例如 [1]、[2]。
4. 用繁體中文（台灣用語）回答，簡潔直接，不要重複問題。"""

SYSTEM_EN = """You are a spec assistant for the GIGABYTE AORUS MASTER 16 AM6H laptop.

Rules:
1. Answer only from the Reference section below. If it is not there, say
   "That information is not in the provided specifications" -- do not guess and
   do not fill gaps from general knowledge.
2. Quote spec values verbatim. Never convert units, round, or rephrase numbers.
3. Cite the source number after each fact, e.g. [1], [2].
4. Answer in English, concise and direct. Do not restate the question."""

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

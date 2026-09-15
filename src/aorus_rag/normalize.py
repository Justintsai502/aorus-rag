"""Derive atomic, directly-answerable facts from the spec rows.

Why this layer exists
---------------------
A spec row like ``Display`` holds ten lines covering panel type, resolution,
refresh rate, brightness, HDR certifications and more. If the whole row is
handed to a 3B model as context, the model has to pick the right number out of
a dozen candidates -- and small models pick wrong. Splitting each measurable
property into its own one-line chunk means "螢幕更新率多少?" retrieves a chunk
that contains exactly one number.

An observation that shapes the design: on the GIGABYTE site the spec *values*
are byte-identical between the Traditional Chinese and English pages -- only
the *keys* are translated. So the bilingual problem lives entirely on the key
side, which is why every fact carries a hand-written zh/en label pair.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .parse import SpecItem


@dataclass
class Fact:
    """One atomic, self-contained spec fact."""

    # A fact is a single value pulled from a spec row, e.g. battery.capacity = 99Wh.
    #   label_zh / label_en  hand-written bilingual labels; they become the chunk anchor
    #   source_key_*         the spec row the fact came from
    fact_id: str
    label_zh: str
    label_en: str
    value: str
    source_key_zh: str = ""
    source_key_en: str = ""
    # Structured data for port facts (side, count); copied into the chunk's meta.
    extra: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# Declarative rules: (fact_id, english spec key, label_zh, label_en, pattern,
# formatter). The pattern runs over the joined value text of that spec row.
# --------------------------------------------------------------------------

# Rules are declarative: a new fact type is one more tuple, not a new function.
Rule = tuple[str, str, str, str, str, str]

# Patterns run against the English page's values (identical to the zh values).
# In a template, \1 is the text captured by the pattern's first group.
RULES: list[Rule] = [
    # fact_id, spec key (EN), zh label, en label, regex, output template
    (
        "cpu.model",
        "CPU",
        "處理器型號",
        "CPU model",
        # The processor name up to its model number (e.g. ... 275HX); the non-greedy
        # [^()]*? cannot run past a parenthesis.
        r"(Intel[^()]*?Ultra\s+\d+\s+Processor\s+\w+)",
        r"\1",
    ),
    ("cpu.cache", "CPU", "處理器快取", "CPU cache", r"(\d+MB)\s+cache", r"\1"),
    (
        "cpu.max_frequency",
        "CPU",
        "處理器最高時脈",
        "CPU max frequency",
        r"up to ([\d.]+ GHz)",
        r"\1",
    ),
    ("cpu.cores", "CPU", "處理器核心數", "CPU cores", r"(\d+)\s+cores", r"\1 cores"),
    ("cpu.threads", "CPU", "處理器執行緒數", "CPU threads", r"(\d+)\s+threads", r"\1 threads"),
    (
        "gpu.model",
        "Video Graphics",
        "顯示晶片型號",
        "GPU model",
        # [^\n]*? keeps the whole match on a single line of the row.
        r"(NVIDIA[^\n]*?GeForce RTX[^\n]*?Laptop GPU)",
        r"\1",
    ),
    ("gpu.vram", "Video Graphics", "顯示記憶體", "GPU memory", r"(\d+GB\s+GDDR\d+)", r"\1"),
    (
        "gpu.max_graphics_power",
        "Video Graphics",
        "顯示晶片最大功耗",
        "GPU max graphics power",
        r"(\d+W)\s+Maximum Graphics Power",
        r"\1 (with Dynamic Boost)",
    ),
    (
        "gpu.boost_clock",
        "Video Graphics",
        "顯示晶片boost時脈",
        "GPU boost clock",
        r"AI Boost\s*:\s*(\d+ MHz)",
        r"\1 (AI Boost)",
    ),
    # The inch mark (") is rewritten as "inch" so the value reads naturally.
    ("display.size", "Display", "螢幕尺寸", "Display size", r'(\d+)"', r"\1 inch"),
    ("display.aspect_ratio", "Display", "螢幕比例", "Display aspect ratio", r"(\d+:\d+)\b", r"\1"),
    (
        "display.panel",
        "Display",
        "面板類型",
        "Display panel type",
        r"\b(OLED|IPS|Mini LED)\b",
        r"\1",
    ),
    (
        "display.resolution",
        "Display",
        "螢幕解析度",
        "Display resolution",
        # Resolution in parentheses, e.g. (2560x1600); both "x" and "×" are accepted.
        r"\((\d{3,4}\s*[x×]\s*\d{3,4})\)",
        r"\1",
    ),
    ("display.refresh_rate", "Display", "螢幕更新率", "Display refresh rate", r"(\d+)Hz", r"\1Hz"),
    (
        "display.response_time",
        "Display",
        "螢幕反應時間",
        "Display response time",
        r"(\d+)ms",
        r"\1ms",
    ),
    (
        "display.color_gamut",
        "Display",
        "色域覆蓋",
        "Display colour gamut",
        r"(DCI?P?-?3\s*\d+%)",
        r"\1",
    ),
    (
        "display.brightness",
        "Display",
        "螢幕亮度",
        "Display brightness",
        # Peak brightness specifically, marked "(peak)" on the page.
        r"(\d+)nits \(peak\)",
        r"\1 nits (peak)",
    ),
    (
        "display.contrast",
        "Display",
        "對比度",
        "Display contrast ratio",
        # Requiring thousands separators stops the 16:10 aspect ratio from being read as a
        # 16:1 contrast ratio (covered by a test).
        r"(\d{1,3}(?:,\d{3})+:1)",
        r"\1",
    ),
    (
        "memory.max_capacity",
        "System Memory",
        "最大記憶體容量",
        "Maximum system memory",
        # Maximum supported capacity ("Up to ..."), not the installed amount.
        r"Up to (\d+GB)",
        r"\1",
    ),
    ("memory.type", "System Memory", "記憶體型號", "Memory type", r"\b(DDR\d)\b", r"\1"),
    ("memory.speed", "System Memory", "記憶體時脈", "Memory speed", r"(\d{4}MHz)", r"\1"),
    (
        "memory.slots",
        "System Memory",
        "記憶體插槽",
        "Memory slots",
        r"(\d+)x SO-DIMM",
        r"\1 x SO-DIMM",
    ),
    ("storage.max_capacity", "Storage", "最大儲存容量", "Maximum storage", r"Up to (\d+TB)", r"\1"),
    (
        "storage.gen5_slot",
        "Storage",
        "PCIe Gen5 M.2 插槽",
        "PCIe Gen5 M.2 slot",
        r"(\d+)x PCIe Gen5 M\.2 slot",
        r"\1 x PCIe Gen5 M.2",
    ),
    (
        "storage.gen4_slot",
        "Storage",
        "PCIe Gen4 M.2 插槽",
        "PCIe Gen4 M.2 slot",
        r"(\d+)x PCIe Gen4x4 M\.2 slot",
        r"\1 x PCIe Gen4x4 M.2",
    ),
    (
        "storage.interface",
        "Storage",
        "儲存介面",
        "Storage interface",
        r"(PCIe NVMe\u2122? M\.2 SSD)",
        r"\1",
    ),
    (
        "keyboard.backlight",
        "Keyboard Type",
        "鍵盤背光",
        "Keyboard backlight",
        r"([\w-]+ RGB Backlit Keyboard)",
        r"\1",
    ),
    (
        "keyboard.key_travel",
        "Keyboard Type",
        "鍵程",
        "Key travel",
        r"Up to ([\d.]+mm) Key-travel",
        r"\1",
    ),
    ("audio.speakers", "Audio", "喇叭配置", "Speakers", r"(\d+x \d+W speakers)", r"\1"),
    ("wireless.wifi", "Communications", "無線網路", "Wi-Fi", r"(WIFI \d[^\n]*)", r"\1"),
    (
        "wireless.bluetooth",
        "Communications",
        "藍牙版本",
        "Bluetooth",
        r"Bluetooth (v[\d.]+)",
        r"\1",
    ),
    ("network.lan", "Communications", "有線網路", "Wired LAN", r"LAN:\s*(\w+)", r"\1"),
    ("webcam.resolution", "Webcam", "視訊鏡頭", "Webcam", r"(FHD \(1080p\) IR Webcam)", r"\1"),
    ("battery.capacity", "Battery", "電池容量", "Battery capacity", r"(\d+Wh)", r"\1"),
    ("battery.type", "Battery", "電池類型", "Battery type", r"\b(Li-ion)\b", r"\1"),
    ("adapter.power", "Adapter", "變壓器功率", "Adapter power", r"(\d+W)", r"\1"),
    (
        "dimensions.mm",
        "Dimensions (W x D x H)",
        "機身尺寸",
        "Dimensions",
        # Width x depth x height; the height allows "~" for a thickness range.
        r"([\d.]+ x [\d.]+ x [\d.~]+ mm)",
        r"\1",
    ),
    ("weight.kg", "Weight", "機身重量", "Weight", r"~?([\d.]+ kg)", r"\1"),
    # ^(.+)$ with re.MULTILINE takes the first line of the colour row.
    ("color.name", "Color", "機身顏色", "Colour", r"^(.+)$", r"\1"),
]


# --------------------------------------------------------------------------
# I/O ports get bespoke handling: the row is side-structured ("Left Side:" /
# "Right Side:") and "which side is Thunderbolt 5 on?" is a natural question
# that a flat text chunk answers badly.
# --------------------------------------------------------------------------

# _SIDE_RE matches section headers like "Left Side:"; _PORT_RE splits
# "2 x USB Type-C ..." into a count and a description.
_SIDE_RE = re.compile(r"^(Left|Right)\s+Side:", re.IGNORECASE)
_PORT_RE = re.compile(r"^(\d+)\s*x\s*(.+)$")

SIDE_ZH = {"Left": "左側", "Right": "右側"}


@dataclass
class Port:
    # One port line from the I/O row: which side it is on, how many, and what it is.
    side: str
    count: int
    description: str


def parse_ports(item: SpecItem) -> list[Port]:
    """Split the I/O row into side-tagged port entries."""
    ports: list[Port] = []
    # Walk the lines: a side header updates the current side, a port line is recorded
    # under it.
    side = "Unknown"
    for line in item.lines:
        m = _SIDE_RE.match(line)
        if m:
            side = m.group(1).capitalize()
            continue
        pm = _PORT_RE.match(line)
        if pm:
            ports.append(Port(side=side, count=int(pm.group(1)), description=pm.group(2).strip()))
    return ports


def _port_facts(item_zh: SpecItem, item_en: SpecItem) -> list[Fact]:
    # The I/O row yields three kinds of fact:
    #   io.port.N    one per port, with its side
    #   io.count.*   totals per port type (USB-C, USB-A, Thunderbolt, HDMI)
    #   io.side.*    which side the notable ports are on
    ports = parse_ports(item_en)
    facts: list[Fact] = []

    for i, port in enumerate(ports):
        side_zh = SIDE_ZH.get(port.side, port.side)
        facts.append(
            Fact(
                # The label carries the side in both languages, so a question about the ports on
                # the left side matches in either language.
                fact_id=f"io.port.{i}",
                label_zh=f"連接埠（{side_zh}）",
                label_en=f"I/O port ({port.side} side)",
                value=f"{port.count} x {port.description}",
                source_key_zh=item_zh.key,
                source_key_en=item_en.key,
                extra={"side": port.side, "count": port.count},
            )
        )

    # Aggregate counts -- "how many USB-C ports does it have?" should not
    # require the model to count list items itself.
    # ``pred`` tests a port description, e.g. lambda d: "Type-C" in d.
    def total(pred) -> int:
        return sum(p.count for p in ports if pred(p.description))

    # (fact_id, zh label, en label, count); types with a zero count are skipped below.
    aggregates = [
        (
            "io.count.usb_c",
            "Type-C 連接埠數量",
            "USB Type-C port count",
            total(lambda d: "Type-C" in d),
        ),
        (
            "io.count.usb_a",
            "Type-A 連接埠數量",
            "USB Type-A port count",
            total(lambda d: "Type-A" in d),
        ),
        (
            "io.count.thunderbolt",
            "Thunderbolt 連接埠數量",
            "Thunderbolt port count",
            total(lambda d: "Thunderbolt" in d),
        ),
        ("io.count.hdmi", "HDMI 連接埠數量", "HDMI port count", total(lambda d: "HDMI" in d)),
    ]
    for fact_id, zh, en, n in aggregates:
        if n:
            facts.append(
                Fact(
                    fact_id=fact_id,
                    label_zh=zh,
                    label_en=en,
                    value=str(n),
                    source_key_zh=item_zh.key,
                    source_key_en=item_en.key,
                )
            )

    # Side lookup for the marquee ports.
    # Only the first port whose description contains the keyword is used.
    for keyword, zh, en in [
        ("Thunderbolt™5", "Thunderbolt 5 位置", "Thunderbolt 5 location"),
        ("Thunderbolt™4", "Thunderbolt 4 位置", "Thunderbolt 4 location"),
        ("HDMI", "HDMI 位置", "HDMI location"),
        ("RJ-45", "RJ-45 網路孔位置", "RJ-45 location"),
        ("MicroSD", "MicroSD 讀卡機位置", "MicroSD reader location"),
    ]:
        match = next((p for p in ports if keyword in p.description), None)
        if match:
            facts.append(
                Fact(
                    # e.g. "Thunderbolt™5" -> io.side.thunderbolt5
                    fact_id=f"io.side.{keyword.lower().replace('™', '').replace('-', '')}",
                    label_zh=zh,
                    label_en=en,
                    value=f"{SIDE_ZH.get(match.side, match.side)} / {match.side} side",
                    source_key_zh=item_zh.key,
                    source_key_en=item_en.key,
                    extra={"side": match.side},
                )
            )
    return facts


# --------------------------------------------------------------------------


def extract_facts(zh_items: list[SpecItem], en_items: list[SpecItem]) -> list[Fact]:
    """Run every rule and return the facts that actually matched."""
    # Rows are paired by position, so both tables must have the same length.
    if len(zh_items) != len(en_items):
        raise ValueError("zh/en spec tables have different row counts")

    # Rules are keyed by English field name; carry the matching zh row alongside.
    by_en_key = {en.key: (zh, en) for zh, en in zip(zh_items, en_items)}
    facts: list[Fact] = []
    # Each fact_id is emitted at most once.
    seen: set[str] = set()

    for fact_id, key, label_zh, label_en, pattern, template in RULES:
        pair = by_en_key.get(key)
        if pair is None:
            continue
        zh_item, en_item = pair
        text = en_item.value
        # First match only; a rule that does not match is simply skipped.
        m = re.search(pattern, text, re.MULTILINE)
        if not m:
            continue
        # expand() substitutes the captured groups into the template.
        value = m.expand(template).strip()
        if not value or fact_id in seen:
            continue
        seen.add(fact_id)
        facts.append(
            Fact(
                fact_id=fact_id,
                label_zh=label_zh,
                label_en=label_en,
                value=value,
                source_key_zh=zh_item.key,
                source_key_en=en_item.key,
            )
        )

    # The I/O row is side-structured, so it gets its own extractor.
    io_pair = by_en_key.get("I/O Port")
    if io_pair:
        facts.extend(_port_facts(*io_pair))

    return facts

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from threading import Lock
import time
import unicodedata
from typing import Any

import pandas as pd
from tqdm import tqdm

from .csv_io import safe_to_csv
from .extract import ExtractSettings, extract_json_object
from .llm import ChatClient
from .measurement_manifest import count_rows, write_measurement_manifest
from .preprocess import (
    PreprocessSettings,
    is_ascii_letter,
    load_keywords,
    normalize_stock_code,
    parse_report_filename,
    read_text_file,
)
from .schemas import (
    KEYWORD_FEATURE_COLUMNS,
    MAIN_FINAL_COLUMNS,
    MAIN_MAPPED_COLUMNS,
    STAGE1_RELEVANCE_COLUMNS,
    STAGE2_INPUT_COLUMNS,
    TEXT_UNIT_AUDIT_COLUMNS,
    TEXT_UNIT_COLUMNS,
    require_columns,
)


DEFAULT_MIN_UNIT_CHARS = 500
DEFAULT_MAX_UNIT_CHARS = 1200
TARGET_CHAPTER_FALLBACK = "全文"
STAGE1_RELEVANCE_VALUES = {"related", "unrelated", "uncertain"}
STAGE1_STRING_FIELD_RE = r'"{field}"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"'


@dataclass(frozen=True)
class TextUnitSettings:
    min_chars: int = DEFAULT_MIN_UNIT_CHARS
    max_chars: int = DEFAULT_MAX_UNIT_CHARS


@dataclass(frozen=True)
class MainRegressionPaths:
    base_dir: Path
    year: str

    @property
    def stage_dir(self) -> Path:
        return self.base_dir / "stage"

    @property
    def results_dir(self) -> Path:
        return self.base_dir / "results"

    @property
    def final_dir(self) -> Path:
        return self.base_dir / "final"

    @property
    def logs_dir(self) -> Path:
        return self.base_dir / "logs"

    @property
    def text_units_path(self) -> Path:
        return self.stage_dir / f"01_text_units_{self.year}.csv"

    @property
    def keyword_features_path(self) -> Path:
        return self.stage_dir / f"02_keyword_features_{self.year}.csv"

    @property
    def text_unit_audit_path(self) -> Path:
        return self.stage_dir / f"01_text_units_{self.year}_audit.csv"

    @property
    def stage1_relevance_path(self) -> Path:
        return self.stage_dir / f"03_stage1_llm_relevance_{self.year}.csv"

    @property
    def stage2_input_path(self) -> Path:
        return self.stage_dir / f"04_stage2_input_{self.year}.csv"

    @property
    def stage2_result_path(self) -> Path:
        return self.results_dir / f"05_stage2_entity_result_{self.year}.csv"

    @property
    def mapped_result_path(self) -> Path:
        return self.results_dir / f"06_mapped_entity_result_{self.year}.csv"

    @property
    def final_output_path(self) -> Path:
        return self.final_dir / f"07_main_regression_firm_year_{self.year}.csv"

    @property
    def manifest_path(self) -> Path:
        return self.logs_dir / f"08_manifest_{self.year}.json"

    def stage1_raw_failure_log_path(self, provider: str) -> Path:
        return self.logs_dir / f"03_stage1_raw_failures_{self.year}_{provider}.jsonl"

    def stage2_log_path(self, provider: str) -> Path:
        return self.logs_dir / f"05_stage2_processed_{self.year}_{provider}.log"

    def ensure_dirs(self) -> None:
        for directory in (self.stage_dir, self.results_dir, self.final_dir, self.logs_dir):
            directory.mkdir(parents=True, exist_ok=True)


def default_main_regression_paths(
    project_root: Path,
    year: str,
    smoke: bool = False,
    base_dir: Path | None = None,
) -> MainRegressionPaths:
    data_root = project_root / "data" / ("measurement_smoke" if smoke else "measurement")
    resolved_base = base_dir or data_root / "main_regression"
    return MainRegressionPaths(base_dir=resolved_base.resolve(), year=str(year))


def settings_from_config(data: dict[str, Any]) -> TextUnitSettings:
    min_chars = int(data.get("text_unit_min_chars", DEFAULT_MIN_UNIT_CHARS))
    max_chars = int(data.get("text_unit_max_chars", DEFAULT_MAX_UNIT_CHARS))
    if max_chars < min_chars:
        raise ValueError("text_unit_max_chars must be greater than or equal to text_unit_min_chars")
    return TextUnitSettings(min_chars=min_chars, max_chars=max_chars)


CHAPTER_PREFIX_PATTERN = re.compile(r"^(第[一二三四五六七八九十百\d]+[章节])[ \t]*(.*?)$")
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[。！？；!?;])")
DOT_LEADER_PATTERN = re.compile(r"(?:\.{4,}|…{2,}|·{4,}|_{4,}|-{6,})")
PAGE_NUMBER_PATTERN = re.compile(r"\d{1,4}")
TRAILING_PAGE_PATTERN = re.compile(r"(?:\s|\.|…|·|-|_)*\d{1,4}\s*$")
FALSE_CHAPTER_TITLE_HINTS = (
    "详见",
    "描述",
    "请投资者",
    "注意",
    "有关内容",
    "相关内容",
    "披露",
    "本报告",
    "部分",
    "中关于",
)
SHORT_TABLE_LINE_MAX_CHARS = 6
TABLE_CELL_MAX_CHARS = 24
SHORT_TABLE_BLOCK_MIN_LINES = 4
MERGED_CJK_PHRASE_MAX_CHARS = 12
TABLE_CONTEXT_LINES = 2
AUDIT_PREVIEW_CHARS = 300
PROTECTED_ANCHOR_TERMS = (
    "标准",
    "標準",
    "准则",
    "準則",
    "认证",
    "認證",
    "证书",
    "證書",
    "注册",
    "备案",
    "注准",
    "准字",
    "登记",
    "登記",
    "许可证",
    "許可證",
    "许可",
    "許可",
    "资质",
    "資質",
    "规范",
    "規範",
    "规程",
    "規程",
    "办法",
    "辦法",
    "公告",
    "指引",
    "指南",
    "体系",
    "體系",
    "合规",
    "合規",
    "达标",
    "達標",
    "超标",
    "超標",
    "符合",
    "执行",
    "執行",
    "通过",
    "通過",
    "取得",
    "获批",
    "獲批",
    "起草",
    "制定",
    "参与",
    "參與",
    "法规",
    "法規",
    "规则",
    "規則",
    "条例",
    "條例",
    "守则",
    "守則",
    "准入",
    "準入",
    "一致性评价",
    "一致性評價",
    "议定书",
    "議定書",
    "合格",
    "ISO",
    "IEC",
    "GB/T",
    "GB",
    "GMP",
    "GSP",
    "HACCP",
    "IATF",
    "CE",
    "FDA",
    "UL",
    "RoHS",
    "REACH",
    "CCC",
    "3C",
    "CNAS",
    "CMA",
    "IFRS",
    "ASTM",
    "AEC",
    "PPAP",
    "HDMI",
    "USB",
    "MIPI",
    "NIST",
    "DALI",
    "Zigbee",
    "FCC",
)
PROTECTED_ANCHOR_PATTERNS = tuple(
    (term, re.compile(re.escape(term), re.IGNORECASE))
    for term in PROTECTED_ANCHOR_TERMS
)
PURE_TABLE_VALUE_PATTERN = re.compile(
    r"^(?:[¥￥$€])?[-+]?"
    r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
    r"(?:%|‰|元|万元|亿元|万|亿|美元|平方米|万平方米|吨|万吨|千克|人|人次|个|项|家|股)?$",
    re.IGNORECASE,
)
DATE_TABLE_LINE_PATTERN = re.compile(
    r"^(?:19|20)\d{2}(?:[-/.年](?:0?[1-9]|1[0-2])(?:[-/.月](?:0?[1-9]|[12]\d|3[01])日?)?)?$"
)


def normalize_unit_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def normalize_chapter_title(title: str) -> str:
    title = unicodedata.normalize("NFKC", title)
    title = re.sub(r"\s+", " ", title).strip()
    return title.strip("- ")


def compact_text_length(text: str) -> int:
    return len(re.sub(r"\s+", "", text))


def is_toc_or_dot_leader_line(text: str) -> bool:
    stripped = normalize_chapter_title(text)
    return bool(DOT_LEADER_PATTERN.search(stripped) and TRAILING_PAGE_PATTERN.search(stripped))


def is_probable_chapter_title(title: str) -> bool:
    normalized = normalize_chapter_title(title)
    compact = re.sub(r"\s+", "", normalized)
    if not compact:
        return False
    if is_toc_or_dot_leader_line(normalized):
        return False
    if compact_text_length(normalized) > 55:
        return False
    if re.search(r"[。；;，,：:]", normalized):
        return False
    if any(hint in normalized for hint in FALSE_CHAPTER_TITLE_HINTS):
        return False
    if "之" in normalized:
        return False
    if re.search(r"(管理层讨论与分析|重要事项|公司治理|环境和社会责任)[”\"']?[之中]", normalized):
        return False
    return True


def is_annual_report_header_line(line: str, year: str | None = None) -> bool:
    compact = re.sub(r"\s+", "", line)
    if "年度报告" not in compact:
        return False
    if year and str(year) not in compact and f"{int(year) - 1}" not in compact:
        return False
    return len(compact) <= 55


def is_company_header_line(line: str) -> bool:
    compact = re.sub(r"\s+", "", line)
    if len(compact) > 45:
        return False
    return compact.endswith(("股份有限公司", "集团股份有限公司", "有限公司"))


def is_page_number_line(line: str) -> bool:
    return bool(PAGE_NUMBER_PATTERN.fullmatch(line.strip()))


def is_near_report_header(lines: list[str], index: int, year: str | None = None) -> bool:
    start = max(0, index - 2)
    end = min(len(lines), index + 3)
    return any(
        offset != index and is_annual_report_header_line(lines[offset], year)
        for offset in range(start, end)
    )


def should_drop_report_line(lines: list[str], index: int, year: str | None = None) -> bool:
    line = lines[index].strip()
    if not line:
        return True
    if line in {"目", "录", "目录"}:
        return True
    if is_toc_or_dot_leader_line(line):
        return True
    if is_annual_report_header_line(line, year):
        return True
    # Standalone numeric lines have no semantic value in the extracted TXT and
    # are overwhelmingly PDF page numbers or isolated numeric table cells.
    if is_page_number_line(line):
        return True
    if is_company_header_line(line) and is_near_report_header(lines, index, year):
        return True
    return False


def is_short_table_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    length = compact_text_length(stripped)
    if length <= SHORT_TABLE_LINE_MAX_CHARS:
        return True
    return length <= 8 and bool(re.fullmatch(r"[\d,.\-+%/%()（）A-Za-z_]+", stripped))


def is_numeric_fragment(text: str) -> bool:
    return bool(re.fullmatch(r"[-+]?[\d,./%()（）]+", text))


def is_cjk_fragment(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def merge_short_line_block(lines: list[str]) -> str:
    phrases: list[str] = []
    current = ""
    current_kind = ""

    def flush_current() -> None:
        nonlocal current, current_kind
        if current:
            phrases.append(current)
        current = ""
        current_kind = ""

    for raw in lines:
        token = raw.strip()
        if not token:
            continue
        if is_numeric_fragment(token):
            if current_kind == "number":
                current += token
            else:
                flush_current()
                current = token
                current_kind = "number"
            continue
        if is_cjk_fragment(token):
            token_len = compact_text_length(token)
            current_len = compact_text_length(current)
            if current_kind == "cjk" and current_len + token_len <= MERGED_CJK_PHRASE_MAX_CHARS:
                current += token
            else:
                flush_current()
                current = token
                current_kind = "cjk"
            continue
        flush_current()
        phrases.append(token)

    flush_current()
    return " ".join(phrases)


def merge_dense_short_line_blocks(lines: list[str]) -> list[str]:
    merged: list[str] = []
    buffer: list[str] = []

    def flush_buffer() -> None:
        nonlocal buffer
        if len(buffer) >= SHORT_TABLE_BLOCK_MIN_LINES:
            merged.append(merge_short_line_block(buffer))
        else:
            merged.extend(buffer)
        buffer = []

    for line in lines:
        if is_short_table_line(line):
            buffer.append(line)
            continue
        flush_buffer()
        merged.append(line)
    flush_buffer()
    return merged


def has_protected_anchor(text: str) -> bool:
    """Return True when text contains evidence that table cleanup must preserve."""
    for term, pattern in PROTECTED_ANCHOR_PATTERNS:
        for match in pattern.finditer(text):
            start, end = match.span()
            if is_ascii_letter(term[0]) and start > 0 and is_ascii_letter(text[start - 1]):
                continue
            if is_ascii_letter(term[-1]) and end < len(text) and is_ascii_letter(text[end]):
                continue
            return True
    return False


def is_strong_table_noise_line(line: str) -> bool:
    compact = re.sub(r"\s+", "", line)
    if not compact or not re.search(r"\d", compact):
        return False
    if is_page_number_line(compact):
        return True
    if DATE_TABLE_LINE_PATTERN.fullmatch(compact):
        return True
    if PURE_TABLE_VALUE_PATTERN.fullmatch(compact):
        return True
    return bool(re.fullmatch(r"[-+]?\d+(?:[./:]\d+){1,3}%?", compact))


def table_line_score(line: str) -> float:
    stripped = line.strip()
    if not stripped:
        return 0.0
    compact = re.sub(r"\s+", "", stripped)
    if is_strong_table_noise_line(stripped):
        return 1.0
    if is_short_table_line(stripped):
        return 0.8
    if compact_text_length(stripped) <= TABLE_CELL_MAX_CHARS and not re.search(
        r"[。！？!?；;]", stripped
    ):
        return 0.6

    digit_ratio = sum(char.isdigit() for char in compact) / max(1, len(compact))
    if digit_ratio >= 0.5 and not re.search(r"[。！？!?；;]", stripped):
        return 0.9
    numeric_tokens = re.findall(r"[-+]?\d[\d,.]*(?:%|‰)?", stripped)
    if len(numeric_tokens) >= 2 and digit_ratio >= 0.25 and not re.search(r"[。！？!?]", stripped):
        return 0.65
    return 0.0


def find_table_like_blocks(lines: list[str]) -> list[tuple[int, int]]:
    """Find half-open runs of table-like lines without crossing blank paragraphs."""
    blocks: list[tuple[int, int]] = []
    start: int | None = None

    def flush(end: int) -> None:
        nonlocal start
        if start is None:
            return
        scores = [table_line_score(lines[index]) for index in range(start, end)]
        if len(scores) >= SHORT_TABLE_BLOCK_MIN_LINES or (
            len(scores) >= 3 and sum(scores) / len(scores) >= 0.85
        ):
            blocks.append((start, end))
        start = None

    for index, line in enumerate(lines):
        if line.strip() and table_line_score(line) >= 0.6:
            if start is None:
                start = index
            continue
        flush(index)
    flush(len(lines))
    return blocks


def protected_table_context(lines: list[str], blocks: list[tuple[int, int]]) -> set[int]:
    protected: set[int] = set()

    def add_context(index: int) -> None:
        protected.add(index)
        for direction in (-1, 1):
            for offset in range(1, TABLE_CONTEXT_LINES + 1):
                neighbor = index + (direction * offset)
                if neighbor < 0 or neighbor >= len(lines) or not lines[neighbor].strip():
                    break
                protected.add(neighbor)

    for index, line in enumerate(lines):
        if has_protected_anchor(line):
            add_context(index)

    # PDF extraction can split one protected phrase over several tiny cells
    # (for example, `认` + `证`).  Inspect short windows inside table blocks so
    # those anchors and their local context survive as well.
    for start, end in blocks:
        joined = "".join(re.sub(r"\s+", "", lines[index]) for index in range(start, end))
        if not has_protected_anchor(joined):
            continue
        if end - start <= 12:
            protected.update(range(start, end))
            continue
        for window_start in range(start, end):
            for window_end in range(window_start + 1, min(end, window_start + 6) + 1):
                window = "".join(lines[index].strip() for index in range(window_start, window_end))
                if has_protected_anchor(window):
                    for index in range(window_start, window_end):
                        add_context(index)
                    break

    return protected


def calculate_table_noise_score(text: str) -> float:
    lines = [line for line in normalize_unit_text(text).split("\n") if line.strip()]
    if not lines:
        return 0.0
    return round(sum(table_line_score(line) for line in lines) / len(lines), 4)


def has_protected_anchor_in_text(text: str) -> bool:
    """Detect direct anchors and anchors split across adjacent table cells."""
    normalized = normalize_unit_text(text)
    if not normalized:
        return False
    lines = normalized.split("\n")
    if any(has_protected_anchor(line) for line in lines):
        return True
    blocks = find_table_like_blocks(lines)
    return bool(protected_table_context(lines, blocks))


def compress_table_noise(text: str) -> str:
    """Remove line-level table noise while preserving anchored evidence and context."""
    normalized = normalize_unit_text(text)
    if not normalized:
        return ""
    lines = normalized.split("\n")
    blocks = find_table_like_blocks(lines)
    table_indices = {
        index
        for start, end in blocks
        for index in range(start, end)
    }
    protected = protected_table_context(lines, blocks)

    filtered: list[str] = []
    for index, line in enumerate(lines):
        if not line.strip():
            filtered.append("")
            continue
        removable = is_strong_table_noise_line(line) or index in table_indices
        if removable and index not in protected:
            continue
        filtered.append(line)

    # Reconstruct short protected phrases after noisy cells have been removed.
    merged = merge_dense_short_line_blocks(filtered)
    result = "\n".join(merged)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def clean_chapter_text_for_units(text: str, year: str | None = None) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    cleaned = [
        "" if not line else line
        for index, line in enumerate(lines)
        if not line or not should_drop_report_line(lines, index, year)
    ]
    return compress_table_noise("\n".join(cleaned))


def match_chapter_title_at(lines: list[str], index: int) -> tuple[str, int] | None:
    """Match a one-line or two-line chapter title at a source-line boundary."""
    line = lines[index].strip()
    if not line:
        return None
    match = CHAPTER_PREFIX_PATTERN.fullmatch(line)
    if not match:
        return None

    prefix, remainder = match.groups()
    if remainder.strip():
        return normalize_chapter_title(f"{prefix} {remainder}"), 1

    # Some extracted reports put `第三节` and its title on adjacent lines.
    # Only look a short distance ahead and never absorb another chapter marker.
    next_index = index + 1
    while next_index < len(lines) and not lines[next_index].strip() and next_index <= index + 2:
        next_index += 1
    if next_index >= len(lines) or next_index > index + 2:
        return None
    continuation = lines[next_index].strip()
    if CHAPTER_PREFIX_PATTERN.fullmatch(continuation):
        return None
    return normalize_chapter_title(f"{prefix} {continuation}"), next_index - index + 1


def extract_target_chapter_blocks(
    raw_text: str,
    settings: PreprocessSettings,
) -> list[tuple[str, str]]:
    """Extract target chapters without turning prose references into boundaries."""
    normalized = unicodedata.normalize("NFKC", raw_text)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")

    sections: list[tuple[str, str]] = []
    current_title = ""
    current_lines: list[str] = []
    found_boundary = False

    def flush_current() -> None:
        nonlocal current_lines
        if current_title:
            sections.append((current_title, "\n".join(current_lines)))
        current_lines = []

    index = 0
    while index < len(lines):
        title_match = match_chapter_title_at(lines, index)
        if title_match and is_probable_chapter_title(title_match[0]):
            candidate_title, consumed = title_match
            found_boundary = True
            if candidate_title == current_title:
                # Repeated chapter titles at every PDF page are running headers,
                # not new semantic sections.
                index += consumed
                continue
            flush_current()
            current_title = candidate_title
            index += consumed
            continue

        # A line that merely contains a chapter reference is ordinary content.
        if current_title:
            current_lines.append(lines[index])
        index += 1

    flush_current()
    if not found_boundary:
        return [(TARGET_CHAPTER_FALLBACK, raw_text)]

    selected = [
        (title, text)
        for title, text in sections
        if any(keyword in title for keyword in settings.target_chapter_keywords)
    ]
    filtered_len = sum(len(text.strip()) for _, text in selected)
    if filtered_len < settings.min_chapter_chars:
        return [(TARGET_CHAPTER_FALLBACK, raw_text)]
    return selected or [(TARGET_CHAPTER_FALLBACK, raw_text)]


def split_paragraphs(text: str) -> list[str]:
    text = normalize_unit_text(text)
    if not text:
        return []

    lines = text.split("\n")
    paragraphs: list[str] = []
    current: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped:
            current.append(stripped)
            continue
        if current:
            paragraphs.append("\n".join(current).strip())
            current = []
    if current:
        paragraphs.append("\n".join(current).strip())

    non_empty_lines = [line.strip() for line in lines if line.strip()]
    if len(paragraphs) <= 1 and len(non_empty_lines) > 1:
        paragraphs = non_empty_lines
    return [paragraph for paragraph in paragraphs if paragraph]


def hard_split_text(text: str, max_chars: int) -> list[str]:
    return [text[index : index + max_chars].strip() for index in range(0, len(text), max_chars) if text[index : index + max_chars].strip()]


def split_long_piece(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    sentences = [sentence.strip() for sentence in SENTENCE_SPLIT_PATTERN.split(text) if sentence.strip()]
    if not sentences:
        return hard_split_text(text, max_chars)

    pieces: list[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > max_chars:
            if current:
                pieces.append(current.strip())
                current = ""
            pieces.extend(hard_split_text(sentence, max_chars))
            continue

        candidate = f"{current}{sentence}" if current else sentence
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                pieces.append(current.strip())
            current = sentence
    if current:
        pieces.append(current.strip())
    return pieces


def merge_short_pieces(pieces: list[str], settings: TextUnitSettings) -> list[str]:
    units: list[str] = []
    current = ""
    soft_overflow = settings.max_chars + max(100, settings.min_chars // 2)

    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        if not current:
            current = piece
            continue

        candidate = f"{current}\n{piece}"
        if len(candidate) <= settings.max_chars:
            current = candidate
        elif len(current) < settings.min_chars and len(candidate) <= soft_overflow:
            current = candidate
        else:
            units.append(current.strip())
            current = piece

    if current:
        if units and len(current) < settings.min_chars:
            candidate = f"{units[-1]}\n{current}"
            if len(candidate) <= soft_overflow:
                units[-1] = candidate.strip()
            else:
                units.append(current.strip())
        else:
            units.append(current.strip())
    return units


def split_text_units(text: str, settings: TextUnitSettings) -> list[str]:
    paragraphs = split_paragraphs(text)
    pieces: list[str] = []
    for paragraph in paragraphs:
        pieces.extend(split_long_piece(paragraph, settings.max_chars))
    return merge_short_pieces(pieces, settings)


def run_build_text_units(
    input_dir: Path,
    output_csv: Path,
    preprocess_settings: PreprocessSettings,
    unit_settings: TextUnitSettings,
    limit: int | None = None,
    input_files: list[Path] | None = None,
) -> pd.DataFrame:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    files = (
        sorted(input_files)
        if input_files is not None
        else sorted(path for path in input_dir.iterdir() if path.suffix.lower() == ".txt")
    )
    invalid_files = [path for path in files if not path.is_file() or path.suffix.lower() != ".txt"]
    if invalid_files:
        raise ValueError(f"input_files contains missing or non-TXT paths: {invalid_files[:5]}")
    if limit is not None:
        files = files[:limit]

    records: list[dict[str, object]] = []
    for file_path in tqdm(files, desc="build-text-units"):
        meta = parse_report_filename(file_path.name)
        if not meta:
            continue
        raw_text = read_text_file(file_path)
        chapter_blocks = extract_target_chapter_blocks(raw_text, preprocess_settings)

        unit_order = 1
        for chapter_title, chapter_text in chapter_blocks:
            clean_text = clean_chapter_text_for_units(chapter_text, meta["year"])
            for text in split_text_units(clean_text, unit_settings):
                records.append(
                    {
                        "text_unit_id": f"{meta['stock_code']}_{meta['year']}_{unit_order:05d}",
                        "stock_code": meta["stock_code"],
                        "company_name": meta["company_name"],
                        "year": meta["year"],
                        "source_file": file_path.name,
                        "chapter_title": chapter_title,
                        "unit_order": unit_order,
                        "text": text,
                    }
                )
                unit_order += 1

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records, columns=TEXT_UNIT_COLUMNS)
    safe_to_csv(df, output_csv)
    return df


def normalize_stock_code_column(df: pd.DataFrame) -> pd.DataFrame:
    if "stock_code" in df.columns:
        df = df.copy()
        df["stock_code"] = df["stock_code"].map(normalize_stock_code)
    return df


def audit_preview(text: str) -> str:
    preview = text.replace("\r\n", "\n").replace("\r", "\n")
    preview = preview.replace("\n", r"\n")
    return preview[:AUDIT_PREVIEW_CHARS]


def run_text_unit_noise_audit(
    input_csv: Path,
    output_csv: Path,
    limit: int | None = None,
    chunksize: int = 5000,
) -> int:
    """Write a dry-run cleanup audit without modifying the input text-unit CSV."""
    if input_csv.resolve() == output_csv.resolve():
        raise ValueError("Audit output must not overwrite the input text-unit CSV")
    if chunksize < 1:
        raise ValueError("chunksize must be positive")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    temp_output = output_csv.with_name(f".{output_csv.name}.tmp")
    temp_output.unlink(missing_ok=True)
    processed = 0
    wrote_header = False

    try:
        reader = pd.read_csv(
            input_csv,
            dtype={"stock_code": str, "year": str, "text_unit_id": str},
            chunksize=chunksize,
        )
        for chunk in tqdm(reader, desc="audit-text-units"):
            require_columns(chunk, TEXT_UNIT_COLUMNS, str(input_csv))
            if limit is not None:
                remaining = limit - processed
                if remaining <= 0:
                    break
                chunk = chunk.head(remaining)

            records: list[dict[str, object]] = []
            for _, row in chunk.iterrows():
                raw_text = "" if pd.isna(row["text"]) else str(row["text"])
                year = None if pd.isna(row["year"]) else str(row["year"])
                filtered_text = clean_chapter_text_for_units(raw_text, year)
                raw_len = len(raw_text)
                filtered_len = len(filtered_text)
                removed_ratio = (raw_len - filtered_len) / raw_len if raw_len else 0.0
                records.append(
                    {
                        "text_unit_id": str(row["text_unit_id"]),
                        "company_name": str(row["company_name"]),
                        "unit_order": row["unit_order"],
                        "table_noise_score": calculate_table_noise_score(raw_text),
                        "has_protected_anchor": has_protected_anchor_in_text(raw_text),
                        "raw_text_len": raw_len,
                        "filtered_text_len": filtered_len,
                        "removed_ratio": round(max(0.0, min(1.0, removed_ratio)), 4),
                        "raw_preview": audit_preview(raw_text),
                        "filtered_preview": audit_preview(filtered_text),
                    }
                )

            audit_chunk = pd.DataFrame(records, columns=TEXT_UNIT_AUDIT_COLUMNS)
            safe_to_csv(
                audit_chunk,
                temp_output,
                mode="a" if wrote_header else "w",
                header=not wrote_header,
            )
            wrote_header = True
            processed += len(audit_chunk)
            if limit is not None and processed >= limit:
                break

        if not wrote_header:
            safe_to_csv(pd.DataFrame(columns=TEXT_UNIT_AUDIT_COLUMNS), temp_output)
        temp_output.replace(output_csv)
    except Exception:
        temp_output.unlink(missing_ok=True)
        raise
    return processed


def match_keyword_terms(text: str, keywords: list[str]) -> list[str]:
    matched_terms: list[str] = []
    used_spans: list[tuple[int, int]] = []
    text_len = len(text)

    for keyword in keywords:
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        for match in pattern.finditer(text):
            start, end = match.span()
            matched = match.group()
            if not matched:
                continue
            if is_ascii_letter(matched[0]) and start > 0 and is_ascii_letter(text[start - 1]):
                continue
            if is_ascii_letter(matched[-1]) and end < text_len and is_ascii_letter(text[end]):
                continue
            if any(start < used_end and end > used_start for used_start, used_end in used_spans):
                continue
            matched_terms.append(keyword)
            used_spans.append((start, end))
            break

    return matched_terms


def run_keyword_features(text_units_csv: Path, keyword_file: Path, output_csv: Path) -> pd.DataFrame:
    text_units = pd.read_csv(text_units_csv, dtype={"stock_code": str, "year": str, "text_unit_id": str})
    text_units = normalize_stock_code_column(text_units)
    require_columns(text_units, TEXT_UNIT_COLUMNS, str(text_units_csv))

    keywords = load_keywords(keyword_file)
    records: list[dict[str, object]] = []
    for _, row in tqdm(text_units.iterrows(), total=len(text_units), desc="keyword-features"):
        matched_terms = match_keyword_terms(str(row["text"]), keywords)
        records.append(
            {
                "text_unit_id": str(row["text_unit_id"]),
                "keyword_candidate": bool(matched_terms),
                "matched_terms": ";".join(matched_terms),
            }
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records, columns=KEYWORD_FEATURE_COLUMNS)
    safe_to_csv(df, output_csv)
    return df


def build_stage1_user_content(row: pd.Series) -> str:
    return (
        "请只判断以下单个 text_unit 是否可能涉及标准、认证、管理体系、市场准入或标准符合性。\n"
        f"text_unit_id: {row['text_unit_id']}\n"
        "text:\n"
        "---\n"
        f"{row['text']}\n"
        "---"
    )


def normalize_stage1_result(data: Any, text_unit_id: str) -> dict[str, object]:
    if not isinstance(data, dict):
        raise ValueError("Stage1 response is not a JSON object")

    relevance = str(data.get("relevance") or "uncertain").strip().lower()
    if relevance not in STAGE1_RELEVANCE_VALUES:
        relevance = "uncertain"

    try:
        confidence_score = float(data.get("confidence_score", 0.0))
    except (TypeError, ValueError):
        confidence_score = 0.0
    confidence_score = max(0.0, min(1.0, confidence_score))

    return {
        "text_unit_id": str(data.get("text_unit_id") or text_unit_id),
        "relevance": relevance,
        "confidence_score": confidence_score,
        "reason": str(data.get("reason") or "").strip(),
        "stage1_status": "OK",
        "stage1_error": "",
    }


def decode_json_string_fragment(value: str) -> str:
    try:
        return str(json.loads(f'"{value}"'))
    except json.JSONDecodeError:
        return value


def extract_stage1_string_field(raw: str, field: str) -> str | None:
    pattern = STAGE1_STRING_FIELD_RE.format(field=re.escape(field))
    match = re.search(pattern, raw, flags=re.DOTALL)
    if not match:
        return None
    return decode_json_string_fragment(match.group(1)).strip()


def extract_stage1_number_field(raw: str, field: str) -> float | None:
    pattern = rf'"{re.escape(field)}"\s*:\s*(-?\d+(?:\.\d+)?)'
    match = re.search(pattern, raw)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def extract_stage1_reason_fragment(raw: str) -> str:
    match = re.search(r'"reason"\s*:\s*"', raw, flags=re.DOTALL)
    if not match:
        return ""

    tail = raw[match.end() :]
    end = tail.rfind("}")
    if end >= 0:
        tail = tail[:end]
    tail = tail.strip()
    if tail.endswith('"'):
        tail = tail[:-1]
    tail = tail.replace("\\n", " ").replace("\\t", " ").replace("\\r", " ")
    tail = re.sub(r"\s+", " ", tail).strip()
    return tail


def repair_stage1_json_object(raw: str, text_unit_id: str) -> dict[str, object]:
    """Recover Stage1's fixed fields when only free-text reason breaks JSON."""
    relevance = extract_stage1_string_field(raw, "relevance")
    confidence_score = extract_stage1_number_field(raw, "confidence_score")
    if relevance is None:
        raise ValueError("Cannot repair Stage1 response without relevance")

    return {
        "text_unit_id": extract_stage1_string_field(raw, "text_unit_id") or text_unit_id,
        "relevance": relevance or "uncertain",
        "confidence_score": 0.0 if confidence_score is None else confidence_score,
        "reason": extract_stage1_reason_fragment(raw),
    }


def stage1_keyword_skip_result(text_unit_id: str) -> dict[str, object]:
    return {
        "text_unit_id": text_unit_id,
        "relevance": "",
        "confidence_score": "",
        "reason": "keyword_candidate routed directly to stage2",
        "stage1_status": "SKIPPED_KEYWORD",
        "stage1_error": "",
    }


def truncate_raw_response(raw: str, max_chars: int = 4000) -> str:
    if len(raw) <= max_chars:
        return raw
    return raw[:max_chars] + f"...[truncated {len(raw) - max_chars} chars]"


def append_stage1_raw_failure(
    log_file: Path | None,
    record: dict[str, object],
    lock: Any | None = None,
) -> None:
    if log_file is None:
        return

    line = json.dumps(record, ensure_ascii=False)
    if lock is None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("a", encoding="utf-8") as file:
            file.write(line + "\n")
        return

    with lock:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("a", encoding="utf-8") as file:
            file.write(line + "\n")


def stage1_parse_failure_record(
    text_unit_id: str,
    attempt: int,
    raw: str,
    error: Exception,
) -> dict[str, object]:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": "stage1",
        "text_unit_id": text_unit_id,
        "attempt": attempt,
        "error_type": type(error).__name__,
        "error": str(error),
        "raw_response": truncate_raw_response(raw),
    }


def call_stage1_with_retry(
    client: ChatClient,
    system_prompt: str,
    row: pd.Series,
    settings: ExtractSettings,
    raw_failure_log: Path | None = None,
    raw_log_lock: Any | None = None,
) -> dict[str, object]:
    user_content = build_stage1_user_content(row)
    text_unit_id = str(row["text_unit_id"])
    last_error: Exception | None = None
    json_parse_failures = 0
    max_json_parse_attempts = min(settings.max_retries, 2)

    for attempt in range(1, settings.max_retries + 1):
        try:
            raw = client.complete_json(system_prompt, user_content)
        except Exception as exc:
            last_error = exc
            if attempt >= settings.max_retries:
                break
            delay = min(settings.retry_max_seconds, settings.retry_min_seconds * (2 ** (attempt - 1)))
            time.sleep(delay)
            continue

        try:
            data = extract_json_object(raw)
            return normalize_stage1_result(data, text_unit_id)
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            json_parse_failures += 1
            append_stage1_raw_failure(
                raw_failure_log,
                stage1_parse_failure_record(text_unit_id, json_parse_failures, raw, exc),
                raw_log_lock,
            )
            try:
                repaired = repair_stage1_json_object(raw, text_unit_id)
                return normalize_stage1_result(repaired, text_unit_id)
            except ValueError:
                pass
            if json_parse_failures >= max_json_parse_attempts:
                log_hint = f"; raw_response_log={raw_failure_log}" if raw_failure_log else ""
                raise RuntimeError(
                    f"Stage1 JSON parse failed after {json_parse_failures} attempts: {exc}{log_hint}"
                ) from exc
            continue
        except Exception as exc:
            last_error = exc
            if attempt >= settings.max_retries:
                break
            delay = min(settings.retry_max_seconds, settings.retry_min_seconds * (2 ** (attempt - 1)))
            time.sleep(delay)
    raise RuntimeError(f"Stage1 LLM call failed after {settings.max_retries} attempts: {last_error}") from last_error


def process_stage1_row(
    row: pd.Series,
    client: ChatClient,
    system_prompt: str,
    settings: ExtractSettings,
    raw_failure_log: Path | None = None,
    raw_log_lock: Any | None = None,
) -> dict[str, object]:
    text_unit_id = str(row["text_unit_id"])
    try:
        result = call_stage1_with_retry(client, system_prompt, row, settings, raw_failure_log, raw_log_lock)
        result["text_unit_id"] = text_unit_id
        return result
    except Exception as exc:
        return {
            "text_unit_id": text_unit_id,
            "relevance": "",
            "confidence_score": "",
            "reason": "",
            "stage1_status": "ERROR",
            "stage1_error": str(exc),
        }


def run_stage1_screening(
    text_units_csv: Path,
    output_csv: Path,
    client: ChatClient,
    system_prompt: str,
    settings: ExtractSettings,
    limit: int | None = None,
    keyword_features_csv: Path | None = None,
    raw_failure_log: Path | None = None,
) -> pd.DataFrame:
    text_units = pd.read_csv(text_units_csv, dtype={"stock_code": str, "year": str, "text_unit_id": str})
    require_columns(text_units, TEXT_UNIT_COLUMNS, str(text_units_csv))
    if limit is not None:
        text_units = text_units.head(limit).copy()

    order = {str(row["text_unit_id"]): index for index, row in text_units.iterrows()}
    records: list[dict[str, object]] = []
    keyword_ids: set[str] = set()
    if keyword_features_csv is not None and keyword_features_csv.exists():
        keyword_features = pd.read_csv(keyword_features_csv, dtype={"text_unit_id": str})
        require_columns(keyword_features, KEYWORD_FEATURE_COLUMNS, str(keyword_features_csv))
        keyword_ids = set(
            keyword_features.loc[
                keyword_features["keyword_candidate"].map(parse_bool).fillna(False),
                "text_unit_id",
            ].astype(str)
        )

    if keyword_ids:
        for _, row in text_units[text_units["text_unit_id"].astype(str).isin(keyword_ids)].iterrows():
            records.append(stage1_keyword_skip_result(str(row["text_unit_id"])))

    work_units = text_units[~text_units["text_unit_id"].astype(str).isin(keyword_ids)].copy()
    raw_log_lock = Lock() if raw_failure_log is not None else None
    if raw_failure_log is not None and raw_failure_log.exists():
        raw_failure_log.unlink()

    if settings.workers <= 1:
        for _, row in tqdm(work_units.iterrows(), total=len(work_units), desc="stage1-screen"):
            records.append(process_stage1_row(row, client, system_prompt, settings, raw_failure_log, raw_log_lock))
    else:
        with ThreadPoolExecutor(max_workers=settings.workers) as executor:
            futures = [
                executor.submit(process_stage1_row, row, client, system_prompt, settings, raw_failure_log, raw_log_lock)
                for _, row in work_units.iterrows()
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc="stage1-screen"):
                records.append(future.result())

    df = pd.DataFrame(records, columns=STAGE1_RELEVANCE_COLUMNS)
    if not df.empty:
        df = df.assign(_order=df["text_unit_id"].map(order)).sort_values("_order").drop(columns="_order")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    safe_to_csv(df, output_csv)
    return df


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def build_route_reason(keyword_candidate: bool, relevance: str) -> str:
    reasons: list[str] = []
    if keyword_candidate:
        reasons.append("keyword")
    if relevance in {"related", "uncertain"}:
        reasons.append(f"stage1_{relevance}")
    return "+".join(reasons)


def run_route_main(
    text_units_csv: Path,
    keyword_features_csv: Path,
    stage1_relevance_csv: Path,
    output_csv: Path,
    limit: int | None = None,
) -> pd.DataFrame:
    text_units = pd.read_csv(text_units_csv, dtype={"stock_code": str, "year": str, "text_unit_id": str})
    text_units = normalize_stock_code_column(text_units)
    keyword_features = pd.read_csv(keyword_features_csv, dtype={"text_unit_id": str})
    stage1 = pd.read_csv(stage1_relevance_csv, dtype={"text_unit_id": str})
    require_columns(text_units, TEXT_UNIT_COLUMNS, str(text_units_csv))
    require_columns(keyword_features, KEYWORD_FEATURE_COLUMNS, str(keyword_features_csv))
    require_columns(stage1, STAGE1_RELEVANCE_COLUMNS, str(stage1_relevance_csv))

    merged = text_units.merge(keyword_features, on="text_unit_id", how="left")
    merged = merged.merge(stage1, on="text_unit_id", how="left")
    merged["keyword_candidate"] = merged["keyword_candidate"].map(parse_bool).fillna(False)
    merged["matched_terms"] = merged["matched_terms"].fillna("")
    merged["relevance"] = merged["relevance"].fillna("").astype(str).str.lower()
    merged["confidence_score"] = merged["confidence_score"].fillna("")
    merged["route_reason"] = [
        build_route_reason(bool(keyword_candidate), str(relevance))
        for keyword_candidate, relevance in zip(merged["keyword_candidate"], merged["relevance"])
    ]

    routed = merged[
        merged["keyword_candidate"] | merged["relevance"].isin(["related", "uncertain"])
    ].copy()
    if limit is not None:
        routed = routed.head(limit).copy()
    routed = routed[STAGE2_INPUT_COLUMNS]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    safe_to_csv(routed, output_csv)
    return routed


def run_aggregate_main(
    text_units_csv: Path,
    mapped_csv: Path,
    output_csv: Path,
) -> pd.DataFrame:
    text_units = pd.read_csv(text_units_csv, dtype={"stock_code": str, "year": str, "text_unit_id": str})
    mapped = pd.read_csv(mapped_csv, dtype={"stock_code": str, "year": str, "text_unit_id": str})
    text_units = normalize_stock_code_column(text_units)
    mapped = normalize_stock_code_column(mapped)
    require_columns(text_units, TEXT_UNIT_COLUMNS, str(text_units_csv))
    require_columns(mapped, MAIN_MAPPED_COLUMNS, str(mapped_csv))

    universe = text_units[["stock_code", "company_name", "year"]].drop_duplicates().copy()
    if universe.empty:
        final = pd.DataFrame(columns=MAIN_FINAL_COLUMNS)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        safe_to_csv(final, output_csv)
        return final

    mapped = mapped.copy()
    mapped["_output"] = pd.to_numeric(mapped["output"], errors="coerce").fillna(0).astype(int)
    dummy = (
        mapped.groupby(["stock_code", "year"], as_index=False)["_output"].max()
        if not mapped.empty
        else pd.DataFrame(columns=["stock_code", "year", "_output"])
    )
    dummy = dummy.rename(columns={"_output": "InternationalStandardDummy"})

    adopted = mapped[(mapped["_output"] == 1) & (mapped["status"].astype(str).str.upper() == "ADOPTED")].copy()
    adopted_counts = (
        adopted.dropna(subset=["entity"])
        .groupby(["stock_code", "year"])["entity"]
        .nunique()
        .reset_index(name="AdoptedEntityCount")
        if not adopted.empty
        else pd.DataFrame(columns=["stock_code", "year", "AdoptedEntityCount"])
    )

    final = universe.merge(dummy, on=["stock_code", "year"], how="left")
    final = final.merge(adopted_counts, on=["stock_code", "year"], how="left")
    final["InternationalStandardDummy"] = final["InternationalStandardDummy"].fillna(0).astype(int)
    final["AdoptedEntityCount"] = final["AdoptedEntityCount"].fillna(0).astype(int)

    sort_key = pd.to_numeric(final["stock_code"], errors="coerce")
    final = final.assign(_sort_key=sort_key).sort_values(["_sort_key", "year"]).drop(columns=["_sort_key"])
    final = final[MAIN_FINAL_COLUMNS]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    safe_to_csv(final, output_csv)
    return final


def safe_read_csv(path: Path, dtype: dict[str, object] | None = None) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, dtype=dtype)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def unique_text_unit_count(df: pd.DataFrame | None) -> int:
    if df is None or df.empty or "text_unit_id" not in df.columns:
        return 0
    return int(df["text_unit_id"].astype(str).nunique())


def write_manifest(
    paths: MainRegressionPaths,
    input_report_dir: Path,
    prompt_stage1_path: Path,
    prompt_stage2_path: Path,
    gb_mapping_path: Path,
    model_stage1: str,
    model_stage2: str,
    stage1_raw_failure_log_path: Path | None = None,
) -> dict[str, object]:
    keyword_features = safe_read_csv(paths.keyword_features_path, dtype={"text_unit_id": str})
    stage1 = safe_read_csv(paths.stage1_relevance_path, dtype={"text_unit_id": str})
    n_keyword_candidate = 0
    if keyword_features is not None and not keyword_features.empty and "keyword_candidate" in keyword_features.columns:
        n_keyword_candidate = int(keyword_features["keyword_candidate"].map(parse_bool).sum())

    n_llm_related = 0
    n_llm_uncertain = 0
    n_stage1_failed = 0
    n_stage1_skipped_keyword = 0
    if stage1 is not None and not stage1.empty:
        relevance = stage1.get("relevance", pd.Series(dtype=str)).astype(str).str.lower()
        n_llm_related = int((relevance == "related").sum())
        n_llm_uncertain = int((relevance == "uncertain").sum())
        if "stage1_status" in stage1.columns:
            stage1_status = stage1["stage1_status"].astype(str).str.upper()
            n_stage1_failed = int((stage1_status == "ERROR").sum())
            n_stage1_skipped_keyword = int((stage1_status == "SKIPPED_KEYWORD").sum())

    required_paths = [
        paths.text_units_path,
        paths.keyword_features_path,
        paths.stage1_relevance_path,
        paths.stage2_input_path,
        paths.stage2_result_path,
        paths.mapped_result_path,
        paths.final_output_path,
    ]

    manifest = write_measurement_manifest(
        paths.manifest_path,
        method="main_regression",
        year=paths.year,
        input_paths={
            "input_report_dir": input_report_dir,
        },
        output_paths={
            "text_units_path": paths.text_units_path,
            "keyword_features_path": paths.keyword_features_path,
            "stage1_relevance_path": paths.stage1_relevance_path,
            "stage1_raw_failure_log_path": stage1_raw_failure_log_path,
            "stage2_input_path": paths.stage2_input_path,
            "stage2_result_path": paths.stage2_result_path,
            "mapped_result_path": paths.mapped_result_path,
            "final_output_path": paths.final_output_path,
            "manifest_path": paths.manifest_path,
        },
        prompt_stage1_path=prompt_stage1_path,
        prompt_stage2_path=prompt_stage2_path,
        gb_mapping_path=gb_mapping_path,
        model_stage1=model_stage1,
        model_stage2=model_stage2,
        text_units_path=paths.text_units_path,
        stage2_input_path=paths.stage2_input_path,
        stage2_result_path=paths.stage2_result_path,
        final_output_path=paths.final_output_path,
        required_paths=required_paths,
        extra_counts={
            "N_keyword_candidate": n_keyword_candidate,
            "N_llm_related": n_llm_related,
            "N_llm_uncertain": n_llm_uncertain,
            "N_stage1_skipped_keyword": n_stage1_skipped_keyword,
            "N_stage1_failed": n_stage1_failed,
            "N_not_routed_to_stage2": max(
                count_rows(paths.text_units_path, dtype={"stock_code": str, "year": str, "text_unit_id": str})
                - count_rows(paths.stage2_input_path, dtype={"stock_code": str, "year": str, "text_unit_id": str}),
                0,
            ),
        },
    )
    if n_stage1_failed:
        manifest["complete"] = False
        paths.manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return manifest

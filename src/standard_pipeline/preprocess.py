from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import re
import unicodedata

import pandas as pd
from tqdm import tqdm

from .csv_io import safe_to_csv
from .schemas import PREPROCESSED_COLUMNS


@dataclass(frozen=True)
class PreprocessSettings:
    window_pre: int = 100
    window_post: int = 100
    max_chars: int = 15000
    min_chapter_chars: int = 500
    target_chapter_keywords: tuple[str, ...] = (
        "经营情况",
        "管理层讨论",
        "核心竞争力",
        "环境",
        "社会责任",
        "公司治理",
        "重要事项",
        "董事会报告",
    )


def settings_from_config(data: dict) -> PreprocessSettings:
    return PreprocessSettings(
        window_pre=int(data.get("window_pre", 100)),
        window_post=int(data.get("window_post", 100)),
        max_chars=int(data.get("max_chars", 15000)),
        min_chapter_chars=int(data.get("min_chapter_chars", 500)),
        target_chapter_keywords=tuple(data.get("target_chapter_keywords", []))
        or PreprocessSettings().target_chapter_keywords,
    )


def load_keywords(path: Path) -> list[str]:
    keywords: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        item = line.strip()
        if item and not item.startswith("#"):
            keywords.append(item)
    if not keywords:
        raise ValueError(f"No keywords found in {path}")
    return sorted(set(keywords), key=len, reverse=True)


def build_keyword_pattern(keywords: list[str]) -> re.Pattern[str]:
    return re.compile("|".join(re.escape(keyword) for keyword in keywords), re.IGNORECASE)


def read_text_file(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")


def is_ascii_letter(char: str) -> bool:
    return bool(char) and char.isascii() and char.isalpha()


def normalize_text_layout(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def normalize_chapter_titles(text: str, chapter_keywords: tuple[str, ...]) -> str:
    if not chapter_keywords:
        return text
    keywords_pattern = "|".join(re.escape(keyword) for keyword in chapter_keywords)
    # A chapter reference inside prose (for example, `详见本报告“第四节董事会报告”`)
    # must not become a structural boundary.  Match horizontal whitespace only and
    # anchor both ends so that only a complete source line is normalized.
    pattern = re.compile(
        rf"^[ \t]*(第[一二三四五六七八九十百\d]+[章节])[ \t]*"
        rf"([^\r\n]*(?:{keywords_pattern})[^\r\n]*)[ \t]*$",
        re.IGNORECASE | re.MULTILINE,
    )
    return pattern.sub(lambda match: f"\n\n{match.group(1)} {match.group(2).strip()}\n\n", text)


def extract_target_chapters(text: str, settings: PreprocessSettings) -> str:
    normalized = normalize_chapter_titles(text, settings.target_chapter_keywords)
    split_pattern = re.compile(
        r"^[ \t]*(第[一二三四五六七八九十百\d]+[章节][ \t]+[^\r\n]+?)[ \t]*$",
        re.MULTILINE,
    )
    segments = split_pattern.split(normalized)
    if len(segments) < 3:
        return text

    selected: list[str] = []
    current_title = ""
    for segment in segments:
        stripped = segment.strip()
        if split_pattern.fullmatch(stripped):
            current_title = stripped
            continue
        if current_title and any(keyword in current_title for keyword in settings.target_chapter_keywords):
            selected.append(f"\n\n--- {current_title} ---\n\n{segment}")

    filtered_text = "".join(selected)
    if len(filtered_text) < settings.min_chapter_chars:
        return text
    return filtered_text


def parse_report_filename(filename: str) -> dict[str, str] | None:
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) < 3:
        return None

    if re.fullmatch(r"\d{4}", parts[1]):
        year, company_name = parts[1], parts[2]
    elif re.fullmatch(r"\d{4}", parts[2]):
        company_name, year = parts[1], parts[2]
    else:
        return None

    return {
        "stock_code": normalize_stock_code(parts[0]),
        "year": year,
        "company_name": company_name,
    }


def normalize_stock_code(value: object) -> str:
    """Keep numeric A-share codes join-safe without changing non-numeric codes."""
    text = str(value).strip()
    if re.fullmatch(r"\d+(?:\.0+)?", text):
        digits = text.split(".", 1)[0]
        if len(digits) <= 6:
            return digits.zfill(6)
    return text


def extract_relevant_text(file_path: Path, pattern: re.Pattern[str], settings: PreprocessSettings) -> str:
    raw_text = read_text_file(file_path)
    filtered_text = extract_target_chapters(raw_text, settings)
    clean_text = normalize_text_layout(filtered_text)
    text_len = len(clean_text)
    intervals: list[tuple[int, int]] = []

    for match in pattern.finditer(clean_text):
        start, end = match.span()
        matched = match.group()
        if is_ascii_letter(matched[0]) and start > 0 and is_ascii_letter(clean_text[start - 1]):
            continue
        if is_ascii_letter(matched[-1]) and end < text_len and is_ascii_letter(clean_text[end]):
            continue
        intervals.append((max(0, start - settings.window_pre), min(text_len, end + settings.window_post)))

    if not intervals:
        return "未匹配到关键词"

    merged: list[tuple[int, int]] = []
    current_start, current_end = sorted(intervals)[0]
    for next_start, next_end in sorted(intervals)[1:]:
        if next_start <= current_end:
            current_end = max(current_end, next_end)
        else:
            merged.append((current_start, current_end))
            current_start, current_end = next_start, next_end
    merged.append((current_start, current_end))

    sentence_delimiters = re.compile(r"[。！？；!?;]")
    segments: list[str] = []
    for start, end in merged:
        search_start = max(0, start - 50)
        pre_matches = list(sentence_delimiters.finditer(clean_text[search_start:start]))
        if pre_matches:
            start = search_start + pre_matches[-1].end()

        search_end = min(text_len, end + 50)
        post_match = sentence_delimiters.search(clean_text[end:search_end])
        if post_match:
            end = end + post_match.end()

        segment = clean_text[start:end].strip()
        if segment:
            segments.append(segment)

    result = "\n\n......(片段分隔)......\n\n".join(segments)
    return result[: settings.max_chars] if result else "未匹配到关键词"


def infer_report_dir(annual_reports_dir: Path, year: str) -> Path:
    matches = sorted(path for path in annual_reports_dir.glob(f"{year}_*") if path.is_dir())
    if not matches:
        raise FileNotFoundError(f"Cannot find annual report directory for year {year} under {annual_reports_dir}")
    if len(matches) > 1:
        names = [str(path) for path in matches]
        raise ValueError(f"Multiple annual report directories matched year {year}: {names}")
    return matches[0]


def run_preprocess(
    input_dir: Path,
    output_csv: Path,
    keyword_file: Path,
    settings: PreprocessSettings,
    limit: int | None = None,
) -> pd.DataFrame:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    keywords = load_keywords(keyword_file)
    pattern = build_keyword_pattern(keywords)
    files = sorted(path for path in input_dir.iterdir() if path.suffix.lower() == ".txt")
    if limit is not None:
        files = files[:limit]

    records: list[dict[str, str]] = []
    for file_path in tqdm(files, desc="preprocess"):
        meta = parse_report_filename(file_path.name)
        if not meta:
            continue
        records.append(
            {
                **meta,
                "full_text": extract_relevant_text(file_path, pattern, settings),
            }
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records, columns=PREPROCESSED_COLUMNS)
    safe_to_csv(df, output_csv)
    return df

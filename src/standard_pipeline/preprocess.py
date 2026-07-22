from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

@dataclass(frozen=True)
class PreprocessSettings:
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


def read_text_file(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")


def is_ascii_letter(char: str) -> bool:
    return bool(char) and char.isascii() and char.isalpha()


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


def infer_report_dir(annual_reports_dir: Path, year: str) -> Path:
    matches = sorted(path for path in annual_reports_dir.glob(f"{year}_*") if path.is_dir())
    if not matches:
        raise FileNotFoundError(f"Cannot find annual report directory for year {year} under {annual_reports_dir}")
    if len(matches) > 1:
        names = [str(path) for path in matches]
        raise ValueError(f"Multiple annual report directories matched year {year}: {names}")
    return matches[0]

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
import re
import unicodedata

import pandas as pd

from .csv_io import safe_to_csv
from .schemas import EXTRACTION_COLUMNS, MAPPED_COLUMNS, require_columns


GB_STANDARD_COLUMNS = ("标准号", "标准编号", "国家标准编号")
INTERNATIONAL_COLUMNS = ("国际标准编号", "国际标准", "采用国际标准编号")
ADOPTION_COLUMNS = ("采标类型", "采标情况", "采用程度")
EFFECTIVE_DATE_COLUMNS = ("生效日期", "实施日期")
EXPIRY_DATE_COLUMNS = ("失效日期",)
CURRENT_STATUS_COLUMNS = ("当前状态", "状态")
TEMPORAL_QUALITY_COLUMNS = ("时间数据质量",)


@dataclass(frozen=True)
class MappingEntry:
    year_value: int
    international_standard: str | None
    adoption_status: str | None
    original_code: str
    effective_from: date | None = None
    effective_to: date | None = None
    current_status: str | None = None
    temporal_quality: str | None = None


def normalize_text(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = unicodedata.normalize("NFKC", str(value)).strip()
    text = text.replace("–", "-").replace("—", "-").replace("－", "-")
    text = text.replace("（", "(").replace("）", ")")
    return text or None


def parse_standard_code(value: object) -> tuple[str | None, str | None]:
    text = normalize_text(value)
    if not text:
        return None, None
    text = re.sub(r"\s+", " ", text).strip().upper()
    match = re.match(r"^([A-Z]+(?:/[A-Z]+)?\s*\d+(?:\.\d+)*)(?:[-:](\d{2,4}))?$", text)
    if match:
        return match.group(1).replace(" ", ""), match.group(2)

    if "-" in text:
        base, suffix = text.rsplit("-", 1)
        if re.fullmatch(r"\d{2,4}", suffix):
            return base.replace(" ", "").replace("-", "").upper(), suffix

    return text.replace(" ", "").replace("-", "").upper(), None


def full_year(value: str | None) -> int:
    if not value:
        return 0
    if len(value) == 4:
        return int(value)
    if len(value) == 2:
        year = int(value)
        return 1900 + year if year > 50 else 2000 + year
    return 0


def pick_column(df: pd.DataFrame, candidates: tuple[str, ...], source: str) -> str:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise ValueError(f"{source} is missing one of required columns: {candidates}")


def pick_optional_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def clean_optional(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text if text else None


def clean_date(value: object) -> date | None:
    if pd.isna(value):
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date()


def parse_report_year(value: object) -> int | None:
    text = normalize_text(value)
    if not text:
        return None
    match = re.search(r"\b(\d{4})\b", text)
    if match:
        return int(match.group(1))
    if re.fullmatch(r"\d{2}", text):
        return full_year(text)
    return None


def build_mapping_lookup(mapping_df: pd.DataFrame) -> dict[str, list[MappingEntry]]:
    gb_col = pick_column(mapping_df, GB_STANDARD_COLUMNS, "GB mapping file")
    intl_col = pick_column(mapping_df, INTERNATIONAL_COLUMNS, "GB mapping file")
    adopt_col = pick_column(mapping_df, ADOPTION_COLUMNS, "GB mapping file")
    effective_col = pick_optional_column(mapping_df, EFFECTIVE_DATE_COLUMNS)
    expiry_col = pick_optional_column(mapping_df, EXPIRY_DATE_COLUMNS)
    current_status_col = pick_optional_column(mapping_df, CURRENT_STATUS_COLUMNS)
    temporal_quality_col = pick_optional_column(mapping_df, TEMPORAL_QUALITY_COLUMNS)

    lookup: dict[str, list[MappingEntry]] = {}
    for _, row in mapping_df.iterrows():
        base, year = parse_standard_code(row[gb_col])
        if not base:
            continue
        entry = MappingEntry(
            year_value=full_year(year),
            international_standard=clean_optional(row[intl_col]),
            adoption_status=clean_optional(row[adopt_col]),
            original_code=str(row[gb_col]),
            effective_from=clean_date(row[effective_col]) if effective_col else None,
            effective_to=clean_date(row[expiry_col]) if expiry_col else None,
            current_status=clean_optional(row[current_status_col]) if current_status_col else None,
            temporal_quality=clean_optional(row[temporal_quality_col]) if temporal_quality_col else None,
        )
        lookup.setdefault(base, []).append(entry)

    for entries in lookup.values():
        entries.sort(key=lambda item: item.year_value, reverse=True)
    return lookup


def find_mapping(
    entity: object,
    lookup: dict[str, list[MappingEntry]],
    report_year: object | None = None,
) -> MappingEntry | None:
    base, year = parse_standard_code(entity)
    if not base or base not in lookup:
        return None
    candidates = lookup[base]
    if year:
        target_year = full_year(year)
        exact_candidates = [candidate for candidate in candidates if candidate.year_value == target_year]
        if not exact_candidates:
            return None
        candidate = max(
            exact_candidates,
            key=lambda item: (item.effective_from or date(item.year_value, 1, 1), item.original_code),
        )
        cutoff_year = parse_report_year(report_year)
        if cutoff_year is not None:
            cutoff = date(cutoff_year, 12, 31)
            if candidate.year_value > cutoff_year:
                return None
            if candidate.effective_from and candidate.effective_from > cutoff:
                return None
        return candidate

    cutoff_year = parse_report_year(report_year)
    if cutoff_year is None:
        return candidates[0] if candidates else None

    cutoff = date(cutoff_year, 12, 31)
    eligible: list[MappingEntry] = []
    for candidate in candidates:
        if candidate.year_value <= 0 or candidate.year_value > cutoff_year:
            continue
        if candidate.temporal_quality and "LEGACY_ONLY_NO_TEMPORAL_METADATA" in candidate.temporal_quality:
            continue
        if candidate.effective_from:
            if candidate.effective_from > cutoff:
                continue
            if candidate.effective_to and cutoff >= candidate.effective_to:
                continue
            if not candidate.effective_to and candidate.current_status == "废止":
                continue
        eligible.append(candidate)

    if not eligible:
        return None
    return max(
        eligible,
        key=lambda item: (item.effective_from or date(item.year_value, 1, 1), item.year_value, item.original_code),
    )


def map_row(row: pd.Series, lookup: dict[str, list[MappingEntry]]) -> tuple[str | None, str, int]:
    entity = row["entity"]
    item_type = row["type"]
    status = row["status"]

    final_standard: str | None = entity
    adoption_status = item_type
    output = 0

    if item_type == "TYPE_B":
        mapping = find_mapping(entity, lookup, row.get("year"))
        if mapping and mapping.international_standard:
            final_standard = mapping.international_standard
            adoption_status = mapping.adoption_status or "已找到采标信息"
            if status == "ADOPTED":
                output = 1
        else:
            final_standard = None
            adoption_status = "未找到采标信息"
    elif item_type in {"TYPE_A", "TYPE_C"} and status == "ADOPTED":
        output = 1

    return final_standard, adoption_status, output


def run_gb_mapping(input_csv: Path, mapping_csv: Path, output_csv: Path) -> pd.DataFrame:
    predictions = pd.read_csv(input_csv, dtype={"stock_code": str, "year": str})
    require_columns(predictions, EXTRACTION_COLUMNS, str(input_csv))

    mapping_df = pd.read_csv(mapping_csv)
    lookup = build_mapping_lookup(mapping_df)

    mapped = predictions.copy()
    mapped_values = mapped.apply(lambda row: map_row(row, lookup), axis=1)
    mapped["国际标准"] = [value[0] for value in mapped_values]
    mapped["采标情况"] = [value[1] for value in mapped_values]
    mapped["output"] = [value[2] for value in mapped_values]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    safe_to_csv(mapped, output_csv)
    return mapped[MAPPED_COLUMNS]

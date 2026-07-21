from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import unicodedata

import pandas as pd

from .csv_io import safe_to_csv
from .schemas import EXTRACTION_COLUMNS, MAPPED_COLUMNS, require_columns


GB_STANDARD_COLUMNS = ("标准号", "标准编号", "国家标准编号")
INTERNATIONAL_COLUMNS = ("国际标准编号", "国际标准", "采用国际标准编号")
ADOPTION_COLUMNS = ("采标类型", "采标情况", "采用程度")


@dataclass(frozen=True)
class MappingEntry:
    year_value: int
    international_standard: str | None
    adoption_status: str | None
    original_code: str


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


def clean_optional(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text if text else None


def build_mapping_lookup(mapping_df: pd.DataFrame) -> dict[str, list[MappingEntry]]:
    gb_col = pick_column(mapping_df, GB_STANDARD_COLUMNS, "GB mapping file")
    intl_col = pick_column(mapping_df, INTERNATIONAL_COLUMNS, "GB mapping file")
    adopt_col = pick_column(mapping_df, ADOPTION_COLUMNS, "GB mapping file")

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
        )
        lookup.setdefault(base, []).append(entry)

    for entries in lookup.values():
        entries.sort(key=lambda item: item.year_value, reverse=True)
    return lookup


def find_mapping(entity: object, lookup: dict[str, list[MappingEntry]]) -> MappingEntry | None:
    base, year = parse_standard_code(entity)
    if not base or base not in lookup:
        return None
    candidates = lookup[base]
    if year:
        target_year = full_year(year)
        for candidate in candidates:
            if candidate.year_value == target_year:
                return candidate
    return candidates[0] if candidates else None


def map_row(row: pd.Series, lookup: dict[str, list[MappingEntry]]) -> tuple[str | None, str, int]:
    entity = row["entity"]
    item_type = row["type"]
    status = row["status"]

    final_standard: str | None = entity
    adoption_status = item_type
    output = 0

    if item_type == "TYPE_B":
        mapping = find_mapping(entity, lookup)
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

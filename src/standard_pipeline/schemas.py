from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


PREPROCESSED_COLUMNS = ["stock_code", "company_name", "year", "full_text"]
EXTRACTION_COLUMNS = ["stock_code", "company_name", "year", "entity", "type", "status", "evidence"]
MAPPED_COLUMNS = EXTRACTION_COLUMNS + ["国际标准", "采标情况", "output"]

TEXT_UNIT_COLUMNS = [
    "text_unit_id",
    "stock_code",
    "company_name",
    "year",
    "source_file",
    "chapter_title",
    "unit_order",
    "text",
]
TEXT_UNIT_AUDIT_COLUMNS = [
    "text_unit_id",
    "company_name",
    "unit_order",
    "table_noise_score",
    "has_protected_anchor",
    "raw_text_len",
    "filtered_text_len",
    "removed_ratio",
    "raw_preview",
    "filtered_preview",
]
KEYWORD_FEATURE_COLUMNS = ["text_unit_id", "keyword_candidate", "matched_terms"]
STAGE1_RELEVANCE_COLUMNS = [
    "text_unit_id",
    "relevance",
    "confidence_score",
    "reason",
    "stage1_status",
    "stage1_error",
]
STAGE2_INPUT_COLUMNS = TEXT_UNIT_COLUMNS.copy()
ROUTE_AUDIT_COLUMNS = TEXT_UNIT_COLUMNS + [
    "keyword_candidate",
    "matched_terms",
    "relevance",
    "confidence_score",
    "route_reason",
]
TEXT_UNIT_EXTRACTION_COLUMNS = [
    "text_unit_id",
    "stock_code",
    "company_name",
    "year",
    "entity",
    "type",
    "status",
    "evidence",
]
MAIN_MAPPED_COLUMNS = TEXT_UNIT_EXTRACTION_COLUMNS + ["国际标准", "采标情况", "output"]
MAIN_FINAL_COLUMNS = [
    "stock_code",
    "company_name",
    "year",
    "InternationalStandardDummy",
    "AdoptedEntityCount",
]


class SchemaError(ValueError):
    pass


def require_columns(df: pd.DataFrame, required: Iterable[str], source: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise SchemaError(f"{source} is missing required columns: {missing}")

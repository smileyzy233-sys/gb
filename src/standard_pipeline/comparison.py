from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from pathlib import Path

import pandas as pd

from .csv_io import safe_to_csv
from .measurement_manifest import write_measurement_manifest
from .schemas import MAIN_FINAL_COLUMNS, require_columns


COMPARISON_BASE_COLUMNS = [
    "stock_code",
    "company_name",
    "year",
    "InternationalStandardDummy_main",
    "InternationalStandardDummy_keyword",
    "InternationalStandardDummy_llm_only",
    "InternationalStandardDummy_full_llm",
]
COMPARISON_COLUMNS = COMPARISON_BASE_COLUMNS + [
    "main_eq_keyword",
    "main_eq_llm_only",
    "main_eq_full_llm",
]


@dataclass(frozen=True)
class ComparisonPaths:
    base_dir: Path
    year: str

    @property
    def stage_dir(self) -> Path:
        return self.base_dir / "stage"

    @property
    def final_dir(self) -> Path:
        return self.base_dir / "final"

    @property
    def logs_dir(self) -> Path:
        return self.base_dir / "logs"

    @property
    def collected_output_path(self) -> Path:
        return self.stage_dir / f"01_collected_firm_year_outputs_{self.year}.csv"

    @property
    def consistency_output_path(self) -> Path:
        return self.final_dir / f"02_dummy_consistency_{self.year}.csv"

    @property
    def manifest_path(self) -> Path:
        return self.logs_dir / f"03_manifest_{self.year}.json"

    def ensure_dirs(self) -> None:
        for directory in (self.stage_dir, self.final_dir, self.logs_dir):
            directory.mkdir(parents=True, exist_ok=True)


def default_comparison_paths(
    project_root: Path,
    year: str,
    smoke: bool = False,
    base_dir: Path | None = None,
) -> ComparisonPaths:
    data_root = project_root / "data" / ("measurement_smoke" if smoke else "measurement")
    resolved_base = base_dir or data_root / "comparison"
    return ComparisonPaths(base_dir=resolved_base.resolve(), year=str(year))


def _load_final(path: Path, suffix: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"stock_code": str, "year": str})
    require_columns(df, MAIN_FINAL_COLUMNS, str(path))
    return df[["stock_code", "company_name", "year", "InternationalStandardDummy"]].rename(
        columns={
            "company_name": f"company_name_{suffix}",
            "InternationalStandardDummy": f"InternationalStandardDummy_{suffix}",
        }
    )


def run_collect_firm_year_outputs(
    main_final_csv: Path,
    keyword_final_csv: Path,
    llm_only_final_csv: Path,
    full_llm_final_csv: Path,
    output_csv: Path,
) -> pd.DataFrame:
    frames = [
        _load_final(main_final_csv, "main"),
        _load_final(keyword_final_csv, "keyword"),
        _load_final(llm_only_final_csv, "llm_only"),
        _load_final(full_llm_final_csv, "full_llm"),
    ]
    merged = reduce(lambda left, right: left.merge(right, on=["stock_code", "year"], how="outer"), frames)
    company_columns = [column for column in merged.columns if column.startswith("company_name_")]
    merged["company_name"] = merged[company_columns].bfill(axis=1).iloc[:, 0]
    merged = merged.drop(columns=company_columns)
    merged = merged[COMPARISON_BASE_COLUMNS]

    sort_key = pd.to_numeric(merged["stock_code"], errors="coerce")
    merged = merged.assign(_sort_key=sort_key).sort_values(["_sort_key", "year"]).drop(columns=["_sort_key"])
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    safe_to_csv(merged, output_csv)
    return merged


def _dummy_eq(df: pd.DataFrame, other_column: str) -> pd.Series:
    main = pd.to_numeric(df["InternationalStandardDummy_main"], errors="coerce")
    other = pd.to_numeric(df[other_column], errors="coerce")
    return main.eq(other) & main.notna() & other.notna()


def run_compare_dummy_consistency(collected_csv: Path, output_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(collected_csv, dtype={"stock_code": str, "year": str})
    require_columns(df, COMPARISON_BASE_COLUMNS, str(collected_csv))
    result = df.copy()
    result["main_eq_keyword"] = _dummy_eq(result, "InternationalStandardDummy_keyword")
    result["main_eq_llm_only"] = _dummy_eq(result, "InternationalStandardDummy_llm_only")
    result["main_eq_full_llm"] = _dummy_eq(result, "InternationalStandardDummy_full_llm")
    result = result[COMPARISON_COLUMNS]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    safe_to_csv(result, output_csv)
    return result


def write_comparison_manifest(
    paths: ComparisonPaths,
    *,
    main_final_csv: Path,
    keyword_final_csv: Path,
    llm_only_final_csv: Path,
    full_llm_final_csv: Path,
) -> dict[str, object]:
    required_paths = [
        paths.collected_output_path,
        paths.consistency_output_path,
    ]
    return write_measurement_manifest(
        paths.manifest_path,
        method="comparison",
        year=paths.year,
        input_paths={
            "main_final_csv": main_final_csv,
            "keyword_final_csv": keyword_final_csv,
            "llm_only_final_csv": llm_only_final_csv,
            "full_llm_final_csv": full_llm_final_csv,
        },
        output_paths={
            "collected_output_path": paths.collected_output_path,
            "consistency_output_path": paths.consistency_output_path,
            "manifest_path": paths.manifest_path,
        },
        prompt_stage1_path=None,
        prompt_stage2_path=None,
        gb_mapping_path=None,
        model_stage1=None,
        model_stage2=None,
        text_units_path=None,
        stage2_input_path=None,
        stage2_result_path=None,
        final_output_path=None,
        required_paths=required_paths,
    )

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .csv_io import safe_to_csv
from .measurement_manifest import count_rows, parse_bool, safe_read_csv, write_measurement_manifest
from .schemas import (
    KEYWORD_FEATURE_COLUMNS,
    STAGE1_RELEVANCE_COLUMNS,
    TEXT_UNIT_COLUMNS,
    TEXT_UNIT_EXTRACTION_COLUMNS,
    require_columns,
)


ROBUSTNESS_METHODS = {
    "robustness_keyword": {
        "unit_slug": "keyword",
        "final_slug": "robustness_keyword",
    },
    "robustness_llm_only": {
        "unit_slug": "llm",
        "final_slug": "robustness_llm_only",
    },
    "robustness_full_llm": {
        "unit_slug": "full",
        "final_slug": "robustness_full_llm",
    },
}


@dataclass(frozen=True)
class RobustnessPaths:
    base_dir: Path
    year: str
    method: str

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
    def unit_slug(self) -> str:
        return ROBUSTNESS_METHODS[self.method]["unit_slug"]

    @property
    def final_slug(self) -> str:
        return ROBUSTNESS_METHODS[self.method]["final_slug"]

    @property
    def units_path(self) -> Path:
        return self.stage_dir / f"01_{self.unit_slug}_units_{self.year}.csv"

    @property
    def stage2_result_path(self) -> Path:
        return self.results_dir / f"02_stage2_entity_result_{self.year}.csv"

    @property
    def mapped_result_path(self) -> Path:
        return self.results_dir / f"03_mapped_entity_result_{self.year}.csv"

    @property
    def final_output_path(self) -> Path:
        return self.final_dir / f"04_{self.final_slug}_firm_year_{self.year}.csv"

    @property
    def manifest_path(self) -> Path:
        return self.logs_dir / f"05_manifest_{self.year}.json"

    def stage2_log_path(self, provider: str) -> Path:
        return self.logs_dir / f"02_stage2_processed_{self.year}_{provider}.log"

    def ensure_dirs(self) -> None:
        for directory in (self.stage_dir, self.results_dir, self.final_dir, self.logs_dir):
            directory.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Stage2SeedReport:
    seeded_rows: int
    completed_ids: int
    missing_ids: list[str]


def default_robustness_paths(
    project_root: Path,
    method: str,
    year: str,
    smoke: bool = False,
    base_dir: Path | None = None,
) -> RobustnessPaths:
    if method not in ROBUSTNESS_METHODS:
        raise ValueError(f"Unknown robustness method: {method}")
    data_root = project_root / "data" / ("measurement_smoke" if smoke else "measurement")
    resolved_base = base_dir or data_root / method
    return RobustnessPaths(base_dir=resolved_base.resolve(), year=str(year), method=method)


def _write_units(df: pd.DataFrame, output_csv: Path, limit: int | None = None) -> pd.DataFrame:
    result = df[TEXT_UNIT_COLUMNS].copy()
    if limit is not None:
        result = result.head(limit).copy()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    safe_to_csv(result, output_csv)
    return result


def run_prepare_keyword_units(
    text_units_csv: Path,
    keyword_features_csv: Path,
    output_csv: Path,
    limit: int | None = None,
) -> pd.DataFrame:
    text_units = pd.read_csv(text_units_csv, dtype={"stock_code": str, "year": str, "text_unit_id": str})
    keyword_features = pd.read_csv(keyword_features_csv, dtype={"text_unit_id": str})
    require_columns(text_units, TEXT_UNIT_COLUMNS, str(text_units_csv))
    require_columns(keyword_features, KEYWORD_FEATURE_COLUMNS, str(keyword_features_csv))

    merged = text_units.merge(keyword_features, on="text_unit_id", how="left")
    routed = merged[merged["keyword_candidate"].map(parse_bool).fillna(False)].copy()
    return _write_units(routed, output_csv, limit=limit)


def run_prepare_llm_units(
    text_units_csv: Path,
    stage1_relevance_csv: Path,
    output_csv: Path,
    limit: int | None = None,
) -> pd.DataFrame:
    text_units = pd.read_csv(text_units_csv, dtype={"stock_code": str, "year": str, "text_unit_id": str})
    stage1 = pd.read_csv(stage1_relevance_csv, dtype={"text_unit_id": str})
    require_columns(text_units, TEXT_UNIT_COLUMNS, str(text_units_csv))
    require_columns(stage1, STAGE1_RELEVANCE_COLUMNS, str(stage1_relevance_csv))

    merged = text_units.merge(stage1, on="text_unit_id", how="left")
    relevance = merged["relevance"].fillna("").astype(str).str.lower()
    routed = merged[relevance.isin(["related", "uncertain"])].copy()
    return _write_units(routed, output_csv, limit=limit)


def run_prepare_full_units(
    text_units_csv: Path,
    output_csv: Path,
    limit: int | None = None,
) -> pd.DataFrame:
    text_units = pd.read_csv(text_units_csv, dtype={"stock_code": str, "year": str, "text_unit_id": str})
    require_columns(text_units, TEXT_UNIT_COLUMNS, str(text_units_csv))
    return _write_units(text_units, output_csv, limit=limit)


def _ordered_unique(values: pd.Series) -> list[str]:
    return list(dict.fromkeys(values.astype(str).tolist()))


def _completed_stage2_ids(output_csv: Path) -> set[str]:
    existing = safe_read_csv(output_csv, dtype={"stock_code": str, "year": str, "text_unit_id": str})
    if existing is None or existing.empty or "text_unit_id" not in existing.columns:
        return set()
    return set(existing["text_unit_id"].astype(str))


def _sync_stage2_log(log_file: Path, completed_ids: set[str]) -> None:
    if not completed_ids:
        return
    logged_ids: set[str] = set()
    if log_file.exists():
        logged_ids = {line.strip() for line in log_file.read_text(encoding="utf-8").splitlines() if line.strip()}
    missing = sorted(completed_ids - logged_ids)
    if not missing:
        return
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as file:
        file.write("\n".join(missing) + "\n")


def seed_stage2_results_from_existing(
    units_csv: Path,
    source_result_csv: Path,
    output_csv: Path,
    log_file: Path,
) -> Stage2SeedReport:
    units = pd.read_csv(units_csv, dtype={"stock_code": str, "year": str, "text_unit_id": str})
    require_columns(units, TEXT_UNIT_COLUMNS, str(units_csv))
    target_ids = _ordered_unique(units["text_unit_id"])
    target_set = set(target_ids)
    already_completed = _completed_stage2_ids(output_csv)
    if not target_ids and not output_csv.exists():
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        safe_to_csv(pd.DataFrame(columns=TEXT_UNIT_EXTRACTION_COLUMNS), output_csv)

    seeded_rows = 0
    if source_result_csv.exists():
        source = pd.read_csv(source_result_csv, dtype={"stock_code": str, "year": str, "text_unit_id": str})
        require_columns(source, TEXT_UNIT_EXTRACTION_COLUMNS, str(source_result_csv))
        reusable = source[
            source["text_unit_id"].astype(str).isin(target_set - already_completed)
        ].copy()
        if not reusable.empty:
            order = {text_unit_id: index for index, text_unit_id in enumerate(target_ids)}
            reusable = (
                reusable.assign(_order=reusable["text_unit_id"].astype(str).map(order))
                .sort_values(["_order", "text_unit_id"])
                .drop(columns=["_order"])
            )
            output_csv.parent.mkdir(parents=True, exist_ok=True)
            header = not output_csv.exists() or output_csv.stat().st_size == 0
            safe_to_csv(
                reusable[TEXT_UNIT_EXTRACTION_COLUMNS],
                output_csv,
                mode="a",
                header=header,
            )
            reusable_ids = _ordered_unique(reusable["text_unit_id"])
            log_file.parent.mkdir(parents=True, exist_ok=True)
            with log_file.open("a", encoding="utf-8") as file:
                file.write("\n".join(reusable_ids) + "\n")
            seeded_rows = int(len(reusable))

    completed_ids = _completed_stage2_ids(output_csv)
    _sync_stage2_log(log_file, completed_ids & target_set)
    missing_ids = [text_unit_id for text_unit_id in target_ids if text_unit_id not in completed_ids]
    return Stage2SeedReport(
        seeded_rows=seeded_rows,
        completed_ids=len(completed_ids & target_set),
        missing_ids=missing_ids,
    )


def robustness_extra_counts(method: str, paths: RobustnessPaths, main_stage1_path: Path | None = None) -> dict[str, int]:
    if method == "robustness_keyword":
        return {"N_keyword_candidate": count_rows(paths.units_path, dtype={"text_unit_id": str})}
    if method == "robustness_full_llm":
        return {"N_full_units": count_rows(paths.units_path, dtype={"text_unit_id": str})}

    stage1 = safe_read_csv(main_stage1_path, dtype={"text_unit_id": str}) if main_stage1_path else None
    if stage1 is None or stage1.empty:
        return {"N_llm_related": 0, "N_llm_uncertain": 0}
    relevance = stage1.get("relevance", pd.Series(dtype=str)).astype(str).str.lower()
    return {
        "N_llm_related": int((relevance == "related").sum()),
        "N_llm_uncertain": int((relevance == "uncertain").sum()),
    }


def write_robustness_manifest(
    paths: RobustnessPaths,
    *,
    source_text_units_path: Path,
    source_keyword_features_path: Path | None,
    source_stage1_relevance_path: Path | None,
    source_stage2_result_path: Path,
    prompt_stage1_path: Path | None,
    prompt_stage2_path: Path,
    gb_mapping_path: Path,
    model_stage1: str | None,
    model_stage2: str | None,
) -> dict[str, object]:
    required_paths = [
        paths.units_path,
        paths.stage2_result_path,
        paths.mapped_result_path,
        paths.final_output_path,
    ]
    return write_measurement_manifest(
        paths.manifest_path,
        method=paths.method,
        year=paths.year,
        input_paths={
            "source_text_units_path": source_text_units_path,
            "source_keyword_features_path": source_keyword_features_path,
            "source_stage1_relevance_path": source_stage1_relevance_path,
            "source_stage2_result_path": source_stage2_result_path,
        },
        output_paths={
            "units_path": paths.units_path,
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
        text_units_path=source_text_units_path,
        stage2_input_path=paths.units_path,
        stage2_result_path=paths.stage2_result_path,
        final_output_path=paths.final_output_path,
        required_paths=required_paths,
        extra_counts=robustness_extra_counts(paths.method, paths, source_stage1_relevance_path),
    )

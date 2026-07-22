from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import pandas as pd


def safe_read_csv(path: Path | None, dtype: dict[str, object] | None = None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    try:
        return pd.read_csv(path, dtype=dtype)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def count_rows(path: Path | None, dtype: dict[str, object] | None = None) -> int:
    df = safe_read_csv(path, dtype=dtype)
    return 0 if df is None else int(len(df))


def is_stage2_failure_sentinel(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(False, index=df.index, dtype=bool)
    return (
        df.get("entity", pd.Series("", index=df.index)).astype(str).str.upper().eq("ERROR")
        | df.get("type", pd.Series("", index=df.index)).astype(str).str.upper().eq("LLM_FAILURE")
        | df.get("status", pd.Series("", index=df.index)).astype(str).str.upper().eq("FAIL")
    )


def read_failure_queue_ids(failure_queue_path: Path | None) -> set[str]:
    if failure_queue_path is None or not failure_queue_path.exists():
        return set()
    failure_ids: set[str] = set()
    for line in failure_queue_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        text_unit_id = str(record.get("text_unit_id") or "").strip()
        if text_unit_id:
            failure_ids.add(text_unit_id)
    return failure_ids


def count_stage2_status(
    stage2_result_path: Path | None,
    failure_queue_path: Path | None = None,
) -> tuple[int, int]:
    stage2_result = safe_read_csv(
        stage2_result_path,
        dtype={"stock_code": str, "year": str, "text_unit_id": str},
    )
    completed_ids: set[str] = set()
    sentinel_ids: set[str] = set()
    if stage2_result is not None and not stage2_result.empty and "text_unit_id" in stage2_result.columns:
        failure_mask = is_stage2_failure_sentinel(stage2_result)
        sentinel_ids = set(stage2_result.loc[failure_mask, "text_unit_id"].astype(str))
        completed_ids = set(stage2_result.loc[~failure_mask, "text_unit_id"].astype(str))
    failed_ids = (read_failure_queue_ids(failure_queue_path) | sentinel_ids) - completed_ids
    return len(completed_ids), len(failed_ids)


def count_output_1(final_output_path: Path | None) -> int:
    final = safe_read_csv(final_output_path, dtype={"stock_code": str, "year": str})
    if final is None or final.empty or "InternationalStandardDummy" not in final.columns:
        return 0
    values = pd.to_numeric(final["InternationalStandardDummy"], errors="coerce").fillna(0).astype(int)
    return int((values == 1).sum())


def path_map_to_strings(paths: dict[str, Path | str | None]) -> dict[str, str | None]:
    return {key: None if value is None else str(value) for key, value in paths.items()}


def write_measurement_manifest(
    manifest_path: Path,
    *,
    method: str,
    year: str,
    input_paths: dict[str, Path | str | None],
    output_paths: dict[str, Path | str | None],
    prompt_stage1_path: Path | str | None,
    prompt_stage2_path: Path | str | None,
    gb_mapping_path: Path | str | None,
    model_stage1: str | None,
    model_stage2: str | None,
    text_units_path: Path | None,
    stage2_input_path: Path | None,
    stage2_result_path: Path | None,
    stage2_failure_queue_path: Path | None = None,
    final_output_path: Path | None,
    required_paths: list[Path],
    extra_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    n_text_units = count_rows(text_units_path, dtype={"stock_code": str, "year": str, "text_unit_id": str})
    routed = safe_read_csv(stage2_input_path, dtype={"stock_code": str, "year": str, "text_unit_id": str})
    n_routed = (
        0
        if routed is None or routed.empty or "text_unit_id" not in routed.columns
        else int(routed["text_unit_id"].astype(str).nunique())
    )
    n_stage2_completed, n_stage2_failed = count_stage2_status(
        stage2_result_path,
        stage2_failure_queue_path,
    )
    n_stage2_pending = max(n_routed - n_stage2_completed, 0)
    n_output_1 = count_output_1(final_output_path)
    complete = (
        all(path.exists() for path in required_paths)
        and n_stage2_completed == n_routed
        and n_stage2_failed == 0
    )
    created_at = datetime.now(timezone.utc).isoformat()

    manifest: dict[str, Any] = {
        "workflow_id": f"{method}_{year}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "year": str(year),
        "method": method,
        "input_paths": path_map_to_strings(input_paths),
        "output_paths": path_map_to_strings(output_paths),
        "prompt_stage1_path": None if prompt_stage1_path is None else str(prompt_stage1_path),
        "prompt_stage2_path": None if prompt_stage2_path is None else str(prompt_stage2_path),
        "gb_mapping_path": None if gb_mapping_path is None else str(gb_mapping_path),
        "model_stage1": model_stage1,
        "model_stage2": model_stage2,
        "N_text_units_available": n_text_units,
        "N_routed_to_stage2": n_routed,
        "N_stage2_completed": n_stage2_completed,
        "N_stage2_failed": n_stage2_failed,
        "N_stage2_pending": n_stage2_pending,
        "N_output_1": n_output_1,
        "complete": complete,
        "created_at": created_at,
    }
    if extra_counts:
        manifest.update(extra_counts)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest

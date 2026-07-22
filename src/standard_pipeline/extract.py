from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time
from typing import Any

import pandas as pd
from tqdm import tqdm

from .csv_io import safe_to_csv
from .llm import ChatClient
from .measurement_manifest import is_stage2_failure_sentinel
from .schemas import STAGE2_INPUT_COLUMNS, TEXT_UNIT_EXTRACTION_COLUMNS, require_columns


VALID_TYPES = {"TYPE_A", "TYPE_B", "TYPE_C", "TYPE_D"}
VALID_STATUSES = {"ADOPTED", "PENDING", "NO"}
RAW_RESPONSE_LIMIT = 4000


@dataclass(frozen=True)
class ExtractSettings:
    workers: int = 3
    batch_size: int = 10
    max_retries: int = 3
    retry_min_seconds: int = 2
    retry_max_seconds: int = 10


@dataclass(frozen=True)
class Stage2Outcome:
    text_unit_id: str
    success: bool
    rows: list[dict[str, str]]
    failure_record: dict[str, object] | None = None


class Stage2AttemptsExhausted(RuntimeError):
    def __init__(self, attempt: int, error: Exception, raw_response: str | None = None) -> None:
        super().__init__(str(error))
        self.attempt = attempt
        self.error_type = type(error).__name__
        self.raw_response = raw_response


def clean_json_string(raw: str) -> str:
    if not raw:
        return ""
    text = re.sub(r"^```json\s*", "", raw.strip(), flags=re.MULTILINE)
    text = re.sub(r"^```\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"```$", "", text, flags=re.MULTILINE)
    return text.strip()


def extract_json_object(raw: str) -> Any:
    text = clean_json_string(raw)
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            data, _ = decoder.raw_decode(text[index:])
            return data
        except json.JSONDecodeError:
            continue
    raise json.JSONDecodeError("Cannot parse a JSON object from model output", text, 0)


def normalize_items(data: Any) -> list[dict[str, str]]:
    if isinstance(data, dict):
        if "standards" not in data:
            raise ValueError("Stage2 response is missing the standards field")
        items = data["standards"]
    elif isinstance(data, list):
        items = data
    else:
        raise ValueError("Stage2 response is neither an object nor a list")
    if not isinstance(items, list):
        raise ValueError("Stage2 standards field is not a list")

    normalized: list[dict[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Stage2 standards contains a non-object item")
        entity = str(item.get("entity") or "无").strip() or "无"
        item_type = str(item.get("type") or "TYPE_D").strip().upper()
        status = str(item.get("status") or "NO").strip().upper()
        evidence = str(item.get("evidence") or "未发现相关描述").strip() or "未发现相关描述"
        normalized.append(
            {
                "entity": entity,
                "type": item_type if item_type in VALID_TYPES else "TYPE_D",
                "status": status if status in VALID_STATUSES else "NO",
                "evidence": evidence,
            }
        )
    return normalized


def no_result_item(reason: str = "未发现相关描述") -> list[dict[str, str]]:
    return [{"entity": "无", "type": "TYPE_D", "status": "NO", "evidence": reason}]


def text_unit_task_id(row: pd.Series) -> str:
    return str(row["text_unit_id"])


def build_text_unit_user_content(row: pd.Series) -> str:
    return (
        "请分析以下年报 text_unit，只能根据给定 text 判断，并且只输出合法 JSON。\n"
        f"text_unit_id: {row['text_unit_id']}\n"
        f"stock_code: {row['stock_code']}\n"
        f"company_name: {row['company_name']}\n"
        f"year: {row['year']}\n"
        "text:\n"
        "---\n"
        f"{row['text']}\n"
        "---"
    )


def _entity_rows(row: pd.Series, items: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {
            "text_unit_id": str(row["text_unit_id"]),
            "stock_code": str(row["stock_code"]),
            "company_name": str(row["company_name"]),
            "year": str(row["year"]),
            "entity": item["entity"],
            "type": item["type"],
            "status": item["status"],
            "evidence": item["evidence"],
        }
        for item in items
    ]


def truncate_raw_response(raw: str | None, max_chars: int = RAW_RESPONSE_LIMIT) -> str | None:
    if raw is None or len(raw) <= max_chars:
        return raw
    return raw[:max_chars] + f"...[truncated {len(raw) - max_chars} chars]"


def stage2_failure_record(
    row: pd.Series | dict[str, Any],
    *,
    provider: str,
    attempt: int,
    error_type: str,
    error: str,
    raw_response: str | None,
    retries_exhausted: bool,
) -> dict[str, object]:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": "stage2",
        "provider": provider,
        "text_unit_id": str(row["text_unit_id"]),
        "stock_code": str(row["stock_code"]),
        "company_name": str(row["company_name"]),
        "year": str(row["year"]),
        "attempt": int(attempt),
        "error_type": error_type,
        "error": error,
        "raw_response": truncate_raw_response(raw_response),
        "retries_exhausted": bool(retries_exhausted),
    }


def call_text_unit_with_retry(
    client: ChatClient,
    system_prompt: str,
    row: pd.Series,
    settings: ExtractSettings,
) -> list[dict[str, str]]:
    text_content = str(row.get("text", ""))
    if not text_content or len(text_content.strip()) < 5:
        return no_result_item("文本内容过短或为空，跳过处理")

    user_content = build_text_unit_user_content(row)
    last_error: Exception = RuntimeError("Stage2 did not run")
    last_raw: str | None = None
    for attempt in range(1, settings.max_retries + 1):
        try:
            last_raw = client.complete_json(system_prompt, user_content)
            items = normalize_items(extract_json_object(last_raw))
            return items or no_result_item()
        except Exception as exc:
            last_error = exc
            if attempt >= settings.max_retries:
                raise Stage2AttemptsExhausted(attempt, exc, last_raw) from exc
            delay = min(settings.retry_max_seconds, settings.retry_min_seconds * (2 ** (attempt - 1)))
            time.sleep(delay)
    raise Stage2AttemptsExhausted(settings.max_retries, last_error, last_raw)


def process_text_unit_row(
    row: pd.Series,
    client: ChatClient,
    system_prompt: str,
    settings: ExtractSettings,
    provider: str = "api",
) -> Stage2Outcome:
    text_unit_id = str(row["text_unit_id"])
    try:
        items = call_text_unit_with_retry(client, system_prompt, row, settings)
        return Stage2Outcome(text_unit_id, True, _entity_rows(row, items))
    except Stage2AttemptsExhausted as exc:
        return Stage2Outcome(
            text_unit_id,
            False,
            [],
            stage2_failure_record(
                row,
                provider=provider,
                attempt=exc.attempt,
                error_type=exc.error_type,
                error=str(exc),
                raw_response=exc.raw_response,
                retries_exhausted=True,
            ),
        )


def stage2_outcome_from_raw(
    row: pd.Series | dict[str, Any],
    raw: str,
    provider: str = "vllm_batch",
    attempt: int = 1,
) -> Stage2Outcome:
    series = row if isinstance(row, pd.Series) else pd.Series(row)
    text_unit_id = str(series["text_unit_id"])
    try:
        items = normalize_items(extract_json_object(raw)) or no_result_item()
        return Stage2Outcome(text_unit_id, True, _entity_rows(series, items))
    except Exception as exc:
        return Stage2Outcome(
            text_unit_id,
            False,
            [],
            stage2_failure_record(
                series,
                provider=provider,
                attempt=attempt,
                error_type=type(exc).__name__,
                error=str(exc),
                raw_response=raw,
                retries_exhausted=True,
            ),
        )


def stage2_outcome_from_exception(
    row: pd.Series | dict[str, Any],
    error: Exception,
    provider: str,
    attempt: int = 1,
) -> Stage2Outcome:
    series = row if isinstance(row, pd.Series) else pd.Series(row)
    text_unit_id = str(series["text_unit_id"])
    return Stage2Outcome(
        text_unit_id,
        False,
        [],
        stage2_failure_record(
            series,
            provider=provider,
            attempt=attempt,
            error_type=type(error).__name__,
            error=str(error),
            raw_response=None,
            retries_exhausted=True,
        ),
    )


def successful_stage2_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=TEXT_UNIT_EXTRACTION_COLUMNS)
    require_columns(df, TEXT_UNIT_EXTRACTION_COLUMNS, "stage2 result")
    return df.loc[~is_stage2_failure_sentinel(df), TEXT_UNIT_EXTRACTION_COLUMNS].copy()


def successful_stage2_ids(output_csv: Path) -> set[str]:
    if not output_csv.exists() or output_csv.stat().st_size == 0:
        return set()
    try:
        existing = pd.read_csv(output_csv, dtype={"stock_code": str, "year": str, "text_unit_id": str})
    except pd.errors.EmptyDataError:
        return set()
    cleaned = successful_stage2_rows(existing)
    return set(cleaned["text_unit_id"].astype(str)) if not cleaned.empty else set()


def reconcile_completed_tasks(output_csv: Path, log_file: Path) -> set[str]:
    if output_csv.exists() and output_csv.stat().st_size:
        try:
            existing = pd.read_csv(output_csv, dtype={"stock_code": str, "year": str, "text_unit_id": str})
        except pd.errors.EmptyDataError:
            existing = pd.DataFrame(columns=TEXT_UNIT_EXTRACTION_COLUMNS)
        cleaned = successful_stage2_rows(existing)
        if len(cleaned) != len(existing):
            safe_to_csv(cleaned, output_csv)
    else:
        cleaned = pd.DataFrame(columns=TEXT_UNIT_EXTRACTION_COLUMNS)
    completed_ids = set(cleaned["text_unit_id"].astype(str)) if not cleaned.empty else set()
    log_file.parent.mkdir(parents=True, exist_ok=True)
    payload = "" if not completed_ids else "\n".join(sorted(completed_ids)) + "\n"
    log_file.write_text(payload, encoding="utf-8")
    return completed_ids


def append_failure_records(failure_queue: Path, records: list[dict[str, object]]) -> None:
    if not records:
        return
    failure_queue.parent.mkdir(parents=True, exist_ok=True)
    with failure_queue.open("a", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_stage2_outcomes(
    output_csv: Path,
    log_file: Path,
    failure_queue: Path,
    outcomes: list[Stage2Outcome],
) -> None:
    successful_rows = [row for outcome in outcomes if outcome.success for row in outcome.rows]
    completed_ids = [outcome.text_unit_id for outcome in outcomes if outcome.success]
    failures = [outcome.failure_record for outcome in outcomes if not outcome.success and outcome.failure_record]
    if successful_rows:
        header = not output_csv.exists() or output_csv.stat().st_size == 0
        safe_to_csv(
            pd.DataFrame(successful_rows, columns=TEXT_UNIT_EXTRACTION_COLUMNS),
            output_csv,
            mode="a",
            header=header,
        )
    if completed_ids:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("a", encoding="utf-8") as file:
            file.write("\n".join(completed_ids) + "\n")
    append_failure_records(failure_queue, failures)


def initialize_stage2_run(
    output_csv: Path,
    log_file: Path,
    failure_queue: Path,
    resume: bool,
) -> set[str]:
    if not resume:
        for path in (output_csv, log_file, failure_queue):
            path.unlink(missing_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if not output_csv.exists():
        safe_to_csv(pd.DataFrame(columns=TEXT_UNIT_EXTRACTION_COLUMNS), output_csv)
    failure_queue.parent.mkdir(parents=True, exist_ok=True)
    failure_queue.touch(exist_ok=True)
    return reconcile_completed_tasks(output_csv, log_file) if resume else set()


def finalize_stage2_output(output_csv: Path, log_file: Path) -> pd.DataFrame:
    df_final = pd.read_csv(output_csv, dtype={"stock_code": str, "year": str, "text_unit_id": str})
    if not df_final.empty:
        sort_key = pd.to_numeric(df_final["stock_code"], errors="coerce")
        df_final = (
            df_final.assign(_sort_key=sort_key)
            .sort_values(["_sort_key", "year", "text_unit_id"])
            .drop(columns=["_sort_key"])
        )
        safe_to_csv(df_final, output_csv)
    reconcile_completed_tasks(output_csv, log_file)
    return df_final


def run_text_unit_extraction(
    input_csv: Path,
    output_csv: Path,
    log_file: Path,
    failure_queue: Path,
    client: ChatClient,
    system_prompt: str,
    settings: ExtractSettings,
    resume: bool = True,
    limit: int | None = None,
    provider: str = "api",
) -> pd.DataFrame:
    df_input = pd.read_csv(input_csv, dtype={"stock_code": str, "year": str, "text_unit_id": str})
    require_columns(df_input, STAGE2_INPUT_COLUMNS, str(input_csv))
    df_input["task_id"] = df_input.apply(text_unit_task_id, axis=1)

    processed_tasks = initialize_stage2_run(output_csv, log_file, failure_queue, resume)
    df_work = df_input[~df_input["task_id"].isin(processed_tasks)].copy()
    if limit is not None:
        df_work = df_work.head(limit)
    if df_work.empty:
        return finalize_stage2_output(output_csv, log_file)

    outcomes_buffer: list[Stage2Outcome] = []

    def flush() -> None:
        nonlocal outcomes_buffer
        append_stage2_outcomes(output_csv, log_file, failure_queue, outcomes_buffer)
        outcomes_buffer = []

    if settings.workers <= 1:
        for _, row in tqdm(df_work.iterrows(), total=len(df_work), desc="stage2-extract"):
            outcomes_buffer.append(process_text_unit_row(row, client, system_prompt, settings, provider))
            if len(outcomes_buffer) >= settings.batch_size:
                flush()
    else:
        with ThreadPoolExecutor(max_workers=settings.workers) as executor:
            futures = [
                executor.submit(process_text_unit_row, row, client, system_prompt, settings, provider)
                for _, row in df_work.iterrows()
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc="stage2-extract"):
                outcomes_buffer.append(future.result())
                if len(outcomes_buffer) >= settings.batch_size:
                    flush()
    flush()
    return finalize_stage2_output(output_csv, log_file)

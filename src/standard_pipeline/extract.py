from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
from pathlib import Path
import re
import time
from typing import Any

import pandas as pd
from tqdm import tqdm

from .csv_io import safe_to_csv
from .llm import ChatClient
from .schemas import (
    EXTRACTION_COLUMNS,
    PREPROCESSED_COLUMNS,
    STAGE2_INPUT_COLUMNS,
    TEXT_UNIT_EXTRACTION_COLUMNS,
    require_columns,
)


VALID_TYPES = {"TYPE_A", "TYPE_B", "TYPE_C", "TYPE_D"}
VALID_STATUSES = {"ADOPTED", "PENDING", "NO"}


@dataclass(frozen=True)
class ExtractSettings:
    workers: int = 3
    batch_size: int = 10
    max_retries: int = 3
    retry_min_seconds: int = 2
    retry_max_seconds: int = 10


def settings_from_config(data: dict) -> ExtractSettings:
    return ExtractSettings(
        workers=int(data.get("workers", 3)),
        batch_size=int(data.get("batch_size", 10)),
        max_retries=int(data.get("max_retries", 3)),
        retry_min_seconds=int(data.get("retry_min_seconds", 2)),
        retry_max_seconds=int(data.get("retry_max_seconds", 10)),
    )


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
        items = data.get("standards", [])
    elif isinstance(data, list):
        items = data
    else:
        items = []

    normalized: list[dict[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
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


def call_with_retry(client: ChatClient, system_prompt: str, text_content: str, settings: ExtractSettings) -> list[dict[str, str]]:
    if not text_content or len(str(text_content).strip()) < 5:
        return no_result_item("文本内容过短或为空，跳过处理")

    user_content = f"请分析以下年报片段，只输出 JSON：\n---\n{text_content}\n---"
    last_error: Exception | None = None
    for attempt in range(1, settings.max_retries + 1):
        try:
            raw = client.complete_json(system_prompt, user_content)
            items = normalize_items(extract_json_object(raw))
            return items or no_result_item()
        except Exception as exc:
            last_error = exc
            if attempt >= settings.max_retries:
                break
            delay = min(settings.retry_max_seconds, settings.retry_min_seconds * (2 ** (attempt - 1)))
            time.sleep(delay)
    raise RuntimeError(f"LLM call failed after {settings.max_retries} attempts: {last_error}") from last_error


def process_row(row: pd.Series, client: ChatClient, system_prompt: str, settings: ExtractSettings) -> list[dict[str, str]]:
    try:
        items = call_with_retry(client, system_prompt, str(row["full_text"]), settings)
    except Exception as exc:
        items = [
            {
                "entity": "ERROR",
                "type": "LLM_FAILURE",
                "status": "FAIL",
                "evidence": str(exc),
            }
        ]

    return [
        {
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


def task_id(row: pd.Series) -> str:
    return f"{row['stock_code']}_{row['year']}"


def text_unit_task_id(row: pd.Series) -> str:
    return str(row["text_unit_id"])


def load_processed_tasks(log_file: Path) -> set[str]:
    if not log_file.exists():
        return set()
    return {line.strip() for line in log_file.read_text(encoding="utf-8").splitlines() if line.strip()}


def append_batch(output_csv: Path, log_file: Path, rows: list[dict[str, str]], task_ids: list[str]) -> None:
    if rows:
        header = not output_csv.exists() or output_csv.stat().st_size == 0
        safe_to_csv(
            pd.DataFrame(rows, columns=EXTRACTION_COLUMNS),
            output_csv,
            mode="a",
            header=header,
        )
    if task_ids:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("a", encoding="utf-8") as file:
            file.write("\n".join(task_ids) + "\n")


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
    last_error: Exception | None = None
    for attempt in range(1, settings.max_retries + 1):
        try:
            raw = client.complete_json(system_prompt, user_content)
            items = normalize_items(extract_json_object(raw))
            return items or no_result_item()
        except Exception as exc:
            last_error = exc
            if attempt >= settings.max_retries:
                break
            delay = min(settings.retry_max_seconds, settings.retry_min_seconds * (2 ** (attempt - 1)))
            time.sleep(delay)
    raise RuntimeError(f"LLM call failed after {settings.max_retries} attempts: {last_error}") from last_error


def process_text_unit_row(
    row: pd.Series,
    client: ChatClient,
    system_prompt: str,
    settings: ExtractSettings,
) -> list[dict[str, str]]:
    try:
        items = call_text_unit_with_retry(client, system_prompt, row, settings)
    except Exception as exc:
        items = [
            {
                "entity": "ERROR",
                "type": "LLM_FAILURE",
                "status": "FAIL",
                "evidence": str(exc),
            }
        ]

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


def append_text_unit_batch(
    output_csv: Path,
    log_file: Path,
    rows: list[dict[str, str]],
    task_ids: list[str],
) -> None:
    if rows:
        header = not output_csv.exists() or output_csv.stat().st_size == 0
        safe_to_csv(
            pd.DataFrame(rows, columns=TEXT_UNIT_EXTRACTION_COLUMNS),
            output_csv,
            mode="a",
            header=header,
        )
    if task_ids:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("a", encoding="utf-8") as file:
            file.write("\n".join(task_ids) + "\n")


def run_extraction(
    input_csv: Path,
    output_csv: Path,
    log_file: Path,
    client: ChatClient,
    system_prompt: str,
    settings: ExtractSettings,
    resume: bool = True,
    limit: int | None = None,
) -> pd.DataFrame:
    df_input = pd.read_csv(input_csv, dtype={"stock_code": str, "year": str})
    require_columns(df_input, PREPROCESSED_COLUMNS, str(input_csv))
    df_input["task_id"] = df_input.apply(task_id, axis=1)

    processed_tasks = load_processed_tasks(log_file) if resume else set()
    df_work = df_input[~df_input["task_id"].isin(processed_tasks)].copy()
    if limit is not None:
        df_work = df_work.head(limit)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if not output_csv.exists():
        safe_to_csv(pd.DataFrame(columns=EXTRACTION_COLUMNS), output_csv)

    if df_work.empty:
        return pd.read_csv(output_csv, dtype={"stock_code": str, "year": str})

    rows_buffer: list[dict[str, str]] = []
    tasks_buffer: list[str] = []

    if settings.workers <= 1:
        iterator = df_work.iterrows()
        for _, row in tqdm(iterator, total=len(df_work), desc="extract"):
            rows_buffer.extend(process_row(row, client, system_prompt, settings))
            tasks_buffer.append(str(row["task_id"]))
            if len(tasks_buffer) >= settings.batch_size:
                append_batch(output_csv, log_file, rows_buffer, tasks_buffer)
                rows_buffer, tasks_buffer = [], []
    else:
        with ThreadPoolExecutor(max_workers=settings.workers) as executor:
            future_to_task = {
                executor.submit(process_row, row, client, system_prompt, settings): str(row["task_id"])
                for _, row in df_work.iterrows()
            }
            for future in tqdm(as_completed(future_to_task), total=len(future_to_task), desc="extract"):
                rows_buffer.extend(future.result())
                tasks_buffer.append(future_to_task[future])
                if len(tasks_buffer) >= settings.batch_size:
                    append_batch(output_csv, log_file, rows_buffer, tasks_buffer)
                    rows_buffer, tasks_buffer = [], []

    append_batch(output_csv, log_file, rows_buffer, tasks_buffer)
    df_final = pd.read_csv(output_csv, dtype={"stock_code": str, "year": str})
    if not df_final.empty:
        sort_key = pd.to_numeric(df_final["stock_code"], errors="coerce")
        df_final = df_final.assign(_sort_key=sort_key).sort_values(["_sort_key", "year"]).drop(columns=["_sort_key"])
        safe_to_csv(df_final, output_csv)
    return df_final


def run_text_unit_extraction(
    input_csv: Path,
    output_csv: Path,
    log_file: Path,
    client: ChatClient,
    system_prompt: str,
    settings: ExtractSettings,
    resume: bool = True,
    limit: int | None = None,
) -> pd.DataFrame:
    df_input = pd.read_csv(input_csv, dtype={"stock_code": str, "year": str, "text_unit_id": str})
    require_columns(df_input, STAGE2_INPUT_COLUMNS, str(input_csv))
    df_input["task_id"] = df_input.apply(text_unit_task_id, axis=1)

    if not resume:
        if output_csv.exists():
            output_csv.unlink()
        if log_file.exists():
            log_file.unlink()

    processed_tasks = load_processed_tasks(log_file) if resume else set()
    df_work = df_input[~df_input["task_id"].isin(processed_tasks)].copy()
    if limit is not None:
        df_work = df_work.head(limit)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if not output_csv.exists():
        safe_to_csv(pd.DataFrame(columns=TEXT_UNIT_EXTRACTION_COLUMNS), output_csv)

    if df_work.empty:
        return pd.read_csv(output_csv, dtype={"stock_code": str, "year": str, "text_unit_id": str})

    rows_buffer: list[dict[str, str]] = []
    tasks_buffer: list[str] = []

    if settings.workers <= 1:
        iterator = df_work.iterrows()
        for _, row in tqdm(iterator, total=len(df_work), desc="stage2-extract"):
            rows_buffer.extend(process_text_unit_row(row, client, system_prompt, settings))
            tasks_buffer.append(str(row["task_id"]))
            if len(tasks_buffer) >= settings.batch_size:
                append_text_unit_batch(output_csv, log_file, rows_buffer, tasks_buffer)
                rows_buffer, tasks_buffer = [], []
    else:
        with ThreadPoolExecutor(max_workers=settings.workers) as executor:
            future_to_task = {
                executor.submit(process_text_unit_row, row, client, system_prompt, settings): str(row["task_id"])
                for _, row in df_work.iterrows()
            }
            for future in tqdm(as_completed(future_to_task), total=len(future_to_task), desc="stage2-extract"):
                rows_buffer.extend(future.result())
                tasks_buffer.append(future_to_task[future])
                if len(tasks_buffer) >= settings.batch_size:
                    append_text_unit_batch(output_csv, log_file, rows_buffer, tasks_buffer)
                    rows_buffer, tasks_buffer = [], []

    append_text_unit_batch(output_csv, log_file, rows_buffer, tasks_buffer)
    df_final = pd.read_csv(output_csv, dtype={"stock_code": str, "year": str, "text_unit_id": str})
    if not df_final.empty:
        sort_key = pd.to_numeric(df_final["stock_code"], errors="coerce")
        df_final = (
            df_final.assign(_sort_key=sort_key)
            .sort_values(["_sort_key", "year", "text_unit_id"])
            .drop(columns=["_sort_key"])
        )
        safe_to_csv(df_final, output_csv)
    return df_final

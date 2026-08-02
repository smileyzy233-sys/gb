from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
import signal
from typing import Any, Callable, Iterable

import pandas as pd
from tqdm import tqdm

from .csv_io import safe_to_csv
from .extract import (
    Stage2Outcome,
    append_failure_records,
    append_stage2_outcomes,
    build_text_unit_user_content,
    extract_json_object,
    finalize_stage2_output,
    initialize_stage2_run,
    stage2_failure_record,
    stage2_outcome_from_exception,
    stage2_outcome_from_raw,
    stage2_outcome_from_raw_safe,
    text_unit_task_id,
)
from .main_regression import (
    append_stage1_raw_failure,
    build_stage1_user_content,
    normalize_stage1_result,
    parse_bool,
    repair_stage1_json_object,
    stage1_keyword_skip_result,
    stage1_parse_failure_record,
)
from .schemas import (
    KEYWORD_FEATURE_COLUMNS,
    STAGE1_RELEVANCE_COLUMNS,
    STAGE2_INPUT_COLUMNS,
    TEXT_UNIT_COLUMNS,
    require_columns,
)


@dataclass(frozen=True)
class VLLMBatchConfig:
    model_path: str
    dtype: str = "auto"
    trust_remote_code: bool = True
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.90
    max_model_len: int = 4096
    chunk_size: int = 64
    temperature: float = 0.0
    max_tokens: int = 384
    enforce_eager: bool = False
    enable_lora: bool = False
    lora_path: str | None = None
    lora_rank: int = 32


@contextmanager
def defer_sigint_until_checkpoint():
    """Let the active vLLM chunk finish and be checkpointed before Ctrl-C exits."""
    requested = False
    previous_handler = signal.getsignal(signal.SIGINT)

    def request_stop(signum, frame):
        nonlocal requested
        requested = True

    signal.signal(signal.SIGINT, request_stop)
    try:
        yield lambda: requested
    finally:
        signal.signal(signal.SIGINT, previous_handler)


def vllm_batch_config_from_dict(
    data: dict,
    model_path: str | None = None,
) -> VLLMBatchConfig:
    path_env = data.get("model_path_env", "LOCAL_MODEL_PATH")
    resolved_model_path = model_path or os.environ.get(path_env) or data.get("model_path")

    if not resolved_model_path:
        raise ValueError(
            f"Missing vLLM batch model path. Set {path_env}, config model_path, or pass --model-path."
        )

    lora_path = data.get("lora_path") or None
    chunk_size = int(data.get("chunk_size", data.get("batch_size", 64)))
    if chunk_size <= 0:
        raise ValueError("vllm_batch chunk_size must be greater than 0.")

    return VLLMBatchConfig(
        model_path=resolved_model_path,
        dtype=str(data.get("dtype", "auto")),
        trust_remote_code=bool(data.get("trust_remote_code", True)),
        tensor_parallel_size=int(data.get("tensor_parallel_size", 1)),
        gpu_memory_utilization=float(data.get("gpu_memory_utilization", 0.90)),
        max_model_len=int(data.get("max_model_len", 4096)),
        chunk_size=chunk_size,
        temperature=float(data.get("temperature", 0.0)),
        max_tokens=int(data.get("max_tokens", 384)),
        enforce_eager=bool(data.get("enforce_eager", False)),
        enable_lora=bool(data.get("enable_lora", False)),
        lora_path=lora_path,
        lora_rank=int(data.get("lora_rank", 32)),
    )


def build_vllm_engine(config: VLLMBatchConfig):
    try:
        from vllm import LLM
    except ImportError as exc:
        raise ImportError("vllm_batch provider requires vllm to be installed.") from exc

    kwargs = {
        "model": config.model_path,
        "trust_remote_code": config.trust_remote_code,
        "tensor_parallel_size": config.tensor_parallel_size,
        "gpu_memory_utilization": config.gpu_memory_utilization,
        "max_model_len": config.max_model_len,
        "dtype": config.dtype,
        "enforce_eager": config.enforce_eager,
        "enable_lora": config.enable_lora,
    }

    if config.enable_lora:
        kwargs["max_lora_rank"] = config.lora_rank

    llm = LLM(**kwargs)
    tokenizer = llm.get_tokenizer()

    try:
        if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception:
        pass

    return llm, tokenizer


def build_sampling_params(
    tokenizer,
    config: VLLMBatchConfig,
    structured_output_kwargs: dict[str, Any] | None = None,
):
    try:
        from vllm import SamplingParams
    except ImportError as exc:
        raise ImportError("vllm_batch provider requires vllm to be installed.") from exc

    stop_token_ids = [
        token_id
        for token_id in (
            getattr(tokenizer, "eos_token_id", None),
            getattr(tokenizer, "pad_token_id", None),
        )
        if token_id is not None
    ]

    kwargs: dict[str, Any] = {
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "stop_token_ids": stop_token_ids,
    }

    if structured_output_kwargs:
        kwargs.update(structured_output_kwargs)

    return SamplingParams(**kwargs)


def apply_chat_template(tokenizer, system_prompt: str, user_content: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return f"System:\n{system_prompt}\n\nUser:\n{user_content}\n\nAssistant:\n"


def generate_raw_batch(
    llm,
    tokenizer,
    rows: Iterable[dict[str, Any]],
    system_prompt: str,
    user_content_builder: Callable[[dict[str, Any]], str],
    config: VLLMBatchConfig,
    structured_output_kwargs: dict[str, Any] | None = None,
) -> list[str]:
    sampling_params = build_sampling_params(
        tokenizer, config, structured_output_kwargs=structured_output_kwargs
    )

    prompts = [
        apply_chat_template(
            tokenizer,
            system_prompt,
            user_content_builder(row),
        )
        for row in rows
    ]

    generate_kwargs = {}

    if config.enable_lora:
        if not config.lora_path:
            raise ValueError("enable_lora=true but lora_path is empty.")

        from vllm.lora.request import LoRARequest

        generate_kwargs["lora_request"] = LoRARequest(
            "standard_pipeline_lora",
            config.lora_rank,
            config.lora_path,
        )

    outputs = llm.generate(
        prompts,
        sampling_params,
        use_tqdm=True,
        **generate_kwargs,
    )

    return [output.outputs[0].text for output in outputs]


def _stage1_error_result(text_unit_id: str, error: Exception) -> dict[str, object]:
    return {
        "text_unit_id": text_unit_id,
        "relevance": "",
        "confidence_score": "",
        "reason": "",
        "stage1_status": "ERROR",
        "stage1_error": str(error),
    }


def _parse_stage1_raw(raw: str, text_unit_id: str, raw_failure_log: Path | None) -> dict[str, object]:
    try:
        data = extract_json_object(raw)
        result = normalize_stage1_result(data, text_unit_id)
    except Exception as exc:
        append_stage1_raw_failure(
            raw_failure_log,
            stage1_parse_failure_record(text_unit_id, 1, raw, exc),
        )
        try:
            repaired = repair_stage1_json_object(raw, text_unit_id)
            result = normalize_stage1_result(repaired, text_unit_id)
        except Exception:
            raise exc
    result["text_unit_id"] = text_unit_id
    return result


def run_stage1_screening_vllm_batch(
    text_units_csv: Path,
    output_csv: Path,
    system_prompt: str,
    config: VLLMBatchConfig,
    limit: int | None = None,
    keyword_features_csv: Path | None = None,
    raw_failure_log: Path | None = None,
    resume: bool = True,
) -> pd.DataFrame:
    text_units = pd.read_csv(
        text_units_csv,
        dtype={
            "stock_code": str,
            "year": str,
            "text_unit_id": str,
        },
    )
    require_columns(text_units, TEXT_UNIT_COLUMNS, str(text_units_csv))

    if limit is not None:
        text_units = text_units.head(limit).copy()

    order = {str(row["text_unit_id"]): index for index, row in text_units.iterrows()}
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if not resume:
        if output_csv.exists():
            output_csv.unlink()
        if raw_failure_log is not None and raw_failure_log.exists():
            raw_failure_log.unlink()

    if output_csv.exists():
        existing = pd.read_csv(output_csv, dtype={"text_unit_id": str})
        require_columns(existing, STAGE1_RELEVANCE_COLUMNS, str(output_csv))
    else:
        existing = pd.DataFrame(columns=STAGE1_RELEVANCE_COLUMNS)
        safe_to_csv(existing, output_csv)

    processed_ids = (
        set(
            existing.loc[
                existing["stage1_status"].astype(str).str.upper().ne("ERROR"),
                "text_unit_id",
            ].dropna().astype(str)
        )
        if resume
        else set()
    )

    def append_checkpoint(records: list[dict[str, object]]) -> None:
        if records:
            safe_to_csv(
                pd.DataFrame(records, columns=STAGE1_RELEVANCE_COLUMNS),
                output_csv,
                mode="a",
                header=False,
            )

    keyword_ids: set[str] = set()
    if keyword_features_csv is not None and keyword_features_csv.exists():
        keyword_features = pd.read_csv(keyword_features_csv, dtype={"text_unit_id": str})
        require_columns(keyword_features, KEYWORD_FEATURE_COLUMNS, str(keyword_features_csv))
        keyword_ids = set(
            keyword_features.loc[
                keyword_features["keyword_candidate"].map(parse_bool).fillna(False),
                "text_unit_id",
            ].astype(str)
        )

    if keyword_ids:
        keyword_records = [
            stage1_keyword_skip_result(str(row["text_unit_id"]))
            for _, row in text_units[
                text_units["text_unit_id"].astype(str).isin(keyword_ids - processed_ids)
            ].iterrows()
        ]
        append_checkpoint(keyword_records)
        processed_ids.update(str(record["text_unit_id"]) for record in keyword_records)

    work_units = text_units[~text_units["text_unit_id"].astype(str).isin(processed_ids)].copy()

    rows = list(work_units.to_dict("records"))
    if rows:
        llm, tokenizer = build_vllm_engine(config)

        for start in tqdm(range(0, len(rows), config.chunk_size), desc="stage1-screen"):
            chunk = rows[start : start + config.chunk_size]
            chunk_records: list[dict[str, object]] = []

            with defer_sigint_until_checkpoint() as stop_requested:
                raw_outputs = generate_raw_batch(
                    llm,
                    tokenizer,
                    chunk,
                    system_prompt,
                    lambda row: build_stage1_user_content(pd.Series(row)),
                    config,
                )

                for row, raw in zip(chunk, raw_outputs):
                    text_unit_id = str(row["text_unit_id"])
                    try:
                        chunk_records.append(_parse_stage1_raw(raw, text_unit_id, raw_failure_log))
                    except Exception as exc:
                        chunk_records.append(_stage1_error_result(text_unit_id, exc))
                append_checkpoint(chunk_records)
            if stop_requested():
                raise KeyboardInterrupt

    df = pd.read_csv(output_csv, dtype={"text_unit_id": str})
    if not df.empty:
        df = df.drop_duplicates(subset=["text_unit_id"], keep="last")
        df = df[df["text_unit_id"].isin(order)]
        df = df.assign(_order=df["text_unit_id"].map(order)).sort_values("_order").drop(columns="_order")
    safe_to_csv(df, output_csv)
    return df


def _run_extraction_original(
    rows: list[dict[str, Any]],
    output_csv: Path,
    log_file: Path,
    failure_queue: Path,
    system_prompt: str,
    config: VLLMBatchConfig,
    llm: Any,
    tokenizer: Any,
) -> pd.DataFrame:
    """Original one-pass extraction (no safe recovery)."""
    outcomes_buffer: list[Stage2Outcome] = []

    for start in tqdm(range(0, len(rows), config.chunk_size), desc="stage2-extract"):
        chunk = rows[start : start + config.chunk_size]

        with defer_sigint_until_checkpoint() as stop_requested:
            try:
                raw_outputs = generate_raw_batch(
                    llm, tokenizer, chunk, system_prompt,
                    lambda row: build_text_unit_user_content(pd.Series(row)),
                    config,
                )
                outcomes_buffer.extend(
                    stage2_outcome_from_raw(row, raw, provider="vllm_batch")
                    for row, raw in zip(chunk, raw_outputs)
                )
                if len(raw_outputs) < len(chunk):
                    error = RuntimeError("vLLM returned fewer outputs than inputs")
                    outcomes_buffer.extend(
                        stage2_outcome_from_exception(row, error, "vllm_batch")
                        for row in chunk[len(raw_outputs):]
                    )
            except Exception as exc:
                outcomes_buffer.extend(
                    stage2_outcome_from_exception(row, exc, "vllm_batch") for row in chunk
                )

            append_stage2_outcomes(output_csv, log_file, failure_queue, outcomes_buffer)
            outcomes_buffer = []
        if stop_requested():
            raise KeyboardInterrupt

    append_stage2_outcomes(output_csv, log_file, failure_queue, outcomes_buffer)
    return finalize_stage2_output(output_csv, log_file)


def _run_extraction_safe(
    rows: list[dict[str, Any]],
    output_csv: Path,
    log_file: Path,
    failure_queue: Path,
    system_prompt: str,
    config: VLLMBatchConfig,
    llm: Any,
    tokenizer: Any,
    recovery: "SafeRecoveryConfig",
    audit_log: Path,
    structured_kwargs: dict[str, Any] | None,
) -> pd.DataFrame:
    """Two-pass extraction with safe recovery."""
    from .stage2_recovery import (  # noqa: F811
        append_audit_entries,
        build_retry_user_content,
        make_audit_entry,
    )

    outcomes_buffer: list[Stage2Outcome] = []
    first_pass_failures: list[dict[str, Any]] = []
    first_pass_failure_ids: set[str] = set()

    # ── First pass ──────────────────────────────────────────────────
    gen_kwargs = {}
    if structured_kwargs:
        gen_kwargs["structured_output_kwargs"] = structured_kwargs

    for start in tqdm(range(0, len(rows), config.chunk_size), desc="stage2-extract"):
        chunk = rows[start : start + config.chunk_size]

        with defer_sigint_until_checkpoint() as stop_requested:
            try:
                raw_outputs = generate_raw_batch(
                    llm, tokenizer, chunk, system_prompt,
                    lambda row: build_text_unit_user_content(pd.Series(row)),
                    config,
                    **gen_kwargs,
                )
            except Exception as exc:
                for row in chunk:
                    tid = str(row["text_unit_id"])
                    outcomes_buffer.append(
                        stage2_outcome_from_exception(row, exc, "vllm_batch")
                    )
                    first_pass_failure_ids.add(tid)
                append_stage2_outcomes(output_csv, log_file, failure_queue, outcomes_buffer)
                outcomes_buffer = []
                if stop_requested():
                    raise KeyboardInterrupt
                continue

            if len(raw_outputs) < len(chunk):
                error = RuntimeError("vLLM returned fewer outputs than inputs")
                for row in chunk[len(raw_outputs):]:
                    tid = str(row["text_unit_id"])
                    outcomes_buffer.append(
                        stage2_outcome_from_exception(row, error, "vllm_batch")
                    )
                    first_pass_failure_ids.add(tid)

            for row, raw in zip(chunk, raw_outputs):
                outcome = stage2_outcome_from_raw_safe(
                    row, raw,
                    provider="vllm_batch",
                    attempt=1,
                    structured_output=recovery.structured_output,
                    audit_log=audit_log,
                    source_text=str(row.get("text", "")) if recovery.grounding_mode == "audit" else None,
                    grounding_mode=recovery.grounding_mode,
                )

                if outcome.success:
                    outcomes_buffer.append(outcome)
                else:
                    first_pass_failures.append(row)
                    first_pass_failure_ids.add(outcome.text_unit_id)

            append_stage2_outcomes(output_csv, log_file, failure_queue, outcomes_buffer)
            outcomes_buffer = []

        if stop_requested():
            raise KeyboardInterrupt

    # ── Second pass (retry first-pass failures only) ────────────────
    if not first_pass_failures:
        return finalize_stage2_output(output_csv, log_file)

    for start in tqdm(
        range(0, len(first_pass_failures), config.chunk_size),
        desc="stage2-extract-retry",
    ):
        retry_chunk = first_pass_failures[start : start + config.chunk_size]

        with defer_sigint_until_checkpoint() as stop_requested:
            try:
                raw_outputs = generate_raw_batch(
                    llm, tokenizer, retry_chunk, system_prompt,
                    lambda row: build_retry_user_content(
                        build_text_unit_user_content(pd.Series(row))
                    ),
                    config,
                    **gen_kwargs,
                )
            except Exception as exc:
                for row in retry_chunk:
                    tid = str(row["text_unit_id"])
                    rec = stage2_failure_record(
                        pd.Series(row), provider="vllm_batch", attempt=2,
                        error_type=type(exc).__name__, error=str(exc),
                        raw_response=None, retries_exhausted=True,
                    )
                    append_failure_records(failure_queue, [rec])
                if stop_requested():
                    raise KeyboardInterrupt
                continue

            if len(raw_outputs) < len(retry_chunk):
                error = RuntimeError("vLLM returned fewer outputs than inputs (retry)")
                for row in retry_chunk[len(raw_outputs):]:
                    tid = str(row["text_unit_id"])
                    rec = stage2_failure_record(
                        pd.Series(row), provider="vllm_batch", attempt=2,
                        error_type=type(error).__name__, error=str(error),
                        raw_response=None, retries_exhausted=True,
                    )
                    append_failure_records(failure_queue, [rec])

            for row, raw in zip(retry_chunk, raw_outputs):
                outcome = stage2_outcome_from_raw_safe(
                    row, raw,
                    provider="vllm_batch",
                    attempt=2,
                    structured_output=recovery.structured_output,
                    audit_log=audit_log,
                    source_text=str(row.get("text", "")) if recovery.grounding_mode == "audit" else None,
                    grounding_mode=recovery.grounding_mode,
                )

                if outcome.success:
                    success_entry = make_audit_entry(
                        text_unit_id=outcome.text_unit_id,
                        attempt=2,
                        structured_output=recovery.structured_output,
                        parse_method="structured_retry",
                        parse_success=True,
                        schema_valid=True,
                        repair_attempted=False,
                        final_disposition="success",
                        raw_response=raw,
                    )
                    append_audit_entries(audit_log, [success_entry])
                    outcomes_buffer.append(outcome)
                else:
                    tid = outcome.text_unit_id
                    rec = stage2_failure_record(
                        pd.Series(row), provider="vllm_batch", attempt=2,
                        error_type=outcome.failure_record.get("error_type", "ParseError") if outcome.failure_record else "ParseError",
                        error=outcome.failure_record.get("error", str(raw)[:200]) if outcome.failure_record else str(raw)[:200],
                        raw_response=raw, retries_exhausted=True,
                    )
                    append_failure_records(failure_queue, [rec])

            append_stage2_outcomes(output_csv, log_file, failure_queue, outcomes_buffer)
            outcomes_buffer = []

        if stop_requested():
            raise KeyboardInterrupt

    append_stage2_outcomes(output_csv, log_file, failure_queue, outcomes_buffer)
    return finalize_stage2_output(output_csv, log_file)


def run_text_unit_extraction_vllm_batch(
    input_csv: Path,
    output_csv: Path,
    log_file: Path,
    failure_queue: Path,
    system_prompt: str,
    config: VLLMBatchConfig,
    resume: bool = True,
    limit: int | None = None,
    safe_recovery_config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    from .stage2_recovery import (
        STAGE2_JSON_SCHEMA,
        build_structured_output_kwargs,
        safe_recovery_config_from_dict,
    )

    recovery = safe_recovery_config_from_dict(safe_recovery_config or {})

    df_input = pd.read_csv(
        input_csv,
        dtype={"stock_code": str, "year": str, "text_unit_id": str},
    )
    require_columns(df_input, STAGE2_INPUT_COLUMNS, str(input_csv))
    df_input["task_id"] = df_input.apply(text_unit_task_id, axis=1)

    processed_tasks = initialize_stage2_run(output_csv, log_file, failure_queue, resume)
    df_work = df_input[~df_input["task_id"].isin(processed_tasks)].copy()
    if limit is not None:
        df_work = df_work.head(limit)

    if df_work.empty:
        return finalize_stage2_output(output_csv, log_file)

    llm, tokenizer = build_vllm_engine(config)
    rows = list(df_work.to_dict("records"))

    if not recovery.enabled:
        return _run_extraction_original(
            rows, output_csv, log_file, failure_queue, system_prompt, config, llm, tokenizer,
        )

    # ── Safe recovery mode ──────────────────────────────────────────
    structured_kwargs: dict[str, Any] | None = None
    if recovery.structured_output:
        structured_kwargs = build_structured_output_kwargs(STAGE2_JSON_SCHEMA)

    audit_log = (
        failure_queue.parent
        / f"05_stage2_recovery_audit_{Path(input_csv).stem.split('_')[-1]}_vllm_batch.jsonl"
    )

    return _run_extraction_safe(
        rows, output_csv, log_file, failure_queue, system_prompt, config,
        llm, tokenizer, recovery, audit_log, structured_kwargs,
    )

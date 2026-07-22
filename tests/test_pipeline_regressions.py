import json
from pathlib import Path
import re

import pandas as pd
import pytest

from standard_pipeline import cli
from standard_pipeline.extract import ExtractSettings, run_text_unit_extraction
from standard_pipeline.gb_mapping import run_gb_mapping
from standard_pipeline.main_regression import run_aggregate_main, run_stage1_screening
from standard_pipeline.measurement_manifest import count_stage2_status, write_measurement_manifest
from standard_pipeline.robustness import (
    default_robustness_paths,
    run_prepare_llm_units,
    write_robustness_manifest,
)
from standard_pipeline.schemas import (
    ROUTE_AUDIT_COLUMNS,
    STAGE1_RELEVANCE_COLUMNS,
    TEXT_UNIT_COLUMNS,
    TEXT_UNIT_EXTRACTION_COLUMNS,
)
from standard_pipeline.vllm_batch import (
    VLLMBatchConfig,
    run_stage1_screening_vllm_batch,
    run_text_unit_extraction_vllm_batch,
)


def unit_rows(ids=("id1", "id2", "id3")):
    return [
        {
            "text_unit_id": text_unit_id,
            "stock_code": f"{index:06d}",
            "company_name": f"C{index}",
            "year": "2024",
            "source_file": f"r{index}.txt",
            "chapter_title": "经营情况",
            "unit_order": 1,
            "text": f"用于测试的年报标准描述 {text_unit_id}",
        }
        for index, text_unit_id in enumerate(ids, start=1)
    ]


def write_units(path: Path, ids=("id1", "id2", "id3"), audit=False) -> None:
    frame = pd.DataFrame(unit_rows(ids), columns=TEXT_UNIT_COLUMNS)
    if audit:
        frame["keyword_candidate"] = [True] + [False] * (len(frame) - 1)
        frame["matched_terms"] = ["ISO"] + [""] * (len(frame) - 1)
        frame["relevance"] = [""] + ["related"] * (len(frame) - 1)
        frame["confidence_score"] = [""] + [0.9] * (len(frame) - 1)
        frame["route_reason"] = ["keyword"] + ["stage1_related"] * (len(frame) - 1)
        frame = frame[ROUTE_AUDIT_COLUMNS]
    frame.to_csv(path, index=False, encoding="utf-8-sig")


class RelatedClient:
    def __init__(self):
        self.calls = 0

    def complete_json(self, system_prompt, user_content):
        self.calls += 1
        match = re.search(r"text_unit_id:\s*(\w+)", user_content)
        text_unit_id = match.group(1) if match else "unknown"
        return json.dumps(
            {
                "text_unit_id": text_unit_id,
                "relevance": "related",
                "confidence_score": 0.9,
                "reason": "semantic decision",
            }
        )


class Stage2Client:
    def __init__(self, failures=()):
        self.failures = set(failures)
        self.calls = []

    def complete_json(self, system_prompt, user_content):
        assert "keyword_candidate:" not in user_content
        assert "route_reason:" not in user_content
        text_unit_id = re.search(r"text_unit_id:\s*(\w+)", user_content).group(1)
        self.calls.append(text_unit_id)
        if text_unit_id in self.failures:
            raise RuntimeError(f"temporary failure for {text_unit_id}")
        if text_unit_id == "id2":
            return '{"standards": []}'
        return (
            '{"standards":[{"entity":"ISO 9001","type":"TYPE_A",'
            '"status":"ADOPTED","evidence":"通过认证"}]}'
        )


def test_llm_only_api_stage1_processes_every_text_unit(tmp_path):
    text_units = tmp_path / "units.csv"
    keyword = tmp_path / "keywords.csv"
    output = tmp_path / "llm_stage1.csv"
    write_units(text_units)
    pd.DataFrame(
        [
            {"text_unit_id": "id1", "keyword_candidate": True, "matched_terms": "ISO"},
            {"text_unit_id": "id2", "keyword_candidate": False, "matched_terms": ""},
            {"text_unit_id": "id3", "keyword_candidate": False, "matched_terms": ""},
        ]
    ).to_csv(keyword, index=False, encoding="utf-8-sig")
    client = RelatedClient()

    result = run_stage1_screening(
        text_units,
        output,
        client,
        "prompt",
        ExtractSettings(workers=1, max_retries=1),
        keyword_features_csv=None,
        resume=False,
    )

    assert client.calls == 3
    assert set(result["stage1_status"]) == {"OK"}
    assert "SKIPPED_KEYWORD" not in result["stage1_status"].tolist()


def test_llm_only_vllm_stage1_processes_every_text_unit(tmp_path, monkeypatch):
    text_units = tmp_path / "units.csv"
    output = tmp_path / "llm_stage1.csv"
    write_units(text_units)
    seen = []

    monkeypatch.setattr("standard_pipeline.vllm_batch.build_vllm_engine", lambda config: (object(), object()))

    def generate(llm, tokenizer, rows, system_prompt, user_content_builder, config):
        seen.extend(row["text_unit_id"] for row in rows)
        return [
            json.dumps({"relevance": "related", "confidence_score": 0.8, "reason": "ok"})
            for _ in rows
        ]

    monkeypatch.setattr("standard_pipeline.vllm_batch.generate_raw_batch", generate)
    result = run_stage1_screening_vllm_batch(
        text_units,
        output,
        "prompt",
        VLLMBatchConfig(model_path="fake", chunk_size=2),
        keyword_features_csv=None,
        resume=False,
    )

    assert seen == ["id1", "id2", "id3"]
    assert set(result["stage1_status"]) == {"OK"}


def test_llm_only_routes_only_related_and_uncertain_without_keywords(tmp_path):
    text_units = tmp_path / "units.csv"
    stage1 = tmp_path / "stage1.csv"
    output = tmp_path / "llm_units.csv"
    write_units(text_units)
    pd.DataFrame(
        [
            {"text_unit_id": "id1", "relevance": "related", "confidence_score": 0.9, "reason": "", "stage1_status": "OK", "stage1_error": ""},
            {"text_unit_id": "id2", "relevance": "uncertain", "confidence_score": 0.5, "reason": "", "stage1_status": "OK", "stage1_error": ""},
            {"text_unit_id": "id3", "relevance": "unrelated", "confidence_score": 0.9, "reason": "", "stage1_status": "OK", "stage1_error": ""},
        ],
        columns=STAGE1_RELEVANCE_COLUMNS,
    ).to_csv(stage1, index=False, encoding="utf-8-sig")

    result = run_prepare_llm_units(text_units, stage1, output)

    assert result["text_unit_id"].tolist() == ["id1", "id2"]
    assert result.columns.tolist() == TEXT_UNIT_COLUMNS


def test_llm_only_manifest_uses_independent_stage1_and_counts_errors(tmp_path):
    paths = default_robustness_paths(tmp_path, "robustness_llm_only", "2024")
    paths.ensure_dirs()
    source_units = tmp_path / "all_units.csv"
    source_main_stage2 = tmp_path / "main_stage2.csv"
    write_units(source_units)
    pd.DataFrame(
        [
            {"text_unit_id": "id1", "relevance": "related", "confidence_score": 0.9, "reason": "", "stage1_status": "OK", "stage1_error": ""},
            {"text_unit_id": "id2", "relevance": "uncertain", "confidence_score": 0.5, "reason": "", "stage1_status": "OK", "stage1_error": ""},
            {"text_unit_id": "id3", "relevance": "", "confidence_score": "", "reason": "", "stage1_status": "ERROR", "stage1_error": "bad"},
        ],
        columns=STAGE1_RELEVANCE_COLUMNS,
    ).to_csv(paths.stage1_relevance_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(unit_rows(("id1", "id2")), columns=TEXT_UNIT_COLUMNS).to_csv(paths.units_path, index=False)
    stage2_rows = [
        {"text_unit_id": text_unit_id, "stock_code": f"00000{index}", "company_name": f"C{index}", "year": "2024", "entity": "无", "type": "TYPE_D", "status": "NO", "evidence": "none"}
        for index, text_unit_id in enumerate(("id1", "id2"), start=1)
    ]
    pd.DataFrame(stage2_rows, columns=TEXT_UNIT_EXTRACTION_COLUMNS).to_csv(paths.stage2_result_path, index=False)
    pd.DataFrame(stage2_rows, columns=TEXT_UNIT_EXTRACTION_COLUMNS).to_csv(source_main_stage2, index=False)
    pd.DataFrame(stage2_rows).assign(**{"国际标准": "", "采标情况": "TYPE_D", "output": 0}).to_csv(paths.mapped_result_path, index=False)
    pd.DataFrame([{"stock_code": "000001", "company_name": "C1", "year": "2024", "InternationalStandardDummy": 0, "AdoptedEntityCount": 0}]).to_csv(paths.final_output_path, index=False)
    raw_log = paths.stage1_raw_failure_log_path("api")
    raw_log.touch()
    failure_queue = paths.stage2_failure_queue_path("api")
    failure_queue.touch()

    manifest = write_robustness_manifest(
        paths,
        source_text_units_path=source_units,
        source_keyword_features_path=None,
        source_stage1_relevance_path=paths.stage1_relevance_path,
        source_stage2_result_path=source_main_stage2,
        prompt_stage1_path=tmp_path / "stage1_prompt.txt",
        prompt_stage2_path=tmp_path / "stage2_prompt.txt",
        gb_mapping_path=tmp_path / "gb.csv",
        model_stage1="fake-stage1",
        model_stage2="fake-stage2",
        stage1_raw_failure_log_path=raw_log,
        stage2_failure_queue_path=failure_queue,
    )

    assert manifest["output_paths"]["stage1_relevance_path"] == str(paths.stage1_relevance_path)
    assert "source_stage1_relevance_path" not in manifest["input_paths"]
    assert manifest["N_llm_related"] == 1
    assert manifest["N_llm_uncertain"] == 1
    assert manifest["N_stage1_failed"] == 1
    assert manifest["complete"] is False


def test_llm_only_no_resume_cleanup_is_isolated(tmp_path):
    paths = default_robustness_paths(tmp_path, "robustness_llm_only", "2024")
    paths.ensure_dirs()
    owned = [
        paths.stage1_relevance_path,
        paths.stage1_raw_failure_log_path("api"),
        paths.stage2_result_path,
        paths.stage2_log_path("api"),
        paths.stage2_failure_queue_path("api"),
    ]
    for path in owned:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("owned", encoding="utf-8")
    main_file = tmp_path / "data" / "measurement" / "main_regression" / "keep.csv"
    other_file = tmp_path / "data" / "measurement" / "robustness_keyword" / "keep.csv"
    for path in (main_file, other_file):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("keep", encoding="utf-8")

    cli.remove_robustness_resume_files(paths, "api", "api")

    assert all(not path.exists() for path in owned)
    assert main_file.exists()
    assert other_file.exists()


def test_api_stage2_failure_is_queued_and_retried_on_resume(tmp_path):
    input_csv = tmp_path / "input.csv"
    output = tmp_path / "output.csv"
    log = tmp_path / "completed.log"
    queue = tmp_path / "failures.jsonl"
    write_units(input_csv, ("id1",), audit=True)
    failing = Stage2Client(failures=("id1",))

    first = run_text_unit_extraction(
        input_csv, output, log, queue, failing, "prompt",
        ExtractSettings(workers=1, batch_size=1, max_retries=2, retry_min_seconds=0, retry_max_seconds=0),
        resume=False,
    )
    record = json.loads(queue.read_text(encoding="utf-8").splitlines()[0])

    assert first.empty
    assert log.read_text(encoding="utf-8") == ""
    assert record["text_unit_id"] == "id1"
    assert record["attempt"] == 2
    assert record["error_type"] == "RuntimeError"
    assert record["retries_exhausted"] is True

    succeeding = Stage2Client()
    second = run_text_unit_extraction(
        input_csv, output, log, queue, succeeding, "prompt",
        ExtractSettings(workers=1, batch_size=1, max_retries=1),
        resume=True,
    )

    assert succeeding.calls == ["id1"]
    assert second["text_unit_id"].tolist() == ["id1"]
    assert log.read_text(encoding="utf-8").splitlines() == ["id1"]
    assert count_stage2_status(output, queue) == (1, 0)


@pytest.mark.parametrize("mode", ["legacy_sentinel", "log_only"])
def test_stage2_resume_reconciles_completed_log_with_success_output(tmp_path, mode):
    input_csv = tmp_path / "input.csv"
    output = tmp_path / "output.csv"
    log = tmp_path / "completed.log"
    queue = tmp_path / "failures.jsonl"
    write_units(input_csv, ("id1",))
    if mode == "legacy_sentinel":
        pd.DataFrame(
            [{"text_unit_id": "id1", "stock_code": "000001", "company_name": "C1", "year": "2024", "entity": "ERROR", "type": "LLM_FAILURE", "status": "FAIL", "evidence": "old"}],
            columns=TEXT_UNIT_EXTRACTION_COLUMNS,
        ).to_csv(output, index=False)
    else:
        pd.DataFrame(columns=TEXT_UNIT_EXTRACTION_COLUMNS).to_csv(output, index=False)
    log.write_text("id1\n", encoding="utf-8")
    client = Stage2Client()

    result = run_text_unit_extraction(
        input_csv, output, log, queue, client, "prompt",
        ExtractSettings(workers=1, max_retries=1),
        resume=True,
    )

    assert client.calls == ["id1"]
    assert result["entity"].tolist() == ["ISO 9001"]
    assert "ERROR" not in result["entity"].tolist()


def test_vllm_stage2_accepts_route_audit_fields_and_queues_parse_failure(tmp_path, monkeypatch):
    input_csv = tmp_path / "input.csv"
    output = tmp_path / "output.csv"
    log = tmp_path / "completed.log"
    queue = tmp_path / "failures.jsonl"
    write_units(input_csv, ("id1",), audit=True)
    monkeypatch.setattr("standard_pipeline.vllm_batch.build_vllm_engine", lambda config: (object(), object()))
    monkeypatch.setattr("standard_pipeline.vllm_batch.generate_raw_batch", lambda *args, **kwargs: ["not json"])

    result = run_text_unit_extraction_vllm_batch(
        input_csv, output, log, queue, "prompt", VLLMBatchConfig(model_path="fake"), resume=False
    )

    assert result.empty
    assert log.read_text(encoding="utf-8") == ""
    assert json.loads(queue.read_text(encoding="utf-8").splitlines()[0])["text_unit_id"] == "id1"


def test_cli_exposes_only_formal_workflow_commands(capsys):
    with pytest.raises(SystemExit) as help_exit:
        cli.parse_args(["--help"])
    assert help_exit.value.code == 0
    help_text = capsys.readouterr().out
    choices = set(re.search(r"\{([^}]+)\}", help_text, re.DOTALL).group(1).replace("\n", "").split(","))
    retained = {
        "build-text-units", "audit-text-units", "stage1-screen", "route-main",
        "stage2-extract", "map-main-gb", "aggregate-main", "main-regression",
        "robustness-keyword", "robustness-llm-only", "robustness-full-llm",
        "compare-measurements",
    }
    assert choices == retained
    for removed in ("preprocess", "extract", "map-gb", "run-all"):
        with pytest.raises(SystemExit) as invalid:
            cli.parse_args([removed])
        assert invalid.value.code == 2
    for command in retained:
        assert cli.parse_args([command, "--year", "2024"]).command == command


def test_minimal_pipeline_failure_then_resume_restores_manifest(tmp_path):
    text_units = tmp_path / "text_units.csv"
    stage2_input = tmp_path / "stage2_input.csv"
    stage2_output = tmp_path / "stage2_output.csv"
    completed = tmp_path / "completed.log"
    queue = tmp_path / "failures.jsonl"
    mapping = tmp_path / "gb.csv"
    mapped = tmp_path / "mapped.csv"
    final = tmp_path / "firm_year.csv"
    manifest_path = tmp_path / "manifest.json"
    write_units(text_units)
    write_units(stage2_input, audit=True)
    pd.DataFrame([{"标准号": "GB/T 19001-2016", "国际标准编号": "ISO 9001", "采标类型": "等同采用"}]).to_csv(mapping, index=False)

    first_client = Stage2Client(failures=("id3",))
    run_text_unit_extraction(
        stage2_input, stage2_output, completed, queue, first_client, "prompt",
        ExtractSettings(workers=1, batch_size=1, max_retries=1), resume=False,
    )
    run_gb_mapping(stage2_output, mapping, mapped)
    first_final = run_aggregate_main(text_units, mapped, final)
    first_stage2 = pd.read_csv(stage2_output, dtype={"text_unit_id": str})
    first_manifest = write_measurement_manifest(
        manifest_path,
        method="integration",
        year="2024",
        input_paths={"text_units": text_units},
        output_paths={"stage2": stage2_output, "failure_queue": queue},
        prompt_stage1_path=None,
        prompt_stage2_path=None,
        gb_mapping_path=mapping,
        model_stage1=None,
        model_stage2="fake",
        text_units_path=text_units,
        stage2_input_path=stage2_input,
        stage2_result_path=stage2_output,
        stage2_failure_queue_path=queue,
        final_output_path=final,
        required_paths=[text_units, stage2_input, stage2_output, mapped, final],
    )

    assert first_final.set_index("stock_code").loc["000001", "InternationalStandardDummy"] == 1
    assert first_final.set_index("stock_code").loc["000002", "InternationalStandardDummy"] == 0
    assert first_stage2.set_index("text_unit_id").loc["id2", "status"] == "NO"
    assert completed.read_text(encoding="utf-8").splitlines() == ["id1", "id2"]
    assert first_manifest["complete"] is False
    assert first_manifest["N_stage2_completed"] == 2
    assert first_manifest["N_stage2_failed"] == 1

    second_client = Stage2Client()
    run_text_unit_extraction(
        stage2_input, stage2_output, completed, queue, second_client, "prompt",
        ExtractSettings(workers=1, batch_size=1, max_retries=1), resume=True,
    )
    run_gb_mapping(stage2_output, mapping, mapped)
    run_aggregate_main(text_units, mapped, final)
    second_manifest = write_measurement_manifest(
        manifest_path,
        method="integration",
        year="2024",
        input_paths={"text_units": text_units},
        output_paths={"stage2": stage2_output, "failure_queue": queue},
        prompt_stage1_path=None,
        prompt_stage2_path=None,
        gb_mapping_path=mapping,
        model_stage1=None,
        model_stage2="fake",
        text_units_path=text_units,
        stage2_input_path=stage2_input,
        stage2_result_path=stage2_output,
        stage2_failure_queue_path=queue,
        final_output_path=final,
        required_paths=[text_units, stage2_input, stage2_output, mapped, final],
    )

    assert second_client.calls == ["id3"]
    assert second_manifest["complete"] is True
    assert second_manifest["N_stage2_completed"] == 3
    assert second_manifest["N_stage2_failed"] == 0
    assert second_manifest["N_stage2_pending"] == 0

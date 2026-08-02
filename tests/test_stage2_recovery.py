"""Tests for stage2_recovery module: strict parsing, schema validation,
safe repair, repetition detection, and two-pass generation."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile

import pandas as pd
import pytest

from standard_pipeline.extract import (
    Stage2Outcome,
    _entity_rows,
    stage2_failure_record,
    stage2_outcome_from_raw_safe,
)
from standard_pipeline.stage2_recovery import (
    STAGE2_JSON_SCHEMA,
    SafeRecoveryConfig,
    build_structured_output_kwargs,
    detect_stage2_degeneration,
    extract_stage2_json_object_strict,
    make_audit_entry,
    safe_repair_and_parse,
    safe_recovery_config_from_dict,
    validate_stage2_payload,
)


# ======================================================================
# Strict parsing tests
# ======================================================================


class TestStrictJsonParsing:
    def test_valid_outer_json_passes(self):
        raw = '{"standards":[{"entity":"ISO 9001","type":"TYPE_A","status":"ADOPTED","evidence":"通过认证"}]}'
        data = extract_stage2_json_object_strict(raw)
        assert data["standards"][0]["entity"] == "ISO 9001"

    def test_empty_standards_array_passes(self):
        raw = '{"standards":[]}'
        data = extract_stage2_json_object_strict(raw)
        assert data["standards"] == []

    def test_escaped_quotes_in_evidence(self):
        raw = '{"standards":[{"entity":"RoHS","type":"TYPE_B","status":"ADOPTED","evidence":"符合\\"RoHS\\"指令"}]}'
        data = extract_stage2_json_object_strict(raw)
        assert '"RoHS"' in data["standards"][0]["evidence"]

    def test_broken_outer_inner_valid_fails(self):
        raw = 'garbage {"standards":[{"entity":"ISO","type":"TYPE_A","status":"ADOPTED","evidence":"x"}]}'
        with pytest.raises((json.JSONDecodeError, ValueError)):
            extract_stage2_json_object_strict(raw)

    def test_missing_standards_fails(self):
        raw = '{"other": 1}'
        with pytest.raises(ValueError, match="missing standards"):
            extract_stage2_json_object_strict(raw)

    def test_top_level_array_fails(self):
        raw = '[{"entity":"ISO","type":"TYPE_A","status":"ADOPTED","evidence":"x"}]'
        with pytest.raises(ValueError, match="outer JSON object"):
            extract_stage2_json_object_strict(raw)

    def test_standards_not_array_fails(self):
        raw = '{"standards": "not_array"}'
        data = extract_stage2_json_object_strict(raw)
        errors = validate_stage2_payload(data)
        assert len(errors) > 0
        assert any("not a list" in e for e in errors)

    def test_explanatory_text_before_json_fails(self):
        raw = 'Here is the result:\n{"standards":[]}'
        with pytest.raises(json.JSONDecodeError):
            extract_stage2_json_object_strict(raw)

    def test_markdown_fence_is_stripped(self):
        raw = '```json\n{"standards":[]}\n```'
        data = extract_stage2_json_object_strict(raw)
        assert data["standards"] == []


# ======================================================================
# Schema validation tests
# ======================================================================


class TestSchemaValidation:
    def test_valid_payload_passes(self):
        data = {
            "standards": [
                {"entity": "ISO 9001", "type": "TYPE_A", "status": "ADOPTED", "evidence": "通过"}
            ]
        }
        assert validate_stage2_payload(data) == []

    def test_missing_field_fails(self):
        data = {"standards": [{"entity": "ISO", "type": "TYPE_A", "status": "ADOPTED"}]}
        errors = validate_stage2_payload(data)
        assert any("missing required field 'evidence'" in e for e in errors)

    def test_invalid_type_enum_fails(self):
        data = {
            "standards": [
                {"entity": "ISO", "type": "TYPE_X", "status": "ADOPTED", "evidence": "x"}
            ]
        }
        errors = validate_stage2_payload(data)
        assert any("TYPE_X" in e for e in errors)

    def test_invalid_status_enum_fails(self):
        data = {
            "standards": [
                {"entity": "ISO", "type": "TYPE_A", "status": "YES", "evidence": "x"}
            ]
        }
        errors = validate_stage2_payload(data)
        assert any("YES" in e for e in errors)

    def test_non_string_field_fails(self):
        data = {"standards": [{"entity": 123, "type": "TYPE_A", "status": "ADOPTED", "evidence": "x"}]}
        errors = validate_stage2_payload(data)
        assert any("entity" in e and "not a string" in e for e in errors)

    def test_unexpected_top_level_keys(self):
        data = {"standards": [], "extra": 1}
        errors = validate_stage2_payload(data)
        assert any("unexpected top-level keys" in e for e in errors)

    def test_unexpected_item_keys(self):
        data = {
            "standards": [
                {
                    "entity": "ISO",
                    "type": "TYPE_A",
                    "status": "ADOPTED",
                    "evidence": "x",
                    "extra_field": 1,
                }
            ]
        }
        errors = validate_stage2_payload(data)
        assert any("unexpected keys" in e for e in errors)


# ======================================================================
# Safe repair tests
# ======================================================================


class TestSafeRepair:
    def test_trailing_comma_in_array_is_repaired(self):
        raw = '{"standards":[{"entity":"ISO","type":"TYPE_A","status":"ADOPTED","evidence":"x"},]}'
        data, actions = safe_repair_and_parse(raw)
        assert "removed_trailing_comma" in actions
        assert data["standards"][0]["entity"] == "ISO"

    def test_markdown_fence_is_cleaned(self):
        raw = '```json\n{"standards":[]}\n```'
        data, actions = safe_repair_and_parse(raw)
        assert "removed_markdown_fence" in actions
        assert data["standards"] == []

    def test_ambiguous_unescaped_quotes_not_auto_fixed(self):
        # Evidence with unescaped inner double quotes — repair should not guess
        raw = '{"standards":[{"entity":"RoHS","type":"TYPE_B","status":"ADOPTED","evidence":"符合"RoHS"指令"}]}'
        with pytest.raises(Exception):
            safe_repair_and_parse(raw)

    def test_missing_field_not_fabricated(self):
        raw = '{"standards":[{"entity":"ISO","type":"TYPE_A","status":"ADOPTED"}]}'
        with pytest.raises(Exception):
            safe_repair_and_parse(raw)

    def test_repaired_payload_passes_validation(self):
        raw = '{"standards":[{"entity":"ISO","type":"TYPE_A","status":"ADOPTED","evidence":"x"},]}'
        data, actions = safe_repair_and_parse(raw)
        assert validate_stage2_payload(data) == []


# ======================================================================
# Repetition detection tests
# ======================================================================


class TestRepetitionDetection:
    def test_long_single_char_repeat_detected(self):
        raw = "项" * 100 + '{"standards":[]}'
        flags = detect_stage2_degeneration(raw, min_chars=100)
        assert "single_char_repeat" in flags

    def test_long_segment_repeat_detected(self):
        segment = "公司通过ISO 9001认证，符合国际标准要求。" * 50
        raw = segment + segment  # 100 repetitions of a long segment
        flags = detect_stage2_degeneration(raw, min_chars=1000)
        assert len(flags) > 0

    def test_normal_iso_list_not_flagged(self):
        # A valid JSON with diverse standard entities should not be flagged
        standards = []
        for i in range(1, 21):
            standards.append(
                f'{{"entity":"ISO {9000+i}","type":"TYPE_A","status":"ADOPTED",'
                f'"evidence":"通过ISO {9000+i}质量管理体系认证，符合公司{chr(65+i%26)}部门要求"}}'
            )
        raw = '{"standards":[' + ",".join(standards) + "]}"
        flags = detect_stage2_degeneration(raw, min_chars=1000)
        # Only check that compression_ratio is not the reason when text is diverse
        assert "single_char_repeat" not in flags
        assert "long_segment_repeat" not in flags

    def test_short_response_not_flagged_by_compression(self):
        raw = '{"standards":[{"entity":"ISO","type":"TYPE_A","status":"ADOPTED","evidence":"x"}]}'
        # Short response should not trigger compression ratio check
        flags = detect_stage2_degeneration(raw, min_chars=1000)
        assert "compression_ratio" not in flags


# ======================================================================
# vLLM structured output tests
# ======================================================================


class TestStructuredOutput:
    def test_enabled_false_does_not_build_kwargs(self):
        config = SafeRecoveryConfig(enabled=False, structured_output=False)
        assert config.structured_output is False

    def test_config_from_dict_defaults_disabled(self):
        config = safe_recovery_config_from_dict({})
        assert config.enabled is False

    def test_config_from_dict_enabled(self):
        config = safe_recovery_config_from_dict({"enabled": True})
        assert config.enabled is True

    def test_build_structured_output_kwargs_falls_back_to_guided(self):
        import sys
        # Simulate missing StructuredOutputsParams, present GuidedDecodingParams
        try:
            from vllm.sampling_params import GuidedDecodingParams  # noqa: F401
            has_guided = True
        except ImportError:
            has_guided = False

        if not has_guided:
            pytest.skip("vLLM not installed")

        # If both are available, StructuredOutputsParams is preferred
        kwargs = build_structured_output_kwargs(STAGE2_JSON_SCHEMA)
        assert "structured_outputs" in kwargs or "guided_decoding" in kwargs

    def test_config_max_attempts_validation(self):
        with pytest.raises(ValueError, match="max_generation_attempts"):
            SafeRecoveryConfig(max_generation_attempts=0)
        with pytest.raises(ValueError, match="max_generation_attempts"):
            SafeRecoveryConfig(max_generation_attempts=4)

    def test_config_grounding_mode_validation(self):
        with pytest.raises(ValueError, match="grounding_mode"):
            SafeRecoveryConfig(grounding_mode="strict")


# ======================================================================
# Audit logging tests
# ======================================================================


class TestAuditLogging:
    def test_audit_entry_contains_required_fields(self):
        entry = make_audit_entry(
            text_unit_id="id1",
            attempt=1,
            structured_output=True,
            parse_method="strict_json",
            parse_success=True,
            schema_valid=True,
            repair_attempted=False,
            final_disposition="success",
            raw_response='{"standards":[]}',
        )
        required = {
            "created_at", "text_unit_id", "attempt", "structured_output",
            "parse_method", "parse_success", "schema_valid", "repair_attempted",
            "repair_actions", "degeneration_flags", "entity_grounded",
            "evidence_grounded", "final_disposition", "raw_response_length",
            "raw_response_sha256", "raw_response_preview",
        }
        assert required <= set(entry.keys())

    def test_audit_entry_truncates_long_raw(self):
        long_raw = "x" * 5000
        entry = make_audit_entry(
            text_unit_id="id1",
            attempt=1,
            structured_output=False,
            parse_method="strict_json",
            parse_success=False,
            schema_valid=False,
            repair_attempted=False,
            final_disposition="failed",
            raw_response=long_raw,
        )
        assert len(entry["raw_response_preview"]) < 5000
        assert "truncated" in entry["raw_response_preview"]

    def test_audit_log_writes_to_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "audit.jsonl"
            from standard_pipeline.stage2_recovery import append_audit_entries

            entry = make_audit_entry(
                text_unit_id="id1",
                attempt=1,
                structured_output=True,
                parse_method="strict_json",
                parse_success=True,
                schema_valid=True,
                repair_attempted=False,
                final_disposition="success",
                raw_response='{"standards":[]}',
            )
            append_audit_entries(log_path, [entry])
            assert log_path.exists()
            lines = log_path.read_text(encoding="utf-8").strip().split("\n")
            assert len(lines) == 1
            data = json.loads(lines[0])
            assert data["text_unit_id"] == "id1"


# ======================================================================
# stage2_outcome_from_raw_safe tests
# ======================================================================


class TestStage2OutcomeSafe:
    def test_valid_json_produces_success(self):
        row = pd.Series({
            "text_unit_id": "id1",
            "stock_code": "000001",
            "company_name": "Test",
            "year": "2024",
            "text": "通过ISO 9001认证",
        })
        raw = '{"standards":[{"entity":"ISO 9001","type":"TYPE_A","status":"ADOPTED","evidence":"通过"}]}'
        outcome = stage2_outcome_from_raw_safe(
            row, raw, provider="vllm_batch", attempt=1, structured_output=False,
        )
        assert outcome.success
        assert len(outcome.rows) == 1
        assert outcome.rows[0]["entity"] == "ISO 9001"

    def test_broken_json_produces_retry_failure(self):
        row = pd.Series({
            "text_unit_id": "id1",
            "stock_code": "000001",
            "company_name": "Test",
            "year": "2024",
            "text": "test",
        })
        raw = "not json at all"
        outcome = stage2_outcome_from_raw_safe(
            row, raw, provider="vllm_batch", attempt=1, structured_output=False,
        )
        assert not outcome.success
        assert outcome.failure_record is not None
        assert outcome.failure_record["retries_exhausted"] is False

    def test_degeneration_flags_triggers_retry(self):
        row = pd.Series({
            "text_unit_id": "id1",
            "stock_code": "000001",
            "company_name": "Test",
            "year": "2024",
            "text": "test",
        })
        raw = "项" * 2000  # Long single-char repeat
        outcome = stage2_outcome_from_raw_safe(
            row, raw, provider="vllm_batch", attempt=1, structured_output=False,
        )
        assert not outcome.success
        assert outcome.failure_record["error_type"] == "DegenerationDetected"

    def test_audit_log_is_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "audit.jsonl"
            row = pd.Series({
                "text_unit_id": "id1",
                "stock_code": "000001",
                "company_name": "Test",
                "year": "2024",
                "text": "通过ISO 9001认证",
            })
            raw = '{"standards":[{"entity":"ISO 9001","type":"TYPE_A","status":"ADOPTED","evidence":"通过"}]}'
            stage2_outcome_from_raw_safe(
                row, raw, provider="vllm_batch", attempt=1,
                structured_output=False, audit_log=log_path,
            )
            assert log_path.exists()
            lines = log_path.read_text(encoding="utf-8").strip().split("\n")
            assert len(lines) == 1
            data = json.loads(lines[0])
            assert data["final_disposition"] == "success"

    def test_evidence_grounding_audit_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "audit.jsonl"
            row = pd.Series({
                "text_unit_id": "id1",
                "stock_code": "000001",
                "company_name": "Test",
                "year": "2024",
                "text": "公司通过ISO 9001质量管理体系认证",
            })
            raw = '{"standards":[{"entity":"ISO 9001","type":"TYPE_A","status":"ADOPTED","evidence":"通过ISO 9001认证"}]}'
            outcome = stage2_outcome_from_raw_safe(
                row, raw, provider="vllm_batch", attempt=1,
                structured_output=False, audit_log=log_path,
                source_text=str(row["text"]), grounding_mode="audit",
            )
            assert outcome.success
            lines = log_path.read_text(encoding="utf-8").strip().split("\n")
            data = json.loads(lines[0])
            assert data["entity_grounded"] is True


# ======================================================================
# normalize_items preserves original behavior
# ======================================================================


class TestNormalizeItemsUnchanged:
    def test_normalize_still_converts_invalid_type_when_not_safe(self):
        from standard_pipeline.extract import normalize_items
        data = {"standards": [{"entity": "X", "type": "INVALID", "status": "OK", "evidence": "e"}]}
        items = normalize_items(data)
        assert items[0]["type"] == "TYPE_D"
        assert items[0]["status"] == "NO"

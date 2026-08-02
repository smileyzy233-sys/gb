"""Stage2 Safe Recovery — strict JSON parsing, schema validation, safe repair,
repetition detection, two-pass generation, audit logging, and evidence grounding.

All behaviour is gated behind ``[stage2.safe_recovery] enabled = true``.
When disabled the original pipeline path is unchanged.
"""

from __future__ import annotations

import hashlib
import json
import re
import zlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# JSON Schema for structured output (vLLM Guided / Structured decoding)
# ---------------------------------------------------------------------------

STAGE2_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "standards": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "entity": {"type": "string"},
                    "type": {
                        "type": "string",
                        "enum": ["TYPE_A", "TYPE_B", "TYPE_C", "TYPE_D"],
                    },
                    "status": {
                        "type": "string",
                        "enum": ["ADOPTED", "PENDING", "NO"],
                    },
                    "evidence": {"type": "string"},
                },
                "required": ["entity", "type", "status", "evidence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["standards"],
    "additionalProperties": False,
}

VALID_TYPES = {"TYPE_A", "TYPE_B", "TYPE_C", "TYPE_D"}
VALID_STATUSES = {"ADOPTED", "PENDING", "NO"}

# ---------------------------------------------------------------------------
# Repetition / degeneration detection
# ---------------------------------------------------------------------------

REPETITION_MIN_CHARS = 1000
REPETITION_COMPRESSION_RATIO = 0.20

# Simple CJK single-char repeat (≥30 consecutive same char)
_SINGLE_CHAR_REPEAT_RE = re.compile(r"([\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff])\1{29,}")

# Repeated sentence segment (≥80 chars repeated ≥5 times)
_LONG_SEGMENT_REPEAT_RE = re.compile(r"(.{80,}?)\1{4,}")


def detect_stage2_degeneration(
    raw: str,
    *,
    min_chars: int = REPETITION_MIN_CHARS,
    compression_ratio: float = REPETITION_COMPRESSION_RATIO,
) -> list[str]:
    """Return a list of degeneration flag strings; empty list means OK."""
    flags: list[str] = []

    encoded = raw.encode("utf-8")
    raw_len = len(encoded)

    if raw_len >= min_chars:
        compressed = zlib.compress(encoded, level=6)
        actual_ratio = len(compressed) / raw_len
        if actual_ratio < compression_ratio:
            flags.append(f"compression_ratio_{actual_ratio:.3f}")

    if _SINGLE_CHAR_REPEAT_RE.search(raw):
        flags.append("single_char_repeat")

    if _LONG_SEGMENT_REPEAT_RE.search(raw):
        flags.append("long_segment_repeat")

    return flags


# ---------------------------------------------------------------------------
# Strict JSON parsing (Stage2-only, does NOT affect extract_json_object)
# ---------------------------------------------------------------------------

_MARKDOWN_FENCE_RE = re.compile(r"^```(?:json)?\s*\n(.*?)\n```\s*$", re.DOTALL)


def _strip_markdown_fence(text: str) -> str:
    """Remove a single outer Markdown JSON fence if it wraps the entire text."""
    m = _MARKDOWN_FENCE_RE.match(text.strip())
    if m:
        return m.group(1).strip()
    # Also try simpler removal (from clean_json_string)
    text = re.sub(r"^```json\s*", "", text.strip(), flags=re.MULTILINE)
    text = re.sub(r"^```\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"```$", "", text, flags=re.MULTILINE)
    return text.strip()


def extract_stage2_json_object_strict(raw: str) -> dict[str, Any]:
    """Parse the *entire* raw string as a single JSON object.

    Unlike ``extract_json_object`` this does NOT scan for inner ``{``.
    The entire string (after fence cleanup) must be valid JSON.

    Raises ``json.JSONDecodeError`` preserving line/col/offset.
    """
    text = _strip_markdown_fence(raw)
    if not text:
        raise json.JSONDecodeError("empty string after cleanup", "", 0)
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Stage2 response must be one outer JSON object")
    if "standards" not in data:
        raise ValueError("Stage2 outer JSON object is missing standards")
    return data


# ---------------------------------------------------------------------------
# Strict payload validation
# ---------------------------------------------------------------------------


def validate_stage2_payload(data: dict[str, Any]) -> list[str]:
    """Validate a parsed Stage2 response strictly.

    Returns a list of error message strings; empty list = valid.
    """
    errors: list[str] = []

    if not isinstance(data, dict):
        errors.append("top-level is not a dict")
        return errors

    extra_keys = set(data.keys()) - {"standards"}
    if extra_keys:
        errors.append(f"unexpected top-level keys: {sorted(extra_keys)}")

    standards = data.get("standards")
    if standards is None:
        errors.append("missing 'standards' key")
        return errors

    if not isinstance(standards, list):
        errors.append(f"'standards' is not a list, got {type(standards).__name__}")
        return errors

    for idx, item in enumerate(standards):
        if not isinstance(item, dict):
            errors.append(f"standards[{idx}] is not a dict, got {type(item).__name__}")
            continue

        # Check for unexpected keys
        item_extra = set(item.keys()) - {"entity", "type", "status", "evidence"}
        if item_extra:
            errors.append(f"standards[{idx}] has unexpected keys: {sorted(item_extra)}")

        for field in ("entity", "type", "status", "evidence"):
            if field not in item:
                errors.append(f"standards[{idx}] missing required field '{field}'")
            elif not isinstance(item[field], str):
                errors.append(
                    f"standards[{idx}].{field} is not a string, got {type(item[field]).__name__}"
                )

        if "type" in item and isinstance(item["type"], str) and item["type"] not in VALID_TYPES:
            errors.append(f"standards[{idx}].type='{item['type']}' not in {sorted(VALID_TYPES)}")

        if "status" in item and isinstance(item["status"], str) and item["status"] not in VALID_STATUSES:
            errors.append(
                f"standards[{idx}].status='{item['status']}' not in {sorted(VALID_STATUSES)}"
            )

    return errors


# ---------------------------------------------------------------------------
# Conservative safe repair (single pass, deterministic only)
# ---------------------------------------------------------------------------

_TRAILING_COMMA_BEFORE_BRACKET = re.compile(r",(\s*[}\]])")


def _safe_repair_json(raw: str) -> tuple[str, list[str]]:
    """Apply one pass of deterministic, semantics-preserving repairs.

    Returns ``(repaired_text, actions)``.
    """
    text = raw
    actions: list[str] = []

    # 1. Strip markdown fences
    stripped = _strip_markdown_fence(text)
    if stripped != text:
        actions.append("removed_markdown_fence")
        text = stripped

    # 2. Remove trailing commas before } or ]
    repaired = _TRAILING_COMMA_BEFORE_BRACKET.sub(r"\1", text)
    if repaired != text:
        actions.append("removed_trailing_comma")
        text = repaired

    # 3. Normalize fullwidth punctuation that commonly appears in structural positions
    #    Only at structural positions (colon after key-like pattern, comma in array/object)
    #    This is conservative: we only fix clear structural cases.
    _fullwidth_fixed = text
    # Fullwidth colon after a quoted key or simple key
    _fullwidth_fixed = re.sub(
        r'("\s*)\uff1a(\s*")', r"\1:\2", _fullwidth_fixed
    )
    # Fullwidth comma in JSON structural positions (after ", after ], after })
    _fullwidth_fixed = re.sub(
        r'(["\}\]\d])\s*\uff0c\s*([\["\{\-\d])', r"\1,\2", _fullwidth_fixed
    )
    if _fullwidth_fixed != text:
        actions.append("normalized_fullwidth_punctuation")
        text = _fullwidth_fixed

    return text, actions


def safe_repair_and_parse(raw: str) -> tuple[dict[str, Any], list[str]]:
    """Try to repair and parse a Stage2 response.

    Returns ``(parsed_dict, repair_actions)``.
    Raises the original exception if repair fails or validation fails.
    """
    repaired, actions = _safe_repair_json(raw)
    try:
        data = extract_stage2_json_object_strict(repaired)
    except Exception:
        # If repair didn't change anything, re-raise original error
        if not actions:
            raise
        # Try original extraction as last resort
        try:
            data = extract_stage2_json_object_strict(raw)
            actions = []
        except Exception:
            raise
    errors = validate_stage2_payload(data)
    if errors:
        raise ValueError(f"Stage2 schema validation failed after repair: {errors}")
    return data, actions


# ---------------------------------------------------------------------------
# vLLM structured-output capability detection
# ---------------------------------------------------------------------------


def build_structured_output_kwargs(
    schema: dict[str, Any],
) -> dict[str, Any]:
    """Detect available vLLM structured-output API and return kwargs for
    ``SamplingParams``.

    Prefers ``StructuredOutputsParams`` (new, vLLM ≥ 0.20), falls back to
    ``GuidedDecodingParams`` (legacy).  Raises ``RuntimeError`` if neither
    is available.
    """
    try:
        from vllm.sampling_params import StructuredOutputsParams  # type: ignore[import-untyped]

        return {
            "structured_outputs": StructuredOutputsParams(
                json=schema,
                disable_additional_properties=True,
            )
        }
    except ImportError:
        pass

    try:
        from vllm.sampling_params import GuidedDecodingParams  # type: ignore[import-untyped]

        return {"guided_decoding": GuidedDecodingParams(json=schema)}
    except ImportError:
        pass

    raise RuntimeError(
        "Configured structured output is unsupported by installed vLLM. "
        "Neither StructuredOutputsParams nor GuidedDecodingParams is available. "
        "Upgrade vLLM or set [stage2.safe_recovery] structured_output = false."
    )


# ---------------------------------------------------------------------------
# Audit logging
# ---------------------------------------------------------------------------

AUDIT_RAW_MAX_CHARS = 4000
AUDIT_PREVIEW_HEAD_CHARS = 2000
AUDIT_PREVIEW_TAIL_CHARS = 1000


@dataclass
class RecoveryAuditEntry:
    created_at: str
    text_unit_id: str
    attempt: int
    structured_output: bool
    parse_method: str
    parse_success: bool
    schema_valid: bool
    repair_attempted: bool
    repair_actions: list[str]
    degeneration_flags: list[str]
    entity_grounded: bool | None
    evidence_grounded: bool | None
    final_disposition: str  # "success", "retry", "failed"
    raw_response_length: int
    raw_response_sha256: str
    raw_response_preview: str
    original_error_type: str | None = None
    parse_error: str | None = None


def _raw_preview(raw: str | None) -> str:
    if raw is None:
        return ""
    head = raw[:AUDIT_PREVIEW_HEAD_CHARS]
    if len(raw) <= AUDIT_PREVIEW_HEAD_CHARS + AUDIT_PREVIEW_TAIL_CHARS:
        return head
    tail = raw[-AUDIT_PREVIEW_TAIL_CHARS:]
    return head + f"\n...[truncated {len(raw) - AUDIT_PREVIEW_HEAD_CHARS - AUDIT_PREVIEW_TAIL_CHARS} chars]...\n" + tail


def make_audit_entry(
    text_unit_id: str,
    attempt: int,
    *,
    structured_output: bool,
    parse_method: str,
    parse_success: bool,
    schema_valid: bool,
    repair_attempted: bool,
    repair_actions: list[str] | None = None,
    degeneration_flags: list[str] | None = None,
    entity_grounded: bool | None = None,
    evidence_grounded: bool | None = None,
    final_disposition: str = "success",
    raw_response: str | None = None,
    original_error_type: str | None = None,
    parse_error: str | None = None,
) -> dict[str, Any]:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "text_unit_id": text_unit_id,
        "attempt": attempt,
        "structured_output": structured_output,
        "parse_method": parse_method,
        "parse_success": parse_success,
        "schema_valid": schema_valid,
        "repair_attempted": repair_attempted,
        "repair_actions": repair_actions or [],
        "degeneration_flags": degeneration_flags or [],
        "entity_grounded": entity_grounded,
        "evidence_grounded": evidence_grounded,
        "final_disposition": final_disposition,
        "raw_response_length": len(raw_response or ""),
        "raw_response_sha256": hashlib.sha256((raw_response or "").encode("utf-8")).hexdigest(),
        "raw_response_preview": _raw_preview(raw_response),
        "original_error_type": original_error_type,
        "parse_error": parse_error,
    }


def append_audit_entries(audit_log: Path | None, entries: list[dict[str, Any]]) -> None:
    if audit_log is None or not entries:
        return
    audit_log.parent.mkdir(parents=True, exist_ok=True)
    with audit_log.open("a", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Evidence grounding (audit-mode: check without auto-rejection)
# ---------------------------------------------------------------------------


def _normalize_for_matching(text: str) -> str:
    """Normalize text for evidence grounding: whitespace, quotes, punctuation."""
    # Collapse all whitespace
    text = re.sub(r"\s+", "", text)
    # Unify quotes
    text = text.replace("\u201c", '"').replace("\u201d", '"')  # Chinese double quotes
    text = text.replace("\u2018", "'").replace("\u2019", "'")  # Chinese single quotes
    text = text.replace("\uff0c", ",").replace("\uff0e", ".")  # Fullwidth comma/period
    text = text.replace("\uff1a", ":").replace("\uff1b", ";")  # Fullwidth colon/semicolon
    text = text.replace("\uff08", "(").replace("\uff09", ")")  # Fullwidth parens
    return text.lower()


def check_evidence_grounding(
    entity: str,
    evidence: str,
    source_text: str,
) -> tuple[bool, bool]:
    """Check if entity and evidence can be found in source text.

    Returns ``(entity_grounded, evidence_grounded)``.
    Uses normalized matching — whitespace/punctuation differences are ignored.
    """
    norm_source = _normalize_for_matching(source_text)
    norm_entity = _normalize_for_matching(entity)
    norm_evidence = _normalize_for_matching(evidence)

    entity_ok = bool(norm_entity) and norm_entity in norm_source
    evidence_ok = bool(norm_evidence) and norm_evidence in norm_source

    return entity_ok, evidence_ok


# ---------------------------------------------------------------------------
# Second-attempt prompt suffix
# ---------------------------------------------------------------------------

RETRY_PROMPT_SUFFIX = (
    "\n\n上一次输出未通过JSON结构或重复退化检查。\n"
    "请重新完成判断，只输出符合指定Schema的JSON。\n"
    "不要重复句子，不要输出解释文字。\n"
    "evidence中的英文双引号必须作为JSON字符串内容正确转义。"
)


def build_retry_user_content(original_user_content: str) -> str:
    """Append the retry suffix to the original user content."""
    return original_user_content + RETRY_PROMPT_SUFFIX


# ---------------------------------------------------------------------------
# Recovery config dataclass
# ---------------------------------------------------------------------------


@dataclass
class SafeRecoveryConfig:
    enabled: bool = False
    strict_outer_json: bool = True
    structured_output: bool = True
    repair_once: bool = True
    detect_repetition: bool = True
    retry_on_parse_failure: bool = True
    retry_on_repetition: bool = True
    max_generation_attempts: int = 2
    grounding_mode: str = "audit"  # "audit" or "off"
    repetition_min_chars: int = REPETITION_MIN_CHARS
    repetition_compression_ratio: float = REPETITION_COMPRESSION_RATIO
    audit_raw_max_chars: int = AUDIT_RAW_MAX_CHARS

    def __post_init__(self) -> None:
        if self.max_generation_attempts < 1:
            raise ValueError("max_generation_attempts must be >= 1")
        if self.max_generation_attempts > 3:
            raise ValueError("max_generation_attempts must be <= 3")
        if self.grounding_mode not in ("audit", "off"):
            raise ValueError("grounding_mode must be 'audit' or 'off'")


def safe_recovery_config_from_dict(data: dict[str, Any]) -> SafeRecoveryConfig:
    return SafeRecoveryConfig(
        enabled=bool(data.get("enabled", False)),
        strict_outer_json=bool(data.get("strict_outer_json", True)),
        structured_output=bool(data.get("structured_output", True)),
        repair_once=bool(data.get("repair_once", True)),
        detect_repetition=bool(data.get("detect_repetition", True)),
        retry_on_parse_failure=bool(data.get("retry_on_parse_failure", True)),
        retry_on_repetition=bool(data.get("retry_on_repetition", True)),
        max_generation_attempts=int(data.get("max_generation_attempts", 2)),
        grounding_mode=str(data.get("grounding_mode", "audit")),
        repetition_min_chars=int(data.get("repetition_min_chars", REPETITION_MIN_CHARS)),
        repetition_compression_ratio=float(
            data.get("repetition_compression_ratio", REPETITION_COMPRESSION_RATIO)
        ),
        audit_raw_max_chars=int(data.get("audit_raw_max_chars", AUDIT_RAW_MAX_CHARS)),
    )

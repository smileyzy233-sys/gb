from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import sys

from .config import PipelineConfig, load_config, resolve_path
from .comparison import (
    default_comparison_paths,
    run_collect_firm_year_outputs,
    run_compare_dummy_consistency,
    write_comparison_manifest,
)
from .extract import (
    ExtractSettings,
    run_extraction,
    run_text_unit_extraction,
    settings_from_config as extract_settings_from_config,
)
from .gb_mapping import run_gb_mapping
from .llm import (
    LocalModelClient,
    OpenAICompatibleClient,
    api_config_from_dict,
    local_config_from_dict,
)
from .main_regression import (
    MainRegressionPaths,
    default_main_regression_paths,
    run_aggregate_main,
    run_build_text_units,
    run_keyword_features,
    run_route_main,
    run_stage1_screening,
    run_text_unit_noise_audit,
    settings_from_config as text_unit_settings_from_config,
    write_manifest,
)
from .preprocess import (
    infer_report_dir,
    load_keywords,
    run_preprocess,
    settings_from_config as preprocess_settings_from_config,
)
from .robustness import (
    default_robustness_paths,
    run_prepare_full_units,
    run_prepare_keyword_units,
    run_prepare_llm_units,
    seed_stage2_results_from_existing,
    write_robustness_manifest,
)
from .vllm_batch import (
    run_stage1_screening_vllm_batch,
    run_text_unit_extraction_vllm_batch,
    vllm_batch_config_from_dict,
)


ROBUSTNESS_COMMAND_METHODS = {
    "robustness-keyword": "robustness_keyword",
    "robustness-llm-only": "robustness_llm_only",
    "robustness-full-llm": "robustness_full_llm",
}


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="configs/pipeline.toml", help="Path to pipeline config TOML.")
    parser.add_argument("--project-root", default=None, help="Project root. Defaults to config parent.")


def add_main_base_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--year", required=True, help="Report year, for example 2024.")
    parser.add_argument(
        "--base-dir",
        default=None,
        help=(
            "Main-regression output directory. Defaults to data/measurement/main_regression, "
            "or data/measurement/main_regression_smoke when --limit is used."
        ),
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit files or rows for smoke tests.")


def add_llm_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--provider", choices=["api", "local", "vllm_batch"], default=None, help="LLM provider.")
    parser.add_argument("--api-key", default=None, help="API key. Prefer environment variables.")
    parser.add_argument("--model-path", default=None, help="Local Hugging Face model path.")
    parser.add_argument("--workers", type=int, default=None, help="Concurrent LLM workers. Overrides config stage/extract workers.")
    parser.add_argument("--batch-size", type=int, default=None, help="Rows to flush per batch.")


def add_robustness_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--year", required=True, help="Report year, for example 2024.")
    parser.add_argument("--base-dir", default=None, help="Output directory for this robustness method.")
    parser.add_argument("--main-base-dir", default=None, help="Main-regression directory to reuse as source.")
    parser.add_argument("--limit", type=int, default=None, help="Limit prepared text units for smoke tests.")
    add_llm_args(parser)
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing method stage2 output and log.")
    parser.add_argument(
        "--no-reuse-main-stage2",
        action="store_true",
        help="Do not seed this method from main-regression stage2 results.",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="standard-pipeline")
    add_common_args(parser)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preprocess = subparsers.add_parser("preprocess", help="Extract relevant annual-report snippets.")
    preprocess.add_argument("--year", required=False, help="Report year, for example 2024.")
    preprocess.add_argument("--input-dir", default=None, help="Annual report txt directory.")
    preprocess.add_argument("--output", default=None, help="Preprocessed CSV output path.")
    preprocess.add_argument("--keywords", default=None, help="Keyword file path.")
    preprocess.add_argument("--limit", type=int, default=None, help="Limit files for smoke tests.")

    extract = subparsers.add_parser("extract", help="Extract standard entities with an LLM.")
    extract.add_argument("--input", required=True, help="Preprocessed CSV path.")
    extract.add_argument("--output", default=None, help="Extraction CSV output path.")
    extract.add_argument("--log", default=None, help="Resume log path.")
    extract.add_argument("--prompt", default=None, help="System prompt text file.")
    extract.add_argument("--provider", choices=["api", "local"], default=None, help="LLM provider.")
    extract.add_argument("--api-key", default=None, help="API key. Prefer environment variables.")
    extract.add_argument("--model-path", default=None, help="Local Hugging Face model path.")
    extract.add_argument("--workers", type=int, default=None, help="Concurrent API workers. Local mode is forced to 1.")
    extract.add_argument("--batch-size", type=int, default=None, help="Rows to flush per batch.")
    extract.add_argument("--limit", type=int, default=None, help="Limit rows for smoke tests.")
    extract.add_argument("--no-resume", action="store_true", help="Ignore existing processed-task log.")

    map_gb = subparsers.add_parser("map-gb", help="Map TYPE_B GB standards to international standards.")
    map_gb.add_argument("--input", required=True, help="Extraction CSV path.")
    map_gb.add_argument("--gb-dict", default=None, help="GB mapping CSV path.")
    map_gb.add_argument("--output", default=None, help="Mapped CSV output path.")

    run_all = subparsers.add_parser("run-all", help="Run preprocess, extract, and GB mapping.")
    run_all.add_argument("--year", required=True, help="Report year, for example 2024.")
    run_all.add_argument("--input-dir", default=None, help="Annual report txt directory.")
    run_all.add_argument("--provider", choices=["api", "local"], default=None, help="LLM provider.")
    run_all.add_argument("--api-key", default=None, help="API key. Prefer environment variables.")
    run_all.add_argument("--model-path", default=None, help="Local Hugging Face model path.")
    run_all.add_argument("--workers", type=int, default=None, help="Concurrent API workers.")
    run_all.add_argument("--limit", type=int, default=None, help="Limit files/rows for smoke tests.")
    run_all.add_argument("--no-resume", action="store_true", help="Ignore existing processed-task log.")

    build_units = subparsers.add_parser("build-text-units", help="Build text_unit rows and keyword features.")
    add_main_base_args(build_units)
    build_units.add_argument("--input-dir", default=None, help="Annual report txt directory.")
    build_units.add_argument("--output", default=None, help="Text units CSV output path.")
    build_units.add_argument("--keywords", default=None, help="Keyword file path.")

    audit_units = subparsers.add_parser(
        "audit-text-units",
        help="Dry-run page/header and table-noise cleanup without overwriting text units.",
    )
    add_main_base_args(audit_units)
    audit_units.add_argument("--input", default=None, help="Existing text units CSV path.")
    audit_units.add_argument("--output", default=None, help="Dry-run audit CSV output path.")
    audit_units.add_argument("--keywords", default=None, help="Keyword file used for table-evidence protection.")

    stage1 = subparsers.add_parser("stage1-screen", help="Run first-stage LLM relevance screening.")
    add_main_base_args(stage1)
    add_llm_args(stage1)
    stage1.add_argument("--input", default=None, help="Text units CSV path.")
    stage1.add_argument("--output", default=None, help="Stage1 relevance CSV output path.")
    stage1.add_argument("--prompt", default=None, help="Stage1 system prompt text file.")
    stage1.add_argument("--keyword-features", default=None, help="Keyword features CSV used to skip keyword candidates.")
    stage1.add_argument("--raw-failure-log", default=None, help="Stage1 raw JSON-parse failure JSONL log path.")
    stage1.add_argument("--no-resume", action="store_true", help="Ignore existing stage1 checkpoint output.")

    route = subparsers.add_parser("route-main", help="Route text units to stage2 using keyword and stage1 channels.")
    add_main_base_args(route)
    route.add_argument("--text-units", default=None, help="Text units CSV path.")
    route.add_argument("--keyword-features", default=None, help="Keyword features CSV path.")
    route.add_argument("--stage1", default=None, help="Stage1 relevance CSV path.")
    route.add_argument("--output", default=None, help="Stage2 input CSV output path.")
    route.add_argument("--keywords", default=None, help="Keyword file path if keyword features need to be built.")

    stage2 = subparsers.add_parser("stage2-extract", help="Extract entities for routed text units.")
    add_main_base_args(stage2)
    add_llm_args(stage2)
    stage2.add_argument("--input", default=None, help="Stage2 input CSV path.")
    stage2.add_argument("--output", default=None, help="Stage2 entity result CSV output path.")
    stage2.add_argument("--log", default=None, help="Stage2 resume log path.")
    stage2.add_argument("--prompt", default=None, help="Stage2 extraction prompt text file.")
    stage2.add_argument("--no-resume", action="store_true", help="Ignore existing processed-task log.")

    map_main = subparsers.add_parser("map-main-gb", help="Map routed stage2 entity results with the GB dictionary.")
    add_main_base_args(map_main)
    map_main.add_argument("--input", default=None, help="Stage2 entity result CSV path.")
    map_main.add_argument("--gb-dict", default=None, help="GB mapping CSV path.")
    map_main.add_argument("--output", default=None, help="Mapped entity result CSV output path.")

    aggregate = subparsers.add_parser("aggregate-main", help="Aggregate mapped text-unit results to firm-year dummies.")
    add_main_base_args(aggregate)
    aggregate.add_argument("--text-units", default=None, help="Text units CSV path.")
    aggregate.add_argument("--mapped", default=None, help="Mapped entity result CSV path.")
    aggregate.add_argument("--output", default=None, help="Firm-year output CSV path.")

    main_regression = subparsers.add_parser("main-regression", help="Run the full main-regression workflow.")
    add_main_base_args(main_regression)
    add_llm_args(main_regression)
    main_regression.add_argument("--input-dir", default=None, help="Annual report txt directory.")
    main_regression.add_argument("--no-resume", action="store_true", help="Ignore existing stage2 processed-task log.")

    robustness_keyword = subparsers.add_parser("robustness-keyword", help="Run keyword-only robustness workflow.")
    add_robustness_args(robustness_keyword)

    robustness_llm_only = subparsers.add_parser("robustness-llm-only", help="Run LLM-only robustness workflow.")
    add_robustness_args(robustness_llm_only)

    robustness_full_llm = subparsers.add_parser("robustness-full-llm", help="Run full-LLM robustness workflow.")
    add_robustness_args(robustness_full_llm)

    compare = subparsers.add_parser("compare-measurements", help="Compare firm-year dummies across methods.")
    compare.add_argument("--year", required=True, help="Report year, for example 2024.")
    compare.add_argument("--base-dir", default=None, help="Comparison output directory.")
    compare.add_argument("--main-base-dir", default=None, help="Main-regression output directory.")
    compare.add_argument("--keyword-base-dir", default=None, help="Keyword robustness output directory.")
    compare.add_argument("--llm-only-base-dir", default=None, help="LLM-only robustness output directory.")
    compare.add_argument("--full-llm-base-dir", default=None, help="Full-LLM robustness output directory.")
    compare.add_argument("--smoke", action="store_true", help="Read and write under data/measurement_smoke.")

    return parser.parse_args(argv)


def infer_year_from_path(path: Path) -> str:
    match = re.search(r"(20\d{2}|19\d{2})", path.stem)
    if not match:
        raise ValueError(f"Cannot infer year from path: {path}")
    return match.group(1)


def default_preprocessed_path(config: PipelineConfig, year: str) -> Path:
    return config.path("preprocessed_dir") / f"yuchuli_{year}.csv"


def default_extraction_path(config: PipelineConfig, year: str) -> Path:
    return config.path("prediction_dir") / f"final_{year}.csv"


def default_log_path(config: PipelineConfig, year: str, provider: str) -> Path:
    return config.path("log_dir") / f"processed_tasks_{year}_{provider}.log"


def default_mapped_path(input_csv: Path) -> Path:
    return input_csv.with_name(f"{input_csv.stem}_mapped.csv")


def default_stage1_prompt_path(config: PipelineConfig) -> Path:
    return config.project_root / "prompts" / "stage1_relevance_zh.txt"


def cli_path(config: PipelineConfig, value: str | None, fallback: Path) -> Path:
    return resolve_path(config.project_root, value) if value else fallback


def main_paths(config: PipelineConfig, args: argparse.Namespace) -> MainRegressionPaths:
    base_dir = resolve_path(config.project_root, args.base_dir) if getattr(args, "base_dir", None) else None
    return default_main_regression_paths(
        config.project_root,
        args.year,
        smoke=getattr(args, "limit", None) is not None and base_dir is None,
        base_dir=base_dir,
    )


def robustness_paths(config: PipelineConfig, args: argparse.Namespace, method: str):
    base_dir = resolve_path(config.project_root, args.base_dir) if getattr(args, "base_dir", None) else None
    return default_robustness_paths(
        config.project_root,
        method,
        args.year,
        smoke=getattr(args, "limit", None) is not None and base_dir is None,
        base_dir=base_dir,
    )


def robustness_source_main_paths(config: PipelineConfig, args: argparse.Namespace) -> MainRegressionPaths:
    base_dir = resolve_path(config.project_root, args.main_base_dir) if getattr(args, "main_base_dir", None) else None
    return default_main_regression_paths(
        config.project_root,
        args.year,
        smoke=getattr(args, "limit", None) is not None and base_dir is None,
        base_dir=base_dir,
    )


def comparison_paths(config: PipelineConfig, args: argparse.Namespace):
    base_dir = resolve_path(config.project_root, args.base_dir) if getattr(args, "base_dir", None) else None
    return default_comparison_paths(
        config.project_root,
        args.year,
        smoke=getattr(args, "smoke", False) and base_dir is None,
        base_dir=base_dir,
    )


def comparison_method_paths(config: PipelineConfig, args: argparse.Namespace, method: str):
    attr = {
        "main_regression": "main_base_dir",
        "robustness_keyword": "keyword_base_dir",
        "robustness_llm_only": "llm_only_base_dir",
        "robustness_full_llm": "full_llm_base_dir",
    }[method]
    value = getattr(args, attr)
    base_dir = resolve_path(config.project_root, value) if value else None
    if method == "main_regression":
        return default_main_regression_paths(
            config.project_root,
            args.year,
            smoke=getattr(args, "smoke", False) and base_dir is None,
            base_dir=base_dir,
        )
    return default_robustness_paths(
        config.project_root,
        method,
        args.year,
        smoke=getattr(args, "smoke", False) and base_dir is None,
        base_dir=base_dir,
    )


def main_input_dir(config: PipelineConfig, year: str, value: str | None = None) -> Path:
    if value:
        return resolve_path(config.project_root, value)
    return infer_report_dir(config.path("annual_reports_dir"), year)


def provider_model_label(
    config: PipelineConfig,
    provider: str,
    model_path: str | None = None,
    stage_name: str | None = None,
) -> str:
    if provider == "api":
        data = config.section(stage_name, "api") if stage_name else {}
        if not data:
            data = config.section("extract", "api")
        model_env = data.get("model_env", "STANDARD_PIPELINE_MODEL")
        return os.environ.get(model_env) or data.get("model", "deepseek-chat")

    if provider == "vllm_batch":
        data = config.section(stage_name, "vllm_batch") if stage_name else {}
        path_env = data.get("model_path_env", "LOCAL_MODEL_PATH")
        resolved_model_path = model_path or os.environ.get(path_env) or data.get("model_path") or "local-model"
        return f"vllm_batch:{resolved_model_path}"

    if provider == "local":
        data = config.section(stage_name, "local") if stage_name else {}
        if not data:
            data = config.section("extract", "local")
        path_env = data.get("model_path_env", "LOCAL_MODEL_PATH")
        return model_path or os.environ.get(path_env) or data.get("model_path", "local-model")

    raise ValueError(f"Unknown provider for model label: {provider}")


def write_main_manifest(
    config: PipelineConfig,
    paths: MainRegressionPaths,
    input_dir: Path,
    provider: str | None = None,
    model_path: str | None = None,
    stage1_prompt: Path | None = None,
    stage2_prompt: Path | None = None,
    gb_mapping: Path | None = None,
    stage1_provider: str | None = None,
    stage2_provider: str | None = None,
    stage1_raw_failure_log: Path | None = None,
) -> None:
    fallback_provider = provider or config.section("extract").get("provider", "api")
    resolved_stage1_provider = stage1_provider or config.section("stage1").get("provider") or fallback_provider
    resolved_stage2_provider = stage2_provider or config.section("stage2").get("provider") or fallback_provider
    model_stage1 = provider_model_label(
        config,
        resolved_stage1_provider,
        model_path=model_path,
        stage_name="stage1",
    )
    model_stage2 = provider_model_label(
        config,
        resolved_stage2_provider,
        model_path=model_path,
        stage_name="stage2",
    )
    write_manifest(
        paths=paths,
        input_report_dir=input_dir,
        prompt_stage1_path=stage1_prompt or default_stage1_prompt_path(config),
        prompt_stage2_path=stage2_prompt or config.path("prompt_file"),
        gb_mapping_path=gb_mapping or config.path("gb_mapping_csv"),
        model_stage1=model_stage1,
        model_stage2=model_stage2,
        stage1_raw_failure_log_path=stage1_raw_failure_log,
    )


def build_client(config: PipelineConfig, provider: str, api_key: str | None = None, model_path: str | None = None):
    if provider == "api":
        return OpenAICompatibleClient(api_config_from_dict(config.section("extract", "api"), api_key=api_key))
    if provider == "local":
        return LocalModelClient(local_config_from_dict(config.section("extract", "local"), model_path=model_path))
    raise ValueError(f"Provider {provider} does not use ChatClient. Use the dedicated runner.")


def stage_provider(config: PipelineConfig, stage_name: str, fallback_provider: str | None = None) -> str:
    return (
        fallback_provider
        or config.section(stage_name).get("provider")
        or config.section("extract").get("provider", "api")
    )


def build_stage_client(
    config: PipelineConfig,
    stage_name: str,
    provider: str,
    api_key: str | None = None,
    model_path: str | None = None,
):
    if provider == "api":
        data = config.section(stage_name, "api")
        if not data:
            data = config.section("extract", "api")
        return OpenAICompatibleClient(api_config_from_dict(data, api_key=api_key))

    if provider == "local":
        data = config.section(stage_name, "local")
        if not data:
            data = config.section("extract", "local")
        return LocalModelClient(local_config_from_dict(data, model_path=model_path))

    raise ValueError(f"Provider {provider} does not use ChatClient. Use the dedicated runner.")


def stage_extract_settings(
    config: PipelineConfig,
    stage_name: str,
    workers: int | None,
    batch_size: int | None,
) -> ExtractSettings:
    base = config.section("extract")
    stage = config.section(stage_name)
    return ExtractSettings(
        workers=workers or int(stage.get("workers", base.get("workers", 3))),
        batch_size=batch_size or int(stage.get("batch_size", base.get("batch_size", 10))),
        max_retries=int(stage.get("max_retries", base.get("max_retries", 3))),
        retry_min_seconds=int(stage.get("retry_min_seconds", base.get("retry_min_seconds", 2))),
        retry_max_seconds=int(stage.get("retry_max_seconds", base.get("retry_max_seconds", 10))),
    )


def extract_settings(config: PipelineConfig, workers: int | None, batch_size: int | None, provider: str) -> ExtractSettings:
    settings = extract_settings_from_config(config.section("extract"))
    resolved_workers = workers or settings.workers
    return ExtractSettings(
        workers=resolved_workers,
        batch_size=batch_size or settings.batch_size,
        max_retries=settings.max_retries,
        retry_min_seconds=settings.retry_min_seconds,
        retry_max_seconds=settings.retry_max_seconds,
    )


def read_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def command_preprocess(args: argparse.Namespace, config: PipelineConfig) -> Path:
    if args.input_dir:
        input_dir = resolve_path(config.project_root, args.input_dir)
    else:
        if not args.year:
            raise ValueError("--year is required when --input-dir is not provided")
        input_dir = infer_report_dir(config.path("annual_reports_dir"), args.year)
    year = args.year or infer_year_from_path(input_dir)
    output = cli_path(config, args.output, default_preprocessed_path(config, year))
    keywords = cli_path(config, args.keywords, config.path("keyword_file"))
    settings = preprocess_settings_from_config(config.section("preprocess"))
    df = run_preprocess(input_dir, output, keywords, settings, limit=args.limit)
    print(f"Preprocessed {len(df)} rows -> {output}")
    return output


def command_extract(args: argparse.Namespace, config: PipelineConfig) -> Path:
    input_csv = resolve_path(config.project_root, args.input)
    year = infer_year_from_path(input_csv)
    provider = args.provider or config.section("extract").get("provider", "api")
    output = cli_path(config, args.output, default_extraction_path(config, year))
    log_file = cli_path(config, args.log, default_log_path(config, year, provider))
    prompt = read_prompt(cli_path(config, args.prompt, config.path("prompt_file")))
    client = build_client(config, provider, api_key=args.api_key, model_path=args.model_path)
    settings = extract_settings(config, args.workers, args.batch_size, provider)
    df = run_extraction(
        input_csv,
        output,
        log_file,
        client,
        prompt,
        settings,
        resume=not args.no_resume,
        limit=args.limit,
    )
    print(f"Extracted {len(df)} rows -> {output}")
    return output


def command_map_gb(args: argparse.Namespace, config: PipelineConfig) -> Path:
    input_csv = resolve_path(config.project_root, args.input)
    gb_dict = cli_path(config, args.gb_dict, config.path("gb_mapping_csv"))
    output = cli_path(config, args.output, default_mapped_path(input_csv))
    df = run_gb_mapping(input_csv, gb_dict, output)
    print(f"Mapped {len(df)} rows -> {output}")
    return output


def command_run_all(args: argparse.Namespace, config: PipelineConfig) -> None:
    preprocess_args = argparse.Namespace(
        year=args.year,
        input_dir=args.input_dir,
        output=None,
        keywords=None,
        limit=args.limit,
    )
    preprocessed_csv = command_preprocess(preprocess_args, config)

    provider = args.provider or config.section("extract").get("provider", "api")
    extraction_csv = default_extraction_path(config, args.year)
    extract_args = argparse.Namespace(
        input=str(preprocessed_csv),
        output=str(extraction_csv),
        log=None,
        prompt=None,
        provider=provider,
        api_key=args.api_key,
        model_path=args.model_path,
        workers=args.workers,
        batch_size=None,
        limit=args.limit,
        no_resume=args.no_resume,
    )
    final_csv = command_extract(extract_args, config)

    map_args = argparse.Namespace(input=str(final_csv), gb_dict=None, output=None)
    command_map_gb(map_args, config)


def command_build_text_units(args: argparse.Namespace, config: PipelineConfig) -> Path:
    paths = main_paths(config, args)
    paths.ensure_dirs()
    input_dir = main_input_dir(config, args.year, args.input_dir)
    text_units_output = cli_path(config, args.output, paths.text_units_path)
    keyword_output = paths.keyword_features_path
    keywords = cli_path(config, args.keywords, config.path("keyword_file"))
    protected_anchor_terms = load_keywords(keywords)
    preprocess_settings = preprocess_settings_from_config(config.section("preprocess"))
    unit_settings = text_unit_settings_from_config(config.section("main_regression"))

    df_units = run_build_text_units(
        input_dir,
        text_units_output,
        preprocess_settings,
        unit_settings,
        limit=args.limit,
        protected_anchor_terms=protected_anchor_terms,
    )
    df_keywords = run_keyword_features(text_units_output, keywords, keyword_output)

    provider = config.section("extract").get("provider", "api")
    write_main_manifest(config, paths, input_dir, provider)
    print(f"Built {len(df_units)} text units -> {text_units_output}")
    print(f"Built {len(df_keywords)} keyword feature rows -> {keyword_output}")
    return text_units_output


def command_audit_text_units(args: argparse.Namespace, config: PipelineConfig) -> Path:
    paths = main_paths(config, args)
    paths.ensure_dirs()
    input_csv = cli_path(config, args.input, paths.text_units_path)
    output_csv = cli_path(config, args.output, paths.text_unit_audit_path)
    keyword_file = cli_path(config, args.keywords, config.path("keyword_file"))
    processed = run_text_unit_noise_audit(
        input_csv,
        output_csv,
        limit=args.limit,
        protected_anchor_terms=load_keywords(keyword_file),
    )
    print(f"Audited {processed} text units without modifying {input_csv} -> {output_csv}")
    return output_csv


def command_stage1_screen(args: argparse.Namespace, config: PipelineConfig) -> Path:
    paths = main_paths(config, args)
    paths.ensure_dirs()
    input_dir = main_input_dir(config, args.year, None)
    text_units = cli_path(config, args.input, paths.text_units_path)
    output = cli_path(config, args.output, paths.stage1_relevance_path)
    prompt_path = cli_path(config, args.prompt, default_stage1_prompt_path(config))
    provider = stage_provider(config, "stage1", args.provider)
    keyword_features = cli_path(config, args.keyword_features, paths.keyword_features_path)
    raw_failure_log = cli_path(config, args.raw_failure_log, paths.stage1_raw_failure_log_path(provider))
    prompt = read_prompt(prompt_path)

    if provider == "vllm_batch":
        batch_config = vllm_batch_config_from_dict(
            config.section("stage1", "vllm_batch"),
            model_path=args.model_path,
        )
        df = run_stage1_screening_vllm_batch(
            text_units,
            output,
            prompt,
            batch_config,
            limit=args.limit,
            keyword_features_csv=keyword_features if keyword_features.exists() else None,
            raw_failure_log=raw_failure_log,
            resume=not args.no_resume,
        )
    else:
        client = build_stage_client(
            config,
            "stage1",
            provider,
            api_key=args.api_key,
            model_path=args.model_path,
        )
        settings = stage_extract_settings(config, "stage1", args.workers, args.batch_size)
        df = run_stage1_screening(
            text_units,
            output,
            client,
            prompt,
            settings,
            limit=args.limit,
            keyword_features_csv=keyword_features if keyword_features.exists() else None,
            raw_failure_log=raw_failure_log,
        )
    write_main_manifest(
        config,
        paths,
        input_dir,
        provider,
        model_path=args.model_path,
        stage1_prompt=prompt_path,
        stage1_provider=provider,
        stage1_raw_failure_log=raw_failure_log,
    )
    print(f"Stage1 screened {len(df)} text units -> {output}")
    return output


def command_route_main(args: argparse.Namespace, config: PipelineConfig) -> Path:
    paths = main_paths(config, args)
    paths.ensure_dirs()
    input_dir = main_input_dir(config, args.year, None)
    text_units = cli_path(config, args.text_units, paths.text_units_path)
    keyword_features = cli_path(config, args.keyword_features, paths.keyword_features_path)
    stage1 = cli_path(config, args.stage1, paths.stage1_relevance_path)
    output = cli_path(config, args.output, paths.stage2_input_path)

    if not keyword_features.exists():
        keywords = cli_path(config, args.keywords, config.path("keyword_file"))
        run_keyword_features(text_units, keywords, keyword_features)

    df = run_route_main(text_units, keyword_features, stage1, output, limit=args.limit)
    provider = config.section("extract").get("provider", "api")
    write_main_manifest(config, paths, input_dir, provider)
    print(f"Routed {len(df)} text units to stage2 -> {output}")
    return output


def command_stage2_extract(args: argparse.Namespace, config: PipelineConfig) -> Path:
    paths = main_paths(config, args)
    paths.ensure_dirs()
    input_dir = main_input_dir(config, args.year, None)
    stage2_input = cli_path(config, args.input, paths.stage2_input_path)
    output = cli_path(config, args.output, paths.stage2_result_path)
    provider = stage_provider(config, "stage2", args.provider)
    log_file = cli_path(config, args.log, paths.stage2_log_path(provider))
    prompt_path = cli_path(config, args.prompt, config.path("prompt_file"))
    prompt = read_prompt(prompt_path)

    if provider == "vllm_batch":
        batch_config = vllm_batch_config_from_dict(
            config.section("stage2", "vllm_batch"),
            model_path=args.model_path,
        )
        df = run_text_unit_extraction_vllm_batch(
            stage2_input,
            output,
            log_file,
            prompt,
            batch_config,
            resume=not args.no_resume,
            limit=args.limit,
        )
    else:
        client = build_stage_client(
            config,
            "stage2",
            provider,
            api_key=args.api_key,
            model_path=args.model_path,
        )
        settings = stage_extract_settings(config, "stage2", args.workers, args.batch_size)
        df = run_text_unit_extraction(
            stage2_input,
            output,
            log_file,
            client,
            prompt,
            settings,
            resume=not args.no_resume,
            limit=args.limit,
        )
    write_main_manifest(
        config,
        paths,
        input_dir,
        provider,
        model_path=args.model_path,
        stage2_prompt=prompt_path,
        stage2_provider=provider,
    )
    print(f"Stage2 extracted {len(df)} entity rows -> {output}")
    return output


def command_map_main_gb(args: argparse.Namespace, config: PipelineConfig) -> Path:
    paths = main_paths(config, args)
    paths.ensure_dirs()
    input_dir = main_input_dir(config, args.year, None)
    input_csv = cli_path(config, args.input, paths.stage2_result_path)
    gb_dict = cli_path(config, args.gb_dict, config.path("gb_mapping_csv"))
    output = cli_path(config, args.output, paths.mapped_result_path)

    df = run_gb_mapping(input_csv, gb_dict, output)
    provider = config.section("extract").get("provider", "api")
    write_main_manifest(config, paths, input_dir, provider, gb_mapping=gb_dict)
    print(f"Mapped {len(df)} entity rows -> {output}")
    return output


def command_aggregate_main(args: argparse.Namespace, config: PipelineConfig) -> Path:
    paths = main_paths(config, args)
    paths.ensure_dirs()
    input_dir = main_input_dir(config, args.year, None)
    text_units = cli_path(config, args.text_units, paths.text_units_path)
    mapped = cli_path(config, args.mapped, paths.mapped_result_path)
    output = cli_path(config, args.output, paths.final_output_path)

    df = run_aggregate_main(text_units, mapped, output)
    provider = config.section("extract").get("provider", "api")
    write_main_manifest(config, paths, input_dir, provider)
    print(f"Aggregated {len(df)} firm-year rows -> {output}")
    return output


def command_main_regression(args: argparse.Namespace, config: PipelineConfig) -> Path:
    paths = main_paths(config, args)
    paths.ensure_dirs()
    input_dir = main_input_dir(config, args.year, args.input_dir)
    fallback_provider = args.provider or config.section("extract").get("provider", "api")
    stage1_provider_value = stage_provider(config, "stage1", args.provider)
    stage2_provider_value = stage_provider(config, "stage2", args.provider)
    preprocess_settings = preprocess_settings_from_config(config.section("preprocess"))
    unit_settings = text_unit_settings_from_config(config.section("main_regression"))
    keyword_file = config.path("keyword_file")
    protected_anchor_terms = load_keywords(keyword_file)
    stage1_prompt_path = default_stage1_prompt_path(config)
    stage2_prompt_path = config.path("prompt_file")
    gb_mapping = config.path("gb_mapping_csv")

    text_units = run_build_text_units(
        input_dir,
        paths.text_units_path,
        preprocess_settings,
        unit_settings,
        limit=args.limit,
        protected_anchor_terms=protected_anchor_terms,
    )
    run_keyword_features(paths.text_units_path, keyword_file, paths.keyword_features_path)
    stage1_raw_failure_log = paths.stage1_raw_failure_log_path(stage1_provider_value)
    stage1_prompt = read_prompt(stage1_prompt_path)
    if stage1_provider_value == "vllm_batch":
        stage1_batch_config = vllm_batch_config_from_dict(
            config.section("stage1", "vllm_batch"),
            model_path=args.model_path,
        )
        run_stage1_screening_vllm_batch(
            paths.text_units_path,
            paths.stage1_relevance_path,
            stage1_prompt,
            stage1_batch_config,
            keyword_features_csv=paths.keyword_features_path,
            raw_failure_log=stage1_raw_failure_log,
            resume=not args.no_resume,
        )
    else:
        stage1_client = build_stage_client(
            config,
            "stage1",
            stage1_provider_value,
            api_key=args.api_key,
            model_path=args.model_path,
        )
        stage1_settings = stage_extract_settings(config, "stage1", args.workers, args.batch_size)
        run_stage1_screening(
            paths.text_units_path,
            paths.stage1_relevance_path,
            stage1_client,
            stage1_prompt,
            stage1_settings,
            keyword_features_csv=paths.keyword_features_path,
            raw_failure_log=stage1_raw_failure_log,
        )
    stage2_input = run_route_main(
        paths.text_units_path,
        paths.keyword_features_path,
        paths.stage1_relevance_path,
        paths.stage2_input_path,
    )
    stage2_prompt = read_prompt(stage2_prompt_path)
    if stage2_provider_value == "vllm_batch":
        stage2_batch_config = vllm_batch_config_from_dict(
            config.section("stage2", "vllm_batch"),
            model_path=args.model_path,
        )
        run_text_unit_extraction_vllm_batch(
            paths.stage2_input_path,
            paths.stage2_result_path,
            paths.stage2_log_path(stage2_provider_value),
            stage2_prompt,
            stage2_batch_config,
            resume=not args.no_resume,
        )
    else:
        stage2_client = build_stage_client(
            config,
            "stage2",
            stage2_provider_value,
            api_key=args.api_key,
            model_path=args.model_path,
        )
        stage2_settings = stage_extract_settings(config, "stage2", args.workers, args.batch_size)
        run_text_unit_extraction(
            paths.stage2_input_path,
            paths.stage2_result_path,
            paths.stage2_log_path(stage2_provider_value),
            stage2_client,
            stage2_prompt,
            stage2_settings,
            resume=not args.no_resume,
        )
    run_gb_mapping(paths.stage2_result_path, gb_mapping, paths.mapped_result_path)
    final = run_aggregate_main(paths.text_units_path, paths.mapped_result_path, paths.final_output_path)
    write_main_manifest(
        config,
        paths,
        input_dir,
        provider=fallback_provider,
        model_path=args.model_path,
        stage1_prompt=stage1_prompt_path,
        stage2_prompt=stage2_prompt_path,
        gb_mapping=gb_mapping,
        stage1_provider=stage1_provider_value,
        stage2_provider=stage2_provider_value,
        stage1_raw_failure_log=stage1_raw_failure_log,
    )
    print(f"Built {len(text_units)} text units")
    print(f"Routed {len(stage2_input)} text units to stage2")
    print(f"Main regression firm-year rows: {len(final)} -> {paths.final_output_path}")
    return paths.final_output_path


def remove_stage2_resume_files(output_csv: Path, log_file: Path) -> None:
    for path in (output_csv, log_file):
        if path.exists():
            path.unlink()


def prepare_robustness_units(method: str, paths, main: MainRegressionPaths, limit: int | None):
    if method == "robustness_keyword":
        return run_prepare_keyword_units(
            main.text_units_path,
            main.keyword_features_path,
            paths.units_path,
            limit=limit,
        )
    if method == "robustness_llm_only":
        return run_prepare_llm_units(
            main.text_units_path,
            main.stage1_relevance_path,
            paths.units_path,
            limit=limit,
        )
    return run_prepare_full_units(main.text_units_path, paths.units_path, limit=limit)


def command_robustness(args: argparse.Namespace, config: PipelineConfig, method: str) -> Path:
    paths = robustness_paths(config, args, method)
    main = robustness_source_main_paths(config, args)
    paths.ensure_dirs()
    provider = stage_provider(config, "stage2", args.provider)
    stage2_prompt_path = config.path("prompt_file")
    stage1_prompt_path = default_stage1_prompt_path(config)
    gb_mapping = config.path("gb_mapping_csv")
    stage1_model_label = provider_model_label(
        config,
        stage_provider(config, "stage1", args.provider),
        model_path=args.model_path,
        stage_name="stage1",
    )
    stage2_model_label = provider_model_label(
        config,
        provider,
        model_path=args.model_path,
        stage_name="stage2",
    )

    units = prepare_robustness_units(method, paths, main, args.limit)
    log_file = paths.stage2_log_path(provider)
    if args.no_resume:
        remove_stage2_resume_files(paths.stage2_result_path, log_file)

    seed_report = None
    if not args.no_reuse_main_stage2:
        seed_report = seed_stage2_results_from_existing(
            paths.units_path,
            main.stage2_result_path,
            paths.stage2_result_path,
            log_file,
        )

    needs_stage2 = seed_report is None or bool(seed_report.missing_ids)
    if needs_stage2:
        stage2_prompt = read_prompt(stage2_prompt_path)
        if provider == "vllm_batch":
            batch_config = vllm_batch_config_from_dict(
                config.section("stage2", "vllm_batch"),
                model_path=args.model_path,
            )
            run_text_unit_extraction_vllm_batch(
                paths.units_path,
                paths.stage2_result_path,
                log_file,
                stage2_prompt,
                batch_config,
                resume=True,
            )
        else:
            client = build_stage_client(
                config,
                "stage2",
                provider,
                api_key=args.api_key,
                model_path=args.model_path,
            )
            settings = stage_extract_settings(config, "stage2", args.workers, args.batch_size)
            run_text_unit_extraction(
                paths.units_path,
                paths.stage2_result_path,
                log_file,
                client,
                stage2_prompt,
                settings,
                resume=True,
            )

    run_gb_mapping(paths.stage2_result_path, gb_mapping, paths.mapped_result_path)
    final = run_aggregate_main(paths.units_path, paths.mapped_result_path, paths.final_output_path)
    write_robustness_manifest(
        paths,
        source_text_units_path=main.text_units_path,
        source_keyword_features_path=main.keyword_features_path if method == "robustness_keyword" else None,
        source_stage1_relevance_path=main.stage1_relevance_path if method == "robustness_llm_only" else None,
        source_stage2_result_path=main.stage2_result_path,
        prompt_stage1_path=stage1_prompt_path if method == "robustness_llm_only" else None,
        prompt_stage2_path=stage2_prompt_path,
        gb_mapping_path=gb_mapping,
        model_stage1=stage1_model_label if method == "robustness_llm_only" else None,
        model_stage2=stage2_model_label,
    )

    if seed_report is not None:
        print(f"Prepared {len(units)} {method} units -> {paths.units_path}")
        print(
            f"Seeded {seed_report.seeded_rows} stage2 rows from main regression; "
            f"{len(seed_report.missing_ids)} text units required extraction."
        )
    else:
        print(f"Prepared {len(units)} {method} units -> {paths.units_path}")
    print(f"{method} firm-year rows: {len(final)} -> {paths.final_output_path}")
    return paths.final_output_path


def command_compare_measurements(args: argparse.Namespace, config: PipelineConfig) -> Path:
    paths = comparison_paths(config, args)
    paths.ensure_dirs()
    main = comparison_method_paths(config, args, "main_regression")
    keyword = comparison_method_paths(config, args, "robustness_keyword")
    llm_only = comparison_method_paths(config, args, "robustness_llm_only")
    full_llm = comparison_method_paths(config, args, "robustness_full_llm")

    collected = run_collect_firm_year_outputs(
        main.final_output_path,
        keyword.final_output_path,
        llm_only.final_output_path,
        full_llm.final_output_path,
        paths.collected_output_path,
    )
    consistency = run_compare_dummy_consistency(paths.collected_output_path, paths.consistency_output_path)
    write_comparison_manifest(
        paths,
        main_final_csv=main.final_output_path,
        keyword_final_csv=keyword.final_output_path,
        llm_only_final_csv=llm_only.final_output_path,
        full_llm_final_csv=full_llm.final_output_path,
    )
    print(f"Collected {len(collected)} firm-year rows -> {paths.collected_output_path}")
    print(f"Compared {len(consistency)} firm-year rows -> {paths.consistency_output_path}")
    return paths.consistency_output_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config, args.project_root)
    try:
        if args.command == "preprocess":
            command_preprocess(args, config)
        elif args.command == "extract":
            command_extract(args, config)
        elif args.command == "map-gb":
            command_map_gb(args, config)
        elif args.command == "run-all":
            command_run_all(args, config)
        elif args.command == "build-text-units":
            command_build_text_units(args, config)
        elif args.command == "audit-text-units":
            command_audit_text_units(args, config)
        elif args.command == "stage1-screen":
            command_stage1_screen(args, config)
        elif args.command == "route-main":
            command_route_main(args, config)
        elif args.command == "stage2-extract":
            command_stage2_extract(args, config)
        elif args.command == "map-main-gb":
            command_map_main_gb(args, config)
        elif args.command == "aggregate-main":
            command_aggregate_main(args, config)
        elif args.command == "main-regression":
            command_main_regression(args, config)
        elif args.command in ROBUSTNESS_COMMAND_METHODS:
            command_robustness(args, config, ROBUSTNESS_COMMAND_METHODS[args.command])
        elif args.command == "compare-measurements":
            command_compare_measurements(args, config)
        else:
            raise ValueError(f"Unknown command: {args.command}")
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

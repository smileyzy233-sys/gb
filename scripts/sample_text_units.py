from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from standard_pipeline.config import load_config, resolve_path
from standard_pipeline.main_regression import (
    run_build_text_units,
    settings_from_config as text_unit_settings_from_config,
)
from standard_pipeline.preprocess import (
    infer_report_dir,
    normalize_stock_code,
    parse_report_filename,
    settings_from_config as preprocess_settings_from_config,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Randomly sample annual-report TXT files and build text-unit CSVs only. "
            "This script does not run keyword features or either LLM stage."
        )
    )
    parser.add_argument("--years", nargs="+", required=True, help="Years to sample, e.g. 2019 2020 2021.")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=10,
        help="Reports sampled per year. Use 0 to process every matching report.",
    )
    parser.add_argument("--seed", type=int, default=20260718, help="Random seed for reproducible samples.")
    parser.add_argument(
        "--stock-codes",
        nargs="*",
        default=None,
        help="Optional codes to select, separated by spaces or commas, e.g. 000001 000002.",
    )
    parser.add_argument("--annual-reports-dir", default=None, help="Annual-report root; defaults to config paths.")
    parser.add_argument(
        "--output-dir",
        default="data/text_unit_samples",
        help="Output root. Each year is written to its own subdirectory.",
    )
    parser.add_argument("--config", default="configs/pipeline.toml", help="Pipeline TOML path.")
    parser.add_argument("--project-root", default=None, help="Project root; defaults to config parent.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing sample CSV.")
    return parser.parse_args(argv)


def parse_stock_codes(values: list[str] | None) -> set[str]:
    if not values:
        return set()
    raw_codes = [part for value in values for part in value.split(",")]
    return {normalize_stock_code(code) for code in raw_codes if code.strip()}


def list_report_files(report_dir: Path, stock_codes: set[str]) -> list[Path]:
    files: list[Path] = []
    for path in sorted(report_dir.glob("*.txt")):
        meta = parse_report_filename(path.name)
        if not meta:
            continue
        if stock_codes and meta["stock_code"] not in stock_codes:
            continue
        files.append(path)
    return files


def select_report_files(files: list[Path], sample_size: int, seed: int, year: str) -> list[Path]:
    if sample_size < 0:
        raise ValueError("--sample-size must be zero or positive")
    if sample_size == 0 or sample_size >= len(files):
        return files
    rng = random.Random(f"{seed}:{year}")
    return sorted(rng.sample(files, sample_size))


def text_length_summary(df) -> str:
    if df.empty:
        return "units=0"
    lengths = df["text"].fillna("").astype(str).str.len()
    return (
        f"units={len(df)}, below_1000={int((lengths < 1000).sum())}, "
        f"median_chars={int(lengths.median())}, min_chars={int(lengths.min())}, "
        f"max_chars={int(lengths.max())}"
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config, args.project_root)
    report_root = (
        resolve_path(config.project_root, args.annual_reports_dir)
        if args.annual_reports_dir
        else config.path("annual_reports_dir")
    )
    output_root = resolve_path(config.project_root, args.output_dir)
    stock_codes = parse_stock_codes(args.stock_codes)
    preprocess_settings = preprocess_settings_from_config(config.section("preprocess"))
    unit_settings = text_unit_settings_from_config(config.section("main_regression"))

    for raw_year in args.years:
        year = str(raw_year).strip()
        report_dir = infer_report_dir(report_root, year)
        candidates = list_report_files(report_dir, stock_codes)
        if not candidates:
            code_hint = f" for stock codes {sorted(stock_codes)}" if stock_codes else ""
            raise FileNotFoundError(f"No report TXT files found for {year}{code_hint} under {report_dir}")

        selected = select_report_files(candidates, args.sample_size, args.seed, year)
        output_csv = output_root / year / f"01_text_units_{year}_sample.csv"
        if output_csv.exists() and not args.overwrite:
            raise FileExistsError(f"Output already exists; pass --overwrite to replace it: {output_csv}")

        df = run_build_text_units(
            report_dir,
            output_csv,
            preprocess_settings,
            unit_settings,
            input_files=selected,
        )
        selected_names = ", ".join(path.name for path in selected)
        print(f"[{year}] reports={len(selected)} -> {output_csv}")
        print(f"[{year}] {text_length_summary(df)}")
        print(f"[{year}] selected: {selected_names}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

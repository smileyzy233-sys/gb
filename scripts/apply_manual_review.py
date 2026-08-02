#!/usr/bin/env python
"""Apply manually reviewed Stage2 results and re-run downstream pipeline.

Usage:
  1. Fill in data/measurement/manual_review/pending_76_review_template.csv
  2. Run:
     python scripts/apply_manual_review.py [--year 2024] [--dry-run]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from standard_pipeline.csv_io import safe_to_csv
from standard_pipeline.gb_mapping import run_gb_mapping
from standard_pipeline.main_regression import (
    MainRegressionPaths,
    default_main_regression_paths,
    run_aggregate_main,
)
from standard_pipeline.schemas import TEXT_UNIT_EXTRACTION_COLUMNS

VALID_TYPES = {"TYPE_A", "TYPE_B", "TYPE_C", "TYPE_D"}
VALID_STATUSES = {"ADOPTED", "PENDING", "NO"}


def validate_manual_entries(df: pd.DataFrame) -> list[str]:
    """Validate manually filled entries. Returns list of error messages."""
    errors: list[str] = []
    
    required = ["text_unit_id", "entity", "type", "status", "evidence"]
    for col in required:
        if col not in df.columns:
            errors.append(f"缺少必需列: {col}")
    
    if errors:
        return errors
    
    for idx, row in df.iterrows():
        entity = str(row.get("entity") or "").strip()
        typ = str(row.get("type") or "").strip().upper()
        status = str(row.get("status") or "").strip().upper()
        
        if not entity or entity.lower() == "nan":
            continue  # skip empty rows (should already be filtered)
        if typ not in VALID_TYPES:
            errors.append(f"行 {idx}: type='{typ}' 不合法，应为 {sorted(VALID_TYPES)}")
        if status not in VALID_STATUSES:
            errors.append(f"行 {idx}: status='{status}' 不合法，应为 {sorted(VALID_STATUSES)}")
    
    return errors


def apply_manual_review(
    review_csv: Path,
    project_root: Path,
    years: list[int] | None = None,
    dry_run: bool = False,
) -> None:
    """Read manual review, validate, merge to Stage2 CSV, re-run mapping+aggregation."""
    review = pd.read_csv(review_csv, dtype={"text_unit_id": str, "stock_code": str, "year": str})
    
    # Filter out empty rows (entity is NaN, empty, "nan", or whitespace)
    entity_raw = review["entity"].fillna("").astype(str).str.strip()
    review = review[~entity_raw.isin(["", "nan", "none", "null"])]
    
    if review.empty:
        print("没有已填写的人工审核行，退出。")
        return
    
    errors = validate_manual_entries(review)
    if errors:
        print("验证错误:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    
    target_years = set(int(row["year"]) for _, row in review.iterrows())
    if years:
        target_years = target_years & set(years)
    
    if not target_years:
        print("没有需要处理的年份。")
        return
    
    print(f"将处理 {len(review)} 条人工审核结果，涉及年份: {sorted(target_years)}")
    
    if dry_run:
        print("\n[Dry-run] 预览:")
        for _, row in review.iterrows():
            print(f"  {row['text_unit_id']}: entity={row['entity']}, type={row['type']}, status={row['status']}")
        return
    
    for year in sorted(target_years):
        measurement_root = project_root / "data" / "measurement"
        base_dir = measurement_root / str(year) / "main_regression"
        paths = default_main_regression_paths(project_root, str(year), base_dir=base_dir)
        year_review = review[review["year"].astype(int) == year]
        
        # Build entity rows
        entity_rows = []
        for _, r in year_review.iterrows():
            entity_rows.append({
                "text_unit_id": str(r["text_unit_id"]),
                "stock_code": str(r["stock_code"]),
                "company_name": str(r["company_name"]),
                "year": str(r["year"]),
                "entity": str(r["entity"]).strip(),
                "type": str(r["type"]).strip().upper(),
                "status": str(r["status"]).strip().upper(),
                "evidence": str(r.get("evidence", "")).strip() or "人工审核",
            })
        
        # Append to existing Stage2 CSV
        stage2_csv = paths.stage2_result_path
        existing = pd.read_csv(stage2_csv, dtype={"text_unit_id": str, "stock_code": str, "year": str})
        new_rows = pd.DataFrame(entity_rows, columns=TEXT_UNIT_EXTRACTION_COLUMNS)
        
        # Remove old failure sentinels for these IDs
        manual_ids = set(new_rows["text_unit_id"])
        existing = existing[
            ~(
                existing["text_unit_id"].isin(manual_ids)
                & existing["entity"].astype(str).str.upper().eq("ERROR")
            )
        ]
        
        merged = pd.concat([existing, new_rows], ignore_index=True)
        safe_to_csv(merged, stage2_csv)
        
        print(f"\n{year}: 追加 {len(new_rows)} 行 → {stage2_csv}")
        
        # Re-run mapping
        gb_dict = project_root / "GB映射参考数据库" / "GB_dict.csv"
        run_gb_mapping(stage2_csv, gb_dict, paths.mapped_result_path)
        print(f"{year}: 映射完成 → {paths.mapped_result_path}")
        
        # Re-run aggregation
        run_aggregate_main(paths.text_units_path, paths.mapped_result_path, paths.final_output_path)
        print(f"{year}: 聚合完成 → {paths.final_output_path}")
    
    print("\n完成！")


def main():
    parser = argparse.ArgumentParser(description="应用人工审核结果")
    parser.add_argument("--review-csv", default="data/measurement/manual_review/pending_76_review_template.csv")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--year", type=int, default=None, help="仅处理指定年份")
    parser.add_argument("--dry-run", action="store_true", help="预览，不实际写入")
    args = parser.parse_args()
    
    years = [args.year] if args.year else None
    apply_manual_review(
        Path(args.review_csv),
        Path(args.project_root).resolve(),
        years=years,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()

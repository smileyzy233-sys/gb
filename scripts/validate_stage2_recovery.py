#!/usr/bin/env python
"""Validate that Stage2 safe recovery preserved all original successful results.

Usage:
    python scripts/validate_stage2_recovery.py \\
        --before-root <backup_dir> \\
        --measurement-root data/measurement
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


TEXT_UNIT_EXTRACTION_COLUMNS = [
    "text_unit_id", "stock_code", "company_name", "year",
    "entity", "type", "status", "evidence",
]


def _read_stage2_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=TEXT_UNIT_EXTRACTION_COLUMNS)
    return pd.read_csv(path, dtype={"stock_code": str, "year": str, "text_unit_id": str})


def _success_ids(df: pd.DataFrame) -> set[str]:
    if df.empty:
        return set()
    mask = ~(
        df.get("entity", pd.Series("", index=df.index)).astype(str).str.upper().eq("ERROR")
        | df.get("type", pd.Series("", index=df.index)).astype(str).str.upper().eq("LLM_FAILURE")
        | df.get("status", pd.Series("", index=df.index)).astype(str).str.upper().eq("FAIL")
    )
    return set(df.loc[mask, "text_unit_id"].astype(str))


def _compare_rows(before: pd.DataFrame, after: pd.DataFrame) -> int:
    """Return number of rows whose content changed for shared IDs."""
    if before.empty or after.empty:
        return 0
    shared = set(before["text_unit_id"].astype(str)) & set(after["text_unit_id"].astype(str))
    if not shared:
        return 0
    before_shared = before[before["text_unit_id"].astype(str).isin(shared)].copy()
    after_shared = after[after["text_unit_id"].astype(str).isin(shared)].copy()

    content_cols = ["entity", "type", "status", "evidence"]

    before_hash = before_shared.groupby("text_unit_id", sort=False).apply(
        lambda g: hashlib.sha256(
            g[content_cols].to_csv(index=False, header=False).encode("utf-8")
        ).hexdigest()
    )
    after_hash = after_shared.groupby("text_unit_id", sort=False).apply(
        lambda g: hashlib.sha256(
            g[content_cols].to_csv(index=False, header=False).encode("utf-8")
        ).hexdigest()
    )

    changed = 0
    for tid in shared:
        if before_hash.get(tid) != after_hash.get(tid):
            changed += 1
    return changed


def validate_recovery(before_root: Path, measurement_root: Path) -> dict:
    """Run full validation. Returns summary dict."""
    years = list(range(2015, 2026))
    results = {
        "years": {},
        "old_success_rows_changed": 0,
        "old_success_ids_regenerated": 0,
        "unexpected_new_ids": 0,
        "resume_duplicate_rows": 0,
        "total_recovered": 0,
    }

    for year in years:
        before_csv = before_root / str(year) / "main_regression" / "results" / f"05_stage2_entity_result_{year}.csv"
        after_csv = measurement_root / str(year) / "main_regression" / "results" / f"05_stage2_entity_result_{year}.csv"

        before_df = _read_stage2_csv(before_csv)
        after_df = _read_stage2_csv(after_csv)

        before_success = _success_ids(before_df)
        after_success = _success_ids(after_df)

        # 1. Old success IDs must all remain
        missing = before_success - after_success
        # 2. New IDs must be subset of previously unresolved
        new_ids = after_success - before_success

        # Load stage2 input to compute unresolved
        stage2_input = measurement_root / str(year) / "main_regression" / "stage" / f"04_stage2_input_{year}.csv"
        unresolved_before: set[str] = set()
        if stage2_input.exists():
            input_df = pd.read_csv(stage2_input, dtype={"text_unit_id": str})
            all_input_ids = set(input_df["text_unit_id"].astype(str))
            unresolved_before = all_input_ids - before_success

        unexpected = new_ids - unresolved_before

        # 3. Check content unchanged for old success IDs
        changed = _compare_rows(before_df, after_df)

        # 4. Check for duplicate rows within after
        duplicate_count = 0
        if not after_df.empty:
            dupes = after_df[after_df.duplicated(subset=["text_unit_id", "entity", "type", "status"], keep=False)]
            duplicate_count = len(dupes)

        year_summary = {
            "before_success_count": len(before_success),
            "after_success_count": len(after_success),
            "missing_old_ids": len(missing),
            "new_ids_count": len(new_ids),
            "unexpected_new_ids": len(unexpected),
            "content_changed_ids": changed,
            "duplicate_rows": duplicate_count,
            "recovered_count": len(new_ids & unresolved_before),
        }

        results["years"][str(year)] = year_summary
        results["old_success_rows_changed"] += changed
        results["old_success_ids_regenerated"] += len(missing)
        results["unexpected_new_ids"] += len(unexpected)
        results["resume_duplicate_rows"] += duplicate_count
        results["total_recovered"] += len(new_ids & unresolved_before)

    return results


def print_summary(results: dict) -> None:
    print("\n" + "=" * 80)
    print("Stage2 Safe Recovery Validation Report")
    print("=" * 80)

    rows = []
    for year, summary in sorted(results["years"].items()):
        rows.append([
            year,
            summary["before_success_count"],
            summary["after_success_count"],
            summary["recovered_count"],
            summary["missing_old_ids"],
            summary["unexpected_new_ids"],
            summary["content_changed_ids"],
            summary["duplicate_rows"],
        ])

    print(f"{'Year':<6} {'Before':>8} {'After':>8} {'Recovered':>10} {'Missing':>8} {'UnexpNew':>9} {'Changed':>8} {'Dupes':>6}")
    print("-" * 80)
    for r in rows:
        print(f"{r[0]:<6} {r[1]:>8} {r[2]:>8} {r[3]:>10} {r[4]:>8} {r[5]:>9} {r[6]:>8} {r[7]:>6}")

    print("-" * 80)
    print(f"\nTotals:")
    print(f"  old_success_rows_changed  = {results['old_success_rows_changed']}")
    print(f"  old_success_ids_regenerated = {results['old_success_ids_regenerated']}")
    print(f"  unexpected_new_ids        = {results['unexpected_new_ids']}")
    print(f"  resume_duplicate_rows     = {results['resume_duplicate_rows']}")
    print(f"  total_recovered           = {results['total_recovered']}")

    all_ok = (
        results["old_success_rows_changed"] == 0
        and results["old_success_ids_regenerated"] == 0
        and results["unexpected_new_ids"] == 0
        and results["resume_duplicate_rows"] == 0
    )

    if all_ok:
        print("\n✓ ALL CHECKS PASSED — Safe recovery preserved original results.")
    else:
        print("\n✗ SOME CHECKS FAILED — Review the issues above before proceeding.")

    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Stage2 safe recovery results.")
    parser.add_argument("--before-root", required=True, help="Backup directory root")
    parser.add_argument("--measurement-root", default="data/measurement", help="Current measurement root")
    parser.add_argument("--json-output", default=None, help="Write full results as JSON")
    args = parser.parse_args()

    before_root = Path(args.before_root)
    measurement_root = Path(args.measurement_root)

    if not before_root.exists():
        print(f"ERROR: Before root does not exist: {before_root}", file=sys.stderr)
        sys.exit(1)

    results = validate_recovery(before_root, measurement_root)
    print_summary(results)

    if args.json_output:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(results, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"\nFull results written to: {output_path}")

    all_ok = (
        results["old_success_rows_changed"] == 0
        and results["old_success_ids_regenerated"] == 0
        and results["unexpected_new_ids"] == 0
        and results["resume_duplicate_rows"] == 0
    )
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

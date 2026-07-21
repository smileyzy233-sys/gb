from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pandas as pd


def safe_to_csv(df: pd.DataFrame, path: str | Path, **kwargs: Any) -> None:
    options: dict[str, Any] = {
        "index": False,
        "encoding": "utf-8-sig",
        "quoting": csv.QUOTE_MINIMAL,
        "quotechar": '"',
        "doublequote": True,
        "escapechar": "\\",
        "lineterminator": "\n",
    }
    options.update(kwargs)
    if options.get("quoting") == csv.QUOTE_NONE and not options.get("escapechar"):
        options["escapechar"] = "\\"
    try:
        df.to_csv(path, **options)
    except csv.Error as exc:
        raise RuntimeError(
            f"Failed to write CSV {path} with quoting={options.get('quoting')} "
            f"escapechar={options.get('escapechar')!r} quotechar={options.get('quotechar')!r}: {exc}"
        ) from exc

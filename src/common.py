from __future__ import annotations

from pathlib import Path
import json
import time
from typing import Any

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "reports" / "outputs"
MODELS = ROOT / "models"


def ensure_dirs() -> None:
    for path in (RAW, INTERIM, PROCESSED, OUTPUTS, MODELS):
        path.mkdir(parents=True, exist_ok=True)


def get_json(url: str, params: dict | None = None, headers: dict | None = None, retries: int = 4, timeout: int = 40) -> Any:
    last_error = None
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=timeout)
            if response.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"GET failed: {url} / {last_error}")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def normalize_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce").dt.tz_convert(None).dt.normalize()


def safe_save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

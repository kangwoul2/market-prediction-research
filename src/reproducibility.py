from __future__ import annotations

from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
import platform
import subprocess
import sys

from .common import ROOT, RAW, INTERIM, PROCESSED, OUTPUTS, write_json
from .research_cache import file_sha256, stable_hash

PACKAGES = [
    "pandas", "numpy", "scikit-learn", "lightgbm", "catboost", "xgboost",
    "yfinance", "requests", "nltk", "ta", "joblib", "pyarrow",
]


def _package_versions() -> dict[str, str]:
    result = {}
    for name in PACKAGES:
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = "NOT_INSTALLED"
    return result


def _git_info() -> dict:
    result = {"head": "UNKNOWN", "status": "UNKNOWN"}
    try:
        result["head"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
        ).stdout.strip() or "UNKNOWN"
        result["status"] = subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=False
        ).stdout.strip()
    except Exception as exc:
        result["error"] = str(exc)
    return result


def _hash_files(paths) -> dict[str, str]:
    result = {}
    for path in paths:
        if path.exists() and path.is_file():
            try:
                key = str(path.relative_to(ROOT))
            except Exception:
                key = str(path)
            result[key] = file_sha256(path)
    return result


def write_reproducibility_manifest() -> dict:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    source_files = [ROOT / "run_pipeline.py", ROOT / "RUN_ME.bat", ROOT / "requirements.txt"]
    source_files += sorted((ROOT / "src").glob("*.py"))
    data_files = []
    for directory in (RAW, INTERIM, PROCESSED):
        if directory.exists():
            data_files += sorted(p for p in directory.iterdir() if p.is_file())

    try:
        freeze = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
    except Exception as exc:
        freeze = f"pip freeze failed: {exc}"
    (OUTPUTS / "pip_freeze.txt").write_text(freeze + "\n", encoding="utf-8")

    config_path = OUTPUTS / "deep_research_config_lock.json"
    move_config_path = OUTPUTS / "move_research_config_lock.json"
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": _package_versions(),
        "git": _git_info(),
        "source_hashes": _hash_files(source_files),
        "data_hashes": _hash_files(data_files),
        "deep_config_sha256": file_sha256(config_path),
        "move_config_sha256": file_sha256(move_config_path),
    }
    manifest["reproducibility_id"] = stable_hash({
        "python": manifest["python_version"],
        "packages": manifest["packages"],
        "source_hashes": manifest["source_hashes"],
        "data_hashes": manifest["data_hashes"],
        "deep_config": manifest["deep_config_sha256"],
        "move_config": manifest["move_config_sha256"],
    })
    write_json(OUTPUTS / "reproducibility_manifest.json", manifest)
    return manifest

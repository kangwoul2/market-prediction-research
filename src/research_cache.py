from __future__ import annotations

from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from .common import ROOT

CACHE_DIR = ROOT / "reports" / "cache"
PIPELINE_STATE = CACHE_DIR / "pipeline_state.json"
EXPERIMENT_CACHE = CACHE_DIR / "deep_experiments.json"
CACHE_SCHEMA = "v1"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in sorted(value.items(), key=lambda x: str(x[0]))}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def stable_hash(value: Any) -> str:
    payload = json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return "MISSING"
    digest = sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def files_fingerprint(paths: list[Path]) -> dict[str, str]:
    return {str(p.relative_to(ROOT) if p.is_absolute() and ROOT in p.parents else p): file_sha256(p) for p in paths}


def stage_signature(*, sources: list[Path] | None = None, inputs: list[Path] | None = None, params: dict | None = None) -> str:
    return stable_hash({
        "schema": CACHE_SCHEMA,
        "sources": files_fingerprint(sources or []),
        "inputs": files_fingerprint(inputs or []),
        "params": params or {},
    })


def _read_json(path: Path, default: Any) -> Any:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return default


def _write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temp.replace(path)


class PipelineCheckpoint:
    def __init__(self) -> None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.state = _read_json(PIPELINE_STATE, {"schema": CACHE_SCHEMA, "stages": {}})
        if self.state.get("schema") != CACHE_SCHEMA:
            self.state = {"schema": CACHE_SCHEMA, "stages": {}}

    def is_fresh(self, name: str, signature: str, outputs: list[Path] | None = None) -> bool:
        record = self.state.get("stages", {}).get(name)
        if not record or record.get("signature") != signature or record.get("status") != "ok":
            return False
        return all(path.exists() for path in (outputs or []))

    def mark_ok(self, name: str, signature: str, meta: dict | None = None) -> None:
        self.state.setdefault("stages", {})[name] = {
            "status": "ok",
            "signature": signature,
            "completed_at": datetime.now().isoformat(timespec="seconds"),
            "meta": meta or {},
        }
        _write_json_atomic(PIPELINE_STATE, self.state)

    def mark_failed(self, name: str, signature: str, error: str) -> None:
        self.state.setdefault("stages", {})[name] = {
            "status": "failed",
            "signature": signature,
            "failed_at": datetime.now().isoformat(timespec="seconds"),
            "error": error,
        }
        _write_json_atomic(PIPELINE_STATE, self.state)


class ExperimentCache:
    def __init__(self, context: dict[str, Any]) -> None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.context_hash = stable_hash({"schema": CACHE_SCHEMA, **context})
        raw = _read_json(EXPERIMENT_CACHE, {"schema": CACHE_SCHEMA, "contexts": {}})
        if raw.get("schema") != CACHE_SCHEMA:
            raw = {"schema": CACHE_SCHEMA, "contexts": {}}
        self.data = raw
        self.records = self.data.setdefault("contexts", {}).setdefault(self.context_hash, {})

    def key(self, kind: str, config: dict[str, Any]) -> str:
        return stable_hash({"kind": kind, "config": config})

    def get(self, kind: str, config: dict[str, Any]) -> dict | None:
        return self.records.get(self.key(kind, config))

    def set(self, kind: str, config: dict[str, Any], payload: dict) -> None:
        key = self.key(kind, config)
        self.records[key] = {
            "kind": kind,
            "config": _jsonable(config),
            "payload": _jsonable(payload),
            "completed_at": datetime.now().isoformat(timespec="seconds"),
        }
        _write_json_atomic(EXPERIMENT_CACHE, self.data)

    def payload(self, kind: str, config: dict[str, Any]) -> dict | None:
        record = self.get(kind, config)
        return record.get("payload") if record else None

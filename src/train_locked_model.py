from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

from .common import PROCESSED, OUTPUTS, MODELS
from .deep_research import _feature_groups, _make_dynamic_labels, _select_features
from .modeling import _optional_models, _validate_temporal_dataset


def train_locked_model() -> dict:
    config_path = OUTPUTS / "deep_research_config_lock.json"
    dataset_path = PROCESSED / "model_dataset.csv"
    if not config_path.exists() or not dataset_path.exists():
        raise RuntimeError("deep research config 또는 model_dataset.csv가 없습니다.")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    raw = pd.read_csv(dataset_path, parse_dates=["date", "target_end_date"])
    frame = _validate_temporal_dataset(raw)
    tuned, label_col, _ = _make_dynamic_labels(
        frame,
        int(config["selected_volatility_window_days"]),
        float(config["selected_dynamic_k"]),
    )

    window = config["selected_training_window_days"]
    train = tuned.dropna(subset=[label_col]).copy()
    if window != "expanding":
        window = int(window)
        cutoff = pd.Timestamp(train["date"].max()) - pd.Timedelta(days=window)
        train = train[train["date"] >= cutoff].copy()

    groups = _feature_groups(train)
    feature_set = config["selected_feature_set"]
    model_name = config["selected_classifier"]
    if feature_set not in groups:
        raise RuntimeError(f"선택 피처군이 없습니다: {feature_set}")
    model_lookup = dict(_optional_models())
    if model_name not in model_lookup:
        raise RuntimeError(f"선택 모델을 사용할 수 없습니다: {model_name}")

    train = train.reset_index(drop=True)
    features = _select_features(
        train,
        np.arange(len(train)),
        groups[feature_set],
        label_col,
        regression=False,
        top_k=100,
    )
    model = clone(model_lookup[model_name])
    model.fit(train[features], train[label_col].astype(int))
    artifact = {
        "model": model,
        "features": features,
        "classes": np.array([-1, 0, 1]),
        "train_window_days": config["selected_training_window_days"],
        "vol_window": config["selected_volatility_window_days"],
        "k": config["selected_dynamic_k"],
        "feature_set": feature_set,
        "train_first_date": str(pd.Timestamp(train["date"].min()).date()),
        "train_last_date": str(pd.Timestamp(train["date"].max()).date()),
        "train_rows": int(len(train)),
        "status": "development_model_not_final_performance_claim",
    }
    joblib.dump(artifact, MODELS / "deep_primary_three_class_model.joblib")
    summary = {k: v for k, v in artifact.items() if k not in {"model", "features", "classes"}}
    summary["feature_count"] = len(features)
    summary["features"] = features
    (OUTPUTS / "locked_model_training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary

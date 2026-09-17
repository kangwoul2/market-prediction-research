from __future__ import annotations

from datetime import timedelta
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)

from .common import ROOT, PROCESSED, OUTPUTS, MODELS, ensure_dirs, write_json
from .deep_research import _classification_cv, _feature_groups, _make_dynamic_labels, _select_features
from .modeling import _optional_models, _validate_temporal_dataset
from .research_cache import ExperimentCache, file_sha256

CLASSES = np.array([0, 1])


def _make_move_target(frame: pd.DataFrame, vol_window: int, k: float):
    temp, _, threshold_col = _make_dynamic_labels(frame, vol_window, k)
    target = f"research_move_v{vol_window}_k{str(k).replace('.', '_')}"
    temp[target] = (temp["target_return_1d"].abs() > temp[threshold_col]).astype(int)
    return temp, target, threshold_col


def _payload_score(payload: dict) -> tuple[float, float]:
    summary = payload.get("summary", {})
    return float(summary.get("robust_score", -1)), float(summary.get("roc_auc", -1))


def _cached_cv(
    cache: ExperimentCache,
    kind: str,
    config: dict,
    frame: pd.DataFrame,
    target: str,
    features: list[str],
    model,
    train_window,
    n_splits: int,
    top_k: int,
) -> dict:
    cached = cache.payload(kind, config)
    if cached is not None:
        print(f"[CACHE HIT] {kind}: {config}")
        return cached
    print(f"[FIT] {kind}: {config}")
    result = _classification_cv(
        frame,
        target,
        CLASSES,
        features,
        model,
        train_window,
        valid_days=180,
        n_splits=n_splits,
        top_k=top_k,
    )
    if result is None:
        payload = {"summary": {"robust_score": -1, "roc_auc": -1}, "selected_features": []}
    else:
        payload = {
            "summary": result["summary"],
            "selected_features": result["selected_features"],
        }
    cache.set(kind, config, payload)
    return payload


def _row(phase: str, extra: dict, payload: dict) -> dict:
    return {"phase": phase, **extra, **payload.get("summary", {})}


def _binary_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
    }


def run_move_research_cached() -> dict:
    ensure_dirs()
    dataset_path = PROCESSED / "model_dataset.csv"
    if not dataset_path.exists():
        raise RuntimeError("model_dataset.csv가 없습니다.")

    raw = pd.read_csv(dataset_path, parse_dates=["date", "target_end_date"])
    frame = _validate_temporal_dataset(raw)
    models = dict(_optional_models())
    if "extra_trees" not in models:
        raise RuntimeError("ExtraTrees 모델을 찾을 수 없습니다.")

    context = {
        "research": "move_binary_targeted_v1",
        "dataset_sha256": file_sha256(dataset_path),
        "source_sha256": file_sha256(ROOT / "src" / "move_research_cached.py"),
        "deep_source_sha256": file_sha256(ROOT / "src" / "deep_research.py"),
    }
    cache = ExperimentCache(context)

    print("[MOVE 1/6] move target 전용 training window 탐색")
    base, base_target, _ = _make_move_target(frame, 7, 0.9)
    base_groups = _feature_groups(base)
    all_features = base_groups.get("all", next(iter(base_groups.values())))
    window_rows = []
    scout = []
    for window in (365, 730, 1095, 1825, None):
        cfg = {"window": window or "expanding", "vol": 7, "k": 0.9, "model": "extra_trees", "folds": 3}
        payload = _cached_cv(cache, "move_window_scout", cfg, base, base_target, all_features, models["extra_trees"], window, 3, 100)
        row = _row("scout", {"train_window_days": window or "expanding", "model": "extra_trees"}, payload)
        window_rows.append(row)
        scout.append((window, payload))
    top_windows = [x[0] for x in sorted(scout, key=lambda x: _payload_score(x[1]), reverse=True)[:2]]
    confirms = []
    for window in top_windows:
        cfg = {"window": window or "expanding", "vol": 7, "k": 0.9, "model": "extra_trees", "folds": 4}
        payload = _cached_cv(cache, "move_window_confirm", cfg, base, base_target, all_features, models["extra_trees"], window, 4, 100)
        window_rows.append(_row("confirm", {"train_window_days": window or "expanding", "model": "extra_trees"}, payload))
        confirms.append((window, payload))
    best_window = max(confirms, key=lambda x: _payload_score(x[1]))[0]
    pd.DataFrame(window_rows).sort_values(["phase", "robust_score"], ascending=[True, False]).to_csv(
        OUTPUTS / "move_window_sweep.csv", index=False
    )

    print("[MOVE 2/6] move target 전용 volatility window와 k 탐색")
    label_rows = []
    label_scout = []
    for vol in (7, 14, 21, 30, 60):
        for k in (0.5, 0.7, 0.9, 1.1):
            temp, target, _ = _make_move_target(frame, vol, k)
            groups = _feature_groups(temp)
            features = groups.get("all", next(iter(groups.values())))
            cfg = {"window": best_window or "expanding", "vol": vol, "k": k, "model": "extra_trees", "folds": 3}
            payload = _cached_cv(cache, "move_label_scout", cfg, temp, target, features, models["extra_trees"], best_window, 3, 100)
            share = float(temp[target].mean())
            label_rows.append(_row("scout", {"vol_window": vol, "k": k, "move_share": share, "model": "extra_trees"}, payload))
            label_scout.append((vol, k, share, payload))
    promoted_labels = sorted(label_scout, key=lambda x: _payload_score(x[3]), reverse=True)[:3]
    label_confirm = []
    for vol, k, share, _ in promoted_labels:
        temp, target, _ = _make_move_target(frame, vol, k)
        features = _feature_groups(temp).get("all")
        cfg = {"window": best_window or "expanding", "vol": vol, "k": k, "model": "extra_trees", "folds": 4}
        payload = _cached_cv(cache, "move_label_confirm", cfg, temp, target, features, models["extra_trees"], best_window, 4, 100)
        label_rows.append(_row("confirm", {"vol_window": vol, "k": k, "move_share": share, "model": "extra_trees"}, payload))
        label_confirm.append((vol, k, payload))
    best_vol, best_k, _ = max(label_confirm, key=lambda x: _payload_score(x[2]))
    pd.DataFrame(label_rows).sort_values(["phase", "robust_score"], ascending=[True, False]).to_csv(
        OUTPUTS / "move_label_sweep.csv", index=False
    )

    print("[MOVE 3/6] move target 전용 feature family와 모델 ablation")
    tuned, target, threshold_col = _make_move_target(frame, best_vol, best_k)
    groups = _feature_groups(tuned)
    feature_rows = []
    feature_scout = []
    for feature_name, features in groups.items():
        cfg = {"feature_set": feature_name, "window": best_window or "expanding", "vol": best_vol, "k": best_k, "model": "extra_trees", "folds": 3}
        payload = _cached_cv(cache, "move_feature_scout", cfg, tuned, target, features, models["extra_trees"], best_window, 3, 100)
        feature_rows.append(_row("scout", {"feature_set": feature_name, "model": "extra_trees"}, payload))
        feature_scout.append((feature_name, payload))
    promoted_features = [x[0] for x in sorted(feature_scout, key=lambda x: _payload_score(x[1]), reverse=True)[:3]]
    confirm_models = [m for m in ("extra_trees", "catboost", "lightgbm") if m in models]
    feature_confirm = []
    for feature_name in promoted_features:
        for model_name in confirm_models:
            cfg = {
                "feature_set": feature_name,
                "window": best_window or "expanding",
                "vol": best_vol,
                "k": best_k,
                "model": model_name,
                "folds": 4,
            }
            payload = _cached_cv(
                cache, "move_feature_confirm", cfg, tuned, target, groups[feature_name],
                models[model_name], best_window, 4, 100,
            )
            feature_rows.append(_row("confirm", {"feature_set": feature_name, "model": model_name}, payload))
            feature_confirm.append((feature_name, model_name, payload))
    best_feature_set, best_model_name, _ = max(feature_confirm, key=lambda x: _payload_score(x[2]))
    pd.DataFrame(feature_rows).sort_values(["phase", "robust_score"], ascending=[True, False]).to_csv(
        OUTPUTS / "move_feature_model_ablation.csv", index=False
    )

    print("[MOVE 4/6] train-only feature top-k 길이 탐색")
    topk_rows = []
    topk_results = []
    for top_k in (30, 50, 75, 100, 150):
        cfg = {
            "feature_set": best_feature_set,
            "model": best_model_name,
            "window": best_window or "expanding",
            "vol": best_vol,
            "k": best_k,
            "top_k": top_k,
            "folds": 4,
        }
        payload = _cached_cv(
            cache, "move_topk", cfg, tuned, target, groups[best_feature_set],
            models[best_model_name], best_window, 4, top_k,
        )
        topk_rows.append(_row("confirm", {"top_k": top_k, "feature_set": best_feature_set, "model": best_model_name}, payload))
        topk_results.append((top_k, payload))
    best_top_k = max(topk_results, key=lambda x: _payload_score(x[1]))[0]
    pd.DataFrame(topk_rows).sort_values("robust_score", ascending=False).to_csv(
        OUTPUTS / "move_feature_topk_sweep.csv", index=False
    )

    print("[MOVE 5/6] 최종 6-fold OOF와 decision threshold 시간순 검증")
    final = _classification_cv(
        tuned,
        target,
        CLASSES,
        groups[best_feature_set],
        models[best_model_name],
        best_window,
        valid_days=180,
        n_splits=6,
        top_k=best_top_k,
    )
    if final is None:
        raise RuntimeError("move final OOF 생성 실패")
    mask = final["mask"]
    idx = np.flatnonzero(mask)
    y = tuned.iloc[idx][target].astype(int).to_numpy()
    p = final["proba"][mask][:, 1]
    cut = max(60, int(len(idx) * 0.60))
    cut = min(cut, len(idx) - 30)
    y_select, p_select = y[:cut], p[:cut]
    y_verify, p_verify = y[cut:], p[cut:]
    threshold_rows = []
    for threshold in np.arange(0.25, 0.751, 0.025):
        pred = (p_select >= threshold).astype(int)
        threshold_rows.append({"threshold": round(float(threshold), 3), **_binary_metrics(y_select, pred)})
    threshold_df = pd.DataFrame(threshold_rows).sort_values(
        ["macro_f1", "balanced_accuracy"], ascending=[False, False]
    ).reset_index(drop=True)
    threshold_df.to_csv(OUTPUTS / "move_threshold_selection.csv", index=False)
    decision_threshold = float(threshold_df.iloc[0]["threshold"])
    verify_pred = (p_verify >= decision_threshold).astype(int)
    verify = {
        "decision_threshold": decision_threshold,
        "selection_rows": int(len(y_select)),
        "verification_rows": int(len(y_verify)),
        **_binary_metrics(y_verify, verify_pred),
        "roc_auc": float(roc_auc_score(y_verify, p_verify)),
        "pr_auc": float(average_precision_score(y_verify, p_verify)),
        "brier": float(brier_score_loss(y_verify, p_verify)),
    }
    majority = np.zeros_like(y_verify)
    current_return = tuned.iloc[idx[cut:]]["btc_return_1d"].abs().to_numpy()
    current_threshold = tuned.iloc[idx[cut:]][threshold_col].to_numpy()
    previous_move = (current_return > current_threshold).astype(int)
    baselines = pd.DataFrame([
        {"baseline": "always_no_move", **_binary_metrics(y_verify, majority)},
        {"baseline": "previous_day_move", **_binary_metrics(y_verify, previous_move)},
    ])
    baselines.to_csv(OUTPUTS / "move_verification_baselines.csv", index=False)
    write_json(OUTPUTS / "move_threshold_verification.json", verify)

    print("[MOVE 6/6] prospective용 development config 잠금과 모델 저장")
    supervised = tuned.dropna(subset=[target, "target_end_date"]).copy()
    reviewed_through = pd.Timestamp(supervised["target_end_date"].max()).date()
    prospective_start = reviewed_through + timedelta(days=1)
    train = supervised.copy()
    if best_window is not None:
        cutoff = pd.Timestamp(train["date"].max()) - pd.Timedelta(days=int(best_window))
        train = train[train["date"] >= cutoff].copy()
    train = train.reset_index(drop=True)
    final_features = _select_features(
        train,
        np.arange(len(train)),
        groups[best_feature_set],
        target,
        regression=False,
        top_k=best_top_k,
    )
    model = clone(models[best_model_name])
    model.fit(train[final_features], train[target].astype(int))
    artifact = {
        "model": model,
        "features": final_features,
        "classes": CLASSES,
        "decision_threshold": decision_threshold,
        "train_window_days": "expanding" if best_window is None else best_window,
        "vol_window": best_vol,
        "k": best_k,
        "feature_set": best_feature_set,
        "top_k": best_top_k,
        "status": "development_model_not_final_performance_claim",
    }
    joblib.dump(artifact, MODELS / "move_binary_model.joblib")

    summary = {
        "research_status": "development_cv_only",
        "task": "next_day_meaningful_move_binary",
        "definition": "t일 종료 시점까지 관측 가능한 정보로 t+1일 BTC 절대수익률이 k×trailing volatility를 넘는지 예측",
        "selected_training_window_days": "expanding" if best_window is None else best_window,
        "selected_volatility_window_days": best_vol,
        "selected_dynamic_k": best_k,
        "selected_feature_set": best_feature_set,
        "selected_classifier": best_model_name,
        "selected_top_k": best_top_k,
        "decision_threshold": decision_threshold,
        "final_oof": final["summary"],
        "chronological_threshold_verification": verify,
        "development_data_reviewed_through": str(reviewed_through),
        "prospective_start_date": str(prospective_start),
        "feature_count": len(final_features),
        "features": final_features,
        "cache_context": cache.context_hash,
        "important_rule": "threshold는 OOF 앞쪽 구간에서 선택하고 뒤쪽 OOF 구간에 고정 검증한다. 최종 성능 주장은 prospective_start_date 이후 새 데이터로만 판단한다.",
    }
    write_json(OUTPUTS / "move_research_summary.json", summary)
    write_json(OUTPUTS / "move_research_config_lock.json", summary)
    return summary

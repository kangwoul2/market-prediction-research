from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.base import clone

from .common import PROCESSED, OUTPUTS, ensure_dirs, write_json
from .deep_research import (
    _classification_cv,
    _feature_groups,
    _make_dynamic_labels,
    _rolling_splits,
)
from .modeling import _metrics, _optional_models, _validate_temporal_dataset

CLASSES = np.array([-1, 0, 1])


def _seeded_model(prototype, seed: int):
    model = clone(prototype)
    params = model.get_params(deep=True)
    if "model__random_state" in params:
        model.set_params(model__random_state=seed)
    if "model__random_seed" in params:
        model.set_params(model__random_seed=seed)
    return model


def run_deep_post_analysis() -> dict:
    ensure_dirs()
    config_path = OUTPUTS / "deep_research_config_lock.json"
    dataset_path = PROCESSED / "model_dataset.csv"
    if not config_path.exists() or not dataset_path.exists():
        raise RuntimeError("deep research 결과가 없습니다.")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    raw = pd.read_csv(dataset_path, parse_dates=["date", "target_end_date"])
    frame = _validate_temporal_dataset(raw)
    tuned, label_col, _ = _make_dynamic_labels(
        frame,
        int(config["selected_volatility_window_days"]),
        float(config["selected_dynamic_k"]),
    )
    groups = _feature_groups(tuned)
    feature_set = config["selected_feature_set"]
    candidates = groups[feature_set]
    window_cfg = config["selected_training_window_days"]
    train_window = None if window_cfg == "expanding" else int(window_cfg)

    models = dict(_optional_models())
    promoted = config.get("promoted_models") or [config.get("selected_classifier")]
    candidate_names = [name for name in promoted if name in models][:3]
    if not candidate_names:
        candidate_names = [m for m in ("random_forest", "lightgbm", "extra_trees") if m in models][:3]

    results = {}
    model_rows = []
    print("[POST 1/4] 상위 후보 모델만 선택 설정에서 OOF 재생성")
    for name in candidate_names:
        result = _classification_cv(
            tuned, label_col, CLASSES, candidates, models[name], train_window
        )
        if result is None:
            continue
        results[name] = result
        model_rows.append({"model": name, **result["summary"]})
    model_df = pd.DataFrame(model_rows).sort_values("robust_score", ascending=False).reset_index(drop=True)
    model_df.to_csv(OUTPUTS / "deep_selected_config_models.csv", index=False)
    if not results:
        summary = {"status": "no_models", "models": []}
        write_json(OUTPUTS / "deep_post_summary.json", summary)
        return summary

    print("[POST 2/4] OOF 확률 앙상블과 confidence curve")
    common_mask = np.ones(len(tuned), dtype=bool)
    for result in results.values():
        common_mask &= result["mask"]
    idx = np.flatnonzero(common_mask)
    truth = tuned.iloc[idx][label_col].astype(int).to_numpy()

    raw_weights = np.array([
        max(float(results[name]["summary"]["robust_score"]), 1e-6)
        for name in results
    ])
    weights = raw_weights / raw_weights.sum()
    ensemble = np.zeros((len(idx), len(CLASSES)), dtype=float)
    member_rows = []
    for weight, name in zip(weights, results):
        ensemble += weight * results[name]["proba"][common_mask]
        member_rows.append({
            "model": name,
            "weight": float(weight),
            "robust_score": float(results[name]["summary"]["robust_score"]),
            "macro_f1": float(results[name]["summary"]["macro_f1"]),
        })
    ensemble_pred = CLASSES[np.argmax(ensemble, axis=1)]
    ensemble_metrics = _metrics(truth, ensemble_pred, ensemble, CLASSES)

    fold_rows = []
    for fold, _, va in _rolling_splits(tuned, train_window, 180, 6):
        active_global = np.intersect1d(idx, va)
        if len(active_global) == 0:
            continue
        pos = np.searchsorted(idx, active_global)
        fold_truth = tuned.iloc[active_global][label_col].astype(int).to_numpy()
        fold_pred = CLASSES[np.argmax(ensemble[pos], axis=1)]
        fold_rows.append({"fold": fold, **_metrics(fold_truth, fold_pred, ensemble[pos], CLASSES)})
    fold_df = pd.DataFrame(fold_rows)
    if not fold_df.empty:
        ensemble_metrics["fold_macro_f1_mean"] = float(fold_df["macro_f1"].mean())
        ensemble_metrics["fold_macro_f1_std"] = float(fold_df["macro_f1"].std(ddof=0))
        ensemble_metrics["fold_macro_f1_worst"] = float(fold_df["macro_f1"].min())
        ensemble_metrics["robust_score"] = float(
            fold_df["macro_f1"].mean() - 0.25 * fold_df["macro_f1"].std(ddof=0)
        )
    pd.DataFrame(member_rows).to_csv(OUTPUTS / "deep_ensemble_members.csv", index=False)
    fold_df.to_csv(OUTPUTS / "deep_ensemble_fold_metrics.csv", index=False)

    confidence = ensemble.max(axis=1)
    selective_rows = []
    for threshold in np.arange(0.35, 0.81, 0.05):
        active = confidence >= threshold
        if active.sum() < 30:
            continue
        selective_rows.append({
            "confidence_threshold": round(float(threshold), 2),
            "coverage": float(active.mean()),
            "samples": int(active.sum()),
            **_metrics(truth[active], ensemble_pred[active]),
        })
    pd.DataFrame(selective_rows).to_csv(OUTPUTS / "deep_selective_curve.csv", index=False)

    prediction_df = tuned.iloc[idx][["date", "target_end_date", "target_return_1d"]].copy()
    prediction_df["label"] = truth
    prediction_df["ensemble_pred"] = ensemble_pred
    prediction_df["confidence"] = confidence
    for i, cls in enumerate(CLASSES):
        prediction_df[f"prob_{cls}"] = ensemble[:, i]
    prediction_df.to_csv(OUTPUTS / "deep_oof_ensemble_predictions.csv", index=False)

    print("[POST 3/4] 변동성 국면과 추세 국면 분석")
    regime_rows = []
    vol_regime = pd.to_numeric(tuned.iloc[idx]["btc_vol_regime_pctile"], errors="coerce").to_numpy()
    trend_series = tuned.iloc[idx]["btc_ema_gap_90d"] if "btc_ema_gap_90d" in tuned.columns else pd.Series(np.nan, index=tuned.iloc[idx].index)
    trend = pd.to_numeric(trend_series, errors="coerce").to_numpy()
    for name, mask in (
        ("vol_low", vol_regime < 0.33),
        ("vol_mid", (vol_regime >= 0.33) & (vol_regime < 0.67)),
        ("vol_high", vol_regime >= 0.67),
        ("trend_above_ema90", trend >= 0),
        ("trend_below_ema90", trend < 0),
    ):
        active = mask & np.isfinite(confidence)
        if active.sum() < 30:
            continue
        regime_rows.append({
            "regime": name,
            "samples": int(active.sum()),
            **_metrics(truth[active], ensemble_pred[active]),
        })
    pd.DataFrame(regime_rows).to_csv(OUTPUTS / "deep_regime_breakdown.csv", index=False)

    print("[POST 4/4] 최종 선택 모델의 랜덤 시드 안정성")
    seed_rows = []
    stability_model = str(config.get("selected_classifier", candidate_names[0]))
    if stability_model not in models:
        stability_model = candidate_names[0]
    for seed in (11, 42, 73, 104, 137):
        seeded = _seeded_model(models[stability_model], seed)
        result = _classification_cv(
            tuned, label_col, CLASSES, candidates, seeded, train_window
        )
        if result is None:
            continue
        seed_rows.append({"model": stability_model, "seed": seed, **result["summary"]})
    seed_df = pd.DataFrame(seed_rows)
    seed_df.to_csv(OUTPUTS / "deep_seed_stability.csv", index=False)

    best_single = model_df.iloc[0].to_dict() if not model_df.empty else {}
    summary = {
        "status": "ok",
        "feature_set": feature_set,
        "train_window_days": config["selected_training_window_days"],
        "volatility_window_days": config["selected_volatility_window_days"],
        "k": config["selected_dynamic_k"],
        "evaluated_models": candidate_names,
        "seed_stability_model": stability_model,
        "best_single": best_single,
        "ensemble": ensemble_metrics,
        "members": member_rows,
        "ensemble_improved_macro_f1": bool(
            best_single and ensemble_metrics.get("macro_f1", -1) > float(best_single.get("macro_f1", -1))
        ),
        "note": "상위 후보만 OOF 앙상블하고 최종 선택 모델만 multi-seed 검증해 중복 계산을 줄였다.",
    }
    write_json(OUTPUTS / "deep_post_summary.json", summary)
    return summary

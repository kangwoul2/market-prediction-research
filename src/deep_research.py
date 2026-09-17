from __future__ import annotations

from collections import Counter
from datetime import timedelta
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline

from .common import PROCESSED, OUTPUTS, MODELS, ensure_dirs, write_json
from .modeling import _aligned_proba, _metrics, _optional_models, _validate_temporal_dataset

RANDOM_STATE = 42
MACRO_PREFIXES = (
    "spy_", "qqq_", "gold_", "dxy_", "vix_", "tlt_",
    "btc_spy_", "btc_qqq_", "btc_gold_", "btc_dxy_", "btc_vix_", "btc_tlt_",
)
TARGET_COLUMNS = {
    "date", "target_end_date", "btc_close", "target_return_1d",
    "threshold_fixed", "threshold_dynamic", "label_fixed", "label_dynamic",
    "target_move_dynamic", "target_direction",
}


def _feature_groups(frame: pd.DataFrame) -> dict[str, list[str]]:
    numeric = [
        c for c in frame.columns
        if c not in TARGET_COLUMNS
        and not c.startswith("research_")
        and pd.api.types.is_numeric_dtype(frame[c])
    ]
    macro = [c for c in numeric if c.startswith(MACRO_PREFIXES)]
    crypto = [c for c in numeric if c.startswith("btc_crypto_")]
    sentiment = [
        c for c in numeric
        if c.startswith("ext_") and (
            "fear_greed" in c or "x_sentiment" in c or "x_post" in c
            or "x_positive" in c or "x_negative" in c or "x_engagement" in c
        )
    ]
    onchain = [c for c in numeric if c.startswith("ext_") and c not in sentiment]
    price_core = [
        c for c in numeric
        if c.startswith("btc_") and c not in crypto and c not in macro
    ]
    groups = {
        "price_core": price_core,
        "price_crypto": price_core + crypto,
        "price_macro": price_core + macro,
        "price_onchain": price_core + onchain,
        "price_crypto_onchain": price_core + crypto + onchain,
        "price_all_external": price_core + crypto + macro + onchain + sentiment,
        "all": numeric,
    }
    return {k: sorted(set(v)) for k, v in groups.items() if len(set(v)) >= 5}


def _rolling_splits(
    frame: pd.DataFrame,
    train_window_days: int | None,
    valid_days: int = 180,
    n_splits: int = 6,
):
    n = len(frame)
    first_valid = n - valid_days * n_splits
    if first_valid < 400:
        valid_days = max(60, (n - 400) // n_splits)
        first_valid = n - valid_days * n_splits
    target_end = pd.to_datetime(frame["target_end_date"])
    dates = pd.to_datetime(frame["date"])

    for fold in range(n_splits):
        va_start = first_valid + fold * valid_days
        va_end = min(va_start + valid_days, n)
        if va_start <= 0 or va_end <= va_start:
            continue
        valid_idx = np.arange(va_start, va_end)
        valid_start_date = dates.iloc[va_start]
        raw_train = np.arange(0, va_start)
        keep = target_end.iloc[raw_train].to_numpy(dtype="datetime64[ns]") < np.datetime64(valid_start_date)
        train_idx = raw_train[keep]
        if train_window_days is not None:
            cutoff = valid_start_date - timedelta(days=int(train_window_days))
            train_idx = train_idx[dates.iloc[train_idx].to_numpy(dtype="datetime64[ns]") >= np.datetime64(cutoff)]
        if len(train_idx) < 180:
            continue
        if pd.Timestamp(target_end.iloc[train_idx].max()) >= valid_start_date:
            raise RuntimeError("rolling split temporal leakage detected")
        yield fold + 1, train_idx, valid_idx


def _select_features(
    frame: pd.DataFrame,
    train_idx: np.ndarray,
    candidates: list[str],
    target: str,
    regression: bool,
    top_k: int = 100,
) -> list[str]:
    train = frame.iloc[train_idx]
    cols = [
        c for c in candidates
        if c in train.columns
        and train[c].notna().mean() >= 0.60
        and train[c].nunique(dropna=True) > 2
    ]
    if len(cols) <= top_k:
        return cols

    X = train[cols].replace([np.inf, -np.inf], np.nan)
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)
    y = pd.to_numeric(train[target], errors="coerce").to_numpy()
    try:
        if regression:
            scores = mutual_info_regression(X_imp, y, random_state=RANDOM_STATE)
        else:
            scores = mutual_info_classif(X_imp, y.astype(int), random_state=RANDOM_STATE)
        order = np.argsort(np.nan_to_num(scores, nan=-1.0))[::-1][:top_k]
        return [cols[i] for i in order]
    except Exception:
        variance = np.nanvar(X_imp, axis=0)
        order = np.argsort(variance)[::-1][:top_k]
        return [cols[i] for i in order]


def _classification_cv(
    frame: pd.DataFrame,
    target: str,
    classes: np.ndarray,
    candidates: list[str],
    prototype,
    train_window_days: int | None,
    valid_days: int = 180,
    n_splits: int = 6,
    top_k: int = 100,
) -> dict | None:
    y = frame[target].astype(int)
    probs = np.full((len(frame), len(classes)), np.nan)
    fold_rows = []
    selected = []

    for fold, tr, va in _rolling_splits(frame, train_window_days, valid_days, n_splits):
        features = _select_features(frame, tr, candidates, target, regression=False, top_k=top_k)
        if len(features) < 5 or y.iloc[tr].nunique() < 2:
            continue
        model = clone(prototype)
        model.fit(frame.iloc[tr][features], y.iloc[tr])
        p = _aligned_proba(model, frame.iloc[va][features], classes)
        pred = classes[np.argmax(p, axis=1)]
        probs[va] = p
        metrics = _metrics(y.iloc[va], pred, p, classes)
        fold_rows.append({
            "fold": fold,
            "train_window_days": "expanding" if train_window_days is None else train_window_days,
            "train_rows": int(len(tr)),
            "valid_rows": int(len(va)),
            "train_first_date": str(pd.Timestamp(frame.iloc[tr]["date"].min()).date()),
            "train_last_target_end_date": str(pd.Timestamp(frame.iloc[tr]["target_end_date"].max()).date()),
            "valid_first_date": str(pd.Timestamp(frame.iloc[va]["date"].min()).date()),
            "valid_last_date": str(pd.Timestamp(frame.iloc[va]["date"].max()).date()),
            "feature_count": len(features),
            **metrics,
        })
        selected.append(features)

    mask = ~np.isnan(probs).any(axis=1)
    if mask.sum() == 0:
        return None
    pred = classes[np.argmax(probs[mask], axis=1)]
    overall = _metrics(y.iloc[np.flatnonzero(mask)], pred, probs[mask], classes)
    fold_f1 = [r["macro_f1"] for r in fold_rows]
    overall.update({
        "fold_macro_f1_mean": float(np.mean(fold_f1)),
        "fold_macro_f1_std": float(np.std(fold_f1)),
        "fold_macro_f1_worst": float(np.min(fold_f1)),
        "robust_score": float(np.mean(fold_f1) - 0.25 * np.std(fold_f1)),
        "oof_rows": int(mask.sum()),
    })
    return {
        "summary": overall,
        "fold_rows": fold_rows,
        "selected_features": selected,
        "proba": probs,
        "mask": mask,
    }


def _regression_models() -> dict[str, object]:
    models = {
        "random_forest_reg": Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("model", RandomForestRegressor(
                n_estimators=500, max_depth=9, min_samples_leaf=6,
                random_state=RANDOM_STATE, n_jobs=-1,
            )),
        ]),
        "extra_trees_reg": Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("model", ExtraTreesRegressor(
                n_estimators=600, max_depth=10, min_samples_leaf=5,
                random_state=RANDOM_STATE, n_jobs=-1,
            )),
        ]),
        "hist_gradient_reg": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingRegressor(
                learning_rate=0.04, max_iter=300, max_leaf_nodes=15,
                l2_regularization=3.0, random_state=RANDOM_STATE,
            )),
        ]),
    }
    try:
        from lightgbm import LGBMRegressor
        models["lightgbm_reg"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("model", LGBMRegressor(
                n_estimators=500, learning_rate=0.025, num_leaves=15,
                min_child_samples=25, subsample=0.85, colsample_bytree=0.8,
                reg_alpha=0.5, reg_lambda=2.0, random_state=RANDOM_STATE,
                n_jobs=-1, verbosity=-1,
            )),
        ])
    except Exception:
        pass
    try:
        from catboost import CatBoostRegressor
        models["catboost_reg"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("model", CatBoostRegressor(
                iterations=450, depth=6, learning_rate=0.035,
                loss_function="RMSE", random_seed=RANDOM_STATE,
                verbose=False, allow_writing_files=False,
            )),
        ])
    except Exception:
        pass
    return models


def _regression_metrics(y_true: np.ndarray, pred: np.ndarray, threshold: np.ndarray) -> dict:
    mae = float(mean_absolute_error(y_true, pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, pred)))
    pearson = float(pd.Series(y_true).corr(pd.Series(pred), method="pearson"))
    spearman = float(pd.Series(y_true).corr(pd.Series(pred), method="spearman"))
    sign_acc = float((np.sign(y_true) == np.sign(pred)).mean())
    true_class = np.select([y_true > threshold, y_true < -threshold], [1, -1], default=0)
    pred_class = np.select([pred > threshold, pred < -threshold], [1, -1], default=0)
    return {
        "mae": mae,
        "rmse": rmse,
        "pearson": pearson,
        "spearman": spearman,
        "sign_accuracy": sign_acc,
        "derived_class_macro_f1": float(f1_score(true_class, pred_class, average="macro", zero_division=0)),
        "derived_class_balanced_accuracy": float(balanced_accuracy_score(true_class, pred_class)),
    }


def _regression_cv(
    frame: pd.DataFrame,
    candidates: list[str],
    prototype,
    train_window_days: int | None,
    threshold_col: str,
    valid_days: int = 180,
    n_splits: int = 6,
) -> dict | None:
    target = "target_return_1d"
    y = pd.to_numeric(frame[target], errors="coerce")
    pred_all = np.full(len(frame), np.nan)
    fold_rows = []
    selected = []

    for fold, tr, va in _rolling_splits(frame, train_window_days, valid_days, n_splits):
        features = _select_features(frame, tr, candidates, target, regression=True, top_k=100)
        if len(features) < 5:
            continue
        model = clone(prototype)
        model.fit(frame.iloc[tr][features], y.iloc[tr])
        pred = model.predict(frame.iloc[va][features])
        pred_all[va] = pred
        threshold = pd.to_numeric(frame.iloc[va][threshold_col], errors="coerce").to_numpy()
        metrics = _regression_metrics(y.iloc[va].to_numpy(), pred, threshold)
        fold_rows.append({
            "fold": fold,
            "train_window_days": "expanding" if train_window_days is None else train_window_days,
            "feature_count": len(features),
            **metrics,
        })
        selected.append(features)

    mask = ~np.isnan(pred_all)
    if mask.sum() == 0:
        return None
    overall = _regression_metrics(
        y.iloc[np.flatnonzero(mask)].to_numpy(),
        pred_all[mask],
        pd.to_numeric(frame.iloc[np.flatnonzero(mask)][threshold_col], errors="coerce").to_numpy(),
    )
    corr = [r["pearson"] for r in fold_rows if np.isfinite(r["pearson"])]
    derived = [r["derived_class_macro_f1"] for r in fold_rows]
    overall.update({
        "fold_pearson_mean": float(np.mean(corr)) if corr else np.nan,
        "fold_pearson_std": float(np.std(corr)) if corr else np.nan,
        "fold_derived_macro_f1_mean": float(np.mean(derived)),
        "fold_derived_macro_f1_std": float(np.std(derived)),
        "oof_rows": int(mask.sum()),
    })
    return {"summary": overall, "fold_rows": fold_rows, "selected_features": selected, "pred": pred_all, "mask": mask}


def _make_dynamic_labels(frame: pd.DataFrame, vol_window: int, k: float) -> tuple[pd.DataFrame, str, str]:
    temp = frame.copy()
    sigma = pd.to_numeric(temp["btc_log_return_1d"], errors="coerce").rolling(
        vol_window, min_periods=max(5, vol_window // 2)
    ).std()
    threshold_col = f"research_threshold_v{vol_window}_k{str(k).replace('.', '_')}"
    label_col = f"research_label_v{vol_window}_k{str(k).replace('.', '_')}"
    temp[threshold_col] = k * sigma
    temp[label_col] = np.select(
        [temp["target_return_1d"] > temp[threshold_col], temp["target_return_1d"] < -temp[threshold_col]],
        [1, -1], default=0,
    ).astype(int)
    return temp, label_col, threshold_col


def _baseline_metrics(frame: pd.DataFrame, label_col: str, threshold_col: str) -> pd.DataFrame:
    rows = []
    for fold, _, va in _rolling_splits(frame, 1095, 180, 6):
        y = frame.iloc[va][label_col].astype(int).to_numpy()
        neutral = np.zeros(len(va), dtype=int)
        momentum = np.select(
            [
                frame.iloc[va]["btc_return_1d"].to_numpy() > frame.iloc[va][threshold_col].to_numpy(),
                frame.iloc[va]["btc_return_1d"].to_numpy() < -frame.iloc[va][threshold_col].to_numpy(),
            ],
            [1, -1], default=0,
        )
        for name, pred in (("always_neutral", neutral), ("previous_day_momentum", momentum)):
            rows.append({
                "fold": fold,
                "baseline": name,
                "accuracy": float(accuracy_score(y, pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
                "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            })
    return pd.DataFrame(rows)


def _selection_frequency(records: list[dict]) -> pd.DataFrame:
    rows = []
    for record in records:
        counter = Counter()
        folds = record.get("selected_features", [])
        for features in folds:
            counter.update(features)
        for feature, count in counter.items():
            rows.append({
                "experiment": record["experiment"],
                "feature_set": record.get("feature_set", "all"),
                "model": record.get("model", ""),
                "feature": feature,
                "selected_folds": int(count),
                "fold_count": int(len(folds)),
                "selection_rate": float(count / max(1, len(folds))),
            })
    return pd.DataFrame(rows).sort_values(["experiment", "selection_rate"], ascending=[True, False]) if rows else pd.DataFrame()


def run_deep_research() -> dict:
    ensure_dirs()
    dataset_path = PROCESSED / "model_dataset.csv"
    if not dataset_path.exists():
        raise RuntimeError("model_dataset.csv가 없습니다.")
    raw = pd.read_csv(dataset_path, parse_dates=["date", "target_end_date"])
    frame = _validate_temporal_dataset(raw)
    groups = _feature_groups(frame)
    all_features = groups.get("all", [])
    class_models = dict(_optional_models())
    strong_models = [m for m in ("lightgbm", "random_forest") if m in class_models]
    if not strong_models:
        strong_models = [next(iter(class_models))]

    selection_records = []
    print("[DEEP 1/7] 학습 슬라이딩 윈도우 길이 비교")
    window_rows = []
    for window in (365, 730, 1095, 1825, None):
        for model_name in strong_models:
            for target, classes, task in (
                ("label_dynamic", np.array([-1, 0, 1]), "three_class"),
                ("target_move_dynamic", np.array([0, 1]), "move_binary"),
            ):
                result = _classification_cv(frame, target, classes, all_features, class_models[model_name], window)
                if result is None:
                    continue
                row = {
                    "task": task,
                    "model": model_name,
                    "train_window_days": "expanding" if window is None else window,
                    **result["summary"],
                }
                window_rows.append(row)
                selection_records.append({
                    "experiment": f"window_{task}", "feature_set": "all", "model": model_name,
                    "selected_features": result["selected_features"],
                })
    window_df = pd.DataFrame(window_rows).sort_values(["task", "robust_score"], ascending=[True, False])
    window_df.to_csv(OUTPUTS / "deep_window_sweep.csv", index=False)

    three_windows = window_df[window_df["task"] == "three_class"]
    best_window_value = three_windows.iloc[0]["train_window_days"] if not three_windows.empty else 1095
    best_window = None if str(best_window_value) == "expanding" else int(best_window_value)
    best_window_model = str(three_windows.iloc[0]["model"]) if not three_windows.empty else strong_models[0]

    print("[DEEP 2/7] 동적 라벨 변동성 창과 k 비교")
    label_rows = []
    for vol_window in (7, 14, 21, 30, 60):
        for k in (0.5, 0.7, 0.9):
            temp, label_col, threshold_col = _make_dynamic_labels(frame, vol_window, k)
            result = _classification_cv(
                temp, label_col, np.array([-1, 0, 1]), all_features,
                class_models[best_window_model], best_window,
            )
            if result is None:
                continue
            shares = temp[label_col].value_counts(normalize=True).to_dict()
            label_rows.append({
                "vol_window": vol_window,
                "k": k,
                "model": best_window_model,
                "train_window_days": "expanding" if best_window is None else best_window,
                "down_share": float(shares.get(-1, 0.0)),
                "neutral_share": float(shares.get(0, 0.0)),
                "up_share": float(shares.get(1, 0.0)),
                **result["summary"],
            })
            selection_records.append({
                "experiment": "label_sweep", "feature_set": "all", "model": best_window_model,
                "selected_features": result["selected_features"],
            })
    label_df = pd.DataFrame(label_rows).sort_values("robust_score", ascending=False).reset_index(drop=True)
    label_df.to_csv(OUTPUTS / "deep_label_window_k_sweep.csv", index=False)
    best_label = label_df.iloc[0] if not label_df.empty else pd.Series({"vol_window": 21, "k": 0.7})
    best_vol = int(best_label["vol_window"])
    best_k = float(best_label["k"])
    tuned, label_col, threshold_col = _make_dynamic_labels(frame, best_vol, best_k)
    tuned["research_move"] = (tuned["target_return_1d"].abs() > tuned[threshold_col]).astype(int)

    print("[DEEP 3/7] 피처군과 모델 ablation")
    ablation_rows = []
    model_names = [m for m in ("random_forest", "extra_trees", "lightgbm", "catboost") if m in class_models]
    for feature_name, features in groups.items():
        for model_name in model_names:
            result = _classification_cv(
                tuned, label_col, np.array([-1, 0, 1]), features,
                class_models[model_name], best_window,
            )
            if result is None:
                continue
            ablation_rows.append({
                "feature_set": feature_name,
                "model": model_name,
                "train_window_days": "expanding" if best_window is None else best_window,
                "vol_window": best_vol,
                "k": best_k,
                **result["summary"],
            })
            selection_records.append({
                "experiment": "feature_model_ablation", "feature_set": feature_name, "model": model_name,
                "selected_features": result["selected_features"],
            })
    ablation_df = pd.DataFrame(ablation_rows).sort_values("robust_score", ascending=False).reset_index(drop=True)
    ablation_df.to_csv(OUTPUTS / "deep_feature_model_ablation.csv", index=False)
    best_ablation = ablation_df.iloc[0] if not ablation_df.empty else pd.Series({"feature_set": "all", "model": best_window_model})
    best_feature_set = str(best_ablation["feature_set"])
    best_model_name = str(best_ablation["model"])
    best_features = groups[best_feature_set]

    print("[DEEP 4/7] 문제 정의 비교: 3분류, 상승하락, 큰 움직임")
    task_rows = []
    task_specs = [
        ("three_class", label_col, np.array([-1, 0, 1])),
        ("direction_binary", "target_direction", np.array([0, 1])),
        ("move_binary", "research_move", np.array([0, 1])),
    ]
    for task, target, classes in task_specs:
        for model_name in model_names:
            result = _classification_cv(tuned, target, classes, best_features, class_models[model_name], best_window)
            if result is None:
                continue
            task_rows.append({"task": task, "model": model_name, **result["summary"]})
            selection_records.append({
                "experiment": f"task_{task}", "feature_set": best_feature_set, "model": model_name,
                "selected_features": result["selected_features"],
            })
    task_df = pd.DataFrame(task_rows).sort_values(["task", "robust_score"], ascending=[True, False])
    task_df.to_csv(OUTPUTS / "deep_task_formulation_comparison.csv", index=False)

    print("[DEEP 5/7] 다음날 수익률 회귀 비교")
    regression_rows = []
    regression_results = {}
    for model_name, prototype in _regression_models().items():
        result = _regression_cv(tuned, best_features, prototype, best_window, threshold_col)
        if result is None:
            continue
        regression_results[model_name] = result
        regression_rows.append({"model": model_name, **result["summary"]})
        selection_records.append({
            "experiment": "regression", "feature_set": best_feature_set, "model": model_name,
            "selected_features": result["selected_features"],
        })
    regression_df = pd.DataFrame(regression_rows).sort_values(
        ["fold_pearson_mean", "fold_derived_macro_f1_mean"], ascending=[False, False]
    ).reset_index(drop=True)
    regression_df.to_csv(OUTPUTS / "deep_regression_comparison.csv", index=False)

    print("[DEEP 6/7] 베이스라인과 피처 선택 안정성")
    baselines = _baseline_metrics(tuned, label_col, threshold_col)
    baselines.to_csv(OUTPUTS / "deep_baselines.csv", index=False)
    frequency = _selection_frequency(selection_records)
    frequency.to_csv(OUTPUTS / "deep_feature_selection_frequency.csv", index=False)

    print("[DEEP 7/7] 추천 설정 잠금")
    best_three = task_df[task_df["task"] == "three_class"].iloc[0].to_dict() if not task_df[task_df["task"] == "three_class"].empty else {}
    best_direction = task_df[task_df["task"] == "direction_binary"].iloc[0].to_dict() if not task_df[task_df["task"] == "direction_binary"].empty else {}
    best_move = task_df[task_df["task"] == "move_binary"].iloc[0].to_dict() if not task_df[task_df["task"] == "move_binary"].empty else {}
    best_regression = regression_df.iloc[0].to_dict() if not regression_df.empty else {}

    summary = {
        "research_status": "development_cv_only",
        "primary_task": "three_class_next_day_return",
        "primary_definition": "t일 종료 시점까지 관측 가능한 정보로 t+1일 BTC 수익률을 DOWN, NEUTRAL, UP으로 분류",
        "secondary_tasks": ["direction_binary", "move_binary", "return_regression"],
        "selected_training_window_days": "expanding" if best_window is None else best_window,
        "selected_volatility_window_days": best_vol,
        "selected_dynamic_k": best_k,
        "selected_feature_set": best_feature_set,
        "selected_classifier": best_model_name,
        "best_three_class": best_three,
        "best_direction_binary": best_direction,
        "best_move_binary": best_move,
        "best_regression": best_regression,
        "important_rule": "모든 선택은 시간 순 개발 CV에서만 수행하며 이미 확인한 과거 holdout을 다시 최종 테스트로 사용하지 않는다.",
    }
    write_json(OUTPUTS / "deep_research_summary.json", summary)
    write_json(OUTPUTS / "deep_research_config_lock.json", summary)

    if best_model_name in class_models:
        full_train = tuned.dropna(subset=[label_col]).reset_index(drop=True)
        final_features = _select_features(
            full_train, np.arange(len(full_train)), best_features, label_col, regression=False, top_k=100
        )
        final_model = clone(class_models[best_model_name])
        final_model.fit(full_train[final_features], full_train[label_col].astype(int))
        joblib.dump(
            {
                "model": final_model,
                "features": final_features,
                "classes": np.array([-1, 0, 1]),
                "train_window_days": best_window,
                "vol_window": best_vol,
                "k": best_k,
                "feature_set": best_feature_set,
                "status": "development_model_not_final_performance_claim",
            },
            MODELS / "deep_primary_three_class_model.joblib",
        )
    return summary

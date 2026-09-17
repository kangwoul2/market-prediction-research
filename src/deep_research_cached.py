from __future__ import annotations

import numpy as np
import pandas as pd

from .common import PROCESSED, OUTPUTS, ROOT, ensure_dirs, write_json
from .deep_research import (
    _baseline_metrics,
    _classification_cv,
    _feature_groups,
    _make_dynamic_labels,
    _regression_cv,
    _regression_models,
    _selection_frequency,
)
from .modeling import _optional_models, _validate_temporal_dataset
from .research_cache import ExperimentCache, file_sha256, stable_hash

CACHE_LOGIC_VERSION = "deep-research-pruned-v2"
SCOUT_SPLITS = 4
CONFIRM_SPLITS = 6


def _model_config(model) -> dict:
    try:
        return model.get_params(deep=True)
    except Exception:
        return {"repr": repr(model)}


def _compact(result: dict) -> dict:
    return {
        "summary": result["summary"],
        "fold_rows": result.get("fold_rows", []),
        "selected_features": result.get("selected_features", []),
    }


def _cached_classification(
    cache: ExperimentCache,
    frame: pd.DataFrame,
    target: str,
    classes: np.ndarray,
    candidates: list[str],
    prototype,
    model_name: str,
    train_window_days: int | None,
    experiment: str,
    valid_days: int = 180,
    n_splits: int = CONFIRM_SPLITS,
    top_k: int = 100,
) -> dict | None:
    config = {
        "logic": CACHE_LOGIC_VERSION,
        "experiment": experiment,
        "target": target,
        "classes": classes.tolist(),
        "candidate_hash": stable_hash(sorted(candidates)),
        "candidate_count": len(candidates),
        "model": model_name,
        "model_params": _model_config(prototype),
        "train_window_days": train_window_days,
        "valid_days": valid_days,
        "n_splits": n_splits,
        "top_k": top_k,
    }
    payload = cache.payload("classification_cv", config)
    if payload is not None:
        print(f"  [CACHE HIT] {experiment} | {model_name} | window={train_window_days or 'expanding'} | folds={n_splits}")
        return payload
    print(f"  [FIT] {experiment} | {model_name} | window={train_window_days or 'expanding'} | folds={n_splits}")
    result = _classification_cv(
        frame, target, classes, candidates, prototype, train_window_days,
        valid_days=valid_days, n_splits=n_splits, top_k=top_k,
    )
    if result is None:
        return None
    payload = _compact(result)
    cache.set("classification_cv", config, payload)
    return payload


def _cached_regression(
    cache: ExperimentCache,
    frame: pd.DataFrame,
    candidates: list[str],
    prototype,
    model_name: str,
    train_window_days: int | None,
    threshold_col: str,
    experiment: str,
    valid_days: int = 180,
    n_splits: int = CONFIRM_SPLITS,
) -> dict | None:
    config = {
        "logic": CACHE_LOGIC_VERSION,
        "experiment": experiment,
        "target": "target_return_1d",
        "threshold_col": threshold_col,
        "candidate_hash": stable_hash(sorted(candidates)),
        "candidate_count": len(candidates),
        "model": model_name,
        "model_params": _model_config(prototype),
        "train_window_days": train_window_days,
        "valid_days": valid_days,
        "n_splits": n_splits,
    }
    payload = cache.payload("regression_cv", config)
    if payload is not None:
        print(f"  [CACHE HIT] {experiment} | {model_name} | window={train_window_days or 'expanding'} | folds={n_splits}")
        return payload
    print(f"  [FIT] {experiment} | {model_name} | window={train_window_days or 'expanding'} | folds={n_splits}")
    result = _regression_cv(
        frame, candidates, prototype, train_window_days, threshold_col,
        valid_days=valid_days, n_splits=n_splits,
    )
    if result is None:
        return None
    payload = _compact(result)
    cache.set("regression_cv", config, payload)
    return payload


def _save_rows(rows: list[dict], path) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(path, index=False)
    return df


def _top_values(df: pd.DataFrame, column: str, n: int) -> list:
    if df.empty:
        return []
    ordered = df.sort_values("robust_score", ascending=False)
    values = []
    for value in ordered[column].tolist():
        if value not in values:
            values.append(value)
        if len(values) >= n:
            break
    return values


def run_deep_research_cached() -> dict:
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
    scout_model = "lightgbm" if "lightgbm" in class_models else strong_models[0]

    cache = ExperimentCache({
        "logic": CACHE_LOGIC_VERSION,
        "dataset_sha256": file_sha256(dataset_path),
        "requirements_sha256": file_sha256(ROOT / "requirements.txt"),
    })
    selection_records = []
    pruning = {
        "strategy": "coarse_to_fine",
        "scout_splits": SCOUT_SPLITS,
        "confirm_splits": CONFIRM_SPLITS,
        "rules": [],
    }

    print("[DEEP 1/7] 학습 윈도우: 빠른 탐색 후 상위 후보만 재검증")
    window_rows = []
    for window in (365, 730, 1095, 1825, None):
        result = _cached_classification(
            cache, frame, "label_dynamic", np.array([-1, 0, 1]), all_features,
            class_models[scout_model], scout_model, window,
            "window_scout_three_class", n_splits=SCOUT_SPLITS,
        )
        if result is None:
            continue
        window_rows.append({
            "phase": "scout", "task": "three_class", "model": scout_model,
            "train_window_days": "expanding" if window is None else window,
            **result["summary"],
        })
        selection_records.append({
            "experiment": "window_scout", "feature_set": "all", "model": scout_model,
            "selected_features": result["selected_features"],
        })
        _save_rows(window_rows, OUTPUTS / "deep_window_sweep.csv")

    scout_window_df = pd.DataFrame(window_rows)
    top_window_values = _top_values(scout_window_df, "train_window_days", 2)
    if not top_window_values:
        top_window_values = [1095]
    pruning["rules"].append({"stage": "window", "scouted": 5, "promoted": top_window_values})

    for window_value in top_window_values:
        window = None if str(window_value) == "expanding" else int(window_value)
        for model_name in strong_models:
            result = _cached_classification(
                cache, frame, "label_dynamic", np.array([-1, 0, 1]), all_features,
                class_models[model_name], model_name, window,
                "window_confirm_three_class", n_splits=CONFIRM_SPLITS,
            )
            if result is None:
                continue
            window_rows.append({
                "phase": "confirm", "task": "three_class", "model": model_name,
                "train_window_days": "expanding" if window is None else window,
                **result["summary"],
            })
            selection_records.append({
                "experiment": "window_confirm", "feature_set": "all", "model": model_name,
                "selected_features": result["selected_features"],
            })
            _save_rows(window_rows, OUTPUTS / "deep_window_sweep.csv")

    window_df = pd.DataFrame(window_rows).sort_values(["phase", "robust_score"], ascending=[True, False]).reset_index(drop=True)
    window_df.to_csv(OUTPUTS / "deep_window_sweep.csv", index=False)
    confirmed_windows = window_df[window_df["phase"] == "confirm"]
    best_window_row = confirmed_windows.iloc[confirmed_windows["robust_score"].argmax()] if not confirmed_windows.empty else window_df.iloc[window_df["robust_score"].argmax()]
    best_window_value = best_window_row["train_window_days"]
    best_window = None if str(best_window_value) == "expanding" else int(best_window_value)
    best_window_model = str(best_window_row["model"])

    print("[DEEP 2/7] 라벨 창과 k: 15개 빠른 탐색 후 상위 3개만 재검증")
    label_rows = []
    for vol_window in (7, 14, 21, 30, 60):
        for k in (0.5, 0.7, 0.9):
            temp, label_col, _ = _make_dynamic_labels(frame, vol_window, k)
            result = _cached_classification(
                cache, temp, label_col, np.array([-1, 0, 1]), all_features,
                class_models[best_window_model], best_window_model, best_window,
                f"label_scout_v{vol_window}_k{k}", n_splits=SCOUT_SPLITS,
            )
            if result is None:
                continue
            shares = temp[label_col].value_counts(normalize=True).to_dict()
            label_rows.append({
                "phase": "scout", "vol_window": vol_window, "k": k,
                "model": best_window_model,
                "train_window_days": "expanding" if best_window is None else best_window,
                "down_share": float(shares.get(-1, 0.0)),
                "neutral_share": float(shares.get(0, 0.0)),
                "up_share": float(shares.get(1, 0.0)),
                **result["summary"],
            })
            selection_records.append({
                "experiment": "label_scout", "feature_set": "all", "model": best_window_model,
                "selected_features": result["selected_features"],
            })
            _save_rows(label_rows, OUTPUTS / "deep_label_window_k_sweep.csv")

    scout_label_df = pd.DataFrame(label_rows).sort_values("robust_score", ascending=False)
    top_label_pairs = []
    for _, row in scout_label_df.head(3).iterrows():
        top_label_pairs.append((int(row["vol_window"]), float(row["k"])))
    if not top_label_pairs:
        top_label_pairs = [(21, 0.7)]
    pruning["rules"].append({"stage": "label", "scouted": 15, "promoted": top_label_pairs})

    for vol_window, k in top_label_pairs:
        temp, label_col, _ = _make_dynamic_labels(frame, vol_window, k)
        result = _cached_classification(
            cache, temp, label_col, np.array([-1, 0, 1]), all_features,
            class_models[best_window_model], best_window_model, best_window,
            f"label_confirm_v{vol_window}_k{k}", n_splits=CONFIRM_SPLITS,
        )
        if result is None:
            continue
        shares = temp[label_col].value_counts(normalize=True).to_dict()
        label_rows.append({
            "phase": "confirm", "vol_window": vol_window, "k": k,
            "model": best_window_model,
            "train_window_days": "expanding" if best_window is None else best_window,
            "down_share": float(shares.get(-1, 0.0)),
            "neutral_share": float(shares.get(0, 0.0)),
            "up_share": float(shares.get(1, 0.0)),
            **result["summary"],
        })
        selection_records.append({
            "experiment": "label_confirm", "feature_set": "all", "model": best_window_model,
            "selected_features": result["selected_features"],
        })
        _save_rows(label_rows, OUTPUTS / "deep_label_window_k_sweep.csv")

    label_df = pd.DataFrame(label_rows).sort_values(["phase", "robust_score"], ascending=[True, False]).reset_index(drop=True)
    label_df.to_csv(OUTPUTS / "deep_label_window_k_sweep.csv", index=False)
    confirmed_labels = label_df[label_df["phase"] == "confirm"]
    best_label = confirmed_labels.iloc[confirmed_labels["robust_score"].argmax()] if not confirmed_labels.empty else label_df.iloc[label_df["robust_score"].argmax()]
    best_vol = int(best_label["vol_window"])
    best_k = float(best_label["k"])
    tuned, label_col, threshold_col = _make_dynamic_labels(frame, best_vol, best_k)
    tuned["research_move"] = (tuned["target_return_1d"].abs() > tuned[threshold_col]).astype(int)

    print("[DEEP 3/7] 피처군: 한 모델로 빠르게 가지치기 후 상위 3개만 다중 모델 재검증")
    ablation_rows = []
    model_names = [m for m in ("random_forest", "extra_trees", "lightgbm", "catboost") if m in class_models]
    for feature_name, features in groups.items():
        result = _cached_classification(
            cache, tuned, label_col, np.array([-1, 0, 1]), features,
            class_models[best_window_model], best_window_model, best_window,
            f"ablation_scout_{feature_name}", n_splits=SCOUT_SPLITS,
        )
        if result is None:
            continue
        ablation_rows.append({
            "phase": "scout", "feature_set": feature_name, "model": best_window_model,
            "train_window_days": "expanding" if best_window is None else best_window,
            "vol_window": best_vol, "k": best_k,
            **result["summary"],
        })
        selection_records.append({
            "experiment": "feature_scout", "feature_set": feature_name, "model": best_window_model,
            "selected_features": result["selected_features"],
        })
        _save_rows(ablation_rows, OUTPUTS / "deep_feature_model_ablation.csv")

    scout_ablation_df = pd.DataFrame(ablation_rows)
    top_feature_sets = _top_values(scout_ablation_df, "feature_set", 3)
    if not top_feature_sets:
        top_feature_sets = ["all"]
    pruning["rules"].append({"stage": "feature_set", "scouted": len(groups), "promoted": top_feature_sets})

    for feature_name in top_feature_sets:
        features = groups[feature_name]
        for model_name in model_names:
            result = _cached_classification(
                cache, tuned, label_col, np.array([-1, 0, 1]), features,
                class_models[model_name], model_name, best_window,
                f"ablation_confirm_{feature_name}", n_splits=CONFIRM_SPLITS,
            )
            if result is None:
                continue
            ablation_rows.append({
                "phase": "confirm", "feature_set": feature_name, "model": model_name,
                "train_window_days": "expanding" if best_window is None else best_window,
                "vol_window": best_vol, "k": best_k,
                **result["summary"],
            })
            selection_records.append({
                "experiment": "feature_confirm", "feature_set": feature_name, "model": model_name,
                "selected_features": result["selected_features"],
            })
            _save_rows(ablation_rows, OUTPUTS / "deep_feature_model_ablation.csv")

    ablation_df = pd.DataFrame(ablation_rows).sort_values(["phase", "robust_score"], ascending=[True, False]).reset_index(drop=True)
    ablation_df.to_csv(OUTPUTS / "deep_feature_model_ablation.csv", index=False)
    confirmed_ablation = ablation_df[ablation_df["phase"] == "confirm"]
    best_ablation = confirmed_ablation.iloc[confirmed_ablation["robust_score"].argmax()] if not confirmed_ablation.empty else ablation_df.iloc[ablation_df["robust_score"].argmax()]
    best_feature_set = str(best_ablation["feature_set"])
    best_model_name = str(best_ablation["model"])
    best_features = groups[best_feature_set]

    ranked_models = _top_values(confirmed_ablation if not confirmed_ablation.empty else ablation_df, "model", 3)
    if not ranked_models:
        ranked_models = [best_model_name]
    pruning["rules"].append({"stage": "model", "available": model_names, "promoted": ranked_models})

    print("[DEEP 4/7] 문제 정의 비교: 상위 모델만 3분류, 방향, 큰 움직임에 재사용")
    task_rows = []
    for task, target, classes in (
        ("three_class", label_col, np.array([-1, 0, 1])),
        ("direction_binary", "target_direction", np.array([0, 1])),
        ("move_binary", "research_move", np.array([0, 1])),
    ):
        for model_name in ranked_models:
            result = _cached_classification(
                cache, tuned, target, classes, best_features,
                class_models[model_name], model_name, best_window,
                f"task_{task}_{best_feature_set}", n_splits=CONFIRM_SPLITS,
            )
            if result is None:
                continue
            task_rows.append({"task": task, "model": model_name, **result["summary"]})
            selection_records.append({
                "experiment": f"task_{task}", "feature_set": best_feature_set, "model": model_name,
                "selected_features": result["selected_features"],
            })
            _save_rows(task_rows, OUTPUTS / "deep_task_formulation_comparison.csv")

    task_df = pd.DataFrame(task_rows).sort_values(["task", "robust_score"], ascending=[True, False]).reset_index(drop=True)
    task_df.to_csv(OUTPUTS / "deep_task_formulation_comparison.csv", index=False)

    print("[DEEP 5/7] 다음날 수익률 회귀 비교")
    regression_rows = []
    for model_name, prototype in _regression_models().items():
        result = _cached_regression(
            cache, tuned, best_features, prototype, model_name,
            best_window, threshold_col, f"regression_{best_feature_set}", n_splits=CONFIRM_SPLITS,
        )
        if result is None:
            continue
        regression_rows.append({"model": model_name, **result["summary"]})
        selection_records.append({
            "experiment": "regression", "feature_set": best_feature_set, "model": model_name,
            "selected_features": result["selected_features"],
        })
        _save_rows(regression_rows, OUTPUTS / "deep_regression_comparison.csv")

    regression_df = pd.DataFrame(regression_rows).sort_values(
        ["fold_pearson_mean", "fold_derived_macro_f1_mean"], ascending=[False, False]
    ).reset_index(drop=True)
    regression_df.to_csv(OUTPUTS / "deep_regression_comparison.csv", index=False)

    print("[DEEP 6/7] 베이스라인과 피처 선택 안정성")
    baselines = _baseline_metrics(tuned, label_col, threshold_col)
    baselines.to_csv(OUTPUTS / "deep_baselines.csv", index=False)
    frequency = _selection_frequency(selection_records)
    frequency.to_csv(OUTPUTS / "deep_feature_selection_frequency.csv", index=False)
    write_json(OUTPUTS / "deep_search_pruning_summary.json", pruning)

    print("[DEEP 7/7] 추천 설정 잠금")
    best_three = task_df[task_df["task"] == "three_class"].iloc[0].to_dict() if not task_df[task_df["task"] == "three_class"].empty else {}
    best_direction = task_df[task_df["task"] == "direction_binary"].iloc[0].to_dict() if not task_df[task_df["task"] == "direction_binary"].empty else {}
    best_move = task_df[task_df["task"] == "move_binary"].iloc[0].to_dict() if not task_df[task_df["task"] == "move_binary"].empty else {}
    best_regression = regression_df.iloc[0].to_dict() if not regression_df.empty else {}

    summary = {
        "research_status": "development_cv_only",
        "execution_mode": "persistent_cache_plus_coarse_to_fine_pruning",
        "primary_task": "three_class_next_day_return",
        "primary_definition": "t일 종료 시점까지 관측 가능한 정보로 t+1일 BTC 수익률을 DOWN, NEUTRAL, UP으로 분류",
        "secondary_tasks": ["direction_binary", "move_binary", "return_regression"],
        "selected_training_window_days": "expanding" if best_window is None else best_window,
        "selected_volatility_window_days": best_vol,
        "selected_dynamic_k": best_k,
        "selected_feature_set": best_feature_set,
        "selected_classifier": best_model_name,
        "promoted_models": ranked_models,
        "best_three_class": best_three,
        "best_direction_binary": best_direction,
        "best_move_binary": best_move,
        "best_regression": best_regression,
        "cache_context": cache.context_hash,
        "important_rule": "빠른 탐색으로 후보를 줄인 뒤 상위 후보만 더 엄격한 CV로 재검증하며 동일 조합은 persistent cache에서 재사용한다.",
    }
    write_json(OUTPUTS / "deep_research_summary.json", summary)
    write_json(OUTPUTS / "deep_research_config_lock.json", summary)
    return summary

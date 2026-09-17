from __future__ import annotations

import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .common import PROCESSED, OUTPUTS, MODELS, ensure_dirs, write_json

warnings.filterwarnings("ignore")
RANDOM_STATE = 42


def _optional_models() -> list[tuple[str, object]]:
    models: list[tuple[str, object]] = [
        ("logistic", Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.3, class_weight="balanced", max_iter=4000, random_state=RANDOM_STATE)),
        ])),
        ("random_forest", Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("model", RandomForestClassifier(
                n_estimators=450, max_depth=9, min_samples_leaf=6,
                class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1,
            )),
        ])),
        ("extra_trees", Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("model", ExtraTreesClassifier(
                n_estimators=550, max_depth=10, min_samples_leaf=5,
                class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
            )),
        ])),
        ("hist_gradient_boosting", Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=False)),
            ("model", HistGradientBoostingClassifier(
                learning_rate=0.04, max_iter=300, max_leaf_nodes=15,
                l2_regularization=3.0, random_state=RANDOM_STATE,
            )),
        ])),
    ]
    try:
        from lightgbm import LGBMClassifier
        models.append(("lightgbm", Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("model", LGBMClassifier(
                n_estimators=500, learning_rate=0.025, num_leaves=15, max_depth=-1,
                min_child_samples=25, subsample=0.85, colsample_bytree=0.8,
                reg_alpha=0.5, reg_lambda=2.0, class_weight="balanced",
                random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1,
            )),
        ])))
    except Exception:
        pass
    try:
        from catboost import CatBoostClassifier
        models.append(("catboost", Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("model", CatBoostClassifier(
                iterations=450, depth=6, learning_rate=0.035,
                auto_class_weights="Balanced", random_seed=RANDOM_STATE,
                verbose=False, allow_writing_files=False,
            )),
        ])))
    except Exception:
        pass
    return models


def _target_classes(target: str) -> np.ndarray:
    if target in {"label_dynamic", "label_fixed"}:
        return np.array([-1, 0, 1])
    return np.array([0, 1])


def _feature_sets(columns: list[str]) -> dict[str, list[str]]:
    excluded = {
        "date", "target_end_date", "btc_close", "target_return_1d",
        "threshold_fixed", "threshold_dynamic", "label_fixed", "label_dynamic",
        "target_move_dynamic", "target_direction",
    }
    usable = [c for c in columns if c not in excluded]
    price = [c for c in usable if c.startswith("btc_") or c.startswith("eth_")]
    macro = [
        c for c in usable
        if any(c.startswith(p) for p in (
            "spy_", "qqq_", "gold_", "dxy_", "vix_", "tlt_",
            "btc_spy_", "btc_qqq_", "btc_gold_", "btc_dxy_", "btc_vix_", "btc_tlt_",
        ))
    ]
    sentiment = [
        c for c in usable
        if "fear_greed" in c or "x_sentiment" in c or "x_post" in c
        or "x_positive" in c or "x_negative" in c or "x_engagement" in c
    ]
    external = [c for c in usable if c.startswith("ext_")]
    onchain = [c for c in external if c not in sentiment]
    result = {
        "price": sorted(set(price)),
        "price_macro": sorted(set(price + macro)),
        "price_onchain": sorted(set(price + onchain)),
        "price_sentiment": sorted(set(price + sentiment)),
        "all": sorted(set(usable)),
    }
    return {k: v for k, v in result.items() if len(v) >= 5}


def _select_by_training_coverage(X_train: pd.DataFrame, candidates: list[str], min_coverage: float = 0.60) -> list[str]:
    selected = [
        c for c in candidates
        if X_train[c].notna().mean() >= min_coverage and X_train[c].nunique(dropna=True) > 1
    ]
    if len(selected) < 5:
        selected = [
            c for c in candidates
            if X_train[c].notna().mean() >= 0.30 and X_train[c].nunique(dropna=True) > 1
        ]
    return selected


def _metrics(y_true, pred, prob=None, classes=None) -> dict:
    out = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
    }
    unique = np.unique(y_true)
    if prob is not None:
        try:
            out["log_loss"] = float(log_loss(y_true, prob, labels=classes))
        except Exception:
            pass
        if len(unique) == 2 and prob.ndim == 2 and prob.shape[1] == 2:
            pos = prob[:, 1]
            try:
                out["roc_auc"] = float(roc_auc_score(y_true, pos))
                out["brier"] = float(brier_score_loss(y_true, pos))
            except Exception:
                pass
    return out


def _aligned_proba(model, X: pd.DataFrame, target_classes: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        raw = model.predict_proba(X)
        classes = np.asarray(model.classes_)
        aligned = np.zeros((len(X), len(target_classes)), dtype=float)
        for i, cls in enumerate(classes):
            matches = np.where(target_classes == cls)[0]
            if len(matches):
                aligned[:, matches[0]] = raw[:, i]
        row_sum = aligned.sum(axis=1, keepdims=True)
        return aligned / np.where(row_sum == 0, 1, row_sum)
    pred = model.predict(X)
    aligned = np.zeros((len(X), len(target_classes)), dtype=float)
    for i, cls in enumerate(target_classes):
        aligned[:, i] = pred == cls
    return aligned


def _validate_temporal_dataset(df: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "target_end_date", "target_return_1d"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"시간 누수 검증에 필요한 열이 없습니다: {sorted(missing)}")

    frame = df.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["target_end_date"] = pd.to_datetime(frame["target_end_date"], errors="coerce")
    frame = frame.dropna(subset=["date", "target_end_date", "target_return_1d"]).sort_values("date").reset_index(drop=True)

    if frame["date"].duplicated().any():
        raise RuntimeError("date가 중복되어 있습니다. 시계열 순서를 확정할 수 없습니다.")
    if not frame["date"].is_monotonic_increasing:
        raise RuntimeError("date가 시간 오름차순이 아닙니다.")
    if (frame["target_end_date"] <= frame["date"]).any():
        raise RuntimeError("target_end_date가 입력 기준일보다 미래가 아닌 행이 있습니다.")
    return frame


def _assert_boundary(frame: pd.DataFrame, train_idx: np.ndarray, valid_idx: np.ndarray, context: str) -> dict:
    if len(train_idx) == 0 or len(valid_idx) == 0:
        raise RuntimeError(f"{context}: 학습 또는 검증 데이터가 비어 있습니다.")

    train_last_feature = pd.Timestamp(frame.iloc[train_idx]["date"].max())
    train_last_target = pd.Timestamp(frame.iloc[train_idx]["target_end_date"].max())
    valid_first_feature = pd.Timestamp(frame.iloc[valid_idx]["date"].min())

    if not train_last_feature < valid_first_feature:
        raise RuntimeError(
            f"{context}: 학습 피처 날짜가 검증 시작일과 겹칩니다. "
            f"train_last_feature={train_last_feature}, valid_first={valid_first_feature}"
        )
    if not train_last_target < valid_first_feature:
        raise RuntimeError(
            f"{context}: 학습 정답을 계산한 미래 날짜가 검증 시작일과 겹칩니다. "
            f"train_last_target={train_last_target}, valid_first={valid_first_feature}"
        )

    return {
        "train_first_feature_date": str(pd.Timestamp(frame.iloc[train_idx]["date"].min()).date()),
        "train_last_feature_date": str(train_last_feature.date()),
        "train_last_target_end_date": str(train_last_target.date()),
        "valid_first_feature_date": str(valid_first_feature.date()),
        "valid_last_feature_date": str(pd.Timestamp(frame.iloc[valid_idx]["date"].max()).date()),
    }


def _strict_time_splits(frame: pd.DataFrame, n_splits: int = 5):
    splitter = TimeSeriesSplit(n_splits=n_splits)
    target_end = frame["target_end_date"].to_numpy(dtype="datetime64[ns]")

    for fold, (raw_train, valid_idx) in enumerate(splitter.split(frame), 1):
        valid_start = np.datetime64(pd.Timestamp(frame.iloc[valid_idx]["date"].min()))
        keep = target_end[raw_train] < valid_start
        train_idx = raw_train[keep]
        purged_rows = int(len(raw_train) - len(train_idx))
        if len(train_idx) < 50:
            continue
        audit = _assert_boundary(frame, train_idx, valid_idx, f"fold {fold}")
        audit.update({
            "fold": fold,
            "raw_train_rows": int(len(raw_train)),
            "train_rows_after_purge": int(len(train_idx)),
            "purged_rows": purged_rows,
            "valid_rows": int(len(valid_idx)),
        })
        yield fold, train_idx, valid_idx, audit


def _strict_holdout_split(frame: pd.DataFrame, train_ratio: float = 0.80):
    split = int(len(frame) * train_ratio)
    if split <= 0 or split >= len(frame):
        raise RuntimeError("holdout 분할을 만들 수 없습니다.")

    test_idx = np.arange(split, len(frame))
    holdout_start = pd.Timestamp(frame.iloc[test_idx]["date"].min())
    raw_train = np.arange(0, split)
    target_end = frame["target_end_date"].to_numpy(dtype="datetime64[ns]")
    train_idx = raw_train[target_end[raw_train] < np.datetime64(holdout_start)]
    audit = _assert_boundary(frame, train_idx, test_idx, "final holdout")
    audit.update({
        "raw_train_rows": int(len(raw_train)),
        "train_rows_after_purge": int(len(train_idx)),
        "purged_rows": int(len(raw_train) - len(train_idx)),
        "holdout_rows": int(len(test_idx)),
    })
    return train_idx, test_idx, audit


def _oof_for_model(frame: pd.DataFrame, target: str, candidate_features: list[str], prototype, classes: np.ndarray, n_splits: int = 5):
    y = frame[target].astype(int)
    proba = np.full((len(frame), len(classes)), np.nan)
    fold_rows = []
    selected_by_fold = []

    for fold, tr, va, audit in _strict_time_splits(frame, n_splits=n_splits):
        features = _select_by_training_coverage(frame.iloc[tr], candidate_features)
        if len(features) < 2 or y.iloc[tr].nunique() < 2:
            continue
        model = clone(prototype)
        model.fit(frame.iloc[tr][features], y.iloc[tr])
        p = _aligned_proba(model, frame.iloc[va][features], classes)
        pred = classes[np.argmax(p, axis=1)]
        proba[va] = p
        row = {
            **audit,
            "train_size": len(tr),
            "valid_size": len(va),
            "feature_count": len(features),
            **_metrics(y.iloc[va], pred, p, classes),
        }
        fold_rows.append(row)
        selected_by_fold.append(features)

    valid_mask = ~np.isnan(proba).any(axis=1)
    if valid_mask.sum() == 0:
        return None
    pred = classes[np.argmax(proba[valid_mask], axis=1)]
    summary = _metrics(y.iloc[np.flatnonzero(valid_mask)], pred, proba[valid_mask], classes)
    summary["oof_rows"] = int(valid_mask.sum())
    summary["fold_macro_f1_mean"] = float(np.mean([r["macro_f1"] for r in fold_rows]))
    summary["fold_macro_f1_std"] = float(np.std([r["macro_f1"] for r in fold_rows]))
    return {
        "proba": proba,
        "mask": valid_mask,
        "summary": summary,
        "fold_rows": fold_rows,
        "selected_by_fold": selected_by_fold,
    }


def _fit_final(prototype, X_train, y_train, X_test, candidates, classes):
    features = _select_by_training_coverage(X_train, candidates)
    if len(features) < 2:
        raise RuntimeError("최종 학습에 사용할 피처가 부족합니다.")
    model = clone(prototype)
    model.fit(X_train[features], y_train)
    p = _aligned_proba(model, X_test[features], classes)
    return model, features, p


def _evaluate_task(df: pd.DataFrame, target: str, task_name: str, output_prefix: str) -> dict:
    clean = df.dropna(subset=[target]).reset_index(drop=True)
    classes = _target_classes(target)
    feature_sets = _feature_sets(list(clean.columns))
    train_idx, test_idx, holdout_audit = _strict_holdout_split(clean)
    train = clean.iloc[train_idx].reset_index(drop=True)
    test = clean.iloc[test_idx].reset_index(drop=True)
    y_train = train[target].astype(int)
    y_test = test[target].astype(int)

    comparison_rows, fold_rows, oof_store = [], [], {}
    model_lookup = dict(_optional_models())

    for feature_name, features in feature_sets.items():
        for model_name, prototype in model_lookup.items():
            try:
                result = _oof_for_model(train, target, features, prototype, classes)
                if result is None:
                    continue
                row = {
                    "task": task_name,
                    "feature_set": feature_name,
                    "model": model_name,
                    "candidate_feature_count": len(features),
                    **result["summary"],
                }
                comparison_rows.append(row)
                for r in result["fold_rows"]:
                    fold_rows.append({"task": task_name, "feature_set": feature_name, "model": model_name, **r})
                oof_store[(feature_name, model_name)] = result
            except Exception as exc:
                print(f"[WARN] OOF failed {task_name}/{feature_name}/{model_name}: {exc}")

    comparison = pd.DataFrame(comparison_rows)
    if comparison.empty:
        raise RuntimeError(f"No successful models for {task_name}")
    comparison = comparison.sort_values(["macro_f1", "fold_macro_f1_std"], ascending=[False, True]).reset_index(drop=True)
    best = comparison.iloc[0]
    best_proto = model_lookup[best["model"]]
    best_features = feature_sets[best["feature_set"]]
    final_model, selected_features, test_prob = _fit_final(best_proto, train, y_train, test, best_features, classes)
    test_pred = classes[np.argmax(test_prob, axis=1)]
    holdout = _metrics(y_test, test_pred, test_prob, classes)

    top = comparison.head(min(3, len(comparison))).copy()
    raw_weights = np.maximum(top["macro_f1"].to_numpy(), 1e-6)
    weights = raw_weights / raw_weights.sum()
    common_mask = np.ones(len(train), dtype=bool)
    for _, row in top.iterrows():
        common_mask &= oof_store[(row["feature_set"], row["model"])]["mask"]

    ensemble_oof = np.zeros((common_mask.sum(), len(classes)))
    final_test_prob = np.zeros((len(test), len(classes)))
    ensemble_members = []
    for weight, (_, row) in zip(weights, top.iterrows()):
        key = (row["feature_set"], row["model"])
        ensemble_oof += weight * oof_store[key]["proba"][common_mask]
        proto = model_lookup[row["model"]]
        _, feats, p = _fit_final(proto, train, y_train, test, feature_sets[row["feature_set"]], classes)
        final_test_prob += weight * p
        ensemble_members.append({
            "model": row["model"],
            "feature_set": row["feature_set"],
            "weight": float(weight),
            "feature_count": len(feats),
        })

    ensemble_oof_pred = classes[np.argmax(ensemble_oof, axis=1)]
    ensemble_oof_metrics = _metrics(
        y_train.iloc[np.flatnonzero(common_mask)], ensemble_oof_pred, ensemble_oof, classes
    )
    ensemble_test_pred = classes[np.argmax(final_test_prob, axis=1)]
    ensemble_holdout_metrics = _metrics(y_test, ensemble_test_pred, final_test_prob, classes)

    predictions = test[["date", "target_end_date", "target_return_1d", target]].copy()
    predictions["best_model_pred"] = test_pred
    predictions["ensemble_pred"] = ensemble_test_pred
    for i, cls in enumerate(classes):
        predictions[f"ensemble_prob_{cls}"] = final_test_prob[:, i]

    comparison.to_csv(OUTPUTS / f"{output_prefix}_model_comparison.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(OUTPUTS / f"{output_prefix}_fold_metrics.csv", index=False)
    predictions.to_csv(OUTPUTS / f"{output_prefix}_holdout_predictions.csv", index=False)
    joblib.dump(
        {"model": final_model, "features": selected_features, "classes": classes},
        MODELS / f"{output_prefix}_best_model.joblib",
    )

    return {
        "task": task_name,
        "rows": len(clean),
        "train_rows": len(train),
        "holdout_rows": len(test),
        "holdout_start": str(pd.Timestamp(test["date"].iloc[0]).date()),
        "holdout_boundary": holdout_audit,
        "best_model": str(best["model"]),
        "best_feature_set": str(best["feature_set"]),
        "best_oof": {
            k: float(best[k])
            for k in ("accuracy", "balanced_accuracy", "macro_f1", "fold_macro_f1_mean", "fold_macro_f1_std")
        },
        "best_holdout": holdout,
        "ensemble_members": ensemble_members,
        "ensemble_oof": ensemble_oof_metrics,
        "ensemble_holdout": ensemble_holdout_metrics,
    }


def _two_stage(df: pd.DataFrame) -> dict:
    clean = df.dropna(subset=["target_move_dynamic", "label_dynamic"]).reset_index(drop=True)
    feature_sets = _feature_sets(list(clean.columns))
    features = feature_sets["all"]
    train_idx, test_idx, holdout_audit = _strict_holdout_split(clean)
    train = clean.iloc[train_idx].reset_index(drop=True)
    test = clean.iloc[test_idx].reset_index(drop=True)
    model_lookup = dict(_optional_models())

    candidate_names = [
        n for n in ("lightgbm", "catboost", "extra_trees", "random_forest")
        if n in model_lookup
    ]
    move_scores, direction_scores = [], []
    move_oof_by_model, dir_oof_by_model = {}, {}
    audit_rows = []

    for name in candidate_names:
        proto = model_lookup[name]
        move_prob = np.full(len(train), np.nan)
        dir_prob = np.full(len(train), np.nan)

        for fold, tr, va, audit in _strict_time_splits(train, n_splits=5):
            feats = _select_by_training_coverage(train.iloc[tr], features)
            if len(feats) < 2 or train.iloc[tr]["target_move_dynamic"].nunique() < 2:
                continue

            move_model = clone(proto).fit(
                train.iloc[tr][feats], train.iloc[tr]["target_move_dynamic"].astype(int)
            )
            move_prob[va] = _aligned_proba(
                move_model, train.iloc[va][feats], np.array([0, 1])
            )[:, 1]

            actionable_tr = tr[train.iloc[tr]["label_dynamic"].to_numpy() != 0]
            if len(actionable_tr) >= 50 and train.iloc[actionable_tr]["target_direction"].nunique() >= 2:
                dir_model = clone(proto).fit(
                    train.iloc[actionable_tr][feats],
                    train.iloc[actionable_tr]["target_direction"].astype(int),
                )
                dir_prob[va] = _aligned_proba(
                    dir_model, train.iloc[va][feats], np.array([0, 1])
                )[:, 1]

            audit_rows.append({"model": name, **audit})

        move_mask = ~np.isnan(move_prob)
        dir_mask = ~np.isnan(dir_prob) & (train["label_dynamic"].to_numpy() != 0)
        if move_mask.sum() > 0:
            move_scores.append((
                f1_score(
                    train.loc[move_mask, "target_move_dynamic"],
                    move_prob[move_mask] >= 0.5,
                    average="macro",
                ),
                name,
            ))
            move_oof_by_model[name] = move_prob
        if dir_mask.sum() > 0:
            direction_scores.append((
                f1_score(
                    train.loc[dir_mask, "target_direction"],
                    dir_prob[dir_mask] >= 0.5,
                    average="macro",
                ),
                name,
            ))
            dir_oof_by_model[name] = dir_prob

    pd.DataFrame(audit_rows).to_csv(OUTPUTS / "two_stage_temporal_audit.csv", index=False)

    if not move_scores or not direction_scores:
        return {
            "status": "skipped",
            "reason": "insufficient successful two-stage models",
            "holdout_boundary": holdout_audit,
        }

    move_name = max(move_scores)[1]
    dir_name = max(direction_scores)[1]
    move_oof = move_oof_by_model[move_name]
    dir_oof = dir_oof_by_model[dir_name]
    mask = ~np.isnan(move_oof) & ~np.isnan(dir_oof)
    y3 = train.loc[mask, "label_dynamic"].astype(int).to_numpy()

    threshold_rows = []
    for threshold in np.arange(0.40, 0.76, 0.025):
        pred = np.zeros(mask.sum(), dtype=int)
        active = move_oof[mask] >= threshold
        pred[active & (dir_oof[mask] >= 0.5)] = 1
        pred[active & (dir_oof[mask] < 0.5)] = -1
        threshold_rows.append({"move_threshold": round(float(threshold), 3), **_metrics(y3, pred)})
    threshold_df = pd.DataFrame(threshold_rows).sort_values(
        ["macro_f1", "balanced_accuracy"], ascending=False
    )
    locked_threshold = float(threshold_df.iloc[0]["move_threshold"])
    threshold_df.to_csv(OUTPUTS / "two_stage_threshold_validation.csv", index=False)

    feats = _select_by_training_coverage(train, features)
    move_model = clone(model_lookup[move_name]).fit(
        train[feats], train["target_move_dynamic"].astype(int)
    )
    actionable = train["label_dynamic"] != 0
    dir_model = clone(model_lookup[dir_name]).fit(
        train.loc[actionable, feats], train.loc[actionable, "target_direction"].astype(int)
    )
    move_test = _aligned_proba(move_model, test[feats], np.array([0, 1]))[:, 1]
    dir_test = _aligned_proba(dir_model, test[feats], np.array([0, 1]))[:, 1]

    pred = np.zeros(len(test), dtype=int)
    active = move_test >= locked_threshold
    pred[active & (dir_test >= 0.5)] = 1
    pred[active & (dir_test < 0.5)] = -1
    metrics = _metrics(test["label_dynamic"].astype(int), pred)

    result = test[["date", "target_end_date", "target_return_1d", "label_dynamic"]].copy()
    result["move_probability"] = move_test
    result["direction_up_probability"] = dir_test
    result["prediction"] = pred
    result.to_csv(OUTPUTS / "two_stage_holdout_predictions.csv", index=False)
    joblib.dump(
        {
            "move_model": move_model,
            "direction_model": dir_model,
            "features": feats,
            "move_threshold": locked_threshold,
        },
        MODELS / "two_stage_models.joblib",
    )

    return {
        "status": "ok",
        "move_model": move_name,
        "direction_model": dir_name,
        "locked_move_threshold": locked_threshold,
        "holdout": metrics,
        "holdout_active_share": float(active.mean()),
        "holdout_boundary": holdout_audit,
    }


def run_research() -> dict:
    ensure_dirs()
    path = PROCESSED / "model_dataset.csv"
    if not path.exists():
        raise RuntimeError("model_dataset.csv가 없습니다. feature 생성부터 실행하세요.")

    df = pd.read_csv(path, parse_dates=["date", "target_end_date"])
    df = _validate_temporal_dataset(df)

    results = {}
    results["three_class_dynamic"] = _evaluate_task(
        df, "label_dynamic", "동적 기준 3분류", "three_class_dynamic"
    )
    results["move_dynamic"] = _evaluate_task(
        df, "target_move_dynamic", "큰 움직임 발생 여부", "move_dynamic"
    )
    results["two_stage"] = _two_stage(df)
    write_json(OUTPUTS / "research_summary.json", results)

    temporal_summary = {
        "rule": "모든 학습 정답 종료일은 해당 검증 또는 holdout 입력 시작일보다 엄격히 이전이어야 함",
        "three_class_dynamic": results["three_class_dynamic"]["holdout_boundary"],
        "move_dynamic": results["move_dynamic"]["holdout_boundary"],
        "two_stage": results["two_stage"].get("holdout_boundary"),
    }
    write_json(OUTPUTS / "temporal_validation_summary.json", temporal_summary)

    rows = []
    for key in ("three_class_dynamic", "move_dynamic"):
        item = results[key]
        rows.append({
            "task": key,
            "best_model": item["best_model"],
            "best_feature_set": item["best_feature_set"],
            "best_oof_macro_f1": item["best_oof"]["macro_f1"],
            "best_holdout_macro_f1": item["best_holdout"]["macro_f1"],
            "ensemble_holdout_macro_f1": item["ensemble_holdout"]["macro_f1"],
            "holdout_balanced_accuracy": item["ensemble_holdout"]["balanced_accuracy"],
        })
    if results["two_stage"].get("status") == "ok":
        rows.append({
            "task": "two_stage",
            "best_model": f"{results['two_stage']['move_model']} -> {results['two_stage']['direction_model']}",
            "best_feature_set": "all",
            "best_oof_macro_f1": np.nan,
            "best_holdout_macro_f1": results["two_stage"]["holdout"]["macro_f1"],
            "ensemble_holdout_macro_f1": np.nan,
            "holdout_balanced_accuracy": results["two_stage"]["holdout"]["balanced_accuracy"],
        })
    pd.DataFrame(rows).to_csv(OUTPUTS / "research_scorecard.csv", index=False)
    return results

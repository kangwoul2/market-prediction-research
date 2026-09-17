from __future__ import annotations

from datetime import date, datetime
import os
from pathlib import Path
import traceback

from src.common import ROOT, RAW, INTERIM, PROCESSED, OUTPUTS, MODELS, ensure_dirs, write_json
from src.collectors import collect_all
from src.sentiment import run_sentiment
from src.features import build_features
from src.modeling import run_research
from src.deep_research_cached import run_deep_research_cached
from src.deep_post_analysis import run_deep_post_analysis
from src.train_locked_model import train_locked_model
from src.move_research_cached import run_move_research_cached
from src.reproducibility import write_reproducibility_manifest
from src.research_cache import PipelineCheckpoint, stage_signature


def _check_project_rules() -> None:
    readme = Path("README.md")
    if readme.exists():
        text = readme.read_text(encoding="utf-8")
        if "·" in text:
            raise RuntimeError("README.md에 사용 금지 문자인 가운데점이 있습니다. 쉼표나 슬래시로 바꿔주세요.")


def _run_stage(checkpoint, status, name, title, signature, outputs, fn):
    print(f"\n{title}")
    if checkpoint.is_fresh(name, signature, outputs):
        print(f"[SKIP] {name}: 입력, 코드, 설정이 이전 완료 실행과 같습니다.")
        status["steps"][name] = {"status": "cached", "signature": signature}
        return None
    try:
        result = fn()
        meta = result if isinstance(result, dict) else {}
        if not meta and hasattr(result, "__len__"):
            try:
                meta = {"rows": int(len(result))}
            except Exception:
                meta = {}
        checkpoint.mark_ok(name, signature, meta)
        status["steps"][name] = {"status": "ok", "signature": signature, **meta}
        return result
    except Exception as exc:
        checkpoint.mark_failed(name, signature, str(exc))
        status["steps"][name] = {
            "status": "failed",
            "signature": signature,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        _finish(status)
        raise


def _compact_move_result() -> dict:
    result = run_move_research_cached()
    keys = [
        "task", "selected_training_window_days", "selected_volatility_window_days",
        "selected_dynamic_k", "selected_feature_set", "selected_classifier",
        "selected_top_k", "decision_threshold", "prospective_start_date",
    ]
    return {k: result.get(k) for k in keys}


def _compact_repro_result() -> dict:
    result = write_reproducibility_manifest()
    return {
        "reproducibility_id": result.get("reproducibility_id"),
        "python_version": result.get("python_version"),
        "git_head": result.get("git", {}).get("head"),
    }


def main() -> None:
    ensure_dirs()
    checkpoint = PipelineCheckpoint()
    status = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "execution_mode": "checkpoint_and_experiment_cache",
        "steps": {},
        "success": False,
    }

    print("\n[0/9] 프로젝트 문서 규칙 검사")
    _check_project_rules()
    status["steps"]["project_rules"] = {"status": "ok"}

    collect_sig = stage_signature(
        sources=[ROOT / "src" / "collectors.py", ROOT / "src" / "common.py"],
        params={
            "date": str(date.today()),
            "start": "2015-01-01",
            "has_x_token": bool(os.getenv("X_BEARER_TOKEN", "").strip()),
            "has_glassnode": bool(os.getenv("GLASSNODE_API_KEY", "").strip()),
            "has_coinmetrics": bool(os.getenv("COINMETRICS_API_KEY", "").strip()),
        },
    )
    _run_stage(
        checkpoint, status, "collect", "[1/9] 시장, 암호시장, 온체인, 심리 원천 데이터 수집",
        collect_sig,
        [RAW / "market_daily.csv", RAW / "collection_inventory.csv"],
        lambda: {"inventory": collect_all(start="2015-01-01")},
    )

    sentiment_sig = stage_signature(
        sources=[ROOT / "src" / "sentiment.py", ROOT / "src" / "common.py"],
        inputs=[RAW / "x_posts.csv"],
        params={"date": str(date.today()), "has_x_token": bool(os.getenv("X_BEARER_TOKEN", "").strip())},
    )
    _run_stage(
        checkpoint, status, "sentiment", "[2/9] X 게시글 VADER 감성 피처 생성",
        sentiment_sig, [],
        lambda: {"rows": int(len(run_sentiment()))},
    )

    feature_inputs = [
        RAW / "market_daily.csv",
        RAW / "onchain_blockchain_com.csv",
        RAW / "onchain_coinmetrics.csv",
        RAW / "onchain_glassnode.csv",
        RAW / "fear_greed.csv",
        INTERIM / "x_sentiment_daily.csv",
    ]
    feature_sig = stage_signature(
        sources=[ROOT / "src" / "features.py", ROOT / "src" / "common.py"],
        inputs=feature_inputs,
        params={"dynamic_k": 0.7},
    )
    _run_stage(
        checkpoint, status, "features", "[3/9] 멀티 스케일 피처, 상대시장 피처, 동적 라벨 생성",
        feature_sig,
        [PROCESSED / "model_dataset.csv", OUTPUTS / "dataset_metadata.json", OUTPUTS / "feature_coverage.csv"],
        lambda: {
            "rows": int(len(dataset := build_features(dynamic_k=0.7))),
            "columns": int(len(dataset.columns)),
        },
    )

    if os.getenv("RUN_LEGACY_BASELINE", "0").strip() == "1":
        baseline_sig = stage_signature(
            sources=[ROOT / "src" / "modeling.py", ROOT / "src" / "common.py"],
            inputs=[PROCESSED / "model_dataset.csv"],
            params={"research": "reference_baseline"},
        )
        _run_stage(
            checkpoint, status, "reference_baseline", "[4/9] 선택 실행: 기준 모델 전체 비교",
            baseline_sig,
            [OUTPUTS / "research_summary.json", OUTPUTS / "research_scorecard.csv", OUTPUTS / "temporal_validation_summary.json"],
            lambda: {"tasks": list(run_research().keys())},
        )
    else:
        print("\n[4/9] [SKIP] 기준 모델 전체 비교는 심화 연구와 중복되어 기본 실행에서 제외합니다.")
        status["steps"]["reference_baseline"] = {"status": "disabled_by_default"}

    deep_sig = stage_signature(
        sources=[
            ROOT / "src" / "deep_research.py",
            ROOT / "src" / "deep_research_cached.py",
            ROOT / "src" / "modeling.py",
            ROOT / "src" / "research_cache.py",
        ],
        inputs=[PROCESSED / "model_dataset.csv", ROOT / "requirements.txt"],
        params={"research": "task_and_feature_search"},
    )
    _run_stage(
        checkpoint, status, "deep_research", "[5/9] 심화 연구: 학습창, 라벨창, 피처군, 문제정의, 회귀",
        deep_sig,
        [
            OUTPUTS / "deep_research_summary.json",
            OUTPUTS / "deep_research_config_lock.json",
            OUTPUTS / "deep_window_sweep.csv",
            OUTPUTS / "deep_label_window_k_sweep.csv",
            OUTPUTS / "deep_feature_model_ablation.csv",
            OUTPUTS / "deep_task_formulation_comparison.csv",
            OUTPUTS / "deep_regression_comparison.csv",
        ],
        run_deep_research_cached,
    )

    post_sig = stage_signature(
        sources=[ROOT / "src" / "deep_post_analysis.py", ROOT / "src" / "deep_research.py", ROOT / "src" / "modeling.py"],
        inputs=[PROCESSED / "model_dataset.csv", OUTPUTS / "deep_research_config_lock.json"],
        params={"analysis": "ensemble_regime_seed"},
    )
    _run_stage(
        checkpoint, status, "deep_post", "[6/9] 선택 설정 OOF 앙상블, confidence, 국면, seed 안정성",
        post_sig,
        [OUTPUTS / "deep_post_summary.json", OUTPUTS / "deep_selected_config_models.csv", OUTPUTS / "deep_seed_stability.csv"],
        run_deep_post_analysis,
    )

    lock_sig = stage_signature(
        sources=[ROOT / "src" / "train_locked_model.py", ROOT / "src" / "deep_research.py"],
        inputs=[PROCESSED / "model_dataset.csv", OUTPUTS / "deep_research_config_lock.json"],
        params={"artifact": "three_class_reference"},
    )
    _run_stage(
        checkpoint, status, "three_class_model", "[7/9] 선택 3-class 설정으로 개발 모델 재학습",
        lock_sig,
        [OUTPUTS / "locked_model_training_summary.json", MODELS / "deep_primary_three_class_model.joblib"],
        train_locked_model,
    )

    move_sig = stage_signature(
        sources=[
            ROOT / "src" / "move_research_cached.py",
            ROOT / "src" / "deep_research.py",
            ROOT / "src" / "modeling.py",
            ROOT / "src" / "research_cache.py",
        ],
        inputs=[PROCESSED / "model_dataset.csv", ROOT / "requirements.txt"],
        params={"research": "meaningful_move_detection"},
    )
    _run_stage(
        checkpoint, status, "move_research", "[8/9] 큰 움직임 binary 전용 탐색과 threshold 시간순 검증",
        move_sig,
        [
            OUTPUTS / "move_research_summary.json",
            OUTPUTS / "move_research_config_lock.json",
            OUTPUTS / "move_window_sweep.csv",
            OUTPUTS / "move_label_sweep.csv",
            OUTPUTS / "move_feature_model_ablation.csv",
            OUTPUTS / "move_feature_topk_sweep.csv",
            OUTPUTS / "move_threshold_verification.json",
            MODELS / "move_binary_model.joblib",
        ],
        _compact_move_result,
    )

    repro_sig = stage_signature(
        sources=[ROOT / "src" / "reproducibility.py", ROOT / "src" / "research_cache.py"],
        inputs=[
            PROCESSED / "model_dataset.csv",
            OUTPUTS / "deep_research_config_lock.json",
            OUTPUTS / "move_research_config_lock.json",
        ],
        params={"manifest": "current"},
    )
    _run_stage(
        checkpoint, status, "reproducibility", "[9/9] 실행 환경, 데이터, 코드 fingerprint 기록",
        repro_sig,
        [OUTPUTS / "reproducibility_manifest.json", OUTPUTS / "pip_freeze.txt"],
        _compact_repro_result,
    )

    status["success"] = True
    _finish(status)
    print("\n완료: 동일 입력의 완료 단계는 다음 실행부터 자동 SKIP됩니다.")
    print("중간에 종료해도 완료된 실험 조합은 reports/cache에서 이어서 사용합니다.")


def _finish(status: dict) -> None:
    status["finished_at"] = datetime.now().isoformat(timespec="seconds")
    write_json(OUTPUTS / "pipeline_status.json", status)

    lines = [
        "# 연구 실행 결과 인계",
        "",
        f"- 실행 시작: {status.get('started_at')}",
        f"- 실행 종료: {status.get('finished_at')}",
        f"- 전체 성공: {status.get('success')}",
        f"- 실행 모드: `{status.get('execution_mode')}`",
        "",
        "## 우선 확인할 문서와 결과",
        "",
        "1. `docs/FINAL_RESEARCH_REPORT.md`",
        "2. `docs/PERSONAL_STUDY_LOG.md`",
        "3. `reports/outputs/move_research_summary.json`",
        "4. `reports/outputs/move_threshold_verification.json`",
        "5. `reports/outputs/move_verification_baselines.csv`",
        "6. `reports/outputs/reproducibility_manifest.json`",
    ]
    (OUTPUTS / "HANDOFF.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

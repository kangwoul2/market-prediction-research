from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

from .common import RAW, INTERIM, ensure_dirs, get_json, normalize_date, safe_save

CRYPTO_LEXICON = {
    "bullish": 2.4, "moon": 2.1, "mooning": 2.2, "breakout": 1.6, "pump": 1.0,
    "hodl": 1.2, "buythedip": 1.7, "ath": 1.3, "adoption": 1.5, "undervalued": 1.4,
    "bearish": -2.4, "dump": -1.8, "crash": -2.5, "rug": -2.8, "rugpull": -3.0,
    "rekt": -2.2, "liquidated": -2.0, "selloff": -2.1, "overvalued": -1.4, "scam": -2.7,
    "fomo": -0.3, "fud": -1.7,
}


def fetch_x_recent(max_pages: int = 5) -> pd.DataFrame:
    """X API 접근 권한이 있을 때 최근 Bitcoin 관련 공개 게시글을 수집합니다.

    장기 학습용 역사 데이터는 X 요금제와 권한에 따라 full-archive 접근이 필요할 수 있으므로
    이 함수는 기본 파이프라인을 막지 않는 선택 모듈입니다.
    """
    ensure_dirs()
    token = os.getenv("X_BEARER_TOKEN", "").strip()
    if not token:
        print("[INFO] X_BEARER_TOKEN 없음: X 최근 게시글 수집 건너뜀")
        return pd.DataFrame()

    endpoint = "https://api.x.com/2/tweets/search/recent"
    headers = {"Authorization": f"Bearer {token}"}
    params = {
        "query": '(bitcoin OR btc OR #bitcoin OR #btc) lang:en -is:retweet',
        "max_results": 100,
        "tweet.fields": "created_at,lang,public_metrics",
    }
    rows = []
    next_token = None
    for _ in range(max_pages):
        call = dict(params)
        if next_token:
            call["next_token"] = next_token
        try:
            payload = get_json(endpoint, params=call, headers=headers)
        except Exception as exc:
            print(f"[WARN] X API 수집 실패: {exc}")
            break
        for item in payload.get("data", []):
            pm = item.get("public_metrics") or {}
            rows.append({
                "id": item.get("id"),
                "created_at": item.get("created_at"),
                "text": item.get("text", ""),
                "lang": item.get("lang"),
                "like_count": pm.get("like_count", 0),
                "retweet_count": pm.get("retweet_count", 0),
                "reply_count": pm.get("reply_count", 0),
                "quote_count": pm.get("quote_count", 0),
            })
        next_token = payload.get("meta", {}).get("next_token")
        if not next_token:
            break
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame.to_csv(RAW / "x_posts.csv", index=False)
    return frame


def _load_posts() -> pd.DataFrame:
    path = RAW / "x_posts.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    text_candidates = [c for c in ("text", "content", "tweet", "post") if c in frame.columns]
    date_candidates = [c for c in ("created_at", "date", "timestamp", "time") if c in frame.columns]
    if not text_candidates or not date_candidates:
        print("[WARN] x_posts.csv에는 text/content/tweet/post 열과 created_at/date/timestamp/time 열이 필요함")
        return pd.DataFrame()
    frame = frame.rename(columns={text_candidates[0]: "text", date_candidates[0]: "created_at"})
    return frame


def _clean_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#", "", text)
    return re.sub(r"\s+", " ", text).strip()


def build_daily_vader() -> pd.DataFrame:
    ensure_dirs()
    frame = _load_posts()
    if frame.empty:
        print("[INFO] X 게시글 데이터 없음: VADER 감성 피처 건너뜀")
        return pd.DataFrame()

    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)

    analyzer = SentimentIntensityAnalyzer()
    analyzer.lexicon.update(CRYPTO_LEXICON)

    frame["date"] = normalize_date(frame["created_at"])
    frame["clean_text"] = frame["text"].fillna("").map(_clean_text)
    scores = frame["clean_text"].map(analyzer.polarity_scores).apply(pd.Series)
    frame = pd.concat([frame, scores.add_prefix("vader_")], axis=1)

    for col in ("like_count", "retweet_count", "reply_count", "quote_count"):
        if col not in frame.columns:
            frame[col] = 0
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0)
    frame["engagement"] = frame[["like_count", "retweet_count", "reply_count", "quote_count"]].sum(axis=1)
    frame["sentiment_weight"] = 1 + np.log1p(frame["engagement"])
    frame["weighted_compound"] = frame["vader_compound"] * frame["sentiment_weight"]

    daily = frame.groupby("date", as_index=False).agg(
        x_post_count=("text", "size"),
        x_sentiment_mean=("vader_compound", "mean"),
        x_sentiment_median=("vader_compound", "median"),
        x_sentiment_std=("vader_compound", "std"),
        x_positive_share=("vader_compound", lambda s: float((s >= 0.05).mean())),
        x_negative_share=("vader_compound", lambda s: float((s <= -0.05).mean())),
        x_engagement=("engagement", "sum"),
        _weighted_sum=("weighted_compound", "sum"),
        _weight_sum=("sentiment_weight", "sum"),
    )
    daily["x_sentiment_weighted"] = daily["_weighted_sum"] / daily["_weight_sum"].replace(0, np.nan)
    daily = daily.drop(columns=["_weighted_sum", "_weight_sum"]).sort_values("date")
    safe_save(frame, INTERIM / "x_posts_scored.csv")
    safe_save(daily, INTERIM / "x_sentiment_daily.csv")
    return daily


def run_sentiment() -> pd.DataFrame:
    if not (RAW / "x_posts.csv").exists():
        fetch_x_recent()
    return build_daily_vader()

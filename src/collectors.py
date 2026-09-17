from __future__ import annotations

from datetime import datetime, timezone
import os

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

from .common import RAW, ensure_dirs, get_json, normalize_date, safe_save, write_json

load_dotenv()

MARKET_TICKERS = {
    "btc": "BTC-USD",
    "eth": "ETH-USD",
    "bnb": "BNB-USD",
    "xrp": "XRP-USD",
    "sol": "SOL-USD",
    "ada": "ADA-USD",
    "doge": "DOGE-USD",
    "ltc": "LTC-USD",
    "spy": "SPY",
    "qqq": "QQQ",
    "gold": "GC=F",
    "dxy": "DX-Y.NYB",
    "vix": "^VIX",
    "tlt": "TLT",
}

BLOCKCHAIN_CHARTS = {
    "hash_rate": "hash-rate",
    "difficulty": "difficulty",
    "transactions": "n-transactions",
    "unique_addresses": "n-unique-addresses",
    "miners_revenue_usd": "miners-revenue",
    "transaction_fees_usd": "transaction-fees-usd",
    "mempool_size": "mempool-size",
    "avg_block_size": "avg-block-size",
    "transactions_per_block": "n-transactions-per-block",
    "estimated_tx_value_usd": "estimated-transaction-volume-usd",
    "mempool_tx_count": "mempool-count",
    "transaction_rate_per_second": "transactions-per-second",
    "median_confirmation_time": "median-confirmation-time",
    "avg_confirmation_time": "avg-confirmation-time",
    "utxo_count": "utxo-count",
    "nvt": "nvt",
}

COINMETRICS_METRICS = [
    "AdrActCnt", "TxCnt", "TxTfrValAdjUSD", "FeeTotUSD", "HashRate",
    "DiffMean", "BlkCnt", "SplyCur", "CapMrktCurUSD", "PriceUSD",
]


def _flatten_yf(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    frame = df.reset_index().copy()
    date_col = "Date" if "Date" in frame.columns else frame.columns[0]
    rename = {}
    for col in ("Open", "High", "Low", "Close", "Adj Close", "Volume"):
        if col in frame.columns:
            rename[col] = f"{prefix}_{col.lower().replace(' ', '_')}"
    keep = [date_col] + [c for c in rename if c in frame.columns]
    frame = frame[keep].rename(columns=rename)
    frame["date"] = normalize_date(frame[date_col])
    frame = frame.drop(columns=[date_col])
    cols = ["date"] + [c for c in frame.columns if c != "date"]
    return frame[cols].dropna(subset=["date"]).drop_duplicates("date").sort_values("date")


def fetch_market_data(start: str = "2015-01-01") -> pd.DataFrame:
    merged = None
    report = []
    for name, ticker in MARKET_TICKERS.items():
        try:
            raw = yf.download(ticker, start=start, interval="1d", auto_adjust=False, progress=False, threads=False)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            frame = _flatten_yf(raw, name)
            if frame.empty:
                report.append({"asset": name, "ticker": ticker, "status": "empty", "rows": 0})
                print(f"[WARN] market data empty: {name} {ticker}")
                continue
            merged = frame if merged is None else merged.merge(frame, on="date", how="outer")
            report.append({"asset": name, "ticker": ticker, "status": "ok", "rows": int(len(frame))})
        except Exception as exc:
            report.append({"asset": name, "ticker": ticker, "status": "failed", "rows": 0, "error": str(exc)})
            print(f"[WARN] market collection failed: {name}: {exc}")
    if merged is None:
        raise RuntimeError("No market data could be collected.")
    merged = merged.sort_values("date").reset_index(drop=True)
    safe_save(merged, RAW / "market_daily.csv")
    write_json(RAW / "market_collection_report.json", {"assets": report})
    return merged


def fetch_fear_greed() -> pd.DataFrame:
    try:
        payload = get_json("https://api.alternative.me/fng/", params={"limit": 0, "format": "json"})
        frame = pd.DataFrame(payload.get("data", []))
        if frame.empty:
            return frame
        frame["date"] = pd.to_datetime(frame["timestamp"].astype(int), unit="s", utc=True).dt.tz_convert(None).dt.normalize()
        frame["fear_greed"] = pd.to_numeric(frame["value"], errors="coerce")
        frame = frame[["date", "fear_greed", "value_classification"]].sort_values("date")
        safe_save(frame, RAW / "fear_greed.csv")
        return frame
    except Exception as exc:
        print(f"[WARN] Fear & Greed collection failed: {exc}")
        return pd.DataFrame()


def fetch_blockchain_charts(timespan: str = "10years") -> pd.DataFrame:
    merged = None
    success, failed = [], []
    for feature, chart in BLOCKCHAIN_CHARTS.items():
        try:
            payload = get_json(
                f"https://api.blockchain.info/charts/{chart}",
                params={"timespan": timespan, "format": "json", "sampled": "false"},
            )
            values = payload.get("values", [])
            if not values:
                failed.append({"feature": feature, "chart": chart, "reason": "empty"})
                continue
            frame = pd.DataFrame(values)
            frame["date"] = pd.to_datetime(frame["x"], unit="s", utc=True).dt.tz_convert(None).dt.normalize()
            frame[feature] = pd.to_numeric(frame["y"], errors="coerce")
            frame = frame[["date", feature]].drop_duplicates("date").sort_values("date")
            merged = frame if merged is None else merged.merge(frame, on="date", how="outer")
            success.append({"feature": feature, "chart": chart, "rows": int(len(frame))})
        except Exception as exc:
            print(f"[WARN] blockchain chart failed: {chart}: {exc}")
            failed.append({"feature": feature, "chart": chart, "reason": str(exc)})
    write_json(
        RAW / "blockchain_com_collection_report.json",
        {"success": success, "failed": failed, "requested_timespan": timespan},
    )
    if merged is None:
        return pd.DataFrame()
    merged = merged.sort_values("date").reset_index(drop=True)
    safe_save(merged, RAW / "onchain_blockchain_com.csv")
    return merged


def fetch_coinmetrics(start: str = "2015-01-01") -> pd.DataFrame:
    endpoint = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
    params = {
        "assets": "btc",
        "metrics": ",".join(COINMETRICS_METRICS),
        "frequency": "1d",
        "start_time": start,
        "page_size": 10000,
    }
    api_key = os.getenv("COINMETRICS_API_KEY", "").strip()
    if api_key:
        params["api_key"] = api_key
    try:
        payload = get_json(endpoint, params=params)
        rows = list(payload.get("data", []))
        next_url = payload.get("next_page_url")
        while next_url:
            nxt = get_json(next_url)
            rows.extend(nxt.get("data", []))
            next_url = nxt.get("next_page_url")
        frame = pd.DataFrame(rows)
        if frame.empty:
            return frame
        frame["date"] = normalize_date(frame["time"])
        keep = ["date"] + [m for m in COINMETRICS_METRICS if m in frame.columns]
        frame = frame[keep]
        for c in frame.columns:
            if c != "date":
                frame[c] = pd.to_numeric(frame[c], errors="coerce")
        frame = frame.rename(columns={c: f"cm_{c}" for c in frame.columns if c != "date"})
        safe_save(frame, RAW / "onchain_coinmetrics.csv")
        return frame
    except Exception as exc:
        print(f"[WARN] Coin Metrics collection failed: {exc}")
        return pd.DataFrame()


def fetch_mempool_snapshot() -> None:
    snapshot = {"collected_at": datetime.now(timezone.utc).isoformat()}
    endpoints = {
        "mempool": "https://mempool.space/api/mempool",
        "recommended_fees": "https://mempool.space/api/v1/fees/recommended",
        "difficulty_adjustment": "https://mempool.space/api/v1/difficulty-adjustment",
    }
    for name, url in endpoints.items():
        try:
            snapshot[name] = get_json(url)
        except Exception as exc:
            snapshot[name] = {"error": str(exc)}
    write_json(RAW / "mempool_latest.json", snapshot)


def fetch_glassnode_optional() -> pd.DataFrame:
    key = os.getenv("GLASSNODE_API_KEY", "").strip()
    if not key:
        print("[INFO] GLASSNODE_API_KEY 없음: Glassnode 선택 수집 건너뜀")
        return pd.DataFrame()
    metrics = {
        "glassnode_active_addresses": "/v1/metrics/addresses/active_count",
        "glassnode_tx_count": "/v1/metrics/transactions/count",
        "glassnode_exchange_balance": "/v1/metrics/distribution/balance_exchanges",
    }
    merged = None
    for name, path in metrics.items():
        try:
            rows = get_json("https://api.glassnode.com" + path, params={"a": "BTC", "i": "24h", "api_key": key})
            frame = pd.DataFrame(rows)
            if frame.empty:
                continue
            frame["date"] = pd.to_datetime(frame["t"], unit="s", utc=True).dt.tz_convert(None).dt.normalize()
            frame[name] = pd.to_numeric(frame["v"], errors="coerce")
            frame = frame[["date", name]]
            merged = frame if merged is None else merged.merge(frame, on="date", how="outer")
        except Exception as exc:
            print(f"[WARN] Glassnode metric failed: {name}: {exc}")
    if merged is not None:
        safe_save(merged.sort_values("date"), RAW / "onchain_glassnode.csv")
        return merged
    return pd.DataFrame()


def collect_all(start: str = "2015-01-01") -> dict[str, int]:
    ensure_dirs()
    inventory = {}
    jobs = {
        "market_daily": lambda: fetch_market_data(start),
        "fear_greed": fetch_fear_greed,
        "blockchain_com": fetch_blockchain_charts,
        "coinmetrics": lambda: fetch_coinmetrics(start),
        "glassnode_optional": fetch_glassnode_optional,
    }
    for name, fn in jobs.items():
        try:
            frame = fn()
            inventory[name] = int(len(frame)) if isinstance(frame, pd.DataFrame) else 0
        except Exception as exc:
            print(f"[WARN] {name} failed: {exc}")
            inventory[name] = 0
    fetch_mempool_snapshot()
    pd.DataFrame([{"source": k, "rows": v} for k, v in inventory.items()]).to_csv(RAW / "collection_inventory.csv", index=False)
    return inventory

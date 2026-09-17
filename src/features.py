from __future__ import annotations

import numpy as np
import pandas as pd

from .common import RAW, INTERIM, PROCESSED, OUTPUTS, ensure_dirs, safe_save, write_json

RETURN_WINDOWS = (1, 2, 3, 5, 7, 10, 14, 21, 30, 45, 60, 90, 180)
ROLL_WINDOWS = (3, 7, 14, 21, 30, 60, 90, 180)
CRYPTO_ASSETS = ("eth", "bnb", "xrp", "sol", "ada", "doge", "ltc")
MACRO_ASSETS = ("spy", "qqq", "gold", "dxy", "vix", "tlt")


def _read_optional(path, date_col="date") -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if date_col not in frame.columns:
        return pd.DataFrame()
    frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce").dt.normalize()
    return frame.dropna(subset=[date_col]).drop_duplicates(date_col).sort_values(date_col)


def _safe_div(a, b):
    return (a / b.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    diff = close.diff()
    gain = diff.clip(lower=0).rolling(window).mean()
    loss = (-diff.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _rolling_beta(asset_ret: pd.Series, market_ret: pd.Series, window: int) -> pd.Series:
    cov = asset_ret.rolling(window, min_periods=max(10, window // 2)).cov(market_ret)
    var = market_ret.rolling(window, min_periods=max(10, window // 2)).var()
    return cov / var.replace(0, np.nan)


def _add_crypto_relative_features(out: pd.DataFrame, market: pd.DataFrame, btc_ret: pd.Series, btc_close: pd.Series) -> list[str]:
    available = []
    return_1d = {}
    return_7d = {}

    for asset in CRYPTO_ASSETS:
        col = f"{asset}_close"
        if col not in market.columns:
            continue
        close = pd.to_numeric(market[col], errors="coerce")
        if close.notna().sum() < 120:
            continue
        available.append(asset)
        r1 = close.pct_change(fill_method=None)
        r7 = close.pct_change(7, fill_method=None)
        return_1d[asset] = r1
        return_7d[asset] = r7
        out[f"btc_crypto_{asset}_return_1d"] = r1
        out[f"btc_crypto_{asset}_return_7d"] = r7
        out[f"btc_crypto_minus_{asset}_1d"] = btc_ret - r1
        out[f"btc_crypto_minus_{asset}_7d"] = btc_close.pct_change(7, fill_method=None) - r7
        out[f"btc_crypto_{asset}_corr_30d"] = btc_ret.rolling(30, min_periods=15).corr(r1)

    if not return_1d:
        return available

    one = pd.DataFrame(return_1d)
    seven = pd.DataFrame(return_7d)
    basket1 = one.mean(axis=1, skipna=True)
    basket7 = seven.mean(axis=1, skipna=True)
    out["btc_crypto_basket_return_1d"] = basket1
    out["btc_crypto_basket_return_7d"] = basket7
    out["btc_crypto_basket_median_1d"] = one.median(axis=1, skipna=True)
    out["btc_crypto_basket_dispersion_1d"] = one.std(axis=1, skipna=True)
    out["btc_crypto_basket_breadth_positive_1d"] = (one > 0).sum(axis=1) / one.notna().sum(axis=1).replace(0, np.nan)
    out["btc_crypto_minus_basket_1d"] = btc_ret - basket1
    out["btc_crypto_minus_basket_7d"] = btc_close.pct_change(7, fill_method=None) - basket7
    out["btc_crypto_basket_corr_30d"] = btc_ret.rolling(30, min_periods=15).corr(basket1)
    out["btc_crypto_basket_corr_90d"] = btc_ret.rolling(90, min_periods=45).corr(basket1)
    out["btc_crypto_basket_beta_30d"] = _rolling_beta(btc_ret, basket1, 30)
    out["btc_crypto_basket_beta_90d"] = _rolling_beta(btc_ret, basket1, 90)
    out["btc_crypto_residual_1d"] = btc_ret - out["btc_crypto_basket_beta_30d"] * basket1
    return available


def build_features(dynamic_k: float = 0.7) -> pd.DataFrame:
    ensure_dirs()
    market = _read_optional(RAW / "market_daily.csv")
    if market.empty or "btc_close" not in market.columns:
        raise RuntimeError("data/raw/market_daily.csv 또는 btc_close가 없습니다. 먼저 데이터 수집을 실행하세요.")

    close = pd.to_numeric(market["btc_close"], errors="coerce")
    open_ = pd.to_numeric(market["btc_open"], errors="coerce")
    high = pd.to_numeric(market["btc_high"], errors="coerce")
    low = pd.to_numeric(market["btc_low"], errors="coerce")
    volume = pd.to_numeric(market["btc_volume"], errors="coerce")
    out = pd.DataFrame({"date": pd.to_datetime(market["date"]), "btc_close": close})

    daily_ret = close.pct_change(fill_method=None)
    log_price = np.log(close)
    log_ret = log_price.diff()
    log_volume = np.log1p(volume)
    range_log = np.log(high / low.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

    for lag in RETURN_WINDOWS:
        out[f"btc_return_{lag}d"] = close.pct_change(lag, fill_method=None)
        out[f"btc_log_return_{lag}d"] = log_price.diff(lag)

    out["btc_body_pct"] = _safe_div(close - open_, open_)
    out["btc_range_pct"] = _safe_div(high - low, close)
    out["btc_upper_wick_pct"] = _safe_div(high - np.maximum(open_, close), close)
    out["btc_lower_wick_pct"] = _safe_div(np.minimum(open_, close) - low, close)
    out["btc_volume_log_change"] = log_volume.diff()

    for window in ROLL_WINDOWS:
        minp = max(3, window // 2)
        ret_roll = daily_ret.rolling(window, min_periods=minp)
        mean = ret_roll.mean()
        vol = ret_roll.std()
        downside = np.sqrt((daily_ret.clip(upper=0).pow(2)).rolling(window, min_periods=minp).mean())
        out[f"btc_return_mean_{window}d"] = mean
        out[f"btc_volatility_{window}d"] = vol
        out[f"btc_downside_vol_{window}d"] = downside
        out[f"btc_return_skew_{window}d"] = ret_roll.skew()
        out[f"btc_price_ma_gap_{window}d"] = close / close.rolling(window, min_periods=minp).mean() - 1
        out[f"btc_price_z_{window}d"] = (close - close.rolling(window, min_periods=minp).mean()) / close.rolling(window, min_periods=minp).std().replace(0, np.nan)
        out[f"btc_volume_z_{window}d"] = (log_volume - log_volume.rolling(window, min_periods=minp).mean()) / log_volume.rolling(window, min_periods=minp).std().replace(0, np.nan)
        out[f"btc_range_mean_{window}d"] = range_log.rolling(window, min_periods=minp).mean()

    out["btc_vol_ratio_7_30"] = _safe_div(out["btc_volatility_7d"], out["btc_volatility_30d"])
    out["btc_vol_ratio_21_90"] = _safe_div(out["btc_volatility_21d"], out["btc_volatility_90d"])
    out["btc_momentum_7_30"] = out["btc_return_7d"] - out["btc_return_30d"]
    out["btc_momentum_30_90"] = out["btc_return_30d"] - out["btc_return_90d"]

    for span in (7, 14, 30, 60, 90, 180):
        ema = close.ewm(span=span, adjust=False).mean()
        out[f"btc_ema_gap_{span}d"] = close / ema - 1
    for window in (7, 14, 28):
        out[f"btc_rsi_{window}"] = _rsi(close, window) / 100.0

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    out["btc_macd_pct"] = macd / close
    out["btc_macd_signal_pct"] = macd.ewm(span=9, adjust=False).mean() / close
    ma20 = close.rolling(20, min_periods=10).mean()
    std20 = close.rolling(20, min_periods=10).std()
    out["btc_bb_position"] = (close - (ma20 - 2 * std20)) / (4 * std20).replace(0, np.nan)
    out["btc_bb_width"] = (4 * std20) / ma20.replace(0, np.nan)

    for window in (30, 90, 180):
        out[f"btc_drawdown_{window}d"] = close / close.rolling(window, min_periods=max(10, window // 2)).max() - 1
        low_roll = close.rolling(window, min_periods=max(10, window // 2)).min()
        high_roll = close.rolling(window, min_periods=max(10, window // 2)).max()
        out[f"btc_range_position_{window}d"] = (close - low_roll) / (high_roll - low_roll).replace(0, np.nan)

    out["btc_price_volume_corr_30d"] = daily_ret.rolling(30, min_periods=15).corr(log_volume.diff())
    out["btc_vol_regime_pctile"] = out["btc_volatility_21d"].rolling(365, min_periods=90).rank(pct=True)
    day_of_week = out["date"].dt.dayofweek.astype(float)
    out["btc_calendar_dow_sin"] = np.sin(2 * np.pi * day_of_week / 7.0)
    out["btc_calendar_dow_cos"] = np.cos(2 * np.pi * day_of_week / 7.0)

    crypto_assets_used = _add_crypto_relative_features(out, market, daily_ret, close)

    macro_assets_used = []
    for asset in MACRO_ASSETS:
        col = f"{asset}_close"
        if col not in market.columns:
            continue
        raw = pd.to_numeric(market[col], errors="coerce")
        if raw.notna().sum() < 30:
            continue
        known = raw.ffill()
        observed_date = out["date"].where(raw.notna()).ffill()
        staleness = (out["date"] - observed_date).dt.days.astype(float)
        macro_ret = known.pct_change(fill_method=None)
        out[f"{asset}_ret_1d_lag1"] = macro_ret.shift(1)
        out[f"{asset}_ret_5d_lag1"] = known.pct_change(5, fill_method=None).shift(1)
        out[f"{asset}_ret_20d_lag1"] = known.pct_change(20, fill_method=None).shift(1)
        out[f"{asset}_vol_20d_lag1"] = macro_ret.rolling(20, min_periods=10).std().shift(1)
        out[f"btc_{asset}_corr_30d_lag1"] = daily_ret.rolling(30, min_periods=15).corr(macro_ret).shift(1)
        out[f"{asset}_staleness_days_lag1"] = staleness.shift(1)
        macro_assets_used.append(asset)

    external_files = [
        RAW / "onchain_blockchain_com.csv",
        RAW / "onchain_coinmetrics.csv",
        RAW / "onchain_glassnode.csv",
        RAW / "fear_greed.csv",
        INTERIM / "x_sentiment_daily.csv",
    ]
    merged_external = out[["date"]].copy()
    external_source_columns = []
    for path in external_files:
        frame = _read_optional(path)
        if frame.empty:
            continue
        cols = [c for c in frame.columns if c != "date"]
        external_source_columns.extend(cols)
        merged_external = merged_external.merge(frame, on="date", how="left")

    for col in [c for c in merged_external.columns if c != "date"]:
        numeric = pd.to_numeric(merged_external[col], errors="coerce")
        if numeric.notna().sum() < 30:
            continue
        shifted = numeric.shift(1)
        out[f"ext_{col}_lag1"] = shifted
        out[f"ext_{col}_diff_1d"] = shifted.diff()
        out[f"ext_{col}_diff_7d"] = shifted.diff(7)
        if (shifted >= 0).mean() > 0.95:
            logged = np.log1p(shifted.clip(lower=0))
            out[f"ext_{col}_log_change_1d"] = logged.diff()
            out[f"ext_{col}_log_change_7d"] = logged.diff(7)
        for window in (30, 90):
            minp = max(10, window // 2)
            mean = shifted.rolling(window, min_periods=minp).mean()
            std = shifted.rolling(window, min_periods=minp).std()
            out[f"ext_{col}_z{window}"] = (shifted - mean) / std.replace(0, np.nan)
        ma7 = shifted.rolling(7, min_periods=4).mean()
        ma30 = shifted.rolling(30, min_periods=15).mean()
        out[f"ext_{col}_ma7_vs_ma30"] = ma7 / ma30.replace(0, np.nan) - 1

    out["target_end_date"] = out["date"].shift(-1)
    out["target_return_1d"] = close.shift(-1) / close - 1
    out["threshold_fixed"] = 0.01
    trailing_vol = log_ret.rolling(21, min_periods=14).std()
    out["threshold_dynamic"] = dynamic_k * trailing_vol
    out["label_fixed"] = np.select(
        [out["target_return_1d"] > 0.01, out["target_return_1d"] < -0.01], [1, -1], default=0
    )
    out["label_dynamic"] = np.select(
        [out["target_return_1d"] > out["threshold_dynamic"], out["target_return_1d"] < -out["threshold_dynamic"]],
        [1, -1], default=0,
    )
    out["target_move_dynamic"] = (out["target_return_1d"].abs() > out["threshold_dynamic"]).astype(int)
    out["target_direction"] = (out["target_return_1d"] > 0).astype(int)

    out = out.replace([np.inf, -np.inf], np.nan)
    safe_save(out, PROCESSED / "model_dataset.csv")

    coverage = [
        {"column": c, "non_null": int(out[c].notna().sum()), "coverage": float(out[c].notna().mean())}
        for c in out.columns
    ]
    pd.DataFrame(coverage).sort_values("coverage").to_csv(OUTPUTS / "feature_coverage.csv", index=False)

    meta = {
        "rows": len(out),
        "columns": len(out.columns),
        "date_start": str(out["date"].min().date()),
        "date_end": str(out["date"].max().date()),
        "dynamic_k": dynamic_k,
        "return_windows": list(RETURN_WINDOWS),
        "rolling_windows": list(ROLL_WINDOWS),
        "crypto_assets_used": crypto_assets_used,
        "external_raw_columns_seen": external_source_columns,
        "macro_assets_used": macro_assets_used,
        "macro_alignment": "휴장일은 직전 관측 종가만 forward-fill하고 계산 결과를 다시 1일 lag 처리",
        "forecast_definition": "date 시점까지 관측 가능한 정보로 target_end_date의 BTC 수익률과 분류 라벨을 예측",
        "temporal_rule": "거시시장과 외생 일 단위 데이터는 보수적으로 지연하고 target_end_date는 누수 검사 전용",
    }
    write_json(OUTPUTS / "dataset_metadata.json", meta)
    return out

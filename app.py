from __future__ import annotations

import json
import re
import threading
import time
from typing import Any

import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

APP_TITLE = "Stock Data API"
DAILY_PERIOD = "1y"
HOURLY_PERIOD = "60d"
CACHE_TTL_SECONDS = 300

app = FastAPI(
    title=APP_TITLE,
    description="yfinanceから日足1年・1時間足60日を取得するAPI",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)

_cache: dict[str, tuple[float, dict[str, Any]]] = {}
_cache_lock = threading.Lock()


def normalize_ticker(code: str) -> tuple[str, str]:
    raw = code.strip().upper().replace(" ", "")
    if not raw:
        raise ValueError("銘柄コードを指定してください。")
    if re.fullmatch(r"[0-9A-Z]{4}", raw):
        return raw, f"{raw}.T"
    return raw, raw


def prepare_ohlcv(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    if df.empty:
        raise ValueError(
            "株価データを取得できませんでした。"
            "銘柄コードまたはYahoo Finance側のデータ提供状況を確認してください。"
        )
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"必要な列がありません: {', '.join(missing)}")
    out = df[required].copy()
    out = out.dropna(subset=["Open", "High", "Low", "Close"], how="all")
    idx = pd.DatetimeIndex(out.index)
    if interval == "1h" and idx.tz is not None:
        idx = idx.tz_convert("Asia/Tokyo").tz_localize(None)
    elif idx.tz is not None:
        idx = idx.tz_localize(None)
    out.index = idx
    if interval == "1d":
        out.index.name = "Date"
        out = out.reset_index()
        out["Date"] = pd.to_datetime(out["Date"]).dt.strftime("%Y-%m-%d")
    else:
        out.index.name = "Datetime"
        out = out.reset_index()
        out["Datetime"] = pd.to_datetime(out["Datetime"]).dt.strftime("%Y-%m-%d %H:%M")
    return out


def dataframe_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(df.to_json(orient="records", force_ascii=False))


def fetch_stock_data(code: str) -> dict[str, Any]:
    file_code, ticker_symbol = normalize_ticker(code)
    with _cache_lock:
        cached = _cache.get(ticker_symbol)
        if cached and time.time() - cached[0] < CACHE_TTL_SECONDS:
            return cached[1]

    ticker = yf.Ticker(ticker_symbol)
    daily_raw = ticker.history(
        period=DAILY_PERIOD,
        interval="1d",
        auto_adjust=False,
        actions=False,
        prepost=False,
        timeout=20,
    )
    hourly_raw = ticker.history(
        period=HOURLY_PERIOD,
        interval="1h",
        auto_adjust=False,
        actions=False,
        prepost=False,
        timeout=20,
    )
    daily = prepare_ohlcv(daily_raw, "1d")
    hourly = prepare_ohlcv(hourly_raw, "1h")
    payload = {
        "code": file_code,
        "ticker": ticker_symbol,
        "daily_period": DAILY_PERIOD,
        "hourly_period": HOURLY_PERIOD,
        "daily_count": len(daily),
        "hourly_count": len(hourly),
        "daily": dataframe_to_records(daily),
        "hourly": dataframe_to_records(hourly),
    }
    with _cache_lock:
        _cache[ticker_symbol] = (time.time(), payload)
    return payload


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "Stock Data API is running",
        "example": "/api/stock/7186",
        "docs": "/docs",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/stock/{code}")
def get_stock(code: str) -> dict[str, Any]:
    try:
        return fetch_stock_data(code)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"データ取得中にエラーが発生しました: {exc}",
        ) from exc

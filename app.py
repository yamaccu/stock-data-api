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
from fastapi.responses import Response

APP_TITLE = "Stock Data API"
DAILY_PERIOD = "1y"
HOURLY_PERIOD = "60d"
CACHE_TTL_SECONDS = 300

app = FastAPI(
    title=APP_TITLE,
    description="yfinanceから日足1年・1時間足60日を取得するAPI",
    version="1.4.0",
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


def pretty_json_response(payload: Any) -> Response:
    content = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    ) + "\n"
    return Response(
        content=content,
        media_type="application/json",
    )


def fetch_interval_data(code: str, interval: str) -> dict[str, Any]:
    file_code, ticker_symbol = normalize_ticker(code)

    if interval == "1d":
        period = DAILY_PERIOD
        interval_name = "daily"
    elif interval == "1h":
        period = HOURLY_PERIOD
        interval_name = "hourly"
    else:
        raise ValueError(f"未対応のintervalです: {interval}")

    cache_key = f"{ticker_symbol}:{interval}"
    with _cache_lock:
        cached = _cache.get(cache_key)
        if cached and time.time() - cached[0] < CACHE_TTL_SECONDS:
            return cached[1]

    ticker = yf.Ticker(ticker_symbol)
    raw = ticker.history(
        period=period,
        interval=interval,
        auto_adjust=False,
        actions=False,
        prepost=False,
        timeout=20,
    )
    data = prepare_ohlcv(raw, interval)

    payload = {
        "code": file_code,
        "ticker": ticker_symbol,
        "interval": interval_name,
        "period": period,
        "count": len(data),
        "data": dataframe_to_records(data),
    }

    with _cache_lock:
        _cache[cache_key] = (time.time(), payload)

    return payload


def fetch_stock_data(code: str) -> dict[str, Any]:
    daily_payload = fetch_interval_data(code, "1d")
    hourly_payload = fetch_interval_data(code, "1h")

    return {
        "code": daily_payload["code"],
        "ticker": daily_payload["ticker"],
        "daily_period": daily_payload["period"],
        "hourly_period": hourly_payload["period"],
        "daily_count": daily_payload["count"],
        "hourly_count": hourly_payload["count"],
        "daily": daily_payload["data"],
        "hourly": hourly_payload["data"],
    }


def handle_api_request(fetcher: Any, *args: Any) -> Response:
    try:
        return pretty_json_response(fetcher(*args))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"データ取得中にエラーが発生しました: {exc}",
        ) from exc


@app.get("/")
def root() -> Response:
    return pretty_json_response(
        {
            "message": "Stock Data API is running",
            "daily_example": "/api/stock/7186/daily",
            "hourly_example": "/api/stock/7186/hourly",
            "combined_example": "/api/stock/7186",
            "docs": "/docs",
        }
    )


@app.get("/health")
def health() -> Response:
    return pretty_json_response({"status": "ok"})


@app.get("/api/stock/{code}/daily")
def get_stock_daily(code: str) -> Response:
    return handle_api_request(fetch_interval_data, code, "1d")


@app.get("/api/stock/{code}/hourly")
def get_stock_hourly(code: str) -> Response:
    return handle_api_request(fetch_interval_data, code, "1h")


@app.get("/api/stock/{code}")
def get_stock(code: str) -> Response:
    return handle_api_request(fetch_stock_data, code)

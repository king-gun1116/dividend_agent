# cache_manager.py
import os
import pickle
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List
import time

import yfinance as yf
from yfinance.exceptions import YFRateLimitError
from ratelimit import limits, sleep_and_retry

# --- 내부 모듈 import (여기에서만!) ---
from .fetcher import download_single
from .stock_api import get_listed_tickers 

# --- 경로 및 디렉토리 설정 ---
BASE_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
CACHE_DIR  = os.path.join(BASE_DIR, "data", "cache")
DB_DIR     = os.path.join(BASE_DIR, "db")
os.makedirs(DB_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
DB_PATH    = os.path.join(DB_DIR, "agent.db")

# --- SQLite 연결 및 테이블 생성 ---
_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
_conn.execute("""
CREATE TABLE IF NOT EXISTS history (
    ticker TEXT,
    date   TEXT,
    open   REAL,
    high   REAL,
    low    REAL,
    close  REAL,
    volume INTEGER,
    PRIMARY KEY(ticker, date)
)
""")
_conn.commit()

# --- 내부 helper 함수 ---
def _date_dir(date: str) -> str:
    dd = os.path.join(CACHE_DIR, date)
    os.makedirs(dd, exist_ok=True)
    return dd

def _cache_path(ticker: str, date: str) -> str:
    return os.path.join(_date_dir(date), f"{ticker}_{date}.pkl")

def get_from_cache(ticker: str, date: str) -> Optional[pd.DataFrame]:
    path = _cache_path(ticker, date)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


# --- 캐시에 저장 (pickle + SQLite upsert) ---
def save_to_cache(df: pd.DataFrame, ticker: str, date: str) -> None:
    if df.empty:
        print(f"[Cache Skip] {ticker}@{date} empty → skip")
        return

    # 1) Pickle 저장
    with open(_cache_path(ticker, date), "wb") as f:
        pickle.dump(df, f)

    # 2) SQLite에 upsert
    rows = [
        (
            ticker,
            idx,
            float(r["Open"]),
            float(r["High"]),
            float(r["Low"]),
            float(r["Close"]),
            int(r["Volume"])
        )
        for idx, r in df.iterrows()
    ]
    _conn.executemany(
        "INSERT OR REPLACE INTO history VALUES (?, ?, ?, ?, ?, ?, ?)",
        rows
    )
    _conn.commit()

# --- 단일 히스토리 조회 ---
def get_history(ticker: str, date: str) -> pd.DataFrame:
    cached = get_from_cache(ticker, date)
    if cached is not None:
        return cached

    df = download_single(ticker, date)
    save_to_cache(df, ticker, date)
    return df

# --- Rate-limit + retry for chunk download ---
CALLS, PERIOD = 2, 1
@sleep_and_retry
@limits(calls=CALLS, period=PERIOD)
def download_chunk(tickers: List[str], date: str) -> pd.DataFrame:
    start = date
    end   = (datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    return yf.download(
        tickers=tickers,
        start=start,
        end=end,
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        threads=False
    )

def download_chunk_with_retry(tickers: List[str], date: str, max_retries: int = 5) -> Optional[pd.DataFrame]:
    """
    RateLimitError 나 기타 오류 발생 시 지수적으로 backoff 하며 재시도
    """
    delay = 1
    for attempt in range(1, max_retries + 1):
        try:
            return download_chunk(tickers, date)
        except YFRateLimitError as e:
            print(f"[download_chunk retry] RateLimitError {attempt}/{max_retries} → {delay}s sleep")
            time.sleep(delay)
            delay *= 2
    return None

# --- bulk_fetch: 전체 티커를 청크 단위로 내려받아 캐시에 저장 ---
def bulk_fetch(date: str, chunk_size: int = 200) -> None:
    """
    1) 상장된 코스피·코스닥 종목 + 지수만
    2) 청크 실패 시 개별 download_single fallback
    """
    all_tickers = get_all_tickers()
    total, saved, skipped = len(all_tickers), 0, 0
    print(f"[Bulk Fetch] {date} ({total} tickers)")

    for i in range(0, total, chunk_size):
        sub      = all_tickers[i : i + chunk_size]
        uncached = [t for t in sub if not os.path.exists(_cache_path(t, date))]
        skipped += len(sub) - len(uncached)
        if not uncached:
            print(f"  ▶ {i+1}-{i+len(sub)}: all cached, skip")
            continue

        time.sleep(5)
        print(f"  ▶ {i+1}-{i+len(sub)}: downloading {len(uncached)} tickers")

        # 1) chunk 단위 시도
        df_chunk = download_chunk_with_retry(uncached, date)
        if df_chunk is None:
            # 2) 전체 청크 실패 시, 개별로 다시 시도
            for tkr in uncached:
                try:
                    df_single = download_single(tkr, date)
                    save_to_cache(df_single, tkr, date)
                    saved += (0 if df_single.empty else 1)
                except Exception as e:
                    print(f"    – [Fallback Error] {tkr}@{date}: {e}")
            continue

        # 3) chunk 성공: MultiIndex 분리 혹은 단일 처리
        if isinstance(df_chunk.columns, pd.MultiIndex):
            for tkr in uncached:
                if tkr in df_chunk.columns.levels[0]:
                    df_tkr = df_chunk[tkr].copy()
                    df_tkr.index = df_tkr.index.strftime("%Y-%m-%d")
                    save_to_cache(df_tkr, tkr, date)
                    saved += 1
                else:
                    print(f"    – {tkr}@{date}: no data in chunk → skip")
        else:
            tkr = uncached[0]
            df  = df_chunk.copy()
            df.index = df.index.strftime("%Y-%m-%d")
            save_to_cache(df, tkr, date)
            saved += 1

    print(f"[Bulk Fetch] done @ {date}: {saved} saved, {skipped} skipped")
    print(f"[DEBUG] Using SQLite DB at: {DB_PATH} (exists: {os.path.exists(DB_PATH)})")
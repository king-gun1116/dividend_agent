# stock_api.py
import os
import pandas as pd
from typing import List

BASE_DIR     = os.path.abspath(os.path.dirname(__file__))
MAPPINGS_DIR = os.path.join(BASE_DIR, "data", "mappings")

KS_PATH = os.path.join(MAPPINGS_DIR, "ticker_mapping_ks.csv")
KQ_PATH = os.path.join(MAPPINGS_DIR, "ticker_mapping_kq.csv")

_df_ks  = pd.read_csv(KS_PATH, encoding="utf-8-sig")
_df_kq  = pd.read_csv(KQ_PATH, encoding="utf-8-sig")
_df_tkr = pd.concat([_df_ks, _df_kq], ignore_index=True)

ticker_to_company = dict(zip(_df_tkr["ticker"], _df_tkr["name"]))

INDEX_TICKERS = {
    'KOSPI':  '^KS11.KS',
    'KOSDAQ': '^KQ11.KQ',
}

def get_listed_tickers() -> List[str]:
    return _df_tkr["ticker"].tolist()

def company_to_ticker(name: str) -> str:
    key     = name.strip().lower()
    matches = _df_tkr[_df_tkr["name"].str.lower() == key]
    if matches.empty:
        raise KeyError(f"Unknown company name: {name}")
    return matches["ticker"].iloc[0]

def get_index_ticker(index_name: str) -> str:
    try:
        return INDEX_TICKERS[index_name.upper()]
    except KeyError:
        raise KeyError(f"Unknown index name: {index_name}")

def get_index_value(index_name: str, date: str, metric: str = "Close") -> float:
    from .cache_manager import get_history  # 내부 import (OK)
    ticker = get_index_ticker(index_name)
    df     = get_history(ticker, date)
    if df.empty:
        raise ValueError(f"No data for index {index_name} on {date}")
    return float(df[metric].iloc[0])

def get_price(name: str, date: str, metric: str = "Close") -> float:
    from .cache_manager import get_history
    ticker = company_to_ticker(name)
    df     = get_history(ticker, date)
    if df.empty:
        raise ValueError(f"No data for {name} on {date}")
    return float(df[metric].iloc[0])
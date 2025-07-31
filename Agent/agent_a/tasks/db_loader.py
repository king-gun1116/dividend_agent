# agent_a/tasks/db_loader.py
import sqlite3
from pathlib import Path
import pandas as pd

# 1) DB 파일 경로
DB_PATH = Path(__file__).parent.parent / "data" / "agent_db.sqlite"

# 2) data_type → 테이블 컬럼 매핑
_COL_MAP = {
    "open":  "open",
    "high":  "high",
    "low":   "low",
    "close": "close",
    "volume":"volume",
}

def load_price_df(data_type: str, tickers: list[str]) -> pd.DataFrame:
    """
    price_daily 테이블에서 date, stock_code, <col> 을 가져와
    date×ticker pivot된 DataFrame을 반환합니다.
    """
    col = _COL_MAP.get(data_type.lower())
    if col is None:
        raise ValueError(f"Unknown data_type: {data_type}")

    # 물음표 플레이스홀더 생성: (?, ?, ?, ...)
    placeholders = ",".join("?" for _ in tickers)

    sql = f"""
        SELECT
          date,
          stock_code AS ticker,
          {col}
        FROM price_daily
        WHERE stock_code IN ({placeholders})
        ORDER BY date
    """

    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(
            sql,
            conn,
            params=tickers,
            parse_dates=["date"]
        )
    # pivot: index=date, columns=ticker, values=col
    return df.pivot(index="date", columns="ticker", values=col).sort_index()
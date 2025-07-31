import os
import sqlite3
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
DB_PATH = os.path.join(DATA_DIR, 'agent_db.sqlite')

def get_history(ticker: str, date: str) -> pd.DataFrame:
    """DB에서 해당 종목, 날짜의 시세 데이터를 불러온다."""
    conn = sqlite3.connect(DB_PATH)
    ticker = ticker.strip().upper()
    date = str(date).strip()
    q = "SELECT * FROM price_daily WHERE stock_code=? AND date=?"
    df = pd.read_sql_query(q, conn, params=(ticker, date))
    conn.close()
    return df

# 사용 예시 (테스트용)
if __name__ == "__main__":
    df = get_history('000020.KS', '2024-11-22')
    print(df)
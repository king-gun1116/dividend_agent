import os
import sqlite3
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)
DB_PATH = os.path.join(DATA_DIR, 'agent_db.sqlite')

def get_connection():
    return sqlite3.connect(DB_PATH)


def get_history(ticker: str, date: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    # 혹시 티커 형식이 다를 수 있으니 .strip().upper()로 보정
    ticker = ticker.strip().upper()
    date = str(date).strip()
    q = "SELECT * FROM price_daily WHERE stock_code=? AND date=?"
    df = pd.read_sql_query(q, conn, params=(ticker, date))
    conn.close()
    return df

if __name__ == "__main__":
    with get_connection() as conn:
        cur = conn.cursor()

        # 1. 일별 시세 테이블 (종목 + 시장 지수 + 각종 질의 대응)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS price_daily (
            date        TEXT NOT NULL,        -- YYYY-MM-DD
            stock_code  TEXT NOT NULL,        -- ex: '005930.KS'
            stock_name  TEXT NOT NULL,        -- ex: '삼성전자'
            market      TEXT,                 -- ex: KOSPI/KOSDAQ
            open        REAL,
            high        REAL,
            low         REAL,
            close       REAL,
            adj_close   REAL,
            volume      REAL,
            -- 계산/추가 지표 컬럼
            value       REAL,                 -- 거래대금(있으면)
            pct_change  REAL,                 -- 등락률(%, close 기준)
            rsi_14      REAL,                 -- RSI(14)
            ma_5        REAL,
            ma_20       REAL,
            ma_60       REAL,
            bb_upper    REAL,                 -- 볼린저밴드 상단
            bb_lower    REAL,                 -- 볼린저밴드 하단
            market_index REAL,                -- 해당 시장 지수 (종목이 아닌 지수 row에서만 값)
            PRIMARY KEY (date, stock_code)
        );
        """)

        # 2. 지표/시그널별 캐시 (신호 탐지/크로스 등)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            stock_code  TEXT,
            stock_name  TEXT,
            date        TEXT,
            type        TEXT,    -- ex: '골든크로스', '데드크로스', 'RSI_과매수', ...
            value       REAL,
            details     TEXT
        );
        """)

        # 3. 질의/응답 캐시 (선택)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS qa_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            answer TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)

        conn.commit()
    print(f"✅ DB 파일 생성 위치: {DB_PATH}")
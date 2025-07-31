import os
import pickle
import sqlite3
import pandas as pd
import numpy as np
from tqdm import tqdm

# === [1] 경로 및 매핑 ===
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

DB_PATH    = os.path.join(DATA_DIR, 'agent_db.sqlite')
CACHE_ROOT = os.path.join(DATA_DIR, 'cache')
MAPPING_FILE = os.path.join(DATA_DIR, 'mappings', 'ticker_name_market.csv')

df_map = pd.read_csv(MAPPING_FILE, dtype=str)
ticker2name   = dict(zip(df_map['ticker'], df_map['name']))
ticker2market = dict(zip(df_map['ticker'], df_map['market']))

DB_COLUMNS = [
    "stock_code", "date", "market", "stock_name",
    "open", "high", "low", "close", "adj_close", "volume",
    "value", "pct_change", "rsi_14",
    "ma_5", "ma_20", "ma_60",
    "bb_upper", "bb_lower",
    "market_index"
]

def calc_derived(df):
    # 원본 컬럼 보정
    if "Open" not in df.columns and "open" in df.columns:
        df = df.rename(columns={k: k.capitalize() for k in df.columns})
    # value
    if 'Open' in df.columns and 'Close' in df.columns and 'Volume' in df.columns:
        df['value'] = df['Close'] * df['Volume']
    # pct_change
    df['pct_change'] = df['Close'].pct_change().fillna(0) * 100
    # 이동평균
    df['ma_5'] = df['Close'].rolling(window=5, min_periods=1).mean()
    df['ma_20'] = df['Close'].rolling(window=20, min_periods=1).mean()
    df['ma_60'] = df['Close'].rolling(window=60, min_periods=1).mean()
    # RSI(14)
    delta = df['Close'].diff()
    up = np.where(delta > 0, delta, 0)
    down = np.where(delta < 0, -delta, 0)
    roll_up = pd.Series(up, index=df.index).rolling(14, min_periods=1).mean()
    roll_down = pd.Series(down, index=df.index).rolling(14, min_periods=1).mean()
    rs = roll_up / (roll_down + 1e-8)
    df['rsi_14'] = 100 - (100 / (1 + rs))
    # 볼린저밴드(20, 2.0)
    ma20 = df['Close'].rolling(20, min_periods=1).mean()
    std20 = df['Close'].rolling(20, min_periods=1).std()
    df['bb_upper'] = ma20 + 2 * std20
    df['bb_lower'] = ma20 - 2 * std20
    return df

def insert_to_db(row, conn):
    sql = f"""
    INSERT OR REPLACE INTO price_daily (
        stock_code, date, market, stock_name,
        open, high, low, close, adj_close, volume,
        value, pct_change, rsi_14, ma_5, ma_20, ma_60,
        bb_upper, bb_lower, market_index
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    conn.execute(sql, tuple(row.get(col, None) for col in DB_COLUMNS))

if __name__ == "__main__":
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # [A] 테이블 생성
    cur.execute("""
    CREATE TABLE IF NOT EXISTS price_daily (
        stock_code  TEXT NOT NULL,
        date        TEXT NOT NULL,
        market      TEXT,
        stock_name  TEXT,
        open        REAL,
        high        REAL,
        low         REAL,
        close       REAL,
        adj_close   REAL,
        volume      REAL,
        value       REAL,
        pct_change  REAL,
        rsi_14      REAL,
        ma_5        REAL,
        ma_20       REAL,
        ma_60       REAL,
        bb_upper    REAL,
        bb_lower    REAL,
        market_index REAL,
        PRIMARY KEY (date, stock_code)
    );
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS signals (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        stock_code  TEXT,
        stock_name  TEXT,
        date        TEXT,
        type        TEXT,
        value       REAL,
        details     TEXT
    );
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS qa_cache (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        question TEXT NOT NULL,
        answer TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    conn.commit()

    # === [B] 피클 파일 경로 모두 모으기
    all_pkl = []
    for date_folder in sorted(os.listdir(CACHE_ROOT)):
        date_path = os.path.join(CACHE_ROOT, date_folder)
        if not os.path.isdir(date_path): continue
        for fn in os.listdir(date_path):
            if fn.endswith('.pkl'):
                all_pkl.append(os.path.join(date_path, fn))

    print(f"총 파일 수: {len(all_pkl):,}")

    # 1. 전체 티커 추출
    all_tickers = set([os.path.basename(fp).split('_')[0] for fp in all_pkl])

    for ticker in tqdm(sorted(all_tickers), desc="티커별 적재"):
        ticker_pkls = [fp for fp in all_pkl if os.path.basename(fp).startswith(f"{ticker}_")]
        df_list = []
        for fp in sorted(ticker_pkls, key=lambda x: os.path.basename(os.path.dirname(x))):
            date = os.path.basename(os.path.dirname(fp))
            try:
                df = pickle.load(open(fp, "rb"))
                if isinstance(df, pd.DataFrame) and not df.empty:
                    # 보통 한 row이지만 혹시라도 multi row라면 모두 날짜 할당
                    df["date"] = date
                    df_list.append(df)
            except Exception as e:
                print(f"[ERROR] {fp}: {e}")
        if not df_list:
            continue
        # === 1. Concat & 정렬
        df_all = pd.concat(df_list).sort_values("date").set_index("date")
        # === 2. 파생지표 한 번에 계산
        df_all = calc_derived(df_all)
        # === 3. DB 적재 (한 날짜 한 row)
        for idx, row in df_all.iterrows():
            try:
                insert_to_db({
                    'stock_code': ticker,
                    'date': idx,
                    'market': ticker2market.get(ticker, ''),
                    'stock_name': ticker2name.get(ticker, ''),
                    'open': row.get('Open'),
                    'high': row.get('High'),
                    'low': row.get('Low'),
                    'close': row.get('Close'),
                    'adj_close': row.get('Adj Close'),
                    'volume': row.get('Volume'),
                    'value': row.get('value'),
                    'pct_change': row.get('pct_change'),
                    'rsi_14': row.get('rsi_14'),
                    'ma_5': row.get('ma_5'),
                    'ma_20': row.get('ma_20'),
                    'ma_60': row.get('ma_60'),
                    'bb_upper': row.get('bb_upper'),
                    'bb_lower': row.get('bb_lower'),
                    'market_index': None
                }, conn)
            except Exception as e:
                print(f"[DB ERROR] {ticker} {idx}: {e}")
    conn.commit()
    conn.close()
    print("✅ DB 적재 완료")
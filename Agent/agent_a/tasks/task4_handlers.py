from pathlib import Path
import pandas as pd
import re
import sqlite3
from datetime import datetime
from inspect import signature
from typing import Dict, Any, Optional, Union
from agent_a.tasks.db_loader import load_price_df

# ====== [경로 세팅] ======
HERE          = Path(__file__).resolve()
AGENT_A_ROOT  = HERE.parent.parent
DATA_DIR      = AGENT_A_ROOT / "data"
DB_PATH       = AGENT_A_ROOT / "data" / "agent_db.sqlite"
MAPPINGS_DIR  = DATA_DIR / "mappings"
KS_PATH       = MAPPINGS_DIR / "ticker_mapping_ks.csv"
KQ_PATH       = MAPPINGS_DIR / "ticker_mapping_kq.csv"

# ====== [매핑 로딩] ======
def load_all_name_map() -> Dict[str, str]:
    ks_df = pd.read_csv(KS_PATH, encoding='utf-8-sig')
    kq_df = pd.read_csv(KQ_PATH, encoding='utf-8-sig')
    name_map_ks = {str(row['ticker']).strip().upper(): str(row['name']).strip()
                   for _, row in ks_df.iterrows()}
    name_map_kq = {str(row['ticker']).strip().upper(): str(row['name']).strip()
                   for _, row in kq_df.iterrows()}
    return {**name_map_ks, **name_map_kq}

def company_to_ticker(company: str, market: Optional[str] = None) -> Optional[str]:
    if not company:
        return None
    name_map = load_all_name_map()
    all_tickers = list(name_map.keys())
    if market == 'KOSPI':
        candidates = [t for t in all_tickers if t.endswith('.KS')]
    elif market == 'KOSDAQ':
        candidates = [t for t in all_tickers if t.endswith('.KQ')]
    else:
        candidates = all_tickers
    company_lower = company.lower()
    for t in candidates:
        if name_map[t].lower() == company_lower or t.lower() == company_lower:
            return t
    for t in candidates:
        if company_lower in name_map[t].lower():
            return t
    return None

def ticker_to_company(ticker: str, market: Optional[str] = None) -> str:
    name_map = load_all_name_map()
    return name_map.get(ticker.upper(), ticker)

# ====== [SQLite 에서 DF 로딩] ======
_column_map = {
    'open':   'open',
    'high':   'high',
    'low':    'low',
    'close':  'close',
    'volume': 'volume',
}

def load_df_from_db(data_type: str, tickers: list[str]) -> pd.DataFrame:
    col = _column_map.get(data_type.lower())
    if col is None:
        raise ValueError(f"Unknown data_type: {data_type}")

    placeholders = ",".join("?" for _ in tickers)
    query = f"""
        SELECT
            date,
            stock_code   AS ticker,
            {col}
        FROM price_daily
        WHERE stock_code IN ({placeholders})
        ORDER BY date
    """

    print(f"[DB로딩] {data_type} for {tickers[:5]} ...({len(tickers)}개)")
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(
            query,
            conn,
            params=tickers,
            parse_dates=['date']
        )
    print(f"[DB로딩] shape={df.shape}, cols={list(df.columns)}")
    df_pivot = df.pivot(index='date', columns='ticker', values=col)
    return df_pivot.sort_index()

# ====== [퍼센트 문자열 자동 변환] ======

def percent_param(val):
    if val is None:
        return None
    if isinstance(val, (float, int)):
        return val
    if isinstance(val, str):
        s = val.replace('%', '').strip()
        import re
        s = re.sub(r'[^\d\.\-\+]', '', s)
        try:
            return float(s) / 100
        except Exception:
            pass
    try:
        return float(val)
    except Exception:
        return None

# ====== [핸들러] ======
def handle_single_metric(
    df_close: pd.DataFrame, df_open: pd.DataFrame, df_high: pd.DataFrame,
    df_low: pd.DataFrame, df_volume: pd.DataFrame,
    ticker: str, date: str, metric: str = "종가", **params
):
    print(f"[handle_single_metric] {ticker=}, {date=}, {metric=}")
    metric_map = {
        "종가": ("close", df_close),
        "시가": ("open", df_open),
        "고가": ("high", df_high),
        "저가": ("low", df_low),
        "거래량": ("volume", df_volume),
        "close": ("close", df_close),
        "open": ("open", df_open),
        "high": ("high", df_high),
        "low": ("low", df_low),
        "volume": ("volume", df_volume),
    }
    col, df = metric_map.get(metric, ("close", df_close))
    date = pd.to_datetime(date)
    if date not in df.index:
        raise ValueError(f"{date.date()} 기준 데이터 없음")
    if ticker not in df.columns:
        raise ValueError(f"{ticker} 종목 데이터 없음")
    value = df.at[date, ticker]
    result = {
        "ticker": ticker,
        "company": ticker_to_company(ticker),
        "date": str(date.date()),
        "metric": metric,
        "value": value,
    }
    print(f"[handle_single_metric] 결과: {result}")
    return result

def handle_recent_rally(df_close: pd.DataFrame, window: int = 7,
                        min_return: Optional[float] = None, threshold: Optional[int] = 10,
                        date: Optional[Union[str, datetime]] = None, **params):
    print(f"[handle_recent_rally] {window=}, {min_return=}, {threshold=}, {date=}")
    min_return = percent_param(min_return)
    if date is not None:
        date = pd.to_datetime(date)
        if date not in df_close.index:
            raise ValueError(f"기준일 {date} 데이터 없음")
        end_idx = df_close.index.get_loc(date)
    else:
        end_idx = -1
    close_now = df_close.iloc[end_idx]
    close_prev = df_close.iloc[end_idx - window]
    returns = (close_now - close_prev) / close_prev
    if min_return is not None:
        returns = returns[returns >= min_return]
    if threshold is not None:
        returns = returns.sort_values(ascending=False).head(threshold)
    result = [
        {
            "company": ticker_to_company(t),
            "ticker": t,
            "return": round(float(r) * 100, 2)
        }
        for t, r in returns.items()
    ]
    print(f"[handle_recent_rally] 결과: {result}")
    return result

def handle_recent_crash(df_close: pd.DataFrame, window: int = 7,
                        max_loss: Optional[float] = None, threshold: Optional[int] = 10,
                        date: Optional[Union[str, datetime]] = None, **params):
    print(f"[handle_recent_crash] {window=}, {max_loss=}, {threshold=}, {date=}")
    max_loss = percent_param(max_loss)
    if date is not None:
        date = pd.to_datetime(date)
        if date not in df_close.index:
            raise ValueError(f"기준일 {date} 데이터 없음")
        end_idx = df_close.index.get_loc(date)
    else:
        end_idx = -1
    close_now = df_close.iloc[end_idx]
    close_prev = df_close.iloc[end_idx - window]
    returns = (close_now - close_prev) / close_prev
    if max_loss is not None:
        returns = returns[returns <= max_loss]
    if threshold is not None:
        returns = returns.sort_values().head(threshold)
    result = [
        {
            "company": ticker_to_company(t),
            "ticker": t,
            "return": round(float(r) * 100, 2)
        }
        for t, r in returns.items()
    ]
    print(f"[handle_recent_crash] 결과: {result}")
    return result

def handle_market_correction(df_close: pd.DataFrame, window: int = 5, drop_pct: float = -0.05,
                             date: Optional[Union[str, datetime]] = None, **params):
    print(f"[handle_market_correction] {window=}, {drop_pct=}, {date=}")
    drop_pct = percent_param(drop_pct)
    if date is not None:
        date = pd.to_datetime(date)
        end_idx = df_close.index.get_loc(date)
    else:
        end_idx = -1
    close_now = df_close.iloc[end_idx]
    close_prev = df_close.iloc[end_idx - window]
    returns = (close_now - close_prev) / close_prev
    result = returns[returns <= drop_pct]
    out = [
        {
            "company": ticker_to_company(t),
            "ticker": t,
            "return": round(float(returns[t]) * 100, 2)
        }
        for t in result.index
    ]
    print(f"[handle_market_correction] 결과: {out}")
    return out

def handle_limit_up_streak(
    df_close: pd.DataFrame,
    df_high: pd.DataFrame,
    window: int = 5,
    limit_up_ratio: float = 0.297,
    min_streak: int = 2,
    date: Optional[Union[str, datetime]] = None,
    **params
):
    print(f"[handle_limit_up_streak] {window=}, {limit_up_ratio=}, {min_streak=}, {date=}")

    if date is not None:
        date = pd.to_datetime(date)
        end_idx = df_close.index.get_loc(date)
        start_idx = end_idx - window + 1
        if start_idx < 0:
            print(f"⚠️ start_idx({start_idx})가 0보다 작음")
            return []
        closes = df_close.iloc[start_idx:end_idx + 1]
        highs  = df_high.iloc[start_idx:end_idx + 1]
    else:
        closes = df_close.iloc[-window:]
        highs  = df_high.iloc[-window:]

    print(f"[handle_limit_up_streak] 실제 window 슬라이스 shape={closes.shape}")

    # 데이터가 충분한지 체크
    if closes.shape[0] < min_streak:
        print("⚠️ 데이터 개수가 min_streak보다 적음.")
        return []

    closes_shift = closes.shift(1)
    is_limit_up = ((closes == highs) & ((closes / closes_shift - 1) >= limit_up_ratio)).astype(int)
    rolling_sum = is_limit_up.rolling(window=min_streak).sum()
    if rolling_sum.shape[0] == 0:
        print("⚠️ rolling 결과가 비었습니다.")
        return []

    streak = rolling_sum.iloc[-1]
    result = streak[streak >= min_streak]

    out = [
        {
            "company": ticker_to_company(t),
            "ticker": t,
            "limit_streak": int(streak[t])
        }
        for t in result.index
    ]
    print(f"[handle_limit_up_streak] 결과: {out}")
    return out

def handle_candlestick_pattern(
    df_open: pd.DataFrame,
    df_close: pd.DataFrame,
    days: int = 3,
    pattern: str = 'three_white_soldiers',
    date: Optional[Union[str, datetime]] = None,
    **params
):
    if date is not None:
        date = pd.to_datetime(date)
        end_idx = df_open.index.get_loc(date)
    else:
        end_idx = -1

    start_idx = end_idx - days + 1
    # 최근 days 구간 슬라이싱
    opens = df_open.iloc[start_idx:end_idx + 1]
    closes = df_close.iloc[start_idx:end_idx + 1]

    if pattern == 'three_white_soldiers':
        # 1. 종가 > 시가, 2. 종가가 전일보다 상승 (시작일 제외)
        above_open = (closes > opens)
        close_increase = closes.diff().iloc[1:].apply(lambda x: x > 0)
        mask = (above_open.all(axis=0)) & (close_increase.all(axis=0))
        result = mask[mask].index.tolist()
    elif pattern == 'three_black_crows':
        # 1. 종가 < 시가, 2. 종가가 전일보다 하락 (시작일 제외)
        below_open = (closes < opens)
        close_decrease = closes.diff().iloc[1:].apply(lambda x: x < 0)
        mask = (below_open.all(axis=0)) & (close_decrease.all(axis=0))
        result = mask[mask].index.tolist()
    else:
        return []

    # 결과 정제: 회사명/티커 반환
    return [
        {
            "company": ticker_to_company(t),
            "ticker": t,
            "signal": pattern,
        }
        for t in result
    ]

def handle_technical_pattern(df_close, ma_short=5, ma_long=20, pattern_type='golden_cross', date=None, **params):
    print(f"[handle_technical_pattern] {ma_short=}, {ma_long=}, {pattern_type=}, {date=}")
    ma_s = df_close.rolling(window=ma_short).mean()
    ma_l = df_close.rolling(window=ma_long).mean()
    cross = (ma_s > ma_l).astype(int)
    if date is not None:
        date = pd.to_datetime(date)
        idx = df_close.index.get_loc(date)
    else:
        idx = -1
    if pattern_type == 'golden_cross':
        result = ((cross.iloc[idx - 1] == 0) & (cross.iloc[idx] == 1))
    else:
        result = ((cross.iloc[idx - 1] == 1) & (cross.iloc[idx] == 0))
    out = [
        {
            "company": ticker_to_company(t),
            "ticker": t,
            "signal": pattern_type
        }
        for t in result[result].index
    ]
    print(f"[handle_technical_pattern] 결과: {out}")
    return out

def handle_historical_extreme(df: pd.DataFrame, metric: str = 'close', start_date: Optional[str] = None,
                              threshold: Optional[int] = 10, **params):
    print(f"[handle_historical_extreme] {metric=}, {start_date=}, {threshold=}")
    if start_date:
        df = df[df.index >= pd.to_datetime(start_date)]
    change = (df.iloc[-1] - df.iloc[0]) / df.iloc[0]
    if threshold:
        change = change.sort_values(ascending=False).head(threshold)
    out = [
        {
            "company": ticker_to_company(t),
            "ticker": t,
            "change": round(float(change[t]) * 100, 2)
        }
        for t in change.index
    ]
    print(f"[handle_historical_extreme] 결과: {out}")
    return out

def handle_panic_sell(df_close: pd.DataFrame, df_volume: pd.DataFrame, window: int = 3,
                      max_loss: float = -0.07, vol_spike: float = 2.0,
                      date: Optional[Union[str, datetime]] = None, **params):
    print(f"[handle_panic_sell] {window=}, {max_loss=}, {vol_spike=}, {date=}")
    max_loss = percent_param(max_loss)
    print(f"[handle_panic_sell] 변환된 max_loss: {max_loss}")
    if date is not None:
        date = pd.to_datetime(date)
        end_idx = df_close.index.get_loc(date)
    else:
        end_idx = -1
    close_now = df_close.iloc[end_idx]
    close_prev = df_close.iloc[end_idx - window]
    ret = (close_now - close_prev) / close_prev
    vol_now = df_volume.iloc[end_idx]
    vol_prev = df_volume.iloc[end_idx - window]
    vol_ratio = vol_now / (vol_prev + 1e-6)
    print(f"[handle_panic_sell] 수익률 min: {ret.min()}, max: {ret.max()}, 거래량비 min: {vol_ratio.min()}, max: {vol_ratio.max()}")
    result = ret[(ret <= max_loss) & (vol_ratio >= vol_spike)]
    print(f"[handle_panic_sell] 조건부 결과 count: {len(result)}")
    out = [
        {
            "company": ticker_to_company(t),
            "ticker": t,
            "return": round(float(ret[t]) * 100, 2),
            "vol_ratio": round(float(vol_ratio[t]), 2)
        }
        for t in result.index
    ]
    print(f"[handle_panic_sell] 결과: {out[:3]} ...총 {len(out)}개")
    return out

# ====== [핸들러 매핑] ======
HANDLER_MAP = {
    'recent_rally': handle_recent_rally,
    'single_metric': handle_single_metric,
    'recent_crash': handle_recent_crash,
    'market_correction': handle_market_correction,
    'limit_up_streak': handle_limit_up_streak,
    'candlestick_pattern': handle_candlestick_pattern,
    'technical_pattern': handle_technical_pattern,
    'historical_extreme': handle_historical_extreme,
    'panic_sell': handle_panic_sell,
}

# ====== [Dispatcher] ======
def task4_dispatch(parsed: Dict[str, Any]) -> Any:
    print("[task4_dispatch] 파싱 결과:", parsed)
    qtype   = parsed.get('type')
    date    = parsed.get('date')
    raw_window = parsed.get('window', 7)
    # 자동 완화
    if isinstance(raw_window, str):
        m = re.match(r'^(\d+)', raw_window)
        window = int(m.group(1)) if m else 7
    else:
        window = int(raw_window)
    # 너무 짧은 window는 7로 보정
    if window < 5:
        print(f"⚠️ window={window} 너무 짧음 → 7로 보정")
        window = 7

    # 변동폭 (max_var) 등 숫자 파라미터도 완화
    max_var = parsed.get('max_var')
    max_var_f = percent_param(max_var)
    if max_var_f is not None and max_var_f < 0.05:   # 5% 미만이면
        print(f"⚠️ max_var={max_var}({max_var_f}) 너무 타이트 → 0.1(10%)로 보정")
        max_var = '10%'   # 또는 0.1 등
        parsed['max_var'] = max_var
    if isinstance(raw_window, str):
        import re
        m = re.match(r'^(\d+)', raw_window)
        window = int(m.group(1)) if m else 7
    else:
        window = int(raw_window)
    market  = parsed.get('market')
    company = parsed.get('ticker')
    params = {k: v for k, v in parsed.items()
              if k not in ('type','date','window','market','ticker')}
    ticker = company_to_ticker(company, market) if company else None
    if ticker:
        tickers = [ticker]
    else:
        name_map = load_all_name_map()
        tickers = [
            t for t in name_map
            if market is None
            or (market=='KOSPI' and t.endswith('.KS'))
            or (market=='KOSDAQ' and t.endswith('.KQ'))
        ]
    print(f"[task4_dispatch] 핸들러={qtype}, ticker={ticker}, tickers={tickers[:5]} ...({len(tickers)}개)")
    df_close  = load_df_from_db('close',  tickers)
    df_open   = load_df_from_db('open',   tickers)
    df_high   = load_df_from_db('high',   tickers)
    df_low    = load_df_from_db('low',    tickers)
    df_volume = load_df_from_db('volume', tickers)
    handler = HANDLER_MAP.get(qtype)
    if handler is None:
        raise ValueError(f"Unsupported type: {qtype}")
    params.pop('window', None)
    df_args = {
        'df_close':  df_close,
        'df_open':   df_open,
        'df_high':   df_high,
        'df_low':    df_low,
        'df_volume': df_volume,
        'window':    window,
        'date':      date,
        **params,
    }
    if qtype == 'single_metric' and company:
        df_args['ticker'] = ticker
    if qtype == 'historical_extreme':
        df_args['df'] = df_close     
    sig = signature(handler)
    filtered = {k: v for k, v in df_args.items() if k in sig.parameters}
    print(f"[task4_dispatch] 핸들러에 전달하는 인자: {filtered}")
    try:
        result = handler(**filtered)
    except Exception as e:
        import traceback
        print("❌ 핸들러 내부에서 에러 발생!")
        print("❌ Error:", e)
        traceback.print_exc()  # 전체 스택트레이스
        return f"❌ 핸들러 실행 중 에러 발생: {str(e)}"

    if isinstance(result, pd.Series):
        result.index = [ticker_to_company(t, market) for t in result.index]
    elif isinstance(result, dict):
        result = {ticker_to_company(k, market): v for k, v in result.items()}
    elif isinstance(result, pd.Index):
        result = pd.Index([ticker_to_company(t, market) for t in result])
    # 결과 없을 때 메시지 반환
    if (isinstance(result, (pd.Series, list, dict, set)) and not result) or result is None:
        print("⚠️ 조건에 맞는 종목이 없습니다.")
        return "⚠️ 조건에 맞는 종목이 없습니다."
    print(f"[task4_dispatch] 최종 결과: {result}")
    return result

if __name__ == "__main__":
    import pprint
    from agent_a.tasks.task4_llm import parse_task4
    TEST_QUESTIONS = [
        "패닉셀 터진 종목 알려줘",
        "골든크로스 신호 발생한 주식 알려줘",
        "데드크로스 신호 뜬 종목",
        "적삼병 신호 종목 알려줘",
        "흑삼병 신호 포착된 종목",
        "최근 박스권에 머문 종목",
        "횡보장 주식 리스트",
        "조정장 종목 알려줘",
        "3일 연속 상한가 간 종목",
        "역대급 수익률 주식 알려줘",
        "바닥찍고 반등한 주식",
    ]
    for question in TEST_QUESTIONS:
        print("="*60)
        print(f"질문: {question}")
        parsed = parse_task4(question)
        print("\n[파싱 결과]")
        pprint.pprint(parsed)
        try:
            print("\n[핸들러 진입 및 결과]")
            result = task4_dispatch(parsed)
            pprint.pprint(result)
        except Exception as e:
            print("❌ 에러 발생:", e)
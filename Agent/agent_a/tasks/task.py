import sqlite3
import os
from typing import Dict, Any, List, Optional
import pandas as pd
from rapidfuzz import process, fuzz

from agent_a.parser import parse_question  # 위 리팩토링된 파서

# ───────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.path.join(BASE_DIR, "agent_a", "data", "agent_db.sqlite")
_conn = sqlite3.connect(DB_PATH, check_same_thread=False, detect_types=sqlite3.PARSE_DECLTYPES)

def _query(sql: str, params: tuple=()) -> List[tuple]:
    return _conn.execute(sql, params).fetchall()

def get_from_db(code: str, date: str, cols: List[str]) -> Dict[str, Any]:
    if not code or not date:
        return {}
    fields = ", ".join(cols)
    sql = f"SELECT {fields} FROM price_daily WHERE stock_code=? AND date=?"
    row = _conn.execute(sql, (code, date)).fetchone()
    return dict(zip(cols, row)) if row else {}

def find_prev_trading_date(date: str) -> Optional[str]:
    row = _conn.execute(
        "SELECT MAX(date) FROM price_daily WHERE date<?", (date,)
    ).fetchone()
    return row[0] if row else None

def get_latest_date(code: str) -> Optional[str]:
    row = _conn.execute(
        "SELECT MAX(date) FROM price_daily WHERE stock_code=?", (code,)
    ).fetchone()
    return row[0] if row else None

def get_listed_tickers() -> List[str]:
    """
    price_daily 테이블에 존재하는 모든 stock_code 목록을 반환합니다.
    """
    rows = _conn.execute("SELECT DISTINCT stock_code FROM price_daily").fetchall()
    return [r[0] for r in rows]

def format_value(col: str, val: Any) -> str:
    if val is None:
        return "데이터 없음"
    if col in ("open","high","low","close","adj_close","market_index","value"):
        return f"{int(val):,}원"
    if col=="volume":
        return f"{int(val):,}주"
    if col=="pct_change":
        return f"{val:.2f}%"
    if col=="rsi_14":
        return f"{val:.1f}"
    return str(val)

# 종목명 매핑
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MAPPING_CSV = os.path.join(BASE_DIR, "agent_a", "data", "mappings", "ticker_name_market.csv")

_map_df = pd.read_csv(MAPPING_CSV, dtype=str)
NAME_MAP = _map_df.set_index("ticker")["name"].to_dict()

def normalize_stock_name(s: str) -> str:
    return "".join(s.lower().split())

def find_code_by_name(name: str) -> Optional[str]:
    norm = normalize_stock_name(name)
    for tk, nm in NAME_MAP.items():
        if normalize_stock_name(nm)==norm:
            return tk
    choices = list(NAME_MAP.values())
    match, score, idx = process.extractOne(norm, choices, scorer=fuzz.partial_ratio)
    return list(NAME_MAP.keys())[idx] if score>=80 else None

# ───────────────────────────────────────────────────────────────
def handle_single_metric(p: Dict[str, Any]) -> str:
    code = find_code_by_name(p["name"])
    date = p.get("date") or get_latest_date(code)
    col  = p["metric"]

    from datetime import datetime
    # 1) 실제 DB 조회
    rec = get_from_db(code, date, [col])

    # 2) 주말(토·일): 거래일 아님
    dt = datetime.strptime(date, "%Y-%m-%d")
    if rec.get(col) is None and dt.weekday() >= 5:
        return "해당일은 거래일이 아닙니다"

    # 3) 평일인데 값 누락: 고가/저가 등 구분 메시지
    if rec.get(col) is None:
        # metric → 한글명 매핑
        km = {"open":"시가","high":"고가","low":"저가","close":"종가","adj_close":"종가",
              "volume":"거래량","pct_change":"등락률","value":"거래대금"}
        txt = km.get(col, col)
        return f"해당 날짜 {txt} 데이터 없음"

    # 4) 정상 응답
    return format_value(col, rec.get(col))

def handle_topn(p: Dict[str, Any]) -> str:
    date, market = p["date"], p.get("market")
    col, n       = p.get("metric","volume"), int(p.get("rank",5))
    sql = f"""
      SELECT stock_name, {col}
        FROM price_daily
       WHERE date=? AND {col} IS NOT NULL
         {' AND market=?' if market else ''}
       ORDER BY {col} DESC
       LIMIT ?
    """
    params = (date, market, n) if market else (date, n)
    rows = _query(sql, params)
    if not rows:
        return f"{date} 데이터 없음"
    return ", ".join(f"{i+1}위 {nm}({val:,.2f})"
                     for i,(nm,val) in enumerate(rows))

def handle_count_by_pchange(p: Dict[str, Any]) -> str:
    date, market, direction = p["date"], p.get("market"), p["direction"]
    op = ">" if direction=="up" else "<"
    sql = """
      SELECT COUNT(*) FROM price_daily
       WHERE date=? AND pct_change {op} 0
         AND market IS NOT NULL
    """.format(op=op)
    params = [date]
    if market:
        sql += " AND market=?"; params.append(market)
    cnt = _conn.execute(sql, tuple(params)).fetchone()[0]
    return f"{cnt}개"

def handle_market_stats(p: Dict[str, Any]) -> str:
    date, market, metric = p["date"], p.get("market"), p["metric"]

    # 지수
    if metric=="market_index":
        mi = pd.read_csv("agent_a/data/ks_kq.csv")
        val = mi.loc[mi.date==date, market].iat[0]
        return f"{val:.2f}"

    # 거래대금 합계
    if metric=="value":
        sql = "SELECT SUM(value) FROM price_daily WHERE date=?"
        params=[date]
        if market and market!="ALL":
            sql += " AND market=?"; params.append(market)
        tot = _conn.execute(sql, tuple(params)).fetchone()[0]
        return f"{int(tot):,}원"

    # 거래된 종목 수
    if metric=="count":
        sql = "SELECT COUNT(DISTINCT stock_code) FROM price_daily WHERE date=?"
        params=[date]
        if market and market!="ALL":
            sql += " AND market=?"; params.append(market)
        cnt = _conn.execute(sql, tuple(params)).fetchone()[0]
        return f"{cnt}개"

    return "지원하지 않는 통계입니다"

# ───────────────────────────────────────────────────────────────
def handle_combined_pct_volume(p: Dict[str,Any]) -> str:
    date    = p["date"]
    market  = p.get("market")
    pct     = p["pct_sign"]
    op_pct  = ">=" if p["pct_comp"] in ("이상","초과") else "<="
    vol_pct = p["vol_pct"]
    op_vol  = ">=" if p["vol_comp"] in ("이상","초과") else "<="
    prev    = find_prev_trading_date(date)

    # 전일 대비 거래량 증감 비율 계산 SQL
    # 서브쿼리에서 전일 volume 을 가져와서 비교
    sql = f"""
      WITH today AS (
        SELECT stock_code, stock_name, pct_change, volume
        FROM price_daily
        WHERE date = ?
        { "AND market=?" if market else "" }
      ),
      yesterday AS (
        SELECT stock_code, volume AS vol_prev
        FROM price_daily
        WHERE date = ?
      )
      SELECT t.stock_name
      FROM today t
      JOIN yesterday y USING(stock_code)
      WHERE t.pct_change {op_pct} ?
        AND ((t.volume*1.0 / y.vol_prev - 1)*100) {op_vol} ?
    """
    params = [date]
    if market:
        params.append(market)
    params += [prev, pct, vol_pct]

    rows = _query(sql, tuple(params))
    if not rows:
        return "조건에 맞는 종목 없음"
    return ", ".join(r[0] for r in rows)

def handle_volume_threshold(p: Dict[str, Any]) -> str:
    """
    날짜·시장·거래량 임계치(절대치) 조건으로 종목 필터링
    """
    date      = p["date"]
    market    = p.get("market")
    thresh    = p["threshold"]
    comp      = p["comp"]
    op        = ">=" if comp in ("이상","초과") else "<="
    sql = f"""
      SELECT stock_name, volume
        FROM price_daily
       WHERE date = ?
         AND volume {op} ?
         { 'AND market = ?' if market else '' }
    """
    params = [date, thresh]
    if market:
        params.append(market)
    rows = _query(sql, tuple(params))
    if not rows:
        return "조건에 맞는 종목 없음"
    # “종목 (거래량)” 형태로 반환
    return ", ".join(f"{nm} ({int(vol):,}주)" for nm,vol in rows)

def handle_rsi_condition(p):
    date, th, comp = p["date"], p["threshold"], p["comp"]
    op = ">=" if comp in ("이상","과매수") else "<="
    sql = f"""
      SELECT stock_name||'(RSI:'||printf('%.1f',rsi_14)||')'
        FROM price_daily
       WHERE date=? AND rsi_14 {op} ?
    """
    rows = _conn.execute(sql, (date, th)).fetchall()
    return ", ".join(r[0] for r in rows) or "조건에 맞는 종목 없음"

def handle_vol_ma_spike(p: Dict[str, Any]) -> str:
    date, win, pct = p["date"], p["window"], p["threshold"]
    prevs = _query(
        "SELECT DISTINCT date FROM price_daily WHERE date<? ORDER BY date DESC LIMIT ?",
        (date, win)
    )
    dates = [d[0] for d in prevs][::-1]
    if not dates:
        return "조건에 맞는 종목 없음"
    res = []
    for code in get_listed_tickers():
        # 1. 20일 평균
        df = pd.read_sql_query(
            "SELECT volume FROM price_daily WHERE stock_code=? AND date IN ({})".format(",".join("?"*len(dates))),
            _conn, params=[code]+dates
        )
        if df.empty or "volume" not in df.columns:
            continue
        ma = df["volume"].mean()
        # 2. 해당일 거래량
        cur = get_from_db(code, date, ["volume"]).get("volume")
        if cur is None or ma == 0:
            continue
        change = (cur / ma - 1) * 100
        if change >= pct:
            res.append(f"{NAME_MAP.get(code,code)}({change:.0f}%)")
    return ", ".join(res) or "조건에 맞는 종목 없음"

def handle_price_range(p: Dict[str, Any]) -> str:
    date, low, high = p["date"], p["low"], p["high"]
    sql = "SELECT stock_name FROM price_daily WHERE date=? AND adj_close BETWEEN ? AND ?"
    rows = _query(sql, (date, low, high))
    return ", ".join(r[0] for r in rows) or "조건에 맞는 종목 없음"

def handle_pct_change_threshold(p: Dict[str, Any]) -> str:
    date = p["date"]
    market = p.get("market")
    threshold = p["threshold"]
    comp = p["comp"]
    op = {
        "이상": ">=",
        "초과": ">",
        "이하": "<=",
        "미만": "<"
    }.get(comp, ">=")
    sql = f"""
      SELECT stock_name
      FROM price_daily
      WHERE date=? AND pct_change {op} ?
      { 'AND market=?' if market else '' }
    """
    params = [date, threshold]
    if market:
        params.append(market)
    rows = _query(sql, tuple(params))
    return ", ".join(r[0] for r in rows) or "조건에 맞는 종목 없음"

def handle_volume_prevday_pct(p: Dict[str, Any]) -> str:
    date = p["date"]
    market = p.get("market")
    threshold = p["threshold"]
    comp = p["comp"]
    op = {
        "이상": ">=",
        "초과": ">",
        "이하": "<=",
        "미만": "<"
    }.get(comp, ">=")
    prev = find_prev_trading_date(date)
    if not prev:
        return "조건에 맞는 종목 없음"
    sql = f"""
      WITH today AS (
        SELECT stock_code, stock_name, volume
        FROM price_daily
        WHERE date = ?
        { "AND market=?" if market else "" }
      ),
      yesterday AS (
        SELECT stock_code, volume AS vol_prev
        FROM price_daily
        WHERE date = ?
      )
      SELECT t.stock_name
      FROM today t
      JOIN yesterday y USING(stock_code)
      WHERE ( (t.volume*1.0 / NULLIF(y.vol_prev,0) - 1) * 100 ) {op} ?
    """
    params = [date]
    if market:
        params.append(market)
    params.append(prev)
    params.append(threshold)
    rows = _query(sql, tuple(params))
    return ", ".join(r[0] for r in rows) or "조건에 맞는 종목 없음"

def handle_ma_condition(p: Dict[str, Any]) -> str:
    date, win, diff, comp = p["date"], p["window"], p["diff"], p["comp"]
    fld = f"ma_{win}"
    op  = ">=" if comp in ("이상","초과") else "<="
    sql = f"""
        SELECT stock_name||'('||printf('%.1f',(adj_close/{fld}-1)*100)||'%)'
          FROM price_daily WHERE date=? AND adj_close/{fld}{op}?
    """
    rows = _query(sql, (date, 1+diff/100))
    return ", ".join(r[0] for r in rows) or "조건에 맞는 종목 없음"

def handle_bollinger_touch(p: Dict[str, Any]) -> str:
    date, band = p["date"], p["band"]
    fld = "bb_upper" if band=="상단" else "bb_lower"
    op  = ">=" if band=="상단" else "<="
    sql = f"SELECT stock_name FROM price_daily WHERE date=? AND adj_close{op}{fld}"
    rows = _query(sql, (date,))
    return ", ".join(r[0] for r in rows) or "조건에 맞는 종목 없음"

def handle_cross_count(p):
    start, end, kind = p["start"], p["end"], p["cross"]
    # 골든크로스인지 데드크로스인지에 따라 SQL 조건을 선택
    if kind == "골든크로스":
        cond = "prev_d <= 0 AND d > 0"
    else:
        cond = "prev_d >= 0 AND d < 0"

    sql = f"""
      WITH diff AS (
        SELECT
          stock_code,
          date,
          ma_5 - ma_20 AS d,
          LAG(ma_5 - ma_20) OVER (PARTITION BY stock_code ORDER BY date) AS prev_d
        FROM price_daily
        WHERE date BETWEEN ? AND ?
      )
      SELECT DISTINCT stock_name
      FROM price_daily
      JOIN diff USING(stock_code, date)
      WHERE date BETWEEN ? AND ?
        AND {cond}
    """
    rows = _conn.execute(sql, (start, end, start, end)).fetchall()
    return ", ".join(r[0] for r in rows) or "조건에 맞는 종목 없음"

# ───────────────────────────────────────────────────────────────
TYPE_HANDLERS = {
    "single_metric":      handle_single_metric,
    "single_metric_flex": handle_single_metric,
    "topn":               handle_topn,
    "count_by_pchange":   handle_count_by_pchange,
    "market_stats":       handle_market_stats,
    "rsi_condition":      handle_rsi_condition,
    "volume_change_pct":    handle_vol_ma_spike,
    "combined_pct_volume":  handle_combined_pct_volume,
    "volume_threshold":   handle_volume_threshold,
    "price_range":        handle_price_range,
    "vol_ma_spike":       handle_vol_ma_spike,
    "ma_condition":       handle_ma_condition,
    "bollinger_touch":    handle_bollinger_touch,
    "cross_count":        handle_cross_count,
    "pct_change_threshold": handle_pct_change_threshold,
    "volume_prevday_pct": handle_volume_prevday_pct,
}

def run(question: str) -> str:
    params = parse_question(question)
    fn = TYPE_HANDLERS.get(params.get("type"))
    if not fn:
        return "죄송합니다. 아직 지원하지 않는 질문입니다."
    try:
        return fn(params)
    except Exception as e:
        return f"처리 중 오류: {e}"

# ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import pandas as pd
    import os

    # 1) CSV 파일들이 위치한 경로를 정확히 지정
    base = "/Users/gun/Desktop/미래에셋 AI 공모전/미래에셋 (1)"

    # 2) 불러올 파일 리스트
    paths = [
        os.path.join(base, "simple_queries.csv"),
        os.path.join(base, "conditional_queries.csv"),
        os.path.join(base, "signal_queries.csv"),
    ]

    # 2) 데이터 프레임으로 합치기
    dfs = []
    for p in paths:
        df = pd.read_csv(p, encoding="utf-8-sig")
        if 'question' not in df.columns:
            df = df.rename(columns={df.columns[0]: 'question'})
        if 'expected_answer' not in df.columns:
            df['expected_answer'] = ""
        dfs.append(df[['question', 'expected_answer']])
    all_df = pd.concat(dfs, ignore_index=True)

    # 3) run & 평가
    correct = 0
    total = len(all_df)
    wrong_rows = []
    for _, row in all_df.iterrows():
        q, exp = row['question'], row['expected_answer']
        out = run(q)
        if out.strip() == str(exp).strip():
            correct += 1
        else:
            wrong_rows.append({
                'question': q,
                'expected_answer': exp,
                'answer': out
            })

    # 4) 결과 출력
    acc = correct / total * 100
    print(f"\n총 {total}개 중 정답 {correct}개 ({acc:.1f}%)")

    # 5) 오답만 CSV로 저장
    if wrong_rows:
        wrong_df = pd.DataFrame(wrong_rows)
        wrong_path = os.path.join("agent_a", "data", "queries", "wrong_only.csv")
        wrong_df.to_csv(wrong_path, index=False, encoding="utf-8-sig")
        print(f"오답 only: {len(wrong_rows)}개 → {wrong_path} 저장")
    else:
        print("오답 없음! 🎉")
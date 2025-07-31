# parser.py

import re
import pandas as pd
from typing import Dict, Any, Optional
from rapidfuzz import process, fuzz
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MAPPING_CSV = os.path.join(BASE_DIR, "data", "mappings", "ticker_name_market.csv")
_map_df = pd.read_csv(MAPPING_CSV, dtype=str)

# ──────────────────────────────
# (복합) 조건형 쿼리 파싱 함수
def korean_num_to_int(s: str) -> int:
    s = s.replace(",", "").replace(" ", "")
    num_map = {'만':10000,'천':1000,'백':100,'십':10}
    res, num = 0, ""
    for ch in s:
        if ch.isdigit():
            num += ch
        elif ch in num_map:
            n = int(num) if num else 1
            res += n * num_map[ch]
            num = ""
    if num:
        res += int(num)
    return res

def parse_conditional_query(text: str) -> Optional[Dict[str, Any]]:
    date_match = re.search(r'(\d{4}-\d{2}-\d{2})', text)
    date = date_match.group(1) if date_match else None
    market = None
    if re.search(r'코스피|KOSPI', text): market="KOSPI"
    if re.search(r'코스닥|KOSDAQ', text): market="KOSDAQ"

    # 거래량 N일 평균 대비 M% 급증
    m_vol_ma = re.search(
        r'거래량[이]? (\d+)일 평균 대비 ([\d,]+)%\s*(이상|초과|이하|미만|급증)',
        text
    )
    if m_vol_ma:
        win, pct, comp = m_vol_ma.groups()
        return {"type":"vol_ma_spike","date":date,"window":int(win),
                "threshold":float(pct.replace(",","")),"comp":comp}

    # 종가 범위
    m_price = re.search(
        r'종가가 *([0-9,만천백십]+)[원]?\s*이상\s*([0-9,만천백십]+)[원]?\s*이하',
        text
    )
    if m_price:
        low, high = map(korean_num_to_int, m_price.groups())
        return {"type":"price_range","date":date,"low":low,"high":high}

    # RSI 조건
    m_rsi = re.search(r'RSI[가\s]*(\d+)\s*(이상|이하)', text, re.IGNORECASE)
    if m_rsi:
        th, comp = m_rsi.groups()
        return {"type":"rsi_condition","date":date,
                "threshold":float(th),"comp":comp}

    # 볼린저밴드 터치
    m_bb = re.search(r'볼린저[ -]?밴드 *(상단|하단)[에를\s]*터치', text)
    if m_bb:
        return {"type":"bollinger_touch","date":date,"band":m_bb.group(1)}

    # MA 돌파
    m_ma = re.search(
        r'종가[가\s]*(\d+)일 이동평균보다 *([+-]?\d+\.?\d*)% *(이상|초과|이하|미만)',
        text
    )
    if m_ma:
        win, diff, comp = m_ma.groups()
        return {"type":"ma_condition","date":date,
                "window":int(win),"diff":float(diff),"comp":comp}

    # 골든/데드 크로스
    m_cross = re.search(
        r'(\d{4}-\d{2}-\d{2})부터\s*(\d{4}-\d{2}-\d{2})까지\s*(골든크로스|데드크로스)',
        text
    )
    if m_cross:
        start, end, kind = m_cross.groups()
        return {"type":"cross_count","start":start,"end":end,"cross":kind}

    return None



# ──────────────────────────────
# 한글 숫자→정수 유틸 (price_range 등에서 사용)
# korean_num_to_int은 위에 이미 정의되어 있습니다.

# ──────────────────────────────
# ticker↔한글명 매핑 로딩
MAPPING_CSV = os.path.join(BASE_DIR, "data", "mappings", "ticker_name_market.csv")
_map_df    = pd.read_csv(MAPPING_CSV, dtype=str)
NAME_MAP   = _map_df.set_index("ticker")["name"].to_dict()
NAMES      = list(NAME_MAP.values())

def detect_name(q: str) -> Optional[str]:
    for nm in NAMES:
        if nm in q:
            return nm
    match, score, _ = process.extractOne(q, NAMES, scorer=fuzz.partial_ratio)
    return match if score >= 80 else None

# ──────────────────────────────
# 메인 파서
def parse_question(q: str) -> Dict[str, Any]:
    q = q.strip()

    # 0) 날짜 & 시장
    m = re.search(r'(\d{4}-\d{2}-\d{2})', q)
    date = m.group(1) if m else None
    market = None
    if re.search(r'코스피|KOSPI', q): market="KOSPI"
    if re.search(r'코스닥|KOSDAQ', q): market="KOSDAQ"

    # 0-1) 거래량 임계치 (예: 2025-03-10에 KOSDAQ 시장에서 거래량이 100만주 이상인 종목을 모두 보여줘)
    m_vol_thresh = re.search(
        r'거래량[이]? *(?:전날대비[\s\S]*?보다)?\s*([\d,]+)(만)?주\s*(이상|초과|이하|미만)',
        q
    )
    if m_vol_thresh:
        num, man, comp = m_vol_thresh.groups()
        # ‘만’ 처리
        mul = 10_000 if man == '만' else 1
        threshold = int(num.replace(',', '')) * mul
        return {
            "type":      "volume_threshold",
            "date":      date,
            "market":    market,
            "threshold": threshold,
            "comp":      comp
        }


    # 1) 복합조건
    cond = parse_conditional_query(q)
    if cond:
        return cond
    
    m_pct   = re.findall(
        r'등락률[이]? *([+-]?\d+\.?\d*)% *(이상|이하|초과|미만)', q)
    m_volchg= re.findall(
        r'거래량[이]? 전날대비 *([\d,]+)% *(이상|이하|초과|미만)', q)
    if m_pct and m_volchg:
        pct, pct_c = m_pct[0]
        vol, vol_c = m_volchg[0]
        return {
            "type":      "combined_pct_volume",
            "date":      date,
            "market":    market,
            "pct_sign":  float(pct),
            "pct_comp":  pct_c,
            "vol_pct":   float(vol.replace(",","")),
            "vol_comp":  vol_c,
        }

    # 3) 지수 & 거래종목수 & 거래대금
    if re.search(r'지수', q):
        return {"type":"market_stats","market":market,"date":date,"metric":"market_index"}
    if re.search(r'거래된 종목 수|거래종목수', q):
        return {"type":"market_stats","market":market,"date":date,"metric":"count"}
    if re.search(r'거래대금', q):
        return {"type":"market_stats","market":market,"date":date,"metric":"value"}

    # 4) 등락률 높은·낮은 TopN
    m_pct_top = re.search(r'(상승률 높은|하락률 높은).*?(상위|TOP)?\s*(\d+)', q)
    if m_pct_top:
        direction = "up" if "상승률" in m_pct_top.group(1) else "down"
        rank      = int(m_pct_top.group(3))
        return {
            "type":      "topn",
            "market":    market,
            "date":      date,
            "metric":    "pct_change",
            "rank":      rank,
            "direction": direction
        }
    # 1. 등락률 임계치 조건
    m_pct_only = re.search(
        r'등락률[이]? *([+-]?\d+\.?\d*)%?\s*(이상|이하|초과|미만|보다 큰|보다 작은)[^\d\w가-힣]*(?:인)?\s*(?:종목|기업)?(?:을|를)?(?:\s*모두)?\s*보여줘?', q)
    if m_pct_only:
        threshold, comp = m_pct_only.groups()[:2]
        # 한글 비교어 통일
        comp_map = {
            "보다 큰": "초과",
            "보다 작은": "미만"
        }
        comp = comp_map.get(comp, comp)
        return {
            "type": "pct_change_threshold",
            "date": date,
            "market": market,
            "threshold": float(threshold),
            "comp": comp
        }
    # 거래량 전날대비 N% 이상 증가(한) 종목
    m_vol_prev = re.search(
        r'거래량[이]?\s*(?:[a-zA-Z]*\s*시장에서)?전날대비\s*([\d,]+)%\s*(이상|이하|초과|미만)?\s*(증가|감소)?(?:한)?\s*(?:종목|기업)?(?:을|를)?(?:\s*모두)?\s*보여줘?', 
        q
    )
    if not m_vol_prev:
        m_vol_prev = re.search(
            r'거래량[이]?\s*(?:[a-zA-Z]*\s*시장에서)?전날대비\s*([\d,]+)%\s*(?:증가|감소)?(?:한)?\s*(?:종목|기업)?(?:을|를)?(?:\s*모두)?\s*보여줘?', 
            q
        )
    if m_vol_prev:
        threshold, comp, change_word = (m_vol_prev.groups() + (None,)*(3-len(m_vol_prev.groups())))
        comp = comp or "이상"
        return {
            "type": "volume_prevday_pct",
            "date": date,
            "market": market,
            "threshold": float(threshold.replace(",", "")),
            "comp": comp
        }

    # 5) 단순 Count by 상승/하락 종목 수
    # “거래된 종목 수” 또는 “거래종목수”
    if re.search(r'거래된\s*종목\s*수|거래종목수', q):
        return {"type":"market_stats", "market":market, "date":date, "metric":"count"}
    if re.search(r'상승.*종목 수|상승한 종목', q):
        return {"type":"count_by_pchange","market":market,"date":date,"direction":"up"}
    if re.search(r'하락.*종목 수|하락한 종목', q):
        return {"type":"count_by_pchange","market":market,"date":date,"direction":"down"}

    # 6) TopN (volume/value/close)
    m_exp = re.search(r'가장\s*비싼.*?(\d+)개', q)
    if m_exp:
        rank = int(m_exp.group(1))
        return {
            "type":   "topn",
            "market": market,
            "date":   date,
            "metric": "close",
            "rank":   rank
        }
    if re.search(r'가장\s*비싼.*종목', q) and not re.search(r'(\d+)개', q):
        return {
            "type":   "topn",
            "market": market,
            "date":   date,
            "metric": "close",
            "rank":   1
        }

    # 7) “가장 많은 거래량 N개”
    m_vol = re.search(r'거래량[이]?.*가장\s*많은.*?(\d+)개', q)
    if m_vol:
        rank = int(m_vol.group(1))
        return {
            "type":   "topn",
            "market": market,
            "date":   date,
            "metric": "volume",
            "rank":   rank
        }
    m = re.search(r'(상위|TOP)\s*(\d+)', q, re.I)

    if re.search(r'거래량[이]?.*가장\s*많은.*종목', q) and not re.search(r'(\d+)개', q):
        return {
            "type":   "topn",
            "market": market,
            "date":   date,
            "metric": "volume",
            "rank":   1
        }

    # “거래량 많은 종목 N개” (market+date+volume TopN)
    m_vol_top = re.search(r'거래량.*?(\d+)개', q)
    if m_vol_top:
        rank = int(m_vol_top.group(1))
        return {
            "type":   "topn",
            "market": market,
            "date":   date,
            "metric": "volume",
            "rank":   rank
        }
    if re.search(r'가장\s+.*\s+많은', q) and '거래량' in q:
        return {"type":"topn", "market":market, "date":date, "metric":"volume", "rank":1}
    if m:
        rank = int(m.group(2))
        if "거래량" in q:
            metric = "volume"
        elif "거래대금" in q:
            metric = "value"
        elif "상승률" in q or "하락률" in q:
            metric = "pct_change"
        else:
            metric = "adj_close"
        return {"type":"topn","market":market,"date":date,"metric":metric,"rank":rank}

    # 7) 단일 종목 질의
    name = detect_name(q)
    metric_map = {
        "시가":"open","고가":"high","저가":"low",
        "종가":"close","거래량":"volume",
        "등락률":"pct_change","RSI":"rsi_14"
    }
    metric = next((v for k,v in metric_map.items() if k in q), "adj_close")
    if name and date:
        return {"type":"single_metric","market":market,
                "name":name,"date":date,"metric":metric}

    return {"type":"unknown","question":q}
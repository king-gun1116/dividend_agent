# agent_a/tasks/task4_parser.py

import json
import re
import difflib
from typing import Dict, Any, Optional
from pathlib import Path
import pandas as pd
from agent_a.tasks.task4_handlers import task4_dispatch

HERE = Path(__file__).resolve()
Agent_A_ROOT = HERE.parent.parent

BASE_DIR    = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MAPPINGS_DIR = DATA_DIR     / "mappings"

ALIAS_PATH = DATA_DIR / "company_alias.json"
with open(ALIAS_PATH, encoding='utf-8') as f:
    ALIAS_MAP: Dict[str, str] = json.load(f)

MAP_CSV = MAPPINGS_DIR / "ticker_mapping_ks.csv"
_df    = pd.read_csv(MAP_CSV, encoding="utf-8-sig")
ticker_col = next(c for c in _df.columns if "ticker" in c.lower())
name_col   = next(c for c in _df.columns if c != ticker_col)
EXACT_MAP = dict(zip(_df['name'], _df['ticker']))

def alias_to_ticker(name: str) -> Optional[str]:
    canonical = ALIAS_MAP.get(name, name)
    if canonical in EXACT_MAP:
        return EXACT_MAP[canonical]
    match = difflib.get_close_matches(canonical, EXACT_MAP.keys(), n=1, cutoff=0.8)
    return EXACT_MAP[match[0]] if match else None

# 3) slang_map.json 로드
SLANG_PATH = DATA_DIR / "slang_map.json"
with open(SLANG_PATH, encoding="utf-8") as f:
    SLANG_MAP = json.load(f)

# --- 패턴 정의 (생략) ---
PAT_SINGLE = re.compile(r"(?P<name>[\w가-힣]+?)(?:의)?\s*(?:의)?\s*(?P<date>\d{4}-\d{2}-\d{2})\s*(?P<metric>시가|종가|고가|저가|거래량|close|open|high|low|volume)?[은는가]?\??")
PAT_TOP_N         = re.compile(r'수익률\s*상위\s*(\d+)개')
PAT_BOTTOM_N      = re.compile(r'수익률\s*최하위\s*(\d+)개')
PAT_PERCENT_UP    = re.compile(r'(\d+)%\s*이상\s*오른')
PAT_PERCENT_DOWN  = re.compile(r'(\d+)%\s*이상\s*하락')
PAT_CORRECT_PCT   = re.compile(r'(\d+)%\s*이상\s*조정받은')
PAT_PANIC_VOL     = re.compile(r'(\d+)%\s*\+?\s*거래량\s*(\d+)배')
PAT_RUSH          = re.compile(r'투매|패닉셀')
PAT_CANDLE_STREAK = re.compile(r'(\d+)일\s*연속\s*양봉')
PAT_RED_MARU      = re.compile(r'적삼병')
PAT_LIMITUP_SPACE = re.compile(r'(\d+)\s*거래일\s*연속\s*상한가')
PAT_LIMITUP_KO    = re.compile(r'상한가를\s*이틀\s*이상')
PAT_CONT_LIMITUP  = re.compile(r'연속\s*상한가')
PAT_VARIATION     = re.compile(r'등락폭\s*±?\s*(\d+)%')
PAT_VAR_INA       = re.compile(r'등락폭\s*(\d+)%\s*이내')
PAT_BOX           = re.compile(r'횡보|박스권|방향성\s*없는|머문')
PAT_HIGHDROP      = re.compile(r'고점\s*대비\s*(\d+)%\s*(?:이상\s*)?(?:하락|빠진)')
PAT_CORRECT_DOWN  = re.compile(r'조정장|조정받은')
PAT_GOLDEN        = re.compile(r'골든크로스|이평선.*교차')
PAT_DEAD          = re.compile(r'데드크로스')
PAT_BOTTOM_REV    = re.compile(r'바닥\s*찍고|저점.*?반등|최저가.*?반등|상승\s*전환')
PAT_DATE_RANGE    = re.compile(r'(\d{4}-\d{2}-\d{2})부터')
PAT_HIST_TOP      = re.compile(r'전체 기간.*?TOP\s*(\d+)')
PAT_HIST_EXT      = re.compile(r'역대급')
PAT_HIST_METRIC   = re.compile(
    r'역대\s*(?:수익률|return)\s*(?:상위|최고)?\s*(\d+)개',
    flags=re.IGNORECASE
)

# --- 유틸 함수 ---
def lookup_slang(text: str) -> Optional[Dict[str, Any]]:
    for token, info in SLANG_MAP.items():
        if token in text:
            return info.copy()
    return None

def extract_ticker(text: str) -> Optional[str]:
    print(f"\n[extract_ticker] 입력 text: {text}")
    # 1. 텍스트 전체가 company alias일 때 우선 체크
    tk_direct = alias_to_ticker(text)
    print(f"  alias_to_ticker({text}) => {tk_direct}")
    if tk_direct:
        print(f"  ⬆️ '{text}'는 바로 매핑됨 → {tk_direct}")
        return tk_direct

    # 2. 토큰 순회
    for token in re.findall(r'[\w가-힣]+', text):
        print(f"  TOKEN: {token}")
        if token.isdigit() and len(token) == 4:
            print(f"    ➡️ 연도(4자리) '{token}' 건너뜀")
            continue
        tk = alias_to_ticker(token)
        print(f"    alias_to_ticker({token}) => {tk}")
        if tk:
            print(f"    ⬆️ 매핑됨: {token} → {tk}")
            return tk
        else:
            print(f"    ✖️ 매핑실패: {token}")

    print("  ↪️ extract_ticker: 매칭 실패 (None)")
    return None

def parse_task4(text: str) -> Dict[str, Any]:
    t = text.strip().rstrip('?')
    extras: Dict[str, Any] = {}

    # 1) 괄호 내부
    for grp in re.findall(r'\(([^)]*)\)', t):
        c = grp.strip()
        if re.match(r'^\d+d$', c):
            extras['window'] = c
        elif re.match(r'^\d+일$', c):
            extras['window'] = c[:-1] + 'd'
        if '±' in c:
            extras['max_var'] = c
    t_clean = re.sub(r'\([^)]*\)', '', t)

    # 2) 슬랭 매핑 우선 (여기서 바로 처리!)
    params = lookup_slang(t_clean) or {}

    # ====== 여기 추가 ======
    if params:
        for k, v in extras.items():
            params.setdefault(k, v)
        return params
    # ======================

    # 3) 개별 룰
    if (m := PAT_SINGLE.search(t_clean)):
        print(f"PAT_SINGLE 매칭! name={m.group('name')}, date={m.group('date')}, metric={m.group('metric')}")
        name_clean = m.group('name').rstrip("의")
        return {
            'type':   'single_metric',
            'ticker': extract_ticker(m.group('name')),
            'date':   m.group('date'),
            'metric': m.group('metric') or "종가"
        }

    # 3.5) 역대 메트릭 상위/최고 N개 룰 (한글 수익률/영어 return 모두 지원)
    if not params and (m := PAT_HIST_METRIC.search(t_clean)):
        return {
            'type':   'historical_extreme',
            'metric': 'return',
            'rank':   f"top{m.group(1)}"
        }

    # 4) 기타 룰 매칭
    if not params:
        if m := PAT_TOP_N.search(t_clean):
            params = {'type':'recent_rally', 'threshold':int(m.group(1))}
        elif m := PAT_BOTTOM_N.search(t_clean):
            params = {'type':'recent_crash','threshold':int(m.group(1))}
        elif m := PAT_PERCENT_UP.search(t_clean):
            params = {'type':'recent_rally','min_return':f'{m.group(1)}%'}
        elif m := PAT_PERCENT_DOWN.search(t_clean):
            params = {'type':'recent_crash','max_loss':f'{m.group(1)}%'}
        elif m := PAT_CORRECT_PCT.search(t_clean):
            params = {'type':'market_correction','drop_pct':f'{m.group(1)}%'}
        elif m := PAT_PANIC_VOL.search(t_clean):
            params = {'type':'panic_sell','max_loss':f'{m.group(1)}%','vol_spike':int(m.group(2))}
        elif PAT_RUSH.search(t_clean):
            params = {'type':'panic_sell','max_loss':'8%','vol_spike':2}
        elif m := PAT_CANDLE_STREAK.search(t_clean):
            params = {'type':'candlestick_pattern','pattern':'three_white_soldiers'}
            extras['days'] = int(m.group(1))
        elif PAT_RED_MARU.search(t_clean):
            params = {'type':'candlestick_pattern','pattern':'three_white_soldiers'}
        elif m := PAT_LIMITUP_SPACE.search(t_clean):
            params = {'type':'limit_up_streak','days':int(m.group(1))}
        elif PAT_LIMITUP_KO.search(t_clean):
            params = {'type':'limit_up_streak','days':2}
        elif PAT_CONT_LIMITUP.search(t_clean):
            params = {'type':'limit_up_streak'}
        elif m := PAT_VARIATION.search(t_clean):
            params = {'type':'sideways_market','max_var':f'±{m.group(1)}%'}
        elif m := PAT_VAR_INA.search(t_clean) and PAT_BOX.search(t_clean):
            params = {'type':'sideways_market','max_var':f'±{m.group(1)}%'}
        elif PAT_BOX.search(t_clean):
            params = {'type':'sideways_market'}
        elif PAT_CORRECT_DOWN.search(t_clean):
            pct = PAT_PERCENT_DOWN.search(t_clean).group(1) if PAT_PERCENT_DOWN.search(t_clean) else '10'
            params = {'type':'market_correction','drop_pct':f'{pct}%'}
        elif m := PAT_HIGHDROP.search(t_clean):
            params = {'type':'market_correction','drop_pct':f'{m.group(1)}%'}
        elif PAT_GOLDEN.search(t_clean):
            params = {'type':'technical_pattern','pattern':'golden_cross'}
        elif PAT_DEAD.search(t_clean):
            params = {'type':'technical_pattern','pattern':'death_cross'}
        elif PAT_BOTTOM_REV.search(t_clean):
            params = {'type':'bottom_reversal'}
        elif m := PAT_DATE_RANGE.search(t_clean):
            params = {'type':'historical_extreme','start':m.group(1),'rank':'top1','metric':'return'}
        elif m := PAT_HIST_TOP.search(t_clean):
            params = {'type':'historical_extreme','rank':f'top{m.group(1)}','metric':'return'}
        elif PAT_HIST_EXT.search(t_clean):
            params = {'type':'historical_extreme','rank':'top1','metric':'return'}
        else:
            return {'type':'unknown','text':text,'ticker':extract_ticker(t)}

    # 5) extras 병합
    for k, v in extras.items():
        params.setdefault(k, v)


    # 6) threshold 문자열 처리 (e.g. "top10" → 10)
    if 'threshold' in params and isinstance(params['threshold'], str):
        m = re.search(r'(\d+)', params['threshold'])
        if m:
           params['threshold'] = int(m.group(1))
        else:
           params.pop('threshold')

    # 6–7) ticker 보강
    if params.get('type') in ('candlestick_pattern','technical_pattern') and 'ticker' not in params:
        params['ticker'] = extract_ticker(t)
    if params.get('type') != 'unknown' and 'ticker' not in params:
        params['ticker'] = extract_ticker(t)

    return params

def handle_task4(text: str) -> Any:
    params = parse_task4(text)
    if params.get('type') == 'unknown':
        return f"죄송해요, '{text}' 질문을 이해하지 못했습니다."
    return task4_dispatch(params)

def llm_fallback(question, get_history, get_all_tickers):
    """
    model.py에서 import하여 사용하는 LLM fallback
    """
    return handle_task4(question)

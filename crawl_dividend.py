#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
crawl_dividend.py
— 증분 수집 후 data/crawl_dividend.csv 에 덮어쓰기
"""

import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
import xmltodict
from tqdm import tqdm

# =========================
# 프로젝트 경로 설정
# =========================
BASE_DIR   = Path(__file__).resolve().parent
DATA_DIR   = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH   = DATA_DIR / "crawl_dividend.csv"
DEFAULT_START = "20130101"   # CSV가 없을 때 시작일

# =========================
# 요청 설정
# =========================
URL = "https://seibro.or.kr/websquare/engine/proworks/callServletService.jsp"
HEADERS = {
    "Accept": "application/xml",
    "Content-Type": 'application/xml; charset="UTF-8"',
    "Origin": "https://seibro.or.kr",
    "Referer": "https://seibro.or.kr/websquare/control.jsp?w2xPath=/IPORTAL/user/company/BIP_CNTS01041V.xml&menuNo=285",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "submissionid": "submission_divStatInfoPList",
}
MARKET_FILTER = ["유가증권시장", "코스닥시장"]

# =========================
# 컬럼 매핑 / 드랍 설정
# =========================
DROP_COLS = [
    "주식 유통(교부)일",
    "현물배당 지급일",
    "주당배당금_차등",
    "주당배당률(일반)_주식",
    "주당배당률(차등)_현금",
    "주당배당률(차등)_주식",
    "주식배당률_일반",
    "주식배당률_차등",
]

COL_MAP = {
    "RGT_STD_DT": "배정기준일",
    "TH1_PAY_TERM_BEGIN_DT": "현금배당 지급일",
    "SHOTN_ISIN": "종목코드",
    "KOR_SECN_NM": "종목명",
    "LIST_TPNM": "시장구분",
    "RGT_RSN_DTAIL_SORT_NM": "배당구분",
    "SECN_DTAIL_KANM": "주식종류",
    "CASH_ALOC_AMT": "주당배당금_일반",
    "CASH_ALOC_RATIO": "주당배당률(일반)_현금",
    "ESTM_STDPRC": "단주기준가",
    "PVAL": "액면가",
    "SETACC_MM": "결산월",
    "AG_ORG_TPNM": "명의개서대리인",
}
KEEP_COLS = [c for c in COL_MAP.values() if c not in DROP_COLS]

# =========================
# 유틸 함수
# =========================
def ext_d(node: dict) -> dict:
    return {k: v["@value"] for k, v in node.items()}

def to_dt(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series.str.replace(r"\D", "", regex=True),
                          format="%Y%m%d", errors="coerce")

def year_ranges(since: str, until: str):
    s = datetime.strptime(since, "%Y%m%d")
    e = datetime.strptime(until, "%Y%m%d")
    cur = s
    while cur <= e:
        end_y = datetime(cur.year, 12, 31)
        if end_y > e:
            end_y = e
        yield cur.strftime("%Y%m%d"), end_y.strftime("%Y%m%d")
        cur = end_y + timedelta(days=1)

def safe_to_csv(df: pd.DataFrame, path: Path, **kwargs):
    try:
        df.to_csv(path, **kwargs)
        print(f"✅ 저장 완료: {path}")
    except PermissionError:
        ts = int(time.time())
        alt = path.with_stem(path.stem + f"_{ts}")
        df.to_csv(alt, **kwargs)
        print(f"⚠️ '{path.name}' 열림 → 임시저장: {alt.name}")

# =========================
# Fetch 한 구간
# =========================
def fetch_range(since: str, until: str) -> pd.DataFrame:
    rows = []
    with requests.Session() as sess:
        sess.headers.update(HEADERS)
        for start in tqdm(range(1, 10001, 15), desc=f"{since[:4]} {since}~{until}"):
            end = start + 14
            payload = f"""
<reqParam action="divStatInfoPList" task="ksd.safe.bip.cnts.Company.process.EntrFnafInfoPTask">
    <RGT_STD_DT_FROM value="{since}"/>
    <RGT_STD_DT_TO   value="{until}"/>
    <START_PAGE value="{start}"/>
    <END_PAGE   value="{end}"/>
    <MENU_NO    value="285"/>
</reqParam>""".strip()

            try:
                r = sess.post(URL, data=payload.encode("utf-8"), timeout=20)
                r.raise_for_status()
            except requests.RequestException as e:
                print(f"[ERROR] {start}-{end}: {e}")
                time.sleep(2)
                continue

            parsed = xmltodict.parse(r.text)
            cnt = parsed["vector"]["@result"]
            if cnt == "0":
                break

            data_nodes = parsed["vector"]["data"]
            if cnt == "1":
                data_nodes = [ext_d(data_nodes["result"])]
            else:
                data_nodes = [ext_d(n["result"]) for n in data_nodes]

            rows.extend(data_nodes)
            time.sleep(0.4)

    if not rows:
        return pd.DataFrame(columns=KEEP_COLS)

    df = pd.DataFrame(rows).rename(columns=COL_MAP)
    return df.reindex(columns=KEEP_COLS)

# =========================
# 메인 로직
# =========================
def main():
    today = datetime.now().strftime("%Y%m%d")

    if CSV_PATH.exists():
        exist_df = pd.read_csv(CSV_PATH, dtype=str)
        last_dt = to_dt(exist_df["배정기준일"]).max()
        since = (last_dt + timedelta(days=1)).strftime("%Y%m%d") if pd.notna(last_dt) else DEFAULT_START
    else:
        exist_df = pd.DataFrame(columns=KEEP_COLS)
        since = DEFAULT_START

    until = today
    print(f"[DEBUG] since={since} | until={until}")

    if since > until:
        print("이미 최신입니다.")
        return

    add_frames = []
    for s, u in year_ranges(since, until):
        part = fetch_range(s, u)
        if part.empty:
            continue
        part = part[part["시장구분"].isin(MARKET_FILTER)]
        add_frames.append(part)

    if not add_frames:
        print("신규 데이터 없음.")
        return

    new_df = pd.concat(add_frames, ignore_index=True)
    merged = (
        pd.concat([exist_df, new_df], ignore_index=True)
          .drop_duplicates(subset=["배정기준일", "종목코드"], keep="last")
          .sort_values("배정기준일")
          .reset_index(drop=True)
    )

    safe_to_csv(merged, CSV_PATH, index=False, encoding="utf-8-sig")
    print(f"신규 {len(new_df)}건 / 총 {len(merged)}건")

if __name__ == "__main__":
    main()
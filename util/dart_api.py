"""
utils/dart_api.py (rev 2025-07-05b)
──────────────────────────────────────────────────────────────────────────────
배당 공시 수집/전처리 + 증분 업데이트 + 임베딩용 JSONL + CSV 저장

갱신 내역 (2025-07-05b)
──────────────────────────────────────────────────────────────────────────────
1. corp_code.xml 응답 형식 다변화 대응 (<list> → <item> 등)
2. 응답이 HTML 오류일 경우도 감지
3. status 태그가 없을 때도 오류 없이 처리
"""

from __future__ import annotations

import os
import json
import time
import zipfile
import warnings
import csv 
from bs4 import XMLParsedAsHTMLWarning
warnings.filterwarnings("ignore", r"\[Selenium\] fallback for")
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import chardet
import pandas as pd
import requests
import xmltodict
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from requests.exceptions import ReadTimeout
from urllib3.util.retry import Retry
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ────────────────────────────────────────────────────────────
# 환경 설정 & 세션
# ────────────────────────────────────────────────────────────
load_dotenv()
API_KEY: str | None = os.getenv("DART_API_KEY")
if not API_KEY:
    raise RuntimeError("❌ 환경변수 DART_API_KEY 를 설정하세요")

HEADERS = {"User-Agent": "Mozilla/5.0"}
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

session = requests.Session()
retry_strategy = Retry(
    total=5,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET", "POST"],
)
session.mount("https://", HTTPAdapter(max_retries=retry_strategy))
session.mount("http://", HTTPAdapter(max_retries=retry_strategy))

# ────────────────────────────────────────────────────────────
# corp_code.xml 로드 (30일 캐시)
# ────────────────────────────────────────────────────────────
_CORP_XML_PATH = os.path.join(DATA_DIR, "corp_code.xml")


def _download_corp_code(force_refresh: bool = False, max_age_days: int = 30) -> None:
    if not force_refresh and os.path.exists(_CORP_XML_PATH):
        mtime = datetime.fromtimestamp(os.path.getmtime(_CORP_XML_PATH))
        if datetime.now() - mtime < timedelta(days=max_age_days):
            return

    print("⏳ [corp_code] 다운로드 중…", flush=True)
    url = f"https://opendart.fss.or.kr/api/corpCode.xml?crtfc_key={API_KEY}"
    resp = session.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()

    content = resp.content
    if content.lstrip().startswith(b"<?xml"):
        with open(_CORP_XML_PATH, "wb") as f:
            f.write(content)
    else:
        zip_path = os.path.join(DATA_DIR, "corp_code.zip")
        with open(zip_path, "wb") as f:
            f.write(content)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(DATA_DIR)
        os.replace(os.path.join(DATA_DIR, "CORPCODE.xml"), _CORP_XML_PATH)
        os.remove(zip_path)
    print("✅ [corp_code] 최신 파일 저장 완료", flush=True)


def load_corps(force_refresh: bool = False) -> pd.DataFrame:
    _download_corp_code(force_refresh=force_refresh)

    raw = open(_CORP_XML_PATH, "rb").read()
    enc = chardet.detect(raw)["encoding"]
    try:
        doc = xmltodict.parse(raw.decode(enc, errors="ignore"))
    except Exception as e:
        snippet = raw[:400].decode(enc, errors="ignore")
        raise RuntimeError(f"corpCode XML 파싱 실패: {e}\n응답 앞부분: {snippet}")

    if "result" in doc:
        result = doc["result"]
        status = result.get("status")
        if status and status != "000":
            raise RuntimeError(f"corpCode API 오류: {status} {result.get('message')}")

        items = result.get("list", [])
        if isinstance(items, dict) and "item" in items:
            items = items["item"]
    else:
        items = doc.get("CORPCODE", {}).get("list", {}).get("item", [])

    if isinstance(items, dict):
        items = [items]

    df = pd.DataFrame(items)
    df["stock_code"] = df["stock_code"].astype(str).str.strip()
    df = df[df["stock_code"].str.len() == 6].reset_index(drop=True)
    df = df[["corp_code", "corp_name", "stock_code"]]

    print(f"✅ [corp_code] 종목 파싱 완료 → {len(df):,}개", flush=True)
    return df

# ────────────────────────────────────────────────────────────
# 공시 목록(list.json) 조회
# ────────────────────────────────────────────────────────────

def list_filings(corp_code: str, bgn: str, end: str, max_pages: int = 10) -> List[dict]:
    """기업 단위 list.json 페이징 조회 (page_count=100)"""
    results: List[dict] = []
    for page in range(1, max_pages + 1):
        url = (
            f"https://opendart.fss.or.kr/api/list.json?crtfc_key={API_KEY}"
            f"&corp_code={corp_code}&bgn_de={bgn}&end_de={end}&page_count=100&page_no={page}"
        )
        try:
            r = session.get(url, headers=HEADERS, timeout=15)
            r.raise_for_status()
            page_items = r.json().get("list", [])
        except Exception:
            break
        if not page_items:
            break
        results.extend(page_items)
        if len(page_items) < 100:
            break  # 마지막 페이지
        time.sleep(0.1)
    return results

# ────────────────────────────────────────────────────────────
# 보고서 본문 가져오기 (document.xml 우선)
# ────────────────────────────────────────────────────────────

def fetch_report_html(rcept_no: str) -> str:
    """document.xml 로 HTML 획득 (Selenium Fallback)"""
    # 1) document.xml API (대부분 배당 보고서 포함)
    url = (
        f"https://opendart.fss.or.kr/api/document.xml?crtfc_key={API_KEY}&rcept_no={rcept_no}"
    )
    try:
        resp = session.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        # API 성공 but status code 내부 JSON이 아닐 때 → XML 문자열 반환
        if resp.content.startswith(b"<?xml"):
            return resp.text
        # 일부 케이스는 JSON {status,message} 반환
    except Exception:
        pass  # fallback

    # 2) 정적 HTML (JS 미포함) – 속도 빠름
    static_url = (
        f"https://dart.fss.or.kr/report/viewer.do?rcpNo={rcept_no}&dcmNo=0&eleId=0"  # dcmNo=0=최신
    )
    try:
        r = session.get(static_url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        if "<html" in r.text.lower():
            return r.text
    except Exception:
        pass

    # 3) Selenium 최후 수단 (드라이버 매번 기동 ×, 경고)
    warnings.warn(f"[Selenium] fallback for {rcept_no}")
    try:
        from selenium import webdriver 
        from selenium.webdriver.chrome.service import Service
        from selenium.webdriver.chrome.options import Options
    except ImportError:
        warnings.warn(f"[Selenium] 설치된 webdriver_manager 없음 → Selenium fallback 스킵 (rcept_no={rcept_no})")
        return ""  # 또는 적절한 빈 html 처리

    options = Options()
    options.add_argument('--headless=new')
    options.add_argument('--no-sandbox')
    options.add_argument('--remote-allow-origins=*')
    service = Service('/opt/homebrew/bin/chromedriver')
    driver = webdriver.Chrome(service=service, options=options)
    driver.get('https://example.com')
    print("Title:", driver.title)
    driver.quit()
    try:
        driver.get(f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={rcept_no}")
        time.sleep(1.2)
        driver.switch_to.frame("ifrm"); time.sleep(0.8)
        html = driver.page_source
    finally:
        driver.quit()
    return html

# ────────────────────────────────────────────────────────────
# 배당 테이블 파싱
# ────────────────────────────────────────────────────────────
_DIV_KEYS = [
    "div_type", "div_kind",
    "per_share_common", "per_share_preferred",
    "yield_common", "yield_preferred",
    "total_amount", "record_date", "payment_date",
    "meeting_held", "meeting_date", "board_decision_date",
]

def parse_dividend_info(html: str) -> Dict[str, str]:
    """XFormD 테이블에서 핵심 배당 정보를 dict 로 추출"""
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", id=lambda x: x and x.startswith("XFormD"))
    info = {k: "-" for k in _DIV_KEYS}
    if not table:
        return info

    for tr in table.find_all("tr"):
        tds = [td.get_text(strip=True) for td in tr.find_all("td")]
        if not tds:
            continue
        h = tds[0]
        if h.startswith("1."):
            info["div_type"] = tds[-1] or "-"
        elif h.startswith("2."):
            info["div_kind"] = tds[-1] or "-"
        elif h.startswith("3."):
            commons = [tds[i] for i in range(len(tds)) if "보통" in tds[i - 1]]
            prefs = [tds[i] for i in range(len(tds)) if "종류" in tds[i - 1]]
            info["per_share_common"] = commons[0] if commons else "-"
            info["per_share_preferred"] = prefs[0] if prefs else "-"
        elif h.startswith("4."):
            commons = [tds[i] for i in range(len(tds)) if "보통" in tds[i - 1]]
            prefs = [tds[i] for i in range(len(tds)) if "종류" in tds[i - 1]]
            info["yield_common"] = commons[0] if commons else "-"
            info["yield_preferred"] = prefs[0] if prefs else "-"
        elif h.startswith("5."):
            info["total_amount"] = tds[-1] or "-"
        elif h.startswith("6."):
            info["record_date"] = tds[-1] or "-"
        elif h.startswith("7."):
            info["payment_date"] = tds[-1] or "-"
        elif h.startswith("8."):
            info["meeting_held"] = tds[-1] or "-"
        elif h.startswith("9."):
            info["meeting_date"] = tds[-1] or "-"
        elif h.startswith("10."):
            info["board_decision_date"] = tds[-1] or "-"
    return info

# ────────────────────────────────────────────────────────────
# 병렬 수집 + 증분 로직
# ────────────────────────────────────────────────────────────

_LAST_SEEN_PATH = os.path.join(DATA_DIR, "last_seen.json")

def _load_last_seen() -> Dict[str, str]:
    if os.path.exists(_LAST_SEEN_PATH):
        return json.load(open(_LAST_SEEN_PATH))
    return {}

def _save_last_seen(d: Dict[str, str]):
    with open(_LAST_SEEN_PATH, "w", encoding="utf-8") as fw:
        json.dump(d, fw, ensure_ascii=False, indent=2)


def collect_dividend_filings_incremental(
    existing_jsonl: Optional[str] = None,
    start: str = "20130101",
    end: str   = datetime.now().strftime("%Y%m%d"),
    save_csv: Optional[str] = None,
    save_jsonl: Optional[str] = None,
    max_workers: int = 10,
) -> List[dict]:
    """기존 JSONL을 참고하여 *신규* 배당 공시만 수집"""
    # ── 1) 이미 수집된 rcept_no 로드
    seen: set[str] = set()
    if existing_jsonl and os.path.exists(existing_jsonl):
        with open(existing_jsonl, "r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                seen.add(rec["rcept_no"])

    # ── 2) last_seen(기업별 마지막 조회일) 로드
    last_seen = _load_last_seen()

    # ── 3) 전체 기업 목록
    corps = load_corps()

    # ── 4) list.json 조회 & 신규 task 생성
    tasks: List[dict] = []
    for _, row in tqdm(corps.iterrows(), total=len(corps), desc="리스트 조회"):
        corp_start = last_seen.get(row.corp_code, start)
        pages = list_filings(row.corp_code, bgn=corp_start, end=end)
        if not pages:
            continue
        # 해당 기업은 이번에 end 날짜까지 조회했으므로 업데이트
        last_seen[row.corp_code] = end
        for f in pages:
            if "배당" not in f["report_nm"]:
                continue
            if f["rcept_no"] in seen:
                continue
            tasks.append({
                "corp_name": row.corp_name,
                "stock_code": row.stock_code,
                "rcept_dt": f["rcept_dt"],
                "report_nm": f["report_nm"],
                "rcept_no": f["rcept_no"],
            })

    print(
        f"▶ 신규 배당 공시: {len(tasks):,}건 (기존 {len(seen):,}건 제외) → 병렬 수집", flush=True
    )

    # ── 5) HTML 병렬 수집 & 파싱
    results: List[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(fetch_report_html, t["rcept_no"]): t for t in tasks}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="본문 수집"):
            meta = futures[fut]
            html = fut.result()
            parsed = parse_dividend_info(html)
            results.append({**meta, "html": html, **parsed})

    # ── 6) JSONL 파일 저장 (경로가 유효할 때만)
    jsonl_path = save_jsonl if save_jsonl else existing_jsonl
    if jsonl_path:
        os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
        with open(jsonl_path, "a", encoding="utf-8") as fw:
            for rec in results:
                fw.write(json.dumps(rec, ensure_ascii=False) + "\n")
        # 기존 JSONL 라인 수 집계
        total_j = 0
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    total_j += 1
        print(f"✅ JSONL 저장: {jsonl_path} (총 {total_j}건)")
    else:
        print("ℹ️ JSONL 경로가 지정되지 않아 저장을 생략합니다.", flush=True)

    if save_csv:
        # 결과가 없으면 저장 건수만 리포트하고 스킵
        if not results:
            try:
                total_c = len(pd.read_csv(save_csv, encoding="utf-8-sig", engine='python',on_bad_lines='skip'))
            except FileNotFoundError:
                total_c = 0
            except pd.errors.EmptyDataError:
                total_c = 0
            print(f"✅ CSV 저장: {save_csv} (총 {total_c}건)")
        else:
            df_new = pd.DataFrame(results)
            # 빈 CSV 처리: 파일이 없거나 비어 있으면 빈 DataFrame
            if os.path.exists(save_csv):
                try:
                    df_old = pd.read_csv(save_csv, encoding="utf-8-sig")
                except pd.errors.EmptyDataError:
                    df_old = pd.DataFrame()
            else:
                df_old = pd.DataFrame()
            df_all = pd.concat([df_old, df_new], ignore_index=True)
            df_all.to_csv(save_csv, index=False, encoding="utf-8-sig")
            # 갱신 후 현재 총 건수
            try:
                total_c = len(pd.read_csv(save_csv, encoding="utf-8-sig"))
            except pd.errors.EmptyDataError:
                total_c = 0
            print(f"✅ CSV 저장: {save_csv} (총 {total_c}건)")

    # ── 7) last_seen 업데이트
    _save_last_seen(last_seen)

    return results

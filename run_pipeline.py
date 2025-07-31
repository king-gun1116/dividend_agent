#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full pipeline (최종 자동화 버전)

- DataFrame을 or 연산에 쓰지 않도록 ValueError 패치 (fetch_one_code)
- HTML 비어있는 레코드 채우기 함수 fill_missing_html 추가
- 기타 로깅/안정화 유지
- 가격 수집·병합 단계를 건너뛰는 --skip-price-fetch 옵션 추가
- 기존 sync 단계 및 skip-sync 옵션 제거
- existing/have 로직 제거
- 중복된 fetch 함수 정리
- DIVIDEND_CSV 대신 csv_path 일관 사용
- HTML/Text fetch 실패 시 “본문 없음” 로그 강화
- regression 모듈에 ret_1d~ret_30d 모두 생성
- Master CSV 병합 시 p_up 컬럼 유지
"""

from __future__ import annotations
import warnings
from urllib3.exceptions import NotOpenSSLWarning
warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated as an API.*")

import os, sys, io, shutil, logging
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm.auto import tqdm
import joblib

# price libs
import FinanceDataReader as fdr
import yfinance as yf
from cache_krx import stock as pykrx_stock
from typing import List

# ML libs
import lightgbm as lgb

# project utils
from util.dart_api import collect_dividend_filings_incremental
from util.data_cleaning import clean_ml_data
from util.feature_engineering import generate_classification_features
from util.price_fetcher import fetch_price_series
from util.model_utils import train_or_load_classifier, train_or_load_regressor

# suppress FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------- PATHS ----------------
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data")
RAW_CSV         = os.path.join(DATA_DIR, "dividend_final_for_ml.csv")
ML_READY_BASE   = os.path.join(DATA_DIR, "dividend_ml_ready.csv")
ML_READY_NOZERO = os.path.join(DATA_DIR, "dividend_ml_ready_nozero.csv")
HIST_PATH       = os.path.join(DATA_DIR, "price_history.csv")
CACHE_DIR       = os.path.join(DATA_DIR, "price_cache")
MODULE_DIR      = os.path.join(DATA_DIR, "module_datasets")
ARTIFACTS_DIR   = os.path.join(BASE_DIR, "artifacts")
RESULT_REG_DIR  = os.path.join(DATA_DIR, "results", "regression")
MODEL_DIR       = os.path.join(DATA_DIR, "models")
MASTER_CSV      = os.path.join(DATA_DIR, "all_stocks_master.csv")
REG_PRED_CSV    = os.path.join(RESULT_REG_DIR, "regression_predictions_for_ensemble.csv")
CLF_MODEL_FP    = os.path.join(MODEL_DIR, "lgbm_classifier.pkl")
REG_MODEL_FP    = os.path.join(MODEL_DIR, "lgbm_regressor.pkl")

for d in (DATA_DIR, CACHE_DIR, MODULE_DIR, RESULT_REG_DIR, MODEL_DIR, ARTIFACTS_DIR):
    os.makedirs(d, exist_ok=True)

# --------------- LOGGING ---------------
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(DATA_DIR, "pipeline.log"), encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("pykrx").setLevel(logging.CRITICAL)
logging.getLogger("lightgbm").setLevel(logging.WARNING)

# ---------- helpers ----------
def _init_http_session(retries=5, backoff_factor=1.0, pool_maxsize=10):
    r = Retry(total=retries, backoff_factor=backoff_factor,
              status_forcelist=[429,500,502,503,504], respect_retry_after_header=True,
              allowed_methods=["GET","POST"])
    s = requests.Session()
    ad = HTTPAdapter(max_retries=r, pool_maxsize=pool_maxsize)
    s.mount("https://", ad); s.mount("http://", ad)
    return s

def _silence(fn, *a, **k):
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = io.StringIO()
    try: return fn(*a, **k)
    finally: sys.stdout, sys.stderr = old_out, old_err

def _chk(df: pd.DataFrame, code: str, name: str):
    if "stock_code" not in df.columns:
        logging.info(f"[CHK] {name} | no stock_code col")
        return
    cnt = int((df["stock_code"].astype(str) == code).sum())
    logging.info(f"[CHK] {name} | {code} count = {cnt} / rows={len(df)}")


# ---------- price fallbacks ----------
def yf_download(code: str, start: str, end: str) -> pd.DataFrame | None:
    for t in (f"{code}.KS", f"{code}.KQ", code):
        try:
            df = _silence(yf.download, t, start=start, end=end,
                          progress=False, threads=False, auto_adjust=False)
            if isinstance(df, pd.DataFrame) and not df.empty:
                out = df.reset_index()[["Date","Close","Volume"]]
                out.columns = ["date","close","volume"]
                return out
        except Exception:
            pass
    return None

def pykrx_download(code: str, start: str, end: str) -> pd.DataFrame | None:
    try:
        s = pd.to_datetime(start).strftime("%Y%m%d")
        e = pd.to_datetime(end).strftime("%Y%m%d")
        df = _silence(pykrx_stock.get_market_ohlcv_by_date, s, e, code)
        if df is None or df.empty: return None
        out = df.reset_index()[["날짜","종가","거래량"]]
        out.columns = ["date","close","volume"]
        return out
    except Exception:
        return None

def fdr_download(code: str, start: str, end: str) -> pd.DataFrame | None:
    try:
        df = _silence(fdr.DataReader, code, start, end)
        if df is None or df.empty:
            df = _silence(fdr.DataReader, f"KRX:{code}", start, end)
        if df is None or df.empty: return None
        df = df.reset_index()
        lower = {c.lower(): c for c in df.columns}
        dc = lower.get("date") or lower.get("날짜")
        cc = lower.get("close") or lower.get("종가")
        vc = lower.get("volume") or lower.get("거래량")
        if not all((dc,cc,vc)): return None
        out = df[[dc,cc,vc]].copy(); out.columns = ["date","close","volume"]
        return out
    except Exception:
        return None

def _first_ok(*dfs):
    for d in dfs:
        if isinstance(d, pd.DataFrame) and not d.empty:
            return d
    return None

def fetch_one_code(code: str, start: str, end: str,
                   chunk_days=365, max_retry=3, backoff=2.5) -> pd.DataFrame | None:
    cache_fp = os.path.join(CACHE_DIR, f"{code}.csv")
    def cache_ok(fp: str) -> bool:
        if not os.path.exists(fp): return False
        try:
            d = pd.read_csv(fp, parse_dates=["date"])
            return d["date"].min() <= pd.to_datetime(start) and d["date"].max() >= pd.to_datetime(end)
        except Exception:
            return False
    if cache_ok(cache_fp):
        return pd.read_csv(cache_fp, parse_dates=["date"])
    cuts = pd.date_range(start, end, freq=f"{chunk_days}D").tolist() + [pd.to_datetime(end)+pd.Timedelta(days=1)]
    frames: List[pd.DataFrame] = []
    for s, e in zip(cuts[:-1], cuts[1:]):
        seg_s, seg_e = s.strftime("%Y-%m-%d"), (e - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        delay = 1.0
        for attempt in range(max_retry):
            df = _first_ok(
                yf_download(code, seg_s, seg_e),
                fdr_download(code, seg_s, seg_e),
                pykrx_download(code, seg_s, seg_e),
            )
            if df is not None:
                frames.append(df)
                break
            time.sleep(delay); delay *= backoff
        else:
            logging.warning("PRICE FAIL %s seg %s~%s", code, seg_s, seg_e)
            return None
    out = pd.concat(frames, ignore_index=True).drop_duplicates("date").sort_values("date")
    out["stock_code"] = code
    out.to_csv(cache_fp, index=False)
    return out

def _load_cache(fp: str, code: str) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(fp, dtype=str)
        cols = {c.lower(): c for c in df.columns}
        date_c = cols.get("date") or next((c for c in df.columns if "날짜" in c), None)
        close_c= cols.get("close") or next((c for c in df.columns if "종가" in c), None)
        vol_c  = cols.get("volume") or next((c for c in df.columns if "거래" in c), None)
        if not all((date_c, close_c, vol_c)):
            df = pd.read_csv(fp, header=None, names=["date","close","volume"], dtype=str)
            date_c, close_c, vol_c = "date","close","volume"
        df = df[[date_c, close_c, vol_c]].copy()
        df.columns = ["date","close","volume"]
        df["date"] = pd.to_datetime(df["date"].str[:10], errors="coerce")
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(int)
        df = df.dropna(subset=["date"]).sort_values("date")
        df["stock_code"] = code
        return df
    except Exception:
        logging.exception("cache load fail: %s", fp)
        return None

def _is_newer(src_paths: list, target_path: str) -> bool:
    if not os.path.exists(target_path): return True
    t = os.path.getmtime(target_path)
    return any(os.path.getmtime(p) > t for p in src_paths if os.path.exists(p))

# ------------- Master CSV builder -------------
def _build_master_csv(module_dir: str, data_dir: str, master_csv_path: str,
                      n_clusters: int = 4, drop_text_emb: bool = True,
                      drop_helper_cols: bool = True) -> pd.DataFrame:
    logging.info("▶ [Master] merge preds / probs / residual clusters")

    # 1) regression 모듈 불러오기
    df_reg  = pd.read_csv(os.path.join(module_dir, "regression.csv"),
                          parse_dates=["rcept_dt"], dtype={"stock_code": str})
    # 2) 기존 예측 결과 불러오기
    df_pred = pd.read_csv(REG_PRED_CSV,
                          parse_dates=["rcept_dt"], dtype={"stock_code": str})

    pup_fp = os.path.join(data_dir, "p_up_temp.csv")
    if os.path.exists(pup_fp):
        df_pup = pd.read_csv(pup_fp, parse_dates=["rcept_dt"], dtype={"stock_code": str})
        # zfill + normalize
        df_pred["stock_code"] = df_pred["stock_code"].str.zfill(6)
        df_pred["rcept_dt"]   = pd.to_datetime(df_pred["rcept_dt"]).dt.normalize()
        df_pup["stock_code"]  = df_pup["stock_code"].str.zfill(6)
        df_pup["rcept_dt"]    = pd.to_datetime(df_pup["rcept_dt"]).dt.normalize()
        # 병합
        df_pred = df_pred.merge(
            df_pup[["stock_code","rcept_dt","p_up"]],
            on=["stock_code","rcept_dt"], how="left"
        )

    # 4) classifier 피처 불러오기
    feat_pq = os.path.join(data_dir, "df_feat.parquet")
    df_clf  = pd.read_parquet(feat_pq)

    # 5) 중복 키 제거 (1:1 매핑 보장)
    for df in (df_reg, df_pred, df_clf):
        df["stock_code"] = df["stock_code"].astype(str).str.zfill(6)
        df["rcept_dt"] = pd.to_datetime(df["rcept_dt"]).dt.normalize()
        df.drop_duplicates(subset=["stock_code","rcept_dt"], keep="last", inplace=True)

    # 6) clustering
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    tmp  = df_pred[["y_pred","residual"]].replace([np.inf,-np.inf], np.nan)
    mask = tmp.notna().all(axis=1)
    if mask.sum() >= n_clusters:
        feats  = StandardScaler().fit_transform(tmp[mask])
        labels = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)\
                 .fit_predict(np.nan_to_num(feats))
        df_pred.loc[mask, "cluster"] = labels.astype("int8")
    else:
        df_pred["cluster"] = np.nan

    # 7) df_pred + df_clf(corp_name) 병합
    df_master = df_pred.merge(
        df_clf[["stock_code","rcept_dt","corp_name"]],
        on=["stock_code","rcept_dt"], how="left"
    )

    # 8) regression 모듈 피처 병합 (suffix _regorig 제거)
    df_master = df_master.merge(
        df_reg,
        on=["stock_code","rcept_dt"], how="left",
        suffixes=("", "_regorig")
    )
    df_master.drop(columns=[c for c in df_master if c.endswith("_regorig")],
                   inplace=True, errors="ignore")

    # 9) 불필요 컬럼 드롭
    if drop_text_emb:
        df_master.drop(columns=[c for c in df_master if c.startswith("text_emb_")],
                       inplace=True, errors="ignore")
    helper_cols = ["valid_reg","valid_clf","valid_clus","valid_all",
                   "corp_name_x","corp_name_y","sector"]
    if drop_helper_cols:
        df_master.drop(columns=[c for c in helper_cols if c in df_master],
                       inplace=True, errors="ignore")

    # --- 중복한 stock_code+rcept_dt 행 완전 제거 ---
    df_master = df_master.drop_duplicates(subset=["stock_code","rcept_dt"],
                                          keep="last")

    # 10) 최종 컬럼 순서 정리 및 저장
    key  = ["corp_name","stock_code","rcept_dt"]
    rest = [c for c in df_master.columns if c not in key]
    df_master = df_master[key + rest]\
                   .sort_values(["stock_code","rcept_dt"])\
                   .reset_index(drop=True)

    df_master.to_csv(master_csv_path, index=False, encoding="utf-8-sig")
    logging.info(f"✅ Master CSV saved → {master_csv_path} (rows={len(df_master)})")
    return df_master
# -------- modules builder (unchanged) --------
def _build_modules(df_div: pd.DataFrame, df_full: pd.DataFrame,
                   windows: dict, reg_days: list, out_dir: str):
    price_map = {c: g.reset_index(drop=True) for c,g in df_full.groupby("stock_code")}
    def _has_full(series, dt, w):
        pos = series["date"].searchsorted(dt, side="left")
        if pos==len(series): return False,0
        if series.iloc[pos]["date"]!=dt:
            fut = series["date"][series["date"]>=dt]
            if fut.empty: return False,0
            pos = series["date"].searchsorted(fut.iloc[0], side="left")
        lo,hi = pos-w, pos+w+1
        got = max(0,hi)-max(0,lo); need = 2*w+1
        return got==need, got

    for mod,w in windows.items():
        rows,drop_cnt = [],0
        for ev in tqdm(df_div.itertuples(index=False), total=len(df_div), desc=f"module:{mod}", leave=False):
            series = price_map.get(ev.stock_code)
            if series is None:
                drop_cnt+=1; continue
            ok,_ = _has_full(series, ev.rcept_dt, w)
            if not ok:
                drop_cnt+=1; continue
            pos = series["date"].searchsorted(ev.rcept_dt, side="left")
            if series.iloc[pos]["date"]!=ev.rcept_dt:
                fut = series["date"][series["date"]>=ev.rcept_dt]
                pos = series["date"].searchsorted(fut.iloc[0], side="left")
            window = series.iloc[pos-w:pos+w+1].reset_index(drop=True)
            base = window["close"].iloc[w]
            feat = {
                "corp_name": ev.corp_name,
                "stock_code": ev.stock_code,
                "rcept_dt": ev.rcept_dt,
                "sector": getattr(ev,"sector",""),
                "per_share_common": ev.per_share_common,
                "yield_common": ev.yield_common,
                "total_amount": ev.total_amount,
                "div_amount_rank": ev.div_amount_rank,
                "month": ev.month,
                "is_year_end": ev.is_year_end,
            }
            if mod=="classification":
                feat["up_1d"] = int(window["close"].iloc[w+1]/base>1)
            elif mod=="regression":
                for d in reg_days:
                    if w + d < len(window):
                        feat[f"ret_{d}d"] = window["close"].iloc[w+d]/base - 1
                    else:
                        feat[f"ret_{d}d"] = np.nan
            rows.append(feat)
        df_mod = pd.DataFrame(rows)
        out_fp = os.path.join(out_dir, f"{mod}.csv")
        df_mod.to_csv(out_fp, index=False, encoding="utf-8-sig")
        logging.info("✅ [%s] kept=%d dropped=%d → %s", mod, len(df_mod), drop_cnt, out_fp)

# ------------- main pipeline -------------
def run_pipeline(
    start_date: str, end_date: str, max_workers: int = 6,
    skip_refetch: bool = False, full_refetch: bool = True,
    rebuild_nozero: bool = True, drop_zero: bool = False,
    check_code: str = "5930", keep_text_emb: bool = False,
    skip_collect: bool = False, csv_path: str = os.path.join(DATA_DIR,"dividend_with_text.csv"),
    skip_price_fetch: bool = False, refill_html_if_empty: bool = False,
):
    load_dotenv()
    session = _init_http_session()
    API_KEY = os.getenv("DART_API_KEY")
    DOC_URL  = "https://opendart.fss.or.kr/api/document.xml"

    # 1) Incremental collect & HTML/Text fill
    if skip_collect:
        logging.info("↪ skip_collect=True → DART API 건너뜀")
        df_comb = pd.read_csv(csv_path, encoding="utf-8-sig") if os.path.exists(csv_path) else pd.DataFrame()
    else:
        logging.info("1. Incremental collect")
        df_old = pd.read_csv(csv_path, encoding="utf-8-sig") if os.path.exists(csv_path) else pd.DataFrame()
        new_recs = collect_dividend_filings_incremental(start=start_date, end=end_date, save_csv=None, max_workers=max_workers,)
        if not new_recs:
            logging.info("No new filings → exit"); print("[SUMMARY] no new filings, exiting."); return
        df_new = pd.DataFrame(new_recs)
        df_comb = pd.concat([df_old, df_new], ignore_index=True)
        df_comb.drop_duplicates(subset=["rcept_no"], keep="last", inplace=True)
        df_comb.to_csv(csv_path, index=False, encoding="utf-8-sig")
        logging.info("   New=%d Total=%d", len(df_new), len(df_comb))

        mask_empty = df_comb["html"].isna() | (df_comb["html"] == "")
        to_fill = df_comb.loc[mask_empty, "rcept_no"].tolist()

        def fetch_html_text(rcept_no: str) -> dict:
            try:
                res = session.get(DOC_URL, params={"crtfc_key": API_KEY, "rcept_no": rcept_no}, timeout=10)
                soup = BeautifulSoup(res.content, "xml")
                tbl = soup.find("XFormD21") or soup.find("xformd21")
                html = str(tbl or "")
                text = BeautifulSoup(html, "html.parser").get_text(" ", strip=True) if html else ""
                if not html:
                    logging.error(f"본문 없음 (rcept_no={rcept_no})")
                return {"rcept_no": rcept_no, "html": html, "text": text}
            except Exception as e:
                logging.warning(f"HTML/Text fetch failed for {rcept_no}: {e}")
                logging.error(f"본문 없음 (rcept_no={rcept_no})")
                return {"rcept_no": rcept_no, "html":"", "text":""}

        if to_fill:
            filled = list(tqdm(ThreadPoolExecutor(max_workers=max_workers)
                               .map(fetch_html_text, to_fill),
                               total=len(to_fill), desc="HTML/Text"))
            fill_map = {r["rcept_no"]: r for r in filled}
            df_comb.loc[mask_empty, "html"] = df_comb.loc[mask_empty, "rcept_no"].map(lambda k: fill_map[k]["html"])
            df_comb.loc[mask_empty, "text"] = df_comb.loc[mask_empty, "rcept_no"].map(lambda k: fill_map[k]["text"])
            df_comb.to_csv(csv_path, index=False, encoding="utf-8-sig")
            logging.info(f"✅ HTML/Text filled for {len(filled)} records")

    # 2) ML data cleaning
    logging.info("2. ML data cleaning")
    if skip_collect:
        # skip_collect 모드면 기존에 저장된 ML_READY_BASE 를 그대로 로드
        logging.info("↪ skip_collect=True → clean_ml_data 건너뜀, ML_READY_BASE 로드")
        df_ml = pd.read_csv(ML_READY_BASE, dtype=str)
    else:
        if 'df_comb' in locals() and not df_comb.empty:
            df_ml = clean_ml_data(df_comb)
        else:
            df_ml = pd.read_csv(ML_READY_BASE, dtype=str)
    df_ml.replace([np.inf, -np.inf], np.nan, inplace=True)
    before = len(df_ml)
    df_ml.dropna(how="any", inplace=True)
    df_ml.to_csv(ML_READY_BASE, index=False, encoding="utf-8-sig")
    logging.info("   drop %d rows → %s", before-len(df_ml), ML_READY_BASE)

    # 3) Filter listed & non-zero
    logging.info("3. Filter listed & non-zero")
    rebuild_flag = rebuild_nozero or not os.path.exists(ML_READY_NOZERO)
    if not rebuild_flag:
        df_f = pd.read_csv(ML_READY_NOZERO, dtype=str)
        if df_f.empty:
            rebuild_flag = True
    if rebuild_flag:
        krx_url = "https://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13"
        resp = session.get(krx_url); resp.encoding="euc-kr"
        krx_df = pd.read_html(resp.text, header=0)[0]
        ccol = [c for c in krx_df.columns if "종목코드" in c or "Code" in c][0]
        listed = set(krx_df[ccol].astype(str).str.extract(r"(\d+)")[0].str.zfill(6))
        df_f = pd.read_csv(ML_READY_BASE, dtype=str)
        df_f["stock_code"] = df_f["stock_code"].str.zfill(6)
        num_cols = ["per_share_common","yield_common","total_amount"]
        for c in num_cols:
            df_f[c] = (df_f[c].str.replace(",","").str.replace("%","").astype(float))
        df_f = df_f[df_f.stock_code.isin(listed)]
        if drop_zero:
            df_f = df_f[~((df_f[num_cols] <= 0).any(axis=1))]
        df_f.dropna(subset=num_cols, inplace=True)
        df_f.to_csv(ML_READY_NOZERO, index=False, encoding="utf-8-sig")
    else:
        df_f = pd.read_csv(ML_READY_NOZERO, dtype=str)
    logging.info("   filtered rows=%d", len(df_f))
    # 4) Price fetch & merge
    if skip_price_fetch:
        logging.info("↪ skip_price_fetch=True → skip price step")
    else:
        logging.info("4. Price fetch & merge")
        codes = df_f.stock_code.unique().tolist()
        if full_refetch:
            shutil.rmtree(CACHE_DIR, ignore_errors=True)
            os.makedirs(CACHE_DIR, exist_ok=True)
            logging.warning("FULL refetch → cache cleared")
        def _fetch_price(code: str) -> bool:
            try:
                fetch_price_series(stock_code=code,
                                   start=f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}",
                                   end=f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}",
                                   cache_dir_path=CACHE_DIR, session=session, verbose=False)
                return True
            except:
                return fetch_one_code(code, start_date, end_date) is not None
        ok_cnt = sum(tqdm(ThreadPoolExecutor(max_workers=max_workers)
                          .map(_fetch_price, codes), total=len(codes), desc="price_fetch"))
        logging.info("   price success %d/%d", ok_cnt, len(codes))

        dfs, bad = [], []
        for code in codes:
            fp = os.path.join(CACHE_DIR, f"{code}.csv")
            d = _load_cache(fp, code) if os.path.exists(fp) else None
            if d is None or d.empty:
                bad.append(code)
            else:
                dfs.append(d)
        if dfs:
            new_hist = (pd.concat(dfs, ignore_index=True)
                        .drop_duplicates(["stock_code","date"])
                        .sort_values(["stock_code","date"]))
            new_hist.to_csv(HIST_PATH, index=False, encoding="utf-8-sig")
            logging.info("   merged → %s", HIST_PATH)
        if bad:
            logging.warning("Missing price for %d codes: %s", len(bad), bad[:10])

    # 5) Resample B-days
    logging.info("5. Resample B-days")
    df_prices = pd.read_csv(HIST_PATH, parse_dates=["date"], dtype={"stock_code": str})
    outs = []
    for code, grp in tqdm(df_prices.groupby("stock_code"), desc="resample"):
        grp = grp.set_index("date").sort_index()
        idx = pd.date_range(grp.index.min(), grp.index.max(), freq="B")
        filled = grp.reindex(idx).ffill().bfill()
        filled["stock_code"] = code
        outs.append(filled.reset_index().rename(columns={"index":"date"}))
    pd.concat(outs, ignore_index=True).to_csv(HIST_PATH, index=False, encoding="utf-8-sig")

    # 6) module CSV 생성 (regression 1~30일)
    df_full = pd.read_csv(HIST_PATH, parse_dates=["date"], dtype={"stock_code":str})
    df_div  = pd.read_csv(ML_READY_NOZERO, dtype={"stock_code":str})
    df_div["rcept_dt"] = pd.to_datetime(df_div.rcept_dt).dt.normalize()
    df_div["period"] = df_div.rcept_dt.dt.to_period("M")
    df_div["div_amount_rank"] = df_div.groupby("period").per_share_common.rank(pct=True)
    df_div["month"] = df_div.rcept_dt.dt.month
    df_div["is_year_end"] = (df_div.month==12).astype(int)
    _build_modules(
        df_div, df_full,
        windows={"classification":1,"regression":30,"clustering":30},
        reg_days=list(range(1,31)),    # 1~30일 모두
        out_dir=MODULE_DIR
    )

    # 7) Feature engineering (ret_1d~ret_30d 포함)
    feat_df = generate_classification_features(
        os.path.join(MODULE_DIR,"classification.csv"),
        HIST_PATH,
        os.path.join(DATA_DIR,"sector_info.csv"),
        ML_READY_NOZERO
    )
    feat_df.replace([np.inf,-np.inf], np.nan, inplace=True)

    # price map for ret calculation
    df_price = pd.read_csv(HIST_PATH, parse_dates=["date"], dtype={"stock_code":str})
    price_map = {code: grp.set_index("date")["close"] for code, grp in df_price.groupby("stock_code")}

    def calc_ret(row, d):
        pr = price_map.get(row.stock_code)
        if pr is None or row.rcept_dt not in pr.index:
            return np.nan
        idx = pr.index.get_loc(row.rcept_dt)
        base = pr.iloc[idx]
        return pr.iloc[idx+d]/base - 1 if idx+d < len(pr) else np.nan

    for d in range(1,31):
        feat_df[f"ret_{d}d"] = feat_df.apply(lambda r, d=d: calc_ret(r,d), axis=1)

    required = ["up_1d","per_share_common","yield_common","total_amount","div_amount_rank","month","is_year_end"]
    feat_df.dropna(subset=required, inplace=True)
    feat_df.to_parquet(os.path.join(DATA_DIR,"df_feat.parquet"), index=False)
    logging.info("   df_feat saved")

        # 8) Train/Load classifier & predict p_up
    clf = train_or_load_classifier(
     os.path.join(DATA_DIR, "df_feat.parquet"),
     CLF_MODEL_FP
 )
    df_feat = pd.read_parquet(os.path.join(DATA_DIR, "df_feat.parquet"))

    # 예측용 X 준비
    Xp = df_feat.drop(columns=[...], errors="ignore")
    booster = clf.booster_ if hasattr(clf, "booster_") else clf
    fns = booster.feature_name()
    for c in fns:
        if c not in Xp.columns:
            Xp[c] = 0
    Xp = Xp[fns]

    # 예측 수행
    pup = clf.predict(Xp) if not Xp.empty else np.array([])
    df_pup = pd.DataFrame({
        "stock_code": df_feat.stock_code.str.zfill(6),
        "rcept_dt": df_feat.rcept_dt.dt.normalize(),
        "p_up": pup
    })
    df_pup.to_csv(os.path.join(DATA_DIR, "p_up_temp.csv"), index=False)

    # df_feat.parquet 에 p_up 병합
    df_feat = df_feat.merge(
        df_pup.assign(
            stock_code=lambda d: d.stock_code.str.zfill(6),
            rcept_dt=lambda d: pd.to_datetime(d.rcept_dt).dt.normalize()
        ),
        on=["stock_code","rcept_dt"], how="left"
    )
    df_feat.to_parquet(os.path.join(DATA_DIR, "df_feat.parquet"), index=False)
    logging.info("▶ p_up 값을 df_feat.parquet에 병합 완료")


    # 9) Regressor train & predict
    logging.info("9. Train/Load regressor & predict")
    train_or_load_regressor(os.path.join(MODULE_DIR,"regression.csv"), REG_MODEL_FP, REG_PRED_CSV)
    
    # 10) Build Master CSV
    _build_master_csv(MODULE_DIR, DATA_DIR, MASTER_CSV, drop_text_emb=not keep_text_emb)

    print("[SUMMARY] Pipeline complete.")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=str, default="20130101")
    p.add_argument("--end",   type=str, default=datetime.today().strftime("%Y%m%d"))
    p.add_argument("--workers",    type=int, default=6)
    p.add_argument("--full-refetch",action="store_true")
    p.add_argument("--rebuild-nozero",action="store_true")
    p.add_argument("--drop-zero",   action="store_true")
    p.add_argument("--check-code",  type=str, default="5930")
    p.add_argument("--keep-text-emb",action="store_true")
    p.add_argument("--skip-collect",action="store_true")
    p.add_argument("--csv-path",    type=str, default=os.path.join(DATA_DIR,"dividend_with_text.csv"))
    p.add_argument("--skip-price-fetch",action="store_true")
    args = p.parse_args()

    run_pipeline(
        start_date       = args.start,
        end_date         = args.end,
        max_workers      = args.workers,
        full_refetch     = args.full_refetch,
        rebuild_nozero   = args.rebuild_nozero,
        drop_zero        = args.drop_zero,
        check_code       = args.check_code,
        keep_text_emb    = args.keep_text_emb,
        skip_collect     = args.skip_collect,
        csv_path         = args.csv_path,
        skip_price_fetch = args.skip_price_fetch,
    )
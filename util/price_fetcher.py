# utils/price_fetcher.py
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from datetime import datetime
import pandas as pd
import yfinance as yf
import requests

def fetch_price_series(
    stock_code: str,
    start: str,
    end: str,
    cache_dir_path: str,
    session: Optional[requests.Session] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    단일 종목의 가격 시계열을 캐시에서 불러오거나 yfinance에서 다운로드하여
    캐시에 저장 후 반환합니다.

    반환 DataFrame 컬럼:
      - Date (DatetimeIndex)
      - Close (float)
      - Volume (int)
      - stock_code (str)

    :param stock_code: 종목 코드 (숫자만) – 내부에서 6자리로 zfill 처리
    :param start: 시작일 (YYYY-MM-DD)
    :param end: 종료일 (YYYY-MM-DD)
    :param cache_dir_path: 캐시 디렉토리 경로
    :param session: HTTP 세션 (선택)
    :return: 가격 DataFrame
    """
    os.makedirs(cache_dir_path, exist_ok=True)
    # 종목 코드를 항상 6자리로
    code6 = str(stock_code).zfill(6)
    out_fp = os.path.join(cache_dir_path, f"{code6}.csv")

    # 1) 캐시 사용 시도
    if os.path.exists(out_fp):
        try:
            df_cache = pd.read_csv(out_fp, parse_dates=["Date"], index_col="Date")
            df_cache["stock_code"] = code6
            if verbose:
                print(f"✓ cache used: {code6}")
            return df_cache
        except Exception:
            if verbose:
                print(f"⚠ cache parse failed, redownload: {code6}")

    # 2) yfinance 다운로드
    ticker_ext = f"{code6}.KS"
    if verbose:
        print(f"• downloading: {ticker_ext}")
    try:
        df_raw = yf.download(
            ticker_ext,
            start=start,
            end=end,
            auto_adjust=False,
            progress=False,
        )
        if isinstance(df_raw, pd.DataFrame):
            df_out = df_raw[["Close", "Volume"]].copy()
        else:
            df_out = df_raw.to_frame(name="Close")
        df_out.index.name = "Date"
        df_out["stock_code"] = code6
        df_out.to_csv(out_fp)
        if verbose:
            print(f"✓ downloaded & cached: {code6}")
        return df_out
    except Exception as e:
        if verbose:
            print(f"⚠ download failed {ticker_ext}: {e}")
        return pd.DataFrame(
            columns=["Close", "Volume", "stock_code"],
            index=pd.DatetimeIndex([], name="Date")
        )


def run_price_fetching(
    div_path: str,
    hist_path: str,
    check_path: str,
    cache_dir_path: str,
    window_days: int,
    max_workers: int = 10,
) -> None:
    """
    배당 공시 날짜 기준으로 과거/미래 window_days 범위 내 가격을 수집하고,
    간단한 윈도우 검증 결과를 저장합니다.
    """
    print("▶ run_price_fetching start")

    # 1) 배당 데이터 로드
    df_div = pd.read_csv(div_path, parse_dates=["rcept_dt"], dtype={"stock_code": str})
    # 종목 코드 6자리 패딩
    df_div["stock_code"] = df_div["stock_code"].astype(str).str.zfill(6)
    tickers = df_div["stock_code"].unique().tolist()
    print(f"  • unique tickers: {len(tickers)}")

    # 2) 캐시 디렉토리 생성
    os.makedirs(cache_dir_path, exist_ok=True)

    # 3) 병렬 다운로드
    def _fetch(tic: str) -> pd.DataFrame:
        return fetch_price_series(
            stock_code     = tic,
            start          = "1900-01-01",
            end            = datetime.today().strftime("%Y-%m-%d"),
            cache_dir_path = cache_dir_path,
        )

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        all_dfs = list(ex.map(_fetch, tickers))

    # 4) 히스토리 병합 및 저장
    df_hist = (
        pd.concat(all_dfs)
          .reset_index()  # Date -> date 컬럼으로 변환
    )
    df_hist.rename(columns={
        "Date":  "date",
        "Close": "close",
        "Volume": "volume"
    }, inplace=True)
    df_hist.to_csv(hist_path, index=False, encoding="utf-8-sig")
    print(f"  ✓ price history saved: {hist_path} (rows: {len(df_hist)})")

    # 5) 윈도우 검증 계산 및 저장
    results = []
    for _, row in df_div.iterrows():
        tic, dt = row.stock_code, row.rcept_dt
        sub = df_hist[
            (df_hist.stock_code == tic) &
            (df_hist.date.between(
                dt - pd.Timedelta(window_days, "D"),
                dt + pd.Timedelta(window_days, "D")
            ))
        ]
        if len(sub) >= 2:
            ret = sub["close"].pct_change().dropna()
            results.append({
                "stock_code": tic,
                "rcept_dt":   dt,
                "window_return": (1 + ret).prod() - 1
            })
    df_chk = pd.DataFrame(results)
    df_chk.to_csv(check_path, index=False, encoding="utf-8-sig")
    print(f"  ✓ window check saved: {check_path} (rows: {len(df_chk)})")

    print("▶ run_price_fetching done")
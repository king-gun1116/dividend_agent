import os
import pickle
from datetime import datetime, timedelta
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from pykrx import stock

CACHE_ROOT = '/Users/gun/Desktop/미래에셋 AI 공모전/Agent/agent_a/data/cache'
KS_CSV = '/Users/gun/Desktop/미래에셋 AI 공모전/Agent/agent_a/data/mappings/ticker_mapping_ks.csv'
KQ_CSV = '/Users/gun/Desktop/미래에셋 AI 공모전/Agent/agent_a/data/mappings/ticker_mapping_kq.csv'

START_DATE = '2024-01-01'
END_DATE   = '2025-07-28'

def daterange(start, end):
    for n in range((end - start).days + 1):
        yield start + timedelta(n)

def save_ohlcv(ticker, date_str, cache_dir):
    """단일 (티커, 날짜)에 대해 pykrx로 ohlcv 받아서 pickle 저장"""
    fn = f"{ticker}_{date_str}.pkl"
    fp = os.path.join(cache_dir, fn)
    # 이미 존재하면 skip
    if os.path.exists(fp):
        try:
            df = pickle.load(open(fp, "rb"))
            # 비어있지 않으면 패스
            if isinstance(df, pd.DataFrame) and not df.empty and not df.isnull().values.all():
                return "SKIP"
        except Exception:
            pass
    # pykrx로 데이터 수집
    try:
        df = stock.get_market_ohlcv_by_date(date_str, date_str, ticker)
        if df.empty:
            result = pd.DataFrame()  # 빈 DF로 저장
        else:
            result = df.iloc[0:1]  # 첫 번째 row만 (당일)
        with open(fp, "wb") as f:
            pickle.dump(result, f)
        return "OK"
    except Exception as e:
        return f"ERROR: {e}"

if __name__ == "__main__":
    # 티커 로드
    ks_map = pd.read_csv(KS_CSV, dtype=str)
    kq_map = pd.read_csv(KQ_CSV, dtype=str)
    tickers = sorted(set(ks_map['ticker']).union(set(kq_map['ticker'])))
    # 날짜 범위 준비
    start = datetime.strptime(START_DATE, "%Y-%m-%d")
    end   = datetime.strptime(END_DATE, "%Y-%m-%d")
    dates = [d.strftime("%Y-%m-%d") for d in daterange(start, end)]

    # 작업 리스트 작성 (date별로 폴더 자동 생성)
    jobs = []
    for date_str in dates:
        cache_dir = os.path.join(CACHE_ROOT, date_str)
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        for ticker in tickers:
            jobs.append((ticker, date_str, cache_dir))

    print(f"총 작업 수: {len(jobs):,} (티커×날짜)")

    # 진행 카운트 변수
    saved_count = 0
    skip_count = 0
    error_count = 0

    # 멀티프로세싱 실행 (최대 6~8 core 권장, 더 높으면 pykrx 요청 timeout 주의)
    N_WORKERS = os.cpu_count() or 4
    N_WORKERS = max(4, min(N_WORKERS, 16))

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = [executor.submit(save_ohlcv, t, d, c) for t, d, c in jobs]
        pbar = tqdm(as_completed(futures), total=len(futures), desc="수집진행", ncols=110)
        for i, f in enumerate(pbar):
            try:
                result = f.result()
                if result == "OK":
                    saved_count += 1
                elif result == "SKIP":
                    skip_count += 1
                elif result and str(result).startswith("ERROR"):
                    error_count += 1
                # tqdm 진행 바에 실시간 표시
                pbar.set_postfix({'Saved': saved_count, 'Skipped': skip_count, 'Errors': error_count})
            except Exception as e:
                error_count += 1
                pbar.set_postfix({'Saved': saved_count, 'Skipped': skip_count, 'Errors': error_count})
            if (i+1) % 1000 == 0:
                print(f"진행 {i+1:,} / {len(futures):,}, 저장: {saved_count}, 스킵: {skip_count}, 에러: {error_count}")

    print("✅ 캐시 자동화 수집 완료")
# utils/data_cleaning.py
# ─────────────────────────────────────────────────────────
# 머신러닝용 데이터 준비 및 정제 로직
#   • 상장 유지 기업 필터링  ← NEW
#   • 자회사 공시 필터링
#   • 희소/불필요 컬럼 제거
#   • 숫자형 변환
#   • 결측치 및 중복 제거
#   • 중앙값 대체
#   • 날짜 파싱
#   • 불필요 이벤트(취소·우선주·실제 배당 0) 필터링
#   • ML 불필요 컬럼 드랍
# ─────────────────────────────────────────────────────────

from __future__ import annotations
import pandas as pd
import FinanceDataReader as fdr
from functools import lru_cache

# ─────────────────────────────────────────────────────
# 상장 종목 리스트 캐싱
# ─────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _get_current_listed_codes() -> set[str]:
    """KRX 상장 종목 6자리 코드 세트를 1회만 로드·캐싱"""
    krx = fdr.StockListing("KRX")
    return set(krx["Code"].astype(str).str.zfill(6).unique())


def filter_listed_companies(df: pd.DataFrame) -> pd.DataFrame:
    """현재 상장된 종목(stock_code)만 남기기"""
    listed = _get_current_listed_codes()
    if "stock_code" in df.columns:
        df["stock_code"] = df["stock_code"].astype(str).str.zfill(6)
        return df[df["stock_code"].isin(listed)].reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────
# 기존 함수들
# ─────────────────────────────────────────────────────
def filter_subsidiary_policies(df: pd.DataFrame) -> pd.DataFrame:
    """report_nm에 '자회사' 키워드 포함된 공시 제거"""
    if "report_nm" in df.columns:
        return df[~df["report_nm"].str.contains("자회사", na=False)]
    return df


def drop_sparse_columns(df: pd.DataFrame) -> pd.DataFrame:
    """div_type, div_kind 등 희소/불필요 컬럼 제거"""
    drop_cols = [
        "div_type",
        "div_kind",
        "per_share_preferred",
        "yield_preferred",
        "html",
    ]
    return df.drop(columns=[c for c in drop_cols if c in df.columns])


def convert_numeric_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """콤마·하이픈 제거 후 지정 컬럼을 float 변환"""
    for col in cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("-", "", regex=False)
                .str.strip()
                .replace("", pd.NA)
                .pipe(pd.to_numeric, errors="coerce")
            )
    return df


def drop_na_and_duplicates(df: pd.DataFrame, subset: list[str]) -> pd.DataFrame:
    """필수 컬럼 결측치 제거 및 완전 중복 제거"""
    df_clean = df.dropna(subset=subset).reset_index(drop=True)
    return df_clean.drop_duplicates().reset_index(drop=True)


def median_impute(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """컬럼의 결측을 중앙값으로 대체"""
    if col in df.columns:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
    return df


def parse_date_columns(df: pd.DataFrame, date_cols: list[str]) -> pd.DataFrame:
    """문자형 YYYYMMDD 날짜 컬럼을 Timestamp로 파싱"""
    for c in date_cols:
        if c in df.columns:
            df[c] = df[c].replace("-", pd.NA)
            df[c] = pd.to_datetime(df[c], format="%Y%m%d", errors="coerce")
    return df


def drop_ml_unused_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """ML용으로 불필요한 메타/날짜/식별자 컬럼 제거"""
    return df.drop(columns=[c for c in cols if c in df.columns])


# ─────────────────────────────────────────────────────
# 메인 함수: clean_ml_data
# ─────────────────────────────────────────────────────
def clean_ml_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    ML 모델 학습을 위한 일괄 정제 함수:
      0) 현재 상장기업 필터링             ← NEW
      1) 자회사 공시 필터링
      2) 희소 컬럼 제거
      3) numeric 변환 (per_share_common, yield_common, total_amount)
      4) 필수 컬럼 결측치·중복 제거
      5) 중앙값 대체 (yield_common)
      6) 날짜 컬럼 파싱
      7) 불필요 이벤트 필터링
      8) ML 불필요 컬럼 드랍
    """

    # 0) 상장 유지 기업만 필터
    df = filter_listed_companies(df)

    # 1) 자회사 공시 제거
    df = filter_subsidiary_policies(df)

    # 2) 희소/불필요 컬럼 제거
    df = drop_sparse_columns(df)

    # 3) 숫자형 변환
    df = convert_numeric_columns(
        df, ["per_share_common", "yield_common", "total_amount"]
    )

    # 4) 필수 결측 + 중복 제거
    df = drop_na_and_duplicates(df, ["per_share_common", "total_amount"])

    # 5) yield_common 중앙값 대체
    df = median_impute(df, "yield_common")

    # 6) 날짜 파싱
    df = parse_date_columns(
        df,
        ["record_date", "payment_date", "meeting_date", "board_decision_date"],
    )

    # 7) 불필요 이벤트 필터링
    mask_cancel = df.get("report_nm", pd.Series()).str.contains("정정", na=False)
    mask_pref_only = (df.per_share_common == 0) & (df.total_amount > 0)
    mask_no_div = (df.per_share_common == 0) & (df.total_amount == 0)
    df = df[~(mask_cancel | mask_pref_only | mask_no_div)].reset_index(drop=True)

    # 8) ML 불필요 컬럼 드랍
    df = drop_ml_unused_columns(
        df,
        [
            "corp_name",
            "stock_code",
            "rcept_dt",
            "report_nm",
            "rcept_no",
            "meeting_held",
            "days_to_payment",
            "days_to_payment_missing",
            "record_date",
            "payment_date",
            "meeting_date",
            "board_decision_date",
        ],
    )
    return df
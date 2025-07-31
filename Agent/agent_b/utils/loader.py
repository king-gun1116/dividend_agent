import pandas as pd
from typing import List, Dict, Any


def extract_table_fields(html: str) -> Dict[str, str]:
    """
    <table> 안에서
      1열은 라벨, 2열은 값 → {'배당금총액(원)': '866,000,000', '1주당 배당금(원)': '271', …}
    으로 반환합니다.
    """
    if not isinstance(html, str) or not html:
        return {}
    soup = BeautifulSoup(html, "lxml")
    tbl = soup.find("table")
    if not tbl:
        return {}

    fields: Dict[str, str] = {}
    for tr in tbl.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) >= 2:
            label = tds[0].get_text(strip=True).rstrip(" :")
            value = " / ".join(td.get_text(strip=True) for td in tds[1:])
            if label and value:
                fields[label] = value
    return fields


def load_dividend_csv(path: str) -> List[Dict[str, Any]]:
    """
    crawl_dividend.csv 로드: 배당금(주당배당금_일반) > 0 인 유효 공시만 포함
    """
    df = pd.read_csv(path, dtype=str)
    records: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        raw = row.get('주당배당금_일반', None)

        # 결측(NaN) 또는 빈 문자열이면 건너뛰기
        if pd.isna(raw) or raw == "":
            continue

        # 문자열이면 콤마 제거 후 float 변환, 아니면 그대로 float
        try:
            per_share = float(raw.replace(',', '')) if isinstance(raw, str) else float(raw)
        except ValueError:
            # 변환 불가능한 값이면 건너뛰기
            continue

        # 0 이하 배당 제외
        if per_share <= 0:
            continue

        code6 = row.get('종목코드', '').zfill(6)
        rec_id = f"{code6}_{row.get('배정기준일','')}"
        text = (
            f"{row.get('종목명','')} 배당기준일 {row.get('배정기준일','')}, "
            f"주당배당금 {int(per_share):,}원"
        )

        # 모든 원본 컬럼을 meta 에 보존
        meta: Dict[str, Any] = row.to_dict()
        meta.update({
            'source': 'crawl',
            'stock_code': code6,
            'dividend_per_share': per_share,
            'detail_text': ''
        })

        records.append({'id': rec_id, 'text': text, 'meta': meta})

    return records


def load_master_csv(path: str) -> List[Dict[str, Any]]:
    """
    all_stocks_master.csv 로드: 모든 컬럼을 meta로 포함
    """
    df = pd.read_csv(path, dtype=str)
    records: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        meta: Dict[str, Any] = row.to_dict()

        # stock_code, y_pred, residual 등은 기존과 동일하게 파싱
        code6 = meta.get('stock_code','').zfill(6)
        rec_id = f"{code6}_{meta.get('rcept_dt','')}"
        text = f"{meta.get('corp_name','')} #{code6} 예상배당 {meta.get('y_pred','')}"

        # p_up도 float으로 변환
        def to_float(x, default=0.0):
            try:
                return float(x)
            except:
                return default

        meta.update({
            'source': 'master',
            'stock_code':   code6,
            'y_pred':       to_float(meta.get('y_pred', 0.0)),
            'residual':     to_float(meta.get('residual', 0.0)),
            'cluster':      int(to_float(meta.get('cluster', -1))),
            'per_share_common': to_float(meta.get('per_share_common', 0.0)),
            'yield_common':     to_float(meta.get('yield_common', 0.0)),
            'total_amount':     to_float(meta.get('total_amount', 0.0)),
            'div_amount_rank':  int(to_float(meta.get('div_amount_rank', 0))),
            'month':            int(to_float(meta.get('month', 0))),
            'is_year_end':      bool(int(to_float(meta.get('is_year_end', 0)))),
            'p_up':              to_float(meta.get('p_up', 0.0)),
        })

        records.append({'id': rec_id, 'text': text, 'meta': meta})
    return records

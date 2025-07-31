# util/model_utils.py

import os
import joblib
import pandas as pd
import lightgbm as lgb

def train_or_load_classifier(df_feat_fp: str, model_fp: str):
    """
    df_feat_fp: parquet 형식의 feature DataFrame (up_1d 포함)
    model_fp: 저장할/불러올 모델 파일 경로 (.pkl)
    """
    df = pd.read_parquet(df_feat_fp)
    X = df.drop(columns=["up_1d", "corp_name", "stock_code", "rcept_dt", "sector"], errors="ignore")
    y = df["up_1d"]

    if os.path.exists(model_fp):
        clf = joblib.load(model_fp)
    else:
        clf = lgb.LGBMClassifier(n_estimators=100, random_state=42)
        clf = clf.fit(X, y)
        joblib.dump(clf, model_fp)

    return clf


def train_or_load_regressor(reg_csv_fp: str, model_fp: str, out_pred_fp: str):
    """
    reg_csv_fp: regression 모듈 CSV (ret_* 컬럼들 포함)
    model_fp: 저장할/불러올 모델 파일 경로 (.pkl)
    out_pred_fp: 예측 결과를 저장할 CSV 경로
    """
    # 1) 데이터 로드
    df = pd.read_csv(reg_csv_fp, parse_dates=["rcept_dt"])
    # regressor 학습에 사용하지 않을 컬럼들 제거
    X = df.drop(
        columns=[c for c in df.columns if c.startswith("ret_")] +
                ["corp_name", "stock_code", "rcept_dt", "sector"],
        errors="ignore"
    )
    # 타깃: 1일 수익률
    y = df["ret_1d"]

    # 2) 모델 로드 또는 학습
    if os.path.exists(model_fp):
        reg = joblib.load(model_fp)
    else:
        reg = lgb.LGBMRegressor(n_estimators=100, random_state=42).fit(X, y)
        joblib.dump(reg, model_fp)

    # 3) 올바른 피처 이름 추출
    if isinstance(reg, lgb.LGBMRegressor):
        # sklearn 래퍼
        feat_names = reg.booster_.feature_name()
        predict_fn = reg.predict
    elif isinstance(reg, lgb.Booster):
        # 순수 Booster
        feat_names = reg.feature_name()
        predict_fn = reg.predict
    else:
        raise ValueError(f"Unsupported model type: {type(reg)}")

    # 4) 입력 데이터 재정렬: 학습 시 피처 순서에 맞추고 누락된 건 0으로
    for c in feat_names:
        if c not in X.columns:
            X[c] = 0
    X = X[feat_names]

    # 5) 예측
    preds = predict_fn(X)
    residuals = y - preds

    # 6) 결과 저장
    pd.DataFrame({
        "stock_code": df.stock_code,
        "rcept_dt": df.rcept_dt,
        "y_pred": preds,
        "residual": residuals
    }).to_csv(out_pred_fp, index=False, encoding="utf-8-sig")

    return reg
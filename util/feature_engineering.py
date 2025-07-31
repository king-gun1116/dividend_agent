import pandas as pd


def generate_classification_features(
    classification_fp: str,
    price_history_fp: str,
    sector_info_fp: str,
    dividend_ml_ready_fp: str
) -> pd.DataFrame:
    """
    Generate features for the classification model based on the classification module output.

    Args:
        classification_fp: Path to the module/classification.csv file.
        price_history_fp: Path to the price history CSV (not used in this basic version).
        sector_info_fp: Path to the sector info CSV (not used in this basic version).
        dividend_ml_ready_fp: Path to the cleaned dividend CSV (not used in this basic version).

    Returns:
        DataFrame with features including:
        - corp_name, stock_code, rcept_dt, sector
        - per_share_common, yield_common, total_amount
        - div_amount_rank, month, is_year_end
        - up_1d (target)
    """
    # Load the classification module output
    df = pd.read_csv(
        classification_fp,
        parse_dates=["rcept_dt"],
        dtype={"stock_code": str}
    )

    # Cast key numeric columns to ensure correct types
    numeric_cols = [
        "per_share_common", "yield_common", "total_amount",
        "div_amount_rank", "month", "is_year_end", "up_1d"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop any rows with missing values in features or target
    df = df.dropna(subset=numeric_cols)

    return df

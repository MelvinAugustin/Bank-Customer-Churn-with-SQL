import pandas as pd


def preprocess(df: pd.DataFrame) -> tuple:
    """Clean, engineer features, and return X, y.
    
    No Streamlit calls — safe to import from any page.
    """
    df = df.copy()

    # Normalize all column names: strip whitespace + lowercase
    df.columns = df.columns.str.strip().str.lower()

    # ── Locate customer_id column (flexible match) ───────────────────────────
    id_candidates = [c for c in df.columns if "customer" in c or c in ("id", "cust_id")]
    if id_candidates:
        id_col = id_candidates[0]
        df = df.drop_duplicates(subset=id_col)
        df = df.set_index(id_col)
    else:
        df = df.drop_duplicates()

    # ── Locate churn target column ───────────────────────────────────────────
    if "churn" not in df.columns:
        churn_candidates = [c for c in df.columns if "churn" in c]
        if churn_candidates:
            df = df.rename(columns={churn_candidates[0]: "churn"})
        else:
            raise ValueError(
                f"No 'churn' target column found. "
                f"Available columns: {list(df.columns)}"
            )

    # ── Feature engineering ──────────────────────────────────────────────────
    if "balance" in df.columns:
        df["zero_balance"] = (df["balance"] == 0).astype(int)

    # Separate target before encoding
    y = df["churn"]
    X = df.drop("churn", axis=1)

    # ── Auto-handle ALL remaining object/string columns ──────────────────────
    string_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    if string_cols:
        low_card  = [c for c in string_cols if X[c].nunique() <= 20]
        high_card = [c for c in string_cols if X[c].nunique() >  20]

        if high_card:
            X = X.drop(columns=high_card)

        if low_card:
            X = pd.get_dummies(X, columns=low_card, drop_first=True)

    # ── Force all remaining columns to numeric ───────────────────────────────
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # Drop columns that are entirely NaN after coercion
    all_nan = X.columns[X.isna().all()].tolist()
    if all_nan:
        X = X.drop(columns=all_nan)

    # Fill remaining NaNs with column median
    X = X.fillna(X.median(numeric_only=True))

    return X, y

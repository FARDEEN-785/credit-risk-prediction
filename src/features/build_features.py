import pandas as pd
from pathlib import Path


INPUT_PATH = Path("data/processed/credit_risk_processed.csv")
OUTPUT_PATH = Path("data/processed/credit_risk_features.csv")
TARGET_COLUMN = "loan_status"


def build_features():
    df = pd.read_csv(INPUT_PATH)

    # Separate target
    y = df[TARGET_COLUMN]
    X = df.drop(columns=[TARGET_COLUMN])

    # ---- Feature Engineering ----

    # Income cushion
    X["income_cushion"] = X["person_income"] - X["loan_amnt"]

    # Loan to income ratio (recomputed)
    X["loan_to_income"] = X["loan_amnt"] / (X["person_income"] + 1)

    # New employee flag
    X["is_new_employee"] = (X["person_emp_length"] < 1).astype(int)

    # Credit history relative to age
    X["credit_history_ratio"] = (
        X["cb_person_cred_hist_length"] / (X["person_age"] + 1)
    )

    # Combine back target
    feature_df = X.copy()
    feature_df[TARGET_COLUMN] = y.values

    # Save engineered dataset
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(OUTPUT_PATH, index=False)

    print("✅ Feature engineering complete")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    build_features()

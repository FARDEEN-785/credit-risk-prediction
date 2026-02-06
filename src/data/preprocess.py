import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

RAW_DATA_PATH = Path("data/raw/credit_risk.csv")
PROCESSED_DATA_PATH = Path("data/processed/credit_risk_processed.csv")
TARGET_COLUMN = "loan_status"


def preprocess_data():
    
    df = pd.read_csv(RAW_DATA_PATH)
    
    y = df[TARGET_COLUMN]
    X = df.drop(columns=[TARGET_COLUMN])


    numerical_cols =[
         "person_age",
        "person_income",
        "person_emp_length",
        "loan_amnt",
        "loan_int_rate",
        "loan_percent_income",
        "cb_person_cred_hist_length"
    ]

    categorical_cols = [
        "person_home_ownership",
        "loan_intent",
        "loan_grade",
        "cb_person_default_on_file"
    ]

    # HANDLE MISSING VALUE

    # person_emp_length: fill with median
    X['person_emp_length'] =   X['person_emp_length'].fillna(
        X['person_emp_length'].median())

     # fill loan_int_rate with median
    X["loan_int_rate"] = X["loan_int_rate"].fillna(
        X["loan_int_rate"].median()
    )
    
     # ---- Encode categorical variables ----
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)


      # ---- Scale numerical features ----
    scaler = StandardScaler()
    X_encoded[numerical_cols] = scaler.fit_transform(X_encoded[numerical_cols])


    # Combine features and target
    processed_df = X_encoded.copy()
    processed_df[TARGET_COLUMN] = y.values

    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(PROCESSED_DATA_PATH, index=False)

    print("✅ Preprocessing complete")
    print(f"Saved to: {PROCESSED_DATA_PATH}")


if __name__ == "__main__":
    preprocess_data()
    


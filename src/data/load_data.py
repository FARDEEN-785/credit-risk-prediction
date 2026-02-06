import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/raw/credit_risk.csv")
TARGET_COLUMN = 'loan_status'


def load_raw_data(path):

    df = pd.read_csv(path)
    return df


def inspect_data(df):

    """
    Inspect dataset structure and return summary statistics.

    Args:
        df (pd.DataFrame): Raw dataset

    Returns:
        dict: Dataset overview
    """

    summary = {
        'num_rows': df.shape[0],
        'num_columns': df.shape[1],
        "columns": df.columns.tolist(),
        "missing_values": df.isnull().sum().to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "target_distribution": df[TARGET_COLUMN].value_counts(normalize=True).to_dict()
    }

    return summary

def split_features_target(df):
    """
    Separate features and target variable.

    Args:
        df (pd.DataFrame): Raw dataset

    Returns:
        X (pd.DataFrame): Feature set
        y (pd.Series): Target variable
    """
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y

if __name__ == "__main__":
    df = load_raw_data(DATA_PATH)
    summary = inspect_data(df)
    X, y = split_features_target(df)

    print("📊 Dataset Summary")
    for key, value in summary.items():
        print(f"{key}: {value}")

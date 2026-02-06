import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
import xgboost as xgb


import mlflow

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Credit_Risk_MLOps")


DATA_PATH = Path("data/processed/credit_risk_features.csv")
TARGET_COLUMN = "loan_status"


def train_models():
    # Load data
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # Train-validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # -------------------------
    # Logistic Regression
    # -------------------------
    with mlflow.start_run(run_name="Logistic_Regression"):
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_train, y_train)

        y_pred = lr.predict(X_val)
        y_prob = lr.predict_proba(X_val)[:, 1]

        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_metric("roc_auc", roc_auc_score(y_val, y_prob))
        mlflow.log_metric("precision", precision_score(y_val, y_pred))
        mlflow.log_metric("recall", recall_score(y_val, y_pred))
        mlflow.log_metric("f1", f1_score(y_val, y_pred))

        mlflow.sklearn.log_model(lr, "model")

    # -------------------------
    # XGBoost
    # -------------------------
    with mlflow.start_run(run_name="XGBoost"):
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42
        )

        xgb_model.fit(X_train, y_train)

        y_pred = xgb_model.predict(X_val)
        y_prob = xgb_model.predict_proba(X_val)[:, 1]

        mlflow.log_param("model", "XGBoost")
        mlflow.log_metric("roc_auc", roc_auc_score(y_val, y_prob))
        mlflow.log_metric("precision", precision_score(y_val, y_pred))
        mlflow.log_metric("recall", recall_score(y_val, y_pred))
        mlflow.log_metric("f1", f1_score(y_val, y_pred))

        mlflow.sklearn.log_model(xgb_model, "model")


if __name__ == "__main__":
    train_models()

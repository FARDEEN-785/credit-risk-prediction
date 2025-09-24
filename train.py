import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
import mlflow
import mlflow.sklearn

def main(args):
    # 1️⃣ Load dataset
    df = pd.read_csv(args.data_path, nrows=args.nrows, engine="python")
    print("Data loaded successfully! Shape:", df.shape)

    # 2️⃣ Select features and target
    features = ['loan_amnt', 'int_rate', 'annual_inc', 'dti',
                'open_acc', 'revol_bal', 'total_acc',
                'term', 'grade', 'emp_length', 'home_ownership', 'purpose']
    target = 'loan_status'
    df = df[features + [target]]

    # 3️⃣ Convert target to binary
    df[target] = df[target].apply(lambda x: 1 if x in ["Charged Off", "Default"] else 0)

    # 4️⃣ Split features / target
    X = df[features]
    y = df[target]

    # 5️⃣ Define preprocessing pipelines
    numeric_features = ['loan_amnt', 'int_rate', 'annual_inc', 'dti',
                        'open_acc', 'revol_bal', 'total_acc']
    categorical_features = ['term', 'grade', 'emp_length', 'home_ownership', 'purpose']

    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipe, numeric_features),
        ('cat', cat_pipe, categorical_features)
    ])

    # 6️⃣ Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 7️⃣ Handle class imbalance
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale = neg / pos
    print(f"Scale_pos_weight: {scale:.2f}")

    # 8️⃣ Full pipeline: preprocessing + model
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            scale_pos_weight=scale,
            eval_metric='logloss'
        ))
    ])

    # 9️⃣ Fit pipeline
    pipeline.fit(X_train, y_train)

    # 10️⃣ Evaluate
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    auc = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC Score: {auc:.4f}")

    # ✅ CORRECTED: Proper indentation for MLflow setup
    mlflow.set_tracking_uri(args.mlflow_uri)  # Now properly indented
    mlflow.set_experiment(args.experiment_name)  # Use the argument

    with mlflow.start_run(run_name=args.run_name):
        mlflow.sklearn.log_model(pipeline, artifact_path="model")
        mlflow.log_metric("roc_auc", auc)
        print(f"\nModel logged to MLflow. Run ID: {mlflow.active_run().info.run_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default='data/loan.csv')
    parser.add_argument('--nrows', type=int, default=None)
    parser.add_argument('--mlflow-uri', type=str, default='file:///D:/Credit_risk_project/mlruns')  # Changed default to local file
    parser.add_argument('--experiment-name', type=str, default='lendingclub_credit_risk_v2')
    parser.add_argument('--run-name', type=str, default='run1')
    args = parser.parse_args()

    main(args)
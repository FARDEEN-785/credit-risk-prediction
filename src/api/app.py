from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import pandas as pd


# -----------------------------
# Configuration
# -----------------------------
MODEL_NAME = "CreditRisk_XGBoost"

mlflow.set_tracking_uri("file:./mlruns")

model_uri = f"models:/{MODEL_NAME}/latest"
model = mlflow.pyfunc.load_model(model_uri)

app = FastAPI(title="Credit Risk Prediction API")


# -----------------------------
# Input Schema
# -----------------------------
class CreditRiskInput(BaseModel):
    person_age: float
    person_income: float
    person_emp_length: float
    loan_amnt: float
    loan_int_rate: float
    loan_percent_income: float
    cb_person_cred_hist_length: float

    person_home_ownership_OTHER: int
    person_home_ownership_OWN: int
    person_home_ownership_RENT: int

    loan_intent_EDUCATION: int
    loan_intent_HOMEIMPROVEMENT: int
    loan_intent_MEDICAL: int
    loan_intent_PERSONAL: int
    loan_intent_VENTURE: int

    loan_grade_B: int
    loan_grade_C: int
    loan_grade_D: int
    loan_grade_E: int
    loan_grade_F: int
    loan_grade_G: int

    cb_person_default_on_file_Y: int

    income_cushion: float
    loan_to_income: float
    is_new_employee: int
    credit_history_ratio: float


# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def health_check():
    return {"status": "Credit Risk API is running"}


@app.post("/predict")
def predict(input_data: CreditRiskInput):
    input_df = pd.DataFrame([input_data.dict()])
    input_array = input_df.values   # 👈 THIS is the key

    prediction = model.predict(input_array)[0]

    return {
        "default_prediction": int(prediction),
        "risk_label": "High Risk" if prediction == 1 else "Low Risk"
    }

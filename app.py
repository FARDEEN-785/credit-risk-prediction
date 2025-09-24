from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os

# --------------------------
# 1️⃣ Define Input Schema
# --------------------------
class LoanInput(BaseModel):
    loan_amnt: float
    int_rate: float
    annual_inc: float
    dti: float
    open_acc: int
    revol_bal: float
    total_acc: int
    term: str
    grade: str
    emp_length: str
    home_ownership: str
    purpose: str


# --------------------------
# 2️⃣ Initialize FastAPI
# --------------------------
app = FastAPI(title="Credit Risk Prediction API")

# --------------------------
# 3️⃣ Load Model
# --------------------------
MODEL_PATH = os.path.join("models", "model.pkl")

try:
    print(f"🔍 Loading model from: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print("❌ Error loading model:", e)
    model = None


# --------------------------
# 4️⃣ Prediction Endpoint
# --------------------------
@app.post("/predict")
def predict(input_data: LoanInput):
    if model is None:
        return {"error": "Model not loaded. Please check the model path."}

    df = pd.DataFrame([input_data.dict()])

    try:
        pred_class = int(model.predict(df)[0])
        prob = float(model.predict_proba(df)[:, 1][0])

        return {
            "predicted_class": pred_class,
            "default_probability": round(prob, 4),
            "status": "success"
        }
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}


# --------------------------
# 5️⃣ Health Check
# --------------------------
@app.get("/")
def root():
    return {"message": "Credit Risk Prediction API is running!"}

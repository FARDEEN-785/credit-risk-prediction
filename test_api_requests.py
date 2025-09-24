import requests

BASE_URL = "http://127.0.0.1:8001"

# 1️⃣ Health check
resp = requests.get(f"{BASE_URL}/")
print(resp.status_code, resp.json())

# 2️⃣ Prediction test
payload = {
    "loan_amnt": 10000,
    "int_rate": 12.5,
    "annual_inc": 50000,
    "dti": 18.2,
    "open_acc": 5,
    "revol_bal": 2000,
    "total_acc": 12,
    "term": "36 months",
    "grade": "B",
    "emp_length": "10+ years",
    "home_ownership": "MORTGAGE",
    "purpose": "credit_card"
}

resp = requests.post(f"{BASE_URL}/predict", json=payload)
print(resp.status_code, resp.json())

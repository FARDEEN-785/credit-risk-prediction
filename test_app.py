import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Credit Risk Prediction API is running!"}

def test_model_info():
    response = client.get("/model-info")
    assert response.status_code == 200
    assert "model_type" in response.json()

def test_prediction_valid():
    valid_data = {
        "loan_amnt": 10000, "int_rate": 12.5, "annual_inc": 50000, "dti": 18.2,
        "open_acc": 5, "revol_bal": 2000, "total_acc": 12,
        "term": "36 months", "grade": "B", "emp_length": "10+ years",
        "home_ownership": "MORTGAGE", "purpose": "credit_card"
    }
    
    response = client.post("/predict", json=valid_data)
    assert response.status_code == 200
    assert "predicted_class" in response.json()
    assert "default_probability" in response.json()

def test_prediction_missing_field():
    invalid_data = {
        "loan_amnt": 10000, "int_rate": 12.5, 
        # Missing other required fields
    }
    
    response = client.post("/predict", json=invalid_data)
    assert response.status_code == 422  # Validation error

def test_prediction_invalid_types():
    invalid_data = {
        "loan_amnt": "invalid_string",  # Should be number
        "int_rate": 12.5, "annual_inc": 50000, "dti": 18.2,
        "open_acc": 5, "revol_bal": 2000, "total_acc": 12,
        "term": "36 months", "grade": "B", "emp_length": "10+ years",
        "home_ownership": "MORTGAGE", "purpose": "credit_card"
    }
    
    response = client.post("/predict", json=invalid_data)
    assert response.status_code == 422
# Credit Risk Prediction System (MLOps)

A production-grade end-to-end Machine Learning project for predicting credit default risk, built with modern MLOps practices.  
The system covers the full ML lifecycle: data processing, model training, experiment tracking, model registry, and API-based deployment.

---

##  Project Overview

This project aims to predict whether a loan applicant is likely to default based on demographic, financial, and credit history features.  
Multiple models were trained and evaluated, with the best-performing model deployed as a REST API.

---

##  Key Features

- End-to-end ML pipeline (data → model → API)
- Feature engineering and preprocessing
- Model comparison and evaluation
- Experiment tracking and model versioning
- Production-ready inference API

---

##  Tech Stack & Skills

- **Programming**: Python  
- **Data Processing**: Pandas, NumPy  
- **Machine Learning**: Scikit-learn, XGBoost  
- **MLOps**: MLflow (tracking, registry)  
- **API**: FastAPI, Uvicorn  
- **Evaluation**: ROC-AUC, Precision, Recall  

---

##  Models Used

- Logistic Regression (baseline)
- XGBoost Classifier (final model)

**XGBoost achieved higher recall (~71%)**, making it more effective at identifying high-risk borrowers.

---

##  Experiment Tracking

MLflow is used to:
- Track experiments and metrics
- Compare multiple models
- Register the best-performing model

> Note: MLflow artifacts (`mlruns/`) are generated locally at runtime and are excluded from version control.

---

##  Project Structure

Credit_risk_2.0/
│
├── src/
│ ├── data/ # Data loading & preprocessing
│ ├── features/ # Feature engineering
│ ├── models/ # Training & evaluation scripts
│ └── api/ # FastAPI inference service
│
├── notebooks/ # Exploration & experiments
├── requirements.txt
├── README.md
└── .gitignore


---

## Running the Project

### 1. Install dependencies
```bash
pip install -r requirements.txt
2. Train models & log experiments
python src/models/train.py
3. Start the API server
uvicorn src.api.app:app --reload
4. Test the API
Open your browser at:

http://127.0.0.1:8000/docs
 Key Insight
Gradient-boosted models captured non-linear patterns in credit data better than linear models, significantly improving recall for default prediction — a critical metric in real-world credit risk systems.

 Future Improvements
Data drift and model monitoring

Automated retraining pipelines

CI/CD for ML workflows

Dockerized deployment

👤 Author
Fardeen
Aspiring AI/ML Engineer | Interested in applied ML, MLOps, and data-driven systems

# 📊 Customer Churn Prediction (ML + API)

## 🔍 Project Overview

This project predicts whether a customer will churn (leave the service) based on historical data.
It combines machine learning, feature engineering, model evaluation, and a deployed API for real-time predictions.

---

## 🚀 Key Features

* End-to-end ML pipeline (data → features → model → evaluation)
* Handling imbalanced data (`class_weight="balanced"`)
* Model evaluation with business-oriented metrics
* ROC AUC analysis and threshold tuning
* Feature importance analysis
* REST API built with FastAPI

---

## 🧠 Problem Statement

Customer churn is costly.
The goal is not just accuracy, but **detecting customers likely to leave (high recall)**.

---

## ⚙️ Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* Matplotlib
* FastAPI
* Joblib

---

## 🏗️ Project Structure

```bash
churn-prediction/
│
├── data.csv
│     
│
├── app/
│   ├── load_data.py
│   ├── features.py
│   ├── main.py
│   ├── models.py
│   ├── split.py
│   ├── target.py
│   ├── feature_importance.py
│   ├── evaluate.py
│   ├── encoding.py
│
├── api.py
│   
│
├── artifacts/
│   ├── model.pkl
│   ├── scaler.pkl
│   └── columns.pkl
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🔄 ML Pipeline

1. Load raw data
2. Clean & preprocess (handle missing values, encoding)
3. Feature engineering (one-hot encoding)
4. Train Logistic Regression model
5. Handle imbalance with `class_weight="balanced"`
6. Evaluate model performance
7. Save model & artifacts

---

## 📈 Model Performance

* Accuracy: ~0.73
* Precision: ~0.50
* Recall: ~0.79 ✅
* F1 Score: ~0.61
* ROC AUC: ~0.83

👉 Focus: **high recall to detect potential churn**

---

## 📊 Key Insights

* Dataset is imbalanced → default models ignore churn
* Using `class_weight="balanced"` significantly improves recall
* There is a trade-off between precision and recall
* Threshold tuning allows adapting model to business needs

### 💡 Business Insight

It is better to identify most churn cases (high recall), even at the cost of false positives.
Missing a churned customer is more expensive than unnecessary intervention.

---

## 🔮 API Usage

### ▶ Run API

```bash
uvicorn api:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

---

### 📥 Example Request

```json
{
  "data": {
    "tenure": 12,
    "MonthlyCharges": 70.5,
    "TotalCharges": 840,
    "gender_Male": 1,
    "Partner_Yes": 0,
    "Dependents_Yes": 0
  }
}
```

---

### 📤 Response

```json
{
  "churn": 1,
  "probability": 0.78
}
```

---

## 💾 Model Persistence

Model and preprocessing artifacts are saved using `joblib`:

* `model.pkl`
* `scaler.pkl`
* `columns.pkl`

---

## ⚡ How to Run

```bash
pip install -r requirements.txt
python main.py
```

---

## 📬 Author

Boris Matenco

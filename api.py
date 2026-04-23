from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# load model + scaler
model = joblib.load("artifacts/model.pkl")
scaler = joblib.load("artifacts/scaler.pkl")
columns = joblib.load("artifacts/columns.pkl")

# Input model
class Customer(BaseModel):
    data: dict

@app.get("/")
def home():
    return {"message:" "Churn prediction API works!"}

@app.post("/predict")
def predict(customer: Customer):
    """
    data = {
        "features": [....] # lista cech
    }
    """
    print(customer)
    input_dict = customer.data
    # Change to dataframe
    df = pd.DataFrame([input_dict])
    df = df.reindex(columns=columns, fill_value=0)
    # Scaling
    features_scaled = scaler.transform(df)
    # Prediction
    prediction = model.predict(features_scaled)[0]
    # Probability
    probability = model.predict_proba(features_scaled)[0][1]

    return {
        "churn" : int(prediction),
        "probability" : float(probability)
    }

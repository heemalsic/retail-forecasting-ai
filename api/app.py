from fastapi import FastAPI
import pandas as pd
import joblib

app = FastAPI()

model = joblib.load("models/xgb_model.pkl")

@app.get("/")
def home():
    return {"message": "Retail AI API running"}

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]

    return {"predicted_sales": float(prediction)}
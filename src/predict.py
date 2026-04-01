import joblib
import pandas as pd

model = joblib.load("models/xgb_model.pkl")

def predict(input_df):
    preds = model.predict(input_df)
    return preds
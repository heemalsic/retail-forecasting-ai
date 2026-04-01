import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

df = pd.read_csv("data/processed/final_dataset.csv")

features = [
    "Store", "Dept", "Temperature", "Fuel_Price",
    "CPI", "Unemployment", "IsHoliday",
    "Size", "year", "month", "week",
    "lag_1", "lag_4", "rolling_mean_4"
]

X = df[features]
y = df["Weekly_Sales"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

model = XGBRegressor(n_estimators=300, learning_rate=0.05)
model.fit(X_train, y_train)

preds = model.predict(X_val)

print("MAE:", mean_absolute_error(y_val, preds))

joblib.dump(model, "models/xgb_model.pkl")
print("Model saved!")
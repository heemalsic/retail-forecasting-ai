import streamlit as st
import pandas as pd
import joblib
import sys
import os

# Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.inventory import simulate_inventory
from src.llm_insights import generate_insights

# Load model
model = joblib.load("models/xgb_model.pkl")

st.set_page_config(page_title="Retail AI", layout="wide")

st.title("📦 Smart Retail Inventory Forecasting")
st.markdown("Upload raw retail data or use sample dataset")

# Sidebar controls
st.sidebar.header("⚙️ Controls")
initial_stock = st.sidebar.slider("Initial Stock", 10000, 100000, 50000)
restock_threshold = st.sidebar.slider("Restock Threshold", 5000, 30000, 10000)
restock_amount = st.sidebar.slider("Restock Amount", 10000, 100000, 40000)

# Upload
uploaded_file = st.file_uploader("Upload CSV (train/test format)")

# Load default if none
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    st.info("Using sample dataset")
    df = pd.read_csv("data/processed/final_dataset.csv")

# Ensure date exists
if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

# --- Feature Engineering ---
def create_features(df):
    df = df.sort_values(["Store", "Dept", "Date"])

    df["year"] = df["Date"].dt.year
    df["month"] = df["Date"].dt.month
    df["week"] = df["Date"].dt.isocalendar().week.astype(int)

    if "Weekly_Sales" in df.columns:
        df["lag_1"] = df.groupby(["Store", "Dept"])["Weekly_Sales"].shift(1)
        df["lag_4"] = df.groupby(["Store", "Dept"])["Weekly_Sales"].shift(4)

        df["rolling_mean_4"] = (
            df.groupby(["Store", "Dept"])["Weekly_Sales"]
            .shift(1)
            .rolling(4)
            .mean()
        )

    df = df.dropna()

    return df


df = create_features(df)

# --- Feature Selection ---
features = [
    "Store", "Dept", "Temperature", "Fuel_Price",
    "CPI", "Unemployment", "IsHoliday",
    "Size", "year", "month", "week",
    "lag_1", "lag_4", "rolling_mean_4"
]

# Ensure columns exist
missing_cols = [col for col in features if col not in df.columns]
if missing_cols:
    st.error(f"Missing required columns: {missing_cols}")
    st.stop()

# --- Filters ---
stores = df["Store"].unique()
depts = df["Dept"].unique()

selected_store = st.selectbox("Select Store", stores)
selected_dept = st.selectbox("Select Department", depts)

filtered_df = df[(df["Store"] == selected_store) & (df["Dept"] == selected_dept)].copy()

# --- Prediction ---
X = filtered_df[features]
preds = model.predict(X)

filtered_df["Predicted_Sales"] = preds

# --- Inventory Simulation ---
def simulate_inventory_custom(predictions):
    stock_levels = []
    current_stock = initial_stock

    for demand in predictions:
        current_stock -= demand

        if current_stock < restock_threshold:
            current_stock += restock_amount

        stock_levels.append(current_stock)

    return stock_levels


filtered_df["Stock_Level"] = simulate_inventory_custom(preds)

# --- Metrics ---
col1, col2, col3 = st.columns(3)

col1.metric("Avg Predicted Sales", f"{filtered_df['Predicted_Sales'].mean():.0f}")
col2.metric("Min Stock Level", f"{filtered_df['Stock_Level'].min():.0f}")
col3.metric(
    "Stockout Risk",
    "⚠️ High" if filtered_df["Stock_Level"].min() < restock_threshold else "✅ Low"
)

# --- Charts ---
st.subheader("📈 Forecast vs Inventory")

st.line_chart(
    filtered_df.set_index("Date")[["Predicted_Sales", "Stock_Level"]]
)

# --- Table ---
with st.expander("📊 View Data"):
    st.dataframe(filtered_df.tail(50))

# --- Alerts ---
if filtered_df["Stock_Level"].min() < restock_threshold:
    st.warning("⚠️ Stock is projected to drop below threshold. Consider increasing restock amount.")

st.success("✅ Forecast generated successfully!")

# =========================
# 🧠 LLM INSIGHTS SECTION
# =========================

st.subheader("🧠 AI Insights")

if st.button("Generate Insights"):
    with st.spinner("Analyzing trends using AI..."):
        try:
            insights = generate_insights(filtered_df)
            st.markdown(insights)
        except Exception as e:
            st.error(f"LLM Error: {str(e)}")
            st.info("Make sure GEMINI_API_KEY is set correctly.")
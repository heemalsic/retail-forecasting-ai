import os
import pandas as pd


def generate_insights(df: pd.DataFrame):
    try:
        from google import genai
    except ImportError:
        return "⚠️ google-genai not installed. Run: pip install google-genai"

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return "⚠️ GEMINI_API_KEY not set."

    try:
        client = genai.Client(api_key=api_key)

        # --- Stats ---
        avg_sales = df["Predicted_Sales"].mean()
        max_sales = df["Predicted_Sales"].max()
        min_stock = df["Stock_Level"].min()
        holiday_weeks = df[df["IsHoliday"] == 1].shape[0]

        prompt = f"""
        You are a retail AI analyst.

        Data summary:
        - Average predicted sales: {avg_sales:.2f}
        - Peak demand: {max_sales:.2f}
        - Minimum stock level: {min_stock:.2f}
        - Holiday weeks: {holiday_weeks}

        Explain:
        1. Key demand trends
        2. Causes (seasonality, holidays)
        3. Inventory risks
        4. Recommendations

        Keep it concise and actionable.
        """

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        return response.text

    except Exception as e:
        return f"⚠️ LLM error: {str(e)}"
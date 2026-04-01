import pandas as pd

def create_features(df):
    df = df.sort_values(["Store", "Dept", "Date"])

    df["year"] = df["Date"].dt.year
    df["month"] = df["Date"].dt.month
    df["week"] = df["Date"].dt.isocalendar().week.astype(int)

    # Lag features
    df["lag_1"] = df.groupby(["Store", "Dept"])["Weekly_Sales"].shift(1)
    df["lag_4"] = df.groupby(["Store", "Dept"])["Weekly_Sales"].shift(4)

    # Rolling mean
    df["rolling_mean_4"] = (
        df.groupby(["Store", "Dept"])["Weekly_Sales"]
        .shift(1)
        .rolling(4)
        .mean()
    )

    df = df.dropna()

    return df


if __name__ == "__main__":
    df = pd.read_csv("data/processed/final_dataset.csv", parse_dates=["Date"])
    df = create_features(df)

    df.to_csv("data/processed/final_dataset.csv", index=False)
    print("Features created!")
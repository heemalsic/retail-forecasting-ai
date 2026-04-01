import pandas as pd

def load_and_merge():
    train = pd.read_csv("data/raw/train.csv")
    features = pd.read_csv("data/raw/features.csv")
    stores = pd.read_csv("data/raw/stores.csv")

    # Robust date parsing
    train["Date"] = pd.to_datetime(train["Date"], errors="coerce")
    features["Date"] = pd.to_datetime(features["Date"], errors="coerce")

    # Merge datasets
    df = train.merge(features, on=["Store", "Date"], how="left")
    df = df.merge(stores, on="Store", how="left")

    return df


def clean_data(df):
    # Handle markdown nulls
    markdown_cols = ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]
    df[markdown_cols] = df[markdown_cols].fillna(0)

    # 🔥 Fix duplicate IsHoliday columns
    if "IsHoliday_x" in df.columns and "IsHoliday_y" in df.columns:
        df["IsHoliday"] = df["IsHoliday_x"]  # keep train version
        df = df.drop(columns=["IsHoliday_x", "IsHoliday_y"])

    elif "IsHoliday" not in df.columns:
        raise ValueError("❌ No IsHoliday column found after merge")

    # Convert to int
    df["IsHoliday"] = df["IsHoliday"].astype(int)

    return df


if __name__ == "__main__":
    df = load_and_merge()
    df = clean_data(df)

    print("Columns after cleaning:\n", df.columns)

    df.to_csv("data/processed/final_dataset.csv", index=False)
    print("✅ Saved processed data!")
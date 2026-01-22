import pandas as pd

def load_data(csv_path):
    df = pd.read_csv(csv_path)

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")

    return df

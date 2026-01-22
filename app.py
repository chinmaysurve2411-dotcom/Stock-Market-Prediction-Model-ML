import tkinter as tk
from tkinter import messagebox

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from src.preprocessing import load_data
from src.features import add_features
from src.models import train_models
from src.visualization import plot_predictions


def run_prediction():
    try:
        df = load_data("data/stock_data.csv")

        if "Close" not in df.columns:
            raise ValueError("CSV file must contain a 'Close' column")

        df = add_features(df)

        feature_cols = ["MA_5", "MA_10", "Return", "Volatility"]
        X = df[feature_cols]
        y = df["Close"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        lr_model, rf_model = train_models(X_train, y_train)

        lr_pred = lr_model.predict(X_test)
        rf_pred = rf_model.predict(X_test)

        lr_mae = mean_absolute_error(y_test, lr_pred)
        rf_mae = mean_absolute_error(y_test, rf_pred)

        result_label.config(
            text=(
                f"Linear Regression MAE: {lr_mae:.2f}\n"
                f"Random Forest MAE: {rf_mae:.2f}"
            )
        )

        plot_predictions(y_test.values, lr_pred, rf_pred)

    except Exception as e:
        messagebox.showerror("Error", str(e))


# ---------- UI ----------
root = tk.Tk()
root.title("Stock Market Prediction (Educational)")
root.geometry("420x280")

title = tk.Label(
    root,
    text="Stock Market Prediction",
    font=("Arial", 14, "bold")
)
title.pack(pady=10)

run_button = tk.Button(
    root,
    text="Run Prediction",
    font=("Arial", 12),
    command=run_prediction
)
run_button.pack(pady=20)

result_label = tk.Label(root, text="", font=("Arial", 11))
result_label.pack(pady=10)

footer = tk.Label(
    root,
    text="Educational use only",
    font=("Arial", 9)
)
footer.pack(side="bottom", pady=5)

root.mainloop()

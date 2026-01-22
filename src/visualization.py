import matplotlib.pyplot as plt

def plot_predictions(actual, lr_pred, rf_pred):
    plt.figure(figsize=(9, 4))
    plt.plot(actual, label="Actual")
    plt.plot(lr_pred, label="Linear Regression")
    plt.plot(rf_pred, label="Random Forest")
    plt.legend()
    plt.title("Stock Price Prediction (Educational)")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.tight_layout()
    plt.show()

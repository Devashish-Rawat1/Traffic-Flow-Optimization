
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# === Paths ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, "model")

xgb_path = os.path.join(MODEL_DIR, "xgboost_predictions.csv")
lstm_path = os.path.join(MODEL_DIR, "lstm_predictions.csv")

# === Load predictions ===
if not os.path.exists(xgb_path):
    raise FileNotFoundError(" XGBoost predictions not found. Run trainXgboost.py first.")
if not os.path.exists(lstm_path):
    raise FileNotFoundError(" LSTM predictions not found. Save lstm_predictions.csv first.")

xgb_preds = pd.read_csv(xgb_path)
lstm_preds = pd.read_csv(lstm_path)

# Align sizes
min_len = min(len(xgb_preds), len(lstm_preds))
xgb_preds = xgb_preds.head(min_len)
lstm_preds = lstm_preds.head(min_len)

actual = xgb_preds["Actual Congestion"].values
xgb_pred = xgb_preds["Predicted Congestion"].values
lstm_pred = lstm_preds["Predicted Congestion"].values

# === Compute Metrics ===
def compute_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    r2 = r2_score(y_true, y_pred)
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "MAPE": mape, "R²": r2}

lstm_metrics = compute_metrics(actual, lstm_pred)
xgb_metrics = compute_metrics(actual, xgb_pred)

# === 1️ Bar Plot: Metric Comparison ===
metrics = list(lstm_metrics.keys())
lstm_values = list(lstm_metrics.values())
xgb_values = list(xgb_metrics.values())

x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(9, 6))
plt.bar(x - width/2, lstm_values, width, label="LSTM", color="skyblue")
plt.bar(x + width/2, xgb_values, width, label="XGBoost", color="orange")

plt.xticks(x, metrics)
plt.ylabel("Metric Value")
plt.title("Model Performance Comparison: LSTM vs XGBoost")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()

bar_plot_path = os.path.join(MODEL_DIR, "LSTM_vs_XGBoost_Metrics.png")
plt.savefig(bar_plot_path)
plt.show()
print(f" Saved: {bar_plot_path}")

# === 2️ Line Plot: Actual vs Predicted Comparison ===
plt.figure(figsize=(10, 6))
plt.plot(actual, label="Actual Congestion", color="black", linewidth=2)
plt.plot(xgb_pred, label="XGBoost Predicted", color="orange", alpha=0.8)
plt.plot(lstm_pred, label="LSTM Predicted", color="blue", alpha=0.7)

plt.title("Actual vs Predicted Congestion Levels — LSTM vs XGBoost")
plt.xlabel("Sample Index")
plt.ylabel("Congestion Level")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()

line_plot_path = os.path.join(MODEL_DIR, "LSTM_vs_XGBoost_Predictions.png")
plt.savefig(line_plot_path)
plt.show()
print(f" Saved: {line_plot_path}")

# === Print Comparison Table ===
print("\n===== Model Performance Comparison =====")
print(f"{'Metric':<12}{'LSTM':>15}{'XGBoost':>15}")
print("-" * 42)
for name in metrics:
    print(f"{name:<12}{lstm_metrics[name]:>15.4f}{xgb_metrics[name]:>15.4f}")

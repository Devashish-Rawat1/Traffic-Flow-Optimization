import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent.parent 
sys.path.insert(0, str(root_dir))

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math
import joblib

from src.dataProcessing.preprocess2 import preprocess_data
from src.app.utils import get_project_root


def train_model():
    # Load preprocessed data
    df, scaler = preprocess_data()

    # Use the exact feature ordering that the scaler was fitted with (if available).
    # This ensures scaler.data_min_/data_max_ indices align with the columns we feed to the model.
    if hasattr(scaler, "feature_names_in_"):
        features = list(scaler.feature_names_in_)
    else:
        # fallback: pick numeric + one-hot weather columns as before
        features = [col for col in df.columns if col not in ["Date", "Area Name", "Road/Intersection Name"]]

    target_col = "Congestion Level"

    # sanity check: make sure target_col is in features (so we can read its min/max from scaler)
    if target_col not in features:
        raise ValueError(
            f"Expected target column '{target_col}' to be present in scaler.feature_names_in_. "
            "Scaler was fitted on a different set of features. Please ensure the scaler was fitted "
            "including the target column or update preprocess to include it."
        )

    # Build data and target using the same feature ordering scaler expects
    data = df[features].values
    target = df[target_col].values

    # Sequence length
    seq_length = 48
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])   # all features in scaler order
        y.append(target[i+seq_length])   # predict congestion only

    X = np.array(X)
    y = np.array(y)

    print("X shape:", X.shape, "y shape:", y.shape)

    # Chronological split
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Build model
    model = Sequential([
        LSTM(128, activation='tanh', input_shape=(seq_length, X.shape[2]), return_sequences=True),
        Dropout(0.3),
        LSTM(64, activation='tanh'),
        Dropout(0.3),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )

    # Save model + scaler
    root = get_project_root()
    model_dir = os.path.join(root, 'model')
    os.makedirs(model_dir, exist_ok=True)
    model.save(os.path.join(model_dir, 'model.h5'))
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))

    print(f"Model saved to {os.path.join(model_dir, 'model.h5')}")
    print(f"Scaler saved to {os.path.join(model_dir, 'scaler.pkl')}")

    # ------------------------
    # Evaluation
    # ------------------------
    y_pred = model.predict(X_test).flatten()

    # Use the same features ordering to find target index inside scaler internals
    congestion_idx = features.index(target_col)

    target_min = scaler.data_min_[congestion_idx]
    target_max = scaler.data_max_[congestion_idx]

    # Manual MinMax inverse transform (safe and explicit)
    y_test_inv = y_test * (target_max - target_min) + target_min
    y_pred_inv = y_pred * (target_max - target_min) + target_min

    mse = mean_squared_error(y_test_inv, y_pred_inv)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    mape = np.mean(np.abs((y_test_inv - y_pred_inv) / (y_test_inv + 1e-8))) * 100
    r2 = r2_score(y_test_inv, y_pred_inv)

    print("Evaluation Metrics:")
    print(f"MSE:   {mse:.4f}")
    print(f"RMSE:  {rmse:.4f}")
    print(f"MAE:   {mae:.4f}")
    print(f"MAPE:  {mape:.2f}%")
    print(f"R²:    {r2:.4f}")

    # ------------------------
    # Predict with last sequence
    # ------------------------
    last_seq = data[-seq_length:]
    last_seq = last_seq.reshape(1, seq_length, X.shape[2])
    pred_scaled = model.predict(last_seq).flatten()

    # manual MinMax inverse transform for the single-step prediction
    pred_inv = pred_scaled * (target_max - target_min) + target_min
    print("Predicted next congestion level:", pred_inv[0])

    # Save LSTM predictions to CSV for comparison (inverse-transformed values)
    import pandas as pd
    results_df = pd.DataFrame({
      "Actual Congestion": y_test_inv,
      "Predicted Congestion": y_pred_inv
    })

    results_path = os.path.join(model_dir, "lstm_predictions.csv")
    results_df.to_csv(results_path, index=False)

    print(f"LSTM predictions saved to: {results_path}")

    return model, history


if __name__ == "__main__":
    train_model()

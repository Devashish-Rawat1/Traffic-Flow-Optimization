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

    # Dynamically pick all features (numeric + one-hot weather)
    features = [col for col in df.columns if col not in ["Date", "Area Name", "Road/Intersection Name"]]
    target_col = "Congestion Level"

    data = df[features].values
    target = df[target_col].values

    # Sequence length
    seq_length = 48
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])   # all features
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

    # Inverse-transform congestion level only (fix applied)
    congestion_idx = features.index(target_col)
    n_features = scaler.n_features_in_

    # Pad arrays to match scaler's expected feature size
    y_test_padded = np.zeros((len(y_test), n_features))
    y_pred_padded = np.zeros((len(y_pred), n_features))

    y_test_padded[:, congestion_idx] = y_test
    y_pred_padded[:, congestion_idx] = y_pred

    y_test_inv = scaler.inverse_transform(y_test_padded)[:, congestion_idx]
    y_pred_inv = scaler.inverse_transform(y_pred_padded)[:, congestion_idx]

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

    pred_padded = np.zeros((len(pred_scaled), n_features))
    pred_padded[:, congestion_idx] = pred_scaled

    pred_inv = scaler.inverse_transform(pred_padded)[:, congestion_idx]

    print("Predicted next congestion level:", pred_inv[0])
    
    # Save LSTM predictions to CSV for comparison
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

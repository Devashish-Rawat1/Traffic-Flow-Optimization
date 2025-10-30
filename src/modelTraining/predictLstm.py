import os
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
#from src.app.utils import get_project_root

def get_project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


def predict_next_congestion(input_sequence):
    """
    Predict the next congestion level given the last 6 values.
    """
    # Load project root
    root_dir = get_project_root()
    model_dir = os.path.join(root_dir, "model")

    # Load the trained model and scaler
    model_path = os.path.join(model_dir, "model.h5")
    scaler_path = os.path.join(model_dir, "scaler.pkl")

    model = load_model(model_path)
    scaler = joblib.load(scaler_path)

    # Scale the input sequence using the same scaler
    input_scaled = scaler.transform(np.array(input_sequence).reshape(-1, 1))

    # Reshape for LSTM [samples, timesteps, features]
    input_scaled = input_scaled.reshape(1, len(input_sequence), 1)

    # Predict
    prediction_scaled = model.predict(input_scaled)

    # Inverse transform to get original congestion level
    prediction = scaler.inverse_transform(prediction_scaled)

    return prediction[0][0]

if __name__ == "__main__":
    # Example: use last 6 congestion values from dataset
    import pandas as pd

    dataset_path = os.path.join(get_project_root(), "data","raw", "Banglore_traffic_Dataset.csv")
    df = pd.read_csv(dataset_path)

    # Take last 6 congestion levels from dataset
    last_6 = df["Congestion Level"].values[-6:]

    print("Last 6 values:", last_6)

    next_value = predict_next_congestion(last_6)
    print("Predicted next congestion level:", next_value)

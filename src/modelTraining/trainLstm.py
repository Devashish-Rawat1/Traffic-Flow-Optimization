import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent.parent 
sys.path.insert(0, str(root_dir))
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from src.dataProcessing.preprocess import preprocess_data
from src.app.utils import ensure_directory_exists, get_project_root
import joblib

def train_model():
    """Train LSTM model on congestion data"""
    # Load and preprocess data
    df, scaler = preprocess_data()
    
    # Prepare data for LSTM
    target = df['Congestion Level'].values
    X = [] # features
    y = [] # labels
    seq_length = 6
    
    for i in range(len(target) - seq_length):
        X.append(target[i:i+seq_length])
        y.append(target[i+seq_length])
        
    X = np.array(X)
    y = np.array(y)
    
    # Reshape for LSTM [samples(no. of windows), timesteps(6), features(1)]
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    
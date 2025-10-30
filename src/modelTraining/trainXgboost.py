import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.dataProcessing.preprocess_xgboost import preprocess_for_xgboost


def train_xgboost_model():
    # 1️ Load & preprocess dataset
    file_path = os.path.join(root_dir, "data", "raw", "Banglore_traffic_Dataset.csv")
    X, y, df = preprocess_for_xgboost(file_path)

    # 2️ Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )

    # 3️ Convert to DMatrix (for advanced training API)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # 4️ Define parameters
    params = {
        "objective": "reg:squarederror",
        "learning_rate": 0.05,
        "max_depth": 8,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 1.5,
        "min_child_weight": 3,
        "random_state": 42,
        "tree_method": "hist",
        "eval_metric": "rmse"
    }

    print("\nTraining XGBoost model (core API with early stopping)...")

    # 5️ Train using xgb.train (supports early stopping)
    evals_result = {}
    evals = [(dtrain, "train"), (dtest, "eval")]
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=500,
        evals=evals,
        early_stopping_rounds=30,
        verbose_eval=50,
        evals_result=evals_result
    )

    # 6️ Evaluate performance
    y_pred = model.predict(dtest)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100
    r2 = r2_score(y_test, y_pred)

    print("\n Evaluation Metrics:")
    print(f"MSE:   {mse:.4f}")
    print(f"RMSE:  {rmse:.4f}")
    print(f"MAE:   {mae:.4f}")
    print(f"MAPE:  {mape:.2f}%")
    print(f"R²:    {r2:.4f}")

    # 7️ Save model
    model_dir = os.path.join(root_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "xgboost_model.json")
    model.save_model(model_path)
    print(f"\n Model saved at: {model_path}")

    # 8️ Plot feature importance
    plt.figure(figsize=(10, 8))
    plot_importance(model, max_num_features=15, importance_type="gain")
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "xgboost_feature_importance.png"))
    plt.close()

    # 9️ Plot training curves (Train vs Eval RMSE)
    train_rmse = evals_result["train"]["rmse"]
    eval_rmse = evals_result["eval"]["rmse"]
    rounds = range(1, len(train_rmse) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(rounds, train_rmse, label="Train RMSE", color="blue")
    plt.plot(rounds, eval_rmse, label="Validation RMSE", color="orange")
    plt.xlabel("Boosting Rounds")
    plt.ylabel("RMSE")
    plt.title("Training vs Validation RMSE (XGBoost)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "xgboost_training_curve.png"))
    plt.show()

    # 10 Actual vs Predicted plot
    plt.figure(figsize=(8, 5))
    plt.plot(range(len(y_test)), y_test, label="Actual", color="green")
    plt.plot(range(len(y_pred)), y_pred, label="Predicted", color="red", alpha=0.7)
    plt.title("Actual vs Predicted Congestion Level")
    plt.xlabel("Test Sample Index")
    plt.ylabel("Congestion Level")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "xgboost_actual_vs_predicted.png"))
    plt.show()

    # 11 Save predictions to CSV
    results_df = pd.DataFrame({
        "Actual Congestion": y_test,
        "Predicted Congestion": y_pred
    })
    results_path = os.path.join(model_dir, "xgboost_predictions.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\n Predictions saved to: {results_path}")

    # 12 Predict latest congestion
    latest_input = xgb.DMatrix(X.iloc[[-1]])
    next_pred = model.predict(latest_input)[0]
    print(f"\nPredicted congestion level for latest record: {next_pred:.2f}")

    return model


if __name__ == "__main__":
    train_xgboost_model()

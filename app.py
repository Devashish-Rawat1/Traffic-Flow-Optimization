import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import tensorflow as tf
import plotly.express as px
import plotly.graph_objects as go
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime

# -----------------------
# Project paths (absolute)
# -----------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
LSTM_MODEL_PATH = os.path.join(MODEL_DIR, "model.h5")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
XGB_MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_model.json")
XGB_PRED_PATH = os.path.join(MODEL_DIR, "xgboost_predictions.csv")
LSTM_PRED_PATH = os.path.join(MODEL_DIR, "lstm_predictions.csv")
XGB_TRAIN_HISTORY_CSV = os.path.join(MODEL_DIR, "xgboost_training_history.csv")
XGB_FEATURE_IMPORTANCE_IMG = os.path.join(MODEL_DIR, "xgboost_feature_importance.png")
XGB_TRAINING_CURVE_IMG = os.path.join(MODEL_DIR, "xgboost_training_curve.png")

# -------------------------------
# Utility: metrics
# -------------------------------
def compute_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    r2 = r2_score(y_true, y_pred)
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2}

# -------------------------------
# Load models (cached)
# -------------------------------
@st.cache_resource
def load_models():
    lstm = tf.keras.models.load_model(LSTM_MODEL_PATH, compile=False)
    scaler = joblib.load(SCALER_PATH)
    xgb_model = xgb.Booster()
    xgb_model.load_model(XGB_MODEL_PATH)
    return lstm, scaler, xgb_model

# -------------------------------
# LSTM helper (unchanged)
# -------------------------------
def lstm_predict_single(lstm, scaler, df, seq_len=48):
    values = df.values
    if len(values) < seq_len:
        return None
    last_seq = values[-seq_len:].reshape(1, seq_len, df.shape[1])
    pred_scaled = lstm.predict(last_seq).flatten()
    return float(pred_scaled[0])

# -------------------------------
# Plotly template (fixed)
# -------------------------------
def plotly_template():
    return "plotly"

# -------------------------------
# XGBoost Feature Importance
# -------------------------------
def get_xgb_importances(booster):
    importance_types = ["weight", "gain", "cover"]
    data = {}
    for itype in importance_types:
        try:
            score = booster.get_score(importance_type=itype)
        except Exception:
            score = {}
        data[itype] = score

    features = set()
    for d in data.values():
        features.update(d.keys())

    rows = []
    for f in sorted(features):
        rows.append({
            "feature": f,
            "weight": data["weight"].get(f, 0.0),
            "gain": data["gain"].get(f, 0.0),
            "cover": data["cover"].get(f, 0.0)
        })

    return pd.DataFrame(rows).sort_values(by="gain", ascending=False).reset_index(drop=True)

# -------------------------------
# UI Start
# -------------------------------
st.set_page_config(page_title="Traffic Congestion Dashboard", layout="wide")

with st.sidebar:
    st.title("Controls")
    with st.expander("Plot Controls"):
        num_plot_samples = st.slider("Samples to plot", min_value=100, max_value=2000, value=500, step=50)
    with st.expander("Files / Status"):
        st.write("XGBoost predictions:", os.path.exists(XGB_PRED_PATH))
        st.write("LSTM predictions:", os.path.exists(LSTM_PRED_PATH))
        st.write("Training History CSV:", os.path.exists(XGB_TRAIN_HISTORY_CSV))

st.title("🚦 Traffic Congestion: LSTM vs XGBoost (Interactive)")

with st.spinner("Loading models..."):
    lstm, scaler, xgb_model = load_models()

# -------------------------------
# Load Predictions
# -------------------------------
xgb_preds = pd.read_csv(XGB_PRED_PATH) if os.path.exists(XGB_PRED_PATH) else pd.DataFrame()
lstm_preds = pd.read_csv(LSTM_PRED_PATH) if os.path.exists(LSTM_PRED_PATH) else pd.DataFrame()

with st.expander("Preview prediction CSVs"):
    c1, c2 = st.columns(2)
    c1.dataframe(xgb_preds.head(), height=200)
    c2.dataframe(lstm_preds.head(), height=200)

if xgb_preds.empty or lstm_preds.empty:
    st.warning("Prediction files missing.")
else:
    min_len = min(len(xgb_preds), len(lstm_preds))
    xgb_trim = xgb_preds.head(min_len).reset_index(drop=True)
    lstm_trim = lstm_preds.head(min_len).reset_index(drop=True)

    y_true = xgb_trim["Actual Congestion"].values
    lstm_vals = lstm_trim["Predicted Congestion"].values
    xgb_vals = xgb_trim["Predicted Congestion"].values

    lstm_metrics = compute_metrics(y_true, lstm_vals)
    xgb_metrics = compute_metrics(y_true, xgb_vals)

    # KPIs
    st.subheader("📌 Key Performance Indicators")
    a, b, c, d = st.columns(4)
    a.metric("LSTM RMSE", round(lstm_metrics["RMSE"], 3))
    b.metric("XGB RMSE", round(xgb_metrics["RMSE"], 3))
    c.metric("LSTM MAE", round(lstm_metrics["MAE"], 3))
    d.metric("XGB MAE", round(xgb_metrics["MAE"], 3))

    best = "LSTM" if lstm_metrics["RMSE"] < xgb_metrics["RMSE"] else "XGBoost"
    st.success(f"🏆 Best Model: **{best}**")

    tab1, tab2, tab3, tab4 = st.tabs(["Metrics", "Interactive Plots", "XGBoost Analysis", "Raw Data"])

    # ------------------- METRICS TABLE -------------------
    with tab1:
        dfm = pd.DataFrame({
            "Metric": lstm_metrics.keys(),
            "LSTM": lstm_metrics.values(),
            "XGBoost": xgb_metrics.values()
        })
        st.table(dfm.set_index("Metric"))

    # ------------------- INTERACTIVE PLOTS -------------------
    with tab2:
        plot_n = min(num_plot_samples, min_len)
        idx = np.arange(plot_n)

        df_plot = pd.DataFrame({
            "index": idx,
            "Actual": y_true[:plot_n],
            "LSTM": lstm_vals[:plot_n],
            "XGBoost": xgb_vals[:plot_n]
        })

        fig_xgb = px.line(df_plot, x="index", y=["Actual", "XGBoost"],
                          title="XGBoost: Actual vs Predicted",
                          template=plotly_template())
        st.plotly_chart(fig_xgb, use_container_width=True)

        fig_lstm = px.line(df_plot, x="index", y=["Actual", "LSTM"],
                           title="LSTM: Actual vs Predicted",
                           template=plotly_template())
        st.plotly_chart(fig_lstm, use_container_width=True)

    # ------------------- XGBOOST ANALYSIS -------------------
    with tab3:
        st.subheader("Feature Importance")

        try:
            fi = get_xgb_importances(xgb_model)
            top_k = st.slider("Top features", 5, min(50, len(fi)), 15)
            fi_top = fi.head(top_k)

            fig_gain = px.bar(fi_top.sort_values("gain"), x="gain", y="feature",
                              orientation="h",
                              title="Gain",
                              template=plotly_template())
            st.plotly_chart(fig_gain, use_container_width=True)

            fig_weight = px.bar(fi_top.sort_values("weight"), x="weight", y="feature",
                                orientation="h",
                                title="Weight",
                                template=plotly_template())
            st.plotly_chart(fig_weight, use_container_width=True)

            fig_cover = px.bar(fi_top.sort_values("cover"), x="cover", y="feature",
                               orientation="h",
                               title="Cover",
                               template=plotly_template())
            st.plotly_chart(fig_cover, use_container_width=True)

        except Exception as e:
            st.error(str(e))

        st.subheader("XGBoost Training Curves")

        if os.path.exists(XGB_TRAIN_HISTORY_CSV):
            hist = pd.read_csv(XGB_TRAIN_HISTORY_CSV)
            numeric_cols = hist.select_dtypes(include=[np.number]).columns

            if len(numeric_cols) >= 2:
                fig_hist = px.line(hist, y=numeric_cols[:2],
                                   title="Training History",
                                   template=plotly_template())
                st.plotly_chart(fig_hist, use_container_width=True)
        elif os.path.exists(XGB_TRAINING_CURVE_IMG):
            st.image(XGB_TRAINING_CURVE_IMG)
        else:
            st.info("No training history found.")

    # ------------------- RAW DATA -------------------
    with tab4:
        dfc = pd.DataFrame({
            "Actual": y_true,
            "XGB Pred": xgb_vals,
            "LSTM Pred": lstm_vals
        })
        st.dataframe(dfc.head(200), height=350)
        st.download_button("Download CSV", dfc.to_csv(index=False), "comparison.csv")

st.write("---")
st.caption("Built with Streamlit · Generated " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

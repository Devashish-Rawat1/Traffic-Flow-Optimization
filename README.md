# 🚦 Traffic Congestion Prediction using Machine Learning

This project focuses on predicting **traffic congestion levels** using
historical traffic data from **Bangalore, India**. The goal was to
compare deep learning (LSTM) and traditional ML (XGBoost) approaches,
evaluate their performance, and deploy the best model using a
**Streamlit web application**.

## 📌 Project Overview

The dataset used consists of: - 8,936 records - 16 traffic-related
features - Multiple roads and intersections across Bangalore

## 🤖 Models Used

### LSTM

Not suitable due to: - Small dataset size - Tabular, non-sequential
features - Underfitting

### XGBoost

-   Best-performing model
-   Handles structured tabular data well
-   Strong evaluation metrics

## 🧪 Evaluation Metrics

-   MAE
-   RMSE
-   R² Score

## 💻 Streamlit App

Run the app:

``` bash
pip install -r requirements.txt
streamlit run app.py
```

## 📂 Project Structure

    ├── data/raw
    ├── documents/
        ├──notes/
        ├──ppt/
        └──research_paper/
    ├── model/
        ├── lstm_predictions.csv
        └── xgboost_predictions.csv
    ├── notebooks
    ├── screenshorts
    ├── src/
        ├── app/
        ├── dataPreprocessing/
            ├── preprocess2.py
            └── preprocess_xgboost.py
        ├── modelTraining/
            ├── trainLstm2.py
            ├── trainXgboost.py
            └── comparePerformancePlots.py
    ├── app.py
    ├── requirements.txt
    └── README.md

## 🚀 Future Improvements

-   Real-time data integration
-   Route optimization
-   Geospatial visualization
-   Cloud deployment

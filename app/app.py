import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import pickle
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sns.set_palette("YlGnBu")

def preprocess_data(df, target_variable):
    try:
        if 'date' not in df.columns or target_variable not in df.columns:
            st.error(f"Error: The uploaded file must contain 'date' and '{target_variable}' columns.")
            return None
        
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date', target_variable])
        df = df.sort_values(by='date')
        df['dayofweek'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
        
        # Renaming the target variable to 'target'
        df = df.rename(columns={target_variable: 'target'})
        
        # Adding lag features
        lags = [1, 2, 3, 7, 14, 30]
        for lag in lags:
            df[f'lag_{lag}'] = df['target'].shift(lag)
        
        df = df.dropna()
        return df
    except Exception as e:
        st.error(f"An error occurred during data preprocessing: {e}")
        return None

def plot_predictions(df, forecast_date, prediction, target_variable):
    df = df.sort_values(by='date')
    plt.figure(figsize=(12, 6))
    
    train_series = df[df['date'] < forecast_date]
    test_series = df[df['date'] >= forecast_date]
    
    sns.lineplot(data=train_series, x='date', y='target', label='Train Data', color='blue')
    sns.lineplot(data=test_series, x='date', y='target', label='Test Data', color='green')
    
    plt.scatter([forecast_date], [prediction], color='red', s=150, label='Prediction', edgecolors='black', linewidth=1.5)
    
    plt.xlabel("Date", fontsize=14)
    plt.ylabel(f"{target_variable} Value", fontsize=14)
    plt.title(f"{target_variable} Forecast for Selected Period", fontsize=16, fontweight='bold')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    st.pyplot(plt)

st.title("Time Series Forecasting with XGBoost")

# User input for target variable
target_variable = st.text_input("Enter the name of your target variable:", value="unit_sales")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        if df is None or df.empty:
            st.error("Error: Uploaded file is empty or not readable.")
        else:
            df = preprocess_data(df, target_variable)
            if df is not None:
                st.write("### Processed Data Sample (First 20 Rows)")
                st.dataframe(df.head(20))
                
                if st.button("Train Model"):
                    model = train_model(df)
                    if model:
                        st.session_state['model'] = model
                        st.success("Model trained successfully!")
                        
                        with open("trained_model.xgb", "rb") as f:
                            st.download_button("Download XGBoost Model", f, file_name="trained_model.xgb")
                        
                        with open("trained_model.pkl", "rb") as f:
                            st.download_button("Download Pickle Model", f, file_name="trained_model.pkl")
                
                # Select forecast date within dataset range
                min_date = df['date'].min().date()
                max_date = df['date'].max().date()
                forecast_date = st.date_input("Select a forecast date", min_value=min_date, max_value=max_date)
                
                if st.button("Predict"):
                    if 'model' in st.session_state:
                        prediction = st.session_state['model'].predict(df[df['date'] == pd.to_datetime(forecast_date)][['dayofweek', 'month', 'year'] + [f'lag_{lag}' for lag in [1, 2, 3, 7, 14, 30]]])
                        if prediction is not None:
                            st.write(f"Predicted {target_variable} for {forecast_date}: {prediction[0]:.2f}")
                            plot_predictions(df, forecast_date, prediction[0], target_variable)
                        else:
                            st.error("Prediction failed. Check input data and try again.")
                    else:
                        st.error("Please train the model first.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
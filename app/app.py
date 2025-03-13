
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import pickle
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sns.set(style="whitegrid")

# Streamlit App Title
st.title("Time Series Forecasting with XGBoost")

# File Format Requirements
st.write("**File Format Requirements:**")
st.markdown("- The uploaded CSV file should contain at least the following columns:\n  - `date`: The date of the recorded sales (YYYY-MM-DD format).\n  - A target variable column (e.g., `unit_sales`).")

# User input for target variable
target_variable = st.text_input("Enter the name of your target variable:", value="unit_sales")

# File Upload Section
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

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
        
        df = df.rename(columns={target_variable: 'target'})
        
        # Keep only necessary lags and add moving average
        lags = [1, 7, 14]
        for lag in lags:
            df[f'lag_{lag}'] = df['target'].shift(lag)
        
        df['rolling_mean_7'] = df['target'].rolling(window=7, min_periods=1).mean()
        
        df = df.dropna()
        return df
    except Exception as e:
        st.error(f"An error occurred during data preprocessing: {e}")
        return None

def calculate_metrics(model, X_train, X_test, y_train, y_test):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    metrics = {
        "Train RMSE": np.sqrt(mean_squared_error(y_train, y_train_pred)),
        "Test RMSE": np.sqrt(mean_squared_error(y_test, y_test_pred)),
        "Train MAE": mean_absolute_error(y_train, y_train_pred),
        "Test MAE": mean_absolute_error(y_test, y_test_pred),
        "Train R² Score": r2_score(y_train, y_train_pred),
        "Test R² Score": r2_score(y_test, y_test_pred)
    }
    return metrics

def plot_predictions(df, forecast_date, prediction, target_variable):
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='date', y='target', label='Train Data', color='blue')
    sns.lineplot(data=df.iloc[-len(df)//5:], x='date', y='target', label='Test Data', color='green')
    sns.scatterplot(x=[forecast_date], y=[prediction], color='red', s=150, label='Prediction', edgecolor='black', linewidth=1.5)
    plt.xlabel("Date", fontsize=14)
    plt.ylabel(f"{target_variable} Value", fontsize=14)
    plt.title(f"{target_variable} Forecast on {forecast_date.date()}", fontsize=16, fontweight='bold')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(plt)

def train_model(df):
    try:
        if df is None:
            return None
        
        features = ['dayofweek', 'month', 'year'] + [f'lag_{lag}' for lag in [1, 7, 14]] + ['rolling_mean_7']
        target = 'target'
        
        X = df[features]
        y = df[target]
        
        tscv = TimeSeriesSplit(n_splits=5)
        train_indices, test_indices = list(tscv.split(X))[-1]
        X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
        y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
        
        param_grid = {
            'n_estimators': [50, 100, 150, 200, 250, 300],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [2, 3, 4, 5, 6],
            'reg_alpha': [0.0, 0.1, 0.5],
            'reg_lambda': [0.0, 0.1, 0.5]
        }
        
        model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
        
        st.info("Please wait, your file is being processed to find the best model...")
        grid_search = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error', cv=tscv, verbose=1, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        st.session_state['model'] = best_model
        st.session_state['best_params'] = best_params
        st.session_state['trained'] = True
        st.session_state['df'] = df
        st.session_state['metrics'] = calculate_metrics(best_model, X_train, X_test, y_train, y_test)
        
        return best_model
    except Exception as e:
        st.error(f"An error occurred during model training: {e}")
        return None

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df = preprocess_data(df, target_variable)
    
    if df is not None:
        st.write("### Processed Data Sample (First 20 Rows)")
        st.dataframe(df.head(20), height=150)
        
        if st.button("Train Model"):
            train_model(df)
        
        if 'trained' in st.session_state and st.session_state['trained']:
            st.write("### Model Performance Metrics")
            st.write(f"- **Train RMSE:** {st.session_state['metrics']['Train RMSE']:.2f} | **Test RMSE:** {st.session_state['metrics']['Test RMSE']:.2f}")
            st.write(f"- **Train MAE:** {st.session_state['metrics']['Train MAE']:.2f} | **Test MAE:** {st.session_state['metrics']['Test MAE']:.2f}")
            st.write(f"- **Train R² Score:** {st.session_state['metrics']['Train R² Score']:.2f} | **Test R² Score:** {st.session_state['metrics']['Test R² Score']:.2f}")
            
            st.write("### Model Hyperparameters")
            for param, value in st.session_state['best_params'].items():
                st.write(f"- **{param.capitalize()}**: {value}")
            
            min_date = df['date'].min().date()
            max_date = df['date'].max().date()
            forecast_date = st.date_input("Select a forecast date", min_value=min_date, max_value=max_date)
            
            if st.button("Predict"):
                model = st.session_state['model']
                forecast_date = pd.to_datetime(forecast_date)
                prediction = model.predict(df[['dayofweek', 'month', 'year'] + [f'lag_{lag}' for lag in [1, 7, 14]] + ['rolling_mean_7']].iloc[-1:])
                
                if prediction is not None:
                    st.write(f"Predicted {target_variable} for {forecast_date.date()}: {prediction[0]:.2f}")
                    plot_predictions(df, forecast_date, prediction[0], target_variable)
                else:
                    st.error("Prediction failed. Check input data and try again.")

            st.write("### Download Trained Model")
            with open("trained_model.xgb", "rb") as f:
                st.download_button("Download XGBoost Model", f, file_name="trained_model.xgb")
            
            with open("trained_model.pkl", "rb") as f:
                st.download_button("Download Pickle Model", f, file_name="trained_model.pkl")
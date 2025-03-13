# Streamlit Time Series Forecasting App

This repository contains a Streamlit-based application for time series forecasting using XGBoost. It allows users to upload a CSV file, train a model, visualize the results, and make predictions.

## Features
- Upload a dataset with `date` and a target variable column
- Preprocesses data and adds time-based features
- Trains an XGBoost model using time series cross-validation
- Displays model performance metrics (MSE, MAE, R²)
- Generates forecasts and visualizes the results
- Allows downloading the trained model

## Setting Up the Project from Scratch

Follow these steps to create and deploy this Streamlit application from scratch.

### **1. Delete any existing repository (if applicable)**
```bash
rm -rf streamlit-time-series
```

### **2. Create a new repository**
```bash
mkdir streamlit-time-series
cd streamlit-time-series
```

### **3. Set up the project structure**
```bash
mkdir Input
mkdir Output
mkdir app
mkdir notebooks
mkdir scripts
touch requirements.txt
touch README.md
touch app/app.py
```

### **4. Move the training dataset to the `Input` folder**
```bash
mv /path/to/downloaded/training_data.csv Input/
```

### **5. Create a virtual environment and install dependencies**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install streamlit pandas numpy matplotlib seaborn xgboost scikit-learn
pip freeze > requirements.txt
```

### **6. Initialize Git and make the first commit**
```bash
git init
git add .
git commit -m "Initial project setup"
```

### **7. Create a new repository on GitHub and push the code**
```bash
git remote add origin https://github.com/your-username/your-repository.git
git push -u origin main
```

### **8. Run the Streamlit application locally**
```bash
streamlit run app/app.py
```
Your application will be available at `http://localhost:8501/`.

---

## Deploying to Streamlit Community Cloud

### **9. Deploy the app on Streamlit Cloud**
1. Go to [Streamlit Cloud](https://share.streamlit.io/)
2. Sign in with your GitHub account
3. Click `New app`
4. Select your GitHub repository
5. Set the main file path to `app/app.py`
6. Click `Deploy`

### **10. Your Streamlit App is Live!**
Once deployed, you will get a live URL like:
```
https://your-app-name.streamlit.app
```

---

## Updating Your App
To make changes to your app:
```bash
git add .
git commit -m "Updated app"
git push origin main
```
Your Streamlit Cloud app will update automatically!

---

## Troubleshooting

### Problem: "ModuleNotFoundError: No module named 'streamlit'"
Run:
```bash
pip install -r requirements.txt
```

### Problem: "Command not found: streamlit"
If using `venv`, ensure it is activated:
```bash
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### Problem: App does not deploy correctly on Streamlit Cloud
- Ensure `requirements.txt` is in the repository
- Check for errors in Streamlit Cloud logs

---

## License
This project is open-source and available under the MIT License.

---

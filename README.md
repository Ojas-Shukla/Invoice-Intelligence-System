# 📊 Invoice Intelligence System

An end-to-end Machine Learning project that predicts freight costs and identifies potentially risky invoices using supervised learning techniques.

The project includes:

- Freight Cost Prediction
- Invoice Risk Detection
- Batch Invoice Processing
- Interactive Streamlit Dashboard
- SQLite Database Integration
- Model Evaluation and Selection

---
## 🚀 Live Demo

🔗 **Streamlit App**: https://invoice-intelligence-system-ki3twtwru9ez2fnnehutch.streamlit.app/

---

## 🚀 Features

### 1. Freight Cost Prediction

Predict freight charges based on:

- Invoice Amount
- Invoice Quantity

Supported Regression Models:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

The best-performing model is automatically selected and saved.

---

### 2. Invoice Risk Flagging

Classify invoices as:

- ✅ Low Risk
- 🚨 High Risk

Features Used:

- Invoice Quantity
- Invoice Dollars
- Freight Cost
- Total Item Quantity
- Total Item Dollars

---

### 3. Batch Processing

Upload CSV or Excel files and:

- Predict Freight Costs
- Detect Risky Invoices
- Calculate Risk Probability
- Download Processed Results

---

### 4. Interactive Dashboard

Built with Streamlit and Plotly.

Includes:

- Real-time predictions
- Interactive charts
- Risk visualizations
- Model status monitoring

---

## 🏗️ Project Structure

```bash
Invoice-Intelligence/
│
├── app2.py
├──Freight_Prediction/
|   ├── data_preprocessing.py
│   ├── model_evaluation.py
│   └── train.py
|
├──invoice_flagging/
|   ├── data_preprocessing.py
│   ├── model_evaluation.py
│   └── train.py
|
├── inventory.db
│
├── models/
│   ├── predict_freight_model.pkl
│   ├── predict_flag_invoice.pkl
│   └── scaler.pkl
│
├── notebooks/
│   ├── model.ipynb
│   └── invoice_flag.ipynb
│
├── requirements.txt
└── README.md
```

---

## 📋 Required Input Columns

For Batch Processing:

| Column Name | Description |
|------------|-------------|
| invoice_quantity | Invoice quantity |
| invoice_dollars | Invoice amount |
| freight | Freight charge |
| total_item_quantity | Total item quantity |
| total_item_dollars | Total item amount |

---

## 📊 Machine Learning Pipeline

### Freight Prediction

Input Features:

- Dollars
- Quantity

Target:

- Freight

Models Evaluated:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

Evaluation Metrics:

- MAE
- RMSE
- R² Score

---

### Invoice Flagging

Input Features:

- Invoice Quantity
- Invoice Dollars
- Freight
- Total Item Quantity
- Total Item Dollars

Output:

- Risk Flag (0/1)
- Risk Probability

---

## 🛠 Tech Stack

### Backend

- Python

### Machine Learning

- Scikit-Learn
- NumPy
- Pandas

### Database

- SQLite

### Visualization

- Plotly

### Frontend

- Streamlit

---

## 🎯 Future Enhancements

- XGBoost Integration
- LightGBM Models
- SHAP Explainability
- Fraud Detection Module
- MLOps Pipeline
- Docker Deployment
- CI/CD Automation
- Cloud Deployment (AWS/Azure/GCP)

---

## 👨‍💻 Author

Ojas Shukla


## ⭐ If you found this project useful, don't forget to star the repository.
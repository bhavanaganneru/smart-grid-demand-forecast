# ⚡ Smart Grid Electricity Demand Forecast Dashboard

An AI-powered electricity demand forecasting web application built using Machine Learning and Streamlit.

## 🚀 Features

- 24-hour electricity demand prediction
- Rolling time-series forecasting
- CO₂ emission estimation based on fossil fuel emission factor
- Peak demand detection
- AI-based demand explanation panel
- Interactive data visualization dashboard
- Premium glass-style UI

## 🧠 Technologies Used

- Python 3.11
- Pandas
- Scikit-learn (Random Forest)
- Plotly
- Streamlit

## 📊 Model Details

The model uses:
- Hour
- Day of week
- Month
- Year
- Lag-1 demand feature

Algorithm:
Random Forest Regressor

Evaluation Metric:
Mean Absolute Error (MAE)

## 🌍 Environmental Impact

The dashboard estimates CO₂ emissions using:
Emission Factor = 0.82 kg CO₂ per kWh

This connects electricity demand prediction with fossil fuel usage and environmental sustainability.

## ▶ How to Run Locally

1. Clone repository
2. Install dependencies:
## 📌 Deployment
Deployed using Streamlit Cloud.



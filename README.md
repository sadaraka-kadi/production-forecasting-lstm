
# Production Forecasting with LSTM

This project focuses on forecasting monthly oil & gas production using a combination of baseline statistical models, classical machine learning, and deep learning (LSTM).
The goal is to compare different modeling approaches, understand production dynamics, and translate results into practical, decision-oriented insights.

## Notebooks
- `01_data_cleaning.ipynb` – Data cleaning and preprocessing: handling missing values, type conversions, monthly aggregation, and dataset validation.
- `02_exploratory_analysis.ipynb` – Exploratory Data Analysis (EDA): trend visualization, seasonality detection, rolling statistics, outlier inspection, and initial conclusions about autocorrelation and variance behavior.
- `03_feature_engineering.ipynb` - Creation of supervised learning features, including lag variables, rolling statistics, and percentage changes. Feature relevance is assessed using correlation analysis.
- `04_baseline_models.ipynb` - Implementation of baseline forecasting models (Naive forecast, Moving Average, Exponential Weighted Moving Average, Holt-Winters) used as performance benchmarks.
- `05_arima_models.ipynb` - Classical time series modeling using ARIMA and Seasonal ARIMA. Includes stationarity testing (ADF), differencing, ACF/PACF analysis, and residual diagnostics.
- `06_ml_models.ipynb` - Machine learning models for time series forecasting, including Linear Regression, Random Forest Regression, and Support Vector Regression.
One-step, multi-step recursive, and multi-output forecasting strategies are evaluated.
- `07_lstm_hyperparameter_tuning.ipynb` - Covers sequence hyperparameter tuning.
- `08_lstm_model.ipynb` - Deep learning approach using LSTM networks. Covers sequence creation, scaling and validation using RMSE.
- `09_24m_forecast,ipynb` - Future forecasting 24 months ahead using LSTM multi-ouput forecasting.
- `10_results_and_business_insights.ipynb` - Contains plots and comparison tables
  

## Data
- Kaggle Dataset: US Oil & Gas Production & Disposition 2015–2025 (https://www.kaggle.com/datasets/pinuto/us-oil-and-gas-production-and-disposition-20152025/data)
- Frequency: Monthly
- Period: 2015 - 2025
- Target Variable: Oil Production Volume
- Data is claeaned and preprocesses before modelling

## Evaluation Metrics
- RMSE
- MAE
- MAPE
- R^2

## Key Takeaways
- Baseline statistical models provide useful benchmarks, with EWMA outperforming other classical approaches on test data.
- ARIMA and SARIMA models exhibit limited generalization, indicating that linear autoregressive structures are insufficient for capturing production dynamics.
- Classical machine learning models (SVR and Random Forest) outperform LSTM in one-step and recursive multi-step forecasting scenarios.
- Recursive multi-step LSTM forecasting suffers from error accumulation, leading to drift and loss of variance.
- Direct multi-output LSTM models significantly improve multi-horizon forecast stability and outperform recursive approaches for fixed-horizon forecasting.
- Model performance is strongly influenced by the forecasting strategy, not just the model architecture.
- LSTM models are most effective when used for fixed-horizon, multi-output forecasting rather than recursive prediction.

## Tools
- Python (pandas, numpy)
- matplotlib, seaborn
- statsmodels
- scikit-learn
- TensorFlow / Keras
- Jupyter Notebook

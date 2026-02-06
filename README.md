# Production Forecasting with LSTM

This project forecasts monthly oil & gas production using baseline statistical models, classical machine learning, and deep learning (LSTM). The goal is to compare different modeling approaches, understand production dynamics, and translate results into practical, decision-oriented insights for operational planning.

## Project Structure

### Notebooks

1. **`01_data_cleaning.ipynb`** – Data cleaning and preprocessing: handling missing values, type conversions, monthly aggregation, and dataset validation.

2. **`02_exploratory_analysis.ipynb`** – Exploratory Data Analysis (EDA): trend visualization, seasonality detection, rolling statistics, outlier inspection, and initial assessment of autocorrelation and variance behavior.

3. **`03_feature_engineering.ipynb`** – Creation of supervised learning features, including lag variables, rolling statistics, and percentage changes. Feature relevance is assessed using correlation analysis.

4. **`04_baseline_models.ipynb`** – Implementation of baseline forecasting models (Naive forecast, Moving Average, Exponential Weighted Moving Average, Holt-Winters) used as performance benchmarks.

5. **`05_arima_models.ipynb`** – Classical time series modeling using ARIMA and Seasonal ARIMA. Includes stationarity testing (ADF), differencing, ACF/PACF analysis, and residual diagnostics.

6. **`06_ml_models.ipynb`** – Machine learning models for time series forecasting, including Linear Regression, Random Forest Regression, and Support Vector Regression. One-step, multi-step recursive, and multi-output forecasting strategies are evaluated.

7. **`07_lstm_hyperparameter_tuning.ipynb`** – Hyperparameter tuning for LSTM architecture: sequence length, batch size, learning rate, and layer configuration.

8. **`08_lstm_model.ipynb`** – Deep learning approach using LSTM networks. Covers sequence creation, train/validation/test splitting, scaling, and performance evaluation. Implements three forecasting strategies:
   - **1-step ahead**: Single time step predictions
   - **Multi-step recursive**: Iterative forecasting with prediction feedback
   - **Multi-output direct**: Simultaneous prediction of multiple future time steps

9. **`09_24m_forecast.ipynb`** – Production of a 24-month forward forecast using the final LSTM multi-output model trained on the complete dataset, with Monte Carlo confidence intervals.

10. **`10_results_and_business_insights.ipynb`** – Consolidated results, visual comparisons across all models, and interpretation of findings from business and operational perspectives.

## Dataset

- **Source**: [US Oil & Gas Production & Disposition 2015–2025](https://www.kaggle.com/datasets/pinuto/us-oil-and-gas-production-and-disposition-20152025/data) (Kaggle)
- **Frequency**: Monthly
- **Period**: January 2015 – Present
- **Target Variable**: Oil Production Volume (barrels)
- **Preprocessing**: Data cleaned and aggregated before modeling; log transformation and differencing applied for stationarity

## Methodology

### Forecasting Strategies

Three distinct forecasting approaches are evaluated:

1. **One-step ahead**: Model predicts only the next time step using actual historical data
2. **Multi-step recursive**: Model iteratively predicts future steps, feeding each prediction back as input
3. **Multi-output direct**: Model predicts all future horizons simultaneously in a single forward pass

### Train/Validation/Test Split

- **Training**: ~80% of data
- **Validation**: ~10% of data (for hyperparameter tuning and early stopping)
- **Test**: Last 12 months (for final performance evaluation)

## Evaluation Metrics

All models are compared on the same scale (log volume) using:

- **RMSE** (Root Mean Squared Error): Penalizes large errors
- **MAE** (Mean Absolute Error): Average prediction error
- **MAPE** (Mean Absolute Percentage Error): Relative error metric
- **R²** (R-squared): Proportion of variance explained

## Key Findings

### Model Performance Comparison

1. **Baseline Models**
   - EWMA (Exponential Weighted Moving Average) outperforms other classical approaches
   - Simple baselines provide strong benchmarks for complex models

2. **ARIMA/SARIMA**
   - Limited generalization on test data
   - Linear autoregressive structures insufficient for capturing production dynamics
   - Useful for understanding autocorrelation patterns

3. **Classical Machine Learning**
   - **SVR and Random Forest outperform LSTM** for one-step and recursive multi-step forecasting
   - Tree-based methods handle non-linearity effectively
   - Require less tuning than deep learning approaches

4. **LSTM Deep Learning**
   - **Recursive multi-step forecasting**: Suffers from error accumulation, leading to drift and loss of variance
   - **Direct multi-output forecasting**: **Significantly better** for multi-horizon predictions
   - **Key insight**: LSTM effectiveness depends heavily on forecasting strategy, not just architecture
   - Best suited for fixed-horizon, multi-output scenarios rather than recursive prediction

### Strategic Recommendations

- **Short-term forecasting (1-3 months)**: Use classical ML models (Random Forest, SVR) or 1-step LSTM
- **Medium-term forecasting (6-12 months)**: Use multi-output LSTM with confidence intervals
- **Long-term planning (12-24 months)**: Multi-output LSTM provides stable trend projections but consider ensemble approaches
- **Model selection**: Forecasting strategy is as important as model architecture

## Technologies Used

- **Python 3.x**
- **Data manipulation**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Statistical modeling**: statsmodels
- **Machine Learning**: scikit-learn
- **Deep Learning**: TensorFlow, Keras
- **Development**: Jupyter Notebook

## Installation
```bash
# Clone repository
git clone <repository-url>
cd production-forecasting-lstm

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

## Requirements
```
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
statsmodels>=0.13.0
scikit-learn>=1.0.0
tensorflow>=2.8.0
jupyter>=1.0.0
```

## Usage

Run notebooks in order (01 through 10) to reproduce the complete analysis pipeline. Each notebook is self-contained with clear documentation and can be run independently after completing prerequisite notebooks.

## Results Summary

| Model | MAPE (Test) | RMSE (Test) | Best Use Case |
|-------|-------------|-------------|---------------|
| Naive Baseline | 3.1% | 3548693.13 | Benchmark |
| ARIMA | 4.1% | 4538933.012| Pattern analysis |
| Random Forest | 0.15% | 0.04| Short-term forecast |
| SVR | 0.17% | 0.047 | Short-term forecast |
| 1-Step LSTM | 0.27% | 0.0035 | Single-step prediction |
| Multi-Step LSTM | 0.37% | 0.0078 | Avoid (error accumulation) |
| Multi-Output LSTM | 0.19% | 0.0016 | **Medium-term forecast** |

## Future Work

- Incorporate exogenous variables (oil prices, rig counts, seasonal demand)
- Implement hybrid models combining statistical and ML approaches
- Add ensemble methods for improved robustness
- Extend to multivariate forecasting (multiple production fields)
- Deploy model as API for real-time forecasting

## License

[ MIT]

## Author

[Kadi Sadaraka]

## Acknowledgments

- Dataset: Kaggle - US Oil & Gas Production & Disposition

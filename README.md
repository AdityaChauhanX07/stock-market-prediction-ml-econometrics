# Developing Predictive Models for Stock Market Behavior Using Machine Learning and Econometrics
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

## Overview

This repository contains the research data, code, and model outputs for the study "Developing Predictive Models for Stock Market Behavior Using Machine Learning and Econometrics." The research focuses on developing and comparing predictive models for stock market behavior using a combination of traditional econometric approaches (ARIMA, GARCH) and modern machine learning techniques (LSTM neural networks), culminating in a hybrid ARIMA-LSTM model.

The dataset encompasses S&P 500 and Tesla (TSLA) stock price data, macroeconomic indicators, and market sentiment data. The study evaluates model performance through rigorous time-series cross-validation and provides comprehensive error analysis to assess the predictive capabilities of each approach.

## File Manifest

```
Developing Predictive Models for Stock Market Behavior Using Machine Learning and Econometrics/
├── 1_Data_Files/
│   ├── 01_Raw_Data/
│   │   ├── 001_Stock_Market_Data/          # Raw S&P 500 and TSLA stock data from Yahoo Finance
│   │   ├── 002_Macroeconomic_Indicators/   # Raw macroeconomic data (GDP, interest rates, etc.)
│   │   └── 003_Sentiment_Data/             # Links to sources for sentiment data
│   └── 02_Cleaned_Data/                    # Processed and cleaned datasets ready for modeling
│
├── 2_Code-Scripts/
│   ├── 01_Data_Acquisition_and_Cleaning_Scripts/    # Scripts to download and clean raw data
│   ├── 02_Feature_Engineering_Scripts/              # Scripts to process raw data and create model features
│   ├── 03_Model_Training_and_Evaluation_Scripts/    # Scripts to train and evaluate all models
│   ├── 04_Visualization_Scripts/                    # Scripts to generate plots and figures
│   ├── 05_Econometric_Model_Scripts/                # Scripts focused on ARIMA and GARCH models
│   └── requirements.txt                             # Python package dependencies with versions
│
├── 3_Model_Outputs/
│   ├── 01_Raw_Predictions/
│   │   ├── arima_predictions.csv           # Raw forecast outputs from ARIMA model
│   │   ├── lstm_predictions.csv            # Raw forecast outputs from LSTM model
│   │   └── hybrid_predictions.csv          # Raw forecast outputs from Hybrid model
│   ├── 02_Evaluation_Metrics/
│   │   ├── cross_validation_results.csv    # 5-fold time-series cross-validation results
│   │   └── error_analysis.csv              # Detailed error breakdown for hybrid model
│   └── 03_Trained_Models/
│       ├── Trained_Models/
│       │   └── hybrid_model.h5             # Saved trained hybrid model
│       ├── generate_hybrid_model.py        # Script to generate hybrid model
│       └── hyperparameter_tuning_results.csv # Grid search hyperparameter results
│
└── README.md                               # This documentation file
```

## Data Description

### Stock Market Data
- **Price Variables**: Open, High, Low, Close, Adjusted Close prices (USD)
- **Volume**: Trading volume (number of shares)
- **Returns**: Daily log returns calculated as ln(P_t/P_{t-1})
- **Volatility**: Rolling standard deviation of returns

### Macroeconomic Indicators
- **GDP**: Gross Domestic Product growth rate (%)
- **Interest Rates**: Federal funds rate (%)
- **Inflation**: Consumer Price Index (CPI) year-over-year change (%)
- **Unemployment**: Unemployment rate (%)
- **VIX**: Volatility Index (fear gauge)

### Sentiment Data
- **Sentiment Scores**: Numerical sentiment indicators ranging from -1 (negative) to +1 (positive)
- **News Volume**: Count of news articles per day
- **Social Media Metrics**: Twitter/Reddit sentiment indicators

### Model Output Variables
- **Predictions**: Forecasted stock prices or returns
- **Confidence Intervals**: Upper and lower bounds for predictions
- **Error Metrics**: MAE (Mean Absolute Error), RMSE (Root Mean Square Error), MAPE (Mean Absolute Percentage Error)

## Methodology/Workflow

1. **Data Acquisition**: Download historical stock data from Yahoo Finance and macroeconomic indicators from FRED database
2. **Data Cleaning**: Handle missing values, outliers, and ensure data consistency across time series
3. **Feature Engineering**: Create technical indicators, lagged variables, and derived features
4. **Model Development**:
   - Train ARIMA models with optimal parameters selected via AIC/BIC criteria
   - Develop LSTM neural networks with hyperparameter tuning
   - Create hybrid ARIMA-LSTM model combining both approaches
5. **Model Evaluation**: Perform 5-fold time-series cross-validation and calculate performance metrics
6. **Visualization**: Generate plots and figures for model comparison and error analysis

## Software Requirements

### Core Dependencies
- **Python**: 3.9.0 or higher
- **NumPy**: 1.21.0
- **Pandas**: 1.5.3
- **Scikit-learn**: 1.2.0
- **TensorFlow**: 2.10.0
- **Keras**: 2.10.0

### Time Series and Econometrics
- **Statsmodels**: 0.13.2
- **Arch**: 5.3.0
- **Pmdarima**: 2.0.1

### Data Acquisition
- **yfinance**: 0.2.10
- **pandas-datareader**: 0.10.0
- **fredapi**: 0.5.0

### Visualization
- **Matplotlib**: 3.6.0
- **Seaborn**: 0.11.2
- **Plotly**: 5.11.0

### Additional Utilities
- **Jupyter**: 1.0.0
- **Joblib**: 1.2.0
- **Tqdm**: 4.64.0

**Installation**: Run `pip install -r requirements.txt` to install all dependencies.

## Usage Notes

### Data Access
- Stock market data is sourced from Yahoo Finance using the `yfinance` library
- Macroeconomic data requires a FRED API key (free registration at https://fred.stlouisfed.org/)
- Sentiment data sources are provided as reference links in the `003_Sentiment_Data` folder

### Model Training
- LSTM models require GPU acceleration for optimal performance (CUDA-compatible GPU recommended)
- Training time varies from 30 minutes (ARIMA) to 4-6 hours (LSTM/Hybrid) depending on hardware
- Models are trained on 80% of data with 20% reserved for final testing

### Cross-Validation
- Time-series cross-validation maintains temporal order and prevents data leakage
- 5-fold validation uses expanding windows with minimum 252 trading days per fold
- Results may vary slightly due to random initialization in neural networks

### Reproducibility
- Set random seeds in all scripts for reproducible results
- Model checkpoints are saved during training to resume interrupted sessions
- All hyperparameters are documented in configuration files

### Performance Considerations
- Large datasets may require significant memory (16GB+ RAM recommended)
- Consider using data sampling for initial experimentation
- Monitor system resources during intensive model training phases

## License

This research data and code are licensed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). You are free to share, copy, redistribute, adapt, remix, transform, and build upon the material for any purpose, even commercially, as long as you provide appropriate credit to the original authors, provide a link to the license, and indicate if changes were made.

For more details, see the [LICENSE](./LICENSE) file in this repository.

**Citation**: When using this dataset or code, please cite the original research paper and provide attribution to the author.

---

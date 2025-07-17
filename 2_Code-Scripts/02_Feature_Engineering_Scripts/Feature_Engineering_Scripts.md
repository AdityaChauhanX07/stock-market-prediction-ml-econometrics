# Code and Scripts

## `feature_engineering.py`

This final script constructs the complete, model-ready dataset. It generates all engineered features and merges the disparate data sources into a single, unified CSV file.

* **Purpose**: To engineer technical and econometric features, merge all data sources, and apply final normalization.
* **Libraries**: `pandas`, `numpy`, `statsmodels`, `arch`, `scikit-learn`.
* **Process**:
   1. **Load Processed Data**: Loads the cleaned stock data and the daily-frequency macroeconomic and sentiment data.
   2. **Technical Feature Engineering**: For each ticker, calculates a suite of technical indicators from the price data, including:
      * 20-day and 50-day Simple Moving Averages (SMA).
      * 20-day rolling volatility (standard deviation of returns).
      * 14-day Relative Strength Index (RSI).
      * Moving Average Convergence Divergence (MACD) with its signal line and histogram.
   3. **Econometric Feature Engineering**:
      * Fits a GARCH(1,1) model to the daily returns of each ticker using the `arch` library to estimate daily conditional volatility.
      * Fits an ARIMA model to the price series using the `statsmodels` library and extracts the model residuals.
   4. **Data Merging**: Merges the stock data (now including technical and econometric features) with the daily macroeconomic and sentiment data frames based on date and ticker.
   5. **Finalization**:
      * Performs a final forward-fill to handle any missing values that may have arisen from rolling window calculations.
      * Applies Min-Max scaling to all numerical feature columns using `scikit-learn`'s `MinMaxScaler` to normalize the data into a range, as required by the models.
* **Output**: `processed_stock_data_2010-2023.csv`
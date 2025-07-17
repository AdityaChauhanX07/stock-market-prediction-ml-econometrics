import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)

# --- Configuration ---
PROCESSED_DATA_DIR = '01_Data_Files/Cleaned_Data'
PROCESSED_DATA_FILE = os.path.join(PROCESSED_DATA_DIR, 'processed_stock_data_2010-2023.csv')
RESULTS_DIR = '03_Results'
ECONOMETRIC_RESULTS_DIR = os.path.join(RESULTS_DIR, 'Econometric_Models')

# Select a ticker for the case study
TICKER_TO_MODEL = 'TSLA'
# BUG FIX: Corrected column name to match the data file
TARGET_VARIABLE = 'Adj Close_normalized'
RETURNS_VARIABLE = 'daily_return_normalized' # The variable for volatility modeling

# --- Helper Functions ---
def create_output_directory():
    """Creates the results directory if it doesn't exist."""
    if not os.path.exists(ECONOMETRIC_RESULTS_DIR):
        os.makedirs(ECONOMETRIC_RESULTS_DIR)
        print(f"Created directory: {ECONOMETRIC_RESULTS_DIR}")

def load_and_prepare_data():
    """Loads and prepares the data for a specific ticker."""
    print(f"Loading data for ticker: {TICKER_TO_MODEL}")
    df = pd.read_csv(PROCESSED_DATA_FILE, parse_dates=['Date'], index_col='Date')
    ticker_df = df[df['Ticker'] == TICKER_TO_MODEL].copy().sort_index()

    # Split data into 80% train and 20% test sets
    train_size = int(len(ticker_df) * 0.8)
    train_df, test_df = ticker_df[:train_size], ticker_df[train_size:]

    print(f"Train set size: {len(train_df)}, Test set size: {len(test_df)}")
    return train_df, test_df

# --- Model Implementations ---

def run_arima_model(train_df, test_df):
    """
    Trains an ARIMA model once on the training data and forecasts the entire test period.
    This approach is significantly more efficient than re-training at every step.
    - p (AR): 5, d (I): 1, q (MA): 0
    """
    print("\n--- Running ARIMA(5,1,0) Model ---")
    train_target = train_df[TARGET_VARIABLE]
    test_target = test_df[TARGET_VARIABLE]

    # Fit the model once on the full training data
    model = ARIMA(train_target, order=(5, 1, 0))
    model_fit = model.fit()

    # Forecast the entire length of the test set
    predictions = model_fit.forecast(steps=len(test_df))

    # Evaluate forecasts
    rmse = np.sqrt(mean_squared_error(test_target, predictions))
    mae = mean_absolute_error(test_target, predictions)
    print(f"ARIMA Test RMSE: {rmse:.4f}")
    print(f"ARIMA Test MAE: {mae:.4f}")

    # Save results
    results = pd.DataFrame({
        'Actual': test_target,
        'Predicted': predictions
    })
    results.to_csv(os.path.join(ECONOMETRIC_RESULTS_DIR, 'arima_predictions.csv'))

    # Plot results
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(15, 7))
    plt.plot(test_target.index, test_target, label='Actual Prices')
    plt.plot(test_target.index, predictions, color='red', linestyle='--', label='ARIMA Forecast')
    plt.title(f'ARIMA Forecast vs Actual Prices for {TICKER_TO_MODEL}')
    plt.xlabel('Date')
    plt.ylabel('Normalized Price')
    plt.legend()
    plt.savefig(os.path.join(ECONOMETRIC_RESULTS_DIR, 'arima_forecast_plot.png'))
    plt.close()
    print("ARIMA predictions and plot saved.")

def run_garch_model(full_df):
    """
    Fits a GARCH(1,1) model to analyze and visualize the volatility of the full time series.
    GARCH models are fitted on returns to model volatility clustering.
    """
    print("\n--- Running GARCH(1,1) Model for Volatility Analysis ---")
    # GARCH models are fit on returns. We use un-normalized returns for better interpretability if available,
    # but normalized is fine. Scaling by 100 often helps model convergence.
    returns = full_df[RETURNS_VARIABLE].dropna() * 100

    # Define and fit the GARCH(1,1) model
    model = arch_model(returns, vol='Garch', p=1, q=1, rescale=False)
    model_fit = model.fit(disp='off')

    # Print the model summary
    print("GARCH Model Summary:")
    print(model_fit.summary())

    # Save the summary to a file
    with open(os.path.join(ECONOMETRIC_RESULTS_DIR, 'garch_model_summary.txt'), 'w') as f:
        f.write(str(model_fit.summary()))

    # Plot the annualized conditional volatility
    fig = model_fit.plot(annualize='D') # 'D' for daily data
    plt.suptitle(f'GARCH Conditional Volatility for {TICKER_TO_MODEL}', y=0.98)
    fig.savefig(os.path.join(ECONOMETRIC_RESULTS_DIR, 'garch_volatility_plot.png'))
    plt.close()
    print("GARCH summary and plot saved.")


# --- Main Execution ---
if __name__ == "__main__":
    create_output_directory()
    train_data, test_data = load_and_prepare_data()

    # Run ARIMA for forecasting
    run_arima_model(train_data, test_data)

    # Run GARCH on the full dataset's returns for analysis
    full_dataset = pd.concat([train_data, test_data])
    run_garch_model(full_dataset)

    print("\n--- Econometric Modeling Script Complete ---")
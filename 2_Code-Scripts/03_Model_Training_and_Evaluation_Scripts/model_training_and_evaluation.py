import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- Configuration ---
PROCESSED_DATA_DIR = '01_Data_Files/Cleaned_Data'
PROCESSED_DATA_FILE = os.path.join(PROCESSED_DATA_DIR, 'processed_stock_data_2010-2023.csv')
RESULTS_DIR = '03_Results'
RESULTS_FILE = os.path.join(RESULTS_DIR, 'model_performance_comparison.csv')

# Select a ticker for the case study
TICKER_TO_MODEL = 'TSLA'
# CRITICAL FIX: Corrected column name to match the output of the feature engineering script.
TARGET_VARIABLE = 'Adj Close_normalized'
# Model Hyperparameters
N_STEPS = 50 # Lookback window for LSTM

# --- Helper Functions ---
def create_output_directory():
    """Creates the results directory if it doesn't exist."""
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        print(f"Created directory: {RESULTS_DIR}")

def load_processed_data():
    """Loads the final processed and feature-engineered dataset."""
    print(f"Loading processed data for ticker: {TICKER_TO_MODEL}")
    df = pd.read_csv(PROCESSED_DATA_FILE, parse_dates=['Date'], index_col='Date')
    ticker_df = df[df['Ticker'] == TICKER_TO_MODEL].copy()
    # Ensure data is sorted by date for time-series analysis
    ticker_df.sort_index(inplace=True)
    return ticker_df

def create_sequences(data, n_steps):
    """Converts a time-series dataset into sequences for LSTM."""
    X, y = [], []
    # Check if data is 1D or 2D
    is_multivariate = data.ndim > 1
    
    for i in range(len(data)):
        end_ix = i + n_steps
        if end_ix > len(data)-1:
            break
        # Input sequence is all features; output is the first feature (target)
        seq_x = data[i:end_ix, :] if is_multivariate else data[i:end_ix]
        seq_y = data[end_ix, 0] if is_multivariate else data[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def evaluate_model(y_true, y_pred, model_name):
    """Calculates and returns performance metrics."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    # MAPE (Mean Absolute Percentage Error) is often useful for price prediction
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print(f"{model_name} Performance -> RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%")
    return {'Model': model_name, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape}

# --- Model Training and Evaluation ---

def train_evaluate_arima(train_data, test_data):
    """
    Trains an ARIMA model once on the training data and forecasts the entire test period.
    This is a much more efficient approach than re-training at every step.
    """
    print("\n--- Training ARIMA Model ---")
    # Fit the model on the entire training history
    model = ARIMA(train_data, order=(5,1,0)) # (p,d,q) order
    model_fit = model.fit()
    
    # Forecast the entire test set length
    predictions = model_fit.forecast(steps=len(test_data))
    
    return evaluate_model(test_data, predictions, "ARIMA")

def train_evaluate_lstm(train_data, test_data, n_steps):
    """Trains and evaluates a standard LSTM model."""
    print("\n--- Training LSTM Model ---")
    # Prepare data for LSTM
    X_train, y_train = create_sequences(train_data, n_steps)
    X_test, y_test = create_sequences(test_data, n_steps)

    # Define LSTM model architecture
    model = Sequential([
        LSTM(50, activation='tanh', input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    
    # Train the model
    print("Training LSTM... this may take a moment.")
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
    
    # Make predictions
    predictions = model.predict(X_test, verbose=0)
    
    return evaluate_model(y_test, predictions.flatten(), "LSTM")

def train_evaluate_hybrid_model(df, train_size, n_steps):
    """
    Trains and evaluates a correct Hybrid ARIMA-LSTM model.
    1. ARIMA models the linear part of the time series.
    2. LSTM models the non-linear part (the errors/residuals from ARIMA).
    3. Final prediction = ARIMA prediction + LSTM error prediction.
    """
    print("\n--- Training Hybrid (ARIMA-LSTM) Model ---")
    
    # 1. Train ARIMA and get forecasts and residuals
    target_series = df[TARGET_VARIABLE]
    train_target, test_target = target_series[:train_size], target_series[train_size:]
    
    arima_model = ARIMA(train_target, order=(5,1,0)).fit()
    # Get residuals from the training period. The first value will be NaN due to differencing.
    train_residuals = arima_model.resid.iloc[1:].values
    # Get ARIMA's forecast for the test period
    arima_test_forecast = arima_model.forecast(steps=len(test_target))

    # 2. Prepare data for the error-predicting LSTM
    # The LSTM will use all features to predict the ARIMA residuals.
    all_features = df.drop(columns=['Ticker']).values
    # Align features with the training residuals (which are shorter by 1)
    train_features_for_lstm = all_features[1:train_size]

    X_train_res, y_train_res = create_sequences(np.column_stack([train_residuals, train_features_for_lstm]), n_steps)
    # The LSTM's target `y` is the residual, which is the first column
    y_train_res = X_train_res[:, -1, 0] 
    # The LSTM's input `X` are the sequences of all features (including past residuals)
    X_train_res = X_train_res[:, :, :]

    # For testing, the LSTM will use the last `n_steps` of the training features to start predicting
    test_features_for_lstm = all_features[train_size-n_steps:]
    X_test_res, _ = create_sequences(test_features_for_lstm, n_steps)

    # 3. Train LSTM to predict residuals
    print("Training Residual-LSTM... this may take a moment.")
    residual_model = Sequential([
        LSTM(50, activation='tanh', input_shape=(X_train_res.shape[1], X_train_res.shape[2])),
        Dropout(0.2),
        Dense(1)
    ])
    residual_model.compile(optimizer='adam', loss='mse')
    residual_model.fit(X_train_res, y_train_res, epochs=50, batch_size=32, verbose=0)
    
    # 4. Make final hybrid prediction
    predicted_residuals = residual_model.predict(X_test_res, verbose=0)
    
    # Final prediction = ARIMA forecast + LSTM's predicted error
    final_predictions = arima_test_forecast.values + predicted_residuals.flatten()
    
    # 5. Evaluate the hybrid model
    # The LSTM part of the model can't predict for the first `n_steps` of the test set,
    # so we align the true values accordingly. This is a small discrepancy from the other models.
    # For a robust comparison, we ensure the lengths match.
    return evaluate_model(test_target.values[:len(final_predictions)], final_predictions, "Hybrid (ARIMA-LSTM)")


# --- Main Execution ---
if __name__ == "__main__":
    create_output_directory()
    df = load_processed_data()

    # Split data (80% train, 20% test)
    train_size = int(len(df) * 0.8)

    # Data for ARIMA (univariate)
    arima_train = df[TARGET_VARIABLE].iloc[:train_size]
    arima_test = df[TARGET_VARIABLE].iloc[train_size:]
    
    # Data for ML models (multivariate)
    ml_features = df.drop(columns=['Ticker']).values
    ml_train, ml_test = ml_features[:train_size], ml_features[train_size:]
    
    # Run models
    performance_results = []
    performance_results.append(train_evaluate_arima(arima_train, arima_test))
    performance_results.append(train_evaluate_lstm(ml_train, ml_test, n_steps=N_STEPS))
    performance_results.append(train_evaluate_hybrid_model(df, train_size, n_steps=N_STEPS))
    
    # Save results
    results_df = pd.DataFrame(performance_results)
    results_df.to_csv(RESULTS_FILE, index=False)
    
    print("\n--- Model Performance Comparison ---")
    print(results_df.round(4))
    print(f"\nResults saved to {RESULTS_FILE}")
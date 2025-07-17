import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
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

# Visualization Output Files
PERFORMANCE_PLOT_FILE = os.path.join(RESULTS_DIR, 'model_performance_comparison.png')
PREDICTION_PLOT_FILE = os.path.join(RESULTS_DIR, 'tesla_prediction_vs_actual.png')

# Model configuration (must match the training script)
TICKER_TO_MODEL = 'TSLA'
# CRITICAL FIX: Corrected column name to prevent KeyError
TARGET_VARIABLE = 'Adj Close_normalized'
N_STEPS = 50

# --- Helper Functions ---
def create_sequences(data, n_steps):
    """Converts a time-series dataset into sequences for LSTM."""
    X, y = [], []
    is_multivariate = data.ndim > 1
    
    for i in range(len(data)):
        end_ix = i + n_steps
        if end_ix > len(data)-1:
            break
        seq_x = data[i:end_ix, :] if is_multivariate else data[i:end_ix]
        # For this function, we assume the target is always the first column
        seq_y = data[end_ix, 0] if is_multivariate else data[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# --- Visualization Functions ---

def plot_model_performance():
    """
    Loads model performance data and creates a bar chart comparing metrics.
    """
    print("Generating model performance comparison plot...")
    if not os.path.exists(RESULTS_FILE):
        print(f"Error: Results file not found at {RESULTS_FILE}. Please run the training script first.")
        return

    results_df = pd.read_csv(RESULTS_FILE)
    
    # Melt the dataframe to make it suitable for seaborn's barplot
    results_melted = results_df.melt(id_vars='Model', var_name='Metric', value_name='Score',
                                     value_vars=['RMSE', 'MAE', 'MAPE'])

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))
    
    sns.barplot(data=results_melted, x='Model', y='Score', hue='Metric', ax=ax)
    
    ax.set_title('Comparative Performance of Predictive Models', fontsize=16, weight='bold')
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Error Score', fontsize=12)
    ax.legend(title='Metric')
    
    plt.savefig(PERFORMANCE_PLOT_FILE)
    print(f"Performance plot saved to {PERFORMANCE_PLOT_FILE}")
    plt.close()

def plot_prediction_vs_actual():
    """
    Retrains the Hybrid model using the CORRECT methodology to get predictions and plots them.
    """
    print("\nRecreating Hybrid model predictions for plotting...")
    df = pd.read_csv(PROCESSED_DATA_FILE, parse_dates=['Date'], index_col='Date')
    ticker_df = df[df['Ticker'] == TICKER_TO_MODEL].copy().sort_index()

    # --- Re-run the CORRECT hybrid model prediction logic ---
    train_size = int(len(ticker_df) * 0.8)
    
    # 1. Get ARIMA forecasts and residuals
    target_series = ticker_df[TARGET_VARIABLE]
    train_target, test_target = target_series[:train_size], target_series[train_size:]
    
    arima_model = ARIMA(train_target, order=(5,1,0)).fit()
    train_residuals = arima_model.resid.iloc[1:].values # Exclude first NaN
    arima_test_forecast = arima_model.forecast(steps=len(test_target))

    # 2. Prepare data for the error-predicting LSTM
    all_features = ticker_df.drop(columns=['Ticker']).values
    # Align features with the shorter residuals series
    train_features_for_lstm = all_features[1:train_size] 
    
    # Combine residuals (as target) and features for sequencing
    lstm_training_data = np.column_stack([train_residuals, train_features_for_lstm])
    
    X_train_res, _ = create_sequences(lstm_training_data, N_STEPS)
    y_train_res = X_train_res[:, -1, 0] # The target is the residual (first column) at the end of the sequence
    
    test_features_for_lstm = all_features[train_size - N_STEPS:]
    X_test_res, _ = create_sequences(test_features_for_lstm, N_STEPS)
    
    # 3. Train LSTM to predict residuals
    print("Training Residual-LSTM for visualization...")
    residual_model = Sequential([
        LSTM(50, activation='tanh', input_shape=(X_train_res.shape[1], X_train_res.shape[2])),
        Dropout(0.2),
        Dense(1)
    ])
    residual_model.compile(optimizer='adam', loss='mse')
    residual_model.fit(X_train_res, y_train_res, epochs=50, batch_size=32, verbose=0)
    
    # 4. Get final hybrid predictions
    predicted_residuals = residual_model.predict(X_test_res, verbose=0)
    final_predictions = arima_test_forecast.values + predicted_residuals.flatten()
    
    # --- Create the plot ---
    prediction_dates = test_target.index[:len(final_predictions)]

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(15, 8))
    
    ax.plot(ticker_df.index, target_series, label='Actual Price (Normalized)', color='royalblue', lw=2)
    ax.plot(prediction_dates, final_predictions, label='Hybrid Model Prediction', color='red', linestyle='--')
    
    ax.set_title(f'{TICKER_TO_MODEL} Stock Prediction: Actual vs. Hybrid Model', fontsize=16, weight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Adjusted Close Price (Normalized)', fontsize=12)
    ax.axvline(ticker_df.index[train_size], color='gray', linestyle=':', lw=2, label='Train/Test Split')
    ax.legend()
    ax.grid(True)

    plt.savefig(PREDICTION_PLOT_FILE)
    print(f"Prediction plot saved to {PREDICTION_PLOT_FILE}")
    plt.close()

# --- Main Execution ---
if __name__ == "__main__":
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    
    plot_model_performance()
    plot_prediction_vs_actual()
    
    print("\n--- Visualization Script Complete ---")
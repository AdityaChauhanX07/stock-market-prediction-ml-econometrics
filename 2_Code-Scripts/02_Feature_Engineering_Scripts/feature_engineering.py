import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from arch import arch_model
import os
import numpy as np

# --- Configuration ---
# Input files from the data acquisition step.
RAW_DATA_DIR = '01_Data_Files/Raw_Data'
PROCESSED_DATA_DIR = '01_Data_Files/Cleaned_Data'
STOCK_DATA_FILE = os.path.join(RAW_DATA_DIR, 'stock_data_2010-2023.csv')
MACRO_DATA_FILE = os.path.join(RAW_DATA_DIR, 'macroeconomic_indicators_raw.csv')

# Output file for the processed data.
PROCESSED_OUTPUT_FILE = os.path.join(PROCESSED_DATA_DIR, 'processed_stock_data_2010-2023.csv')

# --- Helper Functions ---
def create_output_directory():
    """Creates the output directory for processed data if it doesn't exist."""
    if not os.path.exists(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR)
        print(f"Created directory: {PROCESSED_DATA_DIR}")

def load_and_merge_data():
    """Loads raw data, merges them, and performs initial cleaning."""
    print("Loading and merging raw data...")
    stock_df = pd.read_csv(STOCK_DATA_FILE, parse_dates=['Date'])
    macro_df = pd.read_csv(MACRO_DATA_FILE, parse_dates=['Date'])

    # Set Date as index for time-series operations
    stock_df.set_index('Date', inplace=True)
    macro_df.set_index('Date', inplace=True)

    # Resample macro data to daily frequency using forward-fill
    daily_macro_df = macro_df.resample('D').ffill()
    
    # Merge stock data with daily macro data
    merged_df = stock_df.merge(daily_macro_df, left_index=True, right_index=True, how='left')
    
    # Forward fill any gaps in macro data post-merge (e.g., for holidays)
    merged_df.fillna(method='ffill', inplace=True)
    
    # Ensure data types are correct
    merged_df['Volume'] = merged_df['Volume'].astype(float)
    
    print("Data merged successfully.")
    return merged_df

# --- Feature Engineering Functions ---

def create_time_series_features(df):
    """Calculates daily returns, a primary time-series feature."""
    print("Creating time-series features (daily returns)...")
    # Use 'Adj Close' for calculations to account for stock splits and dividends
    df['daily_return'] = df.groupby('Ticker')['Adj Close'].pct_change()
    return df

def create_technical_indicators(df):
    """Calculates technical indicators like moving averages."""
    print("Creating technical indicators (50-day moving average)...")
    df['rolling_avg_50d'] = df.groupby('Ticker')['Adj Close'].transform(lambda x: x.rolling(window=50).mean())
    return df

def create_econometric_variables(df):
    """Calculates GARCH volatility as an econometric feature."""
    print("Creating econometric variables (GARCH volatility)...")
    volatility_list = []
    for ticker in df['Ticker'].unique():
        print(f"  - Calculating GARCH(1,1) volatility for {ticker}...")
        # GARCH model requires a stationary series (returns are better than prices)
        # We drop NaNs from returns and scale by 100 as it helps with model convergence
        ticker_returns = df[df['Ticker'] == ticker]['daily_return'].dropna() * 100
        
        # Define and fit the GARCH(1,1) model
        model = arch_model(ticker_returns, vol='Garch', p=1, q=1, rescale=False)
        model_fit = model.fit(disp='off')
        
        # Get the conditional volatility and scale it back
        volatility = model_fit.conditional_volatility / 100
        volatility.name = 'volatility_garch'
        volatility_list.append(volatility)
    
    # Concatenate all volatility series and merge back to the main dataframe
    all_volatility = pd.concat(volatility_list)
    df = df.merge(all_volatility, left_index=True, right_index=True, how='left')
    return df

# --- Main Execution ---
if __name__ == "__main__":
    print("--- Starting Feature Engineering ---")
    create_output_directory()
    
    # 1. Load and merge data
    data_df = load_and_merge_data()
    
    # 2. Create features
    data_df = create_time_series_features(data_df)
    data_df = create_technical_indicators(data_df)
    data_df = create_econometric_variables(data_df)
    
    # 3. Final Cleaning
    # Drop rows with NaN values that result from initial return and rolling window calculations
    data_df.dropna(inplace=True)
    print("Dropped initial NaN rows.")

    # 4. Normalization
    print("Normalizing all features for model input...")
    features_to_normalize = [
        'Adj Close', 'Volume', 'daily_return', 'rolling_avg_50d', 'volatility_garch',
        'GDP_Growth_Rate_Annualized', 'Interest_Rate_Federal_Funds', 
        'Inflation_Rate_CPI_YoY', 'Unemployment_Rate'
    ]
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(data_df[features_to_normalize])
    
    # Create a new dataframe with normalized columns
    normalized_df = pd.DataFrame(normalized_data, columns=[f"{col}_normalized" for col in features_to_normalize], index=data_df.index)
    
    # Combine with non-normalized columns (like Ticker) for the final output
    final_df = pd.concat([data_df[['Ticker']], normalized_df], axis=1)
    
    # Save the final processed data
    final_df.to_csv(PROCESSED_OUTPUT_FILE)
    print(f"--- Feature Engineering Complete ---")
    print(f"Final dataset with engineered features saved to {PROCESSED_OUTPUT_FILE}")
    print(f"Dataset contains {final_df.shape[0]} rows and {final_df.shape[1]} columns.")

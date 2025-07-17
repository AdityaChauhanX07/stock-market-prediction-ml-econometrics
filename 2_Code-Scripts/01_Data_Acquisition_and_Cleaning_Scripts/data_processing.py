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

# --- Main Execution ---
def create_output_directory():
    """Creates the output directory for processed data if it doesn't exist."""
    if not os.path.exists(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR)
        print(f"Created directory: {PROCESSED_DATA_DIR}")

def load_data():
    """Loads the raw stock and macroeconomic data."""
    print("Loading raw data files...")
    stock_df = pd.read_csv(STOCK_DATA_FILE, parse_dates=['Date'])
    macro_df = pd.read_csv(MACRO_DATA_FILE, parse_dates=['Date'])
    return stock_df, macro_df

def process_data(stock_df, macro_df):
    """
    Performs data cleaning, feature engineering, merging, and normalization.
    """
    print("Processing data...")
    
    # Set Date as index for time-series operations
    stock_df.set_index('Date', inplace=True)
    macro_df.set_index('Date', inplace=True)

    # 1. Merge Data: Resample macro data to daily frequency and merge
    # We use forward-fill to propagate the last known value.
    daily_macro_df = macro_df.resample('D').ffill()
    merged_df = stock_df.merge(daily_macro_df, left_index=True, right_index=True, how='left')
    
    # Forward fill any remaining NaNs in the merged macro columns
    merged_df.fillna(method='ffill', inplace=True)
    
    # 2. Feature Engineering
    print("Performing feature engineering...")
    
    # Use 'Adj Close' for calculations to account for splits/dividends
    merged_df['daily_return'] = merged_df.groupby('Ticker')['Adj Close'].pct_change()
    merged_df['rolling_avg_50d'] = merged_df.groupby('Ticker')['Adj Close'].transform(lambda x: x.rolling(window=50).mean())

    # Calculate GARCH volatility for each ticker
    volatility_list = []
    for ticker in merged_df['Ticker'].unique():
        print(f"Calculating GARCH(1,1) volatility for {ticker}...")
        ticker_returns = merged_df[merged_df['Ticker'] == ticker]['daily_return'].dropna() * 100 # arch model works better with scaled returns
        model = arch_model(ticker_returns, vol='Garch', p=1, q=1, rescale=False)
        model_fit = model.fit(disp='off')
        # Get conditional volatility and align it with the main dataframe
        volatility = model_fit.conditional_volatility / 100 # scale back
        volatility.name = 'volatility_garch'
        volatility_list.append(volatility)
    
    # Concatenate volatility series and merge back
    all_volatility = pd.concat(volatility_list)
    merged_df = merged_df.merge(all_volatility, left_index=True, right_index=True, how='left')

    # Drop rows with NaN values resulting from rolling calculations and returns
    merged_df.dropna(inplace=True)

    # 3. Normalization
    print("Normalizing features...")
    features_to_normalize = [
        'Adj Close', 'Volume', 'daily_return', 'rolling_avg_50d', 'volatility_garch',
        'GDP_Growth_Rate_Annualized', 'Interest_Rate_Federal_Funds', 
        'Inflation_Rate_CPI_YoY', 'Unemployment_Rate'
    ]
    scaler = MinMaxScaler()
    # Fit and transform the data
    normalized_data = scaler.fit_transform(merged_df[features_to_normalize])
    
    # Create a new dataframe with normalized columns
    normalized_df = pd.DataFrame(normalized_data, columns=[f"{col}_normalized" for col in features_to_normalize], index=merged_df.index)
    
    # Combine with non-normalized columns
    final_df = pd.concat([merged_df[['Ticker']], normalized_df], axis=1)
    final_df.reset_index(inplace=True)
    
    # Rename columns to match the data dictionary
    final_df.rename(columns={'Adj Close_normalized': 'adj_close_normalized', 'Volume_normalized': 'volume_normalized'}, inplace=True)
    
    return final_df


def save_processed_data(df):
    """Saves the final processed dataframe to a CSV file."""
    df.to_csv(PROCESSED_OUTPUT_FILE, index=False)
    print(f"Successfully saved processed data to {PROCESSED_OUTPUT_FILE}")

if __name__ == "__main__":
    print("--- Starting Data Processing ---")
    create_output_directory()
    raw_stock_df, raw_macro_df = load_data()
    processed_df = process_data(raw_stock_df, raw_macro_df)
    save_processed_data(processed_df)
    print("--- Data Processing Complete ---")
    print(f"\nFinal processed data has {processed_df.shape[0]} rows and {processed_df.shape[1]} columns.")


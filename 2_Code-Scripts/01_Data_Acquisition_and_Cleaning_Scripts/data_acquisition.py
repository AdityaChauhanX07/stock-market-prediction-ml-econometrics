import yfinance as yf
import pandas as pd
import os

# --- Configuration ---
# Define the stock tickers and the date range for the data download.
TICKERS = ['^GSPC', 'TSLA']
START_DATE = '2010-01-01'
END_DATE = '2023-12-31'

# Define the output directory for the raw data.
OUTPUT_DIR = '01_Data_Files/Raw_Data'
STOCK_DATA_FILE = os.path.join(OUTPUT_DIR, 'stock_data_2010-2023.csv')
MACRO_DATA_FILE = os.path.join(OUTPUT_DIR, 'macroeconomic_indicators_raw.csv') # Note: In a real scenario, this would be downloaded from FRED API. Here we create it from a dictionary.

# --- Main Execution ---
def create_output_directory():
    """Creates the output directory if it doesn't exist."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")

def download_stock_data():
    """Downloads historical stock data from Yahoo Finance."""
    print(f"Downloading stock data for: {', '.join(TICKERS)}...")
    try:
        # Download data for all tickers at once
        stock_data = yf.download(TICKERS, start=START_DATE, end=END_DATE, group_by='ticker')
        
        # Restructure the dataframe from multi-index columns to a long format
        # This makes it easier to work with
        final_df = pd.DataFrame()
        for ticker in TICKERS:
            # For single ticker downloads, yfinance returns a simple DataFrame
            # For multiple, it's a multi-index one. We handle both.
            if len(TICKERS) > 1:
                ticker_df = stock_data[ticker].copy()
            else:
                ticker_df = stock_data.copy()
            
            ticker_df['Ticker'] = ticker
            ticker_df.reset_index(inplace=True)
            final_df = pd.concat([final_df, ticker_df])

        # Save to CSV
        final_df.to_csv(STOCK_DATA_FILE, index=False)
        print(f"Successfully downloaded and saved stock data to {STOCK_DATA_FILE}")
    except Exception as e:
        print(f"An error occurred during stock data download: {e}")

def generate_macro_data():
    """
    Generates a sample macroeconomic dataset.
    In a real-world scenario, this would involve API calls to sources like FRED.
    For reproducibility, we are creating a static file based on the provided research context.
    """
    print("Generating macroeconomic data file...")
    # This data is a simplified representation for demonstration.
    macro_data = {
        'Date': pd.to_datetime(['2010-01-31', '2010-02-28', '2023-11-30', '2023-12-31']),
        'GDP_Growth_Rate_Annualized': [1.5, 1.6, 2.5, 2.5],
        'Interest_Rate_Federal_Funds': [0.11, 0.13, 5.33, 5.33],
        'Inflation_Rate_CPI_YoY': [2.6, 2.1, 3.1, 3.4],
        'Unemployment_Rate': [9.8, 9.8, 3.7, 3.7]
    }
    macro_df = pd.DataFrame(macro_data)
    macro_df.to_csv(MACRO_DATA_FILE, index=False)
    print(f"Successfully generated and saved macroeconomic data to {MACRO_DATA_FILE}")


if __name__ == "__main__":
    print("--- Starting Data Acquisition ---")
    create_output_directory()
    download_stock_data()
    generate_macro_data()
    print("--- Data Acquisition Complete ---")

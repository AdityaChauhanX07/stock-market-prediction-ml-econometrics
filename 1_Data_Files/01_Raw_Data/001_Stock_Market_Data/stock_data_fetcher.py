import yfinance as yf
import pandas as pd

# 1. Define tickers and date range
# --- This list was missing ---
tickers = ['AAPL', 'GOOGL', 'MSFT'] # Example tickers
start_date = '2010-01-01'
end_date = '2023-12-31'

# yfinance's end date is exclusive, so we set it to the day after our target end date
# Note: The end date for yfinance is exclusive, so '2024-01-01' correctly includes data up to '2023-12-31'.
data_raw = yf.download(tickers, start=start_date, end='2024-01-01')

# 2. Process the multi-level column index into a tidy format
# We stack at level=1 (the ticker level) and rename the axes before resetting the index.
data_processed = data_raw.stack(level=1).rename_axis(['Date', 'Ticker']).reset_index()

# 3. Reorder columns to match a standard format (optional but good practice)
# --- This list was missing ---
final_columns = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
stock_data_df = data_processed[final_columns]

# 4. Save to CSV
stock_data_df.to_csv('stock_data_2010-2023.csv', index=False)

print("Data downloaded and saved successfully to 'stock_data_2010-2023.csv'")
# Historical Stock Market Data

## Data Sourcing and Retrieval Protocol

The historical stock data was programmatically retrieved from Yahoo Finance, a widely used and publicly accessible source for financial market data, ensuring that the acquisition process is fully replicable. The specific financial entities collected are those central to the paper's comparative analysis and case studies :  

* **S&P 500 Index** (Ticker: `^GSPC`): Selected as a proxy for the overall U.S. stock market, serving as a broad market benchmark for model scalability assessment.
* **NASDAQ Composite Index** (Ticker: `^IXIC`): Selected to represent the performance of the technology sector, which often exhibits distinct volatility and growth patterns.
* **Tesla Inc.** (Ticker: `TSLA`): Selected as a specific case study of a high-volatility, high-growth individual stock, allowing for a granular evaluation of model performance under challenging conditions.

Data for these tickers was downloaded for the full period from January 1, 2010, to December 31, 2023. The retrieval was executed using the Python `yfinance` library, a robust and popular tool for accessing Yahoo Finance's historical data API. The following code snippet demonstrates the core retrieval logic, which can be executed in any standard Python environment to replicate the raw data pull.

## File Specifications and Data Dictionary

The data is provided in a single, tidy-format CSV file named `stock_data_2010-2023.csv`. In this format, each row represents a single observation (a trading day for a specific ticker), which is ideal for time-series analysis in most software packages. The file structure is defined by the following data dictionary.

## Data Integrity and Validation Notes

The dataset is constructed to reflect actual market activity. Consequently, it contains entries only for days on which the U.S. stock markets were open; weekends and official market holidays are not present. The data is provided as-is from the Yahoo Finance source to maintain a raw, unprocessed state. This allows researchers to precisely follow the preprocessing steps detailed in the original paper's methodology, such as the imputation of missing values (to handle rare data feed errors) and the application of normalization techniques like Min-Max scaling.  

A known characteristic of Yahoo Finance data is the potential for floating-point imprecision in price fields, where a price might be represented as, for example, 249.179993 instead of 249.18. This is a common artifact of how decimal numbers are handled in binary computing systems. For the purposes of the models described in the paper, which operate on scaled values and relative changes, this level of imprecision is considered immaterial. However, users may wish to round the price data to two or four decimal places for display or specific calculation purposes.

**Works cited:**

1. How to Download Historical Data from Yahoo Finance - Macroption, accessed July 16, 2025, https://www.macroption.com/yahoo-finance-download-historical-data/
2. How to Download Historical Price Data In Excel Using Yahoo Finance, accessed July 16, 2025, https://365financialanalyst.com/knowledge-hub/trading-and-investing/how-to-download-historical-price-data-in-excel-using-yahoo-finance/
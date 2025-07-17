# Macroeconomic Indicators

## Data Sourcing from FRED

As justified previously, all macroeconomic indicators were sourced from the Federal Reserve Economic Data (FRED) database. This ensures data quality, transparency, and public accessibility. The indicators selected are those explicitly mentioned in the research paper as key drivers of market behavior: interest rates, GDP growth rates, and inflation. The specific time series retrieved from FRED are detailed below:  

* **Interest Rates:** The **Federal Funds Effective Rate** (FRED Series ID: `FEDFUNDS`) was chosen. This series represents the weighted average rate at which depository institutions lend reserve balances to other depository institutions overnight. It is the primary tool of the Federal Reserve for implementing monetary policy and serves as a benchmark for most short-term interest rates in the U.S.  

* **GDP Growth:** **Real Gross Domestic Product** (FRED Series ID: `GDPC1`) was used. This series measures the inflation-adjusted value of all goods and services produced in the United States. The quarterly growth rate, a key indicator of economic health, is calculated from the period-over-period percentage change in this series.  

* **Inflation:** The **Consumer Price Index for All Urban Consumers: All Items in U.S. City Average** (FRED Series ID: `CPIAUCSL`) was selected. This is the most widely cited measure of inflation, tracking the average change over time in the prices paid by urban consumers for a market basket of consumer goods and services. The annual or monthly inflation rate is calculated from the percentage change in this index.  

## File Specifications and Data Dictionary

The macroeconomic data is presented in an XLSX file named `macroeconomic_indicators_raw.xlsx`, with each indicator housed in a separate sheet. This format was chosen to preserve the valuable metadata provided by FRED, such as the series name, units of measurement, and native frequency, directly alongside the data. This information is crucial for correct interpretation and processing.
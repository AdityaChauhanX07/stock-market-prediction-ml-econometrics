# Cleaned and Processed Model Input Data

## Introduction

This component represents the culmination of the data preprocessing pipeline described in the research paper. The file `processed_stock_data_2010-2023.csv` is the final, analysis-ready dataset used as the direct input for the econometric, machine learning, and hybrid predictive models. It integrates the raw stock, macroeconomic, and sentiment data into a unified daily time-series format and includes a rich set of engineered features designed to capture complex market dynamics.

## Data Processing and Feature Engineering Pipeline

The creation of this processed dataset involves a multi-step pipeline that transforms the raw data sources into a feature-rich matrix. This process is critical for the reproducibility of the paper's results.

1. **Data Unification and Temporal Alignment**: The raw stock, macroeconomic, and sentiment data are first loaded. The lower-frequency macroeconomic data (monthly interest rates/CPI, quarterly GDP) are upsampled to a daily frequency using a **forward-fill** (`ffill`) method. This approach ensures that the value of an indicator is carried forward from its release date until a new value is published, accurately reflecting the information available to the market on any given day.

2. **Sentiment Quantification and Aggregation**: A sentiment score is calculated for each news headline in `sentiment_data_sources.csv`. Following the paper's mention of advanced NLP models, this can be implemented using a pre-trained transformer model like FinBERT. The resulting sentiment scores, which are tied to specific publication timestamps, are then aggregated to a daily level for each ticker by calculating the **mean sentiment score** of all news published for that ticker on a given day. Days with no news for a ticker result in a missing sentiment value, which is handled in a later imputation step.

3. **Technical Indicator Generation**: A suite of standard technical indicators is engineered from the daily price data to capture market momentum, trend, and volatility. These are standard inputs for many financial machine learning models.

   * **Moving Averages (MA)**: 20-day and 50-day simple moving averages of the `Adj Close` price are calculated to capture short- and medium-term trends.

   * **Rolling Volatility**: The 20-day rolling standard deviation of daily returns (`Adj Close` percentage change) is calculated as a simple measure of historical volatility.

   * **Relative Strength Index (RSI)**: A 14-day RSI is calculated to measure the speed and change of price movements, identifying potential overbought or oversold conditions.

   * **Moving Average Convergence Divergence (MACD)**: The MACD line, signal line, and histogram are calculated using standard periods (12-day fast EMA, 26-day slow EMA, 9-day signal EMA) to identify changes in momentum.

4. **Econometric Feature Generation**: In line with the paper's hybrid modeling approach, features are derived from standalone econometric models.

   * **GARCH Conditional Volatility**: A GARCH(1,1) model is fitted to the daily returns of each ticker to generate a series of daily conditional volatility estimates. This provides a more sophisticated, forward-looking measure of volatility than simple rolling standard deviation.

   * **ARIMA Residuals**: An ARIMA(p,d,q) model is fitted to each `Adj Close` price series. The **residuals** (the difference between the actual prices and the model's one-step-ahead predictions) are extracted. These residuals represent the portion of price movement not explained by linear time-series patterns and are used as a feature to help machine learning models capture non-linear dynamics.

5. **Final Cleaning and Normalization**:
   * **Missing Value Imputation**: After merging and feature engineering, missing values (e.g., at the start of rolling window calculations or for sentiment on no-news days) are filled using the **forward-fill** (`ffill`) method to ensure a complete dataset.

   * **Outlier Capping**: As specified in the paper, outliers in the numerical features are handled by capping. For each feature, the Z-score is calculated. Any value with a Z-score greater than 3 or less than -3 is replaced with the value corresponding to the 3rd or -3rd standard deviation, respectively, to mitigate the influence of extreme data points without removing them.

   * **Min-Max Scaling**: Finally, all numerical features in the dataset undergo **Min-Max scaling** to transform them into a uniform range of . This normalization step is crucial for the stable training of many machine learning algorithms, especially neural networks like LSTMs.

## File Specifications and Data Dictionary

The final processed data is stored in `processed_stock_data_2010-2023.csv`. Each row corresponds to a single trading day for a specific ticker, containing the raw price data alongside all the engineered and processed features.

**Works Cited:**

1. Working with missing data — pandas 0.12.0 documentation, accessed July 16, 2025, https://pandas.pydata.org/pandas-docs/version/0.12.0/missing_data.html
2. Understanding pandas.aggregate() with Simple Examples | by why amit - Medium, accessed July 16, 2025, https://medium.com/@whyamit101/understanding-pandas-aggregate-with-simple-examples-aff4ceb9b329
3. Iterate over DateTimeIndex to get overall sentiment per day - Stack Overflow, accessed July 16, 2025, https://stackoverflow.com/questions/57171317/iterate-over-datetimeindex-to-get-overall-sentiment-per-day
4. How to calculate MOVING AVERAGE in a Pandas DataFrame? - GeeksforGeeks, accessed July 16, 2025, https://www.geeksforgeeks.org/pandas/how-to-calculate-moving-average-in-a-pandas-dataframe/
5. pandas.core.window.rolling.Rolling.std — pandas 2.3.1 documentation - PyData |, accessed July 16, 2025, https://pandas.pydata.org/docs/reference/api/pandas.core.window.rolling.Rolling.std.html
6. RSI MACD Crossover - Phoenix, accessed July 16, 2025, https://algobulls.github.io/pyalgotrading/strategies/rsi_macd_crossover/
7. Technical analysis with Python: 3 indicators you can learn in 2.5 minutes - PyQuant News, accessed July 16, 2025, https://www.pyquantnews.com/the-pyquant-newsletter/technical-analysis-python-3-indicators
8. Stock Technical Indicators for TESLA (MACD & RSI) - Kaggle, accessed July 16, 2025, https://www.kaggle.com/code/korfanakis/stock-technical-indicators-for-tesla-macd-rsi
9. Building a MACD Indicator in Python - Medium, accessed July 16, 2025, https://medium.com/@financial_python/building-a-macd-indicator-in-python-190b2a4c1777
10. GARCH vs. GJR-GARCH Models in Python for Volatility Forecasting - QuantInsti Blog, accessed July 16, 2025, https://blog.quantinsti.com/garch-gjr-garch-volatility-forecasting-python/
11. How to Model Volatility with ARCH and GARCH for Time Series Forecasting in Python, accessed July 16, 2025, https://machinelearningmastery.com/develop-arch-and-garch-models-for-time-series-forecasting-in-python/
12. ARCH Modeling - arch 7.2.0, accessed July 16, 2025, https://arch.readthedocs.io/en/latest/univariate/univariate_volatility_modeling.html
13. Model Residuals - Python - Codecademy, accessed July 16, 2025, https://www.codecademy.com/resources/docs/python/statsmodels/model-residuals
14. ARIMA fit model and residuals - GitHub Gist, accessed July 16, 2025, https://gist.github.com/kangeugine/9aaa186e6674d731d03e6594a3a04cdf
15. Python Pandas DataFrame fillna() - Fill Missing Values - Vultr Docs, accessed July 16, 2025, https://docs.vultr.com/python/third-party/pandas/DataFrame/fillna
16. Handling outliers with Pandas - GeeksforGeeks, accessed July 16, 2025, https://www.geeksforgeeks.org/pandas/handling-outliers-with-pandas/
17. How to Detect and Exclude Outliers in a Pandas DataFrame | Saturn Cloud Blog, accessed July 16, 2025, https://saturncloud.io/blog/how-to-detect-and-exclude-outliers-in-a-pandas-dataframe/
18. MinMaxScaler — scikit-learn 1.7.0 documentation, accessed July 16, 2025, https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
19. 7.3. Preprocessing data — scikit-learn 1.7.0 documentation, accessed July 16, 2025, https://scikit-learn.org/stable/modules/preprocessing.html
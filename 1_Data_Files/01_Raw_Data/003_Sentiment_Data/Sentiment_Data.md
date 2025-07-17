# Sentiment Data

## Curation Strategy and Sourcing

The original paper is non-specific about the source of its "sentiment data," a common issue that hinders reproducibility in NLP-based financial modeling. To create a concrete and replicable foundation for this component of the study, this data package utilizes the  

**FNSPID (Financial News and Stock Price Integration Dataset)**. This choice was made based on several key criteria:  

1. **Relevance:** FNSPID focuses specifically on financial news for S&P 500 companies, making it highly relevant to the entities studied in the paper.  

2. **Temporal Coverage:** The dataset spans from 1999 to 2023, fully encompassing the 2010-2023 period required for this research.  

3. **Accessibility and Citatability:** FNSPID is publicly available through a Hugging Face repository and is documented in a citable academic paper (`arXiv:2402.06698`), adhering to best practices in open science.  

The `sentiment_data_sources.csv` file provided in this package is a curated subset of the full FNSPID dataset. It has been pre-filtered to include news headlines from 2010-2023 that are specifically relevant to the S&P 500, NASDAQ, and Tesla, providing a convenient starting point for replicating the paper's sentiment analysis.

## File Specifications and Data Dictionary

The file contains the raw text of news headlines and associated metadata, which serve as the direct input for sentiment scoring algorithms. Each row in the CSV corresponds to a single published news article.

## Role in Reproducing NLP Models

This file provides the essential corpus of text that serves as the input to the natural language processing models mentioned in the paper's literature review, such as BERT. The methodology described in the paper involves several steps that begin with this data:  

1. **Sentiment Scoring:** A pre-trained language model is applied to each entry in the `headline` column to generate a numerical sentiment score (e.g., ranging from -1 for negative to +1 for positive).
2. **Temporal Aggregation:** The sentiment scores, which are event-based (tied to the publication time of each article), are then aggregated to a daily frequency. This typically involves grouping the news by `date` and `ticker` and calculating an aggregate statistic, such as the mean, median, or volume-weighted average sentiment score for each day.
3. **Feature Engineering:** The resulting daily sentiment score for each ticker becomes a new feature, which is then merged with the stock price and macroeconomic data to form the final input for the hybrid forecasting models.

**Works Cited:**

1. Zdong104/FNSPID_Financial_News_Dataset: FNSPID: A ... - GitHub, accessed July 16, 2025, https://github.com/Zdong104/FNSPID_Financial_News_Dataset
2. FNSPID: A Comprehensive Financial News Dataset in Time Series - IDEAS/RePEc, accessed July 16, 2025, https://ideas.repec.org/p/arx/papers/2402.06698.html
3. Beki78/Financial-News-Analysis - GitHub, accessed July 16, 2025, https://github.com/Beki78/Financial-News-Analysis/
4. shewanek/KAIM_W1_FinanceNewsAnalytics - GitHub, accessed July 16, 2025, https://github.com/shewanek/KAIM_W1_FinanceNewsAnalytics
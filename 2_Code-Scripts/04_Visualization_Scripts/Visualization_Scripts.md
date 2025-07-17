# Code and Scripts

## Visualization Scripts

This script is responsible for generating the plots and figures presented in the paper, allowing for visual inspection of model results and data trends.

### `visualization.py`

**Purpose**: To create all visualizations used in the research paper for model diagnostics and comparative analysis.

**Libraries**: `pandas`, `matplotlib`, `seaborn`.

**Process**:

1. **Load Results**: Ingests the performance metrics from `model_performance.csv` and the test data containing actual values and model predictions.

2. **Generate Predicted vs. Actual Plot**: Creates a time-series line plot to replicate Figure 2 from the paper, overlaying the predicted stock prices from the hybrid model against the actual closing prices for the Tesla (TSLA) case study.

3. **Generate Model Comparison Plots**: Produces bar charts to visually represent the data from Table I, comparing the RMSE, MAE, and R-squared metrics across all tested models (ARIMA, GARCH, LSTM, Hybrid). This provides a clear visual summary of the comparative analysis.

4. **Generate Diagnostic Plots**: Creates diagnostic plots for the econometric models, such as plotting the residuals of the ARIMA model to visually inspect for any remaining patterns or biases.

**Output**: Saved image files for each figure (e.g., `figure_2_tesla_prediction.png`, `figure_metrics_comparison.png`).
# Code and Scripts

## Model Training and Evaluation Scripts `(model_training_and_evaluation.py)`

This collection of scripts covers the core modeling phase of the research, from training and tuning the models to evaluating their final performance.

* `model_training`
   * **Purpose**: To train the individual and hybrid models described in the paper using the final processed feature set.
   * **Libraries**: `pandas`, `numpy`, `statsmodels`, `arch`, `scikit-learn`, `tensorflow`, `joblib`.
   * **Process**:
      1. **Load Data**: Loads the `processed_stock_data_2010-2023.csv` dataset.
      2. **Data Splitting**: Splits the data chronologically into training and testing sets.
      3. **Model Implementation**:
         * Trains the **ARIMA** and **GARCH** models on the relevant time series from the training data.
         * Trains the machine learning models (**LSTM**, **Random Forest**, **SVM**) on the full feature matrix from the training data.
         * Implements and trains the **Hybrid Model**, for example, by feeding the residuals from the trained ARIMA model as an additional feature into the LSTM network, as described in the paper.
      4. **Model Persistence**: Saves the trained models to disk using `joblib` for scikit-learn models and the `.h5` format for the TensorFlow/Keras LSTM model.
   * **Output**: Serialized model files (e.g., `arima_model.pkl`, `lstm_model.h5`, `svm_model.joblib`).

* `hyperparameter_tuning`
   * **Purpose**: To systematically find the optimal hyperparameters for the machine learning models to ensure they generalize well and achieve maximum performance, as outlined in the methodology.
   * **Libraries**: `scikit-learn` (`GridSearchCV`), `keras-tuner`.
   * **Process**:
      1. **Load Training Data**: Loads the training subset of the data.
      2. **Define Search Space**: For each machine learning model, a grid of potential hyperparameters is defined.
         * **SVM**: A grid search is performed to find the optimal `C` and `gamma` parameters for the RBF kernel.
         * **Random Forest**: The search space includes parameters like the number of trees and tree depth.
         * **LSTM**: The search space includes the number of LSTM units, dropout rates, and learning rates for the Adam optimizer.
      3. **Execution**: Uses cross-validation techniques like `GridSearchCV` on the training data to evaluate each combination of hyperparameters against a performance metric (e.g., RMSE).
   * **Output**: A report or log file (`hyperparameter_tuning_results.json`) containing the best-performing hyperparameters for each model.

* `model_evaluation`
   * **Purpose**: To evaluate the performance of the final, tuned models on the unseen test data and reproduce the comparative analysis presented in the paper.
   * **Libraries**: `pandas`, `scikit-learn.metrics`, `matplotlib`, `seaborn`.
   * **Process**:
      1. **Load Models and Test Data**: Loads the saved models from `model_training.py` and the test dataset.
      2. **Prediction**: Generates predictions for the test set using each of the trained models (ARIMA, GARCH, LSTM, Hybrid, etc.).
      3. **Metric Calculation**: Calculates the key performance metrics specified in the paper: **RMSE**, **MAE**, and **R-squared**.
      4. **Results Generation**:
         * Creates a summary table, similar to Table I in the paper, comparing the performance metrics across all models.
         * Generates visualizations, such as the predicted vs. actual price plot for the Tesla case study shown in Figure 2.
   * **Output**: A CSV file with the final performance metrics (`model_performance.csv`) and saved figures (`tesla_prediction_plot.png`).
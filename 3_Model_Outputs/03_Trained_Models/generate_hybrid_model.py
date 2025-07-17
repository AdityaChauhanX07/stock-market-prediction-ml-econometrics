import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

# For ARIMA component
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import os

class HybridARIMALSTM:
    def __init__(self, arima_order=(1,1,1), lstm_units=50, sequence_length=10):
        self.arima_order = arima_order
        self.lstm_units = lstm_units
        self.sequence_length = sequence_length
        self.arima_model = None
        self.lstm_model = None
        self.scaler = MinMaxScaler()
        self.residual_scaler = MinMaxScaler()
        
    def generate_sample_data(self, n_samples=1000):
        """Generate synthetic time series data for demonstration"""
        np.random.seed(42)
        
        # Generate trend component
        trend = np.linspace(100, 200, n_samples)
        
        # Generate seasonal component
        seasonal = 10 * np.sin(2 * np.pi * np.arange(n_samples) / 50)
        
        # Generate noise
        noise = np.random.normal(0, 5, n_samples)
        
        # Combine components
        data = trend + seasonal + noise
        
        # Add some non-linear patterns for LSTM to capture
        for i in range(1, n_samples):
            if i % 100 == 0:
                data[i:i+10] += np.random.normal(0, 15, min(10, n_samples-i))
        
        return pd.Series(data, index=pd.date_range('2020-01-01', periods=n_samples, freq='D'))
    
    def prepare_lstm_data(self, residuals):
        """Prepare data for LSTM training"""
        # Scale residuals
        residuals_scaled = self.residual_scaler.fit_transform(residuals.values.reshape(-1, 1))
        
        # Create sequences
        X, y = [], []
        for i in range(self.sequence_length, len(residuals_scaled)):
            X.append(residuals_scaled[i-self.sequence_length:i, 0])
            y.append(residuals_scaled[i, 0])
        
        return np.array(X), np.array(y)
    
    def build_lstm_model(self, input_shape):
        """Build LSTM model for residual forecasting"""
        model = Sequential([
            LSTM(self.lstm_units, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(self.lstm_units // 2, return_sequences=False),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def fit(self, data, validation_split=0.2):
        """Fit the hybrid ARIMA-LSTM model"""
        print("Training Hybrid ARIMA-LSTM Model...")
        
        # Split data
        split_idx = int(len(data) * (1 - validation_split))
        train_data = data[:split_idx]
        val_data = data[split_idx:]
        
        # Step 1: Fit ARIMA model
        print("Step 1: Fitting ARIMA model...")
        try:
            self.arima_model = ARIMA(train_data, order=self.arima_order)
            self.arima_fitted = self.arima_model.fit()
            print(f"ARIMA{self.arima_order} model fitted successfully")
        except Exception as e:
            print(f"ARIMA fitting failed: {e}")
            # Use simpler model if ARIMA fails
            self.arima_model = ARIMA(train_data, order=(1,0,0))
            self.arima_fitted = self.arima_model.fit()
        
        # Get ARIMA predictions and residuals
        arima_pred = self.arima_fitted.fittedvalues
        residuals = train_data - arima_pred
        
        # Step 2: Prepare LSTM data for residuals
        print("Step 2: Preparing LSTM data...")
        X_lstm, y_lstm = self.prepare_lstm_data(residuals)
        
        # Reshape for LSTM
        X_lstm = X_lstm.reshape((X_lstm.shape[0], X_lstm.shape[1], 1))
        
        # Split LSTM data
        lstm_split = int(len(X_lstm) * 0.8)
        X_train, X_val = X_lstm[:lstm_split], X_lstm[lstm_split:]
        y_train, y_val = y_lstm[:lstm_split], y_lstm[lstm_split:]
        
        # Step 3: Build and train LSTM model
        print("Step 3: Building and training LSTM model...")
        self.lstm_model = self.build_lstm_model((X_lstm.shape[1], 1))
        
        # Callbacks
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)
        
        # Train LSTM
        history = self.lstm_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        
        print("Hybrid model training completed!")
        return history
    
    def predict(self, data, steps=1):
        """Make predictions using the hybrid model"""
        predictions = []
        
        for _ in range(steps):
            # ARIMA prediction
            arima_pred = self.arima_fitted.forecast(steps=1)[0]
            
            # Get recent residuals for LSTM
            recent_residuals = data[-self.sequence_length:] - self.arima_fitted.fittedvalues[-self.sequence_length:]
            recent_residuals_scaled = self.residual_scaler.transform(recent_residuals.values.reshape(-1, 1))
            
            # LSTM prediction on residuals
            lstm_input = recent_residuals_scaled.reshape(1, self.sequence_length, 1)
            lstm_pred_scaled = self.lstm_model.predict(lstm_input, verbose=0)
            lstm_pred = self.residual_scaler.inverse_transform(lstm_pred_scaled)[0, 0]
            
            # Combine predictions
            hybrid_pred = arima_pred + lstm_pred
            predictions.append(hybrid_pred)
        
        return np.array(predictions)
    
    def create_combined_model(self):
        """Create a combined Keras model for easy saving/loading"""
        # Create a wrapper model that includes both ARIMA parameters and LSTM
        
        # Input for LSTM part
        lstm_input = Input(shape=(self.sequence_length, 1), name='lstm_input')
        
        # LSTM layers
        lstm_out = LSTM(self.lstm_units, return_sequences=True)(lstm_input)
        lstm_out = Dropout(0.2)(lstm_out)
        lstm_out = LSTM(self.lstm_units // 2, return_sequences=False)(lstm_out)
        lstm_out = Dropout(0.2)(lstm_out)
        lstm_out = Dense(25, activation='relu')(lstm_out)
        lstm_pred = Dense(1, name='lstm_output')(lstm_out)
        
        # Input for ARIMA prediction (as a feature)
        arima_input = Input(shape=(1,), name='arima_input')
        
        # Combine ARIMA and LSTM predictions
        combined = concatenate([arima_input, lstm_pred])
        final_output = Dense(1, activation='linear', name='hybrid_output')(combined)
        
        # Create the combined model
        combined_model = Model(inputs=[arima_input, lstm_input], outputs=final_output)
        combined_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        
        return combined_model

def generate_hybrid_model_file():
    """Generate the hybrid_model.h5 file"""
    print("Generating hybrid_model.h5 file...")
    
    # Create hybrid model instance
    hybrid = HybridARIMALSTM(arima_order=(2,1,1), lstm_units=64, sequence_length=15)
    
    # Generate sample data
    print("Generating sample time series data...")
    sample_data = hybrid.generate_sample_data(n_samples=1000)
    
    # Train the hybrid model
    history = hybrid.fit(sample_data, validation_split=0.2)
    
    # Create the combined Keras model
    print("Creating combined Keras model...")
    combined_model = hybrid.create_combined_model()
    
    # Transfer weights from the trained LSTM to the combined model
    # Get the LSTM layers from both models
    lstm_layers_trained = [layer for layer in hybrid.lstm_model.layers if 'lstm' in layer.name.lower() or 'dense' in layer.name.lower()]
    lstm_layers_combined = [layer for layer in combined_model.layers if 'lstm' in layer.name.lower() or 'dense' in layer.name.lower()]
    
    # Copy weights
    for i, (trained_layer, combined_layer) in enumerate(zip(lstm_layers_trained, lstm_layers_combined)):
        if trained_layer.get_weights():
            combined_layer.set_weights(trained_layer.get_weights())
    
    # Create directory if it doesn't exist
    os.makedirs('Trained_Models', exist_ok=True)
    
    # Save the model
    model_path = 'Trained_Models/hybrid_model.h5'
    combined_model.save(model_path)
    
    print(f"✅ Hybrid model saved successfully to {model_path}")
    
    # Save additional model information
    model_info = {
        'arima_order': hybrid.arima_order,
        'lstm_units': hybrid.lstm_units,
        'sequence_length': hybrid.sequence_length,
        'arima_params': hybrid.arima_fitted.params.tolist() if hybrid.arima_fitted else None,
        'training_samples': len(sample_data),
        'model_type': 'Hybrid ARIMA-LSTM',
        'framework': 'TensorFlow/Keras + statsmodels'
    }
    
    # Save model metadata
    import json
    with open('Trained_Models/hybrid_model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print("✅ Model metadata saved to Trained_Models/hybrid_model_info.json")
    
    # Demonstration of loading the model
    print("\n" + "="*50)
    print("DEMONSTRATION: Loading the saved model")
    print("="*50)
    
    # Load the model
    from tensorflow.keras.models import load_model
    loaded_model = load_model(model_path)
    
    print(f"✅ Model loaded successfully from {model_path}")
    print(f"Model input shape: {loaded_model.input_shape}")
    print(f"Model output shape: {loaded_model.output_shape}")
    print(f"Total parameters: {loaded_model.count_params():,}")
    
    # Show model summary
    print("\nModel Architecture:")
    loaded_model.summary()
    
    return model_path, loaded_model

# Execute the generation
if __name__ == "__main__":
    try:
        model_path, loaded_model = generate_hybrid_model_file()
        print(f"\n🎉 SUCCESS! Your hybrid_model.h5 file has been generated at: {model_path}")
        print("\nTo use this model in another script:")
        print("```python")
        print("from tensorflow.keras.models import load_model")
        print("trained_model = load_model('Trained_Models/hybrid_model.h5')")
        print("```")
        
    except Exception as e:
        print(f"❌ Error generating model: {e}")
        import traceback
        traceback.print_exc()
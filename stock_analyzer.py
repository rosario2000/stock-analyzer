import requests
import pandas as pd
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import time
import csv
import urllib3
import numpy as np
import warnings

# Deep Learning imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    # Create dummy classes for type hints when TensorFlow is not available
    class keras:
        class Model: pass
    class MinMaxScaler:
        def __init__(self): pass
    print("Warning: TensorFlow not available. LSTM predictions will be disabled.")

# Suppress SSL warnings for testing
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

class StockAnalyzer:
    def __init__(self, api_key: str = "R2UOE0T2GAU721CI", use_lstm: bool = True, lstm_lookback: int = 60):
        """
        Initialize the Advanced Stock Analyzer with enhanced accuracy features
        
        Args:
            api_key: AlphaVantage API key
            use_lstm: Whether to use LSTM neural network for predictions
            lstm_lookback: Number of days to look back for LSTM training
        """
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        self.use_lstm = use_lstm and TENSORFLOW_AVAILABLE
        self.lstm_lookback = lstm_lookback
        self.scalers = {}  # To store MinMaxScalers for each stock
        
        # Enhanced accuracy features
        self.use_ensemble = True  # Multiple LSTM models for better predictions
        self.use_technical_indicators = True  # Add technical analysis features
        self.use_confidence_intervals = True  # Provide prediction ranges
        self.use_risk_adjustment = True  # Risk-adjusted predictions
        
        if self.use_lstm:
            # Set TensorFlow to use less verbose logging
            tf.get_logger().setLevel('ERROR')
            print("✓ Advanced LSTM system enabled with accuracy enhancements")
            print("  🎯 Ensemble modeling, technical indicators, confidence intervals")
        
    def fetch_stock_data(self, symbol: str) -> Optional[Dict]:
        """
        Fetch stock data from AlphaVantage API
        """
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': symbol,
            'outputsize': 'full',
            'apikey': self.api_key
        }
        
        try:
            print(f"Fetching data for {symbol}...")
            # Temporarily disable SSL verification for testing (corporate network issue)
            response = requests.get(self.base_url, params=params, verify=False, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Check for API errors
            if "Error Message" in data:
                print(f"API Error for {symbol}: {data['Error Message']}")
                return None
            elif "Note" in data:
                print(f"API Rate limit for {symbol}: {data['Note']}")
                return None
            elif "Time Series (Daily)" not in data:
                print(f"No time series data found for {symbol}")
                return None
                
            return data
        except requests.exceptions.RequestException as e:
            print(f"Request error for {symbol}: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"JSON decode error for {symbol}: {e}")
            return None
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add advanced technical indicators for better prediction accuracy
        
        Returns enhanced DataFrame with technical analysis features
        """
        df = df.copy()
        
        # Price-based indicators
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Moving Averages (multiple timeframes)
        for window in [5, 10, 20, 50, 200]:
            df[f'ma_{window}'] = df['close'].rolling(window=window).mean()
            df[f'ma_{window}_ratio'] = df['close'] / df[f'ma_{window}']
        
        # Exponential Moving Averages
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # RSI (Relative Strength Index)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Volume indicators
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Volatility measures
        df['volatility'] = df['returns'].rolling(window=20).std()
        df['high_low_ratio'] = df['high'] / df['low']
        df['price_range'] = (df['high'] - df['low']) / df['close']
        
        # Momentum indicators
        df['momentum_5'] = df['close'] / df['close'].shift(5)
        df['momentum_10'] = df['close'] / df['close'].shift(10)
        
        # Support/Resistance levels
        df['resistance'] = df['high'].rolling(window=20).max()
        df['support'] = df['low'].rolling(window=20).min()
        df['support_resistance_ratio'] = (df['close'] - df['support']) / (df['resistance'] - df['support'])
        
        return df
    
    def prepare_lstm_data(self, df: pd.DataFrame, lookback: int = None) -> tuple:
        """
        Prepare enhanced data for LSTM training with technical indicators
        
        Args:
            df: DataFrame with stock data and technical indicators
            lookback: Number of days to look back (uses self.lstm_lookback if None)
            
        Returns:
            Tuple of (X_train, y_train, scaler, scaled_data)
        """
        if lookback is None:
            lookback = self.lstm_lookback
        
        # Enhanced features including technical indicators
        if self.use_technical_indicators:
            features = [
                'close', 'high', 'low', 'open', 'volume',  # Basic OHLCV
                'ma_5_ratio', 'ma_20_ratio', 'ma_50_ratio',  # Moving average ratios
                'rsi', 'macd', 'macd_histogram',  # Momentum indicators
                'bb_position', 'volume_ratio', 'volatility',  # Market condition indicators
                'momentum_5', 'momentum_10', 'support_resistance_ratio'  # Additional momentum
            ]
        else:
            features = ['close', 'high', 'low', 'open', 'volume']
        
        # Select available features (some may be NaN due to rolling calculations)
        available_features = [f for f in features if f in df.columns]
        
        # Get the data and handle NaN values (using forward fill and backward fill)
        data = df[available_features].ffill().bfill().values
        
        # Scale the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
        
        # Create sequences
        X, y = [], []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i])
            y.append(scaled_data[i, 0])  # Predict close price (first feature)
        
        return np.array(X), np.array(y), scaler, scaled_data
    
    def create_lstm_model(self, input_shape: tuple) -> keras.Model:
        """
        Create LSTM neural network model
        
        Args:
            input_shape: Shape of input data (timesteps, features)
            
        Returns:
            Compiled Keras model
        """
        model = keras.Sequential()
        
        # Add input layer explicitly to avoid warning
        model.add(keras.Input(shape=input_shape))
        
        # First LSTM layer with dropout for regularization
        model.add(layers.LSTM(32, return_sequences=True))  # Reduced from 50 to 32 for speed
        model.add(layers.Dropout(0.2))
        
        # Second LSTM layer (simplified - removed third layer for speed)
        model.add(layers.LSTM(32, return_sequences=False))
        model.add(layers.Dropout(0.2))
        
        # Dense layers for final prediction
        model.add(layers.Dense(16))  # Reduced from 25 to 16
        model.add(layers.Dense(1))
        
        # Compile model with Adam optimizer
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mean_squared_error',
            metrics=['mae']
        )
        
        return model
    
    def train_lstm_model(self, X_train: np.ndarray, y_train: np.ndarray) -> keras.Model:
        """
        Train LSTM model on historical data
        
        Args:
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Trained Keras model
        """
        # Create model
        model = self.create_lstm_model((X_train.shape[1], X_train.shape[2]))
        
        # Early stopping to prevent overfitting
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train model (reduced epochs for faster training)
        model.fit(
            X_train, y_train,
            epochs=20,  # Reduced from 50 to 20 for speed
            batch_size=64,  # Increased batch size for speed
            verbose=0,  # Silent training
            callbacks=[early_stopping],
            validation_split=0.2
        )
        
        return model
    
    def create_ensemble_models(self, input_shape: tuple, n_models: int = 3) -> List:
        """
        Create multiple LSTM models with different architectures for ensemble predictions
        
        Args:
            input_shape: Shape of input data
            n_models: Number of models in ensemble
            
        Returns:
            List of trained models with different configurations
        """
        models = []
        
        # Model 1: Conservative (smaller, more regularization)
        model1 = keras.Sequential()
        model1.add(keras.Input(shape=input_shape))
        model1.add(layers.LSTM(24, return_sequences=True))
        model1.add(layers.Dropout(0.3))
        model1.add(layers.LSTM(24, return_sequences=False))
        model1.add(layers.Dropout(0.3))
        model1.add(layers.Dense(12))
        model1.add(layers.Dense(1))
        model1.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005), loss='mse')
        
        # Model 2: Balanced (default configuration)
        model2 = self.create_lstm_model(input_shape)
        
        # Model 3: Aggressive (larger, less regularization)
        model3 = keras.Sequential()
        model3.add(keras.Input(shape=input_shape))
        model3.add(layers.LSTM(48, return_sequences=True))
        model3.add(layers.Dropout(0.1))
        model3.add(layers.LSTM(48, return_sequences=False))
        model3.add(layers.Dropout(0.1))
        model3.add(layers.Dense(24))
        model3.add(layers.Dense(1))
        model3.compile(optimizer=keras.optimizers.Adam(learning_rate=0.002), loss='mse')
        
        models = [model1, model2, model3]
        return models[:n_models]
    
    def validate_model_performance(self, model, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Validate model performance using test data
        
        Args:
            model: Trained model
            X_test: Test features  
            y_test: Test targets
            
        Returns:
            Dictionary with performance metrics
        """
        if len(X_test) == 0:
            return {'mse': 0, 'mae': 0, 'accuracy_score': 0}
        
        # Make predictions
        predictions = model.predict(X_test, verbose=0)
        
        # Calculate metrics
        mse = np.mean((predictions.flatten() - y_test) ** 2)
        mae = np.mean(np.abs(predictions.flatten() - y_test))
        
        # Direction accuracy (did we predict the right direction?)
        pred_direction = np.sign(predictions.flatten()[1:] - predictions.flatten()[:-1])
        actual_direction = np.sign(y_test[1:] - y_test[:-1])
        direction_accuracy = np.mean(pred_direction == actual_direction) * 100 if len(pred_direction) > 0 else 0
        
        return {
            'mse': round(mse, 6),
            'mae': round(mae, 6), 
            'direction_accuracy': round(direction_accuracy, 1)
        }
    
    def train_ensemble_models(self, X_train: np.ndarray, y_train: np.ndarray) -> List:
        """
        Train ensemble of LSTM models for robust predictions
        
        Args:
            X_train: Training features
            y_train: Training targets
            
        Returns:
            List of trained models
        """
        models = self.create_ensemble_models((X_train.shape[1], X_train.shape[2]))
        trained_models = []
        
        for i, model in enumerate(models):
            print(f"    🔄 Training ensemble model {i+1}/{len(models)}...")
            
            # Different training strategies for diversity
            if i == 0:  # Conservative model - more epochs, smaller batch
                epochs, batch_size = 30, 32
            elif i == 1:  # Balanced model - default settings
                epochs, batch_size = 20, 64
            else:  # Aggressive model - fewer epochs, larger batch
                epochs, batch_size = 15, 128
            
            early_stopping = keras.callbacks.EarlyStopping(
                monitor='loss', patience=8, restore_best_weights=True
            )
            
            model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                verbose=0,
                callbacks=[early_stopping],
                validation_split=0.2
            )
            
            trained_models.append(model)
        
        return trained_models
    
    def predict_with_ensemble(self, models: List, last_sequence: np.ndarray, 
                            scaler: MinMaxScaler, days_ahead: int = 30) -> Dict:
        """
        Generate ensemble predictions with confidence intervals
        
        Args:
            models: List of trained LSTM models
            last_sequence: Last sequence for prediction
            scaler: Fitted scaler
            days_ahead: Days to predict ahead
            
        Returns:
            Dictionary with ensemble predictions and confidence intervals
        """
        all_predictions = []
        
        # Get predictions from each model
        for model in models:
            predictions = self.predict_with_lstm_single(model, last_sequence, scaler, days_ahead)
            all_predictions.append(predictions)
        
        # Convert to numpy array for easier manipulation
        all_predictions = np.array(all_predictions)  # Shape: (n_models, days_ahead)
        
        # Calculate ensemble statistics
        mean_predictions = np.mean(all_predictions, axis=0)
        std_predictions = np.std(all_predictions, axis=0)
        
        # Calculate confidence intervals (assuming normal distribution)
        confidence_95_lower = mean_predictions - 1.96 * std_predictions
        confidence_95_upper = mean_predictions + 1.96 * std_predictions
        confidence_68_lower = mean_predictions - std_predictions
        confidence_68_upper = mean_predictions + std_predictions
        
        return {
            'mean_predictions': mean_predictions,
            'std_predictions': std_predictions,
            'confidence_95_lower': confidence_95_lower,
            'confidence_95_upper': confidence_95_upper,
            'confidence_68_lower': confidence_68_lower,
            'confidence_68_upper': confidence_68_upper,
            'individual_predictions': all_predictions
        }
    
    def predict_with_lstm_single(self, model: keras.Model, last_sequence: np.ndarray, 
                               scaler: MinMaxScaler, days_ahead: int = 30) -> np.ndarray:
        """
        Make predictions using a single LSTM model (helper method)
        """
        predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(days_ahead):
            pred = model.predict(current_sequence.reshape(1, current_sequence.shape[0], current_sequence.shape[1]), verbose=0)
            predictions.append(pred[0, 0])
            
            # Update sequence
            new_row = current_sequence[-1].copy()
            new_row[0] = pred[0, 0]
            current_sequence = np.roll(current_sequence, -1, axis=0)
            current_sequence[-1] = new_row
        
        # Inverse transform predictions
        predictions = np.array(predictions).reshape(-1, 1)
        dummy_array = np.zeros((len(predictions), scaler.n_features_in_))
        dummy_array[:, 0] = predictions.flatten()
        predictions_rescaled = scaler.inverse_transform(dummy_array)[:, 0]
        
        return predictions_rescaled
    
    def predict_with_lstm(self, model: keras.Model, last_sequence: np.ndarray, 
                         scaler: MinMaxScaler, days_ahead: int = 30) -> np.ndarray:
        """
        Make predictions using trained LSTM model
        
        Args:
            model: Trained LSTM model
            last_sequence: Last sequence of data for prediction
            scaler: Fitted MinMaxScaler
            days_ahead: Number of days to predict ahead
            
        Returns:
            Array of predicted prices
        """
        predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(days_ahead):
            # Predict next day
            pred = model.predict(current_sequence.reshape(1, current_sequence.shape[0], current_sequence.shape[1]), verbose=0)
            predictions.append(pred[0, 0])
            
            # Update sequence for next prediction
            # Create new row with predicted close price and interpolated other features
            new_row = current_sequence[-1].copy()
            new_row[0] = pred[0, 0]  # Update close price
            
            # Shift sequence and add new prediction
            current_sequence = np.roll(current_sequence, -1, axis=0)
            current_sequence[-1] = new_row
        
        # Inverse transform predictions (only for close price)
        predictions = np.array(predictions).reshape(-1, 1)
        dummy_array = np.zeros((len(predictions), scaler.n_features_in_))
        dummy_array[:, 0] = predictions.flatten()
        predictions_rescaled = scaler.inverse_transform(dummy_array)[:, 0]
        
        return predictions_rescaled
    
    def calculate_adaptive_weights(self, lstm_predictions: np.ndarray, current_price: float, 
                                 volatility: float, trend_multiplier: float, df: pd.DataFrame, 
                                 symbol: str) -> Dict:
        """
        Calculate adaptive weights for hybrid LSTM + Statistical predictions
        
        Uses multiple factors to dynamically adjust the weight given to each method:
        1. LSTM prediction confidence and consistency
        2. Market volatility conditions  
        3. Trend agreement between methods
        4. Data quality and quantity
        5. Time horizon considerations
        
        Args:
            lstm_predictions: Array of LSTM predicted prices
            current_price: Current stock price
            volatility: 20-day volatility measure
            trend_multiplier: Statistical trend indicator
            df: Historical data DataFrame
            symbol: Stock symbol for logging
            
        Returns:
            Dictionary with adaptive weights for each time horizon
        """
        
        # Base weights (default: 70% LSTM, 30% Statistical)
        base_lstm_weight = 0.7
        base_stat_weight = 0.3
        
        # Factor 1: LSTM Prediction Consistency Score (0-1)
        lstm_consistency = self.calculate_lstm_consistency(lstm_predictions)
        
        # Factor 2: Volatility Adjustment (higher volatility = trust LSTM less)
        volatility_factor = self.calculate_volatility_adjustment(volatility)
        
        # Factor 3: Trend Agreement Factor (both methods agree = trust LSTM more)
        trend_agreement = self.calculate_trend_agreement(lstm_predictions, current_price, trend_multiplier)
        
        # Factor 4: Data Quality Score (more data = trust LSTM more)
        data_quality = self.calculate_data_quality_score(df)
        
        # Factor 5: Time Horizon Adjustment (LSTM better for longer horizons)
        time_factors = self.calculate_time_horizon_factors()
        
        # Combine all factors to create adaptive weights
        weights = {}
        
        for horizon in ['1d', '1w', '1m']:
            # Calculate confidence multiplier for this horizon
            confidence_multiplier = (
                0.3 * lstm_consistency +     # 30% based on LSTM consistency
                0.2 * volatility_factor +    # 20% based on volatility
                0.25 * trend_agreement +     # 25% based on trend agreement  
                0.15 * data_quality +        # 15% based on data quality
                0.1 * time_factors[horizon]  # 10% based on time horizon
            )
            
            # Adjust base weights using confidence multiplier
            adjusted_lstm_weight = min(0.9, max(0.1, base_lstm_weight * confidence_multiplier))
            adjusted_stat_weight = 1.0 - adjusted_lstm_weight
            
            weights[f'lstm_{horizon}'] = adjusted_lstm_weight
            weights[f'stat_{horizon}'] = adjusted_stat_weight
        
        # Log the adaptive weighting decision
        print(f"    📊 Adaptive weights for {symbol}: "
              f"Consistency={lstm_consistency:.2f}, Vol={volatility_factor:.2f}, "
              f"Trend={trend_agreement:.2f}, Quality={data_quality:.2f}")
        
        return weights
    
    def calculate_lstm_consistency(self, predictions: np.ndarray) -> float:
        """Calculate how consistent LSTM predictions are (higher = more trustworthy)"""
        if len(predictions) < 7:
            return 0.5
        
        # Calculate coefficient of variation (lower = more consistent)
        mean_pred = np.mean(predictions[:7])
        std_pred = np.std(predictions[:7])
        
        if mean_pred == 0:
            return 0.5
        
        cv = std_pred / mean_pred
        # Convert to consistency score (0-1, higher = more consistent)
        consistency = max(0.1, min(1.0, 1.0 - cv))
        return consistency
    
    def calculate_volatility_adjustment(self, volatility: float) -> float:
        """Adjust trust based on market volatility (higher volatility = trust statistical more)"""
        if pd.isna(volatility) or volatility == 0:
            return 0.5
        
        # Normalize volatility (typical range 0.5-5.0%)
        normalized_vol = min(5.0, max(0.5, volatility)) / 5.0
        
        # Higher volatility = lower LSTM trust (inverted)
        volatility_trust = 1.0 - normalized_vol
        return max(0.2, min(0.8, volatility_trust))
    
    def calculate_trend_agreement(self, predictions: np.ndarray, current_price: float, 
                                trend_multiplier: float) -> float:
        """Calculate agreement between LSTM and statistical trend indicators"""
        if len(predictions) < 7:
            return 0.5
        
        # LSTM trend direction (week ahead)
        lstm_week_change = (predictions[6] - current_price) / current_price
        lstm_bullish = lstm_week_change > 0.01  # >1% = bullish
        lstm_bearish = lstm_week_change < -0.01  # <-1% = bearish
        
        # Statistical trend direction
        stat_bullish = trend_multiplier > 1.0
        stat_bearish = trend_multiplier < 1.0
        
        # Calculate agreement score
        if (lstm_bullish and stat_bullish) or (lstm_bearish and stat_bearish):
            agreement = 1.0  # Perfect agreement
        elif (not lstm_bullish and not lstm_bearish) or (trend_multiplier == 1.0):
            agreement = 0.7  # Neutral/mixed signals
        else:
            agreement = 0.3  # Disagreement
        
        return agreement
    
    def calculate_data_quality_score(self, df: pd.DataFrame) -> float:
        """Assess data quality and quantity for LSTM training"""
        data_points = len(df)
        
        # More data points = higher quality for LSTM
        if data_points >= 500:
            quality = 1.0
        elif data_points >= 200:
            quality = 0.8
        elif data_points >= 100:
            quality = 0.6
        else:
            quality = 0.4
        
        # Check for missing values
        missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
        quality *= (1.0 - missing_ratio)
        
        return max(0.2, min(1.0, quality))
    
    def calculate_time_horizon_factors(self) -> Dict:
        """LSTM typically better for longer time horizons"""
        return {
            '1d': 0.6,   # Statistical methods often better for very short term
            '1w': 0.8,   # LSTM starts to shine at weekly predictions  
            '1m': 1.0    # LSTM generally best for monthly predictions
        }
    
    def calculate_upside_movements(self, time_series: Dict, symbol: str = "") -> Dict:
        """
        Calculate potential upside movements for next day, week, and month
        Enhanced with LSTM neural network predictions when available
        """
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame.from_dict(time_series, orient='index')
        df.columns = ['open', 'high', 'low', 'close', 'volume']
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        # Convert price columns to float
        price_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in price_columns:
            df[col] = pd.to_numeric(df[col])
        
        # Add technical indicators for enhanced analysis
        if self.use_technical_indicators:
            df = self.add_technical_indicators(df)
        
        # Get the most recent data
        latest_date = df.index[-1]
        latest_close = df['close'].iloc[-1]
        
        # Calculate historical volatility and trends (if not already added)
        if 'daily_return' not in df.columns:
            df['daily_return'] = df['close'].pct_change()
        if 'high_low_spread' not in df.columns:
            df['high_low_spread'] = (df['high'] - df['low']) / df['close'] * 100
        
        # Calculate moving averages for trend analysis
        df['ma_5'] = df['close'].rolling(window=5).mean()
        df['ma_20'] = df['close'].rolling(window=20).mean()
        df['ma_50'] = df['close'].rolling(window=50).mean()
        
        # Get recent volatility metrics
        recent_volatility_1d = df['daily_return'].tail(1).std() * 100
        recent_volatility_5d = df['daily_return'].tail(5).std() * 100
        recent_volatility_20d = df['daily_return'].tail(20).std() * 100
        
        # Calculate trend indicators
        current_ma5 = df['ma_5'].iloc[-1] if not pd.isna(df['ma_5'].iloc[-1]) else latest_close
        current_ma20 = df['ma_20'].iloc[-1] if not pd.isna(df['ma_20'].iloc[-1]) else latest_close
        current_ma50 = df['ma_50'].iloc[-1] if not pd.isna(df['ma_50'].iloc[-1]) else latest_close
        
        # Trend multiplier (bullish if price > moving averages)
        trend_multiplier = 1.0
        if latest_close > current_ma5 and latest_close > current_ma20:
            trend_multiplier = 1.2  # Bullish trend
        elif latest_close < current_ma5 and latest_close < current_ma20:
            trend_multiplier = 0.8  # Bearish trend
        
        # LSTM Enhanced Predictions
        lstm_predictions = None
        lstm_upside_1d, lstm_upside_1w, lstm_upside_1m = 0, 0, 0
        prediction_method = "Statistical"
        hybrid_weights = None
        
        if self.use_lstm and len(df) >= self.lstm_lookback + 30:
            try:
                print(f"  🧠 Training LSTM model for {symbol}...")
                
                # Prepare LSTM data
                X_train, y_train, scaler, scaled_data = self.prepare_lstm_data(df)
                
                if len(X_train) > 20:  # Need minimum data for training
                    # Enhanced training with ensemble or single model
                    if self.use_ensemble and len(X_train) > 50:
                        print(f"    🎯 Training ensemble models...")
                        models = self.train_ensemble_models(X_train, y_train)
                        
                        # Make ensemble predictions with confidence intervals
                        last_sequence = scaled_data[-self.lstm_lookback:]
                        ensemble_results = self.predict_with_ensemble(models, last_sequence, scaler, days_ahead=30)
                        lstm_predictions = ensemble_results['mean_predictions']
                        
                        # Store additional ensemble metrics
                        prediction_std = ensemble_results['std_predictions']
                        confidence_intervals = {
                            'lower_95': ensemble_results['confidence_95_lower'],
                            'upper_95': ensemble_results['confidence_95_upper'],
                            'lower_68': ensemble_results['confidence_68_lower'],
                            'upper_68': ensemble_results['confidence_68_upper']
                        }
                    else:
                        # Single model training
                        model = self.train_lstm_model(X_train, y_train)
                        
                        # Make predictions
                        last_sequence = scaled_data[-self.lstm_lookback:]
                        lstm_predictions = self.predict_with_lstm(model, last_sequence, scaler, days_ahead=30)
                        prediction_std = np.std(lstm_predictions[:7])  # Use weekly std as approximation
                        confidence_intervals = None
                    
                    # Calculate LSTM-based upside movements with risk adjustment
                    lstm_upside_1d = max(0, (lstm_predictions[0] - latest_close) / latest_close * 100)
                    lstm_upside_1w = max(0, (np.max(lstm_predictions[:7]) - latest_close) / latest_close * 100)
                    lstm_upside_1m = max(0, (np.max(lstm_predictions[:30]) - latest_close) / latest_close * 100)
                    
                    # Risk-adjusted upside calculations
                    if self.use_risk_adjustment:
                        risk_free_rate = 0.05  # Approximate risk-free rate (5% annually)
                        
                        # Calculate expected returns
                        expected_return_1d = (lstm_predictions[0] - latest_close) / latest_close
                        expected_return_1w = (np.mean(lstm_predictions[:7]) - latest_close) / latest_close  
                        expected_return_1m = (np.mean(lstm_predictions[:30]) - latest_close) / latest_close
                        
                        # Calculate risk-adjusted metrics (Sharpe-like ratio)
                        daily_vol = recent_volatility_20d / 100 / np.sqrt(252) if not pd.isna(recent_volatility_20d) else 0.02
                        weekly_vol = daily_vol * np.sqrt(7)
                        monthly_vol = daily_vol * np.sqrt(30)
                        
                        # Adjust upside based on risk-reward ratio
                        risk_adjustment_1d = max(0.5, min(2.0, (expected_return_1d - risk_free_rate/252) / daily_vol))
                        risk_adjustment_1w = max(0.5, min(2.0, (expected_return_1w - risk_free_rate/52) / weekly_vol))  
                        risk_adjustment_1m = max(0.5, min(2.0, (expected_return_1m - risk_free_rate/12) / monthly_vol))
                        
                        # Apply risk adjustments
                        lstm_upside_1d *= risk_adjustment_1d
                        lstm_upside_1w *= risk_adjustment_1w
                        lstm_upside_1m *= risk_adjustment_1m
                    
                    # Store scaler for this symbol
                    self.scalers[symbol] = scaler
                    
            except Exception as e:
                print(f"    ⚠️ LSTM training failed for {symbol}: {str(e)[:50]}...")
                lstm_upside_1d, lstm_upside_1w, lstm_upside_1m = 0, 0, 0
        
        # Statistical-based upside movements (original method)
        stat_upside_1d = (recent_volatility_5d if not pd.isna(recent_volatility_5d) else 2.0) * trend_multiplier
        stat_upside_1w = (recent_volatility_20d * 2.5 if not pd.isna(recent_volatility_20d) else 5.0) * trend_multiplier
        stat_upside_1m = (recent_volatility_20d * 4.0 if not pd.isna(recent_volatility_20d) else 8.0) * trend_multiplier
        
        # Advanced Hybrid Prediction System
        if self.use_lstm and lstm_predictions is not None:
            # Calculate hybrid weights using multiple factors
            hybrid_weights = self.calculate_adaptive_weights(
                lstm_predictions, latest_close, recent_volatility_20d, 
                trend_multiplier, df, symbol
            )
            
            upside_1d = (hybrid_weights['lstm_1d'] * lstm_upside_1d + 
                        hybrid_weights['stat_1d'] * stat_upside_1d)
            upside_1w = (hybrid_weights['lstm_1w'] * lstm_upside_1w + 
                        hybrid_weights['stat_1w'] * stat_upside_1w)
            upside_1m = (hybrid_weights['lstm_1m'] * lstm_upside_1m + 
                        hybrid_weights['stat_1m'] * stat_upside_1m)
                        
            prediction_method = f"Adaptive-Hybrid (L:{hybrid_weights['lstm_1m']:.1f}/S:{hybrid_weights['stat_1m']:.1f})"
        else:
            # Fall back to statistical method only
            upside_1d = stat_upside_1d
            upside_1w = stat_upside_1w
            upside_1m = stat_upside_1m
        
        # Cap the upside movements to reasonable values
        upside_1d = min(max(upside_1d, 0), 15.0)  # Max 15% daily
        upside_1w = min(max(upside_1w, 0), 35.0)  # Max 35% weekly
        upside_1m = min(max(upside_1m, 0), 60.0)  # Max 60% monthly
        
        # Enhanced trend classification
        lstm_trend = ""
        if lstm_predictions is not None and len(lstm_predictions) >= 7:
            lstm_trend_score = (lstm_predictions[6] - latest_close) / latest_close
            if lstm_trend_score > 0.02:
                lstm_trend = "LSTM-Bullish"
            elif lstm_trend_score < -0.02:
                lstm_trend = "LSTM-Bearish"
            else:
                lstm_trend = "LSTM-Neutral"
        
        base_trend = 'Bullish' if trend_multiplier > 1 else 'Bearish' if trend_multiplier < 1 else 'Neutral'
        combined_trend = f"{base_trend}" + (f" ({lstm_trend})" if lstm_trend else "")
        
        result = {
            'latest_close': latest_close,
            'latest_date': latest_date.strftime('%Y-%m-%d'),
            'upside_1_day': round(upside_1d, 2),
            'upside_1_week': round(upside_1w, 2),
            'upside_1_month': round(upside_1m, 2),
            'current_trend': combined_trend,
            'volatility_20d': round(recent_volatility_20d if not pd.isna(recent_volatility_20d) else 0, 2),
            'prediction_method': prediction_method
        }
        
        # Add LSTM-specific metrics if available
        if lstm_predictions is not None:
            base_confidence = min(100, max(0, (1 - np.std(lstm_predictions[:7]) / np.mean(lstm_predictions[:7])) * 100))
            
            result.update({
                'lstm_1d_target': round(lstm_predictions[0], 2),
                'lstm_1w_max': round(np.max(lstm_predictions[:7]), 2),
                'lstm_1m_max': round(np.max(lstm_predictions[:30]), 2),
                'lstm_confidence': round(base_confidence, 1)
            })
            
            # Add ensemble-specific metrics if available
            if 'ensemble_results' in locals() and self.use_ensemble:
                result.update({
                    'model_type': 'Ensemble',
                    'prediction_std_1w': round(prediction_std[:7].mean() if hasattr(prediction_std, '__len__') else prediction_std, 2),
                    'ensemble_agreement': round(100 - (prediction_std[:7].mean() / np.mean(lstm_predictions[:7])) * 100, 1)
                })
                
                # Add confidence intervals if available
                if self.use_confidence_intervals and confidence_intervals:
                    result.update({
                        'ci_95_lower_1w': round(confidence_intervals['lower_95'][6], 2),
                        'ci_95_upper_1w': round(confidence_intervals['upper_95'][6], 2),
                        'ci_68_lower_1w': round(confidence_intervals['lower_68'][6], 2),
                        'ci_68_upper_1w': round(confidence_intervals['upper_68'][6], 2)
                    })
            else:
                result.update({
                    'model_type': 'Single LSTM',
                    'prediction_std_1w': round(prediction_std if isinstance(prediction_std, (int, float)) else prediction_std.mean() if hasattr(prediction_std, 'mean') else 0, 2),
                    'ensemble_agreement': 'N/A'
                })
            
            # Add risk adjustment indicators
            if self.use_risk_adjustment:
                result.update({
                    'risk_adjusted': 'Yes',
                    'sharpe_indicator_1m': round(risk_adjustment_1m if 'risk_adjustment_1m' in locals() else 1.0, 2)
                })
            else:
                result.update({
                    'risk_adjusted': 'No',
                    'sharpe_indicator_1m': 'N/A'
                })
            
            # Add hybrid weighting information
            if hybrid_weights:
                result.update({
                    'lstm_weight_1d': round(hybrid_weights['lstm_1d'] * 100, 1),
                    'lstm_weight_1w': round(hybrid_weights['lstm_1w'] * 100, 1), 
                    'lstm_weight_1m': round(hybrid_weights['lstm_1m'] * 100, 1),
                    'stat_weight_1d': round(hybrid_weights['stat_1d'] * 100, 1),
                    'stat_weight_1w': round(hybrid_weights['stat_1w'] * 100, 1),
                    'stat_weight_1m': round(hybrid_weights['stat_1m'] * 100, 1)
                })
        
        return result
    
    def analyze_stocks(self, symbols: List[str]) -> List[Dict]:
        """
        Analyze multiple stocks and return results
        """
        results = []
        
        for i, symbol in enumerate(symbols):
            print(f"Processing {symbol} ({i+1}/{len(symbols)})")
            
            # Fetch data
            data = self.fetch_stock_data(symbol)
            if not data:
                results.append({
                    'symbol': symbol,
                    'status': 'Failed to fetch data',
                    'latest_close': None,
                    'latest_date': None,
                    'upside_1_day': None,
                    'upside_1_week': None,
                    'upside_1_month': None,
                    'current_trend': None,
                    'volatility_20d': None
                })
                continue
            
            # Get time series data
            time_series = data.get("Time Series (Daily)", {})
            if not time_series:
                results.append({
                    'symbol': symbol,
                    'status': 'No time series data',
                    'latest_close': None,
                    'latest_date': None,
                    'upside_1_day': None,
                    'upside_1_week': None,
                    'upside_1_month': None,
                    'current_trend': None,
                    'volatility_20d': None
                })
                continue
            
            # Calculate upside movements
            try:
                analysis = self.calculate_upside_movements(time_series, symbol)
                result = {
                    'symbol': symbol,
                    'status': 'Success',
                    **analysis
                }
                results.append(result)
                method_indicator = "🧠" if analysis.get('prediction_method') == 'LSTM+Statistical' else "📊"
                print(f"✓ {symbol}: {method_indicator} 1D={analysis['upside_1_day']}%, 1W={analysis['upside_1_week']}%, 1M={analysis['upside_1_month']}%")
                
            except Exception as e:
                print(f"Error analyzing {symbol}: {e}")
                results.append({
                    'symbol': symbol,
                    'status': f'Analysis error: {str(e)}',
                    'latest_close': None,
                    'latest_date': None,
                    'upside_1_day': None,
                    'upside_1_week': None,
                    'upside_1_month': None,
                    'current_trend': None,
                    'volatility_20d': None
                })
            
            # Rate limiting - AlphaVantage allows 5 calls per minute for free tier
            if i < len(symbols) - 1:  # Don't sleep after the last symbol
                time.sleep(12)  # 12 seconds between calls to stay under rate limit
        
        return results
    
    def save_to_csv(self, results: List[Dict], filename: str = None) -> str:
        """
        Save results to CSV file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"stock_analysis_{timestamp}.csv"
        
        # Define CSV headers for comprehensive analysis
        headers = [
            'Symbol',
            'Status',
            'Latest Close Price',
            'Latest Date',
            'Upside 1 Day (%)',
            'Upside 1 Week (%)',
            'Upside 1 Month (%)',
            'Current Trend',
            'Volatility 20D (%)',
            'Prediction Method',
            
            # Core LSTM metrics
            'LSTM 1D Target',
            'LSTM 1W Max',
            'LSTM 1M Max',
            'LSTM Confidence (%)',
            
            # Enhanced accuracy metrics
            'Model Type',
            'Prediction Std 1W',
            'Ensemble Agreement (%)',
            'Risk Adjusted',
            'Sharpe Indicator 1M',
            
            # Confidence intervals
            'CI 95% Lower 1W',
            'CI 95% Upper 1W', 
            'CI 68% Lower 1W',
            'CI 68% Upper 1W',
            
            # Hybrid weighting
            'LSTM Weight 1D (%)',
            'LSTM Weight 1W (%)',
            'LSTM Weight 1M (%)',
            'Stat Weight 1D (%)',
            'Stat Weight 1W (%)',
            'Stat Weight 1M (%)'
        ]
        
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
            
            for result in results:
                row = [
                    result['symbol'],
                    result['status'],
                    result['latest_close'],
                    result['latest_date'],
                    result['upside_1_day'],
                    result['upside_1_week'],
                    result['upside_1_month'],
                    result['current_trend'],
                    result['volatility_20d'],
                    result.get('prediction_method', 'N/A'),
                    
                    # Core LSTM metrics
                    result.get('lstm_1d_target', 'N/A'),
                    result.get('lstm_1w_max', 'N/A'),
                    result.get('lstm_1m_max', 'N/A'),
                    result.get('lstm_confidence', 'N/A'),
                    
                    # Enhanced accuracy metrics
                    result.get('model_type', 'N/A'),
                    result.get('prediction_std_1w', 'N/A'),
                    result.get('ensemble_agreement', 'N/A'),
                    result.get('risk_adjusted', 'N/A'),
                    result.get('sharpe_indicator_1m', 'N/A'),
                    
                    # Confidence intervals
                    result.get('ci_95_lower_1w', 'N/A'),
                    result.get('ci_95_upper_1w', 'N/A'),
                    result.get('ci_68_lower_1w', 'N/A'),
                    result.get('ci_68_upper_1w', 'N/A'),
                    
                    # Hybrid weighting
                    result.get('lstm_weight_1d', 'N/A'),
                    result.get('lstm_weight_1w', 'N/A'),
                    result.get('lstm_weight_1m', 'N/A'),
                    result.get('stat_weight_1d', 'N/A'),
                    result.get('stat_weight_1w', 'N/A'),
                    result.get('stat_weight_1m', 'N/A')
                ]
                writer.writerow(row)
        
        print(f"\nResults saved to: {filename}")
        return filename


def main():
    """
    Main function to run the stock analysis
    """
    # Example stock symbols (you can modify this list)
    stock_symbols = [
        "KPITTECH.BSE"
    ]
    
    print("🧠 Advanced Stock Analysis Tool with Adaptive LSTM-Statistical Hybrid")
    print("=" * 75)
    print(f"Analyzing {len(stock_symbols)} stocks with intelligent prediction fusion...")
    print("This may take a while due to API rate limits and LSTM training...\n")
    
    # Create analyzer instance with LSTM enabled
    analyzer = StockAnalyzer(use_lstm=True)
    
    # Analyze stocks
    results = analyzer.analyze_stocks(stock_symbols)
    
    # Save to CSV
    csv_filename = analyzer.save_to_csv(results)
    
    # Print summary
    print("\n" + "=" * 50)
    print("ANALYSIS SUMMARY")
    print("=" * 50)
    
    successful_analyses = [r for r in results if r['status'] == 'Success']
    failed_analyses = [r for r in results if r['status'] != 'Success']
    
    print(f"Successfully analyzed: {len(successful_analyses)} stocks")
    print(f"Failed to analyze: {len(failed_analyses)} stocks")
    
    if successful_analyses:
        print("\nTop potential upside movements (1 month):")
        sorted_results = sorted(successful_analyses, key=lambda x: x['upside_1_month'] or 0, reverse=True)
        for result in sorted_results[:5]:
            print(f"  {result['symbol']}: {result['upside_1_month']}% ({result['current_trend']})")
    
    print(f"\nDetailed results saved in: {csv_filename}")


if __name__ == "__main__":
    main() 
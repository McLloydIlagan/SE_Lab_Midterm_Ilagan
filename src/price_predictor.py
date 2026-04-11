import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Optional


class PricePredictor:
    """Handles price prediction using Random Forest"""
    
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for prediction
        
        Args:
            df: DataFrame with indicators
            
        Returns:
            DataFrame with feature columns
        """
        feature_cols = ['open', 'high', 'low', 'close', 'volume', 
                       'rsi', 'macd', 'atr', 'cci']
        
        available_cols = [col for col in feature_cols if col in df.columns]
        features = df[available_cols].copy()
        
        # Handle missing values
        features = features.fillna(method='ffill').fillna(method='bfill')
        
        return features
    
    def predict_price(self, df: pd.DataFrame, days_ahead: int = 1) -> float:
        """
        Predict future price
        
        Args:
            df: DataFrame with historical data
            days_ahead: Number of days to predict (1 or 5)
            
        Returns:
            Predicted price
            
        Raises:
            ValueError: If parameters are invalid
        """
        if df is None or len(df) < 30:
            raise ValueError("Insufficient data for prediction")
        
        if days_ahead not in [1, 5]:
            raise ValueError("Days ahead must be 1 or 5")
        
        try:
            df_copy = df.copy()
            df_copy['return'] = df_copy['close'].pct_change().fillna(0)
            df_copy['target'] = df_copy['close'].shift(-days_ahead)
            
            df_clean = df_copy.dropna()
            
            if len(df_clean) < 20:
                return df_copy['close'].iloc[-1]  # Return current price if insufficient data
            
            features = self.prepare_features(df_clean)
            target = df_clean['target']
            
            # Train model on all data
            self.model.fit(features, target)
            self.is_trained = True
            
            # Predict next value
            latest_features = features.iloc[-1:].values
            prediction = self.model.predict(latest_features)[0]
            
            return float(prediction)
            
        except Exception as e:
            raise Exception(f"Prediction failed: {str(e)}")
    
    def predict_with_confidence(self, df: pd.DataFrame, days_ahead: int = 1) -> Dict:
        """
        Predict price with confidence interval
        
        Args:
            df: DataFrame with historical data
            days_ahead: Number of days to predict
            
        Returns:
            Dictionary with prediction and confidence
        """
        try:
            predictions = []
            
            # Multiple predictions with different random states
            for random_state in range(5):
                self.model = RandomForestRegressor(n_estimators=100, 
                                                   random_state=random_state)
                pred = self.predict_price(df, days_ahead)
                predictions.append(pred)
            
            mean_pred = np.mean(predictions)
            std_pred = np.std(predictions)
            
            return {
                'prediction': mean_pred,
                'confidence_lower': mean_pred - 2 * std_pred,
                'confidence_upper': mean_pred + 2 * std_pred,
                'volatility': std_pred / mean_pred if mean_pred > 0 else 0
            }
            
        except Exception as e:
            return {
                'prediction': df['close'].iloc[-1],
                'confidence_lower': df['close'].iloc[-1] * 0.95,
                'confidence_upper': df['close'].iloc[-1] * 1.05,
                'volatility': 0.05
            }
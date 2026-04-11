"""
Unit tests for PricePredictor module
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.price_predictor import PricePredictor
from src.indicator_calculator import IndicatorCalculator


class TestPricePredictor:
    """Test suite for PricePredictor"""
    
    @pytest.fixture
    def predictor(self):
        """Create PricePredictor instance"""
        return PricePredictor()
    
    @pytest.fixture
    def sample_df_with_indicators(self):
        """Create sample DataFrame with indicators"""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        
        # Create price data with some pattern
        np.random.seed(42)
        prices = [100]
        for i in range(1, 100):
            change = np.random.randn() * 2
            prices.append(prices[-1] + change)
        
        df = pd.DataFrame({
            'open': prices,
            'high': [p + 2 for p in prices],
            'low': [p - 2 for p in prices],
            'close': prices,
            'volume': [1000000 + i * 10000 for i in range(100)]
        }, index=dates)
        
        # Add indicators
        calculator = IndicatorCalculator()
        df = calculator.compute_indicators(df)
        
        return df
    
    # Positive Test Cases
    def test_predictor_initialization(self, predictor):
        """Test successful predictor initialization"""
        assert predictor.is_trained == False
        assert predictor.model is not None
        assert predictor.scaler is not None
    
    def test_prepare_features(self, predictor, sample_df_with_indicators):
        """Test feature preparation"""
        features = predictor.prepare_features(sample_df_with_indicators)
        
        assert features is not None
        assert len(features) == len(sample_df_with_indicators)
        assert 'close' in features.columns
        assert 'rsi' in features.columns
    
    def test_predict_price_1_day(self, predictor, sample_df_with_indicators):
        """Test 1-day price prediction"""
        prediction = predictor.predict_price(sample_df_with_indicators, days_ahead=1)
        
        assert isinstance(prediction, float)
        assert prediction > 0
        assert not np.isnan(prediction)
    
    def test_predict_price_5_day(self, predictor, sample_df_with_indicators):
        """Test 5-day price prediction"""
        prediction = predictor.predict_price(sample_df_with_indicators, days_ahead=5)
        
        assert isinstance(prediction, float)
        assert prediction > 0
    
    def test_predict_with_confidence(self, predictor, sample_df_with_indicators):
        """Test prediction with confidence intervals"""
        result = predictor.predict_with_confidence(sample_df_with_indicators, days_ahead=1)
        
        assert 'prediction' in result
        assert 'confidence_lower' in result
        assert 'confidence_upper' in result
        assert 'volatility' in result
        
        assert result['confidence_lower'] <= result['prediction'] <= result['confidence_upper']
    
    # Negative Test Cases
    def test_predict_price_none_dataframe(self, predictor):
        """Test prediction with None DataFrame"""
        with pytest.raises(ValueError, match="Insufficient data for prediction"):
            predictor.predict_price(None, days_ahead=1)
    
    def test_predict_price_insufficient_data(self, predictor):
        """Test prediction with insufficient data"""
        small_df = pd.DataFrame({
            'close': [100, 101, 102],
            'high': [102, 103, 104],
            'low': [98, 99, 100],
            'volume': [1000, 1000, 1000]
        })
        
        with pytest.raises(ValueError, match="Insufficient data for prediction"):
            predictor.predict_price(small_df, days_ahead=1)
    
    def test_predict_price_invalid_days_ahead(self, predictor, sample_df_with_indicators):
        """Test prediction with invalid days ahead"""
        with pytest.raises(ValueError, match="Days ahead must be 1 or 5"):
            predictor.predict_price(sample_df_with_indicators, days_ahead=10)
        
        with pytest.raises(ValueError, match="Days ahead must be 1 or 5"):
            predictor.predict_price(sample_df_with_indicators, days_ahead=0)
    
    # Edge Test Cases
    def test_predict_price_minimum_data(self, predictor):
        """Test prediction with minimum required data"""
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        prices = [100 + i for i in range(30)]
        
        df = pd.DataFrame({
            'open': prices,
            'high': [p + 1 for p in prices],
            'low': [p - 1 for p in prices],
            'close': prices,
            'volume': [1000000] * 30
        }, index=dates)
        
        df = IndicatorCalculator.compute_indicators(df)
        prediction = predictor.predict_price(df, days_ahead=1)
        
        assert isinstance(prediction, float)
        assert prediction > 0
    
    def test_predict_price_volatile_data(self, predictor):
        """Test prediction with highly volatile data"""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        
        # Create highly volatile price data
        np.random.seed(42)
        prices = [100]
        for i in range(1, 100):
            change = np.random.randn() * 10  # High volatility
            prices.append(prices[-1] + change)
        
        df = pd.DataFrame({
            'open': prices,
            'high': [p + 5 for p in prices],
            'low': [p - 5 for p in prices],
            'close': prices,
            'volume': [1000000] * 100
        }, index=dates)
        
        df = IndicatorCalculator.compute_indicators(df)
        prediction = predictor.predict_price(df, days_ahead=1)
        
        assert isinstance(prediction, float)
        # Prediction should be within reasonable range
        assert min(prices) * 0.5 <= prediction <= max(prices) * 1.5
    
    def test_predict_price_trending_data(self, predictor):
        """Test prediction with strong trending data"""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        
        # Strong uptrend
        prices = [100 + i * 3 for i in range(100)]
        
        df = pd.DataFrame({
            'open': prices,
            'high': [p + 2 for p in prices],
            'low': [p - 2 for p in prices],
            'close': prices,
            'volume': [1000000] * 100
        }, index=dates)
        
        df = IndicatorCalculator.compute_indicators(df)
        prediction = predictor.predict_price(df, days_ahead=5)
        
        assert isinstance(prediction, float)
        # Prediction should be higher than current price for uptrend
        assert prediction > df['close'].iloc[-1]
    
    def test_predict_with_confidence_consistency(self, predictor, sample_df_with_indicators):
        """Test confidence prediction consistency"""
        # Run multiple predictions
        results = []
        for _ in range(3):
            result = predictor.predict_with_confidence(sample_df_with_indicators, days_ahead=1)
            results.append(result)
        
        # Predictions should be similar (not wildly different)
        predictions = [r['prediction'] for r in results]
        assert max(predictions) / min(predictions) < 1.5  # Within 50% range
    
    def test_predict_price_returns_float(self, predictor, sample_df_with_indicators):
        """Test that prediction returns float type"""
        prediction = predictor.predict_price(sample_df_with_indicators, days_ahead=1)
        assert isinstance(prediction, float)
        
        prediction_5d = predictor.predict_price(sample_df_with_indicators, days_ahead=5)
        assert isinstance(prediction_5d, float)
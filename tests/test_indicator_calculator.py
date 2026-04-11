import pytest
import pandas as pd
import numpy as np
from src.indicator_calculator import IndicatorCalculator


class TestIndicatorCalculator:
    """Test suite for IndicatorCalculator"""
    
    @pytest.fixture
    def calculator(self):
        """Create IndicatorCalculator instance"""
        return IndicatorCalculator()
    
    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame for testing"""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        prices = [100 + i + np.random.randn() * 2 for i in range(100)]
        
        df = pd.DataFrame({
            'open': prices,
            'high': [p + 2 for p in prices],
            'low': [p - 2 for p in prices],
            'close': prices,
            'volume': [1000000] * 100
        }, index=dates)
        return df
    
    # Positive Test Cases
    def test_compute_indicators_success(self, calculator, sample_df):
        """Test successful indicator computation"""
        result = calculator.compute_indicators(sample_df)
        assert 'rsi' in result.columns
        assert 'macd' in result.columns
        assert 'atr' in result.columns
        assert len(result) == len(sample_df)
    
    def test_get_rsi_signal_buy(self, calculator):
        """Test BUY signal from RSI"""
        signal = calculator.get_rsi_signal(25)
        assert signal == 'BUY'
    
    def test_get_rsi_signal_sell(self, calculator):
        """Test SELL signal from RSI"""
        signal = calculator.get_rsi_signal(75)
        assert signal == 'SELL'
    
    # Negative Test Cases
    def test_compute_indicators_none_df(self, calculator):
        """Test with None DataFrame"""
        with pytest.raises(ValueError, match="DataFrame cannot be None"):
            calculator.compute_indicators(None)
    
    def test_get_rsi_signal_invalid_value(self, calculator):
        """Test with invalid RSI value"""
        with pytest.raises(ValueError, match="Invalid RSI value"):
            calculator.get_rsi_signal(150)
    
    # Edge Test Cases
    def test_compute_indicators_minimum_data(self, calculator):
        """Test with minimum required data points"""
        dates = pd.date_range(start='2024-01-01', periods=14, freq='D')
        df = pd.DataFrame({
            'open': [100] * 14,
            'high': [102] * 14,
            'low': [98] * 14,
            'close': [100] * 14,
            'volume': [1000000] * 14
        }, index=dates)
        
        result = calculator.compute_indicators(df)
        assert result is not None
    
    def test_get_rsi_signal_boundary_buy(self, calculator):
        """Test RSI at oversold boundary"""
        signal = calculator.get_rsi_signal(30)
        assert signal == 'HOLD'
    
    def test_get_rsi_signal_boundary_sell(self, calculator):
        """Test RSI at overbought boundary"""
        signal = calculator.get_rsi_signal(70)
        assert signal == 'HOLD'
    
    def test_detect_wyckoff_phase_insufficient_data(self, calculator):
        """Test Wyckoff detection with insufficient data"""
        small_df = pd.DataFrame({'close': [100, 101, 102]})
        phase = calculator.detect_wyckoff_phase(small_df)
        assert phase == "Insufficient Data"
"""
Unit tests for BacktestEngine module
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.backtest_engine import BacktestEngine
from src.indicator_calculator import IndicatorCalculator


class TestBacktestEngine:
    """Test suite for BacktestEngine"""
    
    @pytest.fixture
    def engine(self):
        """Create BacktestEngine instance"""
        return BacktestEngine(initial_balance=10000, trade_size=0.1)
    
    @pytest.fixture
    def sample_df_with_indicators(self):
        """Create sample DataFrame with indicators"""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        
        # Create price data with some trend
        prices = [100]
        for i in range(1, 100):
            change = np.random.randn() * 2
            prices.append(prices[-1] + change)
        
        df = pd.DataFrame({
            'open': prices,
            'high': [p + 2 for p in prices],
            'low': [p - 2 for p in prices],
            'close': prices,
            'volume': [1000000] * 100
        }, index=dates)
        
        # Add indicators
        calculator = IndicatorCalculator()
        df = calculator.compute_indicators(df)
        
        return df
    
    # Positive Test Cases
    def test_backtest_initialization(self, engine):
        """Test successful backtest engine initialization"""
        assert engine.initial_balance == 10000
        assert engine.trade_size == 0.1
    
    def test_generate_signals(self, engine, sample_df_with_indicators):
        """Test signal generation"""
        signals = engine.generate_signals(sample_df_with_indicators)
        assert len(signals) == len(sample_df_with_indicators)
        assert all(signal in ['BUY', 'SELL', 'HOLD'] for signal in signals)
    
    def test_run_backtest_success(self, engine, sample_df_with_indicators):
        """Test successful backtest execution"""
        results = engine.run_backtest(sample_df_with_indicators)
        
        assert 'initial_balance' in results
        assert 'final_equity' in results
        assert 'total_return' in results
        assert 'num_trades' in results
        assert 'win_rate' in results
        assert 'max_drawdown' in results
        assert 'equity_curve' in results
        
        assert results['initial_balance'] == 10000
        assert isinstance(results['final_equity'], float)
        assert isinstance(results['total_return'], float)
    
    # Negative Test Cases
    def test_backtest_invalid_initial_balance(self):
        """Test with invalid initial balance"""
        with pytest.raises(ValueError, match="Initial balance must be positive"):
            BacktestEngine(initial_balance=-1000, trade_size=0.1)
        
        with pytest.raises(ValueError, match="Initial balance must be positive"):
            BacktestEngine(initial_balance=0, trade_size=0.1)
    
    def test_backtest_invalid_trade_size(self):
        """Test with invalid trade size"""
        with pytest.raises(ValueError, match="Trade size must be between 0 and 1"):
            BacktestEngine(initial_balance=10000, trade_size=1.5)
        
        with pytest.raises(ValueError, match="Trade size must be between 0 and 1"):
            BacktestEngine(initial_balance=10000, trade_size=-0.1)
    
    def test_run_backtest_none_dataframe(self, engine):
        """Test backtest with None DataFrame"""
        with pytest.raises(ValueError, match="Insufficient data for backtesting"):
            engine.run_backtest(None)
    
    def test_run_backtest_insufficient_data(self, engine):
        """Test backtest with insufficient data"""
        small_df = pd.DataFrame({
            'close': [100, 101, 102],
            'high': [102, 103, 104],
            'low': [98, 99, 100],
            'volume': [1000, 1000, 1000]
        })
        
        with pytest.raises(ValueError, match="Insufficient data for backtesting"):
            engine.run_backtest(small_df)
    
    # Edge Test Cases
    def test_backtest_no_trades(self, engine):
        """Test backtest with no trading signals"""
        # Create flat price data that generates no signals
        dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
        df = pd.DataFrame({
            'open': [100] * 50,
            'high': [102] * 50,
            'low': [98] * 50,
            'close': [100] * 50,
            'volume': [1000000] * 50
        }, index=dates)
        
        df = IndicatorCalculator.compute_indicators(df)
        results = engine.run_backtest(df)
        
        assert results['num_trades'] == 0
        assert results['final_equity'] == results['initial_balance']
        assert results['total_return'] == 0
    
    def test_backtest_all_buy_signals(self, engine):
        """Test backtest with all buy signals"""
        dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
        
        # Create uptrending price data
        prices = [100 + i * 2 for i in range(50)]
        
        df = pd.DataFrame({
            'open': prices,
            'high': [p + 1 for p in prices],
            'low': [p - 1 for p in prices],
            'close': prices,
            'volume': [1000000] * 50
        }, index=dates)
        
        df = IndicatorCalculator.compute_indicators(df)
        
        # Override RSI to always be oversold (BUY signal)
        df['rsi'] = 25
        
        results = engine.run_backtest(df)
        
        assert results['num_trades'] >= 1
        assert results['total_return'] > 0
    
    def test_backtest_all_sell_signals(self, engine):
        """Test backtest with all sell signals"""
        dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
        
        # Create downtrending price data
        prices = [200 - i * 2 for i in range(50)]
        
        df = pd.DataFrame({
            'open': prices,
            'high': [p + 1 for p in prices],
            'low': [p - 1 for p in prices],
            'close': prices,
            'volume': [1000000] * 50
        }, index=dates)
        
        df = IndicatorCalculator.compute_indicators(df)
        
        # Override RSI to always be overbought (SELL signal)
        df['rsi'] = 75
        
        results = engine.run_backtest(df)
        
        # Should have no buys, only holds
        assert results['num_trades'] == 0 or results['total_return'] < 0
    
    def test_backtest_max_drawdown_calculation(self, engine, sample_df_with_indicators):
        """Test max drawdown calculation"""
        results = engine.run_backtest(sample_df_with_indicators)
        
        assert 'max_drawdown' in results
        assert results['max_drawdown'] <= 0  # Drawdown should be non-positive
        assert isinstance(results['max_drawdown'], float)
    
    def test_backtest_win_rate_calculation(self, engine, sample_df_with_indicators):
        """Test win rate calculation"""
        results = engine.run_backtest(sample_df_with_indicators)
        
        assert 'win_rate' in results
        assert 0 <= results['win_rate'] <= 100
        assert isinstance(results['win_rate'], float)
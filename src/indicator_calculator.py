import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD, CCIIndicator
from ta.volatility import AverageTrueRange
from typing import Dict, Tuple, Optional


class IndicatorCalculator:
    """Calculates technical indicators for cryptocurrency analysis"""
    
    @staticmethod
    def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all technical indicators
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            DataFrame with added indicator columns
            
        Raises:
            ValueError: If DataFrame is invalid
        """
        if df is None:
            raise ValueError("DataFrame cannot be None")
        
        if len(df) < 14:
            raise ValueError(f"Insufficient data: need at least 14 rows, got {len(df)}")
        
        try:
            df_copy = df.copy()
            
            # RSI
            df_copy['rsi'] = RSIIndicator(df_copy['close']).rsi()
            
            # MACD
            macd = MACD(df_copy['close'])
            df_copy['macd'] = macd.macd_diff()
            
            # ATR
            df_copy['atr'] = AverageTrueRange(
                df_copy['high'], df_copy['low'], df_copy['close']
            ).average_true_range()
            
            # CCI
            df_copy['cci'] = CCIIndicator(
                df_copy['high'], df_copy['low'], df_copy['close']
            ).cci()
            
            return df_copy
            
        except Exception as e:
            raise Exception(f"Failed to compute indicators: {str(e)}")
    
    @staticmethod
    def detect_fibonacci_levels(df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate Fibonacci retracement levels
        
        Args:
            df: DataFrame with price data
            
        Returns:
            Dictionary of Fibonacci levels
        """
        if df is None or len(df) == 0:
            return {}
        
        max_price = df['high'].max()
        min_price = df['low'].min()
        diff = max_price - min_price
        
        levels = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
        fibs = {}
        
        for level in levels:
            if level == 0.5:
                fibs['50%'] = max_price - level * diff
            elif level == 0.618:
                fibs['61.8%'] = max_price - level * diff
            else:
                fibs[f'{int(level*100)}%'] = max_price - level * diff
        
        return fibs
    
    @staticmethod
    def detect_wyckoff_phase(df: pd.DataFrame) -> str:
        """
        Detect Wyckoff market phase
        
        Args:
            df: DataFrame with price data
            
        Returns:
            Phase description string
        """
        if df is None or len(df) < 14:
            return "Insufficient Data"
        
        change = df['close'].pct_change().rolling(14).mean().iloc[-1]
        
        if change > 0.02:
            return "Distribution Phase (Bearish)"
        elif change < -0.02:
            return "Accumulation Phase (Bullish)"
        else:
            return "Consolidation"
    
    @staticmethod
    def detect_support_resistance(df: pd.DataFrame, window: int = 20) -> Tuple[float, float]:
        """
        Detect support and resistance levels
        
        Args:
            df: DataFrame with price data
            window: Rolling window for calculation
            
        Returns:
            Tuple of (support, resistance)
        """
        if df is None or len(df) < window:
            return (0, 0)
        
        support = df['low'].rolling(window=window).min().iloc[-1]
        resistance = df['high'].rolling(window=window).max().iloc[-1]
        
        return (support, resistance)
    
    @staticmethod
    def get_rsi_signal(rsi_value: float) -> str:
        """
        Generate trading signal based on RSI
        
        Args:
            rsi_value: RSI value (0-100)
            
        Returns:
            'BUY', 'SELL', or 'HOLD'
            
        Raises:
            ValueError: If RSI value is invalid
        """
        if not isinstance(rsi_value, (int, float)):
            raise ValueError("RSI value must be a number")
        
        if rsi_value < 0 or rsi_value > 100:
            raise ValueError(f"Invalid RSI value: {rsi_value}")
        
        if rsi_value < 30:
            return 'BUY'
        elif rsi_value > 70:
            return 'SELL'
        else:
            return 'HOLD'
    
    @staticmethod
    def get_macd_signal(macd_value: float) -> str:
        """
        Generate trading signal based on MACD
        
        Args:
            macd_value: MACD value
            
        Returns:
            'BUY' or 'SELL'
        """
        return 'BUY' if macd_value > 0 else 'SELL'
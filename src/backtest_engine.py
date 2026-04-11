import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from src.indicator_calculator import IndicatorCalculator


class BacktestEngine:
    """Handles backtesting of trading strategies"""
    
    def __init__(self, initial_balance: float = 1000, trade_size: float = 0.1):
        """
        Initialize backtest engine
        
        Args:
            initial_balance: Starting balance in USD
            trade_size: Percentage of balance to trade (0.1 = 10%)
            
        Raises:
            ValueError: If parameters are invalid
        """
        if initial_balance <= 0:
            raise ValueError("Initial balance must be positive")
        
        if trade_size <= 0 or trade_size > 1:
            raise ValueError("Trade size must be between 0 and 1")
        
        self.initial_balance = initial_balance
        self.trade_size = trade_size
        self.calculator = IndicatorCalculator()
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on indicators
        
        Args:
            df: DataFrame with indicators
            
        Returns:
            Series of signals ('BUY', 'SELL', 'HOLD')
        """
        signals = []
        
        for i in range(len(df)):
            if i < 14:
                signals.append('HOLD')
                continue
            
            rsi = df['rsi'].iloc[i]
            macd = df['macd'].iloc[i]
            
            rsi_signal = self.calculator.get_rsi_signal(rsi)
            macd_signal = self.calculator.get_macd_signal(macd)
            
            # Combine signals
            if rsi_signal == 'BUY' and macd_signal == 'BUY':
                signals.append('BUY')
            elif rsi_signal == 'SELL' and macd_signal == 'SELL':
                signals.append('SELL')
            else:
                signals.append('HOLD')
        
        return pd.Series(signals, index=df.index)
    
    def run_backtest(self, df: pd.DataFrame) -> Dict:
        """
        Run backtest on historical data
        
        Args:
            df: DataFrame with price and indicator data
            
        Returns:
            Dictionary with backtest results
            
        Raises:
            ValueError: If DataFrame is invalid
        """
        if df is None or len(df) < 30:
            raise ValueError("Insufficient data for backtesting")
        
        try:
            signals = self.generate_signals(df)
            balance = self.initial_balance
            position = 0
            equity_curve = []
            trades = []
            
            for i in range(30, len(df)):
                current_price = df['close'].iloc[i]
                signal = signals.iloc[i]
                
                # Calculate current equity
                current_equity = balance + position * current_price
                equity_curve.append(current_equity)
                
                # Execute signal
                if signal == 'BUY' and position == 0:
                    position_size = (balance * self.trade_size) / current_price
                    balance -= position_size * current_price
                    position = position_size
                    trades.append(('BUY', current_price, df.index[i]))
                
                elif signal == 'SELL' and position > 0:
                    balance += position * current_price
                    trades.append(('SELL', current_price, df.index[i]))
                    position = 0
            
            # Close any open position
            if position > 0:
                final_price = df['close'].iloc[-1]
                balance += position * final_price
                trades.append(('SELL', final_price, df.index[-1]))
            
            final_equity = balance
            total_return = (final_equity - self.initial_balance) / self.initial_balance * 100
            
            # Calculate metrics
            winning_trades = 0
            for i in range(0, len(trades)-1, 2):
                if i+1 < len(trades):
                    buy_price = trades[i][1]
                    sell_price = trades[i+1][1]
                    if sell_price > buy_price:
                        winning_trades += 1
            
            total_trades = len(trades) // 2
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            # Calculate max drawdown
            equity_series = pd.Series(equity_curve)
            rolling_max = equity_series.cummax()
            drawdown = (equity_series - rolling_max) / rolling_max
            max_drawdown = drawdown.min() * 100
            
            return {
                'initial_balance': self.initial_balance,
                'final_equity': final_equity,
                'total_return': total_return,
                'num_trades': total_trades,
                'win_rate': win_rate,
                'max_drawdown': max_drawdown,
                'equity_curve': equity_curve,
                'trades': trades
            }
            
        except Exception as e:
            raise Exception(f"Backtest failed: {str(e)}")
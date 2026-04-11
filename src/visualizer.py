"""
Module for visualizing cryptocurrency analysis with charts
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import matplotlib.patches as mpatches


class Visualizer:
    """Handles all visualization and charting operations"""
    
    def __init__(self, style: str = 'dark_background'):
        """
        Initialize visualizer with styling
        
        Args:
            style: Matplotlib style (default: 'dark_background')
        """
        plt.style.use(style)
        self.fig = None
        self.ax = None
    
    def plot_candlestick_chart(self, df: pd.DataFrame, symbol: str, 
                               fib_levels: Optional[Dict] = None) -> None:
        """
        Plot candlestick chart with Fibonacci levels
        
        Args:
            df: DataFrame with OHLC data
            symbol: Cryptocurrency symbol
            fib_levels: Optional Fibonacci levels dictionary
        """
        fig, ax = plt.subplots(figsize=(16, 10), facecolor='black')
        ax.set_facecolor('#1a1a1a')
        
        # Prepare candlestick data
        for idx in range(len(df)):
            date = df.index[idx]
            open_price = df['open'].iloc[idx]
            high_price = df['high'].iloc[idx]
            low_price = df['low'].iloc[idx]
            close_price = df['close'].iloc[idx]
            
            # Determine candle color
            color = '#00ff00' if close_price >= open_price else '#ff0000'
            
            # Plot wick (high-low line)
            ax.plot([mdates.date2num(date), mdates.date2num(date)], 
                   [low_price, high_price], color=color, linewidth=1.5)
            
            # Plot candle body
            body_width = 0.6
            ax.add_patch(plt.Rectangle(
                (mdates.date2num(date) - body_width/2, min(open_price, close_price)),
                body_width, abs(close_price - open_price),
                facecolor=color, alpha=0.8, edgecolor=color
            ))
        
        # Add Fibonacci levels
        if fib_levels:
            for level, price in fib_levels.items():
                if level in ['50%', '61.8%']:
                    color = 'yellow'
                    linewidth = 1.5
                else:
                    color = 'skyblue'
                    linewidth = 1
                ax.axhline(price, linestyle='--', color=color, 
                          linewidth=linewidth, alpha=0.6, label=f'Fib {level}')
        
        # Format chart
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.xticks(rotation=45)
        
        ax.set_title(f'{symbol.capitalize()} - Candlestick Chart', 
                    fontsize=16, fontweight='bold', color='white')
        ax.set_xlabel('Date', fontsize=12, color='white')
        ax.set_ylabel('Price (USD)', fontsize=12, color='white')
        ax.tick_params(colors='white')
        ax.grid(True, alpha=0.3)
        
        # Add legend
        legend_elements = [
            mpatches.Patch(color='#00ff00', label='Bullish Candle'),
            mpatches.Patch(color='#ff0000', label='Bearish Candle'),
            mpatches.Patch(color='yellow', label='Golden Zone (50-61.8%)')
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
        
        plt.tight_layout()
        self.fig = fig
        plt.show()
    
    def plot_technical_indicators(self, df: pd.DataFrame, symbol: str) -> None:
        """
        Plot technical indicators (RSI, MACD, ATR)
        
        Args:
            df: DataFrame with indicator data
            symbol: Cryptocurrency symbol
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
        
        # Price chart
        ax1.plot(df.index, df['close'], color='cyan', linewidth=2, label='Close Price')
        ax1.set_title(f'{symbol.capitalize()} - Price', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Price (USD)', fontsize=10)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # RSI
        if 'rsi' in df.columns:
            ax2.plot(df.index, df['rsi'], color='purple', linewidth=2, label='RSI')
            ax2.axhline(70, color='red', linestyle='--', alpha=0.7, label='Overbought')
            ax2.axhline(30, color='green', linestyle='--', alpha=0.7, label='Oversold')
            ax2.fill_between(df.index, 70, 100, alpha=0.2, color='red')
            ax2.fill_between(df.index, 0, 30, alpha=0.2, color='green')
            ax2.set_title('RSI (14)', fontsize=12, fontweight='bold')
            ax2.set_ylim(0, 100)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # MACD
        if 'macd' in df.columns:
            ax3.plot(df.index, df['macd'], color='blue', linewidth=2, label='MACD')
            ax3.axhline(0, color='black', linestyle='-', alpha=0.5)
            ax3.set_title('MACD', fontsize=12, fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # ATR
        if 'atr' in df.columns:
            ax4.plot(df.index, df['atr'], color='orange', linewidth=2, label='ATR')
            ax4.set_title('Average True Range', fontsize=12, fontweight='bold')
            ax4.set_xlabel('Date', fontsize=10)
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.fig = fig
        plt.show()
    
    def plot_backtest_results(self, backtest_results: Dict, symbol: str) -> None:
        """
        Plot backtest results including equity curve and drawdown
        
        Args:
            backtest_results: Dictionary from backtest engine
            symbol: Cryptocurrency symbol
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Equity curve
        equity_curve = backtest_results.get('equity_curve', [])
        if equity_curve:
            ax1.plot(equity_curve, color='blue', linewidth=2, label='Equity')
            ax1.axhline(y=backtest_results['initial_balance'], 
                       color='gray', linestyle='--', alpha=0.7, label='Initial Balance')
            ax1.set_title(f'{symbol.capitalize()} - Backtest Results', 
                         fontsize=14, fontweight='bold')
            ax1.set_ylabel('Equity ($)', fontsize=11)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Drawdown
        if equity_curve:
            equity_series = pd.Series(equity_curve)
            rolling_max = equity_series.cummax()
            drawdown = (equity_series - rolling_max) / rolling_max * 100
            ax2.fill_between(range(len(drawdown)), drawdown, 0, 
                           color='red', alpha=0.3, label='Drawdown')
            ax2.set_ylabel('Drawdown (%)', fontsize=11)
            ax2.set_xlabel('Trading Days', fontsize=11)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Add performance metrics as text
        metrics_text = f"""
        Performance Metrics:
        Total Return: {backtest_results.get('total_return', 0):.2f}%
        Win Rate: {backtest_results.get('win_rate', 0):.2f}%
        Max Drawdown: {backtest_results.get('max_drawdown', 0):.2f}%
        Number of Trades: {backtest_results.get('num_trades', 0)}
        """
        ax1.text(0.02, 0.98, metrics_text, transform=ax1.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        self.fig = fig
        plt.show()
    
    def plot_prediction_with_confidence(self, df: pd.DataFrame, 
                                        predictions: Dict, symbol: str) -> None:
        """
        Plot price predictions with confidence intervals
        
        Args:
            df: DataFrame with historical data
            predictions: Dictionary with prediction and confidence intervals
            symbol: Cryptocurrency symbol
        """
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Plot historical data
        ax.plot(df.index[-60:], df['close'].iloc[-60:], 
               color='blue', linewidth=2, label='Historical Price')
        
        # Create future dates for prediction
        last_date = df.index[-1]
        future_dates = pd.date_range(start=last_date, periods=6, freq='D')[1:]
        
        # Plot predictions
        pred_value = predictions.get('prediction', df['close'].iloc[-1])
        confidence_lower = predictions.get('confidence_lower', pred_value * 0.95)
        confidence_upper = predictions.get('confidence_upper', pred_value * 1.05)
        
        # Plot prediction point
        ax.scatter([future_dates[0]], [pred_value], 
                  color='red', s=200, zorder=5, label='Prediction', marker='D')
        
        # Plot confidence interval
        ax.fill_between([future_dates[0], future_dates[0]], 
                       [confidence_lower], [confidence_upper],
                       color='red', alpha=0.3, label='95% Confidence Interval')
        
        # Add prediction line
        ax.plot([last_date, future_dates[0]], 
               [df['close'].iloc[-1], pred_value], 
               'r--', alpha=0.7, linewidth=2)
        
        # Add annotation
        ax.annotate(f'Prediction: ${pred_value:,.2f}', 
                   xy=(future_dates[0], pred_value),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        ax.set_title(f'{symbol.capitalize()} - Price Prediction (1-Day)', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=11)
        ax.set_ylabel('Price (USD)', fontsize=11)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.fig = fig
        plt.show()
    
    def plot_comparison(self, data_dict: Dict[str, pd.Series], title: str) -> None:
        """
        Compare multiple cryptocurrencies
        
        Args:
            data_dict: Dictionary mapping coin names to price series
            title: Chart title
        """
        fig, ax = plt.subplots(figsize=(14, 8))
        
        colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown', 'cyan', 'magenta']
        
        for i, (name, prices) in enumerate(data_dict.items()):
            # Normalize to percentage change
            normalized = (prices / prices.iloc[0] - 1) * 100
            color = colors[i % len(colors)]
            ax.plot(normalized.index, normalized, color=color, 
                   linewidth=2, label=name.capitalize())
        
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Return (%)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        plt.tight_layout()
        self.fig = fig
        plt.show()
    
    def save_plot(self, filename: str, dpi: int = 150) -> None:
        """
        Save current plot to file
        
        Args:
            filename: Output filename
            dpi: Resolution in dots per inch
        """
        if self.fig:
            self.fig.savefig(filename, dpi=dpi, bbox_inches='tight')
            print(f"✓ Plot saved to {filename}")
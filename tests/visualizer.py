"""
Advanced plotting module for cryptocurrency analysis
Creates professional charts with technical indicators
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from typing import Optional, List, Dict


class CryptoPlotter:
    """Handles visualization of cryptocurrency analysis with TA indicators"""
    
    def __init__(self, style: str = 'dark_background'):
        """
        Initialize plotter with styling
        
        Args:
            style: Matplotlib style (default: 'dark_background')
        """
        plt.style.use(style)
        self.fig = None
        
    def plot_full_analysis(self, df: pd.DataFrame, coin_name: str, 
                          predictions: Dict, save_path: Optional[str] = None):
        """
        Create comprehensive 4-panel analysis chart
        
        Args:
            df: DataFrame with OHLC and indicators
            coin_name: Name of the cryptocurrency
            predictions: Dictionary with prediction scores
            save_path: Optional path to save the figure
        """
        if df is None or len(df) < 14:
            print("Insufficient data for plotting")
            return
            
        # Create figure with 4 subplots
        self.fig = plt.figure(figsize=(16, 12))
        
        # 1. Price chart with moving averages and Bollinger Bands
        ax1 = plt.subplot(4, 1, 1)
        self._plot_price_with_indicators(ax1, df, coin_name)
        
        # 2. RSI
        ax2 = plt.subplot(4, 1, 2)
        self._plot_rsi(ax2, df)
        
        # 3. MACD
        ax3 = plt.subplot(4, 1, 3)
        self._plot_macd(ax3, df)
        
        # 4. Predictions panel
        ax4 = plt.subplot(4, 1, 4)
        self._plot_predictions(ax4, predictions, df)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Chart saved to {save_path}")
            
    def _plot_price_with_indicators(self, ax, df: pd.DataFrame, coin_name: str):
        """Plot price chart with technical indicators"""
        
        # Plot candlestick-style price line
        ax.plot(df.index, df['close'], label='Close Price', color='cyan', linewidth=2, alpha=0.8)
        
        # Add moving averages
        if 'ema_20' in df and not df['ema_20'].isna().all():
            ax.plot(df.index, df['ema_20'], label='EMA 20', color='orange', linewidth=1.5, alpha=0.7)
            
        if 'sma_50' in df and not df['sma_50'].isna().all():
            ax.plot(df.index, df['sma_50'], label='SMA 50', color='yellow', linewidth=1.5, alpha=0.7)
        
        # Add Bollinger Bands
        if 'bb_upper' in df and not df['bb_upper'].isna().all():
            ax.fill_between(df.index, df['bb_upper'], df['bb_lower'], 
                           alpha=0.2, color='gray', label='Bollinger Bands')
            ax.plot(df.index, df['bb_upper'], color='gray', linestyle='--', alpha=0.5)
            ax.plot(df.index, df['bb_lower'], color='gray', linestyle='--', alpha=0.5)
        
        # Add buy/sell signals based on RSI
        buy_signals = df[df['rsi'] < 30] if 'rsi' in df else pd.DataFrame()
        sell_signals = df[df['rsi'] > 70] if 'rsi' in df else pd.DataFrame()
        
        if not buy_signals.empty:
            ax.scatter(buy_signals.index, buy_signals['close'], 
                      color='lime', marker='^', s=150, 
                      label='Buy Signal (RSI < 30)', zorder=5, edgecolors='white')
        
        if not sell_signals.empty:
            ax.scatter(sell_signals.index, sell_signals['close'], 
                      color='red', marker='v', s=150, 
                      label='Sell Signal (RSI > 70)', zorder=5, edgecolors='white')
        
        # Customize
        ax.set_ylabel('Price (USD)', fontsize=11, fontweight='bold')
        ax.set_title(f'{coin_name} - Technical Analysis', fontsize=14, fontweight='bold', pad=15)
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
    def _plot_rsi(self, ax, df: pd.DataFrame):
        """Plot RSI indicator"""
        if 'rsi' not in df or df['rsi'].isna().all():
            ax.text(0.5, 0.5, 'RSI data not available', 
                   ha='center', va='center', transform=ax.transAxes)
            return
            
        ax.plot(df.index, df['rsi'], label='RSI (14)', color='purple', linewidth=2)
        
        # Add overbought/oversold zones
        ax.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Overbought (70)')
        ax.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Oversold (30)')
        ax.axhline(y=50, color='gray', linestyle=':', alpha=0.5)
        
        # Fill zones
        ax.fill_between(df.index, 70, 100, alpha=0.2, color='red')
        ax.fill_between(df.index, 0, 30, alpha=0.2, color='green')
        
        ax.set_ylabel('RSI Value', fontsize=11, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        
    def _plot_macd(self, ax, df: pd.DataFrame):
        """Plot MACD indicator"""
        if 'macd' not in df or df['macd'].isna().all():
            ax.text(0.5, 0.5, 'MACD data not available', 
                   ha='center', va='center', transform=ax.transAxes)
            return
            
        # Plot MACD line and signal line
        ax.plot(df.index, df['macd'], label='MACD', color='blue', linewidth=1.5)
        ax.plot(df.index, df['macd_signal'], label='Signal', color='red', linewidth=1.5)
        
        # Plot histogram
        colors = ['green' if val >= 0 else 'red' for val in df['macd_diff']]
        ax.bar(df.index, df['macd_diff'], color=colors, alpha=0.5, width=0.8, label='Histogram')
        
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.set_ylabel('MACD', fontsize=11, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        
    def _plot_predictions(self, ax, predictions: Dict, df: pd.DataFrame):
        """Plot prediction scores"""
        if not predictions:
            ax.text(0.5, 0.5, 'Prediction data not available', 
                   ha='center', va='center', transform=ax.transAxes)
            return
            
        # Create bar chart for predictions
        categories = ['1-Day Buy', '1-Day Sell', '5-Day Buy', '5-Day Sell']
        scores = [
            predictions['1_day']['buy'],
            predictions['1_day']['sell'],
            predictions['5_day']['buy'],
            predictions['5_day']['sell']
        ]
        colors = ['green', 'red', 'green', 'red']
        
        bars = ax.bar(categories, scores, color=colors, alpha=0.7, edgecolor='black')
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            if score > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{score}%', ha='center', va='bottom', fontweight='bold')
        
        # Add current price info
        current_price = df['close'].iloc[-1] if len(df) > 0 else 0
        price_change = df['price_change_1d'].iloc[-1] if 'price_change_1d' in df and len(df) > 0 else 0
        
        info_text = f"Current Price: ${current_price:,.2f}\n24h Change: {price_change*100:.2f}%"
        ax.text(0.98, 0.95, info_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_ylabel('Confidence Score (%)', fontsize=11, fontweight='bold')
        ax.set_title('Price Predictions', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Rotate x-axis labels
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
    def plot_advanced_candlestick(self, df: pd.DataFrame, coin_name: str):
        """
        Create advanced candlestick chart with volume
        
        Args:
            df: DataFrame with OHLC data
            coin_name: Name of the cryptocurrency
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), 
                                        gridspec_kw={'height_ratios': [3, 1]},
                                        sharex=True)
        
        # Plot candlesticks
        for i in range(len(df)):
            date = df.index[i]
            open_price = df['open'].iloc[i]
            close_price = df['close'].iloc[i]
            high_price = df['high'].iloc[i]
            low_price = df['low'].iloc[i]
            
            color = 'green' if close_price >= open_price else 'red'
            
            # Draw candle body
            ax1.add_patch(plt.Rectangle(
                (mdates.date2num(date) - 0.3, min(open_price, close_price)),
                0.6, abs(close_price - open_price),
                facecolor=color, alpha=0.8, edgecolor=color
            ))
            
            # Draw wick (high-low line)
            ax1.plot([mdates.date2num(date), mdates.date2num(date)], 
                    [low_price, high_price], color=color, linewidth=1)
        
        # Add moving averages
        if 'ema_20' in df:
            ax1.plot(df.index, df['ema_20'], label='EMA 20', color='orange', linewidth=1.5)
        if 'sma_50' in df:
            ax1.plot(df.index, df['sma_50'], label='SMA 50', color='yellow', linewidth=1.5)
        
        # Volume subplot
        for i in range(len(df)):
            color = 'green' if df['close'].iloc[i] >= df['open'].iloc[i] else 'red'
            ax2.bar(df.index[i], df['close'].iloc[i] * 1000,  # Scale volume for visibility
                   color=color, alpha=0.5, width=0.8)
        
        ax1.set_ylabel('Price (USD)', fontsize=11, fontweight='bold')
        ax1.set_title(f'{coin_name} - Advanced Candlestick Chart', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper left', fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        ax2.set_ylabel('Volume', fontsize=11, fontweight='bold')
        ax2.set_xlabel('Date', fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Format x-axis
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        self.fig = fig
        
    def show(self):
        """Display the plot"""
        if self.fig:
            plt.show()
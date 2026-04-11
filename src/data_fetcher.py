
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.volatility import BollingerBands
import warnings
warnings.filterwarnings('ignore')


class CryptoAnalyzer:
    """Main class for cryptocurrency analysis with technical indicators"""
    
    def __init__(self):
        self.base_url = "https://api.coingecko.com/api/v3"
        self.headers = {
            "Accept": "application/json",
            "User-Agent": "CryptoAnalysisTool/2.0"
        }
        
    def get_top_coins(self, limit=20):
        """
        Get top cryptocurrencies by market cap
        
        Args:
            limit: Number of top coins to fetch (default: 20)
            
        Returns:
            List of top coin data
            
        Raises:
            Exception: If API request fails or returns empty data
        """
        if not isinstance(limit, int) or limit <= 0:
            raise ValueError("Limit must be a positive integer")
            
        url = f"{self.base_url}/coins/markets"
        params = {
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": limit,
            "page": 1,
            "sparkline": False
        }
        
        try:
            response = requests.get(url, params=params, headers=self.headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                raise Exception("Received empty data from API")
                
            return data
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to fetch top coins: {str(e)}")
    
    def get_coin_data(self, coin_id, days=30):
        """
        Get historical OHLC market data for a specific coin
        
        Args:
            coin_id: CoinGecko coin ID (e.g., 'bitcoin')
            days: Number of days of historical data (default: 30)
            
        Returns:
            DataFrame with OHLC data
            
        Raises:
            ValueError: If coin_id is invalid or days out of range
            Exception: If API request fails
        """
        if not coin_id or not isinstance(coin_id, str):
            raise ValueError("Coin ID must be a non-empty string")
            
        if days <= 0 or days > 365:
            raise ValueError("Days must be between 1 and 365")
            
        url = f"{self.base_url}/coins/{coin_id}/ohlc"
        params = {
            "vs_currency": "usd",
            "days": days
        }
        
        try:
            response = requests.get(url, params=params, headers=self.headers, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            if not data or len(data) == 0:
                raise Exception(f"No historical data found for coin {coin_id}")
                
            # Process the data into a DataFrame
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
            
            if len(df) == 0:
                raise Exception("Empty DataFrame created from API response")
                
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Verify we have sufficient data
            if len(df) < 14:  # Minimum for RSI calculation
                raise Exception(f"Insufficient data: only {len(df)} points (minimum 14 required)")
                
            return df
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to fetch coin data: {str(e)}")
    
    def calculate_technical_indicators(self, df):
        """
        Calculate all technical indicators with data validation
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            DataFrame with added indicator columns
            
        Raises:
            ValueError: If df is invalid or insufficient data
        """
        if df is None:
            raise ValueError("DataFrame cannot be None")
            
        if len(df) < 14:
            raise ValueError(f"Insufficient data: {len(df)} points (minimum 14 required)")
        
        try:
            # Create a copy to avoid modifying original
            df_indicators = df.copy()
            
            # RSI (Relative Strength Index)
            df_indicators['rsi'] = RSIIndicator(df_indicators['close'], window=14).rsi()
            
            # MACD (Moving Average Convergence Divergence)
            if len(df_indicators) >= 26:
                macd = MACD(df_indicators['close'])
                df_indicators['macd'] = macd.macd()
                df_indicators['macd_signal'] = macd.macd_signal()
                df_indicators['macd_diff'] = df_indicators['macd'] - df_indicators['macd_signal']
            else:
                df_indicators['macd'] = np.nan
                df_indicators['macd_signal'] = np.nan
                df_indicators['macd_diff'] = np.nan
            
            # Moving Averages
            df_indicators['ema_20'] = EMAIndicator(df_indicators['close'], window=20).ema_indicator() if len(df_indicators) >= 20 else np.nan
            df_indicators['sma_50'] = SMAIndicator(df_indicators['close'], window=50).sma_indicator() if len(df_indicators) >= 50 else np.nan
            
            # Bollinger Bands
            if len(df_indicators) >= 20:
                bb = BollingerBands(df_indicators['close'], window=20, window_dev=2)
                df_indicators['bb_upper'] = bb.bollinger_hband()
                df_indicators['bb_middle'] = bb.bollinger_mavg()
                df_indicators['bb_lower'] = bb.bollinger_lband()
            else:
                df_indicators['bb_upper'] = np.nan
                df_indicators['bb_middle'] = np.nan
                df_indicators['bb_lower'] = np.nan
            
            # Price changes for momentum
            df_indicators['price_change_1d'] = df_indicators['close'].pct_change()
            df_indicators['price_change_5d'] = df_indicators['close'].pct_change(5)
            
            return df_indicators
            
        except Exception as e:
            raise Exception(f"Failed to calculate indicators: {str(e)}")
    
    def generate_predictions(self, df):
        """
        Generate predictions for 1-day and 5-day periods
        
        Args:
            df: DataFrame with technical indicators
            
        Returns:
            Dictionary with prediction scores
        """
        predictions = {
            '1_day': {'buy': 0, 'sell': 0, 'hold': 0},
            '5_day': {'buy': 0, 'sell': 0, 'hold': 0}
        }
        
        if df is None or len(df) < 14:
            return predictions
        
        try:
            # Get current technical indicators
            rsi = df['rsi'].iloc[-1] if 'rsi' in df and not pd.isna(df['rsi'].iloc[-1]) else np.nan
            macd_diff = df['macd_diff'].iloc[-1] if 'macd_diff' in df and not pd.isna(df['macd_diff'].iloc[-1]) else np.nan
            price_vs_ema = (df['close'].iloc[-1] / df['ema_20'].iloc[-1]) if 'ema_20' in df and not pd.isna(df['ema_20'].iloc[-1]) else np.nan
            
            # Recent price changes
            price_change_1d = df['price_change_1d'].iloc[-1] if 'price_change_1d' in df and not pd.isna(df['price_change_1d'].iloc[-1]) else 0
            price_change_5d = df['price_change_5d'].iloc[-1] if 'price_change_5d' in df and not pd.isna(df['price_change_5d'].iloc[-1]) else 0
            
            # 1-day prediction scoring
            buy_score_1d = 0
            sell_score_1d = 0
            
            # RSI factors (40% weight)
            if not np.isnan(rsi):
                if rsi < 30:
                    buy_score_1d += 40
                elif rsi > 70:
                    sell_score_1d += 40
                else:
                    # Neutral RSI - split 20 each
                    buy_score_1d += 20
                    sell_score_1d += 20
                    
            # MACD factors (30% weight)
            if not np.isnan(macd_diff):
                if macd_diff > 0:
                    buy_score_1d += 30
                else:
                    sell_score_1d += 30
                    
            # Price momentum (30% weight)
            if price_change_1d > 0.01:  # Up more than 1%
                buy_score_1d += 30
            elif price_change_1d < -0.01:  # Down more than 1%
                sell_score_1d += 30
            else:
                buy_score_1d += 15
                sell_score_1d += 15
                
            # Normalize 1-day scores
            total_1d = buy_score_1d + sell_score_1d
            if total_1d > 0:
                predictions['1_day']['buy'] = int((buy_score_1d / total_1d) * 100)
                predictions['1_day']['sell'] = int((sell_score_1d / total_1d) * 100)
                predictions['1_day']['hold'] = 100 - predictions['1_day']['buy'] - predictions['1_day']['sell']
                
            # 5-day prediction scoring
            buy_score_5d = 0
            sell_score_5d = 0
            
            # RSI factors (30% weight)
            if not np.isnan(rsi):
                if rsi < 35:
                    buy_score_5d += 30
                elif rsi > 65:
                    sell_score_5d += 30
                else:
                    buy_score_5d += 15
                    sell_score_5d += 15
                    
            # Moving average factors (30% weight)
            if not np.isnan(price_vs_ema):
                if price_vs_ema < 1.02:  # Price near or below EMA
                    buy_score_5d += 30
                elif price_vs_ema > 1.05:  # Price significantly above EMA
                    sell_score_5d += 30
                else:
                    buy_score_5d += 15
                    sell_score_5d += 15
                    
            # Price momentum (40% weight)
            if price_change_5d > 0.05:  # Up more than 5% over 5 days
                buy_score_5d += 40
            elif price_change_5d < -0.05:  # Down more than 5%
                sell_score_5d += 40
            else:
                buy_score_5d += 20
                sell_score_5d += 20
                
            # Normalize 5-day scores
            total_5d = buy_score_5d + sell_score_5d
            if total_5d > 0:
                predictions['5_day']['buy'] = int((buy_score_5d / total_5d) * 100)
                predictions['5_day']['sell'] = int((sell_score_5d / total_5d) * 100)
                predictions['5_day']['hold'] = 100 - predictions['5_day']['buy'] - predictions['5_day']['sell']
            
            return predictions
            
        except Exception as e:
            print(f"Warning: Error generating predictions: {e}")
            return predictions
    
    def get_trading_signal(self, df):
        """
        Generate overall trading signal based on multiple indicators
        
        Args:
            df: DataFrame with technical indicators
            
        Returns:
            String: 'BUY', 'SELL', or 'HOLD'
        """
        if df is None or len(df) < 14:
            return 'HOLD'
        
        try:
            rsi = df['rsi'].iloc[-1] if 'rsi' in df and not pd.isna(df['rsi'].iloc[-1]) else 50
            macd_diff = df['macd_diff'].iloc[-1] if 'macd_diff' in df and not pd.isna(df['macd_diff'].iloc[-1]) else 0
            price_vs_ema = (df['close'].iloc[-1] / df['ema_20'].iloc[-1]) if 'ema_20' in df and not pd.isna(df['ema_20'].iloc[-1]) else 1
            
            # Score from -10 (strong sell) to +10 (strong buy)
            score = 0
            
            # RSI scoring
            if rsi < 30:
                score += 4
            elif rsi < 40:
                score += 2
            elif rsi > 70:
                score -= 4
            elif rsi > 60:
                score -= 2
                
            # MACD scoring
            if macd_diff > 0:
                score += 3
            else:
                score -= 3
                
            # Moving average scoring
            if price_vs_ema > 1.02:
                score += 3
            elif price_vs_ema < 0.98:
                score -= 3
                
            # Determine signal
            if score >= 5:
                return 'BUY'
            elif score <= -5:
                return 'SELL'
            else:
                return 'HOLD'
                
        except Exception:
            return 'HOLD'
    
    def get_current_price(self, coin_id, vs_currency='usd'):
        """
        Get current price for a coin
        
        Args:
            coin_id: CoinGecko coin ID
            vs_currency: Currency to compare against
            
        Returns:
            Current price as float
        """
        url = f"{self.base_url}/simple/price"
        params = {
            'ids': coin_id,
            'vs_currencies': vs_currency
        }
        
        try:
            response = requests.get(url, params=params, headers=self.headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if coin_id not in data:
                raise ValueError(f"Coin '{coin_id}' not found")
                
            return data[coin_id].get(vs_currency, 0)
            
        except Exception as e:
            raise Exception(f"Failed to get current price: {str(e)}")
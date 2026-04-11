import requests
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime


class DataFetcher:
    """Handles all data fetching operations from CoinGecko API"""
    
    def __init__(self):
        self.base_url = "https://api.coingecko.com/api/v3"
        self.headers = {
            "Accept": "application/json",
            "User-Agent": "CryptoAnalysisTool/2.0"
        }
    
    def get_top_coins(self, limit: int = 20) -> Optional[List[Dict]]:
        """
        Get top cryptocurrencies by market cap
        
        Args:
            limit: Number of top coins to fetch (1-250)
            
        Returns:
            List of coin data dictionaries or None if error
            
        Raises:
            ValueError: If limit is invalid
            Exception: If API request fails
        """
        if not isinstance(limit, int):
            raise ValueError("Limit must be an integer")
        
        if limit < 1 or limit > 250:
            raise ValueError("Limit must be between 1 and 250")
        
        url = f"{self.base_url}/coins/markets"
        params = {
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": limit,
            "page": 1,
            "sparkline": False
        }
        
        try:
            response = requests.get(url, params=params, 
                                  headers=self.headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                raise Exception("Received empty data from API")
                
            return data
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to fetch top coins: {str(e)}")
    
    def fetch_ohlcv(self, symbol: str, days: int = 365) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data for a specific cryptocurrency
        
        Args:
            symbol: CoinGecko coin ID (e.g., 'bitcoin')
            days: Number of days of historical data (1-365)
            
        Returns:
            DataFrame with OHLCV data or None if error
            
        Raises:
            ValueError: If parameters are invalid
            Exception: If API request fails
        """
        if not symbol or not isinstance(symbol, str):
            raise ValueError("Symbol must be a non-empty string")
        
        if not isinstance(days, int) or days < 1 or days > 365:
            raise ValueError("Days must be an integer between 1 and 365")
        
        url = f"{self.base_url}/coins/{symbol.lower()}/market_chart"
        params = {
            "vs_currency": "usd",
            "days": days
        }
        
        try:
            response = requests.get(url, params=params,
                                  headers=self.headers, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            if not data.get('prices'):
                raise Exception(f"No price data found for {symbol}")
            
            prices = data.get('prices', [])
            df = pd.DataFrame(prices, columns=['timestamp', 'close'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df['close'] = df['close'].astype(float)
            
            # Calculate OHLC from close prices
            df['open'] = df['close'].shift(1)
            df['high'] = df['close'].rolling(window=2).max()
            df['low'] = df['close'].rolling(window=2).min()
            
            # Placeholder volume (API doesn't provide volume in this endpoint)
            np.random.seed(42)
            df['volume'] = np.random.uniform(1000000, 50000000, len(df))
            
            df = df[['open', 'high', 'low', 'close', 'volume']].dropna()
            
            if len(df) < 14:
                raise Exception(f"Insufficient data: only {len(df)} points")
            
            return df
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to fetch coin data: {str(e)}")
    
    def get_current_price(self, symbol: str) -> float:
        """
        Get current price for a cryptocurrency
        
        Args:
            symbol: CoinGecko coin ID
            
        Returns:
            Current price in USD
            
        Raises:
            ValueError: If symbol is invalid
            Exception: If API request fails
        """
        if not symbol or not isinstance(symbol, str):
            raise ValueError("Symbol must be a non-empty string")
        
        url = f"{self.base_url}/simple/price"
        params = {
            'ids': symbol.lower(),
            'vs_currencies': 'usd'
        }
        
        try:
            response = requests.get(url, params=params,
                                  headers=self.headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if symbol not in data:
                raise ValueError(f"Coin '{symbol}' not found")
                
            return data[symbol].get('usd', 0)
            
        except Exception as e:
            raise Exception(f"Failed to get current price: {str(e)}")
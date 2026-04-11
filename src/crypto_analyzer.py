"""
Cryptocurrency Analysis Module using CoinGecko API
Provides market data fetching and technical analysis functionality
"""

import requests
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json
import os


class CryptoAnalyzer:
    """Main class for cryptocurrency analysis"""
    
    def __init__(self):
        self.base_url = "https://api.coingecko.com/api/v3"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'CryptoAnalyzer/1.0'
        })
        
    def get_coin_list(self) -> List[Dict]:
        """
        Get list of all supported coins
        
        Returns:
            List of dictionaries with coin information
            
        Raises:
            requests.RequestException: If API call fails
        """
        try:
            response = self.session.get(
                f"{self.base_url}/coins/list",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise Exception(f"Failed to fetch coin list: {str(e)}")
    
    def get_market_data(self, coin_id: str, vs_currency: str = "usd") -> Dict:
        """
        Fetch current market data for a specific coin
        
        Args:
            coin_id: CoinGecko coin ID (e.g., 'bitcoin')
            vs_currency: Currency to compare against (default: 'usd')
            
        Returns:
            Dictionary with market data
            
        Raises:
            ValueError: If coin_id is empty or invalid
            requests.RequestException: If API call fails
        """
        if not coin_id or not isinstance(coin_id, str):
            raise ValueError("Coin ID must be a non-empty string")
        
        if not vs_currency or not isinstance(vs_currency, str):
            raise ValueError("Currency must be a non-empty string")
        
        try:
            response = self.session.get(
                f"{self.base_url}/simple/price",
                params={
                    'ids': coin_id,
                    'vs_currencies': vs_currency,
                    'include_market_cap': 'true',
                    'include_24hr_change': 'true',
                    'include_last_updated_at': 'true'
                },
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            if coin_id not in data:
                raise ValueError(f"Coin '{coin_id}' not found")
                
            return data[coin_id]
            
        except requests.RequestException as e:
            raise Exception(f"Failed to fetch market data: {str(e)}")
    
    def calculate_rsi(self, prices: List[float], period: int = 14) -> Optional[float]:
        """
        Calculate Relative Strength Index (RSI) for price data
        
        Args:
            prices: List of historical prices
            period: RSI calculation period (default: 14)
            
        Returns:
            RSI value between 0-100, or None if insufficient data
            
        Raises:
            ValueError: If period is invalid or prices list is empty
        """
        if period <= 0:
            raise ValueError("Period must be positive")
        
        if not prices:
            raise ValueError("Prices list cannot be empty")
        
        if len(prices) < period + 1:
            return None
        
        gains = []
        losses = []
        
        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            if change >= 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return round(rsi, 2)
    
    def get_signal(self, rsi: Optional[float]) -> str:
        """
        Generate trading signal based on RSI value
        
        Args:
            rsi: RSI value (0-100)
            
        Returns:
            Trading signal: 'BUY', 'SELL', or 'HOLD'
            
        Raises:
            ValueError: If RSI is invalid
        """
        if rsi is None:
            return 'HOLD'
        
        if not isinstance(rsi, (int, float)):
            raise ValueError("RSI must be a number")
        
        if rsi < 0 or rsi > 100:
            raise ValueError(f"Invalid RSI value: {rsi}. Must be between 0-100")
        
        if rsi < 30:
            return 'BUY'
        elif rsi > 70:
            return 'SELL'
        else:
            return 'HOLD'
    
    def get_historical_prices(self, coin_id: str, days: int = 30, vs_currency: str = "usd") -> List[float]:
        """
        Fetch historical price data for a coin
        
        Args:
            coin_id: CoinGecko coin ID
            days: Number of days of historical data
            vs_currency: Currency to compare against
            
        Returns:
            List of closing prices
            
        Raises:
            ValueError: If parameters are invalid
        """
        if days <= 0 or days > 365:
            raise ValueError("Days must be between 1 and 365")
        
        try:
            response = self.session.get(
                f"{self.base_url}/coins/{coin_id}/market_chart",
                params={
                    'vs_currency': vs_currency,
                    'days': days,
                    'interval': 'daily'
                },
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            if 'prices' not in data:
                raise ValueError(f"No price data available for {coin_id}")
            
            # Extract closing prices (second element of each price pair)
            prices = [price[1] for price in data['prices']]
            return prices
            
        except requests.RequestException as e:
            raise Exception(f"Failed to fetch historical prices: {str(e)}")
    
    def analyze_coin(self, coin_id: str, vs_currency: str = "usd") -> Dict:
        """
        Perform complete analysis of a cryptocurrency
        
        Args:
            coin_id: CoinGecko coin ID
            vs_currency: Currency to compare against
            
        Returns:
            Dictionary with complete analysis results
        """
        # Get current market data
        market_data = self.get_market_data(coin_id, vs_currency)
        
        # Get historical prices and calculate RSI
        prices = self.get_historical_prices(coin_id, days=30, vs_currency=vs_currency)
        rsi = self.calculate_rsi(prices)
        
        # Get signal
        signal = self.get_signal(rsi)
        
        # Prepare result
        result = {
            'coin_id': coin_id,
            'currency': vs_currency,
            'current_price': market_data.get(f'{vs_currency}', 0),
            'market_cap': market_data.get(f'{vs_currency}_market_cap', 0),
            'price_change_24h': market_data.get(f'{vs_currency}_24h_change', 0),
            'rsi': rsi,
            'signal': signal,
            'timestamp': datetime.now().isoformat()
        }
        
        return result
import pytest
import pandas as pd
from unittest.mock import Mock, patch
from src.data_fetcher import DataFetcher


class TestDataFetcher:
    """Test suite for DataFetcher"""
    
    @pytest.fixture
    def fetcher(self):
        """Create DataFetcher instance"""
        return DataFetcher()
    
    # Positive Test Cases
    def test_get_top_coins_success(self, fetcher):
        """Test successful retrieval of top coins"""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = [
                {'id': 'bitcoin', 'name': 'Bitcoin', 'symbol': 'btc',
                 'current_price': 50000, 'market_cap': 1000000000}
            ]
            mock_get.return_value = mock_response
            
            result = fetcher.get_top_coins(limit=1)
            assert len(result) == 1
            assert result[0]['id'] == 'bitcoin'
    
    def test_fetch_ohlcv_success(self, fetcher):
        """Test successful OHLCV data fetch"""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                'prices': [[1700000000000, 50000], [1700086400000, 51000]]
            }
            mock_get.return_value = mock_response
            
            df = fetcher.fetch_ohlcv('bitcoin', days=2)
            assert df is not None
            assert 'close' in df.columns
            assert len(df) > 0
    
    # Negative Test Cases
    def test_get_top_coins_invalid_limit(self, fetcher):
        """Test with invalid limit parameter"""
        with pytest.raises(ValueError, match="Limit must be an integer"):
            fetcher.get_top_coins(limit="invalid")
        
        with pytest.raises(ValueError, match="Limit must be between 1 and 250"):
            fetcher.get_top_coins(limit=0)
    
    def test_fetch_ohlcv_empty_symbol(self, fetcher):
        """Test with empty symbol"""
        with pytest.raises(ValueError, match="Symbol must be a non-empty string"):
            fetcher.fetch_ohlcv("")
    
    def test_fetch_ohlcv_invalid_days(self, fetcher):
        """Test with invalid days parameter"""
        with pytest.raises(ValueError, match="Days must be an integer between 1 and 365"):
            fetcher.fetch_ohlcv('bitcoin', days=400)
    
    # Edge Test Cases
    def test_get_top_coins_max_limit(self, fetcher):
        """Test with maximum limit"""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = [{'id': f'coin{i}'} for i in range(250)]
            mock_get.return_value = mock_response
            
            result = fetcher.get_top_coins(limit=250)
            assert len(result) == 250
    
    def test_fetch_ohlcv_minimum_days(self, fetcher):
        """Test with minimum days"""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                'prices': [[1700000000000, 50000]]
            }
            mock_get.return_value = mock_response
            
            df = fetcher.fetch_ohlcv('bitcoin', days=1)
            assert df is not None
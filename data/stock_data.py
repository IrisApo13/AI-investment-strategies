"""
Stock data retrieval and technical analysis module.
"""
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import logging
from typing import Dict, Optional
from datetime import datetime, timedelta

from config.settings import Config

logger = logging.getLogger(__name__)

class StockDataProvider:
    """Provides stock data with technical indicators."""
    
    def __init__(self):
        """Initialize the stock data provider."""
        self.cache = {}
    
    def get_stock_data(self, ticker: str, period: str = None) -> pd.DataFrame:
        """
        Fetch stock data with technical indicators.
        
        Args:
            ticker: Stock symbol (e.g., 'AAPL')
            period: Time period (e.g., '1y', '2y', '5y', '10y')
            
        Returns:
            DataFrame with OHLCV data and technical indicators
        """
        period = period or Config.DEFAULT_PERIOD
        cache_key = f"{ticker}_{period}"
        
        # Check cache first
        if cache_key in self.cache:
            logger.info(f"Using cached data for {ticker}")
            return self.cache[cache_key].copy()
        
        try:
            logger.info(f"Fetching data for {ticker} ({period})")
            
            # Fetch data from yfinance
            stock = yf.Ticker(ticker)
            data = stock.history(period=period, interval=Config.DATA_INTERVAL)
            
            if data.empty:
                raise ValueError(f"No data found for ticker {ticker}")
            
            logger.info(f"Retrieved {len(data)} days of data for {ticker}")
            
            # Add technical indicators
            data = self._add_technical_indicators(data)
            
            # Cache the data
            self.cache[cache_key] = data.copy()
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {str(e)}")
            raise
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the data."""
        logger.info("Calculating technical indicators...")
        
        # Simple Moving Averages
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        
        # Exponential Moving Averages
        data['EMA_12'] = data['Close'].ewm(span=12).mean()
        data['EMA_26'] = data['Close'].ewm(span=26).mean()
        
        # RSI (Relative Strength Index)
        data['RSI'] = ta.momentum.RSIIndicator(data['Close']).rsi()
        
        # MACD
        macd_indicator = ta.trend.MACD(data['Close'])
        data['MACD'] = macd_indicator.macd()
        data['MACD_signal'] = macd_indicator.macd_signal()
        
        # Bollinger Bands
        bb_indicator = ta.volatility.BollingerBands(data['Close'])
        data['BB_upper'] = bb_indicator.bollinger_hband()
        data['BB_middle'] = bb_indicator.bollinger_mavg()
        data['BB_lower'] = bb_indicator.bollinger_lband()
        
        # Average True Range (ATR)
        data['ATR'] = ta.volatility.AverageTrueRange(
            data['High'], data['Low'], data['Close']
        ).average_true_range()
        
        # Volume indicators
        data['volume_sma'] = data['Volume'].rolling(window=20).mean()
        
        # Price change and momentum
        data['price_change'] = data['Close'].pct_change()
        data['volatility'] = data['price_change'].rolling(window=20).std()
        data['momentum'] = data['Close'] / data['Close'].shift(10) - 1
        
        # Additional indicators
        data['price_position'] = (data['Close'] - data['Close'].rolling(window=20).min()) / \
                                (data['Close'].rolling(window=20).max() - data['Close'].rolling(window=20).min())
        
        # Volume ratio
        data['volume_ratio'] = data['Volume'] / data['volume_sma']
        
        # Drop NaN values
        data = data.dropna()
        
        logger.info(f"Added technical indicators. Final dataset has {len(data)} rows")
        
        return data
    
    def get_latest_data(self, ticker: str, days: int = 30) -> pd.DataFrame:
        """Get recent data for a ticker."""
        try:
            stock = yf.Ticker(ticker)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            data = stock.history(start=start_date, end=end_date, interval=Config.DATA_INTERVAL)
            
            if not data.empty:
                data = self._add_technical_indicators(data)
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching latest data for {ticker}: {str(e)}")
            raise
    
    def validate_ticker(self, ticker: str) -> bool:
        """Validate if a ticker symbol exists and has data."""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Check if we can get basic info
            if 'symbol' in info or 'shortName' in info:
                return True
            
            # Fallback: try to get a small amount of recent data
            test_data = stock.history(period="5d")
            return not test_data.empty
            
        except Exception:
            return False
    
    def get_stock_info(self, ticker: str) -> Dict:
        """Get basic information about a stock."""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            return {
                'symbol': info.get('symbol', ticker),
                'name': info.get('longName', info.get('shortName', 'Unknown')),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
                'currency': info.get('currency', 'USD')
            }
            
        except Exception as e:
            logger.warning(f"Could not retrieve info for {ticker}: {str(e)}")
            return {
                'symbol': ticker,
                'name': 'Unknown',
                'sector': 'Unknown',
                'industry': 'Unknown',
                'market_cap': 0,
                'currency': 'USD'
            }
    
    def clear_cache(self):
        """Clear the data cache."""
        self.cache.clear()
        logger.info("Data cache cleared") 
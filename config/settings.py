"""
Configuration settings for the AI backtesting system.
"""
import os
from typing import List

class Config:
    """Configuration class containing all system settings."""
    
    # OpenAI API Configuration
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    OPENAI_MODEL = "gpt-4-turbo-preview"
    MAX_TOKENS = 2000
    TEMPERATURE = 0.7
    
    # Backtesting Configuration
    INITIAL_CAPITAL = 100000.0  # $100,000
    TRANSACTION_COST = 0.001  # 0.1% transaction cost
    SLIPPAGE = 0.0005  # 0.05% slippage
    
    # Data Configuration
    DEFAULT_PERIOD = "2y"  # Default data period
    DATA_INTERVAL = "1d"  # Daily data
    
    # Strategy Configuration
    MAX_POSITION_SIZE = 0.3  # Maximum 30% of portfolio in single trade
    MIN_POSITION_SIZE = 0.05  # Minimum 5% position size
    
    # Performance Thresholds
    MIN_SHARPE_RATIO = 0.5
    MIN_WIN_RATE = 45.0  # 45%
    MAX_DRAWDOWN_THRESHOLD = -20.0  # -20%
    
    # Technical Indicators
    TECHNICAL_INDICATORS = [
        'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26', 'RSI', 'MACD', 'MACD_signal',
        'BB_upper', 'BB_middle', 'BB_lower', 'ATR', 'volume_sma', 'price_change',
        'volatility', 'momentum'
    ]
    
    # Iteration Configuration
    MAX_ITERATIONS = 10
    IMPROVEMENT_THRESHOLD = 5.0  # 5% improvement threshold
    
    # Logging Configuration
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate configuration settings."""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        if cls.INITIAL_CAPITAL <= 0:
            raise ValueError("INITIAL_CAPITAL must be positive")
        
        if not (0 <= cls.TRANSACTION_COST <= 1):
            raise ValueError("TRANSACTION_COST must be between 0 and 1")
        
        return True 
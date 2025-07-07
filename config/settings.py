"""
Configuration settings for the AI backtesting system.
"""
import os
from typing import List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration class containing all system settings."""
    
    # OpenAI API Configuration
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    
    # OpenAI Model Configuration
    # Supported models:
    # - GPT-4/GPT-3.5 series: "gpt-4-turbo-preview", "gpt-4", "gpt-3.5-turbo" (use max_tokens)
    # - O1 series: "o1-mini", "o1-preview" (use max_completion_tokens)
    # - O4 series: "o4-mini" (use max_completion_tokens)
    OPENAI_MODEL = "gpt-4-turbo"
    
    MAX_TOKENS = 2000  # Will be mapped to max_tokens or max_completion_tokens based on model
    TEMPERATURE = 0.7
    
    # Backtesting Configuration
    INITIAL_CAPITAL = 100000.0  # $100,000
    TRANSACTION_COST = 0  # 0.1% transaction cost
    SLIPPAGE = 0.0005  # 0.05% slippage
    
    # Data Configuration
    DEFAULT_PERIOD = "2y"  # Default data period
    DATA_INTERVAL = "1d"  # Daily data
    
    # Strategy Configuration
    MAX_POSITION_SIZE = 0.1  # Maximum 30% of portfolio in single trade
    MIN_POSITION_SIZE = 0.05  # Minimum 5% position size
    
    # Performance Thresholds
    MIN_SHARPE_RATIO = 0.5
    MIN_WIN_RATE = 45.0  # 45%
    MAX_DRAWDOWN_THRESHOLD = -20.0  # -20%


    # feedback strategies configuration
    # no_feedback: each run generates a new strategy 
    # basic_feedback: the feedback is the performance of the last strategy 
    # advanced_feedback: includes analysis of best and worst trades for enhanced feedback
    
    FEEDBACK_STATEGY = "advanced_feedback"

    # Market data configuration
    # summary: provide summarized market data statistics to LLM
    # raw_data: provide raw daily market data table to LLM
    
    MARKET_DATA_MODE = "raw_data"


    # Technical Indicators
    TECHNICAL_INDICATORS = [
        'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26', 'RSI', 'MACD', 'MACD_signal',
        'BB_upper', 'BB_middle', 'BB_lower', 'ATR', 'volume_sma', 'price_change',
        'volatility', 'momentum'
    ]
    
    # Iteration Configuration
    INTERACTIVE = False

    MAX_ITERATIONS = 1
    IMPROVEMENT_THRESHOLD = 5.0  # 5% improvement threshold
    
    # Logging Configuration
    LOG_LEVEL = "DEBUG"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # LLM Debug Configuration
    SHOW_LLM_PROMPTS = True  # Show prompts sent to LLM
    SHOW_LLM_RESPONSES = True  # Show responses from LLM
    
    # Visualization Configuration
    SHOW_STRATEGY_PLOTS = False  # Show strategy visualization charts during backtesting
    
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
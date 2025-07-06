"""
OpenAI LLM client for strategy generation.
"""
from openai import OpenAI
import logging
from typing import Dict, Optional
import time
import json

from config.settings import Config

logger = logging.getLogger(__name__)

class LLMClient:
    """Client for interacting with OpenAI's LLM."""
    
    def __init__(self):
        """Initialize the LLM client."""
        if not Config.OPENAI_API_KEY:
            raise ValueError("OpenAI API key is required")
        
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
        self.model = Config.OPENAI_MODEL
        self.max_tokens = Config.MAX_TOKENS
        self.temperature = Config.TEMPERATURE
        
        logger.info(f"Initialized LLM client with model: {self.model}")
    
    def _get_model_specific_params(self) -> Dict:
        """
        Get model-specific parameters for OpenAI API calls.
        Some models like o1-mini, o4-mini have different parameter requirements:
        - Use 'max_completion_tokens' instead of 'max_tokens'
        - Only support default temperature (1.0)
        - Don't support some parameters like top_p, frequency_penalty, presence_penalty
        """
        base_params = {
            "model": self.model,
        }
        
        # Models that use max_completion_tokens instead of max_tokens
        new_models = ['o1-mini', 'o1-preview', 'o4-mini', 'gpt-o1-mini', 'gpt-o4-mini']
        
        # Check if the current model is one of the new models
        if any(model_name in self.model.lower() for model_name in new_models):
            base_params["max_completion_tokens"] = self.max_tokens
            # New models only support default temperature (1.0)
            # Don't include temperature parameter to use default
            logger.info(f"Using max_completion_tokens and default temperature for model: {self.model}")
        else:
            base_params["max_tokens"] = self.max_tokens
            base_params["temperature"] = self.temperature
            base_params.update({
                "top_p": 1,
                "frequency_penalty": 0,
                "presence_penalty": 0
            })
            logger.info(f"Using max_tokens and custom temperature for model: {self.model}")
        
        return base_params
    
    def generate_strategy(self, prompt: str, retry_count: int = 3, operation_type: str = "STRATEGY GENERATION") -> str:
        """
        Generate investment strategy using the LLM.
        
        Args:
            prompt: The prompt to send to the LLM
            retry_count: Number of retries on failure
            
        Returns:
            Generated strategy text
        """
        for attempt in range(retry_count):
            try:
                logger.info(f"Generating strategy (attempt {attempt + 1}/{retry_count})")
                
                # Display prompt if enabled
                if Config.SHOW_LLM_PROMPTS:
                    self._display_prompt(operation_type, prompt)
                
                # Get model-specific parameters
                api_params = self._get_model_specific_params()
                api_params["messages"] = [
                    {
                        "role": "system",
                        "content": "You are an expert quantitative analyst and investment strategist. "
                                 "Generate precise, executable investment strategies based on technical indicators."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ]
                
                response = self.client.chat.completions.create(**api_params)
                
                content = response.choices[0].message.content.strip()
                
                if not content:
                    raise ValueError("Empty response from LLM")
                
                # Display response if enabled
                if Config.SHOW_LLM_RESPONSES:
                    self._display_response(operation_type, content)
                
                logger.info("Successfully generated strategy")
                return content
                
            except Exception as e:
                if "rate_limit" in str(e).lower():
                    logger.warning(f"Rate limit hit, waiting before retry: {str(e)}")
                    time.sleep(60)  # Wait 1 minute
                elif "api" in str(e).lower():
                    logger.error(f"OpenAI API error: {str(e)}")
                    if attempt == retry_count - 1:
                        raise
                    time.sleep(10)
                else:
                    logger.error(f"Unexpected error generating strategy: {str(e)}")
                    if attempt == retry_count - 1:
                        raise
                    time.sleep(5)
                
        raise Exception("Failed to generate strategy after all retries")
    
    def improve_strategy(self, 
                        current_strategy: str, 
                        performance_feedback: Dict, 
                        market_data_summary: Dict,
                        retry_count: int = 3) -> str:
        """
        Improve existing strategy based on performance feedback.
        
        Args:
            current_strategy: Current strategy description
            performance_feedback: Performance metrics and feedback
            market_data_summary: Summary of market data characteristics
            retry_count: Number of retries on failure
            
        Returns:
            Improved strategy text
        """
        for attempt in range(retry_count):
            try:
                logger.info(f"Improving strategy (attempt {attempt + 1}/{retry_count})")
                
                improvement_prompt = self._create_improvement_prompt(
                    current_strategy, performance_feedback, market_data_summary
                )
                
                # Display prompt if enabled
                if Config.SHOW_LLM_PROMPTS:
                    self._display_prompt("STRATEGY IMPROVEMENT", improvement_prompt)
                
                # Get model-specific parameters
                api_params = self._get_model_specific_params()
                api_params["messages"] = [
                    {
                        "role": "system",
                        "content": "You are an expert quantitative analyst specializing in strategy optimization. "
                                 "Analyze performance feedback and improve investment strategies to achieve better results."
                    },
                    {
                        "role": "user",
                        "content": improvement_prompt
                    }
                ]
                
                response = self.client.chat.completions.create(**api_params)
                
                content = response.choices[0].message.content.strip()
                
                if not content:
                    raise ValueError("Empty response from LLM")
                
                # Display response if enabled
                if Config.SHOW_LLM_RESPONSES:
                    self._display_response("STRATEGY IMPROVEMENT", content)
                
                logger.info("Successfully improved strategy")
                return content
                
            except Exception as e:
                if "rate_limit" in str(e).lower():
                    logger.warning(f"Rate limit hit during improvement, waiting: {str(e)}")
                    time.sleep(60)
                else:
                    logger.error(f"Error improving strategy: {str(e)}")
                    if attempt == retry_count - 1:
                        raise
                    time.sleep(5)
        
        raise Exception("Failed to improve strategy after all retries")
    
    def _create_improvement_prompt(self, 
                                 current_strategy: str, 
                                 performance: Dict, 
                                 market_summary: Dict) -> str:
        """Create a prompt for strategy improvement."""
        
        prompt = f"""
STRATEGY IMPROVEMENT REQUEST

Current Strategy:
{current_strategy}

Performance Results:
- Total Return: {performance.get('total_return', 0):.2f}%
- Buy & Hold Return: {performance.get('buy_hold_return', 0):.2f}%
- Excess Return: {performance.get('excess_return', 0):.2f}%
- Sharpe Ratio: {performance.get('sharpe_ratio', 0):.2f}
- Max Drawdown: {performance.get('max_drawdown', 0):.2f}%
- Win Rate: {performance.get('win_rate', 0):.2f}%
- Number of Trades: {performance.get('num_trades', 0)}

Market Data Summary:
- Total Days: {market_summary.get('total_days', 0)}
- Average Daily Return: {market_summary.get('avg_daily_return', 0):.4f}%
- Volatility: {market_summary.get('volatility', 0):.2f}%
- Trend Direction: {market_summary.get('trend', 'Unknown')}

TASK: Analyze the performance and improve the strategy. Focus on:
1. Addressing poor performance areas (low returns, high drawdown, poor win rate)
2. Creating more executable and specific trading conditions
3. Better risk management
4. Optimizing entry and exit criteria

CRITICAL CONDITION FORMATTING REQUIREMENTS:
- ALL conditions must be valid Python expressions that can be evaluated directly
- Use ONLY these available indicators: RSI, SMA_20, SMA_50, EMA_12, EMA_26, MACD, MACD_signal, BB_upper, BB_middle, BB_lower, volume_sma, ATR, Close, Open, High, Low, Volume
- Use specific numerical thresholds (e.g., "RSI < 30", "Close > SMA_20 * 1.02")
- Avoid explanatory text or parenthetical comments in conditions
- Each condition must be a simple comparison: indicator operator number
- DO NOT include phrases like "to identify", "indicating", "for better", "as this", "when", etc.
- Keep conditions clean and executable: "RSI < 30" NOT "RSI < 30 to identify oversold conditions"

RESPONSE FORMAT (JSON):
{{
    "name": "Strategy Name",
    "description": "Brief strategy description and rationale",
    "buy_conditions": [
        "RSI < 30",
        "Close > SMA_20",
        "Volume > volume_sma * 1.5"
    ],
    "sell_conditions": [
        "RSI > 70",
        "Close < SMA_20 * 0.98",
        "profit > 0.1 OR loss > 0.05"
    ],
    "position_sizing": "Fixed percentage of portfolio (specify %)",
    "risk_management": "Stop loss and position management rules"
}}

Examples of GOOD conditions (clean and executable):
- "RSI < 30"
- "Close > SMA_50"
- "MACD > MACD_signal"
- "Close > BB_upper"
- "Volume > volume_sma * 1.5"
- "Close < SMA_20 * 0.98"
- "RSI > 70"
- "ATR > 2.0"

Examples of BAD conditions (DO NOT USE):
- "RSI < 30 to identify oversold conditions" (contains explanation)
- "Close > SMA_50 for trend confirmation" (contains explanation)
- "MACD > MACD_signal indicating bullish momentum" (contains explanation)
- "Volume > volume_sma * 1.5 as this suggests interest" (contains explanation)
- "Short-term RSI (5)" (indicator not available)
- "Price above 50-day SMA (trend confirmation)" (contains explanation)

Make conditions very specific and executable using available technical indicators.
"""
        
        return prompt
    
    def validate_api_key(self) -> bool:
        """Validate the OpenAI API key."""
        try:
            # Make a simple test request with a basic model that supports max_tokens
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5
            )
            return True
            
        except Exception as e:
            if "authentication" in str(e).lower() or "api_key" in str(e).lower():
                logger.error("Invalid OpenAI API key")
                return False
            else:
                logger.warning(f"Could not validate API key: {str(e)}")
                return False
    
    def _display_prompt(self, operation_type: str, prompt: str):
        """Display the prompt being sent to the LLM."""
        logger.debug(f"\n{'='*80}")
        logger.debug(f"🤖 LLM PROMPT - {operation_type}")
        logger.debug(f"{'='*80}")
        logger.debug(prompt)
        logger.debug(f"{'='*80}\n")
    
    def _display_response(self, operation_type: str, response: str):
        """Display the response received from the LLM."""
        logger.debug(f"\n{'='*80}")
        logger.debug(f"🧠 LLM RESPONSE - {operation_type}")
        logger.debug(f"{'='*80}")
        logger.debug(response)
        logger.debug(f"{'='*80}\n") 
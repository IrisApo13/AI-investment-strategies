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
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert quantitative analyst and investment strategist. "
                                     "Generate precise, executable investment strategies based on technical indicators."
                        },
                        {
                            "role": "user", 
                            "content": prompt
                        }
                    ],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0
                )
                
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
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert quantitative analyst specializing in strategy optimization. "
                                     "Analyze performance feedback and improve investment strategies to achieve better results."
                        },
                        {
                            "role": "user",
                            "content": improvement_prompt
                        }
                    ],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0
                )
                
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

Provide an improved strategy with the same JSON format:
{{
    "name": "Improved Strategy Name",
    "description": "Clear description of the improved approach",
    "buy_conditions": [
        "Specific, executable condition 1",
        "Specific, executable condition 2"
    ],
    "sell_conditions": [
        "Specific, executable condition 1", 
        "Specific, executable condition 2"
    ],
    "position_sizing": "Risk management approach",
    "risk_management": "Stop loss and risk controls"
}}

Make conditions very specific and executable using available technical indicators.
"""
        
        return prompt
    
    def validate_api_key(self) -> bool:
        """Validate the OpenAI API key."""
        try:
            # Make a simple test request
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
        print(f"\n{'='*80}")
        print(f"🤖 LLM PROMPT - {operation_type}")
        print(f"{'='*80}")
        print(prompt)
        print(f"{'='*80}\n")
    
    def _display_response(self, operation_type: str, response: str):
        """Display the response received from the LLM."""
        print(f"\n{'='*80}")
        print(f"🧠 LLM RESPONSE - {operation_type}")
        print(f"{'='*80}")
        print(response)
        print(f"{'='*80}\n") 
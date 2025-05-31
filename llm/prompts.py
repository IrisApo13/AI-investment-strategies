"""
Prompt engineering for LLM strategy generation.
"""
import pandas as pd
from typing import Dict, List
import logging

from config.settings import Config

logger = logging.getLogger(__name__)

class PromptGenerator:
    """Generates prompts for strategy creation and improvement."""
    
    def __init__(self):
        """Initialize the prompt generator."""
        pass
    
    def create_initial_strategy_prompt(self, 
                                     ticker: str, 
                                     market_data_summary: Dict,
                                     stock_info: Dict) -> str:
        """
        Create initial prompt for strategy generation.
        
        Args:
            ticker: Stock symbol
            market_data_summary: Summary statistics of market data
            stock_info: Basic stock information
            
        Returns:
            Formatted prompt string
        """
        
        prompt = f"""
INVESTMENT STRATEGY GENERATION REQUEST

Target Stock: {ticker} ({stock_info.get('name', 'Unknown')})
Sector: {stock_info.get('sector', 'Unknown')}
Industry: {stock_info.get('industry', 'Unknown')}

Market Data Analysis:
- Total Days: {market_data_summary.get('total_days', 0)}
- Price Range: ${market_data_summary.get('price_min', 0):.2f} - ${market_data_summary.get('price_max', 0):.2f}
- Average Daily Return: {market_data_summary.get('avg_daily_return', 0):.4f}%
- Volatility (Daily): {market_data_summary.get('volatility', 0):.4f}%
- Overall Trend: {market_data_summary.get('trend', 'Unknown')}
- Current RSI: {market_data_summary.get('current_rsi', 0):.1f}
- Current Price vs SMA50: {market_data_summary.get('price_vs_sma50', 0):.2f}%

Available Technical Indicators:
{', '.join(Config.TECHNICAL_INDICATORS)}

TASK: Generate a quantitative investment strategy for {ticker}. The strategy should:

1. Use specific, executable conditions based on technical indicators
2. Include clear buy and sell signals
3. Implement proper risk management
4. Be suitable for the stock's characteristics

CRITICAL REQUIREMENTS:
- ALL conditions must be executable using available technical indicators
- Use specific numerical thresholds (e.g., "RSI < 30", "Close > SMA_20 * 1.02")
- Avoid vague terms like "increasing", "trending up", "high volatility"
- Each condition should be a simple comparison that can be evaluated

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

Examples of GOOD conditions:
- "RSI < 30" (oversold)
- "Close > SMA_50" (above moving average)
- "MACD > MACD_signal" (MACD crossover)
- "BB_lower < Close < BB_middle" (Bollinger band position)
- "volume_ratio > 1.5" (high volume)

Examples of BAD conditions (DO NOT USE):
- "ATR is increasing" (not executable)
- "Stock is trending up" (vague)
- "High volatility period" (undefined)
- "Momentum is building" (unclear)

Generate a strategy now:
"""
        
        return prompt
    
    def create_data_summary(self, data: pd.DataFrame, ticker: str) -> Dict:
        """Create a summary of market data for prompt generation."""
        
        try:
            summary = {
                'total_days': len(data),
                'price_min': data['Close'].min(),
                'price_max': data['Close'].max(),
                'avg_daily_return': data['Close'].pct_change().mean() * 100,
                'volatility': data['Close'].pct_change().std() * 100,
                'current_rsi': data['RSI'].iloc[-1] if 'RSI' in data.columns else 0,
                'price_vs_sma50': ((data['Close'].iloc[-1] / data['SMA_50'].iloc[-1]) - 1) * 100 if 'SMA_50' in data.columns else 0
            }
            
            # Determine trend
            start_price = data['Close'].iloc[0]
            end_price = data['Close'].iloc[-1]
            total_return = (end_price / start_price - 1) * 100
            
            if total_return > 20:
                summary['trend'] = 'Strong Uptrend'
            elif total_return > 5:
                summary['trend'] = 'Uptrend'
            elif total_return > -5:
                summary['trend'] = 'Sideways'
            elif total_return > -20:
                summary['trend'] = 'Downtrend'
            else:
                summary['trend'] = 'Strong Downtrend'
            
            logger.info(f"Created market data summary for {ticker}")
            return summary
            
        except Exception as e:
            logger.error(f"Error creating data summary: {str(e)}")
            return {}
    
    def create_feedback_prompt(self, 
                             strategy_text: str,
                             performance: Dict,
                             issues: List[str] = None) -> str:
        """Create a prompt for providing feedback on strategy performance."""
        
        issues_text = ""
        if issues:
            issues_text = f"\nIssues Identified:\n" + "\n".join(f"- {issue}" for issue in issues)
        
        prompt = f"""
STRATEGY PERFORMANCE ANALYSIS

Strategy:
{strategy_text}

Performance Results:
- Total Return: {performance.get('total_return', 0):.2f}%
- Buy & Hold Return: {performance.get('buy_hold_return', 0):.2f}%
- Excess Return: {performance.get('excess_return', 0):.2f}%
- Sharpe Ratio: {performance.get('sharpe_ratio', 0):.3f}
- Max Drawdown: {performance.get('max_drawdown', 0):.2f}%
- Win Rate: {performance.get('win_rate', 0):.1f}%
- Number of Trades: {performance.get('num_trades', 0)}
- Time in Market: {performance.get('time_in_market', 0):.1f}%
{issues_text}

Provide a brief analysis of:
1. What worked well in this strategy
2. Key weaknesses and areas for improvement  
3. Specific suggestions for better performance

Keep the response concise and actionable.
"""
        
        return prompt
    
    def create_comparison_prompt(self, strategies: List[Dict], performances: List[Dict]) -> str:
        """Create a prompt for comparing multiple strategies."""
        
        comparison_data = []
        for i, (strategy, performance) in enumerate(zip(strategies, performances)):
            comparison_data.append(f"""
Strategy {i+1}: {strategy.get('name', 'Unknown')}
- Return: {performance.get('total_return', 0):.2f}%
- Sharpe: {performance.get('sharpe_ratio', 0):.3f}
- Drawdown: {performance.get('max_drawdown', 0):.2f}%
- Trades: {performance.get('num_trades', 0)}
""")
        
        prompt = f"""
STRATEGY COMPARISON ANALYSIS

{chr(10).join(comparison_data)}

Analyze these strategies and provide:
1. Which strategy performed best overall and why
2. Key differences in approach
3. Recommendations for combining the best elements
4. Suggestions for further improvement

Keep the analysis concise and practical.
"""
        
        return prompt 
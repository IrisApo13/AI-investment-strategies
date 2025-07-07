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
                                     market_data: pd.DataFrame,
                                     stock_info: Dict) -> str:
        """
        Create initial prompt for strategy generation with market data.
        
        Args:
            ticker: Stock symbol
            market_data: Raw daily market data with indicators
            stock_info: Basic stock information
            
        Returns:
            Formatted prompt string
        """
        
        # Choose between raw data and summary based on configuration
        if Config.MARKET_DATA_MODE == "raw_data":
            # Format the raw market data for the LLM
            market_data_text = self._format_raw_market_data(market_data)
            market_section_title = "RAW MARKET DATA (Daily Prices and Indicators):"
            task_description = "Analyze the raw market data above and generate a quantitative investment strategy for {ticker}. The strategy should:\n\n1. Use specific, executable conditions based on technical indicators\n2. Include clear buy and sell signals based on patterns you identify in the data\n3. Implement proper risk management\n4. Be tailored to the stock's specific characteristics and patterns you observe"
        else:
            # Use summarized market data
            market_summary = self.create_data_summary(market_data, ticker)
            market_data_text = self._format_market_summary(market_summary)
            market_section_title = "MARKET DATA ANALYSIS:"
            task_description = "Generate a quantitative investment strategy for {ticker} that leverages the market data insights above. The strategy should:\n\n1. Use specific, executable conditions based on technical indicators\n2. Include clear buy and sell signals that align with current market conditions\n3. Implement proper risk management suitable for the current volatility regime\n4. Be tailored to the stock's specific characteristics and current market position\n5. Consider the RSI, trend, and volume patterns identified in the analysis"
        
        prompt = f"""
INVESTMENT STRATEGY GENERATION REQUEST

Target Stock: {ticker} ({stock_info.get('name', 'Unknown')})
Sector: {stock_info.get('sector', 'Unknown')}
Industry: {stock_info.get('industry', 'Unknown')}

{market_section_title}
==============================================

{market_data_text}

Available Technical Indicators:
{', '.join(Config.TECHNICAL_INDICATORS)}

TASK: {task_description}

CRITICAL CONDITION FORMATTING REQUIREMENTS:
- ALL conditions must be valid Python expressions that can be evaluated directly
- Use ONLY these available indicators: RSI, SMA_20, SMA_50, EMA_12, EMA_26, MACD, MACD_signal, BB_upper, BB_middle, BB_lower, volume_sma, ATR, Close, Open, High, Low, Volume
- Use specific numerical thresholds based on your analysis of the market data
- Avoid explanatory text or parenthetical comments in conditions
- Each condition must be a simple comparison: indicator operator number
- DO NOT include phrases like "to identify", "indicating", "for better", "as this", "when", etc.
- Keep conditions clean and executable: "RSI < 30" NOT "RSI < 30 to identify oversold conditions"

RESPONSE FORMAT (JSON):
{{
    "name": "Strategy Name (based on your analysis)",
    "description": "Brief strategy description explaining the patterns you identified and how the strategy works",
    "buy_conditions": [
        "RSI < 30",
        "Close > SMA_20",
        "Volume > volume_sma * 1.5"
    ],
    "sell_conditions": [
        "RSI > 70",
        "Close < SMA_20 * 0.98"
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

Analyze the market data and generate a strategy for {ticker}:
"""
        
        return prompt
    
    def create_data_summary(self, data: pd.DataFrame, ticker: str) -> Dict:
        """Create a comprehensive summary of market data for prompt generation."""
        
        try:
            # Basic statistics
            summary = {
                'total_days': len(data),
                'price_min': data['Close'].min(),
                'price_max': data['Close'].max(),
                'current_price': data['Close'].iloc[-1],
                'avg_daily_return': data['Close'].pct_change().mean() * 100,
                'volatility': data['Close'].pct_change().std() * 100,
                'avg_volume': data['Volume'].mean() if 'Volume' in data.columns else 0,
                'current_volume': data['Volume'].iloc[-1] if 'Volume' in data.columns else 0,
            }
            
            # Technical indicators summary
            if 'RSI' in data.columns:
                summary.update({
                    'current_rsi': data['RSI'].iloc[-1],
                    'rsi_min': data['RSI'].min(),
                    'rsi_max': data['RSI'].max(),
                    'rsi_avg': data['RSI'].mean(),
                    'rsi_std': data['RSI'].std(),
                    'rsi_oversold_count': len(data[data['RSI'] < 30]),
                    'rsi_overbought_count': len(data[data['RSI'] > 70])
                })
            
            if 'SMA_20' in data.columns:
                summary.update({
                    'current_sma20': data['SMA_20'].iloc[-1],
                    'price_vs_sma20': ((data['Close'].iloc[-1] / data['SMA_20'].iloc[-1]) - 1) * 100,
                    'sma20_trend': 'Above' if data['Close'].iloc[-1] > data['SMA_20'].iloc[-1] else 'Below'
                })
            
            if 'SMA_50' in data.columns:
                summary.update({
                    'current_sma50': data['SMA_50'].iloc[-1],
                    'price_vs_sma50': ((data['Close'].iloc[-1] / data['SMA_50'].iloc[-1]) - 1) * 100,
                    'sma50_trend': 'Above' if data['Close'].iloc[-1] > data['SMA_50'].iloc[-1] else 'Below'
                })
            
            if 'MACD' in data.columns and 'MACD_signal' in data.columns:
                summary.update({
                    'current_macd': data['MACD'].iloc[-1],
                    'current_macd_signal': data['MACD_signal'].iloc[-1],
                    'macd_bullish': data['MACD'].iloc[-1] > data['MACD_signal'].iloc[-1],
                    'macd_crossovers': len(data[data['MACD'] > data['MACD_signal']]) - len(data[data['MACD'] < data['MACD_signal']])
                })
            
            if 'BB_upper' in data.columns and 'BB_lower' in data.columns:
                current_price = data['Close'].iloc[-1]
                bb_upper = data['BB_upper'].iloc[-1]
                bb_lower = data['BB_lower'].iloc[-1]
                bb_position = (current_price - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
                
                summary.update({
                    'current_bb_position': bb_position,
                    'bb_squeeze': bb_upper - bb_lower,
                    'bb_avg_squeeze': (data['BB_upper'] - data['BB_lower']).mean(),
                    'price_vs_bb_upper': ((current_price / bb_upper) - 1) * 100,
                    'price_vs_bb_lower': ((current_price / bb_lower) - 1) * 100
                })
            
            if 'ATR' in data.columns:
                summary.update({
                    'current_atr': data['ATR'].iloc[-1],
                    'atr_avg': data['ATR'].mean(),
                    'atr_volatility': data['ATR'].std()
                })
            
            if 'volume_sma' in data.columns:
                summary.update({
                    'current_volume_ratio': data['Volume'].iloc[-1] / data['volume_sma'].iloc[-1] if data['volume_sma'].iloc[-1] > 0 else 1,
                    'volume_above_avg_count': len(data[data['Volume'] > data['volume_sma']])
                })
            
            # Determine overall trend and market conditions
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
            
            # Market regime analysis
            summary['market_regime'] = self._analyze_market_regime(data)
            
            # Recent price action (last 20 days)
            recent_data = data.tail(20)
            if len(recent_data) > 0:
                summary['recent_volatility'] = recent_data['Close'].pct_change().std() * 100
                summary['recent_trend'] = 'Up' if recent_data['Close'].iloc[-1] > recent_data['Close'].iloc[0] else 'Down'
                summary['recent_volume_trend'] = 'Increasing' if recent_data['Volume'].iloc[-1] > recent_data['Volume'].iloc[0] else 'Decreasing'
            
            logger.info(f"Created comprehensive market data summary for {ticker}")
            return summary
            
        except Exception as e:
            logger.error(f"Error creating data summary: {str(e)}")
            return {}
    
    def _analyze_market_regime(self, data: pd.DataFrame) -> str:
        """Analyze the current market regime based on price action and volatility."""
        try:
            # Calculate rolling volatility
            rolling_vol = data['Close'].pct_change().rolling(20).std() * 100
            current_vol = rolling_vol.iloc[-1] if not rolling_vol.empty else 0
            avg_vol = rolling_vol.mean() if not rolling_vol.empty else 0
            
            # Calculate trend strength
            sma_20 = data['SMA_20'].iloc[-1] if 'SMA_20' in data.columns else data['Close'].iloc[-1]
            sma_50 = data['SMA_50'].iloc[-1] if 'SMA_50' in data.columns else data['Close'].iloc[-1]
            price = data['Close'].iloc[-1]
            
            # Determine regime
            if current_vol > avg_vol * 1.5:
                if price > sma_20 and sma_20 > sma_50:
                    return "High Volatility Bull Market"
                elif price < sma_20 and sma_20 < sma_50:
                    return "High Volatility Bear Market"
                else:
                    return "High Volatility Sideways"
            elif current_vol < avg_vol * 0.7:
                if price > sma_20 and sma_20 > sma_50:
                    return "Low Volatility Bull Market"
                elif price < sma_20 and sma_20 < sma_50:
                    return "Low Volatility Bear Market"
                else:
                    return "Low Volatility Sideways"
            else:
                if price > sma_20 and sma_20 > sma_50:
                    return "Normal Volatility Bull Market"
                elif price < sma_20 and sma_20 < sma_50:
                    return "Normal Volatility Bear Market"
                else:
                    return "Normal Volatility Sideways"
                    
        except Exception as e:
            logger.error(f"Error analyzing market regime: {str(e)}")
            return "Unknown"
    
    def _format_raw_market_data(self, data: pd.DataFrame) -> str:
        """Format raw market data for LLM consumption."""
        try:
            # Select key columns for the LLM
            key_columns = ['Date', 'Close', 'Open', 'High', 'Low', 'Volume', 
                          'RSI', 'SMA_20', 'SMA_50', 'MACD', 'MACD_signal',
                          'BB_upper', 'BB_middle', 'BB_lower', 'ATR', 'volume_sma']
            
            # Filter to available columns
            available_columns = [col for col in key_columns if col in data.columns]
            
            # Take the last 50 days of data to keep prompt manageable
            recent_data = data.tail(50)[available_columns].copy()
            
            # Format the data as a table
            formatted_lines = []
            
            # Header
            header = "Date       | Close   | Open    | High    | Low     | Volume  | RSI  | SMA20  | SMA50  | MACD   | Signal | BB_Upper| BB_Mid | BB_Lower| ATR  | Vol_SMA"
            formatted_lines.append(header)
            formatted_lines.append("-" * len(header))
            
            # Data rows (most recent first)
            for idx in reversed(recent_data.index):
                row = recent_data.loc[idx]
                date_str = str(row.get('Date', ''))[:10] if 'Date' in row else str(idx)[:10]
                
                line = f"{date_str} | "
                line += f"{row.get('Close', 0):7.2f} | "
                line += f"{row.get('Open', 0):7.2f} | "
                line += f"{row.get('High', 0):7.2f} | "
                line += f"{row.get('Low', 0):7.2f} | "
                line += f"{row.get('Volume', 0):7.0f} | "
                line += f"{row.get('RSI', 0):5.1f} | "
                line += f"{row.get('SMA_20', 0):6.2f} | "
                line += f"{row.get('SMA_50', 0):6.2f} | "
                line += f"{row.get('MACD', 0):6.3f} | "
                line += f"{row.get('MACD_signal', 0):6.3f} | "
                line += f"{row.get('BB_upper', 0):7.2f} | "
                line += f"{row.get('BB_middle', 0):6.2f} | "
                line += f"{row.get('BB_lower', 0):7.2f} | "
                line += f"{row.get('ATR', 0):4.2f} | "
                line += f"{row.get('volume_sma', 0):7.0f}"
                
                formatted_lines.append(line)
            
            # Add summary statistics
            formatted_lines.append("")
            formatted_lines.append("SUMMARY STATISTICS:")
            formatted_lines.append(f"Total Days: {len(data)}")
            formatted_lines.append(f"Price Range: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")
            formatted_lines.append(f"Current Price: ${data['Close'].iloc[-1]:.2f}")
            formatted_lines.append(f"Average Daily Return: {data['Close'].pct_change().mean() * 100:.4f}%")
            formatted_lines.append(f"Volatility: {data['Close'].pct_change().std() * 100:.4f}%")
            
            if 'RSI' in data.columns:
                formatted_lines.append(f"RSI Range: {data['RSI'].min():.1f} - {data['RSI'].max():.1f}")
                formatted_lines.append(f"Current RSI: {data['RSI'].iloc[-1]:.1f}")
            
            if 'SMA_20' in data.columns and 'SMA_50' in data.columns:
                formatted_lines.append(f"Current SMA20: ${data['SMA_20'].iloc[-1]:.2f}")
                formatted_lines.append(f"Current SMA50: ${data['SMA_50'].iloc[-1]:.2f}")
                formatted_lines.append(f"Price vs SMA20: {((data['Close'].iloc[-1] / data['SMA_20'].iloc[-1]) - 1) * 100:.1f}%")
                formatted_lines.append(f"Price vs SMA50: {((data['Close'].iloc[-1] / data['SMA_50'].iloc[-1]) - 1) * 100:.1f}%")
            
            return "\n".join(formatted_lines)
            
        except Exception as e:
            logger.error(f"Error formatting raw market data: {str(e)}")
            return f"Error formatting market data: {str(e)}"
    
    def _format_market_summary(self, market_summary: Dict) -> str:
        """Format summarized market data for LLM consumption."""
        try:
            formatted_lines = []
            
            # Basic statistics
            formatted_lines.append("Basic Statistics:")
            formatted_lines.append(f"- Total Days: {market_summary.get('total_days', 0)}")
            formatted_lines.append(f"- Price Range: ${market_summary.get('price_min', 0):.2f} - ${market_summary.get('price_max', 0):.2f}")
            formatted_lines.append(f"- Current Price: ${market_summary.get('current_price', 0):.2f}")
            formatted_lines.append(f"- Average Daily Return: {market_summary.get('avg_daily_return', 0):.4f}%")
            formatted_lines.append(f"- Volatility (Daily): {market_summary.get('volatility', 0):.4f}%")
            formatted_lines.append(f"- Overall Trend: {market_summary.get('trend', 'Unknown')}")
            formatted_lines.append(f"- Market Regime: {market_summary.get('market_regime', 'Unknown')}")
            
            # Technical indicators
            formatted_lines.append("")
            formatted_lines.append("Technical Indicators Analysis:")
            formatted_lines.append(f"- Current RSI: {market_summary.get('current_rsi', 0):.1f} (Range: {market_summary.get('rsi_min', 0):.1f}-{market_summary.get('rsi_max', 0):.1f}, Avg: {market_summary.get('rsi_avg', 0):.1f})")
            formatted_lines.append(f"- RSI Oversold Days: {market_summary.get('rsi_oversold_count', 0)} (RSI < 30)")
            formatted_lines.append(f"- RSI Overbought Days: {market_summary.get('rsi_overbought_count', 0)} (RSI > 70)")
            formatted_lines.append(f"- Current SMA20: ${market_summary.get('current_sma20', 0):.2f} (Price {market_summary.get('sma20_trend', 'Unknown')} SMA20 by {abs(market_summary.get('price_vs_sma20', 0)):.1f}%)")
            formatted_lines.append(f"- Current SMA50: ${market_summary.get('current_sma50', 0):.2f} (Price {market_summary.get('sma50_trend', 'Unknown')} SMA50 by {abs(market_summary.get('price_vs_sma50', 0)):.1f}%)")
            formatted_lines.append(f"- MACD: {market_summary.get('current_macd', 0):.4f} vs Signal: {market_summary.get('current_macd_signal', 0):.4f} ({'Bullish' if market_summary.get('macd_bullish', False) else 'Bearish'})")
            formatted_lines.append(f"- MACD Crossovers: {market_summary.get('macd_crossovers', 0)} (positive = more bullish crossovers)")
            formatted_lines.append(f"- Bollinger Band Position: {market_summary.get('current_bb_position', 0):.2f} (0=lower band, 1=upper band)")
            formatted_lines.append(f"- BB Squeeze: ${market_summary.get('bb_squeeze', 0):.2f} (Avg: ${market_summary.get('bb_avg_squeeze', 0):.2f})")
            formatted_lines.append(f"- ATR (Volatility): {market_summary.get('current_atr', 0):.2f} (Avg: {market_summary.get('atr_avg', 0):.2f})")
            formatted_lines.append(f"- Volume Ratio: {market_summary.get('current_volume_ratio', 0):.1f}x average")
            formatted_lines.append(f"- Volume Above Average Days: {market_summary.get('volume_above_avg_count', 0)} out of {market_summary.get('total_days', 0)}")
            
            # Recent market action
            formatted_lines.append("")
            formatted_lines.append("Recent Market Action (Last 20 Days):")
            formatted_lines.append(f"- Recent Volatility: {market_summary.get('recent_volatility', 0):.2f}%")
            formatted_lines.append(f"- Recent Price Trend: {market_summary.get('recent_trend', 'Unknown')}")
            formatted_lines.append(f"- Recent Volume Trend: {market_summary.get('recent_volume_trend', 'Unknown')}")
            
            # Strategy recommendations
            formatted_lines.append("")
            formatted_lines.append("STRATEGY RECOMMENDATIONS BASED ON MARKET DATA:")
            formatted_lines.append("=" * 50)
            formatted_lines.append("")
            formatted_lines.append("Based on the current market conditions:")
            formatted_lines.append(f"1. RSI Analysis: {market_summary.get('current_rsi', 0):.1f} is {'oversold' if market_summary.get('current_rsi', 50) < 30 else 'overbought' if market_summary.get('current_rsi', 50) > 70 else 'neutral'} - consider {'buying opportunities' if market_summary.get('current_rsi', 50) < 40 else 'selling opportunities' if market_summary.get('current_rsi', 50) > 60 else 'trend following'}")
            formatted_lines.append(f"2. Trend Analysis: Price is {market_summary.get('sma20_trend', 'Unknown')} SMA20 and {market_summary.get('sma50_trend', 'Unknown')} SMA50 - {'strong uptrend' if market_summary.get('sma20_trend') == 'Above' and market_summary.get('sma50_trend') == 'Above' else 'strong downtrend' if market_summary.get('sma20_trend') == 'Below' and market_summary.get('sma50_trend') == 'Below' else 'mixed signals'}")
            formatted_lines.append(f"3. Volatility: {market_summary.get('market_regime', 'Unknown')} - {'use wider stops' if 'High Volatility' in market_summary.get('market_regime', '') else 'use tighter stops' if 'Low Volatility' in market_summary.get('market_regime', '') else 'use standard stops'}")
            formatted_lines.append(f"4. Volume: {'High volume suggests strong moves' if market_summary.get('current_volume_ratio', 1) > 1.5 else 'Low volume suggests weak moves' if market_summary.get('current_volume_ratio', 1) < 0.7 else 'Normal volume conditions'}")
            
            return "\n".join(formatted_lines)
            
        except Exception as e:
            logger.error(f"Error formatting market summary: {str(e)}")
            return f"Error formatting market summary: {str(e)}"
    
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
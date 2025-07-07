"""
Iterative strategy improvement system using LLM feedback.
"""
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple
import json
import time

from llm.client import LLMClient
from analysis.performance_analyzer import PerformanceAnalyzer
from strategy.strategy import InvestmentStrategy
from backtesting.simulator import PortfolioSimulator
from data.stock_data import StockDataProvider
from config.settings import Config

logger = logging.getLogger(__name__)

class IterativeStrategyImprover:
    """Improves strategies iteratively using performance feedback."""
    
    def __init__(self):
        """Initialize the iterative improver."""
        self.llm_client = LLMClient()
        self.analyzer = PerformanceAnalyzer()
        self.simulator = PortfolioSimulator()
        self.data_provider = StockDataProvider()
        
        # Track improvement history
        self.improvement_history = []
        
    def improve_strategy_iteratively(self, initial_strategy: Dict, ticker: str, 
                                   period: str = '2y', max_iterations: int = 3) -> Dict:
        """
        Improve a strategy iteratively using performance feedback.
        
        Args:
            initial_strategy: Initial strategy dictionary
            ticker: Stock ticker to test on
            period: Time period for backtesting
            max_iterations: Maximum number of improvement iterations
            
        Returns:
            Dictionary with improvement results and best strategy
        """
        try:
            logger.info(f"Starting iterative improvement for strategy: {initial_strategy.get('name', 'Unknown')}")
            
            # Get market data
            market_data = self.data_provider.get_stock_data(ticker, period)
            
            current_strategy = initial_strategy.copy()
            best_strategy = None
            best_performance = None
            iteration_results = []
            
            for iteration in range(max_iterations):
                logger.info(f"=== Iteration {iteration + 1}/{max_iterations} ===")
                
                # Test current strategy
                result = self._test_and_analyze_strategy(current_strategy, market_data, ticker, iteration)
                iteration_results.append(result)
                
                # Track best performance
                current_total_return = result['performance']['total_return']
                if best_performance is None or current_total_return > best_performance['total_return']:
                    best_strategy = current_strategy.copy()
                    best_performance = result['performance'].copy()
                    logger.info(f"New best strategy found with {current_total_return:.1f}% return")
                
                # Generate feedback for next iteration (if not last iteration)
                if iteration < max_iterations - 1:
                    # Choose feedback strategy based on configuration
                    if Config.FEEDBACK_STATEGY == "advanced_feedback":
                        feedback = self._generate_enhanced_feedback(result)
                        logger.info("Generating improved strategy based on enhanced feedback with trade analysis...")
                    else:
                        feedback = self.analyzer.generate_feedback_for_next_iteration(result['analysis'])
                        logger.info("Generating improved strategy based on basic feedback...")
                    
                    # Generate improved strategy
                    improved_strategy = self._generate_improved_strategy(current_strategy, feedback, market_data)
                    
                    logger.debug(f"Improved strategy: {improved_strategy}")
                    
                    if improved_strategy:
                        current_strategy = improved_strategy
                        logger.info(f"Generated improved strategy: {improved_strategy.get('name', 'Unknown')}")
                    else:
                        logger.warning("Could not generate improved strategy, stopping iterations")
                        break
                
                # Add delay between iterations to respect API limits
                if iteration < max_iterations - 1:
                    time.sleep(2)
            
            # Prepare final results
            improvement_summary = {
                'initial_strategy': initial_strategy,
                'best_strategy': best_strategy,
                'best_performance': best_performance,
                'iteration_results': iteration_results,
                'improvement_achieved': best_performance['total_return'] - iteration_results[0]['performance']['total_return'],
                'total_iterations': len(iteration_results),
                'ticker_tested': ticker,
                'period_tested': period
            }
            
            # Store in history
            self.improvement_history.append(improvement_summary)
            
            logger.info(f"Improvement complete. Best return: {best_performance['total_return']:.1f}%")
            return improvement_summary
            
        except Exception as e:
            logger.error(f"Error in iterative improvement: {str(e)}")
            return {'error': str(e), 'initial_strategy': initial_strategy}
    
    def _test_and_analyze_strategy(self, strategy_dict: Dict, market_data: pd.DataFrame, 
                                 ticker: str, iteration: int) -> Dict:
        """Test a strategy and perform comprehensive analysis."""
        try:
            logger.info(f"Testing strategy: {strategy_dict.get('name', 'Unknown')}")
            
            # Create strategy object
            strategy = InvestmentStrategy(strategy_dict)
            
            # Generate signals
            signals = strategy.generate_signals(market_data)
            
            # Run backtest
            performance = self.simulator.run_backtest(market_data, signals, ticker)
            
            # Get detailed data for analysis
            trades_df = self.simulator.get_trades_df()
            portfolio_df = self.simulator.get_portfolio_history_df()
            
            # Perform comprehensive analysis
            analysis = self.analyzer.analyze_performance(
                performance, strategy_dict, trades_df, portfolio_df, market_data
            )
            
            result = {
                'iteration': iteration + 1,
                'strategy': strategy_dict,
                'performance': performance,
                'analysis': analysis,
                'trades_df': trades_df,  # Include trades for enhanced feedback
                'num_signals': {
                    'buy': signals['buy_signal'].sum(),
                    'sell': signals['sell_signal'].sum()
                },
                'num_trades': len(trades_df)
            }
            
            logger.info(f"Strategy performance: {performance.get('total_return', 0):.1f}% return, "
                       f"{performance.get('sharpe_ratio', 0):.2f} Sharpe, "
                       f"{performance.get('max_drawdown', 0):.1f}% max drawdown")
            
            return result
            
        except Exception as e:
            logger.error(f"Error testing strategy: {str(e)}")
            return {
                'iteration': iteration + 1,
                'strategy': strategy_dict,
                'error': str(e),
                'performance': {},
                'analysis': {},
                'trades_df': pd.DataFrame()
            }
    
    def _analyze_best_and_worst_trades(self, trades_df: pd.DataFrame) -> Tuple[List[Dict], List[Dict]]:
        """
        Analyze and extract the best and worst trades from the trading history.
        
        Args:
            trades_df: DataFrame containing trade history
            
        Returns:
            Tuple of (best_trades, worst_trades) - each containing trade details
        """
        try:
            if trades_df.empty:
                return [], []
            
            # Group trades into buy-sell pairs
            buy_trades = trades_df[trades_df['action'] == 'BUY'].copy()
            sell_trades = trades_df[trades_df['action'] == 'SELL'].copy()
            
            if len(buy_trades) == 0 or len(sell_trades) == 0:
                return [], []
            
            # Calculate trade returns for each buy-sell pair
            trade_pairs = []
            for i, sell_trade in sell_trades.iterrows():
                # Find the most recent buy trade before this sell
                buy_trades_before = buy_trades[buy_trades.index < i]
                if len(buy_trades_before) > 0:
                    buy_trade = buy_trades_before.iloc[-1]  # Most recent buy
                    
                    # Calculate trade return
                    trade_return = (sell_trade['price'] - buy_trade['price']) / buy_trade['price'] * 100
                    
                    # Calculate trade duration
                    duration_days = (sell_trade.name - buy_trade.name).days
                    
                    trade_pair = {
                        'buy_date': buy_trade.name,
                        'sell_date': sell_trade.name,
                        'buy_price': buy_trade['price'],
                        'sell_price': sell_trade['price'],
                        'return_pct': trade_return,
                        'duration_days': duration_days,
                        'shares': buy_trade['shares'],
                        'buy_metrics': buy_trade['metrics'],
                        'sell_metrics': sell_trade['metrics'],
                        'transaction_cost': buy_trade['transaction_cost'] + sell_trade['transaction_cost']
                    }
                    trade_pairs.append(trade_pair)
            
            if not trade_pairs:
                return [], []
            
            # Sort by return to find best and worst
            trade_pairs.sort(key=lambda x: x['return_pct'], reverse=True)
            
            # Get top 2 best and bottom 2 worst trades
            best_trades = trade_pairs[:2]
            worst_trades = trade_pairs[-2:] if len(trade_pairs) >= 2 else trade_pairs
            
            logger.info(f"Analyzed {len(trade_pairs)} trades. Best: {best_trades[0]['return_pct']:.1f}%, Worst: {worst_trades[0]['return_pct']:.1f}%")
            
            return best_trades, worst_trades
            
        except Exception as e:
            logger.error(f"Error analyzing trades: {str(e)}")
            return [], []
    
    def _format_trade_analysis_for_llm(self, best_trades: List[Dict], worst_trades: List[Dict]) -> str:
        """
        Format trade analysis into a structured format for LLM feedback.
        
        Args:
            best_trades: List of best performing trades
            worst_trades: List of worst performing trades
            
        Returns:
            Formatted string for LLM consumption
        """
        try:
            trade_analysis = "\nTRADE ANALYSIS - KEY INSIGHTS:\n"
            trade_analysis += "=" * 50 + "\n\n"
            
            # Best trades analysis
            trade_analysis += "🏆 BEST PERFORMING TRADES:\n"
            trade_analysis += "-" * 30 + "\n"
            
            for i, trade in enumerate(best_trades, 1):
                trade_analysis += f"Best Trade #{i}:\n"
                trade_analysis += f"  Return: {trade['return_pct']:.1f}%\n"
                trade_analysis += f"  Duration: {trade['duration_days']} days\n"
                trade_analysis += f"  Buy Date: {trade['buy_date'].strftime('%Y-%m-%d')}\n"
                trade_analysis += f"  Sell Date: {trade['sell_date'].strftime('%Y-%m-%d')}\n"
                trade_analysis += f"  Buy Price: ${trade['buy_price']:.2f}\n"
                trade_analysis += f"  Sell Price: ${trade['sell_price']:.2f}\n"
                
                # Market conditions at entry
                buy_metrics = trade['buy_metrics']
                trade_analysis += f"  Entry Conditions:\n"
                trade_analysis += f"    - RSI: {buy_metrics.get('RSI', 'N/A'):.1f}\n"
                trade_analysis += f"    - Close: ${buy_metrics.get('Close', 'N/A'):.2f}\n"
                trade_analysis += f"    - SMA_20: ${buy_metrics.get('SMA_20', 'N/A'):.2f}\n"
                trade_analysis += f"    - Volume: {buy_metrics.get('Volume', 'N/A'):,.0f}\n"
                trade_analysis += f"    - MACD: {buy_metrics.get('MACD', 'N/A'):.4f}\n"
                
                # Market conditions at exit
                sell_metrics = trade['sell_metrics']
                trade_analysis += f"  Exit Conditions:\n"
                trade_analysis += f"    - RSI: {sell_metrics.get('RSI', 'N/A'):.1f}\n"
                trade_analysis += f"    - Close: ${sell_metrics.get('Close', 'N/A'):.2f}\n"
                trade_analysis += f"    - SMA_20: ${sell_metrics.get('SMA_20', 'N/A'):.2f}\n"
                trade_analysis += f"    - Volume: {sell_metrics.get('Volume', 'N/A'):,.0f}\n"
                trade_analysis += f"    - MACD: {sell_metrics.get('MACD', 'N/A'):.4f}\n"
                trade_analysis += "\n"
            
            # Worst trades analysis
            trade_analysis += "📉 WORST PERFORMING TRADES:\n"
            trade_analysis += "-" * 30 + "\n"
            
            for i, trade in enumerate(worst_trades, 1):
                trade_analysis += f"Worst Trade #{i}:\n"
                trade_analysis += f"  Return: {trade['return_pct']:.1f}%\n"
                trade_analysis += f"  Duration: {trade['duration_days']} days\n"
                trade_analysis += f"  Buy Date: {trade['buy_date'].strftime('%Y-%m-%d')}\n"
                trade_analysis += f"  Sell Date: {trade['sell_date'].strftime('%Y-%m-%d')}\n"
                trade_analysis += f"  Buy Price: ${trade['buy_price']:.2f}\n"
                trade_analysis += f"  Sell Price: ${trade['sell_price']:.2f}\n"
                
                # Market conditions at entry
                buy_metrics = trade['buy_metrics']
                trade_analysis += f"  Entry Conditions:\n"
                trade_analysis += f"    - RSI: {buy_metrics.get('RSI', 'N/A'):.1f}\n"
                trade_analysis += f"    - Close: ${buy_metrics.get('Close', 'N/A'):.2f}\n"
                trade_analysis += f"    - SMA_20: ${buy_metrics.get('SMA_20', 'N/A'):.2f}\n"
                trade_analysis += f"    - Volume: {buy_metrics.get('Volume', 'N/A'):,.0f}\n"
                trade_analysis += f"    - MACD: {buy_metrics.get('MACD', 'N/A'):.4f}\n"
                
                # Market conditions at exit
                sell_metrics = trade['sell_metrics']
                trade_analysis += f"  Exit Conditions:\n"
                trade_analysis += f"    - RSI: {sell_metrics.get('RSI', 'N/A'):.1f}\n"
                trade_analysis += f"    - Close: ${sell_metrics.get('Close', 'N/A'):.2f}\n"
                trade_analysis += f"    - SMA_20: ${sell_metrics.get('SMA_20', 'N/A'):.2f}\n"
                trade_analysis += f"    - Volume: {sell_metrics.get('Volume', 'N/A'):,.0f}\n"
                trade_analysis += f"    - MACD: {sell_metrics.get('MACD', 'N/A'):.4f}\n"
                trade_analysis += "\n"
            
            # Key insights from trade analysis
            trade_analysis += "🔍 KEY INSIGHTS FROM TRADE ANALYSIS:\n"
            trade_analysis += "-" * 40 + "\n"
            
            if best_trades and worst_trades:
                # Compare entry conditions
                best_avg_rsi = sum(t['buy_metrics'].get('RSI', 50) for t in best_trades) / len(best_trades)
                worst_avg_rsi = sum(t['buy_metrics'].get('RSI', 50) for t in worst_trades) / len(worst_trades)
                
                best_avg_duration = sum(t['duration_days'] for t in best_trades) / len(best_trades)
                worst_avg_duration = sum(t['duration_days'] for t in worst_trades) / len(worst_trades)
                
                trade_analysis += f"• Best trades had average RSI of {best_avg_rsi:.1f} vs {worst_avg_rsi:.1f} for worst trades\n"
                trade_analysis += f"• Best trades held for {best_avg_duration:.0f} days vs {worst_avg_duration:.0f} days for worst trades\n"
                
                # Volume analysis
                best_volume_ratio = []
                worst_volume_ratio = []
                for trade in best_trades:
                    if 'Volume' in trade['buy_metrics'] and 'volume_sma' in trade['buy_metrics']:
                        ratio = trade['buy_metrics']['Volume'] / trade['buy_metrics']['volume_sma']
                        best_volume_ratio.append(ratio)
                
                for trade in worst_trades:
                    if 'Volume' in trade['buy_metrics'] and 'volume_sma' in trade['buy_metrics']:
                        ratio = trade['buy_metrics']['Volume'] / trade['buy_metrics']['volume_sma']
                        worst_volume_ratio.append(ratio)
                
                if best_volume_ratio and worst_volume_ratio:
                    avg_best_volume = sum(best_volume_ratio) / len(best_volume_ratio)
                    avg_worst_volume = sum(worst_volume_ratio) / len(worst_volume_ratio)
                    trade_analysis += f"• Best trades had {avg_best_volume:.1f}x average volume vs {avg_worst_volume:.1f}x for worst trades\n"
            
            return trade_analysis
            
        except Exception as e:
            logger.error(f"Error formatting trade analysis: {str(e)}")
            return "Error analyzing trades"
    
    def _generate_enhanced_feedback(self, result: Dict) -> str:
        """
        Generate enhanced feedback including trade analysis for next iteration.
        
        Args:
            result: Dictionary containing test results and analysis
            
        Returns:
            Enhanced feedback string for LLM
        """
        try:
            # Get basic feedback from analyzer
            basic_feedback = self.analyzer.generate_feedback_for_next_iteration(result['analysis'])
            
            # Get trade analysis
            trades_df = result.get('trades_df', pd.DataFrame())
            best_trades, worst_trades = self._analyze_best_and_worst_trades(trades_df)
            
            # Format trade analysis
            trade_analysis = self._format_trade_analysis_for_llm(best_trades, worst_trades)
            
            # Combine into enhanced feedback
            enhanced_feedback = f"""
{basic_feedback}

{trade_analysis}

STRATEGY IMPROVEMENT RECOMMENDATIONS BASED ON TRADE ANALYSIS:

Based on the analysis of the best and worst trades, focus on these specific improvements:

1. ENTRY CONDITIONS: 
   - Study the market conditions that led to successful trades
   - Avoid conditions that led to losing trades
   - Consider volume confirmation for better entry timing

2. EXIT CONDITIONS:
   - Analyze what market conditions triggered successful exits
   - Improve stop-loss or take-profit mechanisms
   - Consider holding periods that worked well

3. RISK MANAGEMENT:
   - Implement better position sizing based on trade success patterns
   - Add filters to avoid conditions that led to worst trades
   - Consider market volatility at entry points

Generate a new strategy that incorporates these specific insights from the trade analysis.
"""
            
            return enhanced_feedback
            
        except Exception as e:
            logger.error(f"Error generating enhanced feedback: {str(e)}")
            # Fall back to basic feedback
            return self.analyzer.generate_feedback_for_next_iteration(result['analysis'])
    
    def _generate_improved_strategy(self, current_strategy: Dict, feedback: str, market_data: pd.DataFrame = None) -> Optional[Dict]:
        """Generate an improved strategy based on feedback and market data."""
        try:
            # Create raw market data insights if provided
            market_insights = ""
            if market_data is not None:
                from llm.prompts import PromptGenerator
                prompt_gen = PromptGenerator()
                
                # Choose between raw data and summary based on configuration
                if Config.MARKET_DATA_MODE == "raw_data":
                    raw_data_text = prompt_gen._format_raw_market_data(market_data)
                    market_insights = f"""

CURRENT RAW MARKET DATA (Last 50 Days):
=======================================
{raw_data_text}

ANALYZE THIS RAW DATA TO IMPROVE THE STRATEGY:
- Look for patterns in price movements, RSI levels, and volume
- Identify which conditions led to successful trades
- Consider the current market position and recent trends
- Adapt thresholds based on the actual data ranges you observe
"""
                else:
                    market_summary = prompt_gen.create_data_summary(market_data, "current_ticker")
                    market_summary_text = prompt_gen._format_market_summary(market_summary)
                    market_insights = f"""

CURRENT MARKET DATA INSIGHTS:
============================
{market_summary_text}

USE THESE MARKET INSIGHTS TO IMPROVE THE STRATEGY:
- Consider the current market regime and volatility conditions
- Adapt to the RSI, trend, and volume patterns identified
- Use the strategy recommendations provided above
- Adjust thresholds based on the market analysis
"""

            improvement_prompt = f"""
You are an expert quantitative analyst. Based on the performance feedback, trade analysis, and current market data below, create an improved investment strategy.

CURRENT STRATEGY:
{json.dumps(current_strategy, indent=2)}

PERFORMANCE FEEDBACK AND TRADE ANALYSIS:
{feedback}{market_insights}

Generate a new, improved strategy that addresses the specific issues mentioned in the feedback and incorporates insights from both the trade analysis AND current market conditions. The strategy should:

1. Keep the same general structure but improve the conditions based on trade performance
2. Address the specific weaknesses identified in the analysis
3. Implement the suggested improvements from trade analysis
4. Use different technical indicators or thresholds where needed
5. Incorporate patterns from successful trades and avoid patterns from losing trades
6. Adapt to current market conditions (RSI levels, trend, volatility, volume)

CRITICAL CONDITION FORMATTING REQUIREMENTS:
- ALL conditions must be valid Python expressions that can be evaluated directly
- Use ONLY these available indicators: RSI, SMA_20, SMA_50, EMA_12, EMA_26, MACD, MACD_signal, BB_upper, BB_middle, BB_lower, volume_sma, ATR, Close, Open, High, Low, Volume
- Keep conditions clean and executable: "RSI < 30" NOT "RSI < 30 to identify oversold conditions"
- DO NOT include explanatory text, parenthetical comments, or phrases like "to identify", "indicating", "for better", etc.
- Each condition must be a simple comparison: indicator operator number

IMPORTANT: Pay special attention to:
- Trade analysis section: Entry/exit conditions that worked well vs those that didn't
- Market data insights: Current RSI levels, trend direction, volatility regime
- Combine both insights to create optimal conditions for current market state

Respond with a JSON object in this exact format:
{{
    "name": "Improved [Original Name] v2.0 (Market-Adapted)",
    "description": "Brief description of key improvements made based on trade analysis and market conditions",
    "buy_conditions": [
        "RSI < 25",
        "Close > SMA_20",
        "Volume > volume_sma * 1.2"
    ],
    "sell_conditions": [
        "RSI > 75",
        "Close < SMA_20 * 0.98"
    ],
    "position_sizing": "Position sizing strategy (e.g., '15% of portfolio')",
    "risk_management": "Risk management approach (e.g., '3% stop loss')"
}}

GOOD condition examples:
- "RSI < 30", "Close > SMA_50", "MACD > MACD_signal", "Volume > volume_sma * 1.5"

BAD condition examples (DO NOT USE):
- "RSI < 30 to identify oversold conditions"
- "Close > SMA_50 for trend confirmation"
- "Short-term RSI (5)" (indicator not available)
"""
            
            response = self.llm_client.generate_strategy(improvement_prompt, operation_type="ITERATIVE STRATEGY IMPROVEMENT")
            
            # Parse improved strategy
            improved_strategy = self._parse_strategy_response(response)
            
            if improved_strategy and 'name' in improved_strategy:
                return improved_strategy
            else:
                logger.warning("Could not parse improved strategy from LLM response")
                return None
                
        except Exception as e:
            logger.error(f"Error generating improved strategy: {str(e)}")
            return None
    
    def _parse_strategy_response(self, response: str) -> Optional[Dict]:
        """Parse LLM response into strategy dictionary."""
        try:
            import re
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                strategy_dict = json.loads(json_match.group())
                
                # Validate required fields
                required_fields = ['name', 'description', 'buy_conditions', 'sell_conditions']
                if all(field in strategy_dict for field in required_fields):
                    return strategy_dict
                else:
                    logger.warning(f"Strategy missing required fields: {required_fields}")
                    return None
            else:
                logger.warning("Could not find JSON in LLM response")
                return None
                
        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode error: {str(e)}")
            return None
        except Exception as e:
            logger.warning(f"Error parsing strategy response: {str(e)}")
            return None
    
    def compare_strategies(self, improvement_summary: Dict) -> str:
        """Generate a comparison report of the improvement process."""
        try:
            initial_perf = improvement_summary['iteration_results'][0]['performance']
            best_perf = improvement_summary['best_performance']
            
            improvement = best_perf['total_return'] - initial_perf['total_return']
            
            report = f"""
STRATEGY IMPROVEMENT REPORT
{'=' * 50}

INITIAL STRATEGY: {improvement_summary['initial_strategy']['name']}
Initial Return: {initial_perf.get('total_return', 0):.1f}%
Initial Sharpe: {initial_perf.get('sharpe_ratio', 0):.2f}
Initial Max Drawdown: {initial_perf.get('max_drawdown', 0):.1f}%

BEST STRATEGY: {improvement_summary['best_strategy']['name']}
Best Return: {best_perf.get('total_return', 0):.1f}%
Best Sharpe: {best_perf.get('sharpe_ratio', 0):.2f}
Best Max Drawdown: {best_perf.get('max_drawdown', 0):.1f}%

IMPROVEMENT ACHIEVED:
Total Return Improvement: {improvement:+.1f}%
Sharpe Ratio Improvement: {best_perf.get('sharpe_ratio', 0) - initial_perf.get('sharpe_ratio', 0):+.2f}
Max Drawdown Change: {best_perf.get('max_drawdown', 0) - initial_perf.get('max_drawdown', 0):+.1f}%

ITERATION SUMMARY:
Total Iterations: {improvement_summary['total_iterations']}
"""
            
            # Add iteration-by-iteration breakdown
            for i, result in enumerate(improvement_summary['iteration_results']):
                perf = result['performance']
                report += f"""
Iteration {i+1}: {result['strategy']['name']}
  Return: {perf.get('total_return', 0):.1f}%
  Sharpe: {perf.get('sharpe_ratio', 0):.2f}
  Signals: {result['num_signals']['buy']} buy, {result['num_signals']['sell']} sell"""
            
            return report
            
        except Exception as e:
            return f"Error generating comparison report: {str(e)}"
    
    def get_improvement_history(self) -> List[Dict]:
        """Get the history of all improvement sessions."""
        return self.improvement_history
    
    def save_improvement_results(self, improvement_summary: Dict, filename: str = None):
        """Save improvement results to file."""
        try:
            if filename is None:
                timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                filename = f"strategy_improvement_{timestamp}.json"
            
            with open(filename, 'w') as f:
                # Convert pandas objects to serializable format
                serializable_summary = self._make_serializable(improvement_summary)
                json.dump(serializable_summary, f, indent=2, default=str)
            
            logger.info(f"Improvement results saved to {filename}")
            
        except Exception as e:
            logger.error(f"Error saving improvement results: {str(e)}")
    
    def _make_serializable(self, obj):
        """Convert pandas objects to JSON serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, (pd.Series, pd.DataFrame)):
            return obj.to_dict()
        else:
            return obj 
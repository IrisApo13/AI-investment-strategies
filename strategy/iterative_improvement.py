"""
Iterative strategy improvement system using LLM feedback.
"""
import pandas as pd
import logging
from typing import Dict, List, Optional
import json
import time

from llm.client import LLMClient
from analysis.performance_analyzer import PerformanceAnalyzer
from strategy.strategy import InvestmentStrategy
from backtesting.simulator import PortfolioSimulator
from data.stock_data import StockDataProvider

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
                    feedback = self.analyzer.generate_feedback_for_next_iteration(result['analysis'])
                    logger.info("Generating improved strategy based on feedback...")
                    
                    # Generate improved strategy
                    improved_strategy = self._generate_improved_strategy(current_strategy, feedback)
                    
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
                'analysis': {}
            }
    
    def _generate_improved_strategy(self, current_strategy: Dict, feedback: str) -> Optional[Dict]:
        """Generate an improved strategy based on feedback."""
        try:
            improvement_prompt = f"""
You are an expert quantitative analyst. Based on the performance feedback below, create an improved investment strategy.

CURRENT STRATEGY:
{json.dumps(current_strategy, indent=2)}

PERFORMANCE FEEDBACK:
{feedback}

Generate a new, improved strategy that addresses the specific issues mentioned in the feedback. The strategy should:

1. Keep the same general structure but improve the conditions
2. Address the specific weaknesses identified
3. Implement the suggested improvements
4. Use different technical indicators or thresholds where needed

CRITICAL CONDITION FORMATTING REQUIREMENTS:
- ALL conditions must be valid Python expressions that can be evaluated directly
- Use ONLY these available indicators: RSI, SMA_20, SMA_50, EMA_12, EMA_26, MACD, MACD_signal, BB_upper, BB_middle, BB_lower, volume_sma, ATR, Close, Open, High, Low, Volume
- Keep conditions clean and executable: "RSI < 30" NOT "RSI < 30 to identify oversold conditions"
- DO NOT include explanatory text, parenthetical comments, or phrases like "to identify", "indicating", "for better", etc.
- Each condition must be a simple comparison: indicator operator number

Respond with a JSON object in this exact format:
{{
    "name": "Improved [Original Name] v2.0",
    "description": "Brief description of key improvements made",
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
            
            response = self.llm_client.generate_strategy(improvement_prompt)
            
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
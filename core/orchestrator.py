"""
Main orchestrator for the AI backtesting system.
Coordinates strategy generation, backtesting, and iterative improvement.
"""
import logging
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime

from data.stock_data import StockDataProvider
from llm.client import LLMClient
from llm.prompts import PromptGenerator  
from llm.parser import StrategyParser
from strategy.strategy import InvestmentStrategy
from backtesting.simulator import PortfolioSimulator
from backtesting.evaluator import PerformanceEvaluator
from config.settings import Config

logger = logging.getLogger(__name__)

class AIBacktestOrchestrator:
    """Main orchestrator for the AI backtesting system."""
    
    def __init__(self):
        """Initialize the orchestrator with all components."""
        self.data_provider = StockDataProvider()
        self.llm_client = LLMClient()
        self.prompt_generator = PromptGenerator()
        self.strategy_parser = StrategyParser()
        self.simulator = PortfolioSimulator()
        self.evaluator = PerformanceEvaluator()
        
        # Session state
        self.current_ticker = None
        self.market_data = None
        self.stock_info = None
        self.strategy_history = []
        self.performance_history = []
        
        logger.info("AI Backtest Orchestrator initialized")
    
    def run_backtesting_session(self, 
                               ticker: str, 
                               max_iterations: int = None,
                               target_score: float = None) -> Dict:
        """
        Run a complete AI backtesting session with iterative improvement.
        
        Args:
            ticker: Stock symbol to analyze
            max_iterations: Maximum number of iterations (default from config)
            target_score: Target performance score to achieve
            
        Returns:
            Complete session results
        """
        try:
            max_iterations = max_iterations or Config.MAX_ITERATIONS
            target_score = target_score or 80.0
            
            logger.info(f"Starting backtesting session for {ticker}")
            
            # Initialize session
            session_results = self._initialize_session(ticker)
            if not session_results['success']:
                return session_results
            
            # Iterative improvement loop
            for iteration in range(1, max_iterations + 1):
                logger.info(f"=== ITERATION {iteration} ===")
                
                # Generate or improve strategy
                if iteration == 1:
                    strategy_result = self._generate_initial_strategy()
                else:
                    if Config.FEEDBACK_STATEGY == "basic_feedback":
                        strategy_result = self._improve_current_strategy(iteration)
                    else:
                        strategy_result = self._generate_initial_strategy()
                
                if not strategy_result['success']:
                    logger.error(f"Strategy generation failed in iteration {iteration}")
                    continue
                
                # Backtest the strategy
                backtest_result = self._backtest_strategy(strategy_result['strategy'])
                
                if not backtest_result['success']:
                    logger.error(f"Backtesting failed in iteration {iteration}")
                    continue
                
                # Record results
                self.strategy_history.append(strategy_result['strategy'])
                self.performance_history.append(backtest_result['evaluation'])
                
                # Display iteration results
                self._display_iteration_results(iteration, strategy_result, backtest_result)
                
                # Check if target achieved
                current_score = backtest_result['evaluation']['performance_score']
                if current_score >= target_score:
                    logger.info(f"Target score {target_score} achieved! (Score: {current_score})")
                    break
                
                # Ask user whether to continue
                if not self._should_continue_iteration(iteration, max_iterations, current_score):
                    break
            
            # Generate final session summary
            session_summary = self._generate_session_summary()
            
            return {
                'success': True,
                'ticker': ticker,
                'iterations_completed': len(self.strategy_history),
                'best_strategy': self._get_best_strategy(),
                'best_performance': self._get_best_performance(),
                'strategy_history': self.strategy_history,
                'performance_history': self.performance_history,
                'session_summary': session_summary
            }
            
        except Exception as e:
            logger.error(f"Error in backtesting session: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'ticker': ticker
            }
    
    def _initialize_session(self, ticker: str) -> Dict:
        """Initialize a new backtesting session."""
        try:
            logger.info(f"Initializing session for {ticker}")
            
            # Validate ticker
            if not self.data_provider.validate_ticker(ticker):
                return {
                    'success': False,
                    'error': f"Invalid ticker symbol: {ticker}"
                }
            
            # Get market data
            self.market_data = self.data_provider.get_stock_data(ticker)
            if self.market_data.empty:
                return {
                    'success': False,
                    'error': f"No data available for {ticker}"
                }
            
            # Get stock info
            self.stock_info = self.data_provider.get_stock_info(ticker)
            
            # Reset session state
            self.current_ticker = ticker
            self.strategy_history = []
            self.performance_history = []
            
            logger.info(f"Session initialized. Retrieved {len(self.market_data)} days of data")
            
            return {
                'success': True,
                'data_days': len(self.market_data),
                'stock_info': self.stock_info
            }
            
        except Exception as e:
            logger.error(f"Error initializing session: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _generate_initial_strategy(self) -> Dict:
        """Generate the initial strategy for the session."""
        try:
            logger.info("Generating initial strategy")
            
            # Create market data summary
            market_summary = self.prompt_generator.create_data_summary(
                self.market_data, self.current_ticker
            )
            
            # Generate prompt
            prompt = self.prompt_generator.create_initial_strategy_prompt(
                self.current_ticker, market_summary, self.stock_info
            )
            
            # Generate strategy using LLM
            llm_response = self.llm_client.generate_strategy(prompt, operation_type="INITIAL STRATEGY GENERATION")
            
            # Parse the response
            strategy_dict = self.strategy_parser.parse_strategy_response(llm_response)
            
            if not strategy_dict:
                return {
                    'success': False,
                    'error': "Failed to parse strategy from LLM response"
                }
            
            # Create strategy object
            strategy = InvestmentStrategy(strategy_dict)
            
            return {
                'success': True,
                'strategy': strategy,
                'raw_response': llm_response
            }
            
        except Exception as e:
            logger.error(f"Error generating initial strategy: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _improve_current_strategy(self, iteration: int) -> Dict:
        """Improve the current strategy based on performance feedback."""
        try:
            logger.info(f"Improving strategy for iteration {iteration}")
            
            # Get the last strategy and performance
            last_strategy = self.strategy_history[-1]
            last_performance = self.performance_history[-1]
            
            # Create market data summary
            market_summary = self.prompt_generator.create_data_summary(
                self.market_data, self.current_ticker
            )
            
            # Generate improved strategy using LLM
            llm_response = self.llm_client.improve_strategy(
                str(last_strategy), last_performance, market_summary
            )
            
            # Parse the response
            strategy_dict = self.strategy_parser.parse_strategy_response(llm_response)
            
            if not strategy_dict:
                return {
                    'success': False,
                    'error': "Failed to parse improved strategy from LLM response"
                }
            
            # Create strategy object
            strategy = InvestmentStrategy(strategy_dict)
            
            return {
                'success': True,
                'strategy': strategy,
                'raw_response': llm_response
            }
            
        except Exception as e:
            logger.error(f"Error improving strategy: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _backtest_strategy(self, strategy: InvestmentStrategy) -> Dict:
        """Backtest a strategy and evaluate performance."""
        try:
            logger.info(f"Backtesting strategy: {strategy.name}")
            
            # Generate signals
            signals = strategy.generate_signals(self.market_data)
            
            # Run backtest
            performance_metrics = self.simulator.run_backtest(
                self.market_data, signals, self.current_ticker
            )
            
            # Get portfolio history
            portfolio_history = self.simulator.get_portfolio_history_df()
            
            # Evaluate performance
            evaluation = self.evaluator.evaluate_performance(
                performance_metrics, portfolio_history
            )
            
            return {
                'success': True,
                'performance_metrics': performance_metrics,
                'evaluation': evaluation,
                'signals': signals,
                'portfolio_history': portfolio_history
            }
            
        except Exception as e:
            logger.error(f"Error backtesting strategy: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _display_iteration_results(self, 
                                 iteration: int, 
                                 strategy_result: Dict, 
                                 backtest_result: Dict):
        """Display results for a single iteration."""
        try:
            strategy = strategy_result['strategy']
            evaluation = backtest_result['evaluation']
            signals = backtest_result.get('signals')
            
            print(f"\n{'='*60}")
            print(f"ITERATION {iteration} RESULTS")
            print(f"{'='*60}")
            print(f"\nStrategy: {strategy.name}")
            print(f"Description: {strategy.description}")
            
            print(f"\nKey Performance Metrics:")
            print(f"- Performance Score: {evaluation.get('performance_score', 0):.1f}/100")
            print(f"- Total Return: {evaluation.get('total_return', 0):.2f}%")
            print(f"- Buy & Hold Return: {evaluation.get('buy_hold_return', 0):.2f}%")
            print(f"- Sharpe Ratio: {evaluation.get('sharpe_ratio', 0):.3f}")
            print(f"- Max Drawdown: {evaluation.get('max_drawdown', 0):.2f}%")
            print(f"- Win Rate: {evaluation.get('win_rate', 0):.1f}%")
            print(f"- Number of Trades: {evaluation.get('num_trades', 0)}")
            
            print(f"\nBuy Conditions:")
            for i, condition in enumerate(strategy.buy_conditions, 1):
                print(f"  {i}. {condition}")
            
            print(f"\nSell Conditions:")
            for i, condition in enumerate(strategy.sell_conditions, 1):
                print(f"  {i}. {condition}")
            
            # Plot strategy visualization if enabled
            if Config.SHOW_STRATEGY_PLOTS:
                print(f"\nGenerating strategy visualization...")
                try:
                    strategy.plot_strategy(self.market_data, signals, figsize=(16, 12))
                except Exception as plot_error:
                    logger.warning(f"Could not generate plot: {str(plot_error)}")
                    print(f"Note: Visualization could not be generated ({str(plot_error)})")
            else:
                print(f"\nStrategy visualization disabled (Config.SHOW_STRATEGY_PLOTS = False)")
                
        except Exception as e:
            logger.error(f"Error displaying iteration results: {str(e)}")
    
    def _should_continue_iteration(self, 
                                 current_iteration: int, 
                                 max_iterations: int, 
                                 current_score: float) -> bool:
        """Ask user whether to continue with next iteration."""
        try:
            if current_iteration >= max_iterations:
                print(f"\nReached maximum iterations ({max_iterations})")
                return False
            
            print(f"\nCurrent iteration: {current_iteration}/{max_iterations}")
            print(f"Current performance score: {current_score:.1f}/100")
            
            
            if Config.INTERACTIVE:
                while True:
                    response = input("\nContinue to next iteration? (y/n): ").strip().lower()
                    if response in ['y', 'yes']:
                        return True
                    elif response in ['n', 'no']:
                        return False
                    else:
                        print("Please enter 'y' or 'n'")
            else:
                return True

                        
        except Exception as e:
            logger.error(f"Error in user input: {str(e)}")
            return False
    
    def _get_best_strategy(self) -> Optional[InvestmentStrategy]:
        """Get the best performing strategy from the session."""
        if not self.performance_history:
            return None
        
        best_idx = max(range(len(self.performance_history)), 
                      key=lambda i: self.performance_history[i]['performance_score'])
        
        return self.strategy_history[best_idx]
    
    def _get_best_performance(self) -> Optional[Dict]:
        """Get the best performance metrics from the session."""
        if not self.performance_history:
            return None
        
        best_idx = max(range(len(self.performance_history)), 
                      key=lambda i: self.performance_history[i]['performance_score'])
        
        return self.performance_history[best_idx]
    
    def _generate_session_summary(self) -> str:
        """Generate a comprehensive session summary."""
        try:
            if not self.strategy_history:
                return "No strategies were generated in this session."
            
            best_strategy = self._get_best_strategy()
            best_performance = self._get_best_performance()
            
            summary = f"""
AI BACKTESTING SESSION SUMMARY
==============================

Ticker: {self.current_ticker} ({self.stock_info.get('name', 'Unknown')})
Data Period: {len(self.market_data)} days
Iterations Completed: {len(self.strategy_history)}

BEST STRATEGY PERFORMANCE:
{self.evaluator.generate_performance_summary(best_performance)}

BEST STRATEGY DETAILS:
{str(best_strategy)}

ITERATION COMPARISON:
"""
            
            for i, (strategy, performance) in enumerate(zip(self.strategy_history, self.performance_history), 1):
                summary += f"\nIteration {i}: {strategy.name}"
                summary += f"\n  Score: {performance['performance_score']:.1f}/100"
                summary += f" | Return: {performance['total_return']:.2f}%"
                summary += f" | Sharpe: {performance['sharpe_ratio']:.3f}"
                summary += f" | Trades: {performance['num_trades']}"
            
            return summary.strip()
            
        except Exception as e:
            logger.error(f"Error generating session summary: {str(e)}")
            return "Error generating session summary"
    
    def save_session_results(self, results: Dict, filename: str = None) -> bool:
        """Save session results to a file."""
        try:
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"backtest_session_{results['ticker']}_{timestamp}.json"
            
            # Prepare data for serialization
            serializable_results = {
                'success': results['success'],
                'ticker': results['ticker'],
                'iterations_completed': results['iterations_completed'],
                'session_summary': results['session_summary']
            }
            
            # Add strategy and performance data
            if 'strategy_history' in results:
                serializable_results['strategies'] = [
                    strategy.to_dict() for strategy in results['strategy_history']
                ]
            
            if 'performance_history' in results:
                serializable_results['performance_history'] = results['performance_history']
            
            # Save to file
            with open(filename, 'w') as f:
                json.dump(serializable_results, f, indent=2, default=str)
            
            logger.info(f"Session results saved to {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving session results: {str(e)}")
            return False 
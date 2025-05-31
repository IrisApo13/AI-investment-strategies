"""
Test suite for the AI Investment Strategy Backtesting System.
"""
import unittest
import pandas as pd
import numpy as np
import os
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

# Set test environment
os.environ['OPENAI_API_KEY'] = 'test-key-for-testing'

from config.settings import Config
from data.stock_data import StockDataProvider
from strategy.strategy import InvestmentStrategy
from backtesting.simulator import PortfolioSimulator
from backtesting.evaluator import PerformanceEvaluator
from llm.parser import StrategyParser

class TestSystemIntegration(unittest.TestCase):
    """Integration tests for the complete system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_ticker = "AAPL"
        self.sample_strategy = {
            'name': 'Test RSI Strategy',
            'description': 'Simple RSI-based strategy for testing',
            'buy_conditions': ['RSI < 30', 'Close > SMA_20'],
            'sell_conditions': ['RSI > 70', 'Close < SMA_20'],
            'position_sizing': '10% of portfolio',
            'risk_management': '5% stop loss'
        }
        
        # Create sample data
        self.sample_data = self._create_sample_data()
    
    def _create_sample_data(self) -> pd.DataFrame:
        """Create sample stock data for testing."""
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        dates = dates[dates.weekday < 5]  # Remove weekends
        
        # Generate realistic stock data
        np.random.seed(42)
        n_days = len(dates)
        
        # Price walk
        returns = np.random.normal(0.0005, 0.02, n_days)  # Daily returns
        prices = 100 * np.exp(np.cumsum(returns))  # Starting at $100
        
        # OHLCV data
        data = pd.DataFrame({
            'Open': prices * (1 + np.random.normal(0, 0.001, n_days)),
            'High': prices * (1 + np.abs(np.random.normal(0, 0.01, n_days))),
            'Low': prices * (1 - np.abs(np.random.normal(0, 0.01, n_days))),
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, n_days)
        }, index=dates)
        
        # Ensure High >= Close >= Low and High >= Open >= Low
        data['High'] = np.maximum(data['High'], np.maximum(data['Open'], data['Close']))
        data['Low'] = np.minimum(data['Low'], np.minimum(data['Open'], data['Close']))
        
        # Add technical indicators
        data['SMA_20'] = data['Close'].rolling(20).mean()
        data['SMA_50'] = data['Close'].rolling(50).mean()
        data['RSI'] = self._calculate_rsi(data['Close'])
        data['volume_sma'] = data['Volume'].rolling(20).mean()
        
        # Drop NaN values
        data = data.dropna()
        
        return data
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def test_strategy_creation_and_signals(self):
        """Test strategy creation and signal generation."""
        print("\n🔍 Testing strategy creation and signal generation...")
        
        # Create strategy
        strategy = InvestmentStrategy(self.sample_strategy)
        
        # Verify strategy attributes
        self.assertEqual(strategy.name, 'Test RSI Strategy')
        self.assertEqual(len(strategy.buy_conditions), 2)
        self.assertEqual(len(strategy.sell_conditions), 2)
        self.assertGreater(strategy.position_size, 0)
        
        # Generate signals
        signals = strategy.generate_signals(self.sample_data)
        
        # Verify signals structure
        self.assertIsInstance(signals, pd.DataFrame)
        self.assertIn('buy_signal', signals.columns)
        self.assertIn('sell_signal', signals.columns)
        self.assertEqual(len(signals), len(self.sample_data))
        
        # Verify signal types
        self.assertTrue(signals['buy_signal'].dtype == bool)
        self.assertTrue(signals['sell_signal'].dtype == bool)
        
        print(f"✓ Strategy created successfully")
        print(f"✓ Generated {signals['buy_signal'].sum()} buy signals")
        print(f"✓ Generated {signals['sell_signal'].sum()} sell signals")
    
    def test_portfolio_simulation(self):
        """Test portfolio simulation with realistic scenarios."""
        print("\n🔍 Testing portfolio simulation...")
        
        # Create strategy and generate signals
        strategy = InvestmentStrategy(self.sample_strategy)
        signals = strategy.generate_signals(self.sample_data)
        
        # Run simulation
        simulator = PortfolioSimulator()
        performance = simulator.run_backtest(self.sample_data, signals, self.test_ticker)
        
        # Verify performance structure
        required_metrics = [
            'total_return', 'annualized_return', 'buy_hold_return',
            'sharpe_ratio', 'max_drawdown', 'win_rate', 'num_trades'
        ]
        
        for metric in required_metrics:
            self.assertIn(metric, performance)
            self.assertIsInstance(performance[metric], (int, float))
        
        # Verify logical constraints
        self.assertGreaterEqual(performance['final_portfolio_value'], 0)
        self.assertLessEqual(performance['max_drawdown'], 0)  # Drawdown should be negative
        self.assertGreaterEqual(performance['win_rate'], 0)
        self.assertLessEqual(performance['win_rate'], 100)
        
        print(f"✓ Simulation completed successfully")
        print(f"✓ Total return: {performance['total_return']:.2f}%")
        print(f"✓ Number of trades: {performance['num_trades']}")
        print(f"✓ Win rate: {performance['win_rate']:.1f}%")
    
    def test_performance_evaluation(self):
        """Test performance evaluation and scoring."""
        print("\n🔍 Testing performance evaluation...")
        
        # Create sample performance metrics
        sample_metrics = {
            'total_return': 15.5,
            'annualized_return': 12.3,
            'buy_hold_return': 10.2,
            'excess_return': 5.3,
            'sharpe_ratio': 0.85,
            'max_drawdown': -8.2,
            'win_rate': 62.5,
            'num_trades': 24,
            'volatility': 18.5,
            'final_portfolio_value': 115500
        }
        
        # Evaluate performance
        evaluator = PerformanceEvaluator()
        evaluation = evaluator.evaluate_performance(sample_metrics)
        
        # Verify evaluation structure
        self.assertIn('performance_score', evaluation)
        self.assertIn('performance_rating', evaluation)
        self.assertIn('risk_level', evaluation)
        
        # Verify score range
        score = evaluation['performance_score']
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)
        
        # Verify rating
        rating = evaluation['performance_rating']
        valid_ratings = ['Excellent', 'Good', 'Average', 'Poor', 'Very Poor']
        self.assertIn(rating, valid_ratings)
        
        print(f"✓ Performance evaluation completed")
        print(f"✓ Performance score: {score:.1f}/100")
        print(f"✓ Rating: {rating}")
        print(f"✓ Risk level: {evaluation['risk_level']}")
    
    def test_strategy_parsing(self):
        """Test strategy parsing from various formats."""
        print("\n🔍 Testing strategy parsing...")
        
        parser = StrategyParser()
        
        # Test JSON format
        json_response = '''
        {
            "name": "Momentum Strategy",
            "description": "Buy on momentum, sell on reversal",
            "buy_conditions": ["RSI < 35", "Close > SMA_20"],
            "sell_conditions": ["RSI > 65", "Close < SMA_20"],
            "position_sizing": "15% of portfolio",
            "risk_management": "Stop loss at 3%"
        }
        '''
        
        strategy_dict = parser.parse_strategy_response(json_response)
        
        # Verify parsing
        self.assertIsNotNone(strategy_dict)
        self.assertEqual(strategy_dict['name'], 'Momentum Strategy')
        self.assertEqual(len(strategy_dict['buy_conditions']), 2)
        self.assertEqual(len(strategy_dict['sell_conditions']), 2)
        
        # Test with markdown formatting
        markdown_response = '''
        ```json
        {
            "name": "Mean Reversion Strategy",
            "description": "Buy oversold, sell overbought",
            "buy_conditions": ["RSI < 25", "BB_lower > Close"],
            "sell_conditions": ["RSI > 75", "BB_upper < Close"]
        }
        ```
        '''
        
        strategy_dict2 = parser.parse_strategy_response(markdown_response)
        self.assertIsNotNone(strategy_dict2)
        self.assertEqual(strategy_dict2['name'], 'Mean Reversion Strategy')
        
        print(f"✓ JSON parsing successful")
        print(f"✓ Markdown parsing successful")
        print(f"✓ Strategy validation working")

def run_tests():
    """Run all tests with detailed output."""
    print("🧪 Starting AI Backtesting System Test Suite")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSystemIntegration)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=open(os.devnull, 'w'))
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    if result.wasSuccessful():
        print(f"✅ All {result.testsRun} tests passed successfully!")
        print("\n🎉 System is ready for use!")
    else:
        print(f"❌ {len(result.failures)} test(s) failed")
        print(f"❌ {len(result.errors)} error(s) occurred")
        
        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"- {test}: {traceback}")
        
        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"- {test}: {traceback}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1) 
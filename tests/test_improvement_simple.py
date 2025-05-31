#!/usr/bin/env python3
"""
Simple test script for iterative strategy improvement system.
"""
import os
import logging

# Load environment variables (OpenAI API key will be loaded from .env)
from dotenv import load_dotenv
load_dotenv()

from strategy.iterative_improvement import IterativeStrategyImprover

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_simple_improvement():
    """Test the iterative improvement with a simple strategy."""
    print("🔄 Testing Iterative Strategy Improvement")
    print("=" * 50)
    
    # Create a deliberately simple/poor strategy for improvement
    initial_strategy = {
        'name': 'Simple RSI Strategy v1.0',
        'description': 'Basic RSI strategy that likely needs improvement',
        'buy_conditions': [
            'RSI < 30'  # Very basic oversold condition
        ],
        'sell_conditions': [
            'RSI > 70'  # Very basic overbought condition  
        ],
        'position_sizing': '10% of portfolio',
        'risk_management': 'None'
    }
    
    print("📋 Initial Strategy:")
    print(f"  Name: {initial_strategy['name']}")
    print(f"  Description: {initial_strategy['description']}")
    print(f"  Buy Conditions:")
    for condition in initial_strategy['buy_conditions']:
        print(f"    - {condition}")
    print(f"  Sell Conditions:")
    for condition in initial_strategy['sell_conditions']:
        print(f"    - {condition}")
    print(f"  Position Sizing: {initial_strategy['position_sizing']}")
    print(f"  Risk Management: {initial_strategy['risk_management']}")
    
    # Initialize the improver
    improver = IterativeStrategyImprover()
    
    print(f"\n🚀 Starting improvement process...")
    print(f"  Ticker: SPY (S&P 500 ETF)")
    print(f"  Period: 6 months (for faster testing)")
    print(f"  Max iterations: 2")
    
    # Run the improvement
    try:
        results = improver.improve_strategy_iteratively(
            initial_strategy=initial_strategy,
            ticker='SPY',
            period='6mo',
            max_iterations=2
        )
        
        if 'error' in results:
            print(f"❌ Error: {results['error']}")
            return
        
        # Show results
        print(f"\n📊 RESULTS SUMMARY")
        print("=" * 30)
        
        initial_perf = results['iteration_results'][0]['performance']
        best_perf = results['best_performance']
        
        print(f"Initial Strategy Performance:")
        print(f"  Total Return: {initial_perf.get('total_return', 0):.1f}%")
        print(f"  Sharpe Ratio: {initial_perf.get('sharpe_ratio', 0):.2f}")
        print(f"  Max Drawdown: {initial_perf.get('max_drawdown', 0):.1f}%")
        print(f"  Win Rate: {initial_perf.get('win_rate', 0):.1f}%")
        print(f"  Number of Trades: {initial_perf.get('num_trades', 0)}")
        
        print(f"\nBest Strategy Performance:")
        print(f"  Total Return: {best_perf.get('total_return', 0):.1f}%")
        print(f"  Sharpe Ratio: {best_perf.get('sharpe_ratio', 0):.2f}")
        print(f"  Max Drawdown: {best_perf.get('max_drawdown', 0):.1f}%")
        print(f"  Win Rate: {best_perf.get('win_rate', 0):.1f}%")
        print(f"  Number of Trades: {best_perf.get('num_trades', 0)}")
        
        improvement = results['improvement_achieved']
        print(f"\nImprovement Achieved: {improvement:+.1f}%")
        
        if improvement > 0:
            print(f"🎉 Success! Strategy was improved by {improvement:.1f}%")
        elif improvement == 0:
            print(f"📊 No improvement found (original strategy was best)")
        else:
            print(f"📉 Performance declined by {abs(improvement):.1f}% (this can happen)")
        
        # Show the best strategy found
        best_strategy = results['best_strategy']
        print(f"\n🏆 BEST STRATEGY:")
        print(f"  Name: {best_strategy['name']}")
        print(f"  Description: {best_strategy['description']}")
        print(f"  Buy Conditions:")
        for condition in best_strategy['buy_conditions']:
            print(f"    - {condition}")
        print(f"  Sell Conditions:")
        for condition in best_strategy['sell_conditions']:
            print(f"    - {condition}")
        print(f"  Position Sizing: {best_strategy['position_sizing']}")
        print(f"  Risk Management: {best_strategy['risk_management']}")
        
        # Show iteration-by-iteration progress
        print(f"\n📈 ITERATION PROGRESS:")
        for i, iteration_result in enumerate(results['iteration_results']):
            perf = iteration_result['performance']
            strategy_name = iteration_result['strategy']['name']
            signals = iteration_result['num_signals']
            
            print(f"  Iteration {i+1}: {strategy_name}")
            print(f"    Return: {perf.get('total_return', 0):.1f}%")
            print(f"    Sharpe: {perf.get('sharpe_ratio', 0):.2f}")
            print(f"    Signals: {signals['buy']} buy, {signals['sell']} sell")
        
        print(f"\n✅ Iterative improvement test completed successfully!")
        
        # Show key insights
        if 'weaknesses_identified' in results['iteration_results'][0]['analysis']:
            weaknesses = results['iteration_results'][0]['analysis']['weaknesses_identified']
            if 'weaknesses' in weaknesses:
                print(f"\n🔍 KEY WEAKNESSES IDENTIFIED:")
                for i, weakness in enumerate(weaknesses['weaknesses'][:3], 1):
                    print(f"  {i}. {weakness.get('weakness', 'Unknown')}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error during improvement: {str(e)}")
        import traceback
        traceback.print_exc()

def test_performance_analyzer_only():
    """Test just the performance analyzer functionality."""
    print(f"\n🔬 Testing Performance Analyzer")
    print("=" * 40)
    
    # Test with a simple strategy
    test_strategy = {
        'name': 'Test Strategy',
        'description': 'For analyzer testing',
        'buy_conditions': ['RSI < 35'],
        'sell_conditions': ['RSI > 65'],
        'position_sizing': '15%',
        'risk_management': 'None'
    }
    
    try:
        from analysis.performance_analyzer import PerformanceAnalyzer
        from strategy.strategy import InvestmentStrategy
        from backtesting.simulator import PortfolioSimulator
        from data.stock_data import StockDataProvider
        
        # Get data and run backtest
        provider = StockDataProvider()
        data = provider.get_stock_data('SPY', '6mo')
        
        strategy = InvestmentStrategy(test_strategy)
        signals = strategy.generate_signals(data)
        
        simulator = PortfolioSimulator()
        performance = simulator.run_backtest(data, signals, 'AAPL')
        
        trades_df = simulator.get_trades_df()
        portfolio_df = simulator.get_portfolio_history_df()
        
        print(f"📊 Basic Performance:")
        print(f"  Return: {performance.get('total_return', 0):.1f}%")
        print(f"  Trades: {len(trades_df)}")
        print(f"  Signals: {signals['buy_signal'].sum()} buy, {signals['sell_signal'].sum()} sell")
        
        # Test the analyzer
        analyzer = PerformanceAnalyzer()
        analysis = analyzer.analyze_performance(
            performance, test_strategy, trades_df, portfolio_df, data
        )
        
        print(f"\n🧠 LLM Analysis Results:")
        if 'weaknesses_identified' in analysis:
            weaknesses = analysis['weaknesses_identified']
            if 'error' not in weaknesses and 'weaknesses' in weaknesses:
                print(f"  Weaknesses identified: {len(weaknesses['weaknesses'])}")
                for weakness in weaknesses['weaknesses'][:2]:
                    print(f"    - {weakness.get('weakness', 'Unknown')}")
            else:
                print(f"  Weakness analysis failed: {weaknesses.get('error', 'Unknown error')}")
        
        if 'improvement_suggestions' in analysis:
            improvements = analysis['improvement_suggestions']
            if 'error' not in improvements and 'improvements' in improvements:
                print(f"  Improvements suggested: {len(improvements['improvements'])}")
                for improvement in improvements['improvements'][:2]:
                    print(f"    - {improvement.get('category', 'Unknown')}: {improvement.get('suggested_change', 'Unknown')}")
            else:
                print(f"  Improvement suggestions failed: {improvements.get('error', 'Unknown error')}")
        
        print(f"✅ Performance analyzer test completed!")
        
    except Exception as e:
        print(f"❌ Error in performance analyzer test: {str(e)}")

def main():
    """Run the improvement tests."""
    print("🧪 SIMPLE ITERATIVE IMPROVEMENT TEST")
    print("=" * 50)
    
    # Test the performance analyzer first
    test_performance_analyzer_only()
    
    # Then test the full improvement system
    test_simple_improvement()
    
    print(f"\n🎯 All tests completed!")

if __name__ == "__main__":
    main() 
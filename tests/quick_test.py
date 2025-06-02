#!/usr/bin/env python3
"""
Quick test to demonstrate the working AI backtesting system.
"""
import os
# Load environment variables (OpenAI API key will be loaded from .env)
from dotenv import load_dotenv
load_dotenv()

from data.stock_data import StockDataProvider
from strategy.strategy import InvestmentStrategy
from backtesting.simulator import PortfolioSimulator
from backtesting.evaluator import PerformanceEvaluator

def quick_test():
    """Run a quick test to show the system working."""
    print("🚀 Quick System Test")
    print("=" * 40)
    
    # Get real market data
    provider = StockDataProvider()
    data = provider.get_stock_data('AAPL', '1y')
    stock_info = provider.get_stock_info('AAPL')
    
    print(f"📊 Data: {len(data)} days of {stock_info['name']}")
    print(f"📈 Price range: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")
    print(f"📊 RSI range: {data['RSI'].min():.1f} - {data['RSI'].max():.1f}")
    
    # Test simple strategy
    strategy_dict = {
        'name': 'Simple RSI Strategy',
        'description': 'Buy when RSI < 40, sell when RSI > 60',
        'buy_conditions': ['RSI < 40'],
        'sell_conditions': ['RSI > 60'],
        'position_sizing': '20% of portfolio',
        'risk_management': '5% stop loss'
    }
    
    print(f"\n📋 Testing: {strategy_dict['name']}")
    print(f"   Buy when: {', '.join(strategy_dict['buy_conditions'])}")
    print(f"   Sell when: {', '.join(strategy_dict['sell_conditions'])}")
    
    # Create and test strategy
    strategy = InvestmentStrategy(strategy_dict)
    signals = strategy.generate_signals(data)
    
    # Run backtest
    simulator = PortfolioSimulator()
    performance = simulator.run_backtest(data, signals, 'AAPL')
    
    # Evaluate
    evaluator = PerformanceEvaluator()
    evaluation = evaluator.evaluate_performance(performance)
    
    print(f"\n📊 RESULTS:")
    print(f"   Signals: {signals['buy_signal'].sum()} buys, {signals['sell_signal'].sum()} sells")
    print(f"   Return: {performance['total_return']:.2f}%")
    print(f"   Buy & Hold: {performance['buy_hold_return']:.2f}%")
    print(f"   Excess Return: {performance['excess_return']:.2f}%")
    print(f"   Sharpe Ratio: {performance['sharpe_ratio']:.3f}")
    print(f"   Max Drawdown: {performance['max_drawdown']:.2f}%")
    print(f"   Win Rate: {performance['win_rate']:.1f}%")
    print(f"   Performance Score: {evaluation['performance_score']:.1f}/100")
    print(f"   Rating: {evaluation['performance_rating']}")
    
    # Status
    total_signals = signals['buy_signal'].sum() + signals['sell_signal'].sum()
    print(f"\n✅ System Status: {'WORKING!' if total_signals > 0 else 'ISSUE - No signals'}")
    print(f"   Total signals: {total_signals}")

if __name__ == "__main__":
    quick_test() 
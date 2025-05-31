#!/usr/bin/env python3
"""
Debug script to investigate why strategies generate 0 trades.
"""
import os
os.environ['OPENAI_API_KEY'] = 'test-key-for-debugging'

from data.stock_data import StockDataProvider
from strategy.strategy import InvestmentStrategy
import pandas as pd
import numpy as np

def debug_signal_generation():
    """Debug the signal generation process."""
    print("🔍 Debugging Signal Generation")
    print("=" * 50)
    
    # Get real data
    provider = StockDataProvider()
    data = provider.get_stock_data('AAPL', '1y')
    
    print(f"📊 Data Info:")
    print(f"  Shape: {data.shape}")
    print(f"  Date range: {data.index[0]} to {data.index[-1]}")
    print(f"  Columns: {list(data.columns)}")
    
    # Check for NaN values
    nan_counts = data.isnull().sum()
    print(f"\n❌ NaN values per column:")
    for col, count in nan_counts.items():
        if count > 0:
            print(f"  {col}: {count}")
    
    # Check RSI values
    print(f"\n📈 RSI Statistics:")
    rsi_stats = data['RSI'].describe()
    print(f"  Min: {rsi_stats['min']:.2f}")
    print(f"  Max: {rsi_stats['max']:.2f}")
    print(f"  Mean: {rsi_stats['mean']:.2f}")
    print(f"  Values < 30: {(data['RSI'] < 30).sum()}")
    print(f"  Values > 70: {(data['RSI'] > 70).sum()}")
    
    # Test simple strategy
    print(f"\n🎯 Testing Simple RSI Strategy")
    strategy_dict = {
        'name': 'Debug RSI Strategy',
        'description': 'Simple RSI test',
        'buy_conditions': ['RSI < 50'],
        'sell_conditions': ['RSI > 50'],
        'position_sizing': '10%',
        'risk_management': 'None'
    }
    
    strategy = InvestmentStrategy(strategy_dict)
    print(f"  Strategy created: {strategy.name}")
    print(f"  Buy conditions: {strategy.buy_conditions}")
    print(f"  Sell conditions: {strategy.sell_conditions}")
    
    # Generate signals with debugging
    signals = strategy.generate_signals(data)
    
    print(f"\n📊 Signal Results:")
    print(f"  Buy signals: {signals['buy_signal'].sum()}")
    print(f"  Sell signals: {signals['sell_signal'].sum()}")
    print(f"  Total rows: {len(signals)}")
    
    # Check condition evaluation manually
    print(f"\n🔍 Manual Condition Check:")
    rsi_below_50 = (data['RSI'] < 50).sum()
    rsi_above_50 = (data['RSI'] > 50).sum()
    print(f"  RSI < 50 count: {rsi_below_50}")
    print(f"  RSI > 50 count: {rsi_above_50}")
    
    # Test with extreme conditions
    print(f"\n🧪 Testing Extreme Conditions:")
    extreme_strategy = {
        'name': 'Always Buy Strategy',
        'description': 'Should always trigger',
        'buy_conditions': ['RSI > 0'],  # Should always be true
        'sell_conditions': ['RSI < 1000'],  # Should always be true
        'position_sizing': '10%',
        'risk_management': 'None'
    }
    
    extreme_strat = InvestmentStrategy(extreme_strategy)
    extreme_signals = extreme_strat.generate_signals(data)
    print(f"  Extreme buy signals: {extreme_signals['buy_signal'].sum()}")
    print(f"  Extreme sell signals: {extreme_signals['sell_signal'].sum()}")
    
    # Debug individual condition evaluation
    print(f"\n🔬 Testing Individual Condition Evaluation:")
    test_row = data.iloc[-1]  # Last row
    print(f"  Test RSI value: {test_row['RSI']}")
    
    # Test the evaluation function directly
    result = strategy._evaluate_single_condition(test_row, 'RSI < 50')
    print(f"  'RSI < 50' evaluates to: {result}")
    
    result2 = strategy._evaluate_single_condition(test_row, 'RSI > 50')
    print(f"  'RSI > 50' evaluates to: {result2}")

if __name__ == "__main__":
    debug_signal_generation() 
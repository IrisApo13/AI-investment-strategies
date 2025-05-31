#!/usr/bin/env python3
"""
Test script for condition cleaning logic.
"""
import os
os.environ['OPENAI_API_KEY'] = 'test-key-for-testing'

from strategy.strategy import InvestmentStrategy
from data.stock_data import StockDataProvider
import pandas as pd

def test_condition_cleaning():
    """Test the condition cleaning functionality."""
    print("🧪 Testing Condition Cleaning")
    print("=" * 40)
    
    # Test cases based on the warning messages
    verbose_conditions = [
        "RSI < 30 to identify oversold conditions more precisely, indicating a stronger potential rebound",
        "RSI (14) < 30",
        "Close > SMA_20 for better entry points",
        "Volume > volume_sma * 1.5 as this suggests increased interest",
        "MACD > MACD_signal which means bullish momentum",
        "RSI > 70 (indicating overbought)",
        "Price < BB_lower when we want to buy the dip",
        "RSI < 35",  # This should remain unchanged
        "Close > SMA_50 * 1.02",  # This should remain unchanged
    ]
    
    # Create a dummy strategy to test the cleaning
    strategy_dict = {
        'name': 'Test Strategy',
        'description': 'For testing condition cleaning',
        'buy_conditions': ['RSI < 30'],  # Simple condition
        'sell_conditions': ['RSI > 70'],
        'position_sizing': '10%',
        'risk_management': 'None'
    }
    
    strategy = InvestmentStrategy(strategy_dict)
    
    print("🔍 Testing condition cleaning:")
    for i, condition in enumerate(verbose_conditions, 1):
        cleaned = strategy._clean_condition(condition)
        print(f"{i:2d}. Original: '{condition}'")
        print(f"    Cleaned:  '{cleaned}'")
        
        # Test if the cleaned condition can be evaluated
        provider = StockDataProvider()
        data = provider.get_stock_data('AAPL', '1y')
        test_row = data.iloc[-1]
        
        try:
            result = strategy._evaluate_single_condition(test_row, condition)
            status = "✅ WORKS" if result in [True, False] else "❌ ERROR"
        except Exception as e:
            status = f"❌ ERROR: {str(e)[:50]}..."
        
        print(f"    Status:   {status}")
        print()

def test_full_strategy_with_verbose_conditions():
    """Test a complete strategy with verbose AI-generated conditions."""
    print("🎯 Testing Full Strategy with Verbose Conditions")
    print("=" * 50)
    
    # Simulate what AI might generate
    verbose_strategy = {
        'name': 'AI Generated Verbose Strategy',
        'description': 'Strategy with verbose conditions like AI generates',
        'buy_conditions': [
            'RSI < 30 to identify oversold conditions more precisely',
            'Close > SMA_20 for better entry timing'
        ],
        'sell_conditions': [
            'RSI > 70 indicating overbought conditions',
            'Close < SMA_20 * 0.98 as a stop loss measure'
        ],
        'position_sizing': '15% of portfolio',
        'risk_management': '5% stop loss'
    }
    
    print("📋 Testing strategy:")
    print(f"  Name: {verbose_strategy['name']}")
    print(f"  Buy conditions:")
    for condition in verbose_strategy['buy_conditions']:
        print(f"    - {condition}")
    print(f"  Sell conditions:")
    for condition in verbose_strategy['sell_conditions']:
        print(f"    - {condition}")
    
    # Test the strategy
    strategy = InvestmentStrategy(verbose_strategy)
    
    # Get data
    provider = StockDataProvider()
    data = provider.get_stock_data('AAPL', '1y')
    
    # Generate signals
    signals = strategy.generate_signals(data)
    
    buy_signals = signals['buy_signal'].sum()
    sell_signals = signals['sell_signal'].sum()
    
    print(f"\n📊 Results:")
    print(f"  Buy signals: {buy_signals}")
    print(f"  Sell signals: {sell_signals}")
    print(f"  Status: {'✅ WORKING' if (buy_signals + sell_signals) > 0 else '❌ STILL BROKEN'}")
    
    if buy_signals + sell_signals > 0:
        print(f"  🎉 Condition cleaning successfully fixed the verbose condition issue!")
    else:
        print(f"  😞 Still having issues, need further debugging...")

if __name__ == "__main__":
    test_condition_cleaning()
    print("\n" + "="*60 + "\n")
    test_full_strategy_with_verbose_conditions() 
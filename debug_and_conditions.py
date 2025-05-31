#!/usr/bin/env python3
"""
Debug AND condition evaluation.
"""
import os
os.environ['OPENAI_API_KEY'] = 'test-key-for-testing'

from strategy.strategy import InvestmentStrategy
from data.stock_data import StockDataProvider

def debug_and_conditions():
    """Debug AND condition evaluation."""
    print("🔍 Debugging AND Condition Evaluation")
    print("=" * 45)
    
    # Get data
    provider = StockDataProvider()
    data = provider.get_stock_data('AAPL', '1y')
    
    print(f"📊 RSI range: {data['RSI'].min():.1f} - {data['RSI'].max():.1f}")
    print(f"📊 RSI < 30 count: {(data['RSI'] < 30).sum()}/{len(data)} days")
    print(f"📊 RSI > 70 count: {(data['RSI'] > 70).sum()}/{len(data)} days")
    print(f"📊 Close > SMA_20 count: {(data['Close'] > data['SMA_20']).sum()}/{len(data)} days")
    
    # Test different strategies
    strategies = [
        {
            'name': 'Single Condition RSI',
            'buy_conditions': ['RSI < 30'],
            'sell_conditions': ['RSI > 70'],
        },
        {
            'name': 'Two Separate Conditions (should work)',
            'buy_conditions': ['RSI < 40', 'Close > SMA_20'],  # These are OR'd together
            'sell_conditions': ['RSI > 60'],
        },
        {
            'name': 'AND Condition (problematic)',
            'buy_conditions': ['RSI < 30 AND Close > SMA_20'],  # This is a single AND condition
            'sell_conditions': ['RSI > 70'],
        }
    ]
    
    for strategy_dict in strategies:
        print(f"\n🎯 Testing: {strategy_dict['name']}")
        print(f"   Buy: {strategy_dict['buy_conditions']}")
        print(f"   Sell: {strategy_dict['sell_conditions']}")
        
        # Add required fields
        strategy_dict.update({
            'description': 'Test strategy',
            'position_sizing': '10%',
            'risk_management': 'None'
        })
        
        strategy = InvestmentStrategy(strategy_dict)
        signals = strategy.generate_signals(data)
        
        buy_signals = signals['buy_signal'].sum()
        sell_signals = signals['sell_signal'].sum()
        
        print(f"   📊 Signals: {buy_signals} buys, {sell_signals} sells")
        
        # Test individual conditions manually
        test_row = data.iloc[100]  # Use a middle row
        print(f"   🧪 Test row RSI: {test_row['RSI']:.1f}, Close: ${test_row['Close']:.2f}, SMA_20: ${test_row['SMA_20']:.2f}")
        
        for i, condition in enumerate(strategy.buy_conditions):
            result = strategy._evaluate_single_condition(test_row, condition)
            print(f"      Buy condition {i+1}: '{condition}' = {result}")

def debug_specific_and_condition():
    """Debug a specific AND condition that should work."""
    print(f"\n🔬 Deep Dive: AND Condition Analysis")
    print("=" * 40)
    
    provider = StockDataProvider()
    data = provider.get_stock_data('AAPL', '1y')
    
    # Find rows where both conditions should be true
    rsi_under_40 = data['RSI'] < 40
    close_above_sma = data['Close'] > data['SMA_20']
    both_true = rsi_under_40 & close_above_sma
    
    print(f"📊 RSI < 40: {rsi_under_40.sum()} days")
    print(f"📊 Close > SMA_20: {close_above_sma.sum()} days")
    print(f"📊 Both true: {both_true.sum()} days")
    
    if both_true.sum() > 0:
        # Get the first few dates where both are true
        matching_dates = data[both_true].head(3)
        print(f"\n📅 First few dates where both conditions are true:")
        for date, row in matching_dates.iterrows():
            print(f"   {date.date()}: RSI={row['RSI']:.1f}, Close=${row['Close']:.2f}, SMA_20=${row['SMA_20']:.2f}")
        
        # Test our condition evaluation on these rows
        strategy_dict = {
            'name': 'AND Test',
            'description': 'Test AND conditions',
            'buy_conditions': ['RSI < 40 AND Close > SMA_20'],
            'sell_conditions': ['RSI > 60'],
            'position_sizing': '10%',
            'risk_management': 'None'
        }
        
        strategy = InvestmentStrategy(strategy_dict)
        
        print(f"\n🧪 Testing our AND condition evaluation:")
        for date, row in matching_dates.iterrows():
            result = strategy._evaluate_single_condition(row, 'RSI < 40 AND Close > SMA_20')
            print(f"   {date.date()}: 'RSI < 40 AND Close > SMA_20' = {result}")

if __name__ == "__main__":
    debug_and_conditions()
    debug_specific_and_condition() 
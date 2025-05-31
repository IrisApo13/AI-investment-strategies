#!/usr/bin/env python3
"""
Detailed debug script to trace signal generation step by step.
"""
import os
os.environ['OPENAI_API_KEY'] = 'test-key-for-debugging'

from data.stock_data import StockDataProvider
from strategy.strategy import InvestmentStrategy
import pandas as pd

def debug_detailed_signals():
    """Debug signal generation step by step."""
    print("🔍 Detailed Signal Generation Debug")
    print("=" * 50)
    
    # Get data
    provider = StockDataProvider()
    data = provider.get_stock_data('AAPL', '1y')
    
    # Create simple strategy
    strategy_dict = {
        'name': 'Debug Strategy',
        'description': 'Simple debug strategy',
        'buy_conditions': ['RSI < 50'],
        'sell_conditions': ['RSI > 50'],
        'position_sizing': '10%',
        'risk_management': 'None'
    }
    
    strategy = InvestmentStrategy(strategy_dict)
    
    # Manual step-through
    print("🔬 Manual Signal Generation:")
    signals = pd.DataFrame(index=data.index)
    signals['buy_signal'] = False
    signals['sell_signal'] = False
    signals['position_size'] = 0.0
    
    position_entries = []
    last_action_date = None
    min_hold_days = 1
    
    buy_count = 0
    sell_count = 0
    
    # Process first 20 rows to debug
    for i, (date, row) in enumerate(data.head(20).iterrows()):
        print(f"\n📅 Date: {date.date()}")
        print(f"  RSI: {row['RSI']:.2f}")
        print(f"  Close: ${row['Close']:.2f}")
        
        # Calculate days since last action
        days_since_last_action = 0
        if last_action_date is not None:
            days_since_last_action = (date - last_action_date).days
        print(f"  Days since last action: {days_since_last_action}")
        print(f"  Current positions: {len(position_entries)}")
        
        # Test buy conditions
        buy_condition_met = strategy._evaluate_conditions(row, strategy.buy_conditions)
        print(f"  Buy condition (RSI < 50): {buy_condition_met}")
        print(f"  Days check: {days_since_last_action >= min_hold_days}")
        print(f"  Position limit check: {len(position_entries) < 3}")
        
        # Check buy
        if (buy_condition_met and
            days_since_last_action >= min_hold_days and
            len(position_entries) < 3):
            
            print("  ✅ BUY SIGNAL GENERATED!")
            signals.loc[date, 'buy_signal'] = True
            signals.loc[date, 'position_size'] = strategy.position_size
            position_entries.append((date, row['Close']))
            last_action_date = date
            buy_count += 1
        
        # Test sell conditions
        elif position_entries and days_since_last_action >= min_hold_days:
            oldest_entry_date, oldest_entry_price = position_entries[0]
            sell_condition_met = strategy._evaluate_conditions(row, strategy.sell_conditions, oldest_entry_price)
            print(f"  Sell condition (RSI > 50): {sell_condition_met}")
            print(f"  Oldest entry price: ${oldest_entry_price:.2f}")
            
            if sell_condition_met:
                print("  ✅ SELL SIGNAL GENERATED!")
                signals.loc[date, 'sell_signal'] = True
                signals.loc[date, 'position_size'] = strategy.position_size
                position_entries.pop(0)
                last_action_date = date
                sell_count += 1
    
    print(f"\n📊 Results after 20 days:")
    print(f"  Buy signals: {buy_count}")
    print(f"  Sell signals: {sell_count}")
    print(f"  Open positions: {len(position_entries)}")
    
    # Test the actual strategy method
    print(f"\n🎯 Testing actual strategy.generate_signals():")
    real_signals = strategy.generate_signals(data)
    print(f"  Buy signals: {real_signals['buy_signal'].sum()}")
    print(f"  Sell signals: {real_signals['sell_signal'].sum()}")

if __name__ == "__main__":
    debug_detailed_signals() 
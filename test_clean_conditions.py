#!/usr/bin/env python3
"""
Test clean condition generation with improved prompts.
"""
from dotenv import load_dotenv
load_dotenv()

from strategy.iterative_improvement import IterativeStrategyImprover

def test_clean_condition_generation():
    """Test that LLM generates clean, executable conditions."""
    print("🧪 Testing Clean Condition Generation")
    print("=" * 45)
    
    # Test with a multi-indicator strategy
    initial_strategy = {
        'name': 'Multi-Indicator Strategy',
        'description': 'Strategy using multiple indicators for testing',
        'buy_conditions': ['RSI < 35', 'Close > SMA_20'],
        'sell_conditions': ['RSI > 65', 'Close < SMA_20'],
        'position_sizing': '15% of portfolio',
        'risk_management': '3% stop loss'
    }
    
    print("📋 Initial Strategy:")
    print(f"  Buy: {initial_strategy['buy_conditions']}")
    print(f"  Sell: {initial_strategy['sell_conditions']}")
    
    improver = IterativeStrategyImprover()
    
    print(f"\n🚀 Running 1 iteration to test condition generation...")
    
    try:
        results = improver.improve_strategy_iteratively(
            initial_strategy=initial_strategy,
            ticker='SPY',
            period='6mo',
            max_iterations=2  # Need 2 iterations to generate improved strategy
        )
        
        if 'iteration_results' in results and len(results['iteration_results']) > 1:
            improved_strategy = results['iteration_results'][1]['strategy']
            
            print(f"\n📊 GENERATED CONDITIONS:")
            print(f"  Strategy Name: {improved_strategy['name']}")
            print(f"  Buy Conditions:")
            for condition in improved_strategy['buy_conditions']:
                print(f"    - '{condition}'")
            print(f"  Sell Conditions:")
            for condition in improved_strategy['sell_conditions']:
                print(f"    - '{condition}'")
            
            # Check if conditions are clean
            all_conditions = improved_strategy['buy_conditions'] + improved_strategy['sell_conditions']
            
            print(f"\n🔍 CONDITION ANALYSIS:")
            clean_conditions = 0
            for condition in all_conditions:
                is_clean = not any(phrase in condition.lower() for phrase in [
                    'to identify', 'indicating', 'for better', 'as this', 'when',
                    'suggesting', 'confirming', 'trend confirmation', 'momentum'
                ])
                status = "✅ CLEAN" if is_clean else "❌ VERBOSE"
                print(f"    {status}: '{condition}'")
                if is_clean:
                    clean_conditions += 1
            
            print(f"\n📈 RESULTS:")
            print(f"  Clean conditions: {clean_conditions}/{len(all_conditions)}")
            print(f"  Success rate: {clean_conditions/len(all_conditions)*100:.1f}%")
            
            if clean_conditions == len(all_conditions):
                print(f"  🎉 All conditions are clean and executable!")
            else:
                print(f"  ⚠️  Some conditions still contain explanatory text")
        
        else:
            print(f"❌ Did not generate improved strategy")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    test_clean_condition_generation() 
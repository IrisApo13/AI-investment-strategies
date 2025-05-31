#!/usr/bin/env python3
"""
Test script for iterative strategy improvement system.
"""
import os
import logging

# Load environment variables (OpenAI API key will be loaded from .env)
from dotenv import load_dotenv
load_dotenv()

from strategy.iterative_improvement import IterativeStrategyImprover
# from llm.strategy_generator import StrategyGenerator  # Not implemented yet

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_iterative_improvement():
    """Test the iterative improvement system."""
    print("🔄 Testing Iterative Strategy Improvement System")
    print("=" * 60)
    
    # Create a simple initial strategy that likely has room for improvement
    initial_strategy = {
        'name': 'Simple RSI Strategy',
        'description': 'Basic RSI-based strategy for improvement testing',
        'buy_conditions': [
            'RSI < 30'  # Very simple condition
        ],
        'sell_conditions': [
            'RSI > 70'  # Very simple condition
        ],
        'position_sizing': '10% of portfolio',
        'risk_management': 'None'
    }
    
    print("📋 Initial Strategy:")
    print(f"  Name: {initial_strategy['name']}")
    print(f"  Buy: {initial_strategy['buy_conditions']}")
    print(f"  Sell: {initial_strategy['sell_conditions']}")
    print(f"  Position Size: {initial_strategy['position_sizing']}")
    
    # Initialize the improver
    improver = IterativeStrategyImprover()
    
    print(f"\n🚀 Starting iterative improvement process...")
    print(f"  Testing on: AAPL")
    print(f"  Period: 2 years")
    print(f"  Max iterations: 3")
    
    # Run iterative improvement
    try:
        improvement_results = improver.improve_strategy_iteratively(
            initial_strategy=initial_strategy,
            ticker='AAPL',
            period='2y',
            max_iterations=3
        )
        
        if 'error' in improvement_results:
            print(f"❌ Error in improvement: {improvement_results['error']}")
            return
        
        # Display results
        print(f"\n📊 IMPROVEMENT RESULTS")
        print("=" * 40)
        
        initial_return = improvement_results['iteration_results'][0]['performance']['total_return']
        best_return = improvement_results['best_performance']['total_return']
        improvement = improvement_results['improvement_achieved']
        
        print(f"Initial Return: {initial_return:.1f}%")
        print(f"Best Return: {best_return:.1f}%")
        print(f"Improvement: {improvement:+.1f}%")
        
        print(f"\n📈 Iteration Performance:")
        for i, result in enumerate(improvement_results['iteration_results']):
            perf = result['performance']
            print(f"  Iteration {i+1}: {perf.get('total_return', 0):.1f}% return, "
                  f"Sharpe {perf.get('sharpe_ratio', 0):.2f}, "
                  f"Max DD {perf.get('max_drawdown', 0):.1f}%")
        
        # Show the best strategy
        best_strategy = improvement_results['best_strategy']
        print(f"\n🏆 BEST STRATEGY FOUND:")
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
        
        # Generate comparison report
        print(f"\n📄 DETAILED COMPARISON REPORT:")
        comparison_report = improver.compare_strategies(improvement_results)
        print(comparison_report)
        
        # Save results
        improver.save_improvement_results(improvement_results)
        print(f"\n💾 Results saved to file")
        
        return improvement_results
        
    except Exception as e:
        print(f"❌ Error during improvement: {str(e)}")
        import traceback
        traceback.print_exc()

def test_with_ai_generated_strategy():
    """Test improvement with an AI-generated initial strategy."""
    print(f"\n" + "="*60)
    print("🤖 Testing with AI-Generated Initial Strategy")
    print("=" * 60)
    
    try:
        # Skip AI generation test for now - StrategyGenerator not implemented
        print("⚠️  Skipping AI-generated strategy test (StrategyGenerator not implemented)")
        return
        
        print(f"\n📋 AI-Generated Initial Strategy:")
        print(f"  Name: {ai_strategy['name']}")
        print(f"  Description: {ai_strategy['description']}")
        print(f"  Buy Conditions: {len(ai_strategy['buy_conditions'])} conditions")
        print(f"  Sell Conditions: {len(ai_strategy['sell_conditions'])} conditions")
        
        # Improve it iteratively
        improver = IterativeStrategyImprover()
        
        print(f"\n🔄 Improving AI-generated strategy...")
        improvement_results = improver.improve_strategy_iteratively(
            initial_strategy=ai_strategy,
            ticker='AAPL',
            period='1y',  # Shorter period for faster testing
            max_iterations=2  # Fewer iterations for faster testing
        )
        
        if 'error' not in improvement_results:
            initial_return = improvement_results['iteration_results'][0]['performance']['total_return']
            best_return = improvement_results['best_performance']['total_return']
            
            print(f"\n📊 AI Strategy Improvement Results:")
            print(f"  Initial AI Strategy Return: {initial_return:.1f}%")
            print(f"  Improved Strategy Return: {best_return:.1f}%")
            print(f"  Improvement: {best_return - initial_return:+.1f}%")
            
            print(f"\n🎉 Iterative improvement successfully enhanced the AI-generated strategy!")
        else:
            print(f"❌ Error improving AI strategy: {improvement_results['error']}")
            
    except Exception as e:
        print(f"❌ Error in AI strategy test: {str(e)}")

def main():
    """Run all improvement tests."""
    print("🧪 ITERATIVE STRATEGY IMPROVEMENT TEST SUITE")
    print("=" * 60)
    
    # Test with simple strategy
    test_iterative_improvement()
    
    # Test with AI-generated strategy
    test_with_ai_generated_strategy()
    
    print(f"\n✅ All tests completed!")

if __name__ == "__main__":
    main() 
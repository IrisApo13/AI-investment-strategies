#!/usr/bin/env python3
"""
Debug script to test the full application flow with AAPL.
"""
import os

# Check for API key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY or OPENAI_API_KEY == 'test-key':
    print("❌ OPENAI_API_KEY not set or using test key")
    print("For full testing, set a real API key:")
    print("export OPENAI_API_KEY='your_actual_key_here'")
    print()
    print("🔄 Continuing with test strategy (no AI generation)...")
    USE_AI = False
else:
    print(f"✅ OPENAI_API_KEY is set (length: {len(OPENAI_API_KEY)})")
    USE_AI = True

# Load environment variables (OpenAI API key will be loaded from .env)
from dotenv import load_dotenv
load_dotenv()

from data.stock_data import StockDataProvider
from strategy.strategy import InvestmentStrategy
from backtesting.simulator import PortfolioSimulator
from backtesting.evaluator import PerformanceEvaluator

if USE_AI:
    from llm.client import LLMClient
    from llm.prompts import PromptGenerator
    from llm.parser import StrategyParser

def debug_full_flow():
    """Debug the complete application flow."""
    print("🔍 Debugging Full Application Flow")
    print("=" * 50)
    
    ticker = "AAPL"
    
    # Step 1: Data retrieval
    print("📊 Step 1: Data Retrieval")
    provider = StockDataProvider()
    data = provider.get_stock_data(ticker, '1y')
    stock_info = provider.get_stock_info(ticker)
    
    print(f"  ✅ Retrieved {len(data)} days of data for {stock_info['name']}")
    print(f"  📈 Price range: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")
    print(f"  📊 RSI range: {data['RSI'].min():.1f} - {data['RSI'].max():.1f}")
    
    # Step 2: Strategy generation
    print(f"\n🎯 Step 2: Strategy Generation")
    
    if USE_AI:
        try:
            print("  🤖 Using AI to generate strategy...")
            
            # Initialize AI components
            llm_client = LLMClient()
            prompt_generator = PromptGenerator()
            strategy_parser = StrategyParser()
            
            # Create market summary
            market_summary = prompt_generator.create_data_summary(data, ticker)
            print(f"  📈 Market trend: {market_summary.get('trend', 'Unknown')}")
            print(f"  📊 Current RSI: {market_summary.get('current_rsi', 0):.1f}")
            
            # Generate strategy
            prompt = prompt_generator.create_initial_strategy_prompt(ticker, market_summary, stock_info)
            llm_response = llm_client.generate_strategy(prompt)
            strategy_dict = strategy_parser.parse_strategy_response(llm_response)
            
            if strategy_dict:
                print(f"  ✅ AI generated strategy: {strategy_dict['name']}")
                print(f"  📋 Buy conditions: {strategy_dict['buy_conditions']}")
                print(f"  📋 Sell conditions: {strategy_dict['sell_conditions']}")
            else:
                print(f"  ❌ Failed to parse AI strategy, using fallback")
                strategy_dict = get_fallback_strategy()
                
        except Exception as e:
            print(f"  ❌ AI strategy generation failed: {str(e)}")
            print(f"  🔄 Using fallback strategy...")
            strategy_dict = get_fallback_strategy()
    else:
        print("  🔄 Using predefined strategy (no AI)...")
        strategy_dict = get_fallback_strategy()
    
    # Step 3: Signal generation
    print(f"\n📊 Step 3: Signal Generation")
    strategy = InvestmentStrategy(strategy_dict)
    signals = strategy.generate_signals(data)
    
    buy_signals = signals['buy_signal'].sum()
    sell_signals = signals['sell_signal'].sum()
    
    print(f"  📈 Buy signals: {buy_signals}")
    print(f"  📉 Sell signals: {sell_signals}")
    
    if buy_signals == 0 and sell_signals == 0:
        print(f"  ❌ ZERO SIGNALS DETECTED!")
        print(f"  🔍 Analyzing strategy conditions...")
        
        # Debug the conditions
        print(f"  📋 Buy conditions: {strategy.buy_conditions}")
        print(f"  📋 Sell conditions: {strategy.sell_conditions}")
        
        # Test each condition manually
        test_row = data.iloc[-1]
        print(f"  🧪 Testing conditions on last row (RSI={test_row['RSI']:.1f}):")
        
        for i, condition in enumerate(strategy.buy_conditions):
            result = strategy._evaluate_single_condition(test_row, condition)
            print(f"    Buy condition {i+1}: '{condition}' = {result}")
        
        for i, condition in enumerate(strategy.sell_conditions):
            result = strategy._evaluate_single_condition(test_row, condition)
            print(f"    Sell condition {i+1}: '{condition}' = {result}")
        
        # Check how many rows meet individual conditions
        print(f"  📊 Condition frequency analysis:")
        for condition in strategy.buy_conditions:
            count = sum(strategy._evaluate_single_condition(row, condition) for _, row in data.iterrows())
            print(f"    '{condition}' true on {count}/{len(data)} days ({count/len(data)*100:.1f}%)")
    
    # Step 4: Backtesting (if we have signals)
    if buy_signals > 0 or sell_signals > 0:
        print(f"\n💰 Step 4: Backtesting")
        simulator = PortfolioSimulator()
        performance = simulator.run_backtest(data, signals, ticker)
        
        evaluator = PerformanceEvaluator()
        evaluation = evaluator.evaluate_performance(performance)
        
        print(f"  📊 Total return: {performance['total_return']:.2f}%")
        print(f"  📊 Buy & hold: {performance['buy_hold_return']:.2f}%")
        print(f"  📊 Excess return: {performance['excess_return']:.2f}%")
        print(f"  📊 Performance score: {evaluation['performance_score']:.1f}/100")
        print(f"  📊 Number of actual trades: {performance['num_trades']}")
    else:
        print(f"\n❌ Step 4: Skipping backtesting (no signals)")
    
    # Summary
    print(f"\n📋 SUMMARY")
    print("=" * 30)
    print(f"Strategy: {strategy_dict['name']}")
    print(f"Signals: {buy_signals} buys, {sell_signals} sells")
    print(f"Status: {'✅ WORKING' if (buy_signals + sell_signals) > 0 else '❌ ZERO TRADES'}")

def get_fallback_strategy():
    """Get a fallback strategy that should generate signals."""
    return {
        'name': 'Fallback RSI Strategy',
        'description': 'Conservative RSI mean reversion',
        'buy_conditions': ['RSI < 35'],
        'sell_conditions': ['RSI > 65'],
        'position_sizing': '20% of portfolio',
        'risk_management': '5% stop loss'
    }

if __name__ == "__main__":
    debug_full_flow() 
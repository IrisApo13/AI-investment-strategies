#!/usr/bin/env python3
"""
Debug script to test AI strategy generation.
"""
import os

# Set your actual OpenAI API key here for this test
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if OPENAI_API_KEY and OPENAI_API_KEY != 'test-key':
    os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
else:
    print("❌ Please set your OPENAI_API_KEY environment variable to test AI strategy generation")
    print("For manual testing, you can temporarily set it:")
    print("export OPENAI_API_KEY='your_actual_key_here'")
    exit(1)

from data.stock_data import StockDataProvider
from llm.client import LLMClient
from llm.prompts import PromptGenerator
from llm.parser import StrategyParser
from strategy.strategy import InvestmentStrategy

def test_ai_strategy_generation():
    """Test the full AI strategy generation pipeline."""
    print("🤖 Testing AI Strategy Generation")
    print("=" * 50)
    
    try:
        # Initialize components
        data_provider = StockDataProvider()
        llm_client = LLMClient()
        prompt_generator = PromptGenerator()
        strategy_parser = StrategyParser()
        
        # Get data for AAPL
        ticker = "AAPL"
        data = data_provider.get_stock_data(ticker, '1y')
        stock_info = data_provider.get_stock_info(ticker)
        
        print(f"📊 Retrieved data for {ticker}")
        print(f"  Shape: {data.shape}")
        print(f"  Company: {stock_info.get('name', 'Unknown')}")
        
        # Create market summary
        market_summary = prompt_generator.create_data_summary(data, ticker)
        print(f"\n📈 Market Summary:")
        print(f"  Total days: {market_summary.get('total_days', 0)}")
        print(f"  Trend: {market_summary.get('trend', 'Unknown')}")
        print(f"  Current RSI: {market_summary.get('current_rsi', 0):.1f}")
        
        # Generate strategy prompt
        prompt = prompt_generator.create_initial_strategy_prompt(ticker, market_summary, stock_info)
        print(f"\n📝 Generated prompt (first 200 chars):")
        print(f"  {prompt[:200]}...")
        
        # Generate strategy using AI
        print(f"\n🤖 Calling OpenAI API...")
        llm_response = llm_client.generate_strategy(prompt)
        print(f"✓ Received response ({len(llm_response)} characters)")
        
        # Parse strategy
        strategy_dict = strategy_parser.parse_strategy_response(llm_response)
        
        if strategy_dict:
            print(f"\n✅ Successfully parsed strategy:")
            print(f"  Name: {strategy_dict.get('name', 'Unknown')}")
            print(f"  Description: {strategy_dict.get('description', 'None')}")
            print(f"  Buy conditions: {strategy_dict.get('buy_conditions', [])}")
            print(f"  Sell conditions: {strategy_dict.get('sell_conditions', [])}")
            
            # Test strategy
            strategy = InvestmentStrategy(strategy_dict)
            signals = strategy.generate_signals(data)
            
            print(f"\n📊 Strategy Results:")
            print(f"  Buy signals: {signals['buy_signal'].sum()}")
            print(f"  Sell signals: {signals['sell_signal'].sum()}")
            
            # Show first few signals
            buy_dates = signals[signals['buy_signal']].index[:3]
            sell_dates = signals[signals['sell_signal']].index[:3]
            
            print(f"\n📅 First Buy Signals:")
            for date in buy_dates:
                rsi = data.loc[date, 'RSI']
                price = data.loc[date, 'Close']
                print(f"  {date.date()}: Price=${price:.2f}, RSI={rsi:.1f}")
            
            print(f"\n📅 First Sell Signals:")
            for date in sell_dates:
                rsi = data.loc[date, 'RSI']
                price = data.loc[date, 'Close']
                print(f"  {date.date()}: Price=${price:.2f}, RSI={rsi:.1f}")
                
        else:
            print(f"❌ Failed to parse strategy from AI response")
            print(f"Raw response: {llm_response[:500]}...")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ai_strategy_generation() 
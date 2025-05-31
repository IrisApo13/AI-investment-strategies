#!/usr/bin/env python3
"""
Test script to demonstrate LLM prompt and response display functionality.
"""
import os
import logging

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from llm.client import LLMClient
from llm.prompts import PromptGenerator
from data.stock_data import StockDataProvider

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_prompt_display():
    """Test the new prompt and response display functionality."""
    print("🧪 Testing LLM Prompt and Response Display")
    print("=" * 50)
    
    # Initialize components
    llm_client = LLMClient()
    prompt_generator = PromptGenerator()
    data_provider = StockDataProvider()
    
    print(f"📊 Fetching sample data for AAPL...")
    
    try:
        # Get some sample market data
        market_data = data_provider.get_stock_data('AAPL', '6mo')
        
        # Create market data summary
        market_summary = prompt_generator.create_data_summary(market_data, 'AAPL')
        
        # Create stock info (simplified)
        stock_info = {
            'name': 'Apple Inc.',
            'sector': 'Technology',
            'industry': 'Consumer Electronics'
        }
        
        # Generate initial strategy prompt
        prompt = prompt_generator.create_initial_strategy_prompt(
            'AAPL', market_summary, stock_info
        )
        
        print(f"\n🚀 Generating strategy with prompt/response display enabled...")
        print(f"   Note: You will see the full prompt sent to the LLM and its response")
        
        # Generate strategy (this will display prompt and response)
        response = llm_client.generate_strategy(prompt)
        
        print(f"\n✅ Strategy generation completed!")
        print(f"   The LLM prompt and response were displayed above.")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during test: {str(e)}")
        return False

def test_different_operation_types():
    """Test different operation types for better labeling."""
    print(f"\n🔬 Testing Different Operation Types")
    print("=" * 40)
    
    llm_client = LLMClient()
    
    # Test with different operation types
    test_prompts = [
        ("Initial strategy for SPY", "INITIAL STRATEGY GENERATION"),
        ("Analyze strategy performance", "PERFORMANCE ANALYSIS"),
        ("Improve existing strategy", "STRATEGY IMPROVEMENT")
    ]
    
    for prompt_text, operation_type in test_prompts:
        print(f"\n📝 Testing operation type: {operation_type}")
        
        try:
            # This will show different operation type labels
            response = llm_client.generate_strategy(
                f"Simple test prompt: {prompt_text}",
                operation_type=operation_type
            )
            print(f"✅ {operation_type} completed")
            
        except Exception as e:
            print(f"❌ Error in {operation_type}: {str(e)}")

def main():
    """Run the prompt display tests."""
    print("🧪 LLM PROMPT AND RESPONSE DISPLAY TEST SUITE")
    print("=" * 60)
    
    print(f"\nThis test will demonstrate the new functionality that shows:")
    print(f"  🤖 Prompts sent to the LLM")
    print(f"  🧠 Responses received from the LLM")
    print(f"  📋 Different operation types (Strategy Generation, Analysis, etc.)")
    
    # Test basic functionality
    success = test_prompt_display()
    
    if success:
        # Test different operation types
        test_different_operation_types()
        
        print(f"\n🎉 All tests completed successfully!")
        print(f"\nTo disable prompt/response display, set in config/settings.py:")
        print(f"  SHOW_LLM_PROMPTS = False")
        print(f"  SHOW_LLM_RESPONSES = False")
    else:
        print(f"\n❌ Tests failed. Check your OpenAI API key and connection.")

if __name__ == "__main__":
    main() 
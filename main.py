"""
Main CLI application for the AI Investment Strategy Backtesting System.
"""
import argparse
import logging
import os
import sys
from dotenv import load_dotenv

from config.settings import Config
from core.orchestrator import AIBacktestOrchestrator

# Load environment variables
load_dotenv()

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, Config.LOG_LEVEL),
        format=Config.LOG_FORMAT,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('ai_backtest.log')
        ]
    )

def print_banner():
    """Print application banner."""
    banner = """
╔══════════════════════════════════════════════════════════════════╗
║                 AI Investment Strategy Backtesting               ║
║                           System v1.0                           ║
╚══════════════════════════════════════════════════════════════════╝

Generate, test, and iteratively improve investment strategies using AI.
"""
    print(banner)

def validate_environment():
    """Validate environment and configuration."""
    try:
        Config.validate_config()
        print("✓ Configuration validated successfully")
        return True
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        print("\nPlease ensure you have:")
        print("1. Set OPENAI_API_KEY in your environment or .env file")
        print("2. Valid configuration settings")
        return False

def interactive_mode():
    """Run the application in interactive mode."""
    print("\n" + "="*60)
    print("INTERACTIVE MODE")
    print("="*60)
    
    # Get ticker from user
    while True:
        ticker = input("\nEnter stock ticker symbol (e.g., AAPL, TSLA): ").strip().upper()
        if ticker:
            break
        print("Please enter a valid ticker symbol.")
    
    # Get optional parameters
    print(f"\nOptional parameters (press Enter for defaults):")
    
    max_iterations_input = input(f"Maximum iterations (default: {Config.MAX_ITERATIONS}): ").strip()
    max_iterations = int(max_iterations_input) if max_iterations_input else Config.MAX_ITERATIONS
    
    target_score_input = input("Target performance score (default: 80.0): ").strip()
    target_score = float(target_score_input) if target_score_input else 80.0
    
    # Confirm settings
    print(f"\n" + "-"*40)
    print("SESSION CONFIGURATION:")
    print(f"Ticker: {ticker}")
    print(f"Max Iterations: {max_iterations}")
    print(f"Target Score: {target_score}")
    print(f"Initial Capital: ${Config.INITIAL_CAPITAL:,.2f}")
    print("-"*40)
    
    confirm = input("\nProceed with these settings? (y/n): ").strip().lower()
    if confirm not in ['y', 'yes']:
        print("Session cancelled.")
        return
    
    # Run backtesting session
    run_backtesting_session(ticker, max_iterations, target_score)

def run_backtesting_session(ticker: str, max_iterations: int, target_score: float):
    """Run a backtesting session."""
    try:
        print(f"\n🚀 Starting AI backtesting session for {ticker}...")
        
        # Initialize orchestrator
        orchestrator = AIBacktestOrchestrator()
        
        # Run session
        results = orchestrator.run_backtesting_session(
            ticker=ticker,
            max_iterations=max_iterations,
            target_score=target_score
        )
        
        # Handle results
        if results['success']:
            print("\n" + "="*60)
            print("SESSION COMPLETED SUCCESSFULLY!")
            print("="*60)
            
            # Display summary
            print(results['session_summary'])
            
            # Offer to save results
            save_option = input("\nSave session results to file? (y/n): ").strip().lower()
            if save_option in ['y', 'yes']:
                if orchestrator.save_session_results(results):
                    print("✓ Results saved successfully!")
                else:
                    print("❌ Failed to save results")
        
        else:
            print(f"\n❌ Session failed: {results.get('error', 'Unknown error')}")
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Session interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        logging.error(f"Unexpected error in session: {str(e)}", exc_info=True)

def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(
        description="AI Investment Strategy Backtesting System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Interactive mode
  python main.py AAPL                     # Quick run with AAPL
  python main.py TSLA --iterations 5      # Run TSLA with 5 iterations
  python main.py MSFT --target-score 75   # Run MSFT targeting 75% score
        """
    )
    
    parser.add_argument(
        'ticker',
        nargs='?',
        help='Stock ticker symbol (e.g., AAPL, TSLA)'
    )
    
    parser.add_argument(
        '--iterations',
        type=int,
        default=Config.MAX_ITERATIONS,
        help=f'Maximum number of iterations (default: {Config.MAX_ITERATIONS})'
    )
    
    parser.add_argument(
        '--target-score',
        type=float,
        default=80.0,
        help='Target performance score to achieve (default: 80.0)'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default=Config.LOG_LEVEL,
        help=f'Logging level (default: {Config.LOG_LEVEL})'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='AI Backtesting System v1.0'
    )
    
    args = parser.parse_args()
    
    # Override log level if specified
    if args.log_level != Config.LOG_LEVEL:
        Config.LOG_LEVEL = args.log_level
    
    # Setup logging
    setup_logging()
    
    # Print banner
    print_banner()
    
    # Validate environment
    if not validate_environment():
        sys.exit(1)
    
    # Run in appropriate mode
    if args.ticker:
        # Command line mode
        print(f"\n🎯 Running backtesting session for {args.ticker}")
        run_backtesting_session(args.ticker, args.iterations, args.target_score)
    else:
        # Interactive mode
        interactive_mode()

if __name__ == "__main__":
    main() 
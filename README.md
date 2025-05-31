# AI Investment Strategy Backtesting System

An intelligent backtesting system that uses OpenAI's GPT-4 to generate, test, and iteratively improve investment strategies based on technical analysis.

## 🚀 Features

- **AI-Powered Strategy Generation**: Uses GPT-4 to create investment strategies based on market data analysis
- **Comprehensive Backtesting**: Realistic portfolio simulation with transaction costs and slippage
- **Iterative Improvement**: AI learns from performance feedback to generate better strategies
- **Technical Analysis**: 15+ technical indicators including RSI, MACD, Bollinger Bands, and more
- **Performance Evaluation**: Comprehensive metrics including Sharpe ratio, drawdown, win rate, and custom scoring
- **User-Friendly Interface**: CLI application with interactive and command-line modes
- **Session Management**: Save and load backtesting sessions for later analysis

## 📋 Requirements

- Python 3.8+
- OpenAI API key
- Internet connection for stock data retrieval

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd backtestai
   ```

2. **Create and activate virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   ```bash
   # Create .env file
   echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
   ```

## 🏃‍♂️ Quick Start

### Interactive Mode
```bash
python main.py
```

### Command Line Mode
```bash
# Analyze AAPL with default settings
python main.py AAPL

# Custom parameters
python main.py TSLA --iterations 5 --target-score 75
```

### Test the System
```bash
python test_system.py
```

## 📊 Example Usage

```python
from core.orchestrator import AIBacktestOrchestrator

# Initialize the system
orchestrator = AIBacktestOrchestrator()

# Run backtesting session
results = orchestrator.run_backtesting_session(
    ticker="AAPL",
    max_iterations=3,
    target_score=80.0
)

# View results
if results['success']:
    print(results['session_summary'])
    best_strategy = results['best_strategy']
    best_performance = results['best_performance']
```

## 🏗️ System Architecture

```
backtestai/
├── config/           # Configuration settings
├── data/            # Stock data retrieval and technical analysis
├── llm/             # OpenAI integration and prompt engineering
├── strategy/        # Investment strategy representation
├── backtesting/     # Portfolio simulation and evaluation
├── core/            # Main orchestrator
├── utils/           # Utility functions
├── main.py          # CLI application
└── test_system.py   # Test suite
```

### Key Components

1. **Data Provider** (`data/`): Fetches stock data using yfinance and calculates technical indicators
2. **LLM Integration** (`llm/`): Handles OpenAI API communication, prompt engineering, and response parsing
3. **Strategy Engine** (`strategy/`): Represents investment strategies and generates trading signals
4. **Backtesting Engine** (`backtesting/`): Simulates portfolio performance with realistic costs
5. **Orchestrator** (`core/`): Coordinates the entire iterative improvement process

## 📈 Technical Indicators

The system uses 15+ technical indicators for strategy generation:

- **Moving Averages**: SMA_20, SMA_50, EMA_12, EMA_26
- **Momentum**: RSI, MACD, MACD_signal, momentum
- **Volatility**: Bollinger Bands (upper, middle, lower), ATR, volatility
- **Volume**: Volume SMA, volume ratio
- **Price**: Price change, price position

## 🎯 Performance Metrics

- **Returns**: Total return, annualized return, excess return over buy & hold
- **Risk**: Volatility, maximum drawdown, Sharpe ratio, Sortino ratio
- **Trading**: Number of trades, win rate, profit factor, time in market
- **Scoring**: Custom performance score (0-100) with weighted metrics

## 🔧 Configuration

Key configuration options in `config/settings.py`:

```python
# Backtesting
INITIAL_CAPITAL = 100000.0    # Starting capital
TRANSACTION_COST = 0.001      # 0.1% per trade
SLIPPAGE = 0.0005            # 0.05% slippage

# Strategy
MAX_POSITION_SIZE = 0.3       # Max 30% per position
MIN_POSITION_SIZE = 0.05      # Min 5% per position

# AI
OPENAI_MODEL = "gpt-4-turbo-preview"
MAX_TOKENS = 2000
TEMPERATURE = 0.7
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
python test_system.py
```

Tests cover:
- Strategy creation and signal generation
- Portfolio simulation accuracy
- Performance evaluation metrics
- LLM response parsing
- System integration

## 📝 Example Output

```
============================================================
ITERATION 1 RESULTS
============================================================

Strategy: RSI Mean Reversion
Description: Buy oversold conditions, sell overbought

Key Performance Metrics:
- Performance Score: 67.3/100
- Total Return: 23.45%
- Buy & Hold Return: 18.20%
- Sharpe Ratio: 0.842
- Max Drawdown: -12.30%
- Win Rate: 58.3%
- Number of Trades: 24

Buy Conditions:
  1. RSI < 30
  2. Close > SMA_20
  3. Volume > volume_sma * 1.2

Sell Conditions:
  1. RSI > 70
  2. Close < SMA_20 * 0.98
```

## 🚦 Common Issues

### API Key Issues
- Ensure OPENAI_API_KEY is set in environment or .env file
- Verify API key has sufficient credits

### Data Issues
- Check internet connection for stock data retrieval
- Verify ticker symbol is valid and has sufficient history

### Performance Issues
- Reduce max_iterations for faster testing
- Use shorter data periods for development

## 🔮 Future Enhancements

- [ ] Multiple asset portfolio strategies
- [ ] Options and derivatives support
- [ ] Real-time trading integration
- [ ] Web-based dashboard
- [ ] Advanced machine learning models
- [ ] Risk management optimization
- [ ] Multi-timeframe analysis

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## ⚠️ Disclaimer

This software is for educational and research purposes only. It is not financial advice. Trading involves risk and you should only trade with money you can afford to lose. Always do your own research before making investment decisions.

## 📞 Support

- Create an issue for bug reports or feature requests
- Check the test suite for usage examples
- Review the code documentation for technical details 
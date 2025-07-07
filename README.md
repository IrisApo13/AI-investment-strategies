# AI Investment Strategies

An AI-powered system for generating, backtesting, and iteratively improving investment strategies using Large Language Models (LLMs).

## Features

- **AI Strategy Generation**: Generate investment strategies using GPT-4
- **Comprehensive Backtesting**: Test strategies with realistic market conditions
- **Iterative Improvement**: Continuously improve strategies based on performance feedback
- **Advanced Trade Analysis**: Analyze best and worst trades for enhanced feedback
- **Performance Evaluation**: Comprehensive metrics and risk analysis
- **Visualization**: Strategy performance charts and analysis

## Configuration-Based Feedback Strategy

The system now supports different feedback strategies for iterative improvement, controlled by the `FEEDBACK_STRATEGY` configuration setting.

### Available Feedback Strategies

1. **`basic_feedback`** (default): Standard performance-based feedback
   - Uses performance metrics and analysis
   - Provides general improvement suggestions
   - Standard LLM prompts

2. **`advanced_feedback`**: Enhanced feedback with trade analysis
   - Analyzes the two best and two worst trades
   - Provides detailed market condition insights
   - Enhanced LLM prompts with trade data
   - More specific improvement recommendations

## Market Data Configuration Mode

The system supports configurable market data presentation to the LLM through the `MARKET_DATA_MODE` setting.

### Available Market Data Modes

1. **`summary`**: Summarized market data analysis
   - Pre-analyzed market conditions and recommendations
   - Technical indicator analysis and market regime classification
   - More concise prompts (~4,000 characters)
   - Faster processing and lower token usage

2. **`raw_data`** (default): Raw daily market data
   - Last 50 days of prices and technical indicators
   - Enables pattern recognition from actual data
   - More detailed prompts (~11,000 characters)
   - Data-driven threshold selection

### How to Use Market Data Modes

1. **Configure the Mode**:
   ```python
   # In config/settings.py
   MARKET_DATA_MODE = "summary"  # or "raw_data"
   ```

2. **The system will automatically**:
   - Use the configured mode for both initial strategy generation and iterative improvement
   - Provide appropriate market data format to the LLM
   - Maintain consistency across all strategy generation processes

### Benefits of Each Mode

**Summary Mode**:
- Faster strategy generation
- Lower API token usage
- Clear strategy recommendations
- Pre-analyzed market insights

**Raw Data Mode**:
- Pattern recognition from actual data
- Data-driven threshold selection
- Context awareness of recent market behavior
- Flexible analysis capabilities

### How to Use Advanced Feedback

1. **Enable Advanced Feedback**:
   ```python
   # In config/settings.py
   FEEDBACK_STRATEGY = "advanced_feedback"
   ```

2. **Run Your Backtesting Session**:
   ```python
   from core.orchestrator import AIBacktestOrchestrator
   
   orchestrator = AIBacktestOrchestrator()
   results = orchestrator.run_backtesting_session(
       ticker="SPY",
       max_iterations=3,
       target_score=80.0
   )
   ```

3. **The system will automatically**:
   - Use enhanced feedback with trade analysis
   - Provide more specific improvement suggestions
   - Include insights from best/worst trades

### Benefits of Advanced Feedback

- **More Specific Improvements**: Based on actual trade performance data
- **Data-Driven Insights**: Learn from what actually worked and didn't work
- **Better Entry/Exit Conditions**: Understand market conditions that led to success
- **Improved Strategy Evolution**: More targeted strategy improvements

### Example Trade Analysis Output

When using advanced feedback, the system provides detailed analysis like:

```
🏆 BEST PERFORMING TRADES:
Best Trade #1:
  Return: 15.2%
  Duration: 45 days
  Entry Conditions:
    - RSI: 28.5
    - Close: $150.25
    - Volume: 2,500,000
  Exit Conditions:
    - RSI: 72.1
    - Close: $173.15

📉 WORST PERFORMING TRADES:
Worst Trade #1:
  Return: -8.3%
  Duration: 12 days
  Entry Conditions:
    - RSI: 45.2
    - Close: $148.75
    - Volume: 1,800,000
```

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd AI-investment-strategies
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```bash
   cp env_example.txt .env
   # Edit .env with your OpenAI API key
   ```

## Usage

### Basic Usage

```python
from core.orchestrator import AIBacktestOrchestrator

# Initialize orchestrator
orchestrator = AIBacktestOrchestrator()

# Run backtesting session
results = orchestrator.run_backtesting_session(
    ticker="SPY",
    max_iterations=3,
    target_score=80.0
)
```

### Using Enhanced Feedback

```python
from config.settings import Config

# Enable advanced feedback
Config.FEEDBACK_STATEGY = "advanced_feedback"

# Run session with enhanced feedback
results = orchestrator.run_backtesting_session(
    ticker="AAPL",
    max_iterations=5,
    target_score=85.0
)
```

### Testing Different Feedback Strategies

```python
# Test configuration options
python3 tests/test_config_feedback_simple.py

# Test enhanced improvement with trade analysis
python3 tests/test_enhanced_improvement.py
```

### Testing Market Data Modes

```python
# Test both market data modes
python3 tests/test_market_data_config.py

# Test mode switching
python3 tests/test_switch_modes.py
```

## Configuration

Key configuration options in `config/settings.py`:

- `FEEDBACK_STRATEGY`: Choose between "basic_feedback" and "advanced_feedback"
- `MARKET_DATA_MODE`: Choose between "summary" and "raw_data" for LLM market data presentation
- `OPENAI_MODEL`: LLM model to use for strategy generation
- `MAX_ITERATIONS`: Maximum improvement iterations
- `INITIAL_CAPITAL`: Starting portfolio value
- `TRANSACTION_COST`: Trading fees percentage

## Project Structure

```
AI-investment-strategies/
├── analysis/              # Performance analysis
├── backtesting/           # Backtesting engine
├── config/               # Configuration settings
├── core/                 # Main orchestrator
├── data/                 # Data providers
├── llm/                  # LLM integration
├── strategy/             # Strategy classes
├── tests/                # Test scripts
└── utils/                # Utility functions
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
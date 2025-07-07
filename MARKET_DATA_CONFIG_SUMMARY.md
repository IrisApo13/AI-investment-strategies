# Market Data Configuration Mode Summary

## Overview

The system now supports configurable market data presentation to the LLM through the `MARKET_DATA_MODE` setting in `config/settings.py`. This allows you to choose between providing summarized market data insights or raw daily market data to the LLM when generating investment strategies.

## Configuration Options

### `MARKET_DATA_MODE = "summary"`
- Provides the LLM with **summarized market data analysis**
- Includes strategy recommendations based on market conditions
- Contains technical indicator analysis and market regime classification
- More concise prompts (~4,000 characters)
- Focuses on actionable insights rather than raw data

### `MARKET_DATA_MODE = "raw_data"`
- Provides the LLM with **raw daily market data table**
- Includes the last 50 days of prices and technical indicators
- Contains summary statistics for context
- More detailed prompts (~11,000 characters)
- Enables pattern recognition from actual data

## Implementation Details

### Configuration Setting
```python
# In config/settings.py
MARKET_DATA_MODE = "raw_data"  # or "summary"
```

### Affected Components

1. **Prompt Generation** (`llm/prompts.py`)
   - `create_initial_strategy_prompt()` method
   - Conditionally formats either raw data or summary based on configuration

2. **Iterative Improvement** (`strategy/iterative_improvement.py`)
   - `_generate_improved_strategy()` method
   - Uses the same configuration to provide appropriate market data in feedback

3. **Orchestrator** (`core/orchestrator.py`)
   - Passes market data to prompt generation
   - Works with both modes seamlessly

## Benefits of Each Mode

### Summary Mode Benefits
- **Faster processing**: Smaller prompts reduce LLM processing time
- **Focused insights**: Pre-analyzed market conditions and recommendations
- **Lower token usage**: More cost-effective for API calls
- **Clear guidance**: Strategy recommendations guide the LLM

### Raw Data Mode Benefits
- **Pattern recognition**: LLM can identify patterns in actual price movements
- **Data-driven thresholds**: Can set specific levels based on observed data ranges
- **Context awareness**: Understands recent market behavior and trends
- **Flexible analysis**: Can adapt to changing market conditions

## Usage Examples

### Switching Between Modes
```python
# Switch to summary mode
Config.MARKET_DATA_MODE = "summary"

# Switch to raw data mode  
Config.MARKET_DATA_MODE = "raw_data"
```

### Testing Both Modes
```bash
# Test summary mode
python tests/test_market_data_config.py

# Test mode switching
python tests/test_switch_modes.py
```

## Performance Comparison

Based on testing with SPY data:

| Mode | Prompt Length | Key Features |
|------|---------------|--------------|
| Summary | ~4,140 chars | Strategy recommendations, market analysis |
| Raw Data | ~10,990 chars | Daily data table, summary statistics |

## Recommendations

### Use Summary Mode When:
- You want faster strategy generation
- You prefer pre-analyzed market insights
- You're working with limited API tokens
- You want clear strategy recommendations

### Use Raw Data Mode When:
- You want the LLM to identify specific patterns
- You need data-driven threshold selection
- You're optimizing for strategy performance
- You want the LLM to adapt to current market conditions

## Integration with Existing Features

The market data mode works seamlessly with:
- **Advanced feedback**: Both modes are supported in iterative improvement
- **Strategy generation**: Initial and improved strategies use the configured mode
- **Backtesting**: No impact on strategy execution or evaluation
- **Configuration system**: Easy to switch between modes without code changes

## Future Enhancements

Potential improvements could include:
- **Hybrid mode**: Combine summary insights with selective raw data
- **Dynamic selection**: Automatically choose mode based on market conditions
- **Custom ranges**: Allow configuration of how many days of raw data to include
- **Performance tracking**: Compare strategy performance between modes 
# Troubleshooting Guide

## 🚨 Common Issues and Solutions

### 1. OpenAI API Issues

**Error:** `openai.error.AuthenticationError` or similar API errors

**Solutions:**
- Ensure your OpenAI API key is set correctly:
  ```bash
  export OPENAI_API_KEY="your_key_here"
  # or create a .env file with:
  echo "OPENAI_API_KEY=your_key_here" > .env
  ```
- Verify your API key has sufficient credits
- Check if you have access to GPT-4 models

### 2. Import Errors

**Error:** `ModuleNotFoundError` or import issues

**Solutions:**
- Install all dependencies:
  ```bash
  pip install -r requirements.txt
  ```
- Make sure you're in the correct virtual environment:
  ```bash
  source venv/bin/activate  # On Windows: venv\Scripts\activate
  ```

### 3. Data Retrieval Issues

**Error:** "No data found for ticker" or network errors

**Solutions:**
- Check your internet connection
- Verify the ticker symbol is valid and traded
- Try a different ticker (e.g., AAPL, MSFT, GOOGL)
- Wait a few minutes and try again (API rate limits)

### 4. Strategy Generation Issues

**Error:** No signals generated (0 buy/sell signals)

**Possible causes:**
- Strategy conditions are too restrictive
- Data doesn't meet the strategy criteria
- Technical indicators have insufficient data (need >50 days for some indicators)

**Solutions:**
- Try with a different ticker
- Use longer data periods
- The AI will typically improve strategies in subsequent iterations

### 5. Performance Issues

**Problem:** System is slow or times out

**Solutions:**
- Reduce `max_iterations` for testing
- Use shorter data periods during development
- Check your internet connection
- Verify OpenAI API is responding

## 🔧 Quick Diagnostic Commands

### Test the System
```bash
python test_system.py
```

### Check Environment
```bash
python -c "
import os
from config.settings import Config
print('OpenAI API Key:', 'SET' if Config.OPENAI_API_KEY else 'NOT SET')
print('All imports successful!')
"
```

### Test Data Provider
```bash
python -c "
from data.stock_data import StockDataProvider
provider = StockDataProvider()
print('Ticker AAPL valid:', provider.validate_ticker('AAPL'))
"
```

### Minimal Test Run
```bash
# Set test API key for basic functionality
export OPENAI_API_KEY="test-key"
python main.py --help
```

## 📊 Expected Behavior

### Normal Test Output
- All 4 tests should pass
- You may see 0 signals generated (this is normal for test data)
- Performance score may be low (this is expected without real trading)

### Normal Main Program Flow
1. Banner displays ✓
2. Configuration validation ✓  
3. User prompted for ticker ✓
4. Data retrieval starts ✓
5. Strategy generation begins ✓

## 🆘 Getting Help

If you're still experiencing issues:

1. **Copy the exact error message**
2. **Note what you were doing when the error occurred**
3. **Check which Python version you're using:** `python --version`
4. **Verify your environment:** `pip list | grep -E "(openai|yfinance|pandas)"`

### Create a Minimal Example
```python
# test_minimal.py
import os
os.environ['OPENAI_API_KEY'] = 'test-key'

from data.stock_data import StockDataProvider
provider = StockDataProvider()
data = provider.get_stock_data('AAPL', '1y')
print(f"Retrieved {len(data)} days of data")
```

### Environment Check
```bash
# Check Python version (should be 3.8+)
python --version

# Check virtual environment
which python

# Check installed packages
pip list

# Check working directory
pwd
ls -la
```

## 🔄 Clean Reset

If all else fails, try a clean reset:

```bash
# Remove virtual environment
rm -rf venv

# Create new environment
python -m venv venv
source venv/bin/activate

# Install fresh dependencies
pip install -r requirements.txt

# Set API key
export OPENAI_API_KEY="your_key_here"

# Test again
python test_system.py
``` 
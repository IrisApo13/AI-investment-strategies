"""
Investment strategy representation and signal generation.
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
import re
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

from config.settings import Config

logger = logging.getLogger(__name__)

class InvestmentStrategy:
    """Represents an investment strategy with buy/sell conditions."""
    
    def __init__(self, strategy_dict: Dict):
        """
        Initialize strategy from dictionary.
        
        Args:
            strategy_dict: Dictionary containing strategy parameters
        """
        self.name = strategy_dict.get('name', 'Unnamed Strategy')
        self.description = strategy_dict.get('description', '')
        self.buy_conditions = strategy_dict.get('buy_conditions', [])
        self.sell_conditions = strategy_dict.get('sell_conditions', [])
        self.position_sizing = strategy_dict.get('position_sizing', '10% of portfolio')
        self.risk_management = strategy_dict.get('risk_management', '5% stop loss')
        
        # Parse position sizing
        self.position_size = self._parse_position_size(self.position_sizing)
        
        # Validate conditions
        self._validate_conditions()
        
        logger.info(f"Initialized strategy: {self.name}")
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate buy/sell signals for the given data.
        
        Args:
            data: DataFrame with OHLCV and technical indicators
            
        Returns:
            DataFrame with buy_signal, sell_signal, and position_size columns
        """
        try:
            logger.info(f"Generating signals for {len(data)} data points")
            
            # Initialize signals DataFrame
            signals = pd.DataFrame(index=data.index)
            signals['buy_signal'] = False
            signals['sell_signal'] = False
            signals['position_size'] = 0.0
            
            # Track position state for profit/loss calculations
            position_entries = []  # List of (date, price) tuples for open positions
            last_action_date = None
            min_hold_days = 1  # Minimum days between actions
            
            for i, (date, row) in enumerate(data.iterrows()):
                # Calculate days since last action
                days_since_last_action = min_hold_days  # Default to allow first action
                if last_action_date is not None:
                    days_since_last_action = (date - last_action_date).days
                
                # Check buy conditions - allow multiple positions
                if (self._evaluate_conditions(row, self.buy_conditions) and
                    days_since_last_action >= min_hold_days and
                    len(position_entries) < 3):  # Max 3 overlapping positions
                    
                    signals.loc[date, 'buy_signal'] = True
                    signals.loc[date, 'position_size'] = self.position_size
                    position_entries.append((date, row['Close']))
                    last_action_date = date
                    logger.debug(f"BUY signal generated on {date} at ${row['Close']:.2f} position size {self.position_size}")
                
                # Check sell conditions - close oldest position if conditions met
                elif (position_entries and 
                      days_since_last_action >= min_hold_days):
                    
                    # Check sell conditions for the oldest position
                    oldest_entry_date, oldest_entry_price = position_entries[0]
                    
                    if self._evaluate_conditions(row, self.sell_conditions, oldest_entry_price):
                        signals.loc[date, 'sell_signal'] = True
                        signals.loc[date, 'position_size'] = self.position_size
                        position_entries.pop(0)  # Remove the oldest position
                        last_action_date = date
                        logger.debug(f"SELL signal generated on {date} at ${row['Close']:.2f} position size {self.position_size}")
            
            num_buy_signals = signals['buy_signal'].sum()
            num_sell_signals = signals['sell_signal'].sum()
            logger.info(f"Generated {num_buy_signals} buy signals and {num_sell_signals} sell signals")
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}")
            # Return empty signals
            signals = pd.DataFrame(index=data.index)
            signals['buy_signal'] = False
            signals['sell_signal'] = False
            signals['position_size'] = 0.0
            return signals
    
    def _evaluate_conditions(self, row: pd.Series, conditions: List[str], entry_price: float = None) -> bool:
        """
        Evaluate a list of conditions for a given data row.
        
        Args:
            row: Data row with indicators
            conditions: List of condition strings
            entry_price: Entry price for profit/loss calculations
            
        Returns:
            True if any condition is met (OR logic for multiple conditions)
            For explicit AND conditions (single condition with "AND"), all parts must be true
        """
        try:
            if not conditions:
                return False
            
            # Clean and filter valid conditions
            valid_conditions = []
            for condition in conditions:
                cleaned = self._clean_condition(condition.strip())
                if cleaned:  # Only keep non-empty cleaned conditions
                    valid_conditions.append(cleaned)
            
            # If no valid conditions remain after cleaning, return False
            if not valid_conditions:
                logger.debug(f"No valid conditions found after cleaning from: {conditions}")
                return False
            
            # If there's only one valid condition, evaluate it directly
            if len(valid_conditions) == 1:
                return self._evaluate_single_condition(row, valid_conditions[0], entry_price)
            
            # For multiple conditions, use OR logic (any condition can trigger)
            # This makes strategies more likely to generate signals
            for condition in valid_conditions:
                if self._evaluate_single_condition(row, condition, entry_price):
                    return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Error evaluating conditions: {str(e)}")
            return False
    
    def _evaluate_single_condition(self, row: pd.Series, condition: str, entry_price: float = None) -> bool:
        """
        Evaluate a single condition string.
        
        Args:
            row: Data row with indicators
            condition: Condition string (e.g., "RSI < 30")
            entry_price: Entry price for profit/loss calculations
            
        Returns:
            True if condition is met
        """
        try:
            original_condition = condition.strip()
            
            # Clean up the condition - extract just the executable part
            condition = self._clean_condition(original_condition)
            
            # If cleaning resulted in empty condition, skip it
            if not condition:
                logger.debug(f"Skipping empty/invalid condition: '{original_condition}'")
                return False
            
            # Handle OR conditions
            if ' OR ' in condition.upper():
                or_parts = condition.upper().split(' OR ')
                return any(self._evaluate_single_condition(row, part.strip(), entry_price) for part in or_parts)
            
            # Handle AND conditions
            if ' AND ' in condition.upper():
                and_parts = condition.upper().split(' AND ')
                return all(self._evaluate_single_condition(row, part.strip(), entry_price) for part in and_parts)
            
            # Handle profit/loss conditions
            if 'profit' in condition.lower() or 'loss' in condition.lower():
                return self._evaluate_profit_loss_condition(row, condition, entry_price)
            
            # Parse simple comparison conditions
            return self._parse_comparison_condition(row, condition)
            
        except Exception as e:
            logger.warning(f"Error evaluating condition '{original_condition}': {str(e)}")
            return False
    
    def _clean_condition(self, condition: str) -> str:
        """Clean up condition string to extract just the executable part."""
        try:
            original_condition = condition.strip()
            
            # Skip empty conditions
            if not original_condition:
                return ""
            
            # Handle common problematic patterns seen in warnings
            # If it starts with just a number or ends with just a comma, it's likely garbage
            if re.match(r'^\d+[\.,]?\s*$', original_condition):
                logger.debug(f"Skipping numeric fragment: '{original_condition}'")
                return ""
            
            # If it's just text with no operators, it's likely explanatory text
            if not any(op in original_condition for op in ['<', '>', '=', '!=']):
                logger.debug(f"Skipping non-comparison text: '{original_condition}'")
                return ""
            
            # Remove trailing commas and extra whitespace
            condition = re.sub(r',\s*$', '', original_condition)
            
            # More aggressive cleanup patterns
            cleanup_patterns = [
                r'\s+(to\s+.*)$',  # Remove "to identify oversold conditions..."
                r'\s+(for\s+.*)$',  # Remove "for better entry points..."
                r'\s+(indicating\s+.*)$',  # Remove "indicating a stronger potential..."
                r'\s+(as\s+.*)$',  # Remove "as this suggests..."
                r'\s+(which\s+.*)$',  # Remove "which means..."
                r'\s+(when\s+.*)$',  # Remove explanatory "when" clauses
                r'\s+(confirming\s+.*)$',  # Remove "confirming stronger uptrend..."
                r'\s+(adjusted\s*)$',  # Remove "adjusted" at end
                r'\s*\([^)]*\)$',  # Remove parenthetical explanations at end
                r'\s*,\s*[a-z][^<>=]*$',  # Remove trailing explanations after comma
            ]
            
            for pattern in cleanup_patterns:
                condition = re.sub(pattern, '', condition, flags=re.IGNORECASE).strip()
            
            # Extract valid comparison patterns more strictly
            # Look for: INDICATOR OPERATOR VALUE or INDICATOR OPERATOR INDICATOR * VALUE
            valid_patterns = [
                # Pattern: RSI < 30, Close > SMA_20, Volume > volume_sma * 1.5
                r'^([A-Za-z_][A-Za-z0-9_]*)\s*([<>=!]+)\s*([A-Za-z_][A-Za-z0-9_]*(?:\s*\*\s*[\d.]+)?|[\d.]+)$',
                # Pattern: Close > SMA_20 * 1.02
                r'^([A-Za-z_][A-Za-z0-9_]*)\s*([<>=!]+)\s*([A-Za-z_][A-Za-z0-9_]*\s*\*\s*[\d.]+)$'
            ]
            
            for pattern in valid_patterns:
                match = re.match(pattern, condition)
                if match:
                    clean_condition = f"{match.group(1).strip()} {match.group(2)} {match.group(3).strip()}"
                    if clean_condition != original_condition:
                        logger.debug(f"Extracted valid condition: '{original_condition}' -> '{clean_condition}'")
                    return clean_condition
            
            # If no valid pattern found but we have operators, try to salvage
            if any(op in condition for op in ['<', '>', '=']):
                # Remove any remaining explanatory text after the comparison
                condition = re.sub(r'([<>=!]+\s*[\d.]+).*$', r'\1', condition)
                condition = re.sub(r'([<>=!]+\s*[A-Za-z_][A-Za-z0-9_]*(?:\s*\*\s*[\d.]+)?).*$', r'\1', condition)
                
                if condition != original_condition:
                    logger.debug(f"Salvaged condition: '{original_condition}' -> '{condition}'")
                return condition.strip()
            
            # If we get here, the condition is likely not parseable
            logger.debug(f"Could not clean condition, skipping: '{original_condition}'")
            return ""
            
        except Exception as e:
            logger.warning(f"Error cleaning condition '{original_condition}': {str(e)}")
            return ""
    
    def _evaluate_profit_loss_condition(self, row: pd.Series, condition: str, entry_price: float) -> bool:
        """Evaluate profit/loss conditions."""
        if entry_price is None:
            return False
        
        current_price = row['Close']
        profit_loss = (current_price - entry_price) / entry_price
        
        # Extract threshold from condition
        threshold_match = re.search(r'([\d.]+)', condition)
        if not threshold_match:
            return False
        
        threshold = float(threshold_match.group(1))
        
        if 'profit' in condition.lower() and '>' in condition:
            return profit_loss > threshold
        elif 'loss' in condition.lower() and ('>' in condition or '<' in condition):
            return profit_loss < -threshold
        
        return False
    
    def _parse_comparison_condition(self, row: pd.Series, condition: str) -> bool:
        """Parse and evaluate comparison conditions like 'RSI < 30'."""
        try:
            # Normalize condition
            condition = condition.replace('<=', '≤').replace('>=', '≥').replace('==', '=')
            
            # Extract operator
            operators = ['≤', '≥', '<', '>', '=']
            operator = None
            for op in operators:
                if op in condition:
                    operator = op
                    break
            
            if not operator:
                return False
            
            # Split condition
            parts = condition.split(operator)
            if len(parts) != 2:
                return False
            
            left_expr = parts[0].strip()
            right_expr = parts[1].strip()
            
            # Evaluate left and right expressions
            left_value = self._evaluate_expression(row, left_expr)
            right_value = self._evaluate_expression(row, right_expr)
            
            if left_value is None or right_value is None:
                return False
            
            # Apply comparison
            if operator == '<':
                return left_value < right_value
            elif operator == '>':
                return left_value > right_value
            elif operator == '=':
                return abs(left_value - right_value) < 1e-6
            elif operator == '≤':
                return left_value <= right_value
            elif operator == '≥':
                return left_value >= right_value
            
            return False
            
        except Exception as e:
            logger.warning(f"Error parsing comparison condition '{condition}': {str(e)}")
            return False
    
    def _evaluate_expression(self, row: pd.Series, expr: str) -> Optional[float]:
        """Evaluate an expression that can contain indicators, numbers, or simple math."""
        try:
            expr = expr.strip()
            
            # Skip empty expressions
            if not expr:
                return None
            
            # Skip problematic patterns that shouldn't be evaluated
            problematic_patterns = [
                r'^\d+\s*,\s*$',  # Just a number with comma
                r'^[a-z]+\s*,\s*$',  # Just text with comma  
                r'adjusted$',  # Ends with "adjusted"
                r'confirming\s+',  # Contains "confirming"
                r'^[A-Z]+\([^)]*\)$',  # Function calls like RSI(14)
            ]
            
            for pattern in problematic_patterns:
                if re.search(pattern, expr, re.IGNORECASE):
                    logger.debug(f"Skipping problematic expression: {expr}")
                    return None
            
            # If it's a number, return it
            try:
                return float(expr)
            except ValueError:
                pass
            
            # Handle simple multiplication (e.g., "SMA_20 * 1.02")
            if '*' in expr:
                parts = expr.split('*')
                if len(parts) == 2:
                    left = self._evaluate_expression(row, parts[0])
                    right = self._evaluate_expression(row, parts[1])
                    if left is not None and right is not None:
                        return left * right
            
            # Handle simple division
            if '/' in expr:
                parts = expr.split('/')
                if len(parts) == 2:
                    left = self._evaluate_expression(row, parts[0])
                    right = self._evaluate_expression(row, parts[1])
                    if left is not None and right is not None and right != 0:
                        return left / right
            
            # Check if it's a column name
            if expr in row.index:
                value = row[expr]
                return float(value) if pd.notna(value) else None
            
            # Check for common aliases
            aliases = {
                'Close': 'Close',
                'Price': 'Close',
                'Volume': 'Volume'
            }
            
            if expr in aliases and aliases[expr] in row.index:
                value = row[aliases[expr]]
                return float(value) if pd.notna(value) else None
            
            # Only warn if it looks like it should be a valid expression
            if any(c.isalnum() or c == '_' for c in expr):
                logger.debug(f"Could not evaluate expression: {expr}")
            
            return None
            
        except Exception as e:
            logger.debug(f"Error evaluating expression '{expr}': {str(e)}")
            return None
    
    def _parse_position_size(self, sizing_text: str) -> float:
        """Parse position sizing from text."""
        try:
            # Extract percentage
            percentage_match = re.search(r'(\d+(?:\.\d+)?)%', sizing_text)
            if percentage_match:
                percentage = float(percentage_match.group(1))
                return min(percentage / 100, Config.MAX_POSITION_SIZE)
            
            # Default to 10%
            return 0.1
            
        except Exception:
            return 0.1
    
    def _validate_conditions(self):
        """Validate that conditions can be evaluated."""
        if not self.buy_conditions:
            self.buy_conditions = ["RSI < 30"]
            logger.warning("No buy conditions specified, using default: RSI < 30")
        
        if not self.sell_conditions:
            self.sell_conditions = ["RSI > 70"]
            logger.warning("No sell conditions specified, using default: RSI > 70")
    
    def to_dict(self) -> Dict:
        """Convert strategy to dictionary."""
        return {
            'name': self.name,
            'description': self.description,
            'buy_conditions': self.buy_conditions,
            'sell_conditions': self.sell_conditions,
            'position_sizing': self.position_sizing,
            'risk_management': self.risk_management
        }
    
    def __str__(self) -> str:
        """String representation of the strategy."""
        return f"""
Strategy: {self.name}
Description: {self.description}

Buy Conditions:
{chr(10).join(f"  - {cond}" for cond in self.buy_conditions)}

Sell Conditions:
{chr(10).join(f"  - {cond}" for cond in self.sell_conditions)}

Position Sizing: {self.position_sizing}
Risk Management: {self.risk_management}
""".strip()
    
    def plot_strategy(self, data: pd.DataFrame, signals: pd.DataFrame = None, 
                     start_date: str = None, end_date: str = None,
                     figsize: Tuple[int, int] = (15, 10)) -> None:
        """
        Plot stock price, technical indicators used in strategy conditions, and buy/sell signals.
        
        Args:
            data: DataFrame with OHLCV and technical indicators
            signals: DataFrame with buy/sell signals (if None, will generate them)
            start_date: Start date for plotting (YYYY-MM-DD format)
            end_date: End date for plotting (YYYY-MM-DD format)
            figsize: Figure size as (width, height)
        """
        try:
            # Filter data by date range if specified
            plot_data = data.copy()
            if start_date:
                plot_data = plot_data[plot_data.index >= start_date]
            if end_date:
                plot_data = plot_data[plot_data.index <= end_date]
            
            if plot_data.empty:
                logger.warning("No data available for the specified date range")
                return
            
            # Generate signals if not provided
            if signals is None:
                signals = self.generate_signals(plot_data)
            else:
                # Filter signals to match data range
                signals = signals.loc[plot_data.index]
            
            # Extract indicators used in conditions
            indicators_used = self._extract_indicators_from_conditions()
            
            # Remove indicators that don't exist in the data
            available_indicators = [ind for ind in indicators_used if ind in plot_data.columns]
            
            if not available_indicators:
                logger.warning("No technical indicators found in data that match strategy conditions")
                # Just plot price and signals
                available_indicators = []
            
            # Determine number of subplots needed
            num_subplots = 2 + len(available_indicators)  # Price + Volume + Indicators
            
            # Create subplots
            fig, axes = plt.subplots(num_subplots, 1, figsize=figsize, 
                                   gridspec_kw={'height_ratios': [3] + [1] * (num_subplots - 1)})
            
            if num_subplots == 1:
                axes = [axes]
            
            # Plot 1: Stock Price with Buy/Sell Signals
            ax_price = axes[0]
            ax_price.plot(plot_data.index, plot_data['Close'], label='Close Price', linewidth=1.5, color='black')
            
            # Add buy/sell signals
            buy_signals = signals[signals['buy_signal']]
            sell_signals = signals[signals['sell_signal']]
            
            if not buy_signals.empty:
                buy_prices = plot_data.loc[buy_signals.index, 'Close']
                ax_price.scatter(buy_signals.index, buy_prices, 
                               color='green', marker='^', s=100, label='Buy Signal', zorder=5)
            
            if not sell_signals.empty:
                sell_prices = plot_data.loc[sell_signals.index, 'Close']
                ax_price.scatter(sell_signals.index, sell_prices, 
                                color='red', marker='v', s=100, label='Sell Signal', zorder=5)
            
            ax_price.set_title(f'{self.name} - Stock Price and Signals', fontsize=14, fontweight='bold')
            ax_price.set_ylabel('Price ($)', fontsize=12)
            ax_price.legend(loc='upper left')
            ax_price.grid(True, alpha=0.3)
            
            # Plot 2: Volume
            ax_volume = axes[1]
            ax_volume.bar(plot_data.index, plot_data['Volume'], alpha=0.6, color='lightblue')
            ax_volume.set_ylabel('Volume', fontsize=12)
            ax_volume.set_title('Volume', fontsize=12)
            ax_volume.grid(True, alpha=0.3)
            
            # Plot indicators
            for i, indicator in enumerate(available_indicators):
                ax_ind = axes[2 + i]
                ax_ind.plot(plot_data.index, plot_data[indicator], 
                           label=indicator, linewidth=1.5, color=f'C{i}')
                
                # Add horizontal lines for common thresholds
                self._add_indicator_thresholds(ax_ind, indicator)
                
                ax_ind.set_ylabel(indicator, fontsize=12)
                ax_ind.set_title(f'{indicator}', fontsize=12)
                ax_ind.legend(loc='upper left')
                ax_ind.grid(True, alpha=0.3)
            
            # Format x-axis for all subplots
            for ax in axes:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
            # Set x-label only on bottom subplot
            axes[-1].set_xlabel('Date', fontsize=12)
            
            # Adjust layout
            plt.tight_layout()
            
            # Show strategy information
            strategy_info = f"""
Strategy: {self.name}
Buy Conditions: {', '.join(self.buy_conditions[:3])}{'...' if len(self.buy_conditions) > 3 else ''}
Sell Conditions: {', '.join(self.sell_conditions[:3])}{'...' if len(self.sell_conditions) > 3 else ''}
Total Buy Signals: {len(buy_signals)}
Total Sell Signals: {len(sell_signals)}
            """.strip()
            
            fig.suptitle(strategy_info, fontsize=10, y=0.02, ha='left', va='bottom')
            
            plt.show()
            
            logger.info(f"Plotted strategy visualization with {len(available_indicators)} indicators")
            
        except Exception as e:
            logger.error(f"Error plotting strategy: {str(e)}")
            raise
    
    def _extract_indicators_from_conditions(self) -> List[str]:
        """Extract technical indicator names from buy/sell conditions."""
        indicators = set()
        
        all_conditions = self.buy_conditions + self.sell_conditions
        
        for condition in all_conditions:
            # Clean the condition first
            cleaned_condition = self._clean_condition(condition)
            if not cleaned_condition:
                continue
            
            # Extract indicator names using a more comprehensive approach
            # Look for patterns like RSI, SMA_20, MACD, etc.
            indicator_patterns = [
                r'\b([A-Z][A-Z_0-9]*)\b',  # RSI, SMA_20, MACD_SIGNAL
                r'\b(sma_\d+)\b',          # sma_20
                r'\b(ema_\d+)\b',          # ema_20
                r'\b(rsi)\b',              # rsi (lowercase)
                r'\b(macd)\b',             # macd (lowercase)
                r'\b(volume_sma)\b',       # volume_sma
                r'\b(bb_upper|bb_lower|bb_middle)\b',  # Bollinger Bands
                r'\b(atr)\b',              # atr (lowercase)
                r'\b(price_change)\b',     # price_change
                r'\b(volatility)\b',       # volatility
                r'\b(momentum)\b',         # momentum
            ]
            
            for pattern in indicator_patterns:
                matches = re.findall(pattern, cleaned_condition, re.IGNORECASE)
                for match in matches:
                    # Skip basic OHLCV columns
                    if match.upper() not in ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']:
                        indicators.add(match)
        
        # Debug logging
        logger.debug(f"Extracted indicators from conditions: {list(indicators)}")
        return list(indicators)
    
    def _extract_thresholds_from_conditions(self, indicator: str) -> List[Tuple[float, str, str]]:
        """
        Extract thresholds for a specific indicator from strategy conditions.
        
        Args:
            indicator: The indicator name to extract thresholds for
            
        Returns:
            List of tuples: (threshold_value, condition_type, color)
        """
        thresholds = []
        indicator_lower = indicator.lower()
        
        all_conditions = self.buy_conditions + self.sell_conditions
        
        for condition in all_conditions:
            # Clean the condition first
            cleaned_condition = self._clean_condition(condition)
            if not cleaned_condition:
                continue
            
            # Check if this condition involves our indicator
            if indicator_lower not in cleaned_condition.lower():
                continue
            
            # Extract threshold value and operator
            # Pattern: INDICATOR OPERATOR VALUE
            pattern = rf'\b{re.escape(indicator)}\s*([<>=!]+)\s*([\d.]+(?:\s*\*\s*[\d.]+)?)'
            match = re.search(pattern, cleaned_condition, re.IGNORECASE)
            
            if match:
                operator = match.group(1)
                value_str = match.group(2)
                
                # Handle multiplication (e.g., "SMA_20 * 1.02")
                if '*' in value_str:
                    try:
                        parts = value_str.split('*')
                        base_value = float(parts[0].strip())
                        multiplier = float(parts[1].strip())
                        threshold_value = base_value * multiplier
                    except (ValueError, IndexError):
                        continue
                else:
                    try:
                        threshold_value = float(value_str)
                    except ValueError:
                        continue
                
                # Determine condition type and color
                if operator in ['<', '<=']:
                    condition_type = 'Buy Signal'
                    color = 'green'
                elif operator in ['>', '>=']:
                    condition_type = 'Sell Signal'
                    color = 'red'
                else:
                    condition_type = 'Signal'
                    color = 'blue'
                
                thresholds.append((threshold_value, condition_type, color))
        
        return thresholds
    
    def _add_indicator_thresholds(self, ax: plt.Axes, indicator: str) -> None:
        """Add threshold lines for technical indicators based on strategy conditions."""
        indicator_lower = indicator.lower()
        
        # Extract thresholds from strategy conditions
        strategy_thresholds = self._extract_thresholds_from_conditions(indicator)
        
        # Add strategy-specific thresholds
        for threshold_value, condition_type, color in strategy_thresholds:
            ax.axhline(y=threshold_value, color=color, linestyle='--', alpha=0.7, 
                      label=f'{condition_type} ({threshold_value:.1f})')
        
        # Add common default thresholds if no strategy thresholds found
        if not strategy_thresholds:
            if 'rsi' in indicator_lower:
                ax.axhline(y=70, color='red', linestyle='--', alpha=0.3, label='Standard Overbought (70)')
                ax.axhline(y=30, color='green', linestyle='--', alpha=0.3, label='Standard Oversold (30)')
                ax.set_ylim(0, 100)
            
            elif 'stoch' in indicator_lower or 'k%' in indicator_lower or 'd%' in indicator_lower:
                ax.axhline(y=80, color='red', linestyle='--', alpha=0.3, label='Standard Overbought (80)')
                ax.axhline(y=20, color='green', linestyle='--', alpha=0.3, label='Standard Oversold (20)')
                ax.set_ylim(0, 100)
            
            elif 'macd' in indicator_lower and 'signal' not in indicator_lower:
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, label='Zero Line')
            
            elif 'williams' in indicator_lower or '%r' in indicator_lower:
                ax.axhline(y=-20, color='red', linestyle='--', alpha=0.3, label='Standard Overbought (-20)')
                ax.axhline(y=-80, color='green', linestyle='--', alpha=0.3, label='Standard Oversold (-80)')
                ax.set_ylim(-100, 0) 
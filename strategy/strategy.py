"""
Investment strategy representation and signal generation.
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
import re

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
                    logger.debug(f"BUY signal generated on {date} at ${row['Close']:.2f}")
                
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
                        logger.debug(f"SELL signal generated on {date} at ${row['Close']:.2f}")
            
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
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
            position_price = None
            in_position = False
            
            for i, (date, row) in enumerate(data.iterrows()):
                # Check buy conditions
                if not in_position and self._evaluate_conditions(row, self.buy_conditions):
                    signals.loc[date, 'buy_signal'] = True
                    signals.loc[date, 'position_size'] = self.position_size
                    position_price = row['Close']
                    in_position = True
                    logger.debug(f"BUY signal generated on {date} at ${row['Close']:.2f}")
                
                # Check sell conditions (including profit/loss conditions)
                elif in_position and self._evaluate_conditions(row, self.sell_conditions, position_price):
                    signals.loc[date, 'sell_signal'] = True
                    signals.loc[date, 'position_size'] = self.position_size
                    in_position = False
                    logger.debug(f"SELL signal generated on {date} at ${row['Close']:.2f}")
                    position_price = None
            
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
            True if all conditions are met
        """
        try:
            for condition in conditions:
                if not self._evaluate_single_condition(row, condition, entry_price):
                    return False
            return True
            
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
            condition = condition.strip()
            
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
            logger.warning(f"Error evaluating condition '{condition}': {str(e)}")
            return False
    
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
            
            logger.warning(f"Could not evaluate expression: {expr}")
            return None
            
        except Exception as e:
            logger.warning(f"Error evaluating expression '{expr}': {str(e)}")
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
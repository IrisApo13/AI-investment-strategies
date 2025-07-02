"""
Utility module for loading investment strategies from JSON files.
"""
import json
import os
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

def load_strategy_from_file(file_path: str) -> Optional[Dict]:
    """
    Load a strategy from a JSON file.
    
    Args:
        file_path: Path to the JSON file containing the strategy
        
    Returns:
        Dictionary containing the strategy configuration, or None if loading failed
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            logger.error(f"Strategy file not found: {file_path}")
            return None
        
        # Read and parse JSON file
        with open(file_path, 'r', encoding='utf-8') as file:
            strategy_dict = json.load(file)
        
        # Validate required fields
        required_fields = ['name', 'buy_conditions', 'sell_conditions']
        missing_fields = [field for field in required_fields if field not in strategy_dict]
        
        if missing_fields:
            logger.error(f"Strategy file missing required fields: {missing_fields}")
            return None
        
        # Validate that conditions are lists
        if not isinstance(strategy_dict['buy_conditions'], list):
            logger.error("buy_conditions must be a list")
            return None
        
        if not isinstance(strategy_dict['sell_conditions'], list):
            logger.error("sell_conditions must be a list")
            return None
        
        # Set default values for optional fields
        if 'description' not in strategy_dict:
            strategy_dict['description'] = f"Strategy loaded from {file_path}"
        
        if 'position_sizing' not in strategy_dict:
            strategy_dict['position_sizing'] = '10% of portfolio'
        
        if 'risk_management' not in strategy_dict:
            strategy_dict['risk_management'] = '5% stop loss'
        
        logger.info(f"Successfully loaded strategy '{strategy_dict['name']}' from {file_path}")
        return strategy_dict
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in strategy file {file_path}: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Error loading strategy from {file_path}: {str(e)}")
        return None

def save_strategy_to_file(strategy_dict: Dict, file_path: str) -> bool:
    """
    Save a strategy to a JSON file.
    
    Args:
        strategy_dict: Dictionary containing the strategy configuration
        file_path: Path where to save the JSON file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save to JSON file
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(strategy_dict, file, indent=2, ensure_ascii=False)
        
        logger.info(f"Strategy saved to {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving strategy to {file_path}: {str(e)}")
        return False

def create_example_strategy_file(file_path: str = "example_strategy.json") -> bool:
    """
    Create an example strategy JSON file.
    
    Args:
        file_path: Path where to save the example file
        
    Returns:
        True if successful, False otherwise
    """
    example_strategy = {
        "name": "Example RSI Strategy",
        "description": "A simple RSI-based strategy for demonstration purposes",
        "buy_conditions": [
            "RSI < 40",
            "Close > SMA_20"
        ],
        "sell_conditions": [
            "RSI > 70",
            "Close < SMA_20 * 0.98"
        ],
        "position_sizing": "20% of portfolio",
        "risk_management": "5% stop loss"
    }
    
    return save_strategy_to_file(example_strategy, file_path) 
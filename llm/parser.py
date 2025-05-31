"""
Parser for LLM responses and strategy extraction.
"""
import json
import re
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

class StrategyParser:
    """Parses LLM responses and extracts structured strategy information."""
    
    def __init__(self):
        """Initialize the strategy parser."""
        pass
    
    def parse_strategy_response(self, response: str) -> Optional[Dict]:
        """
        Parse LLM response and extract strategy information.
        
        Args:
            response: Raw LLM response text
            
        Returns:
            Parsed strategy dictionary or None if parsing fails
        """
        try:
            # First, try to extract JSON from the response
            strategy_json = self._extract_json(response)
            
            if strategy_json:
                # Validate the strategy structure
                validated_strategy = self._validate_strategy(strategy_json)
                if validated_strategy:
                    logger.info("Successfully parsed strategy from LLM response")
                    return validated_strategy
            
            # If JSON parsing fails, try to extract strategy information manually
            logger.warning("JSON parsing failed, attempting manual extraction")
            return self._manual_parse_strategy(response)
            
        except Exception as e:
            logger.error(f"Error parsing strategy response: {str(e)}")
            return None
    
    def _extract_json(self, text: str) -> Optional[Dict]:
        """Extract JSON object from text."""
        try:
            # Look for JSON block
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
            
            # Try parsing the entire text as JSON
            return json.loads(text)
            
        except json.JSONDecodeError:
            # Try to fix common JSON issues
            return self._fix_and_parse_json(text)
    
    def _fix_and_parse_json(self, text: str) -> Optional[Dict]:
        """Attempt to fix common JSON formatting issues."""
        try:
            # Remove markdown code blocks
            text = re.sub(r'```json\s*', '', text)
            text = re.sub(r'```\s*$', '', text)
            
            # Find JSON-like structure
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
            if not json_match:
                return None
            
            json_str = json_match.group(0)
            
            # Fix common issues
            json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas
            json_str = re.sub(r',\s*]', ']', json_str)  # Remove trailing commas in arrays
            
            # Try to parse
            return json.loads(json_str)
            
        except Exception:
            return None
    
    def _manual_parse_strategy(self, text: str) -> Optional[Dict]:
        """Manually extract strategy information from unstructured text."""
        try:
            strategy = {}
            
            # Extract name
            name_match = re.search(r'(?:name|strategy)[:=]\s*["\']?([^"\'\n]+)["\']?', text, re.IGNORECASE)
            strategy['name'] = name_match.group(1).strip() if name_match else "Manual Parsed Strategy"
            
            # Extract description
            desc_match = re.search(r'(?:description|summary)[:=]\s*["\']?([^"\'\n]+)["\']?', text, re.IGNORECASE)
            strategy['description'] = desc_match.group(1).strip() if desc_match else "Manually extracted strategy"
            
            # Extract buy conditions
            buy_conditions = self._extract_conditions(text, "buy")
            strategy['buy_conditions'] = buy_conditions if buy_conditions else ["RSI < 30", "Close > SMA_20"]
            
            # Extract sell conditions
            sell_conditions = self._extract_conditions(text, "sell")
            strategy['sell_conditions'] = sell_conditions if sell_conditions else ["RSI > 70", "Close < SMA_20"]
            
            # Extract position sizing
            position_match = re.search(r'(?:position.sizing|position.size)[:=]\s*["\']?([^"\'\n]+)["\']?', text, re.IGNORECASE)
            strategy['position_sizing'] = position_match.group(1).strip() if position_match else "10% of portfolio"
            
            # Extract risk management
            risk_match = re.search(r'(?:risk.management|stop.loss)[:=]\s*["\']?([^"\'\n]+)["\']?', text, re.IGNORECASE)
            strategy['risk_management'] = risk_match.group(1).strip() if risk_match else "5% stop loss"
            
            logger.info("Successfully extracted strategy using manual parsing")
            return strategy
            
        except Exception as e:
            logger.error(f"Manual parsing failed: {str(e)}")
            return None
    
    def _extract_conditions(self, text: str, condition_type: str) -> List[str]:
        """Extract buy or sell conditions from text."""
        try:
            pattern = rf'{condition_type}[_\s]*conditions?[:=]\s*\[(.*?)\]'
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            
            if match:
                conditions_text = match.group(1)
                # Extract individual conditions
                conditions = re.findall(r'["\']([^"\']+)["\']', conditions_text)
                return [c.strip() for c in conditions if c.strip()]
            
            # Fallback: look for bullet points or numbered lists
            pattern = rf'{condition_type}[^:]*:(.+?)(?={condition_type}|sell|position|risk|$)'
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            
            if match:
                section = match.group(1)
                # Extract lines that look like conditions
                lines = section.split('\n')
                conditions = []
                for line in lines:
                    line = line.strip()
                    if any(op in line for op in ['<', '>', '=', 'AND', 'OR']) and len(line) > 5:
                        # Clean up the line
                        line = re.sub(r'^[-*•]\s*', '', line)  # Remove bullet points
                        line = re.sub(r'^\d+\.\s*', '', line)  # Remove numbers
                        conditions.append(line)
                
                return conditions[:5]  # Limit to 5 conditions
            
            return []
            
        except Exception:
            return []
    
    def _validate_strategy(self, strategy: Dict) -> Optional[Dict]:
        """Validate strategy structure and content."""
        try:
            required_fields = ['name', 'description', 'buy_conditions', 'sell_conditions']
            
            # Check required fields
            for field in required_fields:
                if field not in strategy:
                    logger.warning(f"Missing required field: {field}")
                    return None
            
            # Validate conditions are lists
            if not isinstance(strategy['buy_conditions'], list):
                if isinstance(strategy['buy_conditions'], str):
                    strategy['buy_conditions'] = [strategy['buy_conditions']]
                else:
                    strategy['buy_conditions'] = ["RSI < 30"]
            
            if not isinstance(strategy['sell_conditions'], list):
                if isinstance(strategy['sell_conditions'], str):
                    strategy['sell_conditions'] = [strategy['sell_conditions']]
                else:
                    strategy['sell_conditions'] = ["RSI > 70"]
            
            # Ensure we have at least one condition of each type
            if not strategy['buy_conditions']:
                strategy['buy_conditions'] = ["RSI < 30"]
            
            if not strategy['sell_conditions']:
                strategy['sell_conditions'] = ["RSI > 70"]
            
            # Add default values for optional fields
            if 'position_sizing' not in strategy:
                strategy['position_sizing'] = "10% of portfolio"
            
            if 'risk_management' not in strategy:
                strategy['risk_management'] = "5% stop loss"
            
            # Validate condition format
            strategy['buy_conditions'] = self._validate_conditions(strategy['buy_conditions'])
            strategy['sell_conditions'] = self._validate_conditions(strategy['sell_conditions'])
            
            return strategy
            
        except Exception as e:
            logger.error(f"Strategy validation failed: {str(e)}")
            return None
    
    def _validate_conditions(self, conditions: List[str]) -> List[str]:
        """Validate and clean up condition strings."""
        valid_conditions = []
        
        for condition in conditions:
            if not isinstance(condition, str):
                continue
            
            condition = condition.strip()
            
            # Check if condition contains comparison operators
            if any(op in condition for op in ['<', '>', '=', 'AND', 'OR']):
                # Clean up common issues
                condition = re.sub(r'\s+', ' ', condition)  # Normalize whitespace
                valid_conditions.append(condition)
            else:
                # Skip non-executable conditions
                logger.warning(f"Skipping non-executable condition: {condition}")
        
        # Ensure we have at least one valid condition
        if not valid_conditions:
            valid_conditions = ["RSI < 50"]  # Default fallback
        
        return valid_conditions
    
    def extract_feedback(self, response: str) -> Dict:
        """Extract feedback and analysis from LLM response."""
        try:
            feedback = {
                'summary': '',
                'strengths': [],
                'weaknesses': [],
                'suggestions': []
            }
            
            # Try to extract structured feedback
            sections = {
                'strengths': r'(?:worked well|strengths?)[:.\-]\s*(.*?)(?=(?:weaknesses?|areas for improvement|suggestions?)|$)',
                'weaknesses': r'(?:weaknesses?|areas for improvement)[:.\-]\s*(.*?)(?=suggestions?|$)',
                'suggestions': r'suggestions?[:.\-]\s*(.*?)$'
            }
            
            for section, pattern in sections.items():
                match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
                if match:
                    content = match.group(1).strip()
                    # Extract bullet points or numbered items
                    items = re.findall(r'(?:^|\n)\s*[-*•\d+\.]\s*([^\n]+)', content)
                    feedback[section] = [item.strip() for item in items if item.strip()]
            
            # Extract overall summary
            summary_match = re.search(r'^(.*?)(?:strengths?|weaknesses?|suggestions?)', response, re.IGNORECASE | re.DOTALL)
            if summary_match:
                feedback['summary'] = summary_match.group(1).strip()[:500]  # Limit length
            
            return feedback
            
        except Exception as e:
            logger.error(f"Error extracting feedback: {str(e)}")
            return {'summary': response[:500], 'strengths': [], 'weaknesses': [], 'suggestions': []} 
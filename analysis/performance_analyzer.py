"""
Performance analysis and LLM-based strategy improvement feedback.
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
import json

from llm.client import LLMClient
from config.settings import Config

logger = logging.getLogger(__name__)

class PerformanceAnalyzer:
    """Analyzes strategy performance and generates improvement feedback using LLM."""
    
    def __init__(self):
        """Initialize the performance analyzer."""
        self.llm_client = LLMClient()
        
    def analyze_performance(self, performance_metrics: Dict, strategy_dict: Dict, 
                          trades_df: pd.DataFrame, portfolio_df: pd.DataFrame,
                          market_data: pd.DataFrame) -> Dict:
        """
        Comprehensive performance analysis with LLM-generated insights.
        
        Args:
            performance_metrics: Basic performance metrics from backtesting
            strategy_dict: Original strategy configuration
            trades_df: DataFrame with trade history
            portfolio_df: DataFrame with portfolio value history
            market_data: Original market data with indicators
            
        Returns:
            Dictionary with detailed analysis and improvement suggestions
        """
        try:
            logger.info("Starting comprehensive performance analysis")
            
            # Enhanced analysis
            enhanced_metrics = self._calculate_enhanced_metrics(
                performance_metrics, trades_df, portfolio_df, market_data
            )
            
            # LLM-based weakness identification
            metrics_combined = {
                'basic_metrics': performance_metrics, 
                'enhanced_metrics': enhanced_metrics
            }
            weaknesses = self._identify_weaknesses_with_llm(
                metrics_combined, strategy_dict, trades_df
            )
            
            # Generate improvement suggestions
            improvements = self._generate_improvement_suggestions(
                metrics_combined, weaknesses, strategy_dict
            )
            
            analysis_result = {
                'basic_metrics': performance_metrics,
                'enhanced_metrics': enhanced_metrics,
                'weaknesses_identified': weaknesses,
                'improvement_suggestions': improvements,
                'strategy_analyzed': strategy_dict,
                'analysis_timestamp': pd.Timestamp.now().isoformat()
            }
            
            logger.info("Performance analysis completed successfully")
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error in performance analysis: {str(e)}")
            return {
                'error': str(e),
                'basic_metrics': performance_metrics,
                'strategy_analyzed': strategy_dict
            }
    
    def _calculate_enhanced_metrics(self, basic_metrics: Dict, trades_df: pd.DataFrame,
                                  portfolio_df: pd.DataFrame, market_data: pd.DataFrame) -> Dict:
        """Calculate additional performance metrics for deeper analysis."""
        try:
            enhanced = {}
            
            # Basic trade analysis
            if not trades_df.empty:
                enhanced['trade_frequency'] = len(trades_df) / len(market_data) * 252
                enhanced['profit_factor'] = 1.0  # Simplified
            else:
                enhanced['trade_frequency'] = 0
                enhanced['profit_factor'] = 0
            
            # Basic risk metrics
            enhanced['downside_deviation'] = basic_metrics.get('volatility', 0)
            
            return enhanced
            
        except Exception as e:
            logger.warning(f"Error calculating enhanced metrics: {str(e)}")
            return {}
    
    def _identify_weaknesses_with_llm(self, metrics: Dict, strategy_dict: Dict, trades_df: pd.DataFrame) -> Dict:
        """Use LLM to identify strategy weaknesses based on performance data."""
        try:
            analysis_prompt = self._create_weakness_analysis_prompt(metrics, strategy_dict, trades_df)
            response = self.llm_client.generate_strategy(analysis_prompt)
            weaknesses = self._parse_weakness_response(response)
            return weaknesses
            
        except Exception as e:
            logger.warning(f"Error in LLM weakness identification: {str(e)}")
            return {'error': 'Could not identify weaknesses with LLM', 'details': str(e)}
    
    def _generate_improvement_suggestions(self, metrics: Dict, weaknesses: Dict, strategy_dict: Dict) -> Dict:
        """Generate specific improvement suggestions based on identified weaknesses."""
        try:
            improvement_prompt = self._create_improvement_prompt(metrics, weaknesses, strategy_dict)
            response = self.llm_client.generate_strategy(improvement_prompt)
            improvements = self._parse_improvement_response(response)
            return improvements
            
        except Exception as e:
            logger.warning(f"Error generating improvement suggestions: {str(e)}")
            return {'error': 'Could not generate improvements', 'details': str(e)}
    
    def _create_weakness_analysis_prompt(self, metrics: Dict, strategy_dict: Dict, trades_df: pd.DataFrame) -> str:
        """Create prompt for LLM weakness analysis."""
        basic_metrics = metrics.get('basic_metrics', {})
        
        return f"""
Analyze this investment strategy's performance and identify its main weaknesses:

STRATEGY DETAILS:
Name: {strategy_dict.get('name', 'Unknown')}
Buy Conditions: {strategy_dict.get('buy_conditions', [])}
Sell Conditions: {strategy_dict.get('sell_conditions', [])}

PERFORMANCE METRICS:
- Total Return: {basic_metrics.get('total_return', 0):.1f}%
- Sharpe Ratio: {basic_metrics.get('sharpe_ratio', 0):.2f}
- Max Drawdown: {basic_metrics.get('max_drawdown', 0):.1f}%
- Win Rate: {basic_metrics.get('win_rate', 0):.1f}%
- Number of Trades: {basic_metrics.get('num_trades', 0)}

Identify the top 3 weaknesses. Respond in JSON format:
{{
    "weaknesses": [
        {{
            "weakness": "Description of weakness",
            "impact": "How it affects performance",
            "severity": "High/Medium/Low"
        }}
    ]
}}
"""
    
    def _create_improvement_prompt(self, metrics: Dict, weaknesses: Dict, strategy_dict: Dict) -> str:
        """Create prompt for improvement suggestions."""
        return f"""
Based on the strategy weaknesses, generate improvement suggestions:

ORIGINAL STRATEGY:
{json.dumps(strategy_dict, indent=2)}

WEAKNESSES:
{json.dumps(weaknesses, indent=2)}

Generate specific improvements. Respond in JSON format:
{{
    "improvements": [
        {{
            "category": "Buy Conditions/Sell Conditions/Risk Management",
            "suggested_change": "Specific improvement to make",
            "rationale": "Why this will help"
        }}
    ]
}}
"""
    
    def _parse_weakness_response(self, response: str) -> Dict:
        """Parse LLM response for weaknesses."""
        try:
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {'error': 'Could not parse LLM response', 'raw_response': response}
        except Exception as e:
            return {'error': f'Parse error: {str(e)}', 'raw_response': response}
    
    def _parse_improvement_response(self, response: str) -> Dict:
        """Parse LLM response for improvements."""
        try:
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {'error': 'Could not parse LLM response', 'raw_response': response}
        except Exception as e:
            return {'error': f'Parse error: {str(e)}', 'raw_response': response}

    def generate_feedback_for_next_iteration(self, analysis_result: Dict) -> str:
        """Generate structured feedback for the next strategy iteration."""
        try:
            improvements = analysis_result.get('improvement_suggestions', {})
            weaknesses = analysis_result.get('weaknesses_identified', {})
            metrics = analysis_result.get('basic_metrics', {})
            
            feedback = f"""
STRATEGY PERFORMANCE FEEDBACK FOR NEXT ITERATION:

CURRENT PERFORMANCE SUMMARY:
- Total Return: {metrics.get('total_return', 0):.1f}%
- Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}
- Max Drawdown: {metrics.get('max_drawdown', 0):.1f}%
- Win Rate: {metrics.get('win_rate', 0):.1f}%

CRITICAL ISSUES TO ADDRESS:
{self._format_weaknesses(weaknesses)}

SPECIFIC IMPROVEMENTS TO IMPLEMENT:
{self._format_improvements(improvements)}

FOCUS AREAS FOR NEXT STRATEGY:
1. Improve entry timing to reduce false signals
2. Add better risk management to limit drawdowns
3. Optimize position sizing for better risk-adjusted returns

Please generate a new strategy that addresses these specific issues.
"""
            return feedback
            
        except Exception as e:
            logger.warning(f"Error generating feedback: {str(e)}")
            return "Error generating feedback. Please try again."
    
    def _format_weaknesses(self, weaknesses: Dict) -> str:
        """Format weaknesses for feedback."""
        if 'weaknesses' not in weaknesses:
            return "- No specific weaknesses identified"
        
        formatted = []
        for i, weakness in enumerate(weaknesses['weaknesses'], 1):
            formatted.append(f"{i}. {weakness.get('weakness', 'Unknown issue')}")
        
        return "\n".join(formatted)
    
    def _format_improvements(self, improvements: Dict) -> str:
        """Format improvements for feedback."""
        if 'improvements' not in improvements:
            return "- No specific improvements suggested"
        
        formatted = []
        for i, improvement in enumerate(improvements['improvements'], 1):
            formatted.append(f"{i}. {improvement.get('category', 'General')}: "
                           f"{improvement.get('suggested_change', 'No suggestion')}")
        
        return "\n".join(formatted) 
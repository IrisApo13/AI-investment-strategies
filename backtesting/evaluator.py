"""
Performance evaluation and scoring for backtesting results.
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional

from config.settings import Config

logger = logging.getLogger(__name__)

class PerformanceEvaluator:
    """Evaluates and scores strategy performance."""
    
    def __init__(self):
        """Initialize the performance evaluator."""
        pass
    
    def evaluate_performance(self, 
                           performance_metrics: Dict, 
                           portfolio_history: pd.DataFrame = None) -> Dict:
        """
        Evaluate comprehensive performance metrics.
        
        Args:
            performance_metrics: Basic performance metrics from simulator
            portfolio_history: Portfolio value history DataFrame
            
        Returns:
            Enhanced performance evaluation
        """
        try:
            evaluation = performance_metrics.copy()
            
            # Calculate performance score
            evaluation['performance_score'] = self._calculate_performance_score(performance_metrics)
            
            # Risk-adjusted metrics
            evaluation['risk_adjusted_return'] = self._calculate_risk_adjusted_return(performance_metrics)
            evaluation['calmar_ratio'] = self._calculate_calmar_ratio(performance_metrics)
            evaluation['sortino_ratio'] = self._calculate_sortino_ratio(portfolio_history)
            
            # Trading efficiency metrics
            evaluation['profit_factor'] = self._calculate_profit_factor(performance_metrics)
            evaluation['trading_efficiency'] = self._calculate_trading_efficiency(performance_metrics)
            
            # Risk assessment
            evaluation['risk_level'] = self._assess_risk_level(performance_metrics)
            evaluation['consistency_score'] = self._calculate_consistency_score(portfolio_history)
            
            # Performance classification
            evaluation['performance_rating'] = self._rate_performance(evaluation['performance_score'])
            
            logger.info(f"Performance evaluation complete. Score: {evaluation['performance_score']:.1f}")
            
            return evaluation
            
        except Exception as e:
            logger.error(f"Error evaluating performance: {str(e)}")
            return performance_metrics
    
    def _calculate_performance_score(self, metrics: Dict) -> float:
        """
        Calculate overall performance score (0-100).
        
        Weighted scoring based on:
        - Total return (30%)
        - Sharpe ratio (25%)
        - Max drawdown (20%)
        - Win rate (15%)
        - Excess return over buy & hold (10%)
        """
        try:
            # Normalize metrics to 0-100 scale
            return_score = min(max(metrics.get('total_return', 0) * 10, 0), 100)  # 50% return = 100 points
            
            sharpe_score = min(max(metrics.get('sharpe_ratio', 0) * 50, 0), 100)  # Sharpe 2.0 = 100 points
            
            drawdown = abs(metrics.get('max_drawdown', 0))
            drawdown_score = max(100 - drawdown * 5, 0)  # -20% drawdown = 0 points
            
            win_rate_score = min(max(metrics.get('win_rate', 0), 0), 100)  # Direct mapping
            
            excess_return = metrics.get('excess_return', 0)
            excess_score = min(max(excess_return * 2 + 50, 0), 100)  # 0% excess = 50 points
            
            # Weighted average
            score = (
                return_score * 0.50 +
                sharpe_score * 0.30 +
                drawdown_score * 0 +
                win_rate_score * 0 +
                excess_score * 0.20
            )
            
            return round(score, 1)
            
        except Exception:
            return 0.0
    
    def _calculate_risk_adjusted_return(self, metrics: Dict) -> float:
        """Calculate risk-adjusted return metric."""
        try:
            total_return = metrics.get('total_return', 0)
            max_drawdown = abs(metrics.get('max_drawdown', 1))
            
            if max_drawdown == 0:
                return total_return
            
            return total_return / max_drawdown
            
        except Exception:
            return 0.0
    
    def _calculate_calmar_ratio(self, metrics: Dict) -> float:
        """Calculate Calmar ratio (annualized return / max drawdown)."""
        try:
            annualized_return = metrics.get('annualized_return', 0)
            max_drawdown = abs(metrics.get('max_drawdown', 1))
            
            if max_drawdown == 0:
                return annualized_return
            
            return annualized_return / max_drawdown
            
        except Exception:
            return 0.0
    
    def _calculate_sortino_ratio(self, portfolio_history: pd.DataFrame) -> float:
        """Calculate Sortino ratio (return / downside deviation)."""
        try:
            if portfolio_history is None or len(portfolio_history) < 2:
                return 0.0
            
            returns = portfolio_history['portfolio_value'].pct_change().dropna()
            
            if len(returns) == 0:
                return 0.0
            
            mean_return = returns.mean()
            downside_returns = returns[returns < 0]
            
            if len(downside_returns) == 0:
                return float('inf') if mean_return > 0 else 0.0
            
            downside_deviation = np.sqrt(np.mean(downside_returns ** 2))
            
            if downside_deviation == 0:
                return float('inf') if mean_return > 0 else 0.0
            
            # Annualize
            sortino = (mean_return * 252) / (downside_deviation * np.sqrt(252))
            
            return round(sortino, 3)
            
        except Exception:
            return 0.0
    
    def _calculate_profit_factor(self, metrics: Dict) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        try:
            # This would need trade-level data for accurate calculation
            # For now, use a proxy based on win rate and average trade return
            win_rate = metrics.get('win_rate', 50) / 100
            avg_return = metrics.get('avg_trade_return', 0) / 100
            
            if win_rate == 0 or avg_return <= 0:
                return 0.0
            
            # Rough approximation
            gross_profit = win_rate * abs(avg_return)
            gross_loss = (1 - win_rate) * abs(avg_return)
            
            if gross_loss == 0:
                return float('inf') if gross_profit > 0 else 0.0
            
            return gross_profit / gross_loss
            
        except Exception:
            return 0.0
    
    def _calculate_trading_efficiency(self, metrics: Dict) -> float:
        """Calculate trading efficiency score."""
        try:
            num_trades = metrics.get('num_trades', 0)
            total_return = metrics.get('total_return', 0)
            
            if num_trades == 0:
                return 0.0
            
            # Return per trade
            return_per_trade = total_return / num_trades
            
            # Efficiency score (higher is better, but penalize overtrading)
            if num_trades > 100:
                efficiency = return_per_trade * 0.5  # Penalty for overtrading
            else:
                efficiency = return_per_trade
            
            return round(efficiency, 2)
            
        except Exception:
            return 0.0
    
    def _assess_risk_level(self, metrics: Dict) -> str:
        """Assess risk level based on volatility and drawdown."""
        try:
            volatility = metrics.get('volatility', 0)
            max_drawdown = abs(metrics.get('max_drawdown', 0))
            
            # Risk score based on volatility and drawdown
            risk_score = volatility * 0.6 + max_drawdown * 0.4
            
            if risk_score > 30:
                return "High"
            elif risk_score > 15:
                return "Medium"
            else:
                return "Low"
                
        except Exception:
            return "Unknown"
    
    def _calculate_consistency_score(self, portfolio_history: pd.DataFrame) -> float:
        """Calculate consistency score based on return variability."""
        try:
            if portfolio_history is None or len(portfolio_history) < 30:
                return 0.0
            
            # Calculate rolling 30-day returns
            portfolio_values = portfolio_history['portfolio_value']
            rolling_returns = portfolio_values.rolling(30).apply(lambda x: (x.iloc[-1] / x.iloc[0] - 1) * 100)
            rolling_returns = rolling_returns.dropna()
            
            if len(rolling_returns) == 0:
                return 0.0
            
            # Consistency is inverse of return variability
            return_std = rolling_returns.std()
            
            if return_std == 0:
                return 100.0
            
            # Score from 0-100 (lower variability = higher score)
            consistency = max(0, 100 - return_std * 2)
            
            return round(consistency, 1)
            
        except Exception:
            return 0.0
    
    def _rate_performance(self, score: float) -> str:
        """Rate overall performance based on score."""
        if score >= 80:
            return "Excellent"
        elif score >= 65:
            return "Good"
        elif score >= 50:
            return "Average"
        elif score >= 35:
            return "Poor"
        else:
            return "Very Poor"
    
    def compare_strategies(self, evaluations: List[Dict]) -> Dict:
        """
        Compare multiple strategy evaluations.
        
        Args:
            evaluations: List of performance evaluation dictionaries
            
        Returns:
            Comparison summary
        """
        try:
            if not evaluations:
                return {}
            
            comparison = {
                'best_overall': None,
                'best_return': None,
                'best_sharpe': None,
                'lowest_risk': None,
                'rankings': []
            }
            
            # Find best performers in each category
            best_score_idx = max(range(len(evaluations)), 
                               key=lambda i: evaluations[i].get('performance_score', 0))
            comparison['best_overall'] = best_score_idx
            
            best_return_idx = max(range(len(evaluations)), 
                                key=lambda i: evaluations[i].get('total_return', 0))
            comparison['best_return'] = best_return_idx
            
            best_sharpe_idx = max(range(len(evaluations)), 
                                key=lambda i: evaluations[i].get('sharpe_ratio', 0))
            comparison['best_sharpe'] = best_sharpe_idx
            
            lowest_risk_idx = min(range(len(evaluations)), 
                                key=lambda i: abs(evaluations[i].get('max_drawdown', 100)))
            comparison['lowest_risk'] = lowest_risk_idx
            
            # Create rankings
            ranked_evaluations = sorted(enumerate(evaluations), 
                                      key=lambda x: x[1].get('performance_score', 0), 
                                      reverse=True)
            
            comparison['rankings'] = [
                {
                    'rank': i + 1,
                    'index': idx,
                    'score': eval_data.get('performance_score', 0),
                    'return': eval_data.get('total_return', 0),
                    'sharpe': eval_data.get('sharpe_ratio', 0),
                    'drawdown': eval_data.get('max_drawdown', 0)
                }
                for i, (idx, eval_data) in enumerate(ranked_evaluations)
            ]
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing strategies: {str(e)}")
            return {}
    
    def generate_performance_summary(self, evaluation: Dict) -> str:
        """Generate a human-readable performance summary."""
        try:
            summary = f"""
PERFORMANCE SUMMARY
==================

Overall Score: {evaluation.get('performance_score', 0):.1f}/100 ({evaluation.get('performance_rating', 'Unknown')})

Returns:
- Total Return: {evaluation.get('total_return', 0):.2f}%
- Annualized Return: {evaluation.get('annualized_return', 0):.2f}%
- Buy & Hold Return: {evaluation.get('buy_hold_return', 0):.2f}%
- Excess Return: {evaluation.get('excess_return', 0):.2f}%

Risk Metrics:
- Volatility: {evaluation.get('volatility', 0):.2f}%
- Max Drawdown: {evaluation.get('max_drawdown', 0):.2f}%
- Sharpe Ratio: {evaluation.get('sharpe_ratio', 0):.3f}
- Risk Level: {evaluation.get('risk_level', 'Unknown')}

Trading Activity:
- Number of Trades: {evaluation.get('num_trades', 0)}
- Win Rate: {evaluation.get('win_rate', 0):.1f}%
- Time in Market: {evaluation.get('time_in_market', 0):.1f}%
- Trading Efficiency: {evaluation.get('trading_efficiency', 0):.2f}

Final Portfolio Value: ${evaluation.get('final_portfolio_value', 0):,.2f}
Total Fees Paid: ${evaluation.get('total_fees', 0):.2f}
""".strip()
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating performance summary: {str(e)}")
            return "Error generating summary" 
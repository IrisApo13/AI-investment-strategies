"""
Portfolio simulation engine for backtesting investment strategies.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

from config.settings import Config

logger = logging.getLogger(__name__)

class PortfolioSimulator:
    """Simulates portfolio performance based on strategy signals."""
    
    def __init__(self, initial_capital: float = None):
        """Initialize the portfolio simulator."""
        self.initial_capital = initial_capital or Config.INITIAL_CAPITAL
        self.transaction_cost = Config.TRANSACTION_COST
        self.slippage = Config.SLIPPAGE
        
        # Portfolio state
        self.cash = self.initial_capital
        self.positions = {}  # ticker -> shares
        self.portfolio_value = self.initial_capital
        
        # Tracking
        self.trades = []
        self.portfolio_history = []
        self.daily_returns = []
        
    def run_backtest(self, data: pd.DataFrame, signals: pd.DataFrame, ticker: str) -> Dict:
        """Run backtest simulation and return performance metrics."""
        
        # Reset portfolio state
        self._reset_portfolio()
        
        # Initialize portfolio history
        self.portfolio_history = []
        self.trades = []
        self.daily_returns = []
        
        # Process each day
        for date in data.index:
            if date not in signals.index:
                continue
                
            current_price = data.loc[date, 'Close']
            signal_row = signals.loc[date]
            
            # Process buy signal
            if signal_row['buy_signal']:
                self._execute_buy(ticker, current_price, signal_row['position_size'], data, date)
            
            # Process sell signal
            elif signal_row['sell_signal']:
                self._execute_sell(ticker, current_price, abs(signal_row['position_size']), data, date)
            
            # Calculate portfolio value
            portfolio_value = self._calculate_portfolio_value(data.loc[date], ticker)
            self.portfolio_history.append({
                'date': date,
                'portfolio_value': portfolio_value,
                'cash': self.cash,
                'stock_value': portfolio_value - self.cash,
                'stock_price': current_price
            })
        
        # Calculate daily returns
        portfolio_values = [entry['portfolio_value'] for entry in self.portfolio_history]
        self.daily_returns = pd.Series(portfolio_values).pct_change().dropna()
        
        # Generate performance metrics
        performance = self._calculate_performance_metrics(data, ticker)
        
        return performance
    
    def _execute_buy(self, ticker: str, price: float, position_size: float, data, date):
        """Execute a buy order."""
        # Calculate trade amount
        trade_amount = self.cash * position_size
        
        # Apply slippage (buy at higher price)
        execution_price = price * (1 + self.slippage)
        
        # Calculate shares to buy (accounting for transaction costs)
        effective_amount = trade_amount * (1 - self.transaction_cost)
        shares_to_buy = effective_amount / execution_price
        
        if shares_to_buy > 0 and trade_amount <= self.cash:
            # Execute trade
            self.cash -= trade_amount
            if ticker not in self.positions:
                self.positions[ticker] = 0
            self.positions[ticker] += shares_to_buy
            
            # Record trade
            self.trades.append({
                'date': date,
                'action': 'BUY',
                'ticker': ticker,
                'shares': shares_to_buy,
                'price': execution_price,
                'amount': trade_amount,
                "metrics": data.loc[date],
                'transaction_cost': trade_amount * self.transaction_cost
            })
            
            logger.debug(f"BUY: {shares_to_buy:.2f} shares of {ticker} at ${execution_price:.2f}")
    
    def _execute_sell(self, ticker: str, price: float, position_size: float, data, date):
        """Execute a sell order."""
        if ticker not in self.positions or self.positions[ticker] <= 0:
            return
        
        # Calculate shares to sell
        shares_to_sell = min(self.positions[ticker], self.positions[ticker] * position_size)
        
        # Apply slippage (sell at lower price)
        execution_price = price * (1 - self.slippage)
        
        # Calculate proceeds
        gross_proceeds = shares_to_sell * execution_price
        net_proceeds = gross_proceeds * (1 - self.transaction_cost)
        
        # Execute trade
        self.cash += net_proceeds
        self.positions[ticker] -= shares_to_sell
        
        # Record trade
        self.trades.append({
            'date': date,
            'action': 'SELL',
            'ticker': ticker,
            'shares': shares_to_sell,
            'price': execution_price,
            'amount': gross_proceeds,
            "metrics": data.loc[date],
            'transaction_cost': gross_proceeds * self.transaction_cost
        })
        
        logger.debug(f"SELL: {shares_to_sell:.2f} shares of {ticker} at ${execution_price:.2f}")
    
    def _calculate_portfolio_value(self, current_data: pd.Series, ticker: str) -> float:
        """Calculate current portfolio value."""
        stock_value = 0
        if ticker in self.positions and self.positions[ticker] > 0:
            stock_value = self.positions[ticker] * current_data['Close']
        
        return self.cash + stock_value
    
    def _calculate_performance_metrics(self, data: pd.DataFrame, ticker: str) -> Dict:
        """Calculate comprehensive performance metrics."""
        if not self.portfolio_history:
            return {}
        
        # Basic metrics
        final_value = self.portfolio_history[-1]['portfolio_value']
        total_return = (final_value / self.initial_capital - 1) * 100
        
        # Buy and hold comparison
        start_price = data['Close'].iloc[0]
        end_price = data['Close'].iloc[-1]
        start_date = data.index[0]
        end_date = data.index[-1]
        buy_hold_return = (end_price / start_price - 1) * 100

        logger.debug(f"END PRICE: {end_price:.2f} START PRICE: {start_price:.2f} BUY_HOLD: {buy_hold_return:.2f}")
        logger.debug(f"END DATE: {end_date} START DATE: {start_date} BUY_HOLD: {buy_hold_return:.2f}")



        # Risk metrics
        if len(self.daily_returns) > 1:
            annualized_return = self.daily_returns.mean() * 252 * 100
            volatility = self.daily_returns.std() * np.sqrt(252) * 100
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        else:
            annualized_return = 0
            volatility = 0
            sharpe_ratio = 0
        
        # Drawdown calculation
        portfolio_values = pd.Series([entry['portfolio_value'] for entry in self.portfolio_history])
        peak = portfolio_values.expanding().max()
        drawdown = (portfolio_values - peak) / peak
        max_drawdown = drawdown.min() * 100
        
        # Trade analysis
        trade_returns = []
        buy_trades = [t for t in self.trades if t['action'] == 'BUY']
        sell_trades = [t for t in self.trades if t['action'] == 'SELL']
        
        for i, sell_trade in enumerate(sell_trades):
            if i < len(buy_trades):
                buy_trade = buy_trades[i]
                trade_return = (sell_trade['price'] - buy_trade['price']) / buy_trade['price'] * 100
                trade_returns.append(trade_return)
        
        winning_trades = [r for r in trade_returns if r > 0]
        win_rate = len(winning_trades) / len(trade_returns) * 100 if trade_returns else 0
        avg_trade_return = np.mean(trade_returns) if trade_returns else 0
        
        # Trading frequency
        days_in_market = len([entry for entry in self.portfolio_history if entry['stock_value'] > 0])
        time_in_market = days_in_market / len(self.portfolio_history) * 100 if self.portfolio_history else 0
        
        performance = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'buy_hold_return': buy_hold_return,
            'excess_return': total_return - buy_hold_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'num_trades': len(trade_returns),
            'avg_trade_return': avg_trade_return,
            'time_in_market': time_in_market,
            'final_portfolio_value': final_value,
            'total_fees': sum(trade['transaction_cost'] for trade in self.trades)
        }
        
        return performance
    
    def _reset_portfolio(self):
        """Reset portfolio to initial state."""
        self.cash = self.initial_capital
        self.positions = {}
        self.portfolio_value = self.initial_capital
        self.trades = []
        self.portfolio_history = []
        self.daily_returns = []
    
    def get_portfolio_history_df(self) -> pd.DataFrame:
        """Get portfolio history as DataFrame."""
        if not self.portfolio_history:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.portfolio_history)
        df.set_index('date', inplace=True)
        return df
    
    def get_trades_df(self) -> pd.DataFrame:
        """Get trades history as DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.trades)
        df.set_index('date', inplace=True)
        return df 
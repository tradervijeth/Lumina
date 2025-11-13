"""
Backtesting engine.

Copyright (c) 2025 Vijeth Ltd. All rights reserved.
Author: Vithushan Jeyapahan <finance@vijeth.com>
"""

from typing import Optional, Dict, List
from datetime import datetime

import numpy as np
import pandas as pd
from loguru import logger

from lumina.backtest.strategy import Strategy


class BacktestEngine:
    """
    Event-driven backtesting engine.

    Simulates strategy execution with realistic transaction costs,
    slippage, and portfolio constraints.
    """

    def __init__(
        self,
        strategy: Strategy,
        data: pd.DataFrame,
        commission: float = 0.001,
        slippage: float = 0.0005,
    ):
        """
        Initialize backtest engine.

        Args:
            strategy: Strategy instance to backtest
            data: Historical price data (DataFrame with DatetimeIndex)
            commission: Commission rate (fraction)
            slippage: Slippage rate (fraction)

        Example:
            >>> engine = BacktestEngine(strategy, price_data)
        """
        self.strategy = strategy
        self.data = data
        self.commission = commission
        self.slippage = slippage

        self.results = None
        self.trades = []

        logger.info(
            f"Initialized BacktestEngine for strategy '{strategy.name}' "
            f"with {len(data)} data points"
        )

    def run(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        rebalance_frequency: str = 'D',
    ) -> pd.DataFrame:
        """
        Run backtest.

        Args:
            start_date: Start date for backtest
            end_date: End date for backtest
            rebalance_frequency: Rebalancing frequency ('D', 'W', 'M')

        Returns:
            DataFrame with backtest results

        Example:
            >>> results = engine.run(start_date='2020-01-01', end_date='2023-12-31')
        """
        # Filter data by date range
        data_filtered = self.data.copy()

        if start_date:
            data_filtered = data_filtered[data_filtered.index >= pd.to_datetime(start_date)]
        if end_date:
            data_filtered = data_filtered[data_filtered.index <= pd.to_datetime(end_date)]

        logger.info(
            f"Running backtest from {data_filtered.index[0]} to {data_filtered.index[-1]}"
        )

        # Reset strategy
        self.strategy.reset()

        # Initialize tracking
        portfolio_values = []
        positions_history = []
        self.trades = []

        # Get rebalancing dates
        if rebalance_frequency == 'D':
            rebalance_dates = data_filtered.index
        else:
            rebalance_dates = data_filtered.resample(rebalance_frequency).last().index

        current_weights = pd.Series(0.0, index=data_filtered.columns)
        current_positions = {}

        for date in data_filtered.index:
            current_prices = data_filtered.loc[date]

            # Update portfolio value
            positions_value = sum(
                qty * current_prices.get(asset, 0)
                for asset, qty in current_positions.items()
            )
            portfolio_value = self.strategy.cash + positions_value

            # Check if rebalancing date
            if date in rebalance_dates:
                # Get historical data up to this point
                historical_data = data_filtered.loc[:date]

                # Get portfolio state
                portfolio_state = {
                    'positions': current_positions.copy(),
                    'cash': self.strategy.cash,
                    'portfolio_value': portfolio_value,
                    'weights': current_weights.to_dict(),
                }

                # Get target weights from strategy
                target_weights = self.strategy.on_data(historical_data, portfolio_state)

                # Execute rebalance
                trades_executed = self._rebalance(
                    current_positions,
                    target_weights,
                    current_prices,
                    portfolio_value,
                    date,
                )

                current_positions = trades_executed['new_positions']
                self.strategy.cash = trades_executed['new_cash']
                current_weights = target_weights

            # Record portfolio state
            portfolio_values.append({
                'date': date,
                'portfolio_value': portfolio_value,
                'cash': self.strategy.cash,
                'positions_value': positions_value,
            })

            positions_history.append({
                'date': date,
                **current_positions,
            })

        # Create results DataFrame
        self.results = pd.DataFrame(portfolio_values).set_index('date')

        # Calculate returns
        self.results['returns'] = self.results['portfolio_value'].pct_change()
        self.results['cumulative_returns'] = (
            (1 + self.results['returns']).cumprod() - 1
        )

        logger.info(
            f"Backtest complete. Final portfolio value: "
            f"${self.results['portfolio_value'].iloc[-1]:,.0f}"
        )

        return self.results

    def _rebalance(
        self,
        current_positions: Dict[str, float],
        target_weights: pd.Series,
        current_prices: pd.Series,
        portfolio_value: float,
        date: datetime,
    ) -> Dict:
        """
        Execute portfolio rebalancing.

        Args:
            current_positions: Current positions (asset -> quantity)
            target_weights: Target weights
            current_prices: Current prices
            portfolio_value: Total portfolio value
            date: Current date

        Returns:
            Dictionary with new positions and cash
        """
        new_positions = {}
        total_costs = 0

        # Calculate target position sizes
        for asset in target_weights.index:
            target_value = target_weights[asset] * portfolio_value
            current_price = current_prices.get(asset, 0)

            if current_price <= 0:
                continue

            # Apply slippage
            execution_price = current_price * (1 + self.slippage)

            target_quantity = target_value / execution_price

            # Calculate trade
            current_quantity = current_positions.get(asset, 0)
            trade_quantity = target_quantity - current_quantity

            if abs(trade_quantity) > 0:
                trade_value = abs(trade_quantity) * execution_price
                cost = trade_value * self.commission

                total_costs += cost

                # Record trade
                self.trades.append({
                    'date': date,
                    'asset': asset,
                    'quantity': trade_quantity,
                    'price': execution_price,
                    'value': trade_quantity * execution_price,
                    'cost': cost,
                })

            new_positions[asset] = target_quantity

        # Update cash
        new_cash = self.strategy.cash - total_costs

        return {
            'new_positions': new_positions,
            'new_cash': new_cash,
            'total_costs': total_costs,
        }

    def get_trades(self) -> pd.DataFrame:
        """
        Get trade history.

        Returns:
            DataFrame of all trades executed

        Example:
            >>> trades = engine.get_trades()
        """
        if not self.trades:
            return pd.DataFrame()

        trades_df = pd.DataFrame(self.trades)
        logger.info(f"Total trades executed: {len(trades_df)}")

        return trades_df

    def get_results(self) -> Optional[pd.DataFrame]:
        """
        Get backtest results.

        Returns:
            DataFrame with backtest results or None if not run

        Example:
            >>> results = engine.get_results()
        """
        return self.results

"""
Backtesting engine with comprehensive transaction cost modeling.

Copyright (c) 2025 Vijeth Ltd. All rights reserved.
Author: Vithushan Jeyapahan <finance@vijeth.com>
"""

from __future__ import annotations

from typing import Optional, Union
from datetime import datetime

import numpy as np
import pandas as pd
from loguru import logger

from lumina.backtest.strategy import Strategy
from lumina.backtest.costs import TransactionCostModel, RealizedCosts, BROKER_MODELS


class BacktestEngine:
    """
    Event-driven backtesting engine with realistic transaction costs.

    Features:
    - Comprehensive transaction cost modeling
    - Multiple slippage models
    - Realistic order execution
    - Detailed cost tracking and analysis

    Example:
        >>> from lumina.backtest.costs import BROKER_MODELS
        >>> engine = BacktestEngine(
        ...     strategy=my_strategy,
        ...     data=price_data,
        ...     cost_model=BROKER_MODELS['interactive_brokers'],
        ... )
    """

    def __init__(
        self,
        strategy: Strategy,
        data: pd.DataFrame,
        cost_model: Optional[Union[TransactionCostModel, str]] = None,
        volume_data: Optional[pd.DataFrame] = None,
        volatility_data: Optional[pd.DataFrame] = None,
    ):
        """
        Initialize backtest engine.

        Args:
            strategy: Strategy instance to backtest
            data: Historical price data (DataFrame with DatetimeIndex)
            cost_model: TransactionCostModel or broker name ('interactive_brokers', etc.)
                       If None, uses default model
            volume_data: Historical volume data (for slippage models)
            volatility_data: Historical volatility data (for impact models)

        Example:
            >>> engine = BacktestEngine(strategy, price_data, cost_model='interactive_brokers')
        """
        self.strategy = strategy
        self.data = data
        self.volume_data = volume_data
        self.volatility_data = volatility_data

        # Setup cost model
        if cost_model is None:
            self.cost_model = TransactionCostModel()
        elif isinstance(cost_model, str):
            if cost_model in BROKER_MODELS:
                self.cost_model = BROKER_MODELS[cost_model]
            else:
                raise ValueError(
                    f"Unknown broker model: {cost_model}. "
                    f"Available: {list(BROKER_MODELS.keys())}"
                )
        else:
            self.cost_model = cost_model

        self.results = None
        self.trades = []
        self.realized_costs = RealizedCosts()

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
        current_positions: dict[str, float],
        target_weights: pd.Series,
        current_prices: pd.Series,
        portfolio_value: float,
        date: datetime,
    ) -> dict:
        """
        Execute portfolio rebalancing with realistic transaction costs.

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
        cost_breakdown = {'commission': 0, 'spread': 0, 'slippage': 0}

        # Calculate target position sizes
        for asset in target_weights.index:
            target_value = target_weights[asset] * portfolio_value
            current_price = current_prices.get(asset, 0)

            if current_price <= 0:
                continue

            # Calculate target quantity
            target_quantity = target_value / current_price

            # Calculate trade
            current_quantity = current_positions.get(asset, 0)
            trade_quantity = target_quantity - current_quantity

            if abs(trade_quantity) > 1e-8:  # Minimum trade size
                # Determine side
                side = 'buy' if trade_quantity > 0 else 'sell'

                # Get volume and volatility if available
                volume = None
                if self.volume_data is not None and asset in self.volume_data.columns:
                    try:
                        volume = self.volume_data.loc[date, asset]
                    except (KeyError, IndexError):
                        pass

                volatility = None
                if self.volatility_data is not None and asset in self.volatility_data.columns:
                    try:
                        volatility = self.volatility_data.loc[date, asset]
                    except (KeyError, IndexError):
                        pass

                # Calculate comprehensive transaction costs
                costs = self.cost_model.calculate_total_cost(
                    price=current_price,
                    shares=abs(trade_quantity),
                    side=side,
                    volume=volume,
                    volatility=volatility,
                )

                total_costs += costs['total']
                cost_breakdown['commission'] += costs['commission']
                cost_breakdown['spread'] += costs['spread']
                cost_breakdown['slippage'] += costs['slippage']

                # Record trade
                trade_record = {
                    'date': date,
                    'asset': asset,
                    'quantity': trade_quantity,
                    'price': current_price,
                    'side': side,
                    'value': trade_quantity * current_price,
                    **costs,
                }
                self.trades.append(trade_record)

                # Record in realized costs tracker
                self.realized_costs.add_trade(
                    timestamp=date,
                    symbol=asset,
                    price=current_price,
                    shares=trade_quantity,
                    side=side,
                    costs=costs,
                )

            new_positions[asset] = target_quantity

        # Update cash
        new_cash = self.strategy.cash - total_costs

        return {
            'new_positions': new_positions,
            'new_cash': new_cash,
            'total_costs': total_costs,
            'cost_breakdown': cost_breakdown,
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

    def get_cost_analysis(self) -> dict:
        """
        Get comprehensive cost analysis.

        Returns:
            Dictionary with cost statistics and breakdowns

        Example:
            >>> cost_analysis = engine.get_cost_analysis()
            >>> print(f"Total costs: ${cost_analysis['summary']['total_costs']:.2f}")
            >>> print(f"Average cost: {cost_analysis['summary']['avg_bps']:.1f} bps")
        """
        summary = self.realized_costs.get_summary()
        by_symbol = self.realized_costs.get_costs_by_symbol()
        over_time = self.realized_costs.get_costs_over_time(freq='D')

        return {
            'summary': summary,
            'by_symbol': by_symbol,
            'over_time': over_time,
        }

    def get_performance_after_costs(self) -> dict:
        """
        Get performance metrics adjusted for transaction costs.

        Returns:
            Dictionary with gross and net performance metrics

        Example:
            >>> perf = engine.get_performance_after_costs()
            >>> print(f"Gross return: {perf['gross_return']:.2%}")
            >>> print(f"Net return: {perf['net_return']:.2%}")
            >>> print(f"Cost drag: {perf['cost_drag']:.2%}")
        """
        if self.results is None:
            raise ValueError("Backtest not run yet")

        # Calculate gross returns (without costs)
        trades_df = self.get_trades()
        if trades_df.empty:
            return {
                'gross_return': 0.0,
                'net_return': 0.0,
                'cost_drag': 0.0,
                'turnover': 0.0,
            }

        gross_returns = self.results['returns'].copy()
        total_cost = trades_df['total'].sum()

        initial_value = self.results['portfolio_value'].iloc[0]
        final_value = self.results['portfolio_value'].iloc[-1]

        gross_return = (final_value / initial_value) - 1

        # Estimate net return by adding back costs
        final_value_with_costs = final_value + total_cost
        net_return = (final_value_with_costs / initial_value) - 1

        cost_drag = net_return - gross_return

        # Calculate turnover
        total_volume = trades_df['value'].abs().sum()
        avg_portfolio_value = self.results['portfolio_value'].mean()
        turnover = total_volume / (avg_portfolio_value * len(self.results) / 252)  # Annualized

        return {
            'gross_return': gross_return,
            'net_return': net_return,
            'cost_drag': cost_drag,
            'total_costs': total_cost,
            'turnover': turnover,
            'costs_bps': (total_cost / total_volume * 10000) if total_volume > 0 else 0,
        }

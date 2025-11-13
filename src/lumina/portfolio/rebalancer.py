"""
Portfolio rebalancing strategies.

Copyright (c) 2025 Vijeth Ltd. All rights reserved.
Author: Vithushan Jeyapahan <finance@vijeth.com>
"""

from typing import Dict, List, Optional
from datetime import datetime

import numpy as np
import pandas as pd
from loguru import logger

from lumina.utils.validation import validate_weights


class PortfolioRebalancer:
    """
    Portfolio rebalancing engine.

    Implements various rebalancing strategies:
    - Periodic rebalancing (calendar-based)
    - Threshold rebalancing (drift-based)
    - Volatility-targeted rebalancing
    """

    def __init__(
        self,
        target_weights: pd.Series,
        transaction_cost: float = 0.001,
    ):
        """
        Initialize rebalancer.

        Args:
            target_weights: Target portfolio weights
            transaction_cost: Transaction cost (as fraction, e.g., 0.001 = 10bps)

        Example:
            >>> target = pd.Series({'AAPL': 0.6, 'GOOGL': 0.4})
            >>> rebalancer = PortfolioRebalancer(target)
        """
        self.target_weights = validate_weights(target_weights)
        self.transaction_cost = transaction_cost

        logger.info(
            f"Initialized PortfolioRebalancer with {len(target_weights)} assets, "
            f"transaction cost {transaction_cost:.2%}"
        )

    def calculate_drift(
        self,
        current_weights: pd.Series,
    ) -> pd.Series:
        """
        Calculate drift from target weights.

        Args:
            current_weights: Current portfolio weights

        Returns:
            Series showing drift for each asset

        Example:
            >>> drift = rebalancer.calculate_drift(current_weights)
        """
        drift = current_weights - self.target_weights

        max_drift = drift.abs().max()
        logger.debug(f"Maximum weight drift: {max_drift:.2%}")

        return drift

    def should_rebalance_threshold(
        self,
        current_weights: pd.Series,
        threshold: float = 0.05,
    ) -> bool:
        """
        Check if rebalancing is needed based on drift threshold.

        Args:
            current_weights: Current portfolio weights
            threshold: Maximum allowed drift (e.g., 0.05 = 5%)

        Returns:
            True if rebalancing is needed

        Example:
            >>> needs_rebalance = rebalancer.should_rebalance_threshold(current, 0.05)
        """
        drift = self.calculate_drift(current_weights)
        max_drift = drift.abs().max()

        should_rebalance = max_drift > threshold

        if should_rebalance:
            logger.info(f"Rebalancing triggered: drift {max_drift:.2%} > threshold {threshold:.2%}")
        else:
            logger.debug(f"No rebalancing needed: drift {max_drift:.2%} <= threshold {threshold:.2%}")

        return should_rebalance

    def calculate_trades(
        self,
        current_weights: pd.Series,
        portfolio_value: float,
    ) -> pd.DataFrame:
        """
        Calculate required trades to rebalance.

        Args:
            current_weights: Current portfolio weights
            portfolio_value: Total portfolio value

        Returns:
            DataFrame with trade details

        Example:
            >>> trades = rebalancer.calculate_trades(current_weights, 1000000)
        """
        # Align current weights with target
        current_aligned = current_weights.reindex(self.target_weights.index, fill_value=0)

        # Calculate trade amounts
        weight_changes = self.target_weights - current_aligned
        trade_values = weight_changes * portfolio_value

        # Calculate transaction costs
        costs = trade_values.abs() * self.transaction_cost

        trades_df = pd.DataFrame({
            'current_weight': current_aligned,
            'target_weight': self.target_weights,
            'weight_change': weight_changes,
            'trade_value': trade_values,
            'transaction_cost': costs,
        })

        total_cost = costs.sum()
        total_turnover = trade_values.abs().sum()

        logger.info(
            f"Rebalance requires ${total_turnover:,.0f} turnover, "
            f"${total_cost:,.0f} in costs ({total_cost/portfolio_value:.2%})"
        )

        return trades_df

    def optimal_rebalance_with_costs(
        self,
        current_weights: pd.Series,
        portfolio_value: float,
        min_trade_size: float = 100,
    ) -> pd.Series:
        """
        Calculate optimal rebalance considering transaction costs.

        Uses a simple heuristic: only trade if benefit > cost.

        Args:
            current_weights: Current portfolio weights
            portfolio_value: Total portfolio value
            min_trade_size: Minimum trade size in dollars

        Returns:
            Optimal new weights after cost-aware rebalancing

        Example:
            >>> new_weights = rebalancer.optimal_rebalance_with_costs(current, 1000000)
        """
        trades_df = self.calculate_trades(current_weights, portfolio_value)

        # Filter trades below minimum size
        significant_trades = trades_df[trades_df['trade_value'].abs() > min_trade_size]

        # Start with current weights
        new_weights = current_weights.copy()

        # Apply significant trades
        for asset in significant_trades.index:
            new_weights[asset] = self.target_weights[asset]

        # Renormalize weights
        new_weights = new_weights / new_weights.sum()

        n_trades = len(significant_trades)
        logger.info(f"Executing {n_trades} trades above ${min_trade_size} threshold")

        return new_weights

    def simulate_periodic_rebalancing(
        self,
        returns: pd.DataFrame,
        rebalance_frequency: str = 'M',
        initial_value: float = 1000000,
    ) -> pd.DataFrame:
        """
        Simulate periodic rebalancing strategy.

        Args:
            returns: Historical returns DataFrame
            rebalance_frequency: Rebalancing frequency ('D', 'W', 'M', 'Q', 'Y')
            initial_value: Initial portfolio value

        Returns:
            DataFrame with portfolio value over time

        Example:
            >>> results = rebalancer.simulate_periodic_rebalancing(returns, 'M')
        """
        # Align returns with target weights
        returns_aligned = returns[self.target_weights.index]

        portfolio_values = []
        current_weights = self.target_weights.copy()
        portfolio_value = initial_value

        # Get rebalancing dates
        rebalance_dates = returns_aligned.resample(rebalance_frequency).last().index

        for date, daily_return in returns_aligned.iterrows():
            # Update portfolio value with returns
            portfolio_return = (current_weights * daily_return).sum()
            portfolio_value *= (1 + portfolio_return)

            # Update weights due to price changes
            weight_multipliers = 1 + daily_return
            current_weights = current_weights * weight_multipliers
            current_weights = current_weights / current_weights.sum()

            # Rebalance if scheduled
            if date in rebalance_dates:
                # Calculate and deduct transaction costs
                trades = self.calculate_trades(current_weights, portfolio_value)
                total_costs = trades['transaction_cost'].sum()
                portfolio_value -= total_costs

                # Rebalance to target weights
                current_weights = self.target_weights.copy()

                logger.debug(f"Rebalanced on {date}, cost: ${total_costs:,.0f}")

            portfolio_values.append({
                'date': date,
                'portfolio_value': portfolio_value,
                'portfolio_return': portfolio_return,
            })

        results_df = pd.DataFrame(portfolio_values).set_index('date')

        total_return = (portfolio_value / initial_value) - 1
        logger.info(
            f"Simulation complete: {rebalance_frequency} rebalancing, "
            f"total return {total_return:.2%}"
        )

        return results_df

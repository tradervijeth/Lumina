"""
Base strategy class for backtesting.

Copyright (c) 2025 Vijeth Ltd. All rights reserved.
Author: Vithushan Jeyapahan <finance@vijeth.com>
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Any
from datetime import datetime

import pandas as pd
from loguru import logger


class Strategy(ABC):
    """
    Abstract base class for trading strategies.

    All strategy implementations should inherit from this class and implement
    the on_data() method which generates trading signals.
    """

    def __init__(
        self,
        name: str,
        initial_capital: float = 1000000,
        commission: float = 0.001,
    ):
        """
        Initialize strategy.

        Args:
            name: Strategy name
            initial_capital: Initial capital
            commission: Commission rate (fraction, e.g., 0.001 = 10bps)

        Example:
            >>> class MyStrategy(Strategy):
            ...     def on_data(self, data, portfolio):
            ...         return signals
            >>> strategy = MyStrategy("My Strategy")
        """
        self.name = name
        self.initial_capital = initial_capital
        self.commission = commission

        self.current_positions = {}
        self.cash = initial_capital
        self.portfolio_value = initial_capital

        logger.info(
            f"Initialized strategy '{name}' with ${initial_capital:,.0f} capital, "
            f"{commission:.2%} commission"
        )

    @abstractmethod
    def on_data(
        self,
        data: pd.DataFrame,
        portfolio: Dict[str, Any],
    ) -> pd.Series:
        """
        Generate trading signals based on new data.

        This method is called for each time step during backtesting.

        Args:
            data: Historical data up to current time
            portfolio: Current portfolio state with keys:
                      'positions', 'cash', 'portfolio_value', 'weights'

        Returns:
            Series of target weights for each asset

        Raises:
            NotImplementedError: This method must be implemented by subclasses

        Example:
            >>> def on_data(self, data, portfolio):
            ...     # Simple momentum strategy
            ...     returns = data['close'].pct_change(20)
            ...     signals = (returns > 0).astype(float)
            ...     return signals / signals.sum()
        """
        raise NotImplementedError("Subclasses must implement on_data()")

    def get_portfolio_state(self) -> Dict[str, Any]:
        """
        Get current portfolio state.

        Returns:
            Dictionary with portfolio information

        Example:
            >>> state = strategy.get_portfolio_state()
        """
        total_value = self.cash

        positions_value = {}
        for asset, quantity in self.current_positions.items():
            # This will be updated by backtest engine with current prices
            positions_value[asset] = quantity

        weights = {}
        if self.portfolio_value > 0:
            for asset in self.current_positions:
                weights[asset] = positions_value.get(asset, 0) / self.portfolio_value

        return {
            'positions': self.current_positions.copy(),
            'cash': self.cash,
            'portfolio_value': self.portfolio_value,
            'weights': weights,
        }

    def reset(self) -> None:
        """
        Reset strategy to initial state.

        Example:
            >>> strategy.reset()
        """
        self.current_positions = {}
        self.cash = self.initial_capital
        self.portfolio_value = self.initial_capital

        logger.debug(f"Reset strategy '{self.name}' to initial state")


class MomentumStrategy(Strategy):
    """
    Simple momentum strategy implementation.

    Buys assets with positive momentum and sells those with negative momentum.
    """

    def __init__(
        self,
        lookback_period: int = 20,
        top_n: int = 5,
        **kwargs,
    ):
        """
        Initialize momentum strategy.

        Args:
            lookback_period: Lookback period for momentum calculation
            top_n: Number of top assets to hold
            **kwargs: Additional arguments for Strategy base class

        Example:
            >>> strategy = MomentumStrategy(lookback_period=20, top_n=5)
        """
        super().__init__(name="Momentum Strategy", **kwargs)
        self.lookback_period = lookback_period
        self.top_n = top_n

    def on_data(
        self,
        data: pd.DataFrame,
        portfolio: Dict[str, Any],
    ) -> pd.Series:
        """
        Generate momentum-based signals.

        Args:
            data: Historical price data
            portfolio: Current portfolio state

        Returns:
            Target weights
        """
        if len(data) < self.lookback_period:
            # Not enough data, return equal weights
            return pd.Series(
                1.0 / len(data.columns),
                index=data.columns,
            )

        # Calculate momentum
        returns = data.pct_change(self.lookback_period).iloc[-1]

        # Select top N assets
        top_assets = returns.nlargest(self.top_n)

        # Equal weight among top assets
        weights = pd.Series(0.0, index=data.columns)
        weights[top_assets.index] = 1.0 / self.top_n

        return weights


class MeanReversionStrategy(Strategy):
    """
    Mean reversion strategy implementation.

    Buys oversold assets and sells overbought assets based on z-score.
    """

    def __init__(
        self,
        lookback_period: int = 20,
        entry_threshold: float = 2.0,
        **kwargs,
    ):
        """
        Initialize mean reversion strategy.

        Args:
            lookback_period: Lookback period for mean/std calculation
            entry_threshold: Z-score threshold for entry signals
            **kwargs: Additional arguments for Strategy base class

        Example:
            >>> strategy = MeanReversionStrategy(lookback_period=20)
        """
        super().__init__(name="Mean Reversion Strategy", **kwargs)
        self.lookback_period = lookback_period
        self.entry_threshold = entry_threshold

    def on_data(
        self,
        data: pd.DataFrame,
        portfolio: Dict[str, Any],
    ) -> pd.Series:
        """
        Generate mean reversion signals.

        Args:
            data: Historical price data
            portfolio: Current portfolio state

        Returns:
            Target weights
        """
        if len(data) < self.lookback_period:
            return pd.Series(0.0, index=data.columns)

        # Calculate z-scores
        recent_data = data.iloc[-self.lookback_period:]
        mean = recent_data.mean()
        std = recent_data.std()

        current_prices = data.iloc[-1]
        z_scores = (current_prices - mean) / std

        # Generate signals: buy oversold, sell overbought
        signals = pd.Series(0.0, index=data.columns)

        # Buy oversold (z-score < -threshold)
        oversold = z_scores < -self.entry_threshold
        signals[oversold] = 1.0

        # Normalize to equal weight
        if signals.sum() > 0:
            signals = signals / signals.sum()

        return signals

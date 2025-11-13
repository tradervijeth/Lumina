"""
Position sizing algorithms.

Copyright (c) 2025 Vijeth Ltd. All rights reserved.
Author: Vithushan Jeyapahan <finance@vijeth.com>
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


class PositionSizer:
    """
    Calculate position sizes using various risk management techniques.

    Methods:
    - Fixed fraction
    - Kelly criterion
    - Volatility targeting
    - Risk parity
    """

    @staticmethod
    def fixed_fraction(
        capital: float,
        fraction: float = 0.02,
        price: float = 100.0,
    ) -> int:
        """
        Fixed fraction position sizing.

        Position = Capital * Fraction / Price

        Args:
            capital: Total capital
            fraction: Fraction of capital to risk (e.g., 0.02 = 2%)
            price: Current asset price

        Returns:
            Number of shares to buy

        Example:
            >>> shares = PositionSizer.fixed_fraction(100000, 0.02, 150.0)
        """
        position_value = capital * fraction
        shares = int(position_value / price)

        logger.debug(f"Fixed fraction: ${position_value:.2f} -> {shares} shares")

        return shares

    @staticmethod
    def kelly_criterion(
        win_rate: float,
        win_loss_ratio: float,
        capital: float,
        price: float,
        kelly_fraction: float = 0.25,
    ) -> int:
        """
        Kelly criterion position sizing.

        Kelly% = (p * b - q) / b
        where:
            p = win probability
            q = loss probability (1 - p)
            b = win/loss ratio

        Args:
            win_rate: Probability of winning trade
            win_loss_ratio: Average win / average loss
            capital: Total capital
            price: Current asset price
            kelly_fraction: Fraction of full Kelly to use (conservative)

        Returns:
            Number of shares to buy

        Example:
            >>> shares = PositionSizer.kelly_criterion(0.55, 1.5, 100000, 150.0)
        """
        p = win_rate
        q = 1 - win_rate
        b = win_loss_ratio

        # Full Kelly
        kelly_pct = (p * b - q) / b

        # Apply fraction for safety
        kelly_pct *= kelly_fraction

        # Ensure non-negative
        kelly_pct = max(0, min(kelly_pct, 1.0))

        position_value = capital * kelly_pct
        shares = int(position_value / price)

        logger.debug(f"Kelly criterion: {kelly_pct:.2%} -> {shares} shares")

        return shares

    @staticmethod
    def volatility_target(
        capital: float,
        target_vol: float,
        asset_vol: float,
        price: float,
        lookback: int = 20,
    ) -> int:
        """
        Volatility targeting position sizing.

        Position = (Target Vol / Asset Vol) * Capital / Price

        Args:
            capital: Total capital
            target_vol: Target portfolio volatility (annualized)
            asset_vol: Asset volatility (annualized)
            price: Current asset price
            lookback: Lookback period for volatility calculation

        Returns:
            Number of shares to buy

        Example:
            >>> shares = PositionSizer.volatility_target(100000, 0.15, 0.25, 150.0)
        """
        if asset_vol <= 0:
            logger.warning("Asset volatility is zero or negative")
            return 0

        vol_scalar = target_vol / asset_vol
        position_value = capital * vol_scalar
        shares = int(position_value / price)

        logger.debug(f"Vol targeting: scalar={vol_scalar:.2f} -> {shares} shares")

        return shares

    @staticmethod
    def risk_based(
        capital: float,
        risk_per_trade: float,
        stop_loss_pct: float,
        price: float,
    ) -> int:
        """
        Risk-based position sizing.

        Position = (Capital * Risk%) / (Price * Stop Loss%)

        Args:
            capital: Total capital
            risk_per_trade: Risk per trade as fraction (e.g., 0.01 = 1%)
            stop_loss_pct: Stop loss as fraction (e.g., 0.05 = 5%)
            price: Current asset price

        Returns:
            Number of shares to buy

        Example:
            >>> shares = PositionSizer.risk_based(100000, 0.01, 0.05, 150.0)
        """
        if stop_loss_pct <= 0:
            logger.warning("Stop loss percentage is zero or negative")
            return 0

        risk_amount = capital * risk_per_trade
        shares = int(risk_amount / (price * stop_loss_pct))

        logger.debug(f"Risk-based: risk=${risk_amount:.2f} -> {shares} shares")

        return shares

    @staticmethod
    def calculate_portfolio_positions(
        weights: pd.Series,
        capital: float,
        prices: pd.Series,
        method: str = "proportional",
        **kwargs,
    ) -> pd.Series:
        """
        Calculate number of shares for entire portfolio.

        Args:
            weights: Target portfolio weights
            capital: Total capital
            prices: Current prices for each asset
            method: Position sizing method
            **kwargs: Additional arguments for sizing method

        Returns:
            Series of share quantities

        Example:
            >>> positions = PositionSizer.calculate_portfolio_positions(
            ...     weights, 100000, prices
            ... )
        """
        positions = pd.Series(index=weights.index, dtype=int)

        for asset in weights.index:
            if asset not in prices.index:
                logger.warning(f"Price not available for {asset}")
                positions[asset] = 0
                continue

            weight = weights[asset]
            price = prices[asset]
            allocation = capital * weight

            if method == "proportional":
                shares = int(allocation / price)
            else:
                # Use other methods
                shares = 0

            positions[asset] = shares

        logger.info(f"Calculated positions for {len(positions)} assets")

        return positions

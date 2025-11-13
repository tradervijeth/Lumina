"""
Risk limits and constraints.

Copyright (c) 2025 Vijeth Ltd. All rights reserved.
Author: Vithushan Jeyapahan <finance@vijeth.com>
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from loguru import logger


class RiskLimits:
    """Enforce risk limits on portfolio positions."""

    def __init__(
        self,
        max_position_size: float = 0.10,
        max_sector_exposure: float = 0.40,
        max_leverage: float = 1.0,
        max_drawdown: float = 0.20,
    ):
        """
        Initialize risk limits.

        Args:
            max_position_size: Max weight per asset (e.g., 0.10 = 10%)
            max_sector_exposure: Max exposure per sector
            max_leverage: Maximum leverage allowed
            max_drawdown: Max acceptable drawdown before stopping

        Example:
            >>> limits = RiskLimits(max_position_size=0.05, max_drawdown=0.15)
        """
        self.max_position_size = max_position_size
        self.max_sector_exposure = max_sector_exposure
        self.max_leverage = max_leverage
        self.max_drawdown = max_drawdown

        logger.info(f"Initialized risk limits: pos={max_position_size:.1%}, dd={max_drawdown:.1%}")

    def enforce_position_limits(self, weights: pd.Series) -> pd.Series:
        """
        Enforce position size limits.

        Args:
            weights: Portfolio weights

        Returns:
            Adjusted weights

        Example:
            >>> adjusted = limits.enforce_position_limits(weights)
        """
        # Clip weights
        clipped = weights.clip(-self.max_position_size, self.max_position_size)

        # Renormalize
        if clipped.sum() != 0:
            clipped = clipped / clipped.sum()

        violations = (np.abs(weights) > self.max_position_size).sum()
        if violations > 0:
            logger.warning(f"Clipped {violations} positions exceeding limits")

        return clipped

    def check_drawdown(self, returns: pd.Series) -> bool:
        """
        Check if drawdown limit breached.

        Args:
            returns: Historical returns

        Returns:
            True if within limits, False if breached

        Example:
            >>> ok = limits.check_drawdown(returns)
        """
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max

        max_dd = abs(drawdown.min())

        if max_dd > self.max_drawdown:
            logger.error(f"Drawdown limit breached: {max_dd:.2%} > {self.max_drawdown:.2%}")
            return False

        return True

    def check_leverage(self, weights: pd.Series) -> bool:
        """
        Check if leverage within limits.

        Args:
            weights: Portfolio weights

        Returns:
            True if within limits

        Example:
            >>> ok = limits.check_leverage(weights)
        """
        leverage = weights.abs().sum()

        if leverage > self.max_leverage:
            logger.error(f"Leverage limit breached: {leverage:.2f}x > {self.max_leverage:.2f}x")
            return False

        return True

"""
Value at Risk (VaR) calculations.

Copyright (c) 2025 Vijeth Ltd. All rights reserved.
Author: Vithushan Jeyapahan <finance@vijeth.com>
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


class VaRCalculator:
    """Calculate Value at Risk and Expected Shortfall."""

    @staticmethod
    def historical_var(
        returns: pd.Series,
        confidence: float = 0.95,
        portfolio_value: float = 1000000,
    ) -> float:
        """
        Historical VaR calculation.

        Args:
            returns: Historical returns
            confidence: Confidence level (e.g., 0.95 = 95%)
            portfolio_value: Portfolio value

        Returns:
            VaR in dollars

        Example:
            >>> var_95 = VaRCalculator.historical_var(returns, 0.95)
        """
        percentile = (1 - confidence) * 100
        var_return = np.percentile(returns.dropna(), percentile)
        var_dollar = abs(var_return * portfolio_value)

        logger.debug(f"Historical VaR ({confidence:.0%}): ${var_dollar:,.0f}")

        return var_dollar

    @staticmethod
    def parametric_var(
        returns: pd.Series,
        confidence: float = 0.95,
        portfolio_value: float = 1000000,
    ) -> float:
        """
        Parametric VaR (assumes normal distribution).

        Args:
            returns: Historical returns
            confidence: Confidence level
            portfolio_value: Portfolio value

        Returns:
            VaR in dollars

        Example:
            >>> var_95 = VaRCalculator.parametric_var(returns, 0.95)
        """
        from scipy.stats import norm

        mean = returns.mean()
        std = returns.std()

        z_score = norm.ppf(1 - confidence)
        var_return = abs(mean + z_score * std)
        var_dollar = var_return * portfolio_value

        logger.debug(f"Parametric VaR ({confidence:.0%}): ${var_dollar:,.0f}")

        return var_dollar

    @staticmethod
    def expected_shortfall(
        returns: pd.Series,
        confidence: float = 0.95,
        portfolio_value: float = 1000000,
    ) -> float:
        """
        Expected Shortfall (Conditional VaR).

        Average loss beyond VaR threshold.

        Args:
            returns: Historical returns
            confidence: Confidence level
            portfolio_value: Portfolio value

        Returns:
            Expected Shortfall in dollars

        Example:
            >>> es_95 = VaRCalculator.expected_shortfall(returns, 0.95)
        """
        percentile = (1 - confidence) * 100
        var_return = np.percentile(returns.dropna(), percentile)

        # Average of returns worse than VaR
        tail_returns = returns[returns <= var_return]

        if len(tail_returns) == 0:
            return 0.0

        es_return = abs(tail_returns.mean())
        es_dollar = es_return * portfolio_value

        logger.debug(f"Expected Shortfall ({confidence:.0%}): ${es_dollar:,.0f}")

        return es_dollar

    @staticmethod
    def monte_carlo_var(
        returns: pd.Series,
        confidence: float = 0.95,
        portfolio_value: float = 1000000,
        n_simulations: int = 10000,
    ) -> float:
        """
        Monte Carlo VaR simulation.

        Args:
            returns: Historical returns
            confidence: Confidence level
            portfolio_value: Portfolio value
            n_simulations: Number of simulations

        Returns:
            VaR in dollars

        Example:
            >>> var_95 = VaRCalculator.monte_carlo_var(returns, n_simulations=10000)
        """
        mean = returns.mean()
        std = returns.std()

        # Simulate returns
        simulated_returns = np.random.normal(mean, std, n_simulations)

        percentile = (1 - confidence) * 100
        var_return = abs(np.percentile(simulated_returns, percentile))
        var_dollar = var_return * portfolio_value

        logger.debug(f"Monte Carlo VaR ({confidence:.0%}, {n_simulations} sims): ${var_dollar:,.0f}")

        return var_dollar

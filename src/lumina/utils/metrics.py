"""
Performance metrics calculations.

Copyright (c) 2025 Vijeth Ltd. All rights reserved.
Author: Vithushan Jeyapahan <finance@vijeth.com>
"""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd
from loguru import logger


def calculate_sharpe_ratio(
    returns: Union[pd.Series, np.ndarray],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Calculate annualized Sharpe ratio.

    The Sharpe ratio measures risk-adjusted return:
        SR = (E[R] - Rf) / σ

    where:
        E[R] = expected return
        Rf = risk-free rate
        σ = standard deviation of returns

    Args:
        returns: Historical returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year (252 for daily, 12 for monthly)

    Returns:
        Annualized Sharpe ratio

    Example:
        >>> returns = pd.Series([0.01, 0.02, -0.01, 0.015])
        >>> sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.02)
    """
    if isinstance(returns, pd.Series):
        returns = returns.values

    if len(returns) == 0:
        raise ValueError("Returns array is empty")

    excess_returns = returns - (risk_free_rate / periods_per_year)
    mean_excess = np.mean(excess_returns)
    std_excess = np.std(excess_returns, ddof=1)

    if std_excess == 0:
        logger.warning("Standard deviation is zero, returning 0.0 for Sharpe ratio")
        return 0.0

    sharpe_ratio = mean_excess / std_excess * np.sqrt(periods_per_year)
    logger.debug(f"Calculated Sharpe ratio: {sharpe_ratio:.4f}")

    return float(sharpe_ratio)


def calculate_sortino_ratio(
    returns: Union[pd.Series, np.ndarray],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Calculate annualized Sortino ratio.

    The Sortino ratio is similar to Sharpe but only penalizes downside volatility:
        Sortino = (E[R] - Rf) / σ_downside

    where σ_downside is the standard deviation of negative returns only.

    Args:
        returns: Historical returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year (252 for daily, 12 for monthly)

    Returns:
        Annualized Sortino ratio

    Example:
        >>> returns = pd.Series([0.01, 0.02, -0.01, 0.015])
        >>> sortino = calculate_sortino_ratio(returns, risk_free_rate=0.02)
    """
    if isinstance(returns, pd.Series):
        returns = returns.values

    if len(returns) == 0:
        raise ValueError("Returns array is empty")

    excess_returns = returns - (risk_free_rate / periods_per_year)
    mean_excess = np.mean(excess_returns)

    downside_returns = excess_returns[excess_returns < 0]
    if len(downside_returns) == 0:
        logger.warning("No negative returns, returning inf for Sortino ratio")
        return float('inf')

    downside_std = np.std(downside_returns, ddof=1)

    if downside_std == 0:
        logger.warning("Downside deviation is zero, returning inf for Sortino ratio")
        return float('inf')

    sortino_ratio = mean_excess / downside_std * np.sqrt(periods_per_year)
    logger.debug(f"Calculated Sortino ratio: {sortino_ratio:.4f}")

    return float(sortino_ratio)


def calculate_max_drawdown(
    returns: Union[pd.Series, np.ndarray],
) -> float:
    """
    Calculate maximum drawdown.

    Maximum drawdown is the largest peak-to-trough decline:
        MDD = max((Peak - Trough) / Peak)

    Args:
        returns: Historical returns

    Returns:
        Maximum drawdown (positive value)

    Example:
        >>> returns = pd.Series([0.01, 0.02, -0.05, 0.03])
        >>> mdd = calculate_max_drawdown(returns)
    """
    if isinstance(returns, pd.Series):
        returns = returns.values

    if len(returns) == 0:
        raise ValueError("Returns array is empty")

    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max

    max_drawdown = abs(np.min(drawdown))
    logger.debug(f"Calculated maximum drawdown: {max_drawdown:.4%}")

    return float(max_drawdown)


def calculate_calmar_ratio(
    returns: Union[pd.Series, np.ndarray],
    periods_per_year: int = 252,
) -> float:
    """
    Calculate Calmar ratio.

    The Calmar ratio is the annualized return divided by maximum drawdown:
        Calmar = Annualized Return / Maximum Drawdown

    Args:
        returns: Historical returns
        periods_per_year: Number of periods per year (252 for daily, 12 for monthly)

    Returns:
        Calmar ratio

    Example:
        >>> returns = pd.Series([0.01, 0.02, -0.01, 0.015])
        >>> calmar = calculate_calmar_ratio(returns)
    """
    if isinstance(returns, pd.Series):
        returns = returns.values

    if len(returns) == 0:
        raise ValueError("Returns array is empty")

    annualized_return = np.mean(returns) * periods_per_year
    max_dd = calculate_max_drawdown(returns)

    if max_dd == 0:
        logger.warning("Maximum drawdown is zero, returning inf for Calmar ratio")
        return float('inf')

    calmar_ratio = annualized_return / max_dd
    logger.debug(f"Calculated Calmar ratio: {calmar_ratio:.4f}")

    return float(calmar_ratio)

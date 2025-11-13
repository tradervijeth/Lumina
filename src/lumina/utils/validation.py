"""
Data validation utilities.

Copyright (c) 2025 Vijeth Ltd. All rights reserved.
Author: Vithushan Jeyapahan <finance@vijeth.com>
"""

from typing import Union

import numpy as np
import pandas as pd
from loguru import logger


def validate_returns(
    returns: Union[pd.DataFrame, pd.Series, np.ndarray],
    allow_nan: bool = False,
) -> pd.DataFrame:
    """
    Validate and standardize returns data.

    Args:
        returns: Historical returns data
        allow_nan: Whether to allow NaN values

    Returns:
        Validated DataFrame of returns

    Raises:
        ValueError: If returns data is invalid

    Example:
        >>> returns = pd.DataFrame({'AAPL': [0.01, 0.02], 'GOOGL': [0.015, 0.012]})
        >>> validated = validate_returns(returns)
    """
    if isinstance(returns, np.ndarray):
        returns = pd.DataFrame(returns)
    elif isinstance(returns, pd.Series):
        returns = returns.to_frame()
    elif not isinstance(returns, pd.DataFrame):
        raise ValueError("Returns must be a DataFrame, Series, or numpy array")

    if returns.empty:
        raise ValueError("Returns data is empty")

    if not allow_nan and returns.isna().any().any():
        logger.warning("Returns contain NaN values")
        raise ValueError("Returns contain NaN values. Set allow_nan=True to proceed.")

    if not np.isfinite(returns.select_dtypes(include=[np.number])).all().all():
        raise ValueError("Returns contain infinite values")

    logger.debug(f"Validated returns with shape {returns.shape}")
    return returns


def validate_weights(
    weights: Union[pd.Series, dict, np.ndarray],
    tolerance: float = 1e-6,
) -> pd.Series:
    """
    Validate portfolio weights.

    Args:
        weights: Portfolio weights
        tolerance: Tolerance for sum of weights to equal 1.0

    Returns:
        Validated Series of weights

    Raises:
        ValueError: If weights are invalid

    Example:
        >>> weights = pd.Series({'AAPL': 0.6, 'GOOGL': 0.4})
        >>> validated = validate_weights(weights)
    """
    if isinstance(weights, dict):
        weights = pd.Series(weights)
    elif isinstance(weights, np.ndarray):
        weights = pd.Series(weights)
    elif not isinstance(weights, pd.Series):
        raise ValueError("Weights must be a Series, dict, or numpy array")

    if weights.empty:
        raise ValueError("Weights are empty")

    if (weights < 0).any():
        logger.warning("Weights contain negative values (short positions)")

    if not np.isfinite(weights).all():
        raise ValueError("Weights contain non-finite values")

    weight_sum = weights.sum()
    if abs(weight_sum - 1.0) > tolerance:
        raise ValueError(
            f"Weights sum to {weight_sum:.6f}, expected 1.0 (tolerance: {tolerance})"
        )

    logger.debug(f"Validated {len(weights)} weights summing to {weight_sum:.6f}")
    return weights


def validate_covariance_matrix(
    cov_matrix: Union[pd.DataFrame, np.ndarray],
) -> pd.DataFrame:
    """
    Validate covariance matrix.

    Args:
        cov_matrix: Covariance matrix

    Returns:
        Validated DataFrame covariance matrix

    Raises:
        ValueError: If covariance matrix is invalid

    Example:
        >>> cov = pd.DataFrame([[0.04, 0.01], [0.01, 0.09]])
        >>> validated = validate_covariance_matrix(cov)
    """
    if isinstance(cov_matrix, np.ndarray):
        cov_matrix = pd.DataFrame(cov_matrix)
    elif not isinstance(cov_matrix, pd.DataFrame):
        raise ValueError("Covariance matrix must be a DataFrame or numpy array")

    if cov_matrix.shape[0] != cov_matrix.shape[1]:
        raise ValueError("Covariance matrix must be square")

    if not np.allclose(cov_matrix, cov_matrix.T):
        raise ValueError("Covariance matrix must be symmetric")

    eigenvalues = np.linalg.eigvals(cov_matrix.values)
    if (eigenvalues < -1e-8).any():
        raise ValueError("Covariance matrix must be positive semi-definite")

    logger.debug(f"Validated {cov_matrix.shape[0]}x{cov_matrix.shape[1]} covariance matrix")
    return cov_matrix

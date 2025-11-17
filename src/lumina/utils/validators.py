"""
Comprehensive data validation for quantitative finance.

Copyright (c) 2025 Vijeth Ltd. All rights reserved.
Author: Vithushan Jeyapahan <finance@vijeth.com>
"""

from __future__ import annotations

from typing import Optional, Union
import warnings

import pandas as pd
import numpy as np
from loguru import logger


class ValidationError(Exception):
    """Raised when validation fails."""
    pass


class Validators:
    """
    Comprehensive validators for financial data and parameters.

    All validation methods raise ValidationError on failure and return
    the validated (and potentially cleaned) data on success.
    """

    @staticmethod
    def validate_returns(
        returns: Union[pd.Series, pd.DataFrame],
        min_periods: int = 20,
        max_missing_pct: float = 0.1,
        check_outliers: bool = True,
        outlier_threshold: float = 0.5,
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Validate return series/dataframe.

        Args:
            returns: Returns data
            min_periods: Minimum required periods
            max_missing_pct: Maximum allowed missing data percentage
            check_outliers: Whether to check for extreme outliers
            outlier_threshold: Threshold for outliers (e.g., 0.5 = 50% return)

        Returns:
            Validated returns

        Raises:
            ValidationError: If validation fails

        Example:
            >>> returns = Validators.validate_returns(returns, min_periods=252)
        """
        if returns.empty:
            raise ValidationError("Returns data is empty")

        # Check length
        if len(returns) < min_periods:
            raise ValidationError(
                f"Insufficient data: {len(returns)} periods, need at least {min_periods}"
            )

        # Check for missing values
        missing_pct = returns.isnull().sum() / len(returns)
        if isinstance(missing_pct, pd.Series):
            high_missing = missing_pct[missing_pct > max_missing_pct]
            if not high_missing.empty:
                raise ValidationError(
                    f"High missing data in assets: {high_missing.to_dict()}"
                )
        else:
            if missing_pct > max_missing_pct:
                raise ValidationError(f"Missing data: {missing_pct:.1%} > {max_missing_pct:.1%}")

        # Check for inf values
        if np.isinf(returns).any().any() if isinstance(returns, pd.DataFrame) else np.isinf(returns).any():
            raise ValidationError("Returns contain infinite values")

        # Check for outliers
        if check_outliers:
            if isinstance(returns, pd.DataFrame):
                for col in returns.columns:
                    outliers = np.abs(returns[col]) > outlier_threshold
                    if outliers.sum() > 0:
                        logger.warning(
                            f"Extreme returns in {col}: {outliers.sum()} periods > {outlier_threshold:.1%}"
                        )
            else:
                outliers = np.abs(returns) > outlier_threshold
                if outliers.sum() > 0:
                    logger.warning(
                        f"Extreme returns: {outliers.sum()} periods > {outlier_threshold:.1%}"
                    )

        # Check for constant returns (likely data error)
        if isinstance(returns, pd.DataFrame):
            for col in returns.columns:
                if returns[col].std() == 0:
                    raise ValidationError(f"Constant returns in {col} (likely data error)")
        else:
            if returns.std() == 0:
                raise ValidationError("Constant returns (likely data error)")

        return returns

    @staticmethod
    def validate_prices(
        prices: Union[pd.Series, pd.DataFrame],
        allow_negative: bool = False,
        allow_zero: bool = False,
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Validate price data.

        Args:
            prices: Price data
            allow_negative: Whether to allow negative prices
            allow_zero: Whether to allow zero prices

        Returns:
            Validated prices

        Raises:
            ValidationError: If validation fails
        """
        if prices.empty:
            raise ValidationError("Price data is empty")

        # Check for NaN
        if prices.isnull().any().any() if isinstance(prices, pd.DataFrame) else prices.isnull().any():
            logger.warning("Price data contains NaN values")

        # Check for negative prices
        if not allow_negative:
            if isinstance(prices, pd.DataFrame):
                negative_mask = (prices < 0).any()
                if negative_mask.any():
                    raise ValidationError(f"Negative prices in: {negative_mask[negative_mask].index.tolist()}")
            else:
                if (prices < 0).any():
                    raise ValidationError("Negative prices detected")

        # Check for zero prices
        if not allow_zero:
            if isinstance(prices, pd.DataFrame):
                zero_mask = (prices == 0).any()
                if zero_mask.any():
                    logger.warning(f"Zero prices in: {zero_mask[zero_mask].index.tolist()}")
            else:
                if (prices == 0).any():
                    logger.warning("Zero prices detected")

        return prices

    @staticmethod
    def validate_weights(
        weights: Union[pd.Series, np.ndarray, dict],
        min_weight: float = -1.0,
        max_weight: float = 1.0,
        sum_to_one: bool = True,
        tolerance: float = 0.01,
    ) -> pd.Series:
        """
        Validate portfolio weights.

        Args:
            weights: Portfolio weights
            min_weight: Minimum allowed weight
            max_weight: Maximum allowed weight
            sum_to_one: Whether weights should sum to 1 (or -1/1 for long-short)
            tolerance: Tolerance for sum check

        Returns:
            Validated weights as Series

        Raises:
            ValidationError: If validation fails

        Example:
            >>> weights = Validators.validate_weights(weights, min_weight=0, max_weight=0.1)
        """
        # Convert to Series
        if isinstance(weights, dict):
            weights = pd.Series(weights)
        elif isinstance(weights, np.ndarray):
            weights = pd.Series(weights)
        elif not isinstance(weights, pd.Series):
            raise ValidationError(f"Invalid weights type: {type(weights)}")

        if weights.empty:
            raise ValidationError("Weights are empty")

        # Check for NaN/inf
        if weights.isnull().any():
            raise ValidationError("Weights contain NaN values")
        if np.isinf(weights).any():
            raise ValidationError("Weights contain infinite values")

        # Check bounds
        if (weights < min_weight).any():
            violations = weights[weights < min_weight]
            raise ValidationError(
                f"Weights below minimum {min_weight}: {violations.to_dict()}"
            )
        if (weights > max_weight).any():
            violations = weights[weights > max_weight]
            raise ValidationError(
                f"Weights above maximum {max_weight}: {violations.to_dict()}"
            )

        # Check sum
        if sum_to_one:
            weight_sum = weights.sum()
            # Allow for long-short portfolios (sum to -1 or 1)
            if not (abs(abs(weight_sum) - 1.0) < tolerance):
                raise ValidationError(
                    f"Weights sum to {weight_sum:.4f}, expected ±1.0 (tolerance: {tolerance})"
                )

        return weights

    @staticmethod
    def validate_covariance_matrix(
        cov_matrix: Union[pd.DataFrame, np.ndarray],
        min_periods: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Validate covariance matrix.

        Args:
            cov_matrix: Covariance matrix
            min_periods: Minimum periods used to estimate (for warnings)

        Returns:
            Validated covariance matrix

        Raises:
            ValidationError: If validation fails

        Example:
            >>> cov = Validators.validate_covariance_matrix(cov_matrix)
        """
        # Convert to DataFrame if needed
        if isinstance(cov_matrix, np.ndarray):
            cov_matrix = pd.DataFrame(cov_matrix)

        if cov_matrix.empty:
            raise ValidationError("Covariance matrix is empty")

        # Check shape
        if cov_matrix.shape[0] != cov_matrix.shape[1]:
            raise ValidationError(
                f"Covariance matrix not square: {cov_matrix.shape}"
            )

        # Check for NaN/inf
        if cov_matrix.isnull().any().any():
            raise ValidationError("Covariance matrix contains NaN")
        if np.isinf(cov_matrix.values).any():
            raise ValidationError("Covariance matrix contains infinite values")

        # Check symmetry
        if not np.allclose(cov_matrix, cov_matrix.T):
            logger.warning("Covariance matrix not symmetric (will symmetrize)")
            cov_matrix = (cov_matrix + cov_matrix.T) / 2

        # Check positive semi-definite
        try:
            eigenvalues = np.linalg.eigvalsh(cov_matrix.values)
            min_eigenvalue = eigenvalues.min()

            if min_eigenvalue < -1e-8:
                raise ValidationError(
                    f"Covariance matrix not positive semi-definite (min eigenvalue: {min_eigenvalue})"
                )
            elif min_eigenvalue < 0:
                logger.warning(
                    f"Small negative eigenvalue ({min_eigenvalue}), likely numerical error"
                )
        except np.linalg.LinAlgError:
            raise ValidationError("Failed to compute eigenvalues")

        # Check condition number
        condition_number = np.linalg.cond(cov_matrix.values)
        if condition_number > 1e10:
            logger.warning(
                f"High condition number ({condition_number:.2e}), matrix may be ill-conditioned"
            )

        # Warn about estimation periods
        if min_periods is not None and min_periods < cov_matrix.shape[0] * 2:
            logger.warning(
                f"Low estimation periods ({min_periods}) for {cov_matrix.shape[0]} assets"
            )

        return cov_matrix

    @staticmethod
    def validate_correlation_matrix(
        corr_matrix: Union[pd.DataFrame, np.ndarray],
    ) -> pd.DataFrame:
        """
        Validate correlation matrix.

        Args:
            corr_matrix: Correlation matrix

        Returns:
            Validated correlation matrix

        Raises:
            ValidationError: If validation fails
        """
        # Convert to DataFrame if needed
        if isinstance(corr_matrix, np.ndarray):
            corr_matrix = pd.DataFrame(corr_matrix)

        if corr_matrix.empty:
            raise ValidationError("Correlation matrix is empty")

        # Check shape
        if corr_matrix.shape[0] != corr_matrix.shape[1]:
            raise ValidationError(f"Correlation matrix not square: {corr_matrix.shape}")

        # Check for NaN/inf
        if corr_matrix.isnull().any().any():
            raise ValidationError("Correlation matrix contains NaN")
        if np.isinf(corr_matrix.values).any():
            raise ValidationError("Correlation matrix contains infinite values")

        # Check diagonal is 1
        diagonal = np.diag(corr_matrix.values)
        if not np.allclose(diagonal, 1.0):
            raise ValidationError(f"Correlation matrix diagonal not 1: {diagonal}")

        # Check values in [-1, 1]
        if (corr_matrix.values < -1.0).any() or (corr_matrix.values > 1.0).any():
            raise ValidationError("Correlation values outside [-1, 1]")

        # Check symmetry
        if not np.allclose(corr_matrix, corr_matrix.T):
            logger.warning("Correlation matrix not symmetric (will symmetrize)")
            corr_matrix = (corr_matrix + corr_matrix.T) / 2

        return corr_matrix

    @staticmethod
    def validate_positive(
        value: float,
        name: str = "value",
        allow_zero: bool = False,
    ) -> float:
        """
        Validate that a value is positive.

        Args:
            value: Value to check
            name: Name for error messages
            allow_zero: Whether to allow zero

        Returns:
            Validated value

        Raises:
            ValidationError: If validation fails
        """
        if pd.isna(value):
            raise ValidationError(f"{name} is NaN")
        if np.isinf(value):
            raise ValidationError(f"{name} is infinite")

        if allow_zero:
            if value < 0:
                raise ValidationError(f"{name} must be non-negative, got {value}")
        else:
            if value <= 0:
                raise ValidationError(f"{name} must be positive, got {value}")

        return value

    @staticmethod
    def validate_probability(
        value: float,
        name: str = "probability",
    ) -> float:
        """
        Validate that a value is a valid probability [0, 1].

        Args:
            value: Value to check
            name: Name for error messages

        Returns:
            Validated value

        Raises:
            ValidationError: If validation fails
        """
        if pd.isna(value):
            raise ValidationError(f"{name} is NaN")
        if value < 0 or value > 1:
            raise ValidationError(f"{name} must be in [0, 1], got {value}")

        return value

    @staticmethod
    def validate_dataframe_aligned(
        *dfs: pd.DataFrame,
        check_columns: bool = False,
    ) -> None:
        """
        Validate that DataFrames have aligned indices (and optionally columns).

        Args:
            *dfs: DataFrames to check
            check_columns: Whether to also check column alignment

        Raises:
            ValidationError: If DataFrames not aligned
        """
        if len(dfs) < 2:
            return

        first_df = dfs[0]

        for i, df in enumerate(dfs[1:], 1):
            if not first_df.index.equals(df.index):
                raise ValidationError(
                    f"DataFrame {i} has misaligned index with DataFrame 0"
                )

            if check_columns and not first_df.columns.equals(df.columns):
                raise ValidationError(
                    f"DataFrame {i} has misaligned columns with DataFrame 0"
                )

    @staticmethod
    def sanitize_weights(
        weights: pd.Series,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
        normalize: bool = True,
    ) -> pd.Series:
        """
        Sanitize weights by clipping and normalizing.

        Args:
            weights: Input weights
            min_weight: Minimum weight
            max_weight: Maximum weight
            normalize: Whether to normalize to sum to 1

        Returns:
            Sanitized weights

        Example:
            >>> weights = Validators.sanitize_weights(weights, min_weight=0, max_weight=0.1)
        """
        # Clip to bounds
        weights = weights.clip(lower=min_weight, upper=max_weight)

        # Normalize
        if normalize:
            weight_sum = weights.sum()
            if abs(weight_sum) > 1e-10:
                weights = weights / weight_sum
            else:
                logger.warning("Cannot normalize zero-sum weights, using equal weights")
                weights = pd.Series(1.0 / len(weights), index=weights.index)

        return weights

    @staticmethod
    def check_enough_data(
        data: Union[pd.Series, pd.DataFrame],
        min_periods: int,
        name: str = "data",
    ) -> None:
        """
        Check if data has enough periods.

        Args:
            data: Data to check
            min_periods: Minimum required periods
            name: Name for error messages

        Raises:
            ValidationError: If insufficient data
        """
        n_periods = len(data)
        if n_periods < min_periods:
            raise ValidationError(
                f"Insufficient {name}: {n_periods} periods, need at least {min_periods}"
            )

    @staticmethod
    def handle_missing_data(
        data: pd.DataFrame,
        method: str = "drop",
        threshold: float = 0.1,
    ) -> pd.DataFrame:
        """
        Handle missing data in DataFrame.

        Args:
            data: Input data
            method: 'drop', 'ffill', 'bfill', or 'interpolate'
            threshold: Maximum missing percentage before raising error

        Returns:
            Data with missing values handled

        Raises:
            ValidationError: If too much missing data

        Example:
            >>> data = Validators.handle_missing_data(data, method='ffill')
        """
        missing_pct = data.isnull().sum() / len(data)
        high_missing = missing_pct[missing_pct > threshold]

        if not high_missing.empty:
            raise ValidationError(
                f"Too much missing data: {high_missing.to_dict()}"
            )

        if method == "drop":
            data = data.dropna()
        elif method == "ffill":
            data = data.ffill()
        elif method == "bfill":
            data = data.bfill()
        elif method == "interpolate":
            data = data.interpolate(method='linear')
        else:
            raise ValueError(f"Unknown method: {method}")

        # Check for any remaining NaN
        if data.isnull().any().any():
            logger.warning(f"Some NaN values remain after {method}")

        return data

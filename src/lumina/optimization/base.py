"""
Base optimizer class.

Copyright (c) 2025 Vijeth Ltd. All rights reserved.
Author: Vithushan Jeyapahan <finance@vijeth.com>
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Any

import numpy as np
import pandas as pd
from loguru import logger

from lumina.utils.validation import validate_returns, validate_covariance_matrix


class BaseOptimizer(ABC):
    """
    Abstract base class for portfolio optimizers.

    All optimizer implementations should inherit from this class and implement
    the optimize() method.

    Attributes:
        returns: Historical returns DataFrame
        cov_matrix: Covariance matrix of returns
        expected_returns: Expected returns for each asset
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        expected_returns: Optional[pd.Series] = None,
        cov_matrix: Optional[pd.DataFrame] = None,
    ):
        """
        Initialize the base optimizer.

        Args:
            returns: Historical returns DataFrame with assets as columns
            expected_returns: Expected returns for each asset. If None, uses historical mean
            cov_matrix: Covariance matrix. If None, computed from returns

        Example:
            >>> returns = pd.DataFrame({'AAPL': [0.01, 0.02], 'GOOGL': [0.015, 0.012]})
            >>> optimizer = BaseOptimizer(returns)
        """
        self.returns = validate_returns(returns)
        self.assets = self.returns.columns.tolist()
        self.n_assets = len(self.assets)

        if expected_returns is None:
            self.expected_returns = self.returns.mean()
            logger.debug("Using historical mean returns as expected returns")
        else:
            if not all(asset in expected_returns.index for asset in self.assets):
                raise ValueError("Expected returns must contain all assets")
            self.expected_returns = expected_returns[self.assets]

        if cov_matrix is None:
            self.cov_matrix = self.returns.cov()
            logger.debug("Computed covariance matrix from returns")
        else:
            self.cov_matrix = validate_covariance_matrix(cov_matrix)
            if not all(asset in self.cov_matrix.index for asset in self.assets):
                raise ValueError("Covariance matrix must contain all assets")
            self.cov_matrix = self.cov_matrix.loc[self.assets, self.assets]

        logger.info(f"Initialized {self.__class__.__name__} with {self.n_assets} assets")

    @abstractmethod
    def optimize(self, **kwargs) -> pd.Series:
        """
        Optimize portfolio weights.

        Args:
            **kwargs: Optimizer-specific parameters

        Returns:
            Optimal portfolio weights as a Series

        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement optimize()")

    def portfolio_performance(
        self,
        weights: pd.Series,
        risk_free_rate: float = 0.0,
    ) -> Dict[str, float]:
        """
        Calculate portfolio performance metrics.

        Args:
            weights: Portfolio weights
            risk_free_rate: Annual risk-free rate

        Returns:
            Dictionary containing:
                - expected_return: Expected annual return
                - volatility: Annual volatility
                - sharpe_ratio: Sharpe ratio

        Example:
            >>> weights = pd.Series({'AAPL': 0.6, 'GOOGL': 0.4})
            >>> metrics = optimizer.portfolio_performance(weights)
        """
        weights_array = weights[self.assets].values

        expected_return = float(np.dot(weights_array, self.expected_returns.values))
        variance = float(
            np.dot(weights_array, np.dot(self.cov_matrix.values, weights_array))
        )
        volatility = float(np.sqrt(variance))

        sharpe_ratio = (
            (expected_return - risk_free_rate) / volatility if volatility > 0 else 0.0
        )

        return {
            "expected_return": expected_return * 252,  # Annualized
            "volatility": volatility * np.sqrt(252),  # Annualized
            "sharpe_ratio": sharpe_ratio * np.sqrt(252),  # Annualized
        }

    def _create_weight_series(self, weights_array: np.ndarray) -> pd.Series:
        """
        Create a pandas Series from weights array.

        Args:
            weights_array: NumPy array of weights

        Returns:
            Series with asset names as index
        """
        return pd.Series(weights_array, index=self.assets, name="weights")

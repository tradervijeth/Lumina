"""
Efficient frontier construction using PyPortfolioOpt.

Copyright (c) 2025 Vijeth Ltd. All rights reserved.
Author: Vithushan Jeyapahan <finance@vijeth.com>
"""

from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd
from loguru import logger

try:
    from pypfopt import EfficientFrontier, risk_models, expected_returns
    from pypfopt import objective_functions
    PYPFOPT_AVAILABLE = True
except ImportError:
    PYPFOPT_AVAILABLE = False
    logger.warning("PyPortfolioOpt not available. Install with: pip install PyPortfolioOpt")


class EfficientFrontierBuilder:
    """
    Efficient frontier construction using PyPortfolioOpt.

    Provides convenient interface to PyPortfolioOpt's efficient frontier
    optimization capabilities with additional analysis tools.
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        returns_method: str = 'mean_historical_return',
        risk_method: str = 'sample_cov',
    ):
        """
        Initialize efficient frontier builder.

        Args:
            prices: Historical price DataFrame with assets as columns
            returns_method: Method for expected returns ('mean_historical_return', 'ema_historical_return')
            risk_method: Method for risk model ('sample_cov', 'semicovariance', 'exp_cov')

        Example:
            >>> builder = EfficientFrontierBuilder(prices)
        """
        if not PYPFOPT_AVAILABLE:
            raise ImportError("PyPortfolioOpt is required. Install with: pip install PyPortfolioOpt")

        self.prices = prices

        # Calculate expected returns
        if returns_method == 'mean_historical_return':
            self.expected_returns = expected_returns.mean_historical_return(prices)
        elif returns_method == 'ema_historical_return':
            self.expected_returns = expected_returns.ema_historical_return(prices)
        else:
            raise ValueError(f"Unknown returns_method: {returns_method}")

        # Calculate risk model
        if risk_method == 'sample_cov':
            self.cov_matrix = risk_models.sample_cov(prices)
        elif risk_method == 'semicovariance':
            self.cov_matrix = risk_models.semicovariance(prices)
        elif risk_method == 'exp_cov':
            self.cov_matrix = risk_models.exp_cov(prices)
        else:
            raise ValueError(f"Unknown risk_method: {risk_method}")

        logger.info(
            f"Initialized EfficientFrontierBuilder with {len(prices.columns)} assets, "
            f"using {returns_method} and {risk_method}"
        )

    def max_sharpe(
        self,
        risk_free_rate: float = 0.02,
        **kwargs,
    ) -> Tuple[pd.Series, Dict[str, float]]:
        """
        Find the maximum Sharpe ratio portfolio.

        Args:
            risk_free_rate: Annual risk-free rate
            **kwargs: Additional arguments for EfficientFrontier

        Returns:
            Tuple of (weights, performance_dict)

        Example:
            >>> weights, perf = builder.max_sharpe(risk_free_rate=0.02)
        """
        ef = EfficientFrontier(
            self.expected_returns,
            self.cov_matrix,
            **kwargs,
        )

        weights = ef.max_sharpe(risk_free_rate=risk_free_rate)
        cleaned_weights = ef.clean_weights()

        perf = ef.portfolio_performance(
            verbose=False,
            risk_free_rate=risk_free_rate,
        )

        weights_series = pd.Series(cleaned_weights, name='weights')
        perf_dict = {
            'expected_return': perf[0],
            'volatility': perf[1],
            'sharpe_ratio': perf[2],
        }

        logger.info(
            f"Max Sharpe portfolio: Return={perf[0]:.2%}, "
            f"Vol={perf[1]:.2%}, Sharpe={perf[2]:.2f}"
        )

        return weights_series, perf_dict

    def min_volatility(
        self,
        **kwargs,
    ) -> Tuple[pd.Series, Dict[str, float]]:
        """
        Find the minimum volatility portfolio.

        Args:
            **kwargs: Additional arguments for EfficientFrontier

        Returns:
            Tuple of (weights, performance_dict)

        Example:
            >>> weights, perf = builder.min_volatility()
        """
        ef = EfficientFrontier(
            self.expected_returns,
            self.cov_matrix,
            **kwargs,
        )

        weights = ef.min_volatility()
        cleaned_weights = ef.clean_weights()

        perf = ef.portfolio_performance(verbose=False)

        weights_series = pd.Series(cleaned_weights, name='weights')
        perf_dict = {
            'expected_return': perf[0],
            'volatility': perf[1],
            'sharpe_ratio': perf[2],
        }

        logger.info(
            f"Min volatility portfolio: Return={perf[0]:.2%}, Vol={perf[1]:.2%}"
        )

        return weights_series, perf_dict

    def efficient_return(
        self,
        target_return: float,
        **kwargs,
    ) -> Tuple[pd.Series, Dict[str, float]]:
        """
        Find the minimum volatility portfolio for a target return.

        Args:
            target_return: Target annual return
            **kwargs: Additional arguments for EfficientFrontier

        Returns:
            Tuple of (weights, performance_dict)

        Example:
            >>> weights, perf = builder.efficient_return(target_return=0.15)
        """
        ef = EfficientFrontier(
            self.expected_returns,
            self.cov_matrix,
            **kwargs,
        )

        weights = ef.efficient_return(target_return=target_return)
        cleaned_weights = ef.clean_weights()

        perf = ef.portfolio_performance(verbose=False)

        weights_series = pd.Series(cleaned_weights, name='weights')
        perf_dict = {
            'expected_return': perf[0],
            'volatility': perf[1],
            'sharpe_ratio': perf[2],
        }

        logger.info(
            f"Efficient return portfolio: Return={perf[0]:.2%}, Vol={perf[1]:.2%}"
        )

        return weights_series, perf_dict

    def efficient_risk(
        self,
        target_volatility: float,
        **kwargs,
    ) -> Tuple[pd.Series, Dict[str, float]]:
        """
        Find the maximum return portfolio for a target volatility.

        Args:
            target_volatility: Target annual volatility
            **kwargs: Additional arguments for EfficientFrontier

        Returns:
            Tuple of (weights, performance_dict)

        Example:
            >>> weights, perf = builder.efficient_risk(target_volatility=0.15)
        """
        ef = EfficientFrontier(
            self.expected_returns,
            self.cov_matrix,
            **kwargs,
        )

        weights = ef.efficient_risk(target_volatility=target_volatility)
        cleaned_weights = ef.clean_weights()

        perf = ef.portfolio_performance(verbose=False)

        weights_series = pd.Series(cleaned_weights, name='weights')
        perf_dict = {
            'expected_return': perf[0],
            'volatility': perf[1],
            'sharpe_ratio': perf[2],
        }

        logger.info(
            f"Efficient risk portfolio: Return={perf[0]:.2%}, Vol={perf[1]:.2%}"
        )

        return weights_series, perf_dict

    def generate_frontier(
        self,
        n_points: int = 100,
    ) -> pd.DataFrame:
        """
        Generate the efficient frontier.

        Args:
            n_points: Number of points on the frontier

        Returns:
            DataFrame with frontier points

        Example:
            >>> frontier = builder.generate_frontier(n_points=50)
        """
        min_ret = self.expected_returns.min()
        max_ret = self.expected_returns.max()

        target_returns = np.linspace(min_ret, max_ret, n_points)

        frontier_points = []

        for target_ret in target_returns:
            try:
                ef = EfficientFrontier(
                    self.expected_returns,
                    self.cov_matrix,
                )
                ef.efficient_return(target_return=target_ret)
                perf = ef.portfolio_performance(verbose=False)

                frontier_points.append({
                    'expected_return': perf[0],
                    'volatility': perf[1],
                    'sharpe_ratio': perf[2],
                })
            except Exception:
                # Skip infeasible points
                continue

        frontier_df = pd.DataFrame(frontier_points)

        logger.info(f"Generated efficient frontier with {len(frontier_df)} points")

        return frontier_df

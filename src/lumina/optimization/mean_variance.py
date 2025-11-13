"""
Mean-variance portfolio optimization using cvxpy.

Copyright (c) 2025 Vijeth Ltd. All rights reserved.
Author: Vithushan Jeyapahan <finance@vijeth.com>
"""

from typing import Optional, Dict, Any

import cvxpy as cp
import numpy as np
import pandas as pd
from loguru import logger

from lumina.optimization.base import BaseOptimizer


class MeanVarianceOptimizer(BaseOptimizer):
    """
    Mean-variance portfolio optimizer using Markowitz theory.

    Solves the optimization problem:
        minimize: (1/2) * w^T * Σ * w - λ * μ^T * w
        subject to: w^T * 1 = 1 (weights sum to 1)
                    w >= 0 (long-only, optional)

    where:
        w = portfolio weights
        Σ = covariance matrix
        μ = expected returns
        λ = risk aversion parameter

    References:
        Markowitz, H. (1952). Portfolio Selection. The Journal of Finance, 7(1), 77-91.
    """

    def optimize(
        self,
        target_return: Optional[float] = None,
        risk_aversion: float = 1.0,
        max_weight: float = 1.0,
        min_weight: float = 0.0,
        long_only: bool = True,
        sector_constraints: Optional[Dict[str, float]] = None,
    ) -> pd.Series:
        """
        Optimize portfolio using mean-variance optimization.

        Args:
            target_return: Target annual return. If specified, finds minimum variance portfolio
                          achieving this return. Otherwise, uses risk_aversion parameter
            risk_aversion: Risk aversion parameter (λ). Higher values prioritize lower risk
            max_weight: Maximum weight for any single asset
            min_weight: Minimum weight for any single asset
            long_only: If True, constrains weights to be non-negative
            sector_constraints: Dict mapping sector names to max allocation (not implemented)

        Returns:
            Optimal portfolio weights

        Example:
            >>> optimizer = MeanVarianceOptimizer(returns)
            >>> weights = optimizer.optimize(target_return=0.12, long_only=True)
        """
        # Define optimization variable
        w = cp.Variable(self.n_assets)

        # Expected return and variance
        mu = self.expected_returns.values
        Sigma = self.cov_matrix.values

        # Objective: minimize variance - risk_aversion * expected_return
        portfolio_return = mu @ w
        portfolio_variance = cp.quad_form(w, Sigma)

        if target_return is not None:
            # Minimum variance for target return
            objective = cp.Minimize(portfolio_variance)
            target_daily = target_return / 252
            constraints = [
                cp.sum(w) == 1,
                portfolio_return >= target_daily,
            ]
            logger.info(f"Optimizing for minimum variance with target return {target_return:.2%}")
        else:
            # Risk-adjusted return maximization
            objective = cp.Minimize(
                portfolio_variance - risk_aversion * portfolio_return
            )
            constraints = [cp.sum(w) == 1]
            logger.info(f"Optimizing with risk aversion parameter λ={risk_aversion}")

        # Weight constraints
        if long_only:
            constraints.append(w >= 0)

        constraints.append(w >= min_weight)
        constraints.append(w <= max_weight)

        # Solve optimization problem
        problem = cp.Problem(objective, constraints)

        try:
            problem.solve(solver=cp.ECOS)
        except cp.SolverError:
            logger.warning("ECOS solver failed, trying SCS")
            problem.solve(solver=cp.SCS)

        if problem.status not in ["optimal", "optimal_inaccurate"]:
            raise ValueError(
                f"Optimization failed with status: {problem.status}. "
                f"Try adjusting constraints or target_return."
            )

        weights = self._create_weight_series(w.value)

        # Round very small weights to zero
        weights[np.abs(weights) < 1e-6] = 0
        weights = weights / weights.sum()  # Renormalize

        logger.info(f"Optimization successful. Objective value: {problem.value:.6f}")
        logger.debug(f"Active positions: {(weights > 1e-4).sum()}/{self.n_assets}")

        return weights

    def efficient_frontier(
        self,
        n_points: int = 100,
        long_only: bool = True,
    ) -> pd.DataFrame:
        """
        Generate the efficient frontier.

        Args:
            n_points: Number of points to generate on the frontier
            long_only: If True, constrains weights to be non-negative

        Returns:
            DataFrame with columns: expected_return, volatility, sharpe_ratio

        Example:
            >>> optimizer = MeanVarianceOptimizer(returns)
            >>> frontier = optimizer.efficient_frontier(n_points=50)
        """
        min_return = self.expected_returns.min() * 252
        max_return = self.expected_returns.max() * 252
        target_returns = np.linspace(min_return, max_return, n_points)

        results = []
        for target_ret in target_returns:
            try:
                weights = self.optimize(
                    target_return=target_ret,
                    long_only=long_only,
                )
                perf = self.portfolio_performance(weights)
                results.append(perf)
            except ValueError:
                # Skip infeasible points
                continue

        frontier_df = pd.DataFrame(results)
        logger.info(f"Generated efficient frontier with {len(frontier_df)} points")

        return frontier_df

    def max_sharpe_portfolio(
        self,
        risk_free_rate: float = 0.0,
        long_only: bool = True,
    ) -> pd.Series:
        """
        Find the maximum Sharpe ratio portfolio (tangency portfolio).

        Args:
            risk_free_rate: Annual risk-free rate
            long_only: If True, constrains weights to be non-negative

        Returns:
            Optimal portfolio weights

        Example:
            >>> optimizer = MeanVarianceOptimizer(returns)
            >>> weights = optimizer.max_sharpe_portfolio(risk_free_rate=0.02)
        """
        w = cp.Variable(self.n_assets)

        mu = self.expected_returns.values
        Sigma = self.cov_matrix.values
        rf_daily = risk_free_rate / 252

        # Maximize (μ - rf)^T w / sqrt(w^T Σ w)
        # Equivalent to minimizing variance for a unit excess return
        portfolio_return = mu @ w
        portfolio_variance = cp.quad_form(w, Sigma)

        objective = cp.Minimize(portfolio_variance)
        constraints = [
            portfolio_return - rf_daily == 1,  # Normalize excess return to 1
        ]

        if long_only:
            constraints.append(w >= 0)

        problem = cp.Problem(objective, constraints)

        try:
            problem.solve(solver=cp.ECOS)
        except cp.SolverError:
            logger.warning("ECOS solver failed, trying SCS")
            problem.solve(solver=cp.SCS)

        if problem.status not in ["optimal", "optimal_inaccurate"]:
            raise ValueError(f"Optimization failed with status: {problem.status}")

        # Normalize weights to sum to 1
        weights = w.value / np.sum(w.value)
        weights = self._create_weight_series(weights)

        perf = self.portfolio_performance(weights, risk_free_rate)
        logger.info(f"Maximum Sharpe ratio: {perf['sharpe_ratio']:.4f}")

        return weights

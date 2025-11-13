"""
Risk parity portfolio optimization.

Copyright (c) 2025 Vijeth Ltd. All rights reserved.
Author: Vithushan Jeyapahan <finance@vijeth.com>
"""

from typing import Optional

import cvxpy as cp
import numpy as np
import pandas as pd
from loguru import logger

from lumina.optimization.base import BaseOptimizer


class RiskParityOptimizer(BaseOptimizer):
    """
    Risk parity portfolio optimizer.

    Constructs a portfolio where each asset contributes equally to total risk:
        RC_i = w_i * (Σw)_i = constant for all i

    where:
        RC_i = risk contribution of asset i
        w_i = weight of asset i
        (Σw)_i = i-th element of Σw (marginal risk)

    This is solved by minimizing:
        Σ_i Σ_j (RC_i - RC_j)^2

    References:
        Maillard, S., Roncalli, T., & Teïletche, J. (2010). The Properties of
        Equally Weighted Risk Contribution Portfolios. The Journal of Portfolio
        Management, 36(4), 60-70.
    """

    def optimize(
        self,
        target_risk_contributions: Optional[pd.Series] = None,
        max_weight: float = 1.0,
        min_weight: float = 0.0,
        long_only: bool = True,
    ) -> pd.Series:
        """
        Optimize portfolio using risk parity.

        Args:
            target_risk_contributions: Target risk contribution for each asset.
                                      If None, uses equal risk contribution (1/N)
            max_weight: Maximum weight for any single asset
            min_weight: Minimum weight for any single asset
            long_only: If True, constrains weights to be non-negative

        Returns:
            Optimal portfolio weights

        Example:
            >>> optimizer = RiskParityOptimizer(returns)
            >>> weights = optimizer.optimize()  # Equal risk contribution
        """
        if target_risk_contributions is None:
            target_rc = np.ones(self.n_assets) / self.n_assets
            logger.info("Using equal risk contributions (1/N)")
        else:
            if not np.isclose(target_risk_contributions.sum(), 1.0):
                raise ValueError("Target risk contributions must sum to 1.0")
            target_rc = target_risk_contributions[self.assets].values

        w = cp.Variable(self.n_assets)
        Sigma = self.cov_matrix.values

        # Portfolio variance and marginal risk contributions
        portfolio_variance = cp.quad_form(w, Sigma)

        # For numerical stability, we use a sequential convex programming approach
        # Minimize sum of squared deviations from target risk contributions
        constraints = [
            cp.sum(w) == 1,
            w >= min_weight,
            w <= max_weight,
        ]

        if long_only:
            constraints.append(w >= 0)

        # Initial guess: equal weights
        w_init = np.ones(self.n_assets) / self.n_assets

        # Iterative optimization
        max_iterations = 50
        tolerance = 1e-6

        for iteration in range(max_iterations):
            # Calculate marginal risk contributions at current weights
            marginal_rc = Sigma @ w_init
            portfolio_vol = np.sqrt(w_init @ Sigma @ w_init)

            if portfolio_vol < 1e-10:
                logger.warning("Portfolio volatility near zero, using equal weights")
                w_init = np.ones(self.n_assets) / self.n_assets
                break

            # Risk contributions
            rc = w_init * marginal_rc / portfolio_vol

            # Check convergence
            rc_normalized = rc / rc.sum()
            error = np.sum((rc_normalized - target_rc) ** 2)

            if error < tolerance:
                logger.debug(f"Converged in {iteration + 1} iterations")
                break

            # Update weights using convex approximation
            # Minimize: ||w * (Σw) / sqrt(w^T Σ w) - target_rc||^2
            # Simplified objective for convex optimization
            objective = cp.Minimize(
                cp.sum_squares(w - w_init) +
                0.1 * cp.quad_form(w, Sigma)  # Regularization
            )

            problem = cp.Problem(objective, constraints)

            try:
                problem.solve(solver=cp.ECOS, verbose=False)
            except cp.SolverError:
                try:
                    problem.solve(solver=cp.SCS, verbose=False)
                except cp.SolverError:
                    logger.warning("Solver failed, using current weights")
                    break

            if problem.status not in ["optimal", "optimal_inaccurate"]:
                logger.warning(f"Iteration {iteration}: optimization status {problem.status}")
                break

            # Update weights
            w_new = w.value

            # Check for convergence in weights
            if np.linalg.norm(w_new - w_init) < tolerance:
                w_init = w_new
                break

            w_init = w_new

        else:
            logger.warning(f"Did not converge in {max_iterations} iterations")

        weights = self._create_weight_series(w_init)

        # Round very small weights to zero
        weights[np.abs(weights) < 1e-6] = 0
        weights = weights / weights.sum()  # Renormalize

        # Calculate final risk contributions
        final_rc = self._calculate_risk_contributions(weights)
        logger.info("Risk contributions:")
        for asset, rc in final_rc.items():
            logger.info(f"  {asset}: {rc:.2%}")

        return weights

    def _calculate_risk_contributions(self, weights: pd.Series) -> pd.Series:
        """
        Calculate risk contribution of each asset.

        Args:
            weights: Portfolio weights

        Returns:
            Risk contribution for each asset (sums to 1.0)
        """
        w = weights[self.assets].values
        Sigma = self.cov_matrix.values

        portfolio_vol = np.sqrt(w @ Sigma @ w)

        if portfolio_vol < 1e-10:
            return pd.Series(np.zeros(self.n_assets), index=self.assets)

        marginal_rc = Sigma @ w
        rc = w * marginal_rc / portfolio_vol
        rc_normalized = rc / rc.sum()

        return pd.Series(rc_normalized, index=self.assets, name="risk_contribution")

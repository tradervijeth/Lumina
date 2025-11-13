"""
Black-Litterman portfolio optimization.

Copyright (c) 2025 Vijeth Ltd. All rights reserved.
Author: Vithushan Jeyapahan <finance@vijeth.com>
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from lumina.optimization.base import BaseOptimizer


class BlackLittermanOptimizer(BaseOptimizer):
    """
    Black-Litterman model combining market equilibrium with investor views.

    The model updates expected returns based on:
    1. Market equilibrium (CAPM)
    2. Investor views with confidence levels

    E[R] = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ [(τΣ)⁻¹Π + P'Ω⁻¹Q]

    where:
        Π = equilibrium returns
        P = pick matrix (views)
        Q = view returns
        Ω = uncertainty in views
        τ = scaling factor

    References:
        Black, F., & Litterman, R. (1992). Global Portfolio Optimization.
    """

    def optimize(
        self,
        market_caps: Optional[pd.Series] = None,
        views: Optional[dict] = None,
        view_confidences: Optional[dict] = None,
        risk_aversion: float = 2.5,
        tau: float = 0.05,
        **kwargs,
    ) -> pd.Series:
        """
        Optimize using Black-Litterman model.

        Args:
            market_caps: Market capitalizations for equilibrium weights
            views: Dictionary of views {asset: expected_return} or
                   {(asset1, asset2): relative_return}
            view_confidences: Confidence in each view (0-1)
            risk_aversion: Market risk aversion parameter
            tau: Scaling factor for uncertainty

        Returns:
            Optimal portfolio weights

        Example:
            >>> views = {'AAPL': 0.15, 'GOOGL': 0.12}
            >>> confidences = {'AAPL': 0.8, 'GOOGL': 0.6}
            >>> weights = optimizer.optimize(market_caps, views, confidences)
        """
        # Calculate market equilibrium returns (reverse optimization)
        if market_caps is None:
            # Use equal weights as proxy
            market_weights = pd.Series(1 / self.n_assets, index=self.assets)
        else:
            market_weights = market_caps / market_caps.sum()

        # Equilibrium returns: Π = δΣw_mkt
        equilibrium_returns = risk_aversion * (self.cov_matrix.values @ market_weights.values)
        equilibrium_returns = pd.Series(equilibrium_returns, index=self.assets)

        # If no views, return market portfolio
        if not views:
            logger.info("No views specified, returning market portfolio")
            return market_weights

        # Process views
        n_views = len(views)
        P = np.zeros((n_views, self.n_assets))  # Pick matrix
        Q = np.zeros(n_views)  # View returns
        Omega = np.zeros((n_views, n_views))  # View uncertainty

        for i, (view_key, view_return) in enumerate(views.items()):
            Q[i] = view_return

            if isinstance(view_key, tuple):
                # Relative view: asset1 - asset2
                asset1, asset2 = view_key
                P[i, self.assets.index(asset1)] = 1
                P[i, self.assets.index(asset2)] = -1
            else:
                # Absolute view on single asset
                asset = view_key
                P[i, self.assets.index(asset)] = 1

            # View uncertainty
            if view_confidences and view_key in view_confidences:
                confidence = view_confidences[view_key]
                # Higher confidence = lower uncertainty
                Omega[i, i] = (1 - confidence) * tau * (P[i] @ self.cov_matrix.values @ P[i])
            else:
                Omega[i, i] = tau * (P[i] @ self.cov_matrix.values @ P[i])

        # Black-Litterman formula
        tau_sigma = tau * self.cov_matrix.values
        tau_sigma_inv = np.linalg.inv(tau_sigma)
        omega_inv = np.linalg.inv(Omega)

        # Posterior covariance
        M_inv = tau_sigma_inv + P.T @ omega_inv @ P
        M = np.linalg.inv(M_inv)

        # Posterior returns
        posterior_returns = M @ (
            tau_sigma_inv @ equilibrium_returns.values +
            P.T @ omega_inv @ Q
        )

        posterior_returns = pd.Series(posterior_returns, index=self.assets)

        # Optimize with posterior returns
        posterior_cov = self.cov_matrix + M

        # Mean-variance optimization
        weights_array = np.linalg.solve(
            risk_aversion * posterior_cov.values,
            posterior_returns.values,
        )

        # Normalize
        weights = pd.Series(weights_array, index=self.assets)
        weights = weights / weights.sum()

        # Clip small weights
        weights[np.abs(weights) < 1e-6] = 0
        weights = weights / weights.sum()

        logger.info(f"Black-Litterman optimization complete with {n_views} views")

        return weights

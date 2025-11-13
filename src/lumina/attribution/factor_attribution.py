"""
Factor-based performance attribution.

Copyright (c) 2025 Vijeth Ltd. All rights reserved.
Author: Vithushan Jeyapahan <finance@vijeth.com>
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from loguru import logger
from sklearn.linear_model import LinearRegression


class FactorAttribution:
    """
    Attribute portfolio returns to risk factors.

    Supports:
    - Fama-French factors
    - Custom factor models
    - Regression-based attribution
    """

    def __init__(self, portfolio_returns: pd.Series):
        """
        Initialize factor attribution.

        Args:
            portfolio_returns: Portfolio returns to analyze

        Example:
            >>> attribution = FactorAttribution(portfolio_returns)
        """
        self.portfolio_returns = portfolio_returns
        logger.info(f"Initialized attribution for {len(portfolio_returns)} periods")

    def attribute_to_factors(
        self,
        factor_returns: pd.DataFrame,
        include_intercept: bool = True,
    ) -> dict:
        """
        Attribute returns to factors using regression.

        Model: R_p = α + β₁F₁ + β₂F₂ + ... + ε

        Args:
            factor_returns: DataFrame of factor returns
            include_intercept: Include alpha (intercept)

        Returns:
            Dictionary with factor exposures and statistics

        Example:
            >>> results = attribution.attribute_to_factors(factor_returns)
        """
        # Align data
        aligned_portfolio = self.portfolio_returns.reindex(factor_returns.index).dropna()
        aligned_factors = factor_returns.reindex(aligned_portfolio.index)

        # Run regression
        X = aligned_factors.values
        y = aligned_portfolio.values

        model = LinearRegression(fit_intercept=include_intercept)
        model.fit(X, y)

        # Extract results
        factor_exposures = pd.Series(
            model.coef_,
            index=factor_returns.columns,
            name='beta',
        )

        alpha = model.intercept_ if include_intercept else 0.0

        # Calculate R-squared
        y_pred = model.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        # Calculate factor contributions
        factor_contributions = {}
        for factor in factor_returns.columns:
            beta = factor_exposures[factor]
            factor_ret = aligned_factors[factor].mean() * 252  # Annualized
            contribution = beta * factor_ret
            factor_contributions[factor] = contribution

        logger.info(f"Factor attribution: R²={r_squared:.3f}, Alpha={alpha*252:.2%}")

        return {
            'alpha': alpha * 252,  # Annualized
            'factor_exposures': factor_exposures,
            'factor_contributions': factor_contributions,
            'r_squared': r_squared,
            'residual_variance': np.var(y - y_pred),
        }

    def fama_french_attribution(self) -> dict:
        """
        Attribute to Fama-French 3-factor model (simulated).

        Factors:
        - Market (Rm - Rf)
        - SMB (Small Minus Big)
        - HML (High Minus Low)

        Returns:
            Attribution results

        Example:
            >>> results = attribution.fama_french_attribution()
        """
        # Generate simulated factor returns
        dates = self.portfolio_returns.index

        # Simplified factor simulation
        np.random.seed(42)
        market_premium = pd.Series(
            np.random.normal(0.0004, 0.01, len(dates)),
            index=dates,
            name='Market',
        )
        smb = pd.Series(
            np.random.normal(0.0001, 0.005, len(dates)),
            index=dates,
            name='SMB',
        )
        hml = pd.Series(
            np.random.normal(0.0001, 0.005, len(dates)),
            index=dates,
            name='HML',
        )

        factor_returns = pd.concat([market_premium, smb, hml], axis=1)

        results = self.attribute_to_factors(factor_returns)

        logger.info("Fama-French attribution complete")

        return results

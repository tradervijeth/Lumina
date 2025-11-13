"""
Portfolio performance analysis.

Copyright (c) 2025 Vijeth Ltd. All rights reserved.
Author: Vithushan Jeyapahan <finance@vijeth.com>
"""

from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd
from loguru import logger

from lumina.utils.metrics import (
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    calculate_calmar_ratio,
)
from lumina.utils.validation import validate_weights, validate_returns


class PortfolioAnalyzer:
    """
    Comprehensive portfolio performance analysis.

    Provides detailed performance metrics, risk analytics, and
    attribution analysis for portfolios.
    """

    def __init__(
        self,
        weights: pd.Series,
        returns: pd.DataFrame,
        benchmark_returns: Optional[pd.Series] = None,
    ):
        """
        Initialize portfolio analyzer.

        Args:
            weights: Portfolio weights
            returns: Historical returns DataFrame with assets as columns
            benchmark_returns: Optional benchmark returns for comparison

        Example:
            >>> weights = pd.Series({'AAPL': 0.6, 'GOOGL': 0.4})
            >>> analyzer = PortfolioAnalyzer(weights, returns)
        """
        self.weights = validate_weights(weights)
        self.returns = validate_returns(returns)

        # Align weights with returns columns
        if not all(asset in self.returns.columns for asset in self.weights.index):
            raise ValueError("All assets in weights must be in returns DataFrame")

        self.weights = self.weights[self.returns.columns]
        self.benchmark_returns = benchmark_returns

        # Calculate portfolio returns
        self.portfolio_returns = (self.returns * self.weights).sum(axis=1)

        logger.info(f"Initialized PortfolioAnalyzer with {len(self.weights)} assets")

    def performance_metrics(
        self,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252,
    ) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics.

        Args:
            risk_free_rate: Annual risk-free rate
            periods_per_year: Periods per year (252 for daily, 12 for monthly)

        Returns:
            Dictionary of performance metrics

        Example:
            >>> metrics = analyzer.performance_metrics(risk_free_rate=0.02)
        """
        returns = self.portfolio_returns

        # Return metrics
        total_return = (1 + returns).prod() - 1
        annual_return = (1 + returns).prod() ** (periods_per_year / len(returns)) - 1
        mean_return = returns.mean() * periods_per_year

        # Risk metrics
        volatility = returns.std() * np.sqrt(periods_per_year)
        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(periods_per_year)

        # Risk-adjusted returns
        sharpe_ratio = calculate_sharpe_ratio(
            returns, risk_free_rate, periods_per_year
        )
        sortino_ratio = calculate_sortino_ratio(
            returns, risk_free_rate, periods_per_year
        )

        # Drawdown metrics
        max_drawdown = calculate_max_drawdown(returns)
        calmar_ratio = calculate_calmar_ratio(returns, periods_per_year)

        # Win rate
        win_rate = (returns > 0).sum() / len(returns)

        # Value at Risk (95% and 99%)
        var_95 = returns.quantile(0.05)
        var_99 = returns.quantile(0.01)

        # Conditional Value at Risk (Expected Shortfall)
        cvar_95 = returns[returns <= var_95].mean()
        cvar_99 = returns[returns <= var_99].mean()

        metrics = {
            'total_return': total_return,
            'annual_return': annual_return,
            'mean_return': mean_return,
            'volatility': volatility,
            'downside_volatility': downside_volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99,
        }

        # Benchmark comparison if available
        if self.benchmark_returns is not None:
            aligned_benchmark = self.benchmark_returns.reindex(
                self.portfolio_returns.index
            )
            tracking_error = (
                (self.portfolio_returns - aligned_benchmark).std() *
                np.sqrt(periods_per_year)
            )
            information_ratio = (
                (self.portfolio_returns - aligned_benchmark).mean() /
                (self.portfolio_returns - aligned_benchmark).std() *
                np.sqrt(periods_per_year)
            )
            metrics['tracking_error'] = tracking_error
            metrics['information_ratio'] = information_ratio

        logger.info(
            f"Portfolio metrics: Return={annual_return:.2%}, "
            f"Vol={volatility:.2%}, Sharpe={sharpe_ratio:.2f}"
        )

        return metrics

    def risk_contribution(self) -> pd.Series:
        """
        Calculate risk contribution of each asset.

        Returns:
            Series with risk contribution for each asset

        Example:
            >>> contributions = analyzer.risk_contribution()
        """
        cov_matrix = self.returns.cov()
        weights = self.weights.values

        # Portfolio variance
        portfolio_variance = weights @ cov_matrix.values @ weights
        portfolio_vol = np.sqrt(portfolio_variance)

        # Marginal risk contribution
        marginal_contrib = cov_matrix.values @ weights

        # Risk contribution: w_i * (Σw)_i / σ_p
        risk_contrib = weights * marginal_contrib / portfolio_vol

        # Normalize to percentage
        risk_contrib_pct = risk_contrib / risk_contrib.sum()

        contrib_series = pd.Series(
            risk_contrib_pct,
            index=self.weights.index,
            name='risk_contribution',
        )

        logger.debug("Calculated risk contributions")

        return contrib_series

    def return_attribution(self) -> pd.DataFrame:
        """
        Perform return attribution analysis.

        Returns:
            DataFrame with return contribution by asset

        Example:
            >>> attribution = analyzer.return_attribution()
        """
        # Return contribution
        asset_contributions = self.returns * self.weights

        attribution_df = pd.DataFrame({
            'weight': self.weights,
            'total_return': self.returns.sum(),
            'contribution': asset_contributions.sum(),
            'contribution_pct': asset_contributions.sum() / self.portfolio_returns.sum(),
        })

        attribution_df = attribution_df.sort_values('contribution', ascending=False)

        logger.debug("Performed return attribution analysis")

        return attribution_df

    def drawdown_analysis(self) -> pd.DataFrame:
        """
        Analyze portfolio drawdowns over time.

        Returns:
            DataFrame with drawdown statistics

        Example:
            >>> drawdowns = analyzer.drawdown_analysis()
        """
        cumulative = (1 + self.portfolio_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max

        drawdown_df = pd.DataFrame({
            'cumulative_return': cumulative,
            'running_max': running_max,
            'drawdown': drawdown,
        })

        logger.debug("Completed drawdown analysis")

        return drawdown_df

    def correlation_analysis(self) -> pd.DataFrame:
        """
        Analyze correlation between portfolio and assets.

        Returns:
            DataFrame with correlation statistics

        Example:
            >>> correlations = analyzer.correlation_analysis()
        """
        correlations = self.returns.corrwith(self.portfolio_returns)

        corr_df = pd.DataFrame({
            'correlation': correlations,
            'weight': self.weights,
        }).sort_values('correlation', ascending=False)

        logger.debug("Completed correlation analysis")

        return corr_df

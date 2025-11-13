"""
Unit tests for optimization module.

Copyright (c) 2025 Vijeth Ltd. All rights reserved.
Author: Vithushan Jeyapahan <finance@vijeth.com>
"""

import pytest
import numpy as np
import pandas as pd

from lumina.optimization import MeanVarianceOptimizer, RiskParityOptimizer


@pytest.fixture
def sample_returns():
    """Generate sample returns data for testing."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=252, freq='D')
    assets = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']

    returns = pd.DataFrame(
        np.random.randn(252, 4) * 0.01,
        index=dates,
        columns=assets,
    )

    return returns


class TestMeanVarianceOptimizer:
    """Test cases for MeanVarianceOptimizer."""

    def test_initialization(self, sample_returns):
        """Test optimizer initialization."""
        optimizer = MeanVarianceOptimizer(sample_returns)

        assert optimizer.n_assets == 4
        assert len(optimizer.assets) == 4
        assert optimizer.cov_matrix.shape == (4, 4)

    def test_optimize_target_return(self, sample_returns):
        """Test optimization with target return."""
        optimizer = MeanVarianceOptimizer(sample_returns)
        weights = optimizer.optimize(target_return=0.10, long_only=True)

        # Check weights properties
        assert len(weights) == 4
        assert np.isclose(weights.sum(), 1.0, atol=1e-4)
        assert (weights >= 0).all()  # Long only

    def test_optimize_risk_aversion(self, sample_returns):
        """Test optimization with risk aversion parameter."""
        optimizer = MeanVarianceOptimizer(sample_returns)
        weights = optimizer.optimize(risk_aversion=2.0, long_only=True)

        assert len(weights) == 4
        assert np.isclose(weights.sum(), 1.0, atol=1e-4)

    def test_max_sharpe_portfolio(self, sample_returns):
        """Test maximum Sharpe ratio portfolio."""
        optimizer = MeanVarianceOptimizer(sample_returns)
        weights = optimizer.max_sharpe_portfolio(risk_free_rate=0.02)

        assert len(weights) == 4
        assert np.isclose(weights.sum(), 1.0, atol=1e-4)

    def test_portfolio_performance(self, sample_returns):
        """Test portfolio performance calculation."""
        optimizer = MeanVarianceOptimizer(sample_returns)
        weights = pd.Series([0.25, 0.25, 0.25, 0.25], index=sample_returns.columns)

        perf = optimizer.portfolio_performance(weights, risk_free_rate=0.02)

        assert 'expected_return' in perf
        assert 'volatility' in perf
        assert 'sharpe_ratio' in perf
        assert perf['volatility'] > 0


class TestRiskParityOptimizer:
    """Test cases for RiskParityOptimizer."""

    def test_initialization(self, sample_returns):
        """Test optimizer initialization."""
        optimizer = RiskParityOptimizer(sample_returns)

        assert optimizer.n_assets == 4
        assert len(optimizer.assets) == 4

    def test_optimize_equal_risk(self, sample_returns):
        """Test equal risk contribution optimization."""
        optimizer = RiskParityOptimizer(sample_returns)
        weights = optimizer.optimize()

        # Check weights properties
        assert len(weights) == 4
        assert np.isclose(weights.sum(), 1.0, atol=1e-4)
        assert (weights >= 0).all()

    def test_risk_contributions(self, sample_returns):
        """Test risk contribution calculation."""
        optimizer = RiskParityOptimizer(sample_returns)
        weights = optimizer.optimize()

        risk_contrib = optimizer._calculate_risk_contributions(weights)

        # Risk contributions should sum to 1
        assert np.isclose(risk_contrib.sum(), 1.0, atol=1e-4)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

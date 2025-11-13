"""
Unit tests for utility functions.

Copyright (c) 2025 Vijeth Ltd. All rights reserved.
Author: Vithushan Jeyapahan <finance@vijeth.com>
"""

import pytest
import numpy as np
import pandas as pd

from lumina.utils.validation import validate_returns, validate_weights
from lumina.utils.metrics import (
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
)


class TestValidation:
    """Test cases for validation functions."""

    def test_validate_returns_dataframe(self):
        """Test returns validation with DataFrame."""
        returns = pd.DataFrame({
            'AAPL': [0.01, 0.02, -0.01],
            'GOOGL': [0.015, -0.005, 0.02],
        })

        validated = validate_returns(returns)

        assert isinstance(validated, pd.DataFrame)
        assert validated.shape == (3, 2)

    def test_validate_returns_series(self):
        """Test returns validation with Series."""
        returns = pd.Series([0.01, 0.02, -0.01])

        validated = validate_returns(returns)

        assert isinstance(validated, pd.DataFrame)
        assert validated.shape == (3, 1)

    def test_validate_returns_with_nan(self):
        """Test that NaN values raise error by default."""
        returns = pd.DataFrame({
            'AAPL': [0.01, np.nan, -0.01],
        })

        with pytest.raises(ValueError, match="NaN values"):
            validate_returns(returns, allow_nan=False)

    def test_validate_weights_series(self):
        """Test weights validation with Series."""
        weights = pd.Series([0.6, 0.4], index=['AAPL', 'GOOGL'])

        validated = validate_weights(weights)

        assert isinstance(validated, pd.Series)
        assert np.isclose(validated.sum(), 1.0)

    def test_validate_weights_dict(self):
        """Test weights validation with dict."""
        weights = {'AAPL': 0.6, 'GOOGL': 0.4}

        validated = validate_weights(weights)

        assert isinstance(validated, pd.Series)
        assert np.isclose(validated.sum(), 1.0)

    def test_validate_weights_invalid_sum(self):
        """Test that weights not summing to 1 raise error."""
        weights = pd.Series([0.5, 0.3])

        with pytest.raises(ValueError, match="sum to"):
            validate_weights(weights)


class TestMetrics:
    """Test cases for metric calculations."""

    def test_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        returns = pd.Series([0.01, 0.02, -0.01, 0.015, 0.005])

        sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.02, periods_per_year=252)

        assert isinstance(sharpe, float)
        assert not np.isnan(sharpe)

    def test_sortino_ratio(self):
        """Test Sortino ratio calculation."""
        returns = pd.Series([0.01, 0.02, -0.01, 0.015, -0.005])

        sortino = calculate_sortino_ratio(returns, risk_free_rate=0.02, periods_per_year=252)

        assert isinstance(sortino, float)
        assert not np.isnan(sortino)

    def test_max_drawdown(self):
        """Test maximum drawdown calculation."""
        returns = pd.Series([0.05, 0.03, -0.10, -0.05, 0.08])

        mdd = calculate_max_drawdown(returns)

        assert isinstance(mdd, float)
        assert mdd >= 0  # Always positive
        assert mdd <= 1.0  # Cannot exceed 100%

    def test_max_drawdown_no_losses(self):
        """Test max drawdown with only positive returns."""
        returns = pd.Series([0.01, 0.02, 0.01, 0.015])

        mdd = calculate_max_drawdown(returns)

        assert mdd >= 0
        assert mdd < 0.01  # Should be very small


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

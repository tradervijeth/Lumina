"""
Real data example with data loading and risk management.

Copyright (c) 2025 Vijeth Ltd. All rights reserved.
Author: Vithushan Jeyapahan <finance@vijeth.com>
"""

import numpy as np
import pandas as pd
from pathlib import Path

from lumina.data import DataLoader, DataCleaner, Universe
from lumina.risk import PositionSizer, RiskLimits
from lumina.optimization import MeanVarianceOptimizer
from lumina.utils import setup_logger


def main():
    """Run real data example with risk management."""
    setup_logger(log_level="INFO")

    print("=" * 60)
    print("Lumina Real Data & Risk Management Example")
    print("=" * 60)

    # 1. Load Data
    print("\n1. Loading Data...")
    loader = DataLoader()

    # Use predefined universe
    universe = Universe.from_predefined("TEST")
    print(f"   Universe: {universe.get_symbols()}")

    # Load synthetic data (in production, use real data)
    data = loader.load_synthetic(
        symbols=universe.get_symbols(),
        start_date="2021-01-01",
        end_date="2023-12-31",
        mu=0.0005,
        sigma=0.02,
    )

    print(f"   Loaded {len(data)} days of price data")

    # 2. Clean Data
    print("\n2. Cleaning Data...")
    returns = DataCleaner.calculate_returns(data, method='simple')
    returns = DataCleaner.handle_missing_data(returns, method='drop')  # Drop first row with NaN

    print(f"   Calculated returns for {len(returns.columns)} assets")

    # 3. Portfolio Optimization
    print("\n3. Optimizing Portfolio...")
    optimizer = MeanVarianceOptimizer(returns)
    weights = optimizer.max_sharpe_portfolio(risk_free_rate=0.02)

    print("\n   Portfolio Weights:")
    for asset, weight in weights.items():
        if weight > 0.01:
            print(f"      {asset}: {weight:.2%}")

    # 4. Apply Risk Limits
    print("\n4. Applying Risk Limits...")
    limits = RiskLimits(
        max_position_size=0.30,
        max_leverage=1.0,
        max_drawdown=0.20,
    )

    adjusted_weights = limits.enforce_position_limits(weights)

    print("\n   Adjusted Weights:")
    for asset, weight in adjusted_weights.items():
        if weight > 0.01:
            print(f"      {asset}: {weight:.2%}")

    # 5. Position Sizing
    print("\n5. Calculating Position Sizes...")
    capital = 1000000
    current_prices = data.iloc[-1]

    # Fixed fraction sizing
    positions = {}
    for asset in adjusted_weights.index:
        if adjusted_weights[asset] > 0:
            shares = PositionSizer.fixed_fraction(
                capital=capital * adjusted_weights[asset],
                fraction=1.0,
                price=current_prices[asset],
            )
            positions[asset] = shares
            position_value = shares * current_prices[asset]
            print(f"      {asset}: {shares} shares (${position_value:,.0f})")

    # 6. Risk Metrics
    print("\n6. Risk Analysis...")
    from lumina.risk import VaRCalculator

    portfolio_returns = (returns * adjusted_weights).sum(axis=1)

    var_95 = VaRCalculator.historical_var(portfolio_returns, 0.95, capital)
    es_95 = VaRCalculator.expected_shortfall(portfolio_returns, 0.95, capital)

    print(f"   Value at Risk (95%):    ${var_95:,.0f}")
    print(f"   Expected Shortfall (95%): ${es_95:,.0f}")

    # Check if within limits
    if limits.check_drawdown(portfolio_returns):
        print(f"   ✓ Drawdown within limits")

    if limits.check_leverage(adjusted_weights):
        print(f"   ✓ Leverage within limits")

    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

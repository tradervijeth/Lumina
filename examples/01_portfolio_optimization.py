"""
Portfolio optimization example.

Demonstrates mean-variance and risk parity optimization.

Copyright (c) 2025 Vijeth Ltd. All rights reserved.
Author: Vithushan Jeyapahan <finance@vijeth.com>
"""

import numpy as np
import pandas as pd
from pathlib import Path

from lumina.optimization import MeanVarianceOptimizer, RiskParityOptimizer
from lumina.utils.logging import setup_logger


def generate_sample_data(n_assets: int = 5, n_periods: int = 252) -> pd.DataFrame:
    """Generate sample returns data."""
    np.random.seed(42)

    assets = [f"Asset_{i+1}" for i in range(n_assets)]
    dates = pd.date_range('2023-01-01', periods=n_periods, freq='D')

    # Generate correlated returns
    mean_returns = np.random.uniform(0.0001, 0.0005, n_assets)
    volatilities = np.random.uniform(0.01, 0.03, n_assets)

    returns = pd.DataFrame(
        np.random.randn(n_periods, n_assets) * volatilities + mean_returns,
        index=dates,
        columns=assets,
    )

    return returns


def main():
    """Run portfolio optimization examples."""
    # Setup logging
    setup_logger(log_level="INFO")

    print("=" * 60)
    print("Lumina Portfolio Optimization Example")
    print("=" * 60)

    # Generate sample data
    print("\n1. Generating sample returns data...")
    returns = generate_sample_data(n_assets=5, n_periods=252)
    print(f"   Generated {len(returns)} days of returns for {len(returns.columns)} assets")

    # Mean-Variance Optimization
    print("\n2. Mean-Variance Optimization")
    print("-" * 60)

    mv_optimizer = MeanVarianceOptimizer(returns)

    # Maximum Sharpe ratio portfolio
    print("\n   a) Maximum Sharpe Ratio Portfolio:")
    max_sharpe_weights = mv_optimizer.max_sharpe_portfolio(risk_free_rate=0.02)
    print("\n   Weights:")
    for asset, weight in max_sharpe_weights.items():
        if weight > 0.01:  # Only show significant positions
            print(f"      {asset}: {weight:.2%}")

    perf = mv_optimizer.portfolio_performance(max_sharpe_weights, risk_free_rate=0.02)
    print(f"\n   Expected Return: {perf['expected_return']:.2%}")
    print(f"   Volatility:      {perf['volatility']:.2%}")
    print(f"   Sharpe Ratio:    {perf['sharpe_ratio']:.2f}")

    # Target return optimization
    print("\n   b) Minimum Variance for Target Return (12%):")
    target_weights = mv_optimizer.optimize(target_return=0.12, long_only=True)
    print("\n   Weights:")
    for asset, weight in target_weights.items():
        if weight > 0.01:
            print(f"      {asset}: {weight:.2%}")

    perf = mv_optimizer.portfolio_performance(target_weights, risk_free_rate=0.02)
    print(f"\n   Expected Return: {perf['expected_return']:.2%}")
    print(f"   Volatility:      {perf['volatility']:.2%}")
    print(f"   Sharpe Ratio:    {perf['sharpe_ratio']:.2f}")

    # Risk Parity Optimization
    print("\n3. Risk Parity Optimization")
    print("-" * 60)

    rp_optimizer = RiskParityOptimizer(returns)
    rp_weights = rp_optimizer.optimize()

    print("\n   Equal Risk Contribution Weights:")
    for asset, weight in rp_weights.items():
        print(f"      {asset}: {weight:.2%}")

    risk_contrib = rp_optimizer._calculate_risk_contributions(rp_weights)
    print("\n   Risk Contributions:")
    for asset, contrib in risk_contrib.items():
        print(f"      {asset}: {contrib:.2%}")

    perf = rp_optimizer.portfolio_performance(rp_weights, risk_free_rate=0.02)
    print(f"\n   Expected Return: {perf['expected_return']:.2%}")
    print(f"   Volatility:      {perf['volatility']:.2%}")
    print(f"   Sharpe Ratio:    {perf['sharpe_ratio']:.2f}")

    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

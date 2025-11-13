"""
Backtesting example.

Demonstrates strategy backtesting with the Lumina framework.

Copyright (c) 2025 Vijeth Ltd. All rights reserved.
Author: Vithushan Jeyapahan <finance@vijeth.com>
"""

import numpy as np
import pandas as pd

from lumina.backtest import BacktestEngine, MomentumStrategy, PerformanceAnalyzer
from lumina.utils.logging import setup_logger


def generate_sample_prices(n_assets: int = 10, n_periods: int = 500) -> pd.DataFrame:
    """Generate sample price data."""
    np.random.seed(42)

    assets = [f"Stock_{i+1}" for i in range(n_assets)]
    dates = pd.date_range('2022-01-01', periods=n_periods, freq='D')

    # Generate price paths using geometric Brownian motion
    initial_prices = np.random.uniform(50, 200, n_assets)
    returns = np.random.randn(n_periods, n_assets) * 0.015 + 0.0002

    prices = pd.DataFrame(index=dates, columns=assets)
    prices.iloc[0] = initial_prices

    for i in range(1, n_periods):
        prices.iloc[i] = prices.iloc[i-1] * (1 + returns[i])

    return prices


def main():
    """Run backtesting examples."""
    # Setup logging
    setup_logger(log_level="INFO")

    print("=" * 60)
    print("Lumina Backtesting Example")
    print("=" * 60)

    # Generate sample data
    print("\n1. Generating sample price data...")
    prices = generate_sample_prices(n_assets=10, n_periods=500)
    print(f"   Generated {len(prices)} days of prices for {len(prices.columns)} assets")

    # Create momentum strategy
    print("\n2. Creating Momentum Strategy...")
    strategy = MomentumStrategy(
        lookback_period=20,
        top_n=5,
        initial_capital=1000000,
        commission=0.001,
    )
    print(f"   Strategy: {strategy.name}")
    print(f"   Lookback: {strategy.lookback_period} days")
    print(f"   Top N: {strategy.top_n} assets")

    # Run backtest
    print("\n3. Running Backtest...")
    engine = BacktestEngine(
        strategy=strategy,
        data=prices,
        commission=0.001,
        slippage=0.0005,
    )

    results = engine.run(rebalance_frequency='W')
    print(f"   Backtest complete!")

    # Analyze performance
    print("\n4. Performance Analysis")
    print("-" * 60)

    analyzer = PerformanceAnalyzer(results)

    # Print summary statistics
    print(analyzer.summary_statistics())

    # Get monthly returns
    print("\n5. Monthly Returns")
    print("-" * 60)
    monthly_returns = analyzer.monthly_returns()
    print(monthly_returns)

    # Trade analysis
    print("\n6. Trade Statistics")
    print("-" * 60)
    trades = engine.get_trades()
    print(f"   Total Trades: {len(trades)}")
    print(f"   Total Costs:  ${trades['cost'].sum():,.0f}")
    print(f"   Avg Trade:    ${trades['value'].abs().mean():,.0f}")

    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

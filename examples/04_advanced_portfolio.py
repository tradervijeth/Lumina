"""
Advanced portfolio construction with Black-Litterman and attribution.

Copyright (c) 2025 Vijeth Ltd. All rights reserved.
Author: Vithushan Jeyapahan <finance@vijeth.com>
"""

import numpy as np
import pandas as pd

from lumina.data import DataLoader
from lumina.optimization import MeanVarianceOptimizer
from lumina.optimization.black_litterman import BlackLittermanOptimizer
from lumina.attribution import FactorAttribution
from lumina.portfolio import PortfolioAnalyzer
from lumina.utils import setup_logger


def main():
    """Run advanced portfolio construction example."""
    setup_logger(log_level="INFO")

    print("=" * 60)
    print("Lumina Advanced Portfolio Construction Example")
    print("=" * 60)

    # 1. Generate Data
    print("\n1. Generating Data...")
    loader = DataLoader()
    symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "NVDA"]

    data = loader.load_synthetic(
        symbols=symbols,
        start_date="2021-01-01",
        end_date="2023-12-31",
        mu=0.0004,
        sigma=0.018,
    )

    from lumina.data import DataCleaner
    returns = DataCleaner.calculate_returns(data, method='log')
    returns = returns.dropna()  # Remove first row with NaN

    print(f"   Generated data for {len(symbols)} assets")

    # 2. Traditional Mean-Variance
    print("\n2. Mean-Variance Optimization (Baseline)...")
    mv_optimizer = MeanVarianceOptimizer(returns)
    mv_weights = mv_optimizer.max_sharpe_portfolio(risk_free_rate=0.02)

    print("\n   Mean-Variance Weights:")
    for asset, weight in mv_weights.items():
        if weight > 0.01:
            print(f"      {asset}: {weight:.2%}")

    # 3. Black-Litterman with Views
    print("\n3. Black-Litterman with Investor Views...")

    # Market caps (simulated)
    market_caps = pd.Series({
        'AAPL': 2800,
        'GOOGL': 1700,
        'MSFT': 2600,
        'AMZN': 1400,
        'NVDA': 1200,
    })

    # Investor views
    views = {
        'AAPL': 0.15,  # Expect 15% return
        'NVDA': 0.20,  # Expect 20% return
        ('GOOGL', 'MSFT'): 0.03,  # GOOGL will outperform MSFT by 3%
    }

    view_confidences = {
        'AAPL': 0.7,
        'NVDA': 0.6,
        ('GOOGL', 'MSFT'): 0.5,
    }

    bl_optimizer = BlackLittermanOptimizer(returns)
    bl_weights = bl_optimizer.optimize(
        market_caps=market_caps,
        views=views,
        view_confidences=view_confidences,
    )

    print("\n   Black-Litterman Weights:")
    for asset, weight in bl_weights.items():
        if weight > 0.01:
            print(f"      {asset}: {weight:.2%}")

    # 4. Performance Comparison
    print("\n4. Performance Comparison...")

    mv_returns = (returns * mv_weights).sum(axis=1)
    bl_returns = (returns * bl_weights).sum(axis=1)

    from lumina.utils.metrics import calculate_sharpe_ratio

    mv_sharpe = calculate_sharpe_ratio(mv_returns, risk_free_rate=0.02)
    bl_sharpe = calculate_sharpe_ratio(bl_returns, risk_free_rate=0.02)

    print(f"\n   Mean-Variance Sharpe:  {mv_sharpe:.2f}")
    print(f"   Black-Litterman Sharpe: {bl_sharpe:.2f}")

    # 5. Factor Attribution
    print("\n5. Factor Attribution...")

    attribution = FactorAttribution(bl_returns)
    results = attribution.fama_french_attribution()

    print(f"\n   Alpha (annualized):    {results['alpha']:.2%}")
    print(f"   R-squared:             {results['r_squared']:.3f}")

    print("\n   Factor Exposures:")
    for factor, beta in results['factor_exposures'].items():
        print(f"      {factor}: {beta:.3f}")

    print("\n   Factor Contributions:")
    for factor, contrib in results['factor_contributions'].items():
        print(f"      {factor}: {contrib:.2%}")

    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

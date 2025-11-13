"""
Walk-forward optimization for strategy validation.

Copyright (c) 2025 Vijeth Ltd. All rights reserved.
Author: Vithushan Jeyapahan <finance@vijeth.com>
"""

from __future__ import annotations

from typing import Callable
from datetime import timedelta

import pandas as pd
import numpy as np
from loguru import logger

from lumina.backtest.engine import BacktestEngine
from lumina.backtest.strategy import Strategy


class WalkForwardOptimizer:
    """
    Walk-forward optimization to validate strategy robustness.

    Prevents overfitting by:
    - Training on in-sample data
    - Testing on out-of-sample data
    - Rolling forward through time
    """

    def __init__(
        self,
        strategy_class: type[Strategy],
        data: pd.DataFrame,
        train_period: int = 252,
        test_period: int = 63,
        step_size: int = 63,
    ):
        """
        Initialize walk-forward optimizer.

        Args:
            strategy_class: Strategy class to optimize
            data: Historical price data
            train_period: Training window size (days)
            test_period: Testing window size (days)
            step_size: Step size for rolling forward

        Example:
            >>> wfo = WalkForwardOptimizer(MomentumStrategy, data, train_period=252)
        """
        self.strategy_class = strategy_class
        self.data = data
        self.train_period = train_period
        self.test_period = test_period
        self.step_size = step_size

        logger.info(
            f"Initialized walk-forward: train={train_period}, test={test_period}, step={step_size}"
        )

    def optimize_parameters(
        self,
        param_grid: dict,
        train_data: pd.DataFrame,
    ) -> dict:
        """
        Optimize strategy parameters on training data.

        Args:
            param_grid: Dictionary of parameters to test
            train_data: Training data

        Returns:
            Best parameters

        Example:
            >>> best_params = wfo.optimize_parameters({'lookback': [10, 20, 30]}, train_data)
        """
        best_sharpe = -np.inf
        best_params = {}

        # Generate parameter combinations
        from itertools import product

        keys = list(param_grid.keys())
        values = [param_grid[k] for k in keys]

        for combo in product(*values):
            params = dict(zip(keys, combo))

            try:
                # Create strategy with these parameters
                strategy = self.strategy_class(**params)

                # Backtest
                engine = BacktestEngine(strategy, train_data)
                results = engine.run(rebalance_frequency='W')

                # Calculate Sharpe
                sharpe = results['returns'].mean() / results['returns'].std() * np.sqrt(252)

                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = params

            except Exception as e:
                logger.warning(f"Failed for params {params}: {e}")
                continue

        logger.info(f"Best params: {best_params} (Sharpe: {best_sharpe:.2f})")

        return best_params

    def run(
        self,
        param_grid: dict,
    ) -> pd.DataFrame:
        """
        Run walk-forward optimization.

        Args:
            param_grid: Parameter grid to optimize over

        Returns:
            DataFrame with results from each fold

        Example:
            >>> results = wfo.run({'lookback_period': [10, 20, 30], 'top_n': [3, 5]})
        """
        results = []

        start_idx = self.train_period
        end_idx = len(self.data)

        fold = 0

        while start_idx + self.test_period <= end_idx:
            fold += 1

            # Define train and test windows
            train_start = start_idx - self.train_period
            train_end = start_idx
            test_end = min(start_idx + self.test_period, end_idx)

            train_data = self.data.iloc[train_start:train_end]
            test_data = self.data.iloc[start_idx:test_end]

            logger.info(f"Fold {fold}: train={len(train_data)}, test={len(test_data)}")

            # Optimize on training data
            best_params = self.optimize_parameters(param_grid, train_data)

            # Test on out-of-sample data
            strategy = self.strategy_class(**best_params)
            engine = BacktestEngine(strategy, test_data)
            test_results = engine.run(rebalance_frequency='W')

            # Store results
            results.append({
                'fold': fold,
                'train_start': train_data.index[0],
                'train_end': train_data.index[-1],
                'test_start': test_data.index[0],
                'test_end': test_data.index[-1],
                'best_params': str(best_params),
                'total_return': test_results['returns'].sum(),
                'sharpe_ratio': test_results['returns'].mean() / test_results['returns'].std() * np.sqrt(252),
                'max_drawdown': abs(test_results['cumulative_returns'].min()),
            })

            # Roll forward
            start_idx += self.step_size

        results_df = pd.DataFrame(results)
        logger.info(f"Walk-forward complete: {len(results_df)} folds")

        return results_df

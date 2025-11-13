"""
Performance decomposition and analysis.

Copyright (c) 2025 Vijeth Ltd. All rights reserved.
Author: Vithushan Jeyapahan <finance@vijeth.com>
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from loguru import logger


class PerformanceDecomposition:
    """
    Decompose portfolio performance into components.

    - Asset allocation effect
    - Security selection effect
    - Timing effect
    """

    @staticmethod
    def brinson_attribution(
        portfolio_weights: pd.DataFrame,
        portfolio_returns: pd.DataFrame,
        benchmark_weights: pd.DataFrame,
        benchmark_returns: pd.DataFrame,
    ) -> dict:
        """
        Brinson-Hood-Beebower attribution.

        Decomposes excess return into:
        - Allocation effect: (w_p - w_b) * (r_b - R_b)
        - Selection effect: w_b * (r_p - r_b)
        - Interaction: (w_p - w_b) * (r_p - r_b)

        Args:
            portfolio_weights: Portfolio weights over time
            portfolio_returns: Portfolio returns by asset
            benchmark_weights: Benchmark weights over time
            benchmark_returns: Benchmark returns by asset

        Returns:
            Dictionary with attribution results

        Example:
            >>> results = PerformanceDecomposition.brinson_attribution(
            ...     port_weights, port_returns, bench_weights, bench_returns
            ... )
        """
        # Align all dataframes
        common_index = portfolio_weights.index.intersection(portfolio_returns.index)
        common_index = common_index.intersection(benchmark_weights.index)
        common_index = common_index.intersection(benchmark_returns.index)

        pw = portfolio_weights.loc[common_index]
        pr = portfolio_returns.loc[common_index]
        bw = benchmark_weights.loc[common_index]
        br = benchmark_returns.loc[common_index]

        # Calculate benchmark total return
        benchmark_total_return = (bw * br).sum(axis=1)

        # Allocation effect
        weight_diff = pw - bw
        benchmark_excess = br.sub(benchmark_total_return, axis=0)
        allocation_effect = (weight_diff * benchmark_excess).sum(axis=1).mean()

        # Selection effect
        return_diff = pr - br
        selection_effect = (bw * return_diff).sum(axis=1).mean()

        # Interaction effect
        interaction_effect = (weight_diff * return_diff).sum(axis=1).mean()

        total_excess = allocation_effect + selection_effect + interaction_effect

        logger.info(
            f"Brinson attribution: Allocation={allocation_effect:.4f}, "
            f"Selection={selection_effect:.4f}, Interaction={interaction_effect:.4f}"
        )

        return {
            'allocation_effect': allocation_effect * 252,  # Annualized
            'selection_effect': selection_effect * 252,
            'interaction_effect': interaction_effect * 252,
            'total_excess_return': total_excess * 252,
        }

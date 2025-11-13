"""
Performance analysis for backtests.

Copyright (c) 2025 Vijeth Ltd. All rights reserved.
Author: Vithushan Jeyapahan <finance@vijeth.com>
"""

from typing import Dict, Optional
import numpy as np
import pandas as pd
from loguru import logger

from lumina.utils.metrics import (
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    calculate_calmar_ratio,
)


class PerformanceAnalyzer:
    """
    Comprehensive performance analysis for backtest results.

    Calculates various performance metrics, risk analytics, and
    provides visualization data for backtest results.
    """

    def __init__(
        self,
        results: pd.DataFrame,
        benchmark_returns: Optional[pd.Series] = None,
    ):
        """
        Initialize performance analyzer.

        Args:
            results: Backtest results DataFrame with 'returns' column
            benchmark_returns: Optional benchmark returns for comparison

        Example:
            >>> analyzer = PerformanceAnalyzer(backtest_results)
        """
        if 'returns' not in results.columns:
            raise ValueError("Results must contain 'returns' column")

        self.results = results
        self.returns = results['returns'].dropna()
        self.benchmark_returns = benchmark_returns

        logger.info(f"Initialized PerformanceAnalyzer with {len(self.returns)} return periods")

    def calculate_metrics(
        self,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252,
    ) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics.

        Args:
            risk_free_rate: Annual risk-free rate
            periods_per_year: Periods per year (252 for daily)

        Returns:
            Dictionary of performance metrics

        Example:
            >>> metrics = analyzer.calculate_metrics(risk_free_rate=0.02)
        """
        returns = self.returns

        # Return metrics
        total_return = (1 + returns).prod() - 1
        annual_return = (1 + returns).prod() ** (periods_per_year / len(returns)) - 1

        # Risk metrics
        volatility = returns.std() * np.sqrt(periods_per_year)

        # Risk-adjusted metrics
        sharpe_ratio = calculate_sharpe_ratio(returns, risk_free_rate, periods_per_year)
        sortino_ratio = calculate_sortino_ratio(returns, risk_free_rate, periods_per_year)

        # Drawdown metrics
        max_drawdown = calculate_max_drawdown(returns)
        calmar_ratio = calculate_calmar_ratio(returns, periods_per_year)

        # Additional metrics
        win_rate = (returns > 0).sum() / len(returns)
        best_day = returns.max()
        worst_day = returns.min()

        # Skewness and kurtosis
        skewness = returns.skew()
        kurtosis = returns.kurtosis()

        metrics = {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'best_day': best_day,
            'worst_day': worst_day,
            'skewness': skewness,
            'kurtosis': kurtosis,
        }

        # Benchmark comparison
        if self.benchmark_returns is not None:
            aligned_benchmark = self.benchmark_returns.reindex(returns.index).dropna()
            aligned_returns = returns.reindex(aligned_benchmark.index)

            # Tracking error
            tracking_error = (
                (aligned_returns - aligned_benchmark).std() *
                np.sqrt(periods_per_year)
            )

            # Information ratio
            excess_returns = aligned_returns - aligned_benchmark
            information_ratio = (
                excess_returns.mean() / excess_returns.std() *
                np.sqrt(periods_per_year)
            )

            # Beta
            covariance = np.cov(aligned_returns, aligned_benchmark)[0, 1]
            benchmark_variance = aligned_benchmark.var()
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 0

            # Alpha
            benchmark_return = aligned_benchmark.mean() * periods_per_year
            alpha = annual_return - (risk_free_rate + beta * (benchmark_return - risk_free_rate))

            metrics.update({
                'tracking_error': tracking_error,
                'information_ratio': information_ratio,
                'beta': beta,
                'alpha': alpha,
            })

        logger.info(
            f"Performance: Return={annual_return:.2%}, Vol={volatility:.2%}, "
            f"Sharpe={sharpe_ratio:.2f}, MaxDD={max_drawdown:.2%}"
        )

        return metrics

    def monthly_returns(self) -> pd.DataFrame:
        """
        Calculate monthly returns matrix.

        Returns:
            DataFrame with years as rows and months as columns

        Example:
            >>> monthly = analyzer.monthly_returns()
        """
        returns_series = self.returns.copy()
        returns_series.index = pd.to_datetime(returns_series.index)

        monthly = returns_series.resample('M').apply(lambda x: (1 + x).prod() - 1)

        # Pivot to year x month matrix
        monthly_df = pd.DataFrame({
            'year': monthly.index.year,
            'month': monthly.index.month,
            'return': monthly.values,
        })

        monthly_matrix = monthly_df.pivot(
            index='year',
            columns='month',
            values='return',
        )

        # Add yearly totals
        monthly_matrix['YTD'] = monthly_matrix.apply(
            lambda row: (1 + row.dropna()).prod() - 1,
            axis=1,
        )

        logger.debug("Calculated monthly returns matrix")

        return monthly_matrix

    def rolling_metrics(
        self,
        window: int = 252,
        risk_free_rate: float = 0.0,
    ) -> pd.DataFrame:
        """
        Calculate rolling performance metrics.

        Args:
            window: Rolling window size
            risk_free_rate: Annual risk-free rate

        Returns:
            DataFrame with rolling metrics

        Example:
            >>> rolling = analyzer.rolling_metrics(window=252)
        """
        returns = self.returns

        rolling_df = pd.DataFrame(index=returns.index)

        # Rolling return
        rolling_df['rolling_return'] = (
            returns.rolling(window=window).apply(lambda x: (1 + x).prod() - 1)
        )

        # Rolling volatility
        rolling_df['rolling_volatility'] = (
            returns.rolling(window=window).std() * np.sqrt(252)
        )

        # Rolling Sharpe ratio
        rolling_df['rolling_sharpe'] = (
            returns.rolling(window=window).apply(
                lambda x: calculate_sharpe_ratio(x, risk_free_rate, 252)
            )
        )

        # Rolling max drawdown
        rolling_df['rolling_max_drawdown'] = (
            returns.rolling(window=window).apply(calculate_max_drawdown)
        )

        logger.debug(f"Calculated rolling metrics with {window}-period window")

        return rolling_df

    def drawdown_periods(self) -> pd.DataFrame:
        """
        Identify and analyze drawdown periods.

        Returns:
            DataFrame with drawdown periods and statistics

        Example:
            >>> drawdowns = analyzer.drawdown_periods()
        """
        cumulative = (1 + self.returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max

        # Identify drawdown periods
        is_drawdown = drawdown < 0
        drawdown_starts = is_drawdown & ~is_drawdown.shift(1, fill_value=False)
        drawdown_ends = ~is_drawdown & is_drawdown.shift(1, fill_value=False)

        periods = []
        start_idx = None

        for date, is_start in drawdown_starts.items():
            if is_start:
                start_idx = date

        for date, is_end in drawdown_ends.items():
            if is_end and start_idx is not None:
                dd_period = drawdown[start_idx:date]
                periods.append({
                    'start': start_idx,
                    'end': date,
                    'duration': len(dd_period),
                    'max_drawdown': dd_period.min(),
                })
                start_idx = None

        periods_df = pd.DataFrame(periods)

        if not periods_df.empty:
            periods_df = periods_df.sort_values('max_drawdown')

        logger.debug(f"Identified {len(periods_df)} drawdown periods")

        return periods_df

    def summary_statistics(self) -> str:
        """
        Generate formatted summary statistics.

        Returns:
            Formatted string with summary statistics

        Example:
            >>> print(analyzer.summary_statistics())
        """
        metrics = self.calculate_metrics()

        summary = f"""
Backtest Performance Summary
{'=' * 60}

Return Metrics:
  Total Return:          {metrics['total_return']:>10.2%}
  Annual Return:         {metrics['annual_return']:>10.2%}
  Volatility (Annual):   {metrics['volatility']:>10.2%}

Risk-Adjusted Returns:
  Sharpe Ratio:          {metrics['sharpe_ratio']:>10.2f}
  Sortino Ratio:         {metrics['sortino_ratio']:>10.2f}
  Calmar Ratio:          {metrics['calmar_ratio']:>10.2f}

Risk Metrics:
  Maximum Drawdown:      {metrics['max_drawdown']:>10.2%}
  Best Day:              {metrics['best_day']:>10.2%}
  Worst Day:             {metrics['worst_day']:>10.2%}

Distribution:
  Win Rate:              {metrics['win_rate']:>10.2%}
  Skewness:              {metrics['skewness']:>10.2f}
  Kurtosis:              {metrics['kurtosis']:>10.2f}
"""

        if 'alpha' in metrics:
            summary += f"""
Benchmark Comparison:
  Alpha:                 {metrics['alpha']:>10.2%}
  Beta:                  {metrics['beta']:>10.2f}
  Information Ratio:     {metrics['information_ratio']:>10.2f}
  Tracking Error:        {metrics['tracking_error']:>10.2%}
"""

        summary += "=" * 60

        return summary

"""
Stress testing and scenario analysis for risk management.

Copyright (c) 2025 Vijeth Ltd. All rights reserved.
Author: Vithushan Jeyapahan <finance@vijeth.com>
"""

from __future__ import annotations

from typing import Optional
from dataclasses import dataclass

import pandas as pd
import numpy as np
from loguru import logger


@dataclass
class StressScenario:
    """
    Stress test scenario definition.

    Example:
        >>> scenario = StressScenario(
        ...     name='Market Crash',
        ...     returns={'SPY': -0.20, 'TLT': 0.05},
        ...     correlation_shock=1.0,
        ... )
    """
    name: str
    returns: dict[str, float]  # Asset -> return shock
    correlation_shock: float = 0.0  # Increase in correlations
    volatility_multiplier: float = 1.0  # Volatility scaling factor


class StressTester:
    """
    Comprehensive stress testing framework.

    Features:
    - Historical scenario replay
    - Hypothetical scenarios
    - Correlation stress
    - Volatility stress
    - Portfolio impact analysis

    Example:
        >>> tester = StressTester(portfolio_weights, returns_data)
        >>> results = tester.run_scenario(market_crash_scenario)
        >>> print(f"Portfolio loss: {results['portfolio_return']:.2%}")
    """

    def __init__(
        self,
        portfolio_weights: pd.Series,
        returns: pd.DataFrame,
        cov_matrix: Optional[pd.DataFrame] = None,
    ):
        """
        Initialize stress tester.

        Args:
            portfolio_weights: Current portfolio weights
            returns: Historical returns
            cov_matrix: Covariance matrix (computed if not provided)

        Example:
            >>> tester = StressTester(weights, returns)
        """
        self.weights = portfolio_weights
        self.returns = returns
        self.cov_matrix = cov_matrix if cov_matrix is not None else returns.cov()

        logger.info(f"Initialized StressTester for {len(portfolio_weights)} assets")

    def run_scenario(
        self,
        scenario: StressScenario,
    ) -> dict:
        """
        Run a stress test scenario.

        Args:
            scenario: Scenario definition

        Returns:
            Dictionary with stress test results

        Example:
            >>> crash = StressScenario('Crash', {'SPY': -0.25, 'TLT': 0.10})
            >>> results = tester.run_scenario(crash)
        """
        # Calculate asset returns under scenario
        asset_returns = pd.Series(0.0, index=self.weights.index)

        for asset, shock in scenario.returns.items():
            if asset in asset_returns.index:
                asset_returns[asset] = shock

        # Apply correlation stress
        if scenario.correlation_shock != 0:
            stressed_cov = self._stress_correlations(
                self.cov_matrix,
                scenario.correlation_shock,
            )
        else:
            stressed_cov = self.cov_matrix

        # Apply volatility stress
        if scenario.volatility_multiplier != 1.0:
            stressed_cov = stressed_cov * (scenario.volatility_multiplier ** 2)

        # Calculate portfolio impact
        portfolio_return = (self.weights * asset_returns).sum()
        portfolio_var = np.dot(self.weights, np.dot(stressed_cov, self.weights))
        portfolio_vol = np.sqrt(portfolio_var)

        # Calculate marginal contributions
        marginal_contrib = np.dot(stressed_cov, self.weights) / portfolio_vol
        contrib_to_risk = self.weights * marginal_contrib

        results = {
            'scenario_name': scenario.name,
            'portfolio_return': portfolio_return,
            'portfolio_volatility': portfolio_vol,
            'asset_returns': asset_returns.to_dict(),
            'contribution_to_risk': contrib_to_risk.to_dict(),
            'worst_asset': asset_returns.idxmin(),
            'worst_return': asset_returns.min(),
        }

        logger.info(
            f"Scenario '{scenario.name}': portfolio return = {portfolio_return:.2%}, "
            f"volatility = {portfolio_vol:.2%}"
        )

        return results

    def _stress_correlations(
        self,
        cov_matrix: pd.DataFrame,
        shock: float,
    ) -> pd.DataFrame:
        """
        Apply correlation stress (correlations move toward 1).

        Args:
            cov_matrix: Original covariance matrix
            shock: Correlation shock (0 to 1, where 1 = all correlations go to 1)

        Returns:
            Stressed covariance matrix
        """
        # Extract correlation matrix
        vol = np.sqrt(np.diag(cov_matrix.values))
        corr_matrix = cov_matrix.values / np.outer(vol, vol)

        # Stress correlations toward 1
        stressed_corr = corr_matrix * (1 - shock) + shock

        # Ensure diagonal stays at 1
        np.fill_diagonal(stressed_corr, 1.0)

        # Convert back to covariance
        stressed_cov = stressed_corr * np.outer(vol, vol)

        return pd.DataFrame(stressed_cov, index=cov_matrix.index, columns=cov_matrix.columns)

    def run_historical_scenario(
        self,
        start_date: str,
        end_date: str,
    ) -> dict:
        """
        Replay historical period as stress scenario.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            Dictionary with scenario results

        Example:
            >>> results = tester.run_historical_scenario('2008-09-01', '2008-12-31')
        """
        period_returns = self.returns.loc[start_date:end_date]

        if period_returns.empty:
            raise ValueError(f"No data for period {start_date} to {end_date}")

        # Calculate cumulative returns for the period
        cumulative_returns = (1 + period_returns).prod() - 1

        scenario = StressScenario(
            name=f"Historical: {start_date} to {end_date}",
            returns=cumulative_returns.to_dict(),
        )

        return self.run_scenario(scenario)

    def run_multiple_scenarios(
        self,
        scenarios: list[StressScenario],
    ) -> pd.DataFrame:
        """
        Run multiple stress scenarios.

        Args:
            scenarios: List of scenarios

        Returns:
            DataFrame with results for all scenarios

        Example:
            >>> scenarios = [market_crash, rate_shock, inflation_shock]
            >>> results = tester.run_multiple_scenarios(scenarios)
        """
        results = []

        for scenario in scenarios:
            result = self.run_scenario(scenario)
            results.append(result)

        df = pd.DataFrame(results)
        df = df.set_index('scenario_name')

        logger.info(f"Completed {len(scenarios)} stress scenarios")

        return df

    def correlation_sensitivity(
        self,
        correlation_range: tuple[float, float] = (0.0, 1.0),
        n_points: int = 10,
    ) -> pd.DataFrame:
        """
        Analyze portfolio sensitivity to correlation changes.

        Args:
            correlation_range: Range of correlation shocks to test
            n_points: Number of points to test

        Returns:
            DataFrame with correlation shock results

        Example:
            >>> sensitivity = tester.correlation_sensitivity()
        """
        shocks = np.linspace(correlation_range[0], correlation_range[1], n_points)
        results = []

        for shock in shocks:
            stressed_cov = self._stress_correlations(self.cov_matrix, shock)
            portfolio_var = np.dot(self.weights, np.dot(stressed_cov, self.weights))
            portfolio_vol = np.sqrt(portfolio_var)

            results.append({
                'correlation_shock': shock,
                'portfolio_volatility': portfolio_vol,
                'vol_change_pct': (portfolio_vol / np.sqrt(np.dot(self.weights, np.dot(self.cov_matrix, self.weights))) - 1) * 100,
            })

        df = pd.DataFrame(results)

        logger.info(
            f"Correlation sensitivity: vol ranges from {df['portfolio_volatility'].min():.2%} "
            f"to {df['portfolio_volatility'].max():.2%}"
        )

        return df


# Predefined stress scenarios
PREDEFINED_SCENARIOS = {
    'market_crash': StressScenario(
        name='Market Crash (-25%)',
        returns={
            'SPY': -0.25,
            'QQQ': -0.30,
            'IWM': -0.28,
            'TLT': 0.10,
            'GLD': 0.05,
        },
        correlation_shock=0.5,
        volatility_multiplier=2.0,
    ),
    'rate_shock': StressScenario(
        name='Rate Shock (+200bps)',
        returns={
            'SPY': -0.10,
            'TLT': -0.15,
            'HYG': -0.12,
            'GLD': 0.02,
        },
        correlation_shock=0.3,
    ),
    'inflation_shock': StressScenario(
        name='Inflation Shock',
        returns={
            'SPY': -0.08,
            'TLT': -0.12,
            'TIP': 0.05,
            'GLD': 0.15,
            'DBC': 0.20,
        },
    ),
    'credit_crisis': StressScenario(
        name='Credit Crisis',
        returns={
            'SPY': -0.20,
            'HYG': -0.30,
            'LQD': -0.15,
            'TLT': 0.15,
            'GLD': 0.10,
        },
        correlation_shock=0.7,
        volatility_multiplier=3.0,
    ),
    'volatility_spike': StressScenario(
        name='Volatility Spike',
        returns={},  # No return shock, just vol
        volatility_multiplier=2.5,
        correlation_shock=0.4,
    ),
}


class ScenarioGenerator:
    """
    Generate stress scenarios based on historical data.

    Example:
        >>> generator = ScenarioGenerator(returns_data)
        >>> worst_month = generator.worst_historical_period(window=21)
        >>> scenario = generator.to_scenario(worst_month)
    """

    def __init__(self, returns: pd.DataFrame):
        """
        Initialize scenario generator.

        Args:
            returns: Historical returns data
        """
        self.returns = returns

    def worst_historical_period(
        self,
        window: int = 21,
        metric: str = 'return',
    ) -> dict:
        """
        Find worst historical period.

        Args:
            window: Rolling window size (days)
            metric: 'return' or 'volatility'

        Returns:
            Dictionary with period information

        Example:
            >>> worst = generator.worst_historical_period(window=252)
        """
        if metric == 'return':
            rolling_returns = self.returns.rolling(window).sum()
            portfolio_returns = rolling_returns.mean(axis=1)
            worst_idx = portfolio_returns.idxmin()
            worst_value = portfolio_returns.min()

            start_date = worst_idx - pd.Timedelta(days=window)
            period_returns = self.returns.loc[start_date:worst_idx].sum()

        elif metric == 'volatility':
            rolling_vol = self.returns.rolling(window).std()
            portfolio_vol = rolling_vol.mean(axis=1)
            worst_idx = portfolio_vol.idxmax()
            worst_value = portfolio_vol.max()

            start_date = worst_idx - pd.Timedelta(days=window)
            period_returns = self.returns.loc[start_date:worst_idx].sum()

        else:
            raise ValueError(f"Unknown metric: {metric}")

        return {
            'start_date': start_date,
            'end_date': worst_idx,
            'metric': metric,
            'value': worst_value,
            'returns': period_returns.to_dict(),
        }

    def to_scenario(
        self,
        period_info: dict,
        name: Optional[str] = None,
    ) -> StressScenario:
        """
        Convert historical period to stress scenario.

        Args:
            period_info: Period information from worst_historical_period()
            name: Optional scenario name

        Returns:
            StressScenario

        Example:
            >>> worst = generator.worst_historical_period()
            >>> scenario = generator.to_scenario(worst)
        """
        if name is None:
            name = f"Historical worst ({period_info['start_date']} to {period_info['end_date']})"

        return StressScenario(
            name=name,
            returns=period_info['returns'],
        )

"""
Transaction cost models for realistic backtesting.

Copyright (c) 2025 Vijeth Ltd. All rights reserved.
Author: Vithushan Jeyapahan <finance@vijeth.com>
"""

from __future__ import annotations

from typing import Optional
from enum import Enum

import pandas as pd
import numpy as np
from loguru import logger


class SlippageModel(Enum):
    """Slippage modeling approaches."""
    NONE = "none"
    FIXED = "fixed"
    VOLUME_BASED = "volume_based"
    SQRT_VOLUME = "sqrt_volume"
    LINEAR_IMPACT = "linear_impact"


class TransactionCostModel:
    """
    Comprehensive transaction cost model.

    Includes:
    - Fixed commission
    - Percentage commission
    - Minimum commission
    - Bid-ask spread
    - Market impact (slippage)

    Example:
        >>> model = TransactionCostModel(
        ...     commission_pct=0.001,
        ...     min_commission=1.0,
        ...     spread_bps=5.0,
        ... )
        >>> cost = model.calculate_cost(price=100, shares=1000, side='buy')
    """

    def __init__(
        self,
        commission_pct: float = 0.001,
        commission_fixed: float = 0.0,
        min_commission: float = 0.0,
        spread_bps: float = 5.0,
        slippage_model: SlippageModel = SlippageModel.FIXED,
        slippage_bps: float = 5.0,
        slippage_coeff: float = 0.1,
    ):
        """
        Initialize transaction cost model.

        Args:
            commission_pct: Commission as percentage of trade value (e.g., 0.001 = 0.1%)
            commission_fixed: Fixed commission per trade
            min_commission: Minimum commission per trade
            spread_bps: Bid-ask spread in basis points
            slippage_model: Slippage calculation model
            slippage_bps: Fixed slippage in basis points (for FIXED model)
            slippage_coeff: Coefficient for volume-based slippage

        Example:
            >>> # Interactive Brokers-like costs
            >>> model = TransactionCostModel(
            ...     commission_pct=0.0005,
            ...     min_commission=1.0,
            ...     spread_bps=2.0,
            ... )
        """
        self.commission_pct = commission_pct
        self.commission_fixed = commission_fixed
        self.min_commission = min_commission
        self.spread_bps = spread_bps
        self.slippage_model = slippage_model
        self.slippage_bps = slippage_bps
        self.slippage_coeff = slippage_coeff

        logger.info(
            f"TransactionCostModel: commission={commission_pct:.4f}, "
            f"spread={spread_bps}bps, slippage={slippage_model.value}"
        )

    def calculate_commission(
        self,
        trade_value: float,
    ) -> float:
        """
        Calculate commission cost.

        Args:
            trade_value: Absolute value of trade (price * shares)

        Returns:
            Commission cost

        Example:
            >>> cost = model.calculate_commission(trade_value=10000)
        """
        commission = self.commission_fixed + self.commission_pct * trade_value
        commission = max(commission, self.min_commission)
        return commission

    def calculate_spread_cost(
        self,
        price: float,
        shares: float,
        side: str,
    ) -> float:
        """
        Calculate bid-ask spread cost.

        Args:
            price: Mid price
            shares: Number of shares (absolute value)
            side: 'buy' or 'sell'

        Returns:
            Spread cost

        Example:
            >>> cost = model.calculate_spread_cost(price=100, shares=1000, side='buy')
        """
        spread_pct = self.spread_bps / 10000.0

        if side.lower() == 'buy':
            # Pay half spread to get to ask price
            effective_price = price * (1 + spread_pct / 2)
        else:  # sell
            # Lose half spread to get to bid price
            effective_price = price * (1 - spread_pct / 2)

        spread_cost = abs(effective_price - price) * abs(shares)
        return spread_cost

    def calculate_slippage(
        self,
        price: float,
        shares: float,
        volume: Optional[float] = None,
        volatility: Optional[float] = None,
    ) -> float:
        """
        Calculate slippage (market impact).

        Args:
            price: Current price
            shares: Number of shares to trade (absolute value)
            volume: Average daily volume (for volume-based models)
            volatility: Price volatility (for impact models)

        Returns:
            Slippage cost

        Example:
            >>> cost = model.calculate_slippage(
            ...     price=100,
            ...     shares=1000,
            ...     volume=1000000,
            ... )
        """
        if self.slippage_model == SlippageModel.NONE:
            return 0.0

        elif self.slippage_model == SlippageModel.FIXED:
            # Fixed slippage in basis points
            slippage_pct = self.slippage_bps / 10000.0
            return price * abs(shares) * slippage_pct

        elif self.slippage_model == SlippageModel.VOLUME_BASED:
            # Slippage proportional to percentage of daily volume
            if volume is None or volume <= 0:
                logger.warning("Volume required for VOLUME_BASED slippage, using FIXED")
                return self.calculate_slippage(price, shares, None, None)

            participation_rate = abs(shares) / volume
            slippage_pct = self.slippage_coeff * participation_rate
            return price * abs(shares) * slippage_pct

        elif self.slippage_model == SlippageModel.SQRT_VOLUME:
            # Square root model: impact ~ sqrt(shares / volume)
            if volume is None or volume <= 0:
                logger.warning("Volume required for SQRT_VOLUME slippage, using FIXED")
                return self.calculate_slippage(price, shares, None, None)

            participation_rate = abs(shares) / volume
            slippage_pct = self.slippage_coeff * np.sqrt(participation_rate)
            return price * abs(shares) * slippage_pct

        elif self.slippage_model == SlippageModel.LINEAR_IMPACT:
            # Linear price impact model
            if volatility is None:
                logger.warning("Volatility required for LINEAR_IMPACT, using FIXED")
                return self.calculate_slippage(price, shares, None, None)

            if volume is None or volume <= 0:
                volume = 1e6  # Default assumption

            # Almgren-Chriss model approximation
            participation_rate = abs(shares) / volume
            impact = self.slippage_coeff * volatility * np.sign(shares) * participation_rate
            return price * abs(shares) * abs(impact)

        else:
            raise ValueError(f"Unknown slippage model: {self.slippage_model}")

    def calculate_total_cost(
        self,
        price: float,
        shares: float,
        side: str,
        volume: Optional[float] = None,
        volatility: Optional[float] = None,
    ) -> dict:
        """
        Calculate total transaction cost.

        Args:
            price: Execution price
            shares: Number of shares (absolute value)
            side: 'buy' or 'sell'
            volume: Average daily volume
            volatility: Price volatility

        Returns:
            Dictionary with cost breakdown

        Example:
            >>> costs = model.calculate_total_cost(
            ...     price=100,
            ...     shares=1000,
            ...     side='buy',
            ...     volume=1000000,
            ... )
            >>> print(f"Total cost: ${costs['total']:.2f}")
        """
        trade_value = abs(price * shares)

        commission = self.calculate_commission(trade_value)
        spread = self.calculate_spread_cost(price, shares, side)
        slippage = self.calculate_slippage(price, shares, volume, volatility)

        total = commission + spread + slippage

        return {
            'commission': commission,
            'spread': spread,
            'slippage': slippage,
            'total': total,
            'bps': (total / trade_value) * 10000 if trade_value > 0 else 0,
        }


class TieredCommissionModel(TransactionCostModel):
    """
    Tiered commission model (volume discounts).

    Example:
        >>> model = TieredCommissionModel(
        ...     tiers=[
        ...         (10000, 0.001),   # Up to 10k: 0.1%
        ...         (100000, 0.0008), # 10k-100k: 0.08%
        ...         (float('inf'), 0.0005),  # >100k: 0.05%
        ...     ]
        ... )
    """

    def __init__(
        self,
        tiers: list[tuple[float, float]],
        **kwargs,
    ):
        """
        Initialize tiered commission model.

        Args:
            tiers: List of (threshold, rate) tuples
            **kwargs: Additional arguments for base class

        Example:
            >>> tiers = [
            ...     (50000, 0.001),
            ...     (500000, 0.0005),
            ...     (float('inf'), 0.0002),
            ... ]
        """
        super().__init__(**kwargs)
        self.tiers = sorted(tiers, key=lambda x: x[0])

        logger.info(f"TieredCommissionModel with {len(self.tiers)} tiers")

    def calculate_commission(self, trade_value: float) -> float:
        """Calculate commission using tiered structure."""
        # Find applicable tier
        commission_pct = self.commission_pct
        for threshold, rate in self.tiers:
            if trade_value <= threshold:
                commission_pct = rate
                break

        commission = self.commission_fixed + commission_pct * trade_value
        commission = max(commission, self.min_commission)
        return commission


class RealizedCosts:
    """
    Track and analyze realized transaction costs.

    Example:
        >>> costs = RealizedCosts()
        >>> costs.add_trade(
        ...     timestamp='2023-01-01',
        ...     symbol='AAPL',
        ...     price=150.0,
        ...     shares=100,
        ...     side='buy',
        ...     costs={'total': 15.50},
        ... )
        >>> summary = costs.get_summary()
    """

    def __init__(self):
        """Initialize cost tracker."""
        self.trades = []

    def add_trade(
        self,
        timestamp: pd.Timestamp,
        symbol: str,
        price: float,
        shares: float,
        side: str,
        costs: dict,
    ) -> None:
        """
        Record a trade and its costs.

        Args:
            timestamp: Trade timestamp
            symbol: Asset symbol
            price: Execution price
            shares: Number of shares
            side: 'buy' or 'sell'
            costs: Cost breakdown from calculate_total_cost()
        """
        self.trades.append({
            'timestamp': timestamp,
            'symbol': symbol,
            'price': price,
            'shares': shares,
            'side': side,
            **costs,
        })

    def get_dataframe(self) -> pd.DataFrame:
        """Get all trades as DataFrame."""
        if not self.trades:
            return pd.DataFrame()

        df = pd.DataFrame(self.trades)
        df = df.set_index('timestamp')
        return df

    def get_summary(self) -> dict:
        """
        Get summary statistics of transaction costs.

        Returns:
            Dictionary with cost statistics

        Example:
            >>> summary = costs.get_summary()
            >>> print(f"Total costs: ${summary['total_costs']:.2f}")
            >>> print(f"Average cost: {summary['avg_bps']:.1f} bps")
        """
        if not self.trades:
            return {}

        df = self.get_dataframe()

        total_volume = (df['price'] * df['shares'].abs()).sum()

        return {
            'num_trades': len(df),
            'total_volume': total_volume,
            'total_costs': df['total'].sum(),
            'total_commission': df['commission'].sum(),
            'total_spread': df['spread'].sum(),
            'total_slippage': df['slippage'].sum(),
            'avg_bps': df['bps'].mean(),
            'median_bps': df['bps'].median(),
            'costs_pct_of_volume': (df['total'].sum() / total_volume * 100) if total_volume > 0 else 0,
        }

    def get_costs_by_symbol(self) -> pd.DataFrame:
        """Get cost breakdown by symbol."""
        if not self.trades:
            return pd.DataFrame()

        df = self.get_dataframe()
        return df.groupby('symbol').agg({
            'total': 'sum',
            'commission': 'sum',
            'spread': 'sum',
            'slippage': 'sum',
            'bps': 'mean',
            'shares': lambda x: x.abs().sum(),
        })

    def get_costs_over_time(self, freq: str = 'D') -> pd.DataFrame:
        """
        Get cost time series.

        Args:
            freq: Resampling frequency ('D', 'W', 'M', etc.)

        Returns:
            DataFrame with costs aggregated by time period
        """
        if not self.trades:
            return pd.DataFrame()

        df = self.get_dataframe()
        return df.resample(freq).agg({
            'total': 'sum',
            'commission': 'sum',
            'spread': 'sum',
            'slippage': 'sum',
            'bps': 'mean',
        })


# Predefined cost models for common brokers
BROKER_MODELS = {
    'interactive_brokers': TransactionCostModel(
        commission_pct=0.0005,
        min_commission=1.0,
        spread_bps=2.0,
        slippage_bps=3.0,
    ),
    'charles_schwab': TransactionCostModel(
        commission_pct=0.0,
        commission_fixed=0.0,
        spread_bps=5.0,
        slippage_bps=5.0,
    ),
    'robinhood': TransactionCostModel(
        commission_pct=0.0,
        commission_fixed=0.0,
        spread_bps=10.0,  # Higher spread due to payment for order flow
        slippage_bps=5.0,
    ),
    'institutional': TransactionCostModel(
        commission_pct=0.0002,
        spread_bps=1.0,
        slippage_model=SlippageModel.SQRT_VOLUME,
        slippage_coeff=0.1,
    ),
    'none': TransactionCostModel(
        commission_pct=0.0,
        commission_fixed=0.0,
        spread_bps=0.0,
        slippage_model=SlippageModel.NONE,
    ),
}

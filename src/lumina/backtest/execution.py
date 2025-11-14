"""
Order execution simulation for backtesting.

Copyright (c) 2025 Vijeth Ltd. All rights reserved.
Author: Vithushan Jeyapahan <finance@vijeth.com>
"""

from __future__ import annotations

from enum import Enum
from typing import Optional
from dataclasses import dataclass

import pandas as pd
import numpy as np
from loguru import logger


class OrderType(Enum):
    """Order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(Enum):
    """Order status."""
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


@dataclass
class Order:
    """
    Trading order.

    Example:
        >>> order = Order(
        ...     symbol='AAPL',
        ...     order_type=OrderType.LIMIT,
        ...     side='buy',
        ...     quantity=100,
        ...     limit_price=150.0,
        ... )
    """
    symbol: str
    order_type: OrderType
    side: str  # 'buy' or 'sell'
    quantity: float
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = 'GTC'  # GTC, DAY, IOC, FOK
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    filled_price: Optional[float] = None
    submitted_time: Optional[pd.Timestamp] = None
    filled_time: Optional[pd.Timestamp] = None


class ExecutionSimulator:
    """
    Simulate order execution with realistic fills.

    Features:
    - Market, limit, stop orders
    - Partial fills
    - Price improvement
    - Fill probability based on volume

    Example:
        >>> simulator = ExecutionSimulator()
        >>> order = Order('AAPL', OrderType.LIMIT, 'buy', 100, limit_price=150)
        >>> fill = simulator.simulate_fill(order, current_bar)
    """

    def __init__(
        self,
        fill_probability: float = 1.0,
        partial_fill_threshold: float = 0.1,
        price_improvement_prob: float = 0.3,
    ):
        """
        Initialize execution simulator.

        Args:
            fill_probability: Probability of fill for market orders
            partial_fill_threshold: Volume threshold for partial fills (as fraction of ADV)
            price_improvement_prob: Probability of getting price improvement
        """
        self.fill_probability = fill_probability
        self.partial_fill_threshold = partial_fill_threshold
        self.price_improvement_prob = price_improvement_prob

    def simulate_fill(
        self,
        order: Order,
        bar: pd.Series,
        volume: Optional[float] = None,
    ) -> Order:
        """
        Simulate order execution for one bar.

        Args:
            order: Order to execute
            bar: Price bar with open, high, low, close
            volume: Trading volume (for partial fill logic)

        Returns:
            Updated order with fill information

        Example:
            >>> bar = pd.Series({'open': 100, 'high': 102, 'low': 99, 'close': 101})
            >>> filled_order = simulator.simulate_fill(order, bar)
        """
        if order.status != OrderStatus.PENDING:
            return order

        if order.order_type == OrderType.MARKET:
            return self._fill_market_order(order, bar, volume)
        elif order.order_type == OrderType.LIMIT:
            return self._fill_limit_order(order, bar, volume)
        elif order.order_type == OrderType.STOP:
            return self._fill_stop_order(order, bar, volume)
        elif order.order_type == OrderType.STOP_LIMIT:
            return self._fill_stop_limit_order(order, bar, volume)
        else:
            raise ValueError(f"Unknown order type: {order.order_type}")

    def _fill_market_order(
        self,
        order: Order,
        bar: pd.Series,
        volume: Optional[float],
    ) -> Order:
        """Fill market order at open price."""
        # Market orders fill immediately at next open
        fill_price = bar['open']

        # Check for partial fill based on volume
        if volume is not None and volume > 0:
            participation_rate = order.quantity / volume
            if participation_rate > self.partial_fill_threshold:
                # Large order relative to volume - partial fill
                fill_quantity = order.quantity * np.random.uniform(0.5, 0.9)
                order.status = OrderStatus.PARTIAL
            else:
                fill_quantity = order.quantity
                order.status = OrderStatus.FILLED
        else:
            # No volume data - assume full fill
            fill_quantity = order.quantity
            order.status = OrderStatus.FILLED

        order.filled_quantity = fill_quantity
        order.filled_price = fill_price

        return order

    def _fill_limit_order(
        self,
        order: Order,
        bar: pd.Series,
        volume: Optional[float],
    ) -> Order:
        """Fill limit order if price reaches limit."""
        if order.limit_price is None:
            raise ValueError("Limit order must have limit_price")

        filled = False

        if order.side == 'buy':
            # Buy limit: fill if price drops to or below limit
            if bar['low'] <= order.limit_price:
                filled = True
                # Fill at limit or better (price improvement)
                if np.random.random() < self.price_improvement_prob:
                    # Price improvement between low and limit
                    fill_price = np.random.uniform(bar['low'], order.limit_price)
                else:
                    fill_price = order.limit_price
        else:  # sell
            # Sell limit: fill if price rises to or above limit
            if bar['high'] >= order.limit_price:
                filled = True
                # Fill at limit or better
                if np.random.random() < self.price_improvement_prob:
                    fill_price = np.random.uniform(order.limit_price, bar['high'])
                else:
                    fill_price = order.limit_price

        if filled:
            order.filled_quantity = order.quantity
            order.filled_price = fill_price
            order.status = OrderStatus.FILLED

        return order

    def _fill_stop_order(
        self,
        order: Order,
        bar: pd.Series,
        volume: Optional[float],
    ) -> Order:
        """Fill stop order when price reaches stop level."""
        if order.stop_price is None:
            raise ValueError("Stop order must have stop_price")

        triggered = False

        if order.side == 'buy':
            # Buy stop: trigger if price rises to or above stop
            if bar['high'] >= order.stop_price:
                triggered = True
                # Fill at stop or worse (slippage)
                fill_price = max(order.stop_price, bar['open'])
        else:  # sell
            # Sell stop: trigger if price drops to or below stop
            if bar['low'] <= order.stop_price:
                triggered = True
                # Fill at stop or worse
                fill_price = min(order.stop_price, bar['open'])

        if triggered:
            order.filled_quantity = order.quantity
            order.filled_price = fill_price
            order.status = OrderStatus.FILLED

        return order

    def _fill_stop_limit_order(
        self,
        order: Order,
        bar: pd.Series,
        volume: Optional[float],
    ) -> Order:
        """Fill stop-limit order (stop triggers, then limit order)."""
        if order.stop_price is None or order.limit_price is None:
            raise ValueError("Stop-limit order must have both stop_price and limit_price")

        # First check if stop is triggered
        triggered = False

        if order.side == 'buy':
            if bar['high'] >= order.stop_price:
                triggered = True
        else:
            if bar['low'] <= order.stop_price:
                triggered = True

        if not triggered:
            return order

        # Stop triggered, now treat as limit order
        order.order_type = OrderType.LIMIT
        return self._fill_limit_order(order, bar, volume)

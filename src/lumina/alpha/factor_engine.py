"""
Factor engineering and extraction.

Copyright (c) 2025 Vijeth Ltd. All rights reserved.
Author: Vithushan Jeyapahan <finance@vijeth.com>
"""

from typing import List, Dict, Optional, Union
from datetime import datetime

import numpy as np
import pandas as pd
from loguru import logger


class FactorEngine:
    """
    Factor engineering engine for alpha research.

    Provides common technical and fundamental factors used in quantitative trading.
    Integrates with qlib for advanced factor research capabilities.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize factor engine.

        Args:
            data: Price/volume data with columns: open, high, low, close, volume
                 MultiIndex with (date, symbol) or single index with date

        Example:
            >>> data = pd.DataFrame({
            ...     'open': [100, 101],
            ...     'close': [102, 103],
            ...     'volume': [1000, 1100]
            ... })
            >>> engine = FactorEngine(data)
        """
        required_cols = ['close']
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"Data must contain at least: {required_cols}")

        self.data = data.copy()
        self.factors = pd.DataFrame(index=data.index)

        logger.info(f"Initialized FactorEngine with {len(data)} rows")

    def add_momentum_factors(
        self,
        windows: List[int] = [5, 10, 20, 60],
    ) -> 'FactorEngine':
        """
        Add momentum factors.

        Args:
            windows: List of lookback windows in days

        Returns:
            Self for method chaining

        Example:
            >>> engine.add_momentum_factors(windows=[10, 20])
        """
        for window in windows:
            factor_name = f'momentum_{window}d'
            self.factors[factor_name] = (
                self.data['close'].pct_change(window)
            )
            logger.debug(f"Added factor: {factor_name}")

        return self

    def add_reversal_factors(
        self,
        windows: List[int] = [1, 3, 5],
    ) -> 'FactorEngine':
        """
        Add mean reversion factors.

        Args:
            windows: List of lookback windows in days

        Returns:
            Self for method chaining

        Example:
            >>> engine.add_reversal_factors(windows=[1, 5])
        """
        for window in windows:
            factor_name = f'reversal_{window}d'
            # Negative of return (buy losers, sell winners)
            self.factors[factor_name] = (
                -self.data['close'].pct_change(window)
            )
            logger.debug(f"Added factor: {factor_name}")

        return self

    def add_volatility_factors(
        self,
        windows: List[int] = [10, 20, 60],
    ) -> 'FactorEngine':
        """
        Add volatility factors.

        Args:
            windows: List of lookback windows in days

        Returns:
            Self for method chaining

        Example:
            >>> engine.add_volatility_factors(windows=[20])
        """
        returns = self.data['close'].pct_change()

        for window in windows:
            factor_name = f'volatility_{window}d'
            self.factors[factor_name] = (
                returns.rolling(window=window).std()
            )
            logger.debug(f"Added factor: {factor_name}")

        return self

    def add_volume_factors(
        self,
        windows: List[int] = [5, 20],
    ) -> 'FactorEngine':
        """
        Add volume-based factors.

        Args:
            windows: List of lookback windows in days

        Returns:
            Self for method chaining

        Example:
            >>> engine.add_volume_factors(windows=[5, 20])
        """
        if 'volume' not in self.data.columns:
            logger.warning("Volume data not available, skipping volume factors")
            return self

        for window in windows:
            # Volume ratio: current volume / average volume
            factor_name = f'volume_ratio_{window}d'
            avg_volume = self.data['volume'].rolling(window=window).mean()
            self.factors[factor_name] = self.data['volume'] / avg_volume

            # Money flow: price change weighted by volume
            factor_name = f'money_flow_{window}d'
            price_change = self.data['close'].pct_change()
            self.factors[factor_name] = (
                (price_change * self.data['volume'])
                .rolling(window=window)
                .mean()
            )

            logger.debug(f"Added volume factors for {window}d window")

        return self

    def add_technical_indicators(self) -> 'FactorEngine':
        """
        Add common technical indicators.

        Returns:
            Self for method chaining

        Example:
            >>> engine.add_technical_indicators()
        """
        close = self.data['close']

        # RSI (Relative Strength Index)
        self.factors['rsi_14'] = self._calculate_rsi(close, window=14)

        # MACD (Moving Average Convergence Divergence)
        macd, signal = self._calculate_macd(close)
        self.factors['macd'] = macd
        self.factors['macd_signal'] = signal

        # Bollinger Bands
        bb_upper, bb_lower = self._calculate_bollinger_bands(close)
        self.factors['bb_position'] = (close - bb_lower) / (bb_upper - bb_lower)

        logger.debug("Added technical indicators: RSI, MACD, Bollinger Bands")

        return self

    def get_factors(
        self,
        normalize: bool = True,
        fill_na: bool = True,
    ) -> pd.DataFrame:
        """
        Get factor DataFrame.

        Args:
            normalize: If True, z-score normalize factors
            fill_na: If True, forward-fill and then backward-fill NaN values

        Returns:
            DataFrame of factors

        Example:
            >>> factors = engine.get_factors(normalize=True)
        """
        factors = self.factors.copy()

        if fill_na:
            factors = factors.fillna(method='ffill').fillna(method='bfill')

        if normalize:
            # Z-score normalization
            factors = (factors - factors.mean()) / factors.std()
            logger.debug("Normalized factors using z-score")

        logger.info(f"Returning {len(factors.columns)} factors")
        return factors

    def _calculate_rsi(
        self,
        prices: pd.Series,
        window: int = 14,
    ) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _calculate_macd(
        self,
        prices: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> tuple:
        """Calculate MACD and signal line."""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()

        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()

        return macd, signal_line

    def _calculate_bollinger_bands(
        self,
        prices: pd.Series,
        window: int = 20,
        num_std: float = 2.0,
    ) -> tuple:
        """Calculate Bollinger Bands."""
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()

        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)

        return upper_band, lower_band

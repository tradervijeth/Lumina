"""
Trading signal generation from alpha predictions.

Copyright (c) 2025 Vijeth Ltd. All rights reserved.
Author: Vithushan Jeyapahan <finance@vijeth.com>
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd
from loguru import logger


class SignalGenerator:
    """
    Generate trading signals from alpha predictions.

    Converts continuous alpha scores into discrete trading signals with
    position sizing and risk management.
    """

    def __init__(
        self,
        alpha_scores: pd.Series,
        signal_type: str = 'long_short',
    ):
        """
        Initialize signal generator.

        Args:
            alpha_scores: Alpha predictions (higher = better expected return)
            signal_type: Type of signals ('long_only', 'long_short', 'binary')

        Example:
            >>> generator = SignalGenerator(alpha_predictions, signal_type='long_short')
        """
        if signal_type not in ['long_only', 'long_short', 'binary']:
            raise ValueError("signal_type must be 'long_only', 'long_short', or 'binary'")

        self.alpha_scores = alpha_scores
        self.signal_type = signal_type

        logger.info(f"Initialized SignalGenerator with {len(alpha_scores)} alpha scores")

    def generate_percentile_signals(
        self,
        long_threshold: float = 0.8,
        short_threshold: float = 0.2,
    ) -> pd.Series:
        """
        Generate signals based on percentile thresholds.

        Args:
            long_threshold: Percentile threshold for long signals (0-1)
            short_threshold: Percentile threshold for short signals (0-1)

        Returns:
            Series of signals: 1 (long), -1 (short), 0 (neutral)

        Example:
            >>> signals = generator.generate_percentile_signals(long_threshold=0.8)
        """
        signals = pd.Series(0, index=self.alpha_scores.index)

        # Calculate percentile thresholds
        long_cutoff = self.alpha_scores.quantile(long_threshold)
        short_cutoff = self.alpha_scores.quantile(short_threshold)

        # Generate signals
        signals[self.alpha_scores >= long_cutoff] = 1

        if self.signal_type == 'long_short':
            signals[self.alpha_scores <= short_cutoff] = -1

        n_long = (signals == 1).sum()
        n_short = (signals == -1).sum()

        logger.info(f"Generated {n_long} long, {n_short} short, {len(signals) - n_long - n_short} neutral signals")

        return signals

    def generate_zscore_signals(
        self,
        long_threshold: float = 1.0,
        short_threshold: float = -1.0,
    ) -> pd.Series:
        """
        Generate signals based on z-score thresholds.

        Args:
            long_threshold: Z-score threshold for long signals
            short_threshold: Z-score threshold for short signals

        Returns:
            Series of signals: 1 (long), -1 (short), 0 (neutral)

        Example:
            >>> signals = generator.generate_zscore_signals(long_threshold=1.5)
        """
        # Z-score normalization
        z_scores = (self.alpha_scores - self.alpha_scores.mean()) / self.alpha_scores.std()

        signals = pd.Series(0, index=self.alpha_scores.index)
        signals[z_scores >= long_threshold] = 1

        if self.signal_type == 'long_short':
            signals[z_scores <= short_threshold] = -1

        n_long = (signals == 1).sum()
        n_short = (signals == -1).sum()

        logger.info(f"Generated {n_long} long, {n_short} short signals using z-score")

        return signals

    def generate_position_sizes(
        self,
        signals: pd.Series,
        max_position: float = 0.05,
        scale_by_alpha: bool = True,
    ) -> pd.Series:
        """
        Generate position sizes from signals.

        Args:
            signals: Trading signals (1, -1, or 0)
            max_position: Maximum position size per asset
            scale_by_alpha: If True, scale positions by alpha magnitude

        Returns:
            Series of position sizes

        Example:
            >>> positions = generator.generate_position_sizes(signals, max_position=0.1)
        """
        positions = signals.copy().astype(float)

        if scale_by_alpha:
            # Scale by normalized alpha scores
            alpha_norm = np.abs(self.alpha_scores) / np.abs(self.alpha_scores).max()
            positions = positions * alpha_norm

        # Apply max position constraint
        positions = positions.clip(-max_position, max_position)

        # Normalize to sum to 0 (market neutral) or 1 (long only)
        if self.signal_type == 'long_short':
            long_positions = positions[positions > 0]
            short_positions = positions[positions < 0]

            if len(long_positions) > 0:
                positions[positions > 0] = (
                    long_positions / long_positions.sum() * 0.5
                )
            if len(short_positions) > 0:
                positions[positions < 0] = (
                    short_positions / abs(short_positions.sum()) * -0.5
                )
        else:
            # Long only: normalize to sum to 1
            if positions.sum() > 0:
                positions = positions / positions.sum()

        logger.info(
            f"Generated positions: sum={positions.sum():.4f}, "
            f"gross_exposure={positions.abs().sum():.4f}"
        )

        return positions

    def apply_filters(
        self,
        signals: pd.Series,
        min_alpha_magnitude: Optional[float] = None,
        excluded_assets: Optional[list] = None,
    ) -> pd.Series:
        """
        Apply filters to signals.

        Args:
            signals: Trading signals
            min_alpha_magnitude: Minimum absolute alpha score to trade
            excluded_assets: List of assets to exclude from trading

        Returns:
            Filtered signals

        Example:
            >>> filtered = generator.apply_filters(signals, min_alpha_magnitude=0.1)
        """
        filtered_signals = signals.copy()

        # Filter by alpha magnitude
        if min_alpha_magnitude is not None:
            weak_alpha = np.abs(self.alpha_scores) < min_alpha_magnitude
            filtered_signals[weak_alpha] = 0
            n_filtered = weak_alpha.sum()
            logger.debug(f"Filtered {n_filtered} signals with weak alpha")

        # Exclude specific assets
        if excluded_assets is not None:
            for asset in excluded_assets:
                if asset in filtered_signals.index:
                    filtered_signals[asset] = 0
            logger.debug(f"Excluded {len(excluded_assets)} assets")

        return filtered_signals

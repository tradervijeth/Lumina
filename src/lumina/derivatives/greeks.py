"""
Option Greeks calculation utilities.

Copyright (c) 2025 Vijeth Ltd. All rights reserved.
Author: Vithushan Jeyapahan <finance@vijeth.com>
"""

from __future__ import annotations


import numpy as np
from scipy.stats import norm
from loguru import logger


class GreeksCalculator:
    """
    Calculate option Greeks using analytical formulas.

    Provides analytical formulas for option Greeks based on
    Black-Scholes-Merton model. Useful for quick calculations
    without QuantLib dependency.
    """

    @staticmethod
    def d1_d2(
        spot: float,
        strike: float,
        time: float,
        rate: float,
        volatility: float,
        dividend: float = 0.0,
    ) -> tuple:
        """
        Calculate d1 and d2 for Black-Scholes formula.

        d₁ = (ln(S/K) + (r - q + σ²/2)T) / (σ√T)
        d₂ = d₁ - σ√T

        Args:
            spot: Spot price
            strike: Strike price
            time: Time to maturity (years)
            rate: Risk-free rate
            volatility: Volatility
            dividend: Dividend yield

        Returns:
            Tuple of (d1, d2)
        """
        d1 = (
            np.log(spot / strike) +
            (rate - dividend + 0.5 * volatility**2) * time
        ) / (volatility * np.sqrt(time))

        d2 = d1 - volatility * np.sqrt(time)

        return d1, d2

    @staticmethod
    def calculate_call_greeks(
        spot: float,
        strike: float,
        time: float,
        rate: float,
        volatility: float,
        dividend: float = 0.0,
    ) -> Dict[str, float]:
        """
        Calculate all Greeks for a European call option.

        Args:
            spot: Spot price
            strike: Strike price
            time: Time to maturity (years)
            rate: Risk-free rate
            volatility: Annual volatility
            dividend: Dividend yield

        Returns:
            Dictionary with all Greeks

        Example:
            >>> calc = GreeksCalculator()
            >>> greeks = calc.calculate_call_greeks(100, 105, 1.0, 0.05, 0.2)
        """
        d1, d2 = GreeksCalculator.d1_d2(
            spot, strike, time, rate, volatility, dividend
        )

        # Price
        call_price = (
            spot * np.exp(-dividend * time) * norm.cdf(d1) -
            strike * np.exp(-rate * time) * norm.cdf(d2)
        )

        # Delta: ∂V/∂S
        delta = np.exp(-dividend * time) * norm.cdf(d1)

        # Gamma: ∂²V/∂S²
        gamma = (
            np.exp(-dividend * time) * norm.pdf(d1) /
            (spot * volatility * np.sqrt(time))
        )

        # Vega: ∂V/∂σ (per 1% change)
        vega = (
            spot * np.exp(-dividend * time) * norm.pdf(d1) * np.sqrt(time) / 100
        )

        # Theta: ∂V/∂t (per day)
        theta_annual = (
            -spot * np.exp(-dividend * time) * norm.pdf(d1) * volatility / (2 * np.sqrt(time)) -
            rate * strike * np.exp(-rate * time) * norm.cdf(d2) +
            dividend * spot * np.exp(-dividend * time) * norm.cdf(d1)
        )
        theta = theta_annual / 365

        # Rho: ∂V/∂r (per 1% change)
        rho = (
            strike * time * np.exp(-rate * time) * norm.cdf(d2) / 100
        )

        logger.debug(f"Calculated call Greeks: Delta={delta:.4f}, Gamma={gamma:.4f}")

        return {
            'price': call_price,
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'theta': theta,
            'rho': rho,
        }

    @staticmethod
    def calculate_put_greeks(
        spot: float,
        strike: float,
        time: float,
        rate: float,
        volatility: float,
        dividend: float = 0.0,
    ) -> Dict[str, float]:
        """
        Calculate all Greeks for a European put option.

        Args:
            spot: Spot price
            strike: Strike price
            time: Time to maturity (years)
            rate: Risk-free rate
            volatility: Annual volatility
            dividend: Dividend yield

        Returns:
            Dictionary with all Greeks

        Example:
            >>> calc = GreeksCalculator()
            >>> greeks = calc.calculate_put_greeks(100, 105, 1.0, 0.05, 0.2)
        """
        d1, d2 = GreeksCalculator.d1_d2(
            spot, strike, time, rate, volatility, dividend
        )

        # Price
        put_price = (
            strike * np.exp(-rate * time) * norm.cdf(-d2) -
            spot * np.exp(-dividend * time) * norm.cdf(-d1)
        )

        # Delta: ∂V/∂S
        delta = -np.exp(-dividend * time) * norm.cdf(-d1)

        # Gamma: ∂²V/∂S² (same as call)
        gamma = (
            np.exp(-dividend * time) * norm.pdf(d1) /
            (spot * volatility * np.sqrt(time))
        )

        # Vega: ∂V/∂σ (same as call, per 1% change)
        vega = (
            spot * np.exp(-dividend * time) * norm.pdf(d1) * np.sqrt(time) / 100
        )

        # Theta: ∂V/∂t (per day)
        theta_annual = (
            -spot * np.exp(-dividend * time) * norm.pdf(d1) * volatility / (2 * np.sqrt(time)) +
            rate * strike * np.exp(-rate * time) * norm.cdf(-d2) -
            dividend * spot * np.exp(-dividend * time) * norm.cdf(-d1)
        )
        theta = theta_annual / 365

        # Rho: ∂V/∂r (per 1% change)
        rho = (
            -strike * time * np.exp(-rate * time) * norm.cdf(-d2) / 100
        )

        logger.debug(f"Calculated put Greeks: Delta={delta:.4f}, Gamma={gamma:.4f}")

        return {
            'price': put_price,
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'theta': theta,
            'rho': rho,
        }

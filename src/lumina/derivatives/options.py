"""
Options pricing using QuantLib.

Copyright (c) 2025 Vijeth Ltd. All rights reserved.
Author: Vithushan Jeyapahan <finance@vijeth.com>
"""

from datetime import datetime, date
from typing import Optional, Dict

import numpy as np
from loguru import logger

try:
    import QuantLib as ql
    QUANTLIB_AVAILABLE = True
except ImportError:
    QUANTLIB_AVAILABLE = False
    logger.warning("QuantLib not available. Install with: pip install QuantLib")


class OptionPricer:
    """
    Options pricing engine using QuantLib.

    Supports European and American options with various pricing methods:
    - Black-Scholes-Merton (European)
    - Binomial trees (American/European)
    - Finite differences (American/European)
    """

    def __init__(self):
        """Initialize option pricer."""
        if not QUANTLIB_AVAILABLE:
            raise ImportError("QuantLib is required for options pricing")

        self.calendar = ql.UnitedStates(ql.UnitedStates.NYSE)
        self.day_counter = ql.Actual365Fixed()

        logger.info("Initialized OptionPricer with QuantLib")

    def price_european_option(
        self,
        option_type: str,
        spot_price: float,
        strike_price: float,
        risk_free_rate: float,
        volatility: float,
        time_to_maturity: float,
        dividend_yield: float = 0.0,
    ) -> Dict[str, float]:
        """
        Price a European option using Black-Scholes-Merton model.

        The Black-Scholes-Merton formula for a European call option:
            C = S₀e^(-qT)N(d₁) - Ke^(-rT)N(d₂)

        where:
            d₁ = (ln(S₀/K) + (r - q + σ²/2)T) / (σ√T)
            d₂ = d₁ - σ√T

        Args:
            option_type: 'call' or 'put'
            spot_price: Current spot price
            strike_price: Strike price
            risk_free_rate: Annual risk-free rate
            volatility: Annual volatility (standard deviation)
            time_to_maturity: Time to maturity in years
            dividend_yield: Continuous dividend yield

        Returns:
            Dictionary containing:
                - price: Option price
                - delta: Option delta
                - gamma: Option gamma
                - vega: Option vega
                - theta: Option theta
                - rho: Option rho

        Example:
            >>> pricer = OptionPricer()
            >>> result = pricer.price_european_option('call', 100, 105, 0.05, 0.2, 1.0)
        """
        if option_type.lower() not in ['call', 'put']:
            raise ValueError("option_type must be 'call' or 'put'")

        # Set up QuantLib objects
        today = ql.Date.todaysDate()
        ql.Settings.instance().evaluationDate = today

        maturity_date = today + int(time_to_maturity * 365)

        option_type_ql = (
            ql.Option.Call if option_type.lower() == 'call' else ql.Option.Put
        )

        payoff = ql.PlainVanillaPayoff(option_type_ql, strike_price)
        exercise = ql.EuropeanExercise(maturity_date)
        option = ql.VanillaOption(payoff, exercise)

        # Market data
        spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot_price))
        flat_ts = ql.YieldTermStructureHandle(
            ql.FlatForward(today, risk_free_rate, self.day_counter)
        )
        dividend_ts = ql.YieldTermStructureHandle(
            ql.FlatForward(today, dividend_yield, self.day_counter)
        )
        flat_vol_ts = ql.BlackVolTermStructureHandle(
            ql.BlackConstantVol(today, self.calendar, volatility, self.day_counter)
        )

        # Black-Scholes-Merton process
        bsm_process = ql.BlackScholesMertonProcess(
            spot_handle, dividend_ts, flat_ts, flat_vol_ts
        )

        # Pricing engine
        option.setPricingEngine(ql.AnalyticEuropeanEngine(bsm_process))

        # Calculate Greeks
        price = option.NPV()
        delta = option.delta()
        gamma = option.gamma()
        vega = option.vega() / 100  # QuantLib returns vega per 1% change
        theta = option.theta() / 365  # Convert to per-day
        rho = option.rho() / 100  # QuantLib returns rho per 1% change

        logger.debug(
            f"Priced European {option_type} option: "
            f"Price={price:.4f}, Delta={delta:.4f}"
        )

        return {
            'price': price,
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'theta': theta,
            'rho': rho,
        }

    def price_american_option(
        self,
        option_type: str,
        spot_price: float,
        strike_price: float,
        risk_free_rate: float,
        volatility: float,
        time_to_maturity: float,
        dividend_yield: float = 0.0,
        steps: int = 200,
    ) -> Dict[str, float]:
        """
        Price an American option using binomial tree.

        Uses the Cox-Ross-Rubinstein (CRR) binomial tree model for pricing
        American options which allow early exercise.

        Args:
            option_type: 'call' or 'put'
            spot_price: Current spot price
            strike_price: Strike price
            risk_free_rate: Annual risk-free rate
            volatility: Annual volatility
            time_to_maturity: Time to maturity in years
            dividend_yield: Continuous dividend yield
            steps: Number of time steps in binomial tree

        Returns:
            Dictionary containing price and Greeks

        Example:
            >>> pricer = OptionPricer()
            >>> result = pricer.price_american_option('put', 100, 105, 0.05, 0.2, 1.0)
        """
        if option_type.lower() not in ['call', 'put']:
            raise ValueError("option_type must be 'call' or 'put'")

        # Set up QuantLib objects
        today = ql.Date.todaysDate()
        ql.Settings.instance().evaluationDate = today

        maturity_date = today + int(time_to_maturity * 365)

        option_type_ql = (
            ql.Option.Call if option_type.lower() == 'call' else ql.Option.Put
        )

        payoff = ql.PlainVanillaPayoff(option_type_ql, strike_price)
        exercise = ql.AmericanExercise(today, maturity_date)
        option = ql.VanillaOption(payoff, exercise)

        # Market data
        spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot_price))
        flat_ts = ql.YieldTermStructureHandle(
            ql.FlatForward(today, risk_free_rate, self.day_counter)
        )
        dividend_ts = ql.YieldTermStructureHandle(
            ql.FlatForward(today, dividend_yield, self.day_counter)
        )
        flat_vol_ts = ql.BlackVolTermStructureHandle(
            ql.BlackConstantVol(today, self.calendar, volatility, self.day_counter)
        )

        # Black-Scholes-Merton process
        bsm_process = ql.BlackScholesMertonProcess(
            spot_handle, dividend_ts, flat_ts, flat_vol_ts
        )

        # Binomial pricing engine
        option.setPricingEngine(
            ql.BinomialVanillaEngine(bsm_process, "crr", steps)
        )

        price = option.NPV()
        delta = option.delta()
        gamma = option.gamma()

        logger.debug(f"Priced American {option_type} option: Price={price:.4f}")

        return {
            'price': price,
            'delta': delta,
            'gamma': gamma,
        }

    def implied_volatility(
        self,
        option_type: str,
        spot_price: float,
        strike_price: float,
        risk_free_rate: float,
        time_to_maturity: float,
        market_price: float,
        dividend_yield: float = 0.0,
    ) -> float:
        """
        Calculate implied volatility from market price.

        Args:
            option_type: 'call' or 'put'
            spot_price: Current spot price
            strike_price: Strike price
            risk_free_rate: Annual risk-free rate
            time_to_maturity: Time to maturity in years
            market_price: Observed market price
            dividend_yield: Continuous dividend yield

        Returns:
            Implied volatility (annual)

        Example:
            >>> pricer = OptionPricer()
            >>> iv = pricer.implied_volatility('call', 100, 105, 0.05, 1.0, 8.5)
        """
        if option_type.lower() not in ['call', 'put']:
            raise ValueError("option_type must be 'call' or 'put'")

        # Set up QuantLib objects
        today = ql.Date.todaysDate()
        ql.Settings.instance().evaluationDate = today

        maturity_date = today + int(time_to_maturity * 365)

        option_type_ql = (
            ql.Option.Call if option_type.lower() == 'call' else ql.Option.Put
        )

        payoff = ql.PlainVanillaPayoff(option_type_ql, strike_price)
        exercise = ql.EuropeanExercise(maturity_date)
        option = ql.VanillaOption(payoff, exercise)

        # Market data
        spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot_price))
        flat_ts = ql.YieldTermStructureHandle(
            ql.FlatForward(today, risk_free_rate, self.day_counter)
        )
        dividend_ts = ql.YieldTermStructureHandle(
            ql.FlatForward(today, dividend_yield, self.day_counter)
        )

        # Use initial guess for volatility
        vol_guess = 0.20
        flat_vol_ts = ql.BlackVolTermStructureHandle(
            ql.BlackConstantVol(today, self.calendar, vol_guess, self.day_counter)
        )

        bsm_process = ql.BlackScholesMertonProcess(
            spot_handle, dividend_ts, flat_ts, flat_vol_ts
        )

        option.setPricingEngine(ql.AnalyticEuropeanEngine(bsm_process))

        # Calculate implied volatility
        try:
            implied_vol = option.impliedVolatility(
                market_price,
                bsm_process,
                accuracy=1e-6,
                maxEvaluations=1000,
            )
        except RuntimeError as e:
            logger.error(f"Failed to calculate implied volatility: {e}")
            raise ValueError("Could not converge to implied volatility")

        logger.debug(f"Calculated implied volatility: {implied_vol:.4f}")

        return implied_vol

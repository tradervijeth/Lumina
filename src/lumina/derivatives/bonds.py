"""
Bond pricing using QuantLib.

Copyright (c) 2025 Vijeth Ltd. All rights reserved.
Author: Vithushan Jeyapahan <finance@vijeth.com>
"""

from __future__ import annotations

from datetime import datetime, date


from loguru import logger

try:
    import QuantLib as ql
    QUANTLIB_AVAILABLE = True
except ImportError:
    QUANTLIB_AVAILABLE = False
    logger.warning("QuantLib not available")


class BondPricer:
    """
    Bond pricing engine using QuantLib.

    Supports various bond types:
    - Zero-coupon bonds
    - Fixed-rate bonds
    - Floating-rate bonds
    """

    def __init__(self):
        """Initialize bond pricer."""
        if not QUANTLIB_AVAILABLE:
            raise ImportError("QuantLib is required for bond pricing")

        self.calendar = ql.UnitedStates(ql.UnitedStates.NYSE)
        self.day_counter = ql.Actual365Fixed()
        self.business_convention = ql.Following

        logger.info("Initialized BondPricer with QuantLib")

    def price_zero_coupon_bond(
        self,
        face_value: float,
        maturity_years: float,
        yield_rate: float,
    ) -> Dict[str, float]:
        """
        Price a zero-coupon bond.

        The price of a zero-coupon bond is:
            P = F / (1 + y)^T

        where:
            F = face value
            y = yield to maturity
            T = time to maturity

        Args:
            face_value: Face value of bond
            maturity_years: Years to maturity
            yield_rate: Annual yield to maturity

        Returns:
            Dictionary containing:
                - price: Bond price
                - ytm: Yield to maturity
                - duration: Macaulay duration

        Example:
            >>> pricer = BondPricer()
            >>> result = pricer.price_zero_coupon_bond(1000, 5, 0.05)
        """
        today = ql.Date.todaysDate()
        ql.Settings.instance().evaluationDate = today

        maturity_date = today + int(maturity_years * 365)

        # Create zero-coupon bond
        bond = ql.ZeroCouponBond(
            0,  # settlement days
            self.calendar,
            face_value,
            maturity_date,
            self.business_convention,
        )

        # Yield term structure
        spot_curve = ql.YieldTermStructureHandle(
            ql.FlatForward(today, yield_rate, self.day_counter)
        )

        bond_engine = ql.DiscountingBondEngine(spot_curve)
        bond.setPricingEngine(bond_engine)

        price = bond.NPV()
        ytm = bond.bondYield(self.day_counter, ql.Compounded, ql.Annual)

        # Duration is equal to time to maturity for zero-coupon bonds
        duration = maturity_years

        logger.debug(f"Priced zero-coupon bond: Price={price:.2f}, YTM={ytm:.4%}")

        return {
            'price': price,
            'ytm': ytm,
            'duration': duration,
        }

    def price_fixed_rate_bond(
        self,
        face_value: float,
        coupon_rate: float,
        maturity_years: float,
        yield_rate: float,
        coupon_frequency: int = 2,
    ) -> Dict[str, float]:
        """
        Price a fixed-rate coupon bond.

        Args:
            face_value: Face value of bond
            coupon_rate: Annual coupon rate
            maturity_years: Years to maturity
            yield_rate: Annual yield to maturity
            coupon_frequency: Coupons per year (1=annual, 2=semiannual, 4=quarterly)

        Returns:
            Dictionary containing bond metrics

        Example:
            >>> pricer = BondPricer()
            >>> result = pricer.price_fixed_rate_bond(1000, 0.05, 10, 0.04, frequency=2)
        """
        today = ql.Date.todaysDate()
        ql.Settings.instance().evaluationDate = today

        issue_date = today
        maturity_date = today + int(maturity_years * 365)

        # Schedule
        if coupon_frequency == 1:
            period = ql.Period(ql.Annual)
        elif coupon_frequency == 2:
            period = ql.Period(ql.Semiannual)
        elif coupon_frequency == 4:
            period = ql.Period(ql.Quarterly)
        else:
            raise ValueError("coupon_frequency must be 1, 2, or 4")

        schedule = ql.Schedule(
            issue_date,
            maturity_date,
            period,
            self.calendar,
            self.business_convention,
            self.business_convention,
            ql.DateGeneration.Backward,
            False,
        )

        # Create fixed-rate bond
        bond = ql.FixedRateBond(
            0,  # settlement days
            face_value,
            schedule,
            [coupon_rate],
            self.day_counter,
        )

        # Yield term structure
        spot_curve = ql.YieldTermStructureHandle(
            ql.FlatForward(today, yield_rate, self.day_counter)
        )

        bond_engine = ql.DiscountingBondEngine(spot_curve)
        bond.setPricingEngine(bond_engine)

        price = bond.NPV()
        clean_price = bond.cleanPrice()
        accrued_interest = bond.accruedAmount()

        ytm = bond.bondYield(self.day_counter, ql.Compounded, ql.Annual)

        # Macaulay duration and modified duration
        duration = ql.BondFunctions.duration(
            bond,
            yield_rate,
            self.day_counter,
            ql.Compounded,
            ql.Annual,
            ql.Duration.Macaulay,
        )

        modified_duration = ql.BondFunctions.duration(
            bond,
            yield_rate,
            self.day_counter,
            ql.Compounded,
            ql.Annual,
            ql.Duration.Modified,
        )

        convexity = ql.BondFunctions.convexity(
            bond,
            yield_rate,
            self.day_counter,
            ql.Compounded,
            ql.Annual,
        )

        logger.debug(
            f"Priced fixed-rate bond: "
            f"Price={price:.2f}, YTM={ytm:.4%}, Duration={duration:.2f}"
        )

        return {
            'price': price,
            'clean_price': clean_price,
            'accrued_interest': accrued_interest,
            'ytm': ytm,
            'duration': duration,
            'modified_duration': modified_duration,
            'convexity': convexity,
        }

    def calculate_duration(
        self,
        face_value: float,
        coupon_rate: float,
        maturity_years: float,
        yield_rate: float,
        coupon_frequency: int = 2,
    ) -> Dict[str, float]:
        """
        Calculate Macaulay and modified duration.

        Args:
            face_value: Face value of bond
            coupon_rate: Annual coupon rate
            maturity_years: Years to maturity
            yield_rate: Annual yield to maturity
            coupon_frequency: Coupons per year

        Returns:
            Dictionary with duration metrics

        Example:
            >>> pricer = BondPricer()
            >>> duration = pricer.calculate_duration(1000, 0.05, 10, 0.04)
        """
        result = self.price_fixed_rate_bond(
            face_value,
            coupon_rate,
            maturity_years,
            yield_rate,
            coupon_frequency,
        )

        return {
            'macaulay_duration': result['duration'],
            'modified_duration': result['modified_duration'],
            'convexity': result['convexity'],
        }

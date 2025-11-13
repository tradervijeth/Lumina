"""
Machine learning models for alpha generation.

Copyright (c) 2025 Vijeth Ltd. All rights reserved.
Author: Vithushan Jeyapahan <finance@vijeth.com>
"""

from __future__ import annotations

from typing import Optional, Any
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score


class AlphaModel:
    """
    Machine learning model for alpha prediction.

    Supports various ML algorithms for predicting asset returns based on factors.
    Designed for integration with qlib's ML infrastructure.
    """

    SUPPORTED_MODELS = {
        'random_forest': RandomForestRegressor,
        'gradient_boosting': GradientBoostingRegressor,
        'ridge': Ridge,
        'lasso': Lasso,
    }

    def __init__(
        self,
        model_type: str = 'random_forest',
        model_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize alpha model.

        Args:
            model_type: Type of model ('random_forest', 'gradient_boosting', 'ridge', 'lasso')
            model_params: Parameters to pass to the model

        Example:
            >>> model = AlphaModel(model_type='random_forest', model_params={'n_estimators': 100})
        """
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Model type must be one of {list(self.SUPPORTED_MODELS.keys())}"
            )

        self.model_type = model_type
        self.model_params = model_params or {}

        # Set default parameters
        if model_type == 'random_forest' and 'n_estimators' not in self.model_params:
            self.model_params['n_estimators'] = 100
            self.model_params['random_state'] = 42

        self.model = self.SUPPORTED_MODELS[model_type](**self.model_params)
        self.feature_importance_ = None
        self.is_fitted = False

        logger.info(f"Initialized {model_type} alpha model")

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_split: float = 0.2,
        time_series_split: bool = True,
    ) -> Dict[str, float]:
        """
        Train the alpha model.

        Args:
            X: Feature DataFrame (factors)
            y: Target Series (future returns)
            validation_split: Fraction of data to use for validation
            time_series_split: If True, uses time-series aware split

        Returns:
            Dictionary of training metrics

        Example:
            >>> metrics = model.train(factors, future_returns)
        """
        # Remove NaN values
        valid_idx = ~(X.isna().any(axis=1) | y.isna())
        X_clean = X[valid_idx]
        y_clean = y[valid_idx]

        logger.info(f"Training on {len(X_clean)} samples with {X_clean.shape[1]} features")

        if time_series_split:
            # Use chronological split for time series
            split_idx = int(len(X_clean) * (1 - validation_split))
            X_train, X_val = X_clean.iloc[:split_idx], X_clean.iloc[split_idx:]
            y_train, y_val = y_clean.iloc[:split_idx], y_clean.iloc[split_idx:]
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X_clean, y_clean,
                test_size=validation_split,
                random_state=42,
            )

        # Train model
        self.model.fit(X_train, y_train)
        self.is_fitted = True

        # Evaluate
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)

        metrics = {
            'train_mse': mean_squared_error(y_train, train_pred),
            'val_mse': mean_squared_error(y_val, val_pred),
            'train_r2': r2_score(y_train, train_pred),
            'val_r2': r2_score(y_val, val_pred),
            'train_ic': np.corrcoef(y_train, train_pred)[0, 1],
            'val_ic': np.corrcoef(y_val, val_pred)[0, 1],
        }

        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance_ = pd.Series(
                self.model.feature_importances_,
                index=X.columns,
                name='importance',
            ).sort_values(ascending=False)

        logger.info(f"Training complete. Val R²: {metrics['val_r2']:.4f}, Val IC: {metrics['val_ic']:.4f}")

        return metrics

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict alpha scores.

        Args:
            X: Feature DataFrame

        Returns:
            Predicted alpha scores

        Raises:
            RuntimeError: If model hasn't been trained

        Example:
            >>> predictions = model.predict(new_factors)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before prediction")

        predictions = self.model.predict(X)
        pred_series = pd.Series(predictions, index=X.index, name='alpha_prediction')

        logger.debug(f"Generated predictions for {len(pred_series)} samples")

        return pred_series

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
    ) -> pd.DataFrame:
        """
        Perform time-series cross-validation.

        Args:
            X: Feature DataFrame
            y: Target Series
            n_splits: Number of cross-validation splits

        Returns:
            DataFrame with metrics for each fold

        Example:
            >>> cv_results = model.cross_validate(factors, returns, n_splits=5)
        """
        # Remove NaN values
        valid_idx = ~(X.isna().any(axis=1) | y.isna())
        X_clean = X[valid_idx]
        y_clean = y[valid_idx]

        tscv = TimeSeriesSplit(n_splits=n_splits)
        results = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_clean)):
            X_train = X_clean.iloc[train_idx]
            X_val = X_clean.iloc[val_idx]
            y_train = y_clean.iloc[train_idx]
            y_val = y_clean.iloc[val_idx]

            # Train on fold
            fold_model = self.SUPPORTED_MODELS[self.model_type](**self.model_params)
            fold_model.fit(X_train, y_train)

            # Evaluate
            val_pred = fold_model.predict(X_val)

            results.append({
                'fold': fold + 1,
                'val_mse': mean_squared_error(y_val, val_pred),
                'val_r2': r2_score(y_val, val_pred),
                'val_ic': np.corrcoef(y_val, val_pred)[0, 1],
                'n_train': len(X_train),
                'n_val': len(X_val),
            })

            logger.info(f"Fold {fold + 1}/{n_splits}: Val IC = {results[-1]['val_ic']:.4f}")

        cv_df = pd.DataFrame(results)
        logger.info(f"Cross-validation complete. Mean Val IC: {cv_df['val_ic'].mean():.4f}")

        return cv_df

    def get_feature_importance(self, top_n: int = 10) -> pd.Series:
        """
        Get feature importance scores.

        Args:
            top_n: Number of top features to return

        Returns:
            Series of feature importances

        Raises:
            ValueError: If model doesn't support feature importance

        Example:
            >>> importance = model.get_feature_importance(top_n=5)
        """
        if self.feature_importance_ is None:
            raise ValueError("Feature importance not available for this model type")

        return self.feature_importance_.head(top_n)

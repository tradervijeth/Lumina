"""
Database integration for storing backtest results and market data.

Copyright (c) 2025 Vijeth Ltd. All rights reserved.
Author: Vithushan Jeyapahan <finance@vijeth.com>
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
from datetime import datetime

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from loguru import logger


class DatabaseManager:
    """
    Database manager for Lumina.

    Supports SQLite and PostgreSQL for storing:
    - Historical market data
    - Backtest results
    - Portfolio snapshots
    - Trade history

    Example:
        >>> db = DatabaseManager('sqlite:///lumina.db')
        >>> db.save_backtest_results('my_strategy', results_df)
        >>> results = db.load_backtest_results('my_strategy')
    """

    def __init__(
        self,
        connection_string: str,
        create_tables: bool = True,
    ):
        """
        Initialize database connection.

        Args:
            connection_string: SQLAlchemy connection string
                - SQLite: 'sqlite:///path/to/db.db'
                - PostgreSQL: 'postgresql://user:pass@localhost:5432/lumina'
            create_tables: Whether to create tables if they don't exist

        Example:
            >>> db = DatabaseManager('sqlite:///lumina.db')
        """
        self.engine = create_engine(connection_string, echo=False)
        self.connection_string = connection_string

        logger.info(f"Connected to database: {connection_string}")

        if create_tables:
            self._create_tables()

    def _create_tables(self) -> None:
        """Create database tables if they don't exist."""
        with self.engine.connect() as conn:
            # Market data table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS market_data (
                    symbol TEXT NOT NULL,
                    date DATE NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    adj_close REAL,
                    PRIMARY KEY (symbol, date)
                )
            """))

            # Backtest results table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS backtest_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_name TEXT NOT NULL,
                    run_timestamp TIMESTAMP NOT NULL,
                    date DATE NOT NULL,
                    portfolio_value REAL,
                    cash REAL,
                    positions_value REAL,
                    returns REAL,
                    cumulative_returns REAL
                )
            """))

            # Trade history table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_name TEXT NOT NULL,
                    run_timestamp TIMESTAMP NOT NULL,
                    date DATE NOT NULL,
                    asset TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL NOT NULL,
                    side TEXT NOT NULL,
                    value REAL,
                    commission REAL,
                    spread REAL,
                    slippage REAL,
                    total_cost REAL
                )
            """))

            # Portfolio snapshots table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_name TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    asset TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    weight REAL,
                    value REAL
                )
            """))

            conn.commit()

        logger.info("Database tables initialized")

    def save_market_data(
        self,
        symbol: str,
        data: pd.DataFrame,
    ) -> None:
        """
        Save market data to database.

        Args:
            symbol: Asset symbol
            data: OHLCV data with DatetimeIndex

        Example:
            >>> db.save_market_data('AAPL', price_data)
        """
        df = data.copy()
        df['symbol'] = symbol
        df = df.reset_index()
        df = df.rename(columns={'index': 'date'})

        df.to_sql(
            'market_data',
            self.engine,
            if_exists='append',
            index=False,
            method='multi',
        )

        logger.info(f"Saved {len(df)} rows for {symbol}")

    def load_market_data(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load market data from database.

        Args:
            symbol: Asset symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with market data

        Example:
            >>> data = db.load_market_data('AAPL', '2020-01-01', '2023-12-31')
        """
        query = f"SELECT * FROM market_data WHERE symbol = '{symbol}'"

        if start_date:
            query += f" AND date >= '{start_date}'"
        if end_date:
            query += f" AND date <= '{end_date}'"

        query += " ORDER BY date"

        df = pd.read_sql(query, self.engine, parse_dates=['date'])
        df = df.set_index('date')
        df = df.drop(columns=['symbol'])

        logger.info(f"Loaded {len(df)} rows for {symbol}")

        return df

    def save_backtest_results(
        self,
        strategy_name: str,
        results: pd.DataFrame,
        run_timestamp: Optional[datetime] = None,
    ) -> None:
        """
        Save backtest results to database.

        Args:
            strategy_name: Name of strategy
            results: Backtest results DataFrame
            run_timestamp: Timestamp of backtest run (default: now)

        Example:
            >>> db.save_backtest_results('momentum_strategy', results)
        """
        if run_timestamp is None:
            run_timestamp = datetime.now()

        df = results.copy()
        df['strategy_name'] = strategy_name
        df['run_timestamp'] = run_timestamp
        df = df.reset_index()
        df = df.rename(columns={'index': 'date'})

        df.to_sql(
            'backtest_results',
            self.engine,
            if_exists='append',
            index=False,
            method='multi',
        )

        logger.info(f"Saved backtest results for '{strategy_name}': {len(df)} rows")

    def load_backtest_results(
        self,
        strategy_name: str,
        run_timestamp: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Load backtest results from database.

        Args:
            strategy_name: Name of strategy
            run_timestamp: Specific run to load (default: latest)

        Returns:
            DataFrame with backtest results

        Example:
            >>> results = db.load_backtest_results('momentum_strategy')
        """
        if run_timestamp is None:
            # Get latest run
            query = f"""
                SELECT * FROM backtest_results
                WHERE strategy_name = '{strategy_name}'
                AND run_timestamp = (
                    SELECT MAX(run_timestamp)
                    FROM backtest_results
                    WHERE strategy_name = '{strategy_name}'
                )
                ORDER BY date
            """
        else:
            query = f"""
                SELECT * FROM backtest_results
                WHERE strategy_name = '{strategy_name}'
                AND run_timestamp = '{run_timestamp}'
                ORDER BY date
            """

        df = pd.read_sql(query, self.engine, parse_dates=['date', 'run_timestamp'])
        df = df.set_index('date')
        df = df.drop(columns=['id', 'strategy_name', 'run_timestamp'])

        logger.info(f"Loaded backtest results for '{strategy_name}': {len(df)} rows")

        return df

    def save_trades(
        self,
        strategy_name: str,
        trades: pd.DataFrame,
        run_timestamp: Optional[datetime] = None,
    ) -> None:
        """
        Save trade history to database.

        Args:
            strategy_name: Name of strategy
            trades: Trade history DataFrame
            run_timestamp: Timestamp of backtest run (default: now)

        Example:
            >>> db.save_trades('momentum_strategy', trades_df)
        """
        if run_timestamp is None:
            run_timestamp = datetime.now()

        df = trades.copy()
        df['strategy_name'] = strategy_name
        df['run_timestamp'] = run_timestamp

        if 'date' not in df.columns and df.index.name in ['date', 'timestamp']:
            df = df.reset_index()

        df.to_sql(
            'trades',
            self.engine,
            if_exists='append',
            index=False,
            method='multi',
        )

        logger.info(f"Saved {len(df)} trades for '{strategy_name}'")

    def list_strategies(self) -> list[str]:
        """
        List all strategies with saved results.

        Returns:
            List of strategy names

        Example:
            >>> strategies = db.list_strategies()
        """
        query = "SELECT DISTINCT strategy_name FROM backtest_results ORDER BY strategy_name"
        df = pd.read_sql(query, self.engine)
        return df['strategy_name'].tolist()

    def get_strategy_runs(self, strategy_name: str) -> pd.DataFrame:
        """
        Get all runs for a strategy.

        Args:
            strategy_name: Name of strategy

        Returns:
            DataFrame with run information

        Example:
            >>> runs = db.get_strategy_runs('momentum_strategy')
        """
        query = f"""
            SELECT
                run_timestamp,
                MIN(date) as start_date,
                MAX(date) as end_date,
                COUNT(*) as num_periods
            FROM backtest_results
            WHERE strategy_name = '{strategy_name}'
            GROUP BY run_timestamp
            ORDER BY run_timestamp DESC
        """

        df = pd.read_sql(query, self.engine, parse_dates=['run_timestamp', 'start_date', 'end_date'])
        return df

    def close(self) -> None:
        """Close database connection."""
        self.engine.dispose()
        logger.info("Database connection closed")

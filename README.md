# Lumina

**Institutional-Grade Quantitative Trading Platform**

Developed by Vijeth Ltd
Author: Vithushan Jeyapahan

**Version 0.3.0** - Production-Ready Release

---

## Overview

Lumina is a comprehensive quantitative trading platform that integrates state-of-the-art optimization techniques, machine learning models, derivatives pricing, and portfolio management tools. Built for institutional traders and quantitative researchers, Lumina provides a robust framework for alpha generation, risk management, and systematic trading strategy development.

**Now with production-ready features**: comprehensive transaction cost modeling, advanced risk management with stress testing, database integration, and extensive validation.

## Key Features

### 🎯 Multi-Library Integration
- **cvxpy**: Convex optimization for portfolio construction
- **qlib**: Machine learning-based alpha factor research
- **QuantLib**: Advanced derivatives pricing and risk analytics
- **PyPortfolioOpt**: Modern portfolio theory implementations

### 📊 Core Capabilities
- **Data Integration**: Multi-source data loading with automatic fallback (yfinance, pandas_datareader, Alpha Vantage), caching, and validation
- **Alpha Generation**: ML-based factor models and signal generation
- **Portfolio Optimization**: Mean-variance, risk parity, Black-Litterman with investor views
- **Transaction Cost Modeling**: Comprehensive cost modeling (commission, spread, slippage) with multiple broker presets
- **Order Execution Simulation**: Market, limit, stop, and stop-limit orders with realistic fill simulation
- **Derivatives Pricing**: Options, bonds, and structured products using QuantLib
- **Backtesting**: Event-driven engine with transaction costs, walk-forward optimization, and performance analysis
- **Risk Management**: Position sizing (Kelly, volatility targeting), VaR/ES, risk limits, correlation stress testing
- **Stress Testing**: Historical scenario replay, hypothetical scenarios, correlation/volatility stress
- **Performance Attribution**: Factor-based and Brinson-Hood-Beebower attribution
- **Database Integration**: SQLite/PostgreSQL support for storing market data, backtest results, and trade history
- **Comprehensive Validation**: Data validation, portfolio validation, covariance matrix checks

## Architecture

```
lumina/
├── src/lumina/
│   ├── data/           # Data loading and cleaning
│   ├── alpha/          # ML models and factor research
│   ├── optimization/   # Portfolio optimization (cvxpy)
│   ├── derivatives/    # Derivatives pricing (QuantLib)
│   ├── portfolio/      # Portfolio management (PyPortfolioOpt)
│   ├── backtest/       # Backtesting with walk-forward
│   ├── risk/           # Risk management and position sizing
│   ├── attribution/    # Performance attribution
│   └── utils/          # Shared utilities and helpers
├── tests/              # Comprehensive test suite
├── config/             # Configuration examples
├── docs/               # Documentation
└── examples/           # Usage examples
```

## Installation

### Prerequisites
- Python 3.10 or higher
- pip or conda package manager

### Quick Start

```bash
# Clone the repository
git clone https://github.com/tradervijeth/Lumina.git
cd Lumina

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Development Installation

```bash
# Install with development dependencies
pip install -e ".[dev]"
```

## Quick Example

```python
from lumina.optimization import MeanVarianceOptimizer
from lumina.portfolio import PortfolioAnalyzer
import pandas as pd

# Load historical returns
returns = pd.read_csv("returns.csv", index_col=0, parse_dates=True)

# Optimize portfolio
optimizer = MeanVarianceOptimizer(returns)
weights = optimizer.optimize(target_return=0.12, risk_free_rate=0.02)

# Analyze portfolio
analyzer = PortfolioAnalyzer(weights, returns)
metrics = analyzer.performance_metrics()
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"Annual Return: {metrics['annual_return']:.2%}")
print(f"Annual Volatility: {metrics['annual_volatility']:.2%}")
```

## What's New in v0.3.0

### 🚀 Production-Ready Features

**Transaction Cost Modeling**
- Comprehensive cost model with commission, bid-ask spread, and market impact
- Multiple slippage models (fixed, volume-based, square-root, linear impact)
- Predefined broker models (Interactive Brokers, Schwab, Robinhood, institutional)
- Cost analysis and attribution by symbol and over time

**Order Execution Simulation**
- Market, limit, stop, and stop-limit order types
- Realistic fill simulation with partial fills and price improvement
- Volume-based fill probability

**Enhanced Risk Management**
- Correlation stress testing (analyze portfolio sensitivity to correlation shocks)
- Scenario analysis (market crash, rate shock, inflation, credit crisis, etc.)
- Historical scenario replay (e.g., 2008 financial crisis)
- Scenario generator to find worst historical periods

**Data Pipeline Improvements**
- Automatic source fallback (yfinance → pandas_datareader → Alpha Vantage)
- Exponential backoff retry logic for network failures
- Two-layer caching (memory + disk) with TTL
- Rate limiting for API calls
- Comprehensive data validation (OHLC consistency, outliers, missing values)

**Comprehensive Validation**
- Returns validation (minimum periods, missing data, outliers)
- Price data validation (negative prices, zero prices)
- Portfolio weights validation (bounds checking, sum to one)
- Covariance matrix validation (symmetry, positive semi-definite, condition number)

**Database Integration**
- SQLite and PostgreSQL support
- Store market data, backtest results, trade history, portfolio snapshots
- Query historical runs and compare strategies
- Automated table creation and schema management

**Performance Optimizations**
- Vectorized operations throughout codebase
- Efficient caching mechanisms
- Optimized VaR calculations

### 📦 New Modules

- `lumina.backtest.costs` - Transaction cost modeling
- `lumina.backtest.execution` - Order execution simulation
- `lumina.risk.stress_testing` - Stress testing and scenario analysis
- `lumina.utils.validators` - Comprehensive data validation
- `lumina.utils.database` - Database integration

### 🔧 Enhanced Modules

- `lumina.data.loader` - Improved with fallback, caching, and validation
- `lumina.backtest.engine` - Now uses comprehensive cost model
- `lumina.risk.*` - Added stress testing capabilities

## Core Modules

### Data (`lumina.data`)
Multi-source data integration and preprocessing:
- DataLoader: Alpha Vantage, CSV, synthetic data
- DataCleaner: Missing data, outliers, returns calculation
- Universe: Tradable universe management (S&P 500, custom)

### Optimization (`lumina.optimization`)
Advanced portfolio optimization techniques using convex optimization:
- Mean-Variance Optimization
- Risk Parity
- Black-Litterman Model (market equilibrium + investor views)
- Minimum Variance
- Maximum Sharpe Ratio
- Custom constraint handling

### Alpha Generation (`lumina.alpha`)
Machine learning-based alpha factor research:
- Factor extraction and engineering
- ML model training and prediction
- Signal generation and combination
- Performance attribution

### Derivatives (`lumina.derivatives`)
Comprehensive derivatives pricing and analytics:
- Options pricing (Black-Scholes, binomial trees)
- Bond pricing and yield curve construction
- Interest rate derivatives
- Greeks calculation

### Portfolio Management (`lumina.portfolio`)
Modern portfolio theory and risk management:
- Efficient frontier construction
- Portfolio rebalancing
- Risk decomposition
- Performance attribution

### Backtesting (`lumina.backtest`)
Event-driven backtesting framework:
- Vectorized and event-driven modes
- Walk-forward optimization (prevents overfitting)
- Transaction cost modeling
- Slippage simulation
- Performance analytics
- Built-in strategies (Momentum, Mean Reversion)

### Risk Management (`lumina.risk`)
Comprehensive risk management tools:
- Position sizing (Kelly criterion, fixed fraction, volatility targeting)
- Risk limits (position size, leverage, drawdown)
- VaR & Expected Shortfall (historical, parametric, Monte Carlo)

### Attribution (`lumina.attribution`)
Performance attribution analysis:
- Factor-based attribution (Fama-French factors)
- Brinson-Hood-Beebower attribution
- Return decomposition

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=lumina --cov-report=html

# Run specific test suite
pytest tests/unit/optimization/
```

## Documentation

Comprehensive documentation is available in the `docs/` directory:
- API Reference
- Mathematical Formulations
- Usage Examples
- Performance Benchmarks

## Performance

Lumina is designed for institutional-grade performance:
- Vectorized operations using NumPy/Pandas
- Efficient convex optimization with cvxpy
- GPU acceleration support for ML models
- Parallel backtesting capabilities

## License

Copyright (c) 2025 Vijeth Ltd. All rights reserved.

This software is proprietary and confidential. Unauthorized copying, distribution, or use is strictly prohibited.

## Contact

**Vithushan Jeyapahan**
Email: finance@vijeth.com
Company: Vijeth Ltd

## Acknowledgments

Built on industry-leading open-source libraries:
- cvxpy for convex optimization
- qlib for quantitative investment
- QuantLib for quantitative finance
- PyPortfolioOpt for portfolio optimization

---

**Disclaimer**: This software is for informational and educational purposes only. Past performance does not guarantee future results. Trading involves risk of loss.

## Examples

### Basic Portfolio Optimization
See `examples/01_portfolio_optimization.py` for mean-variance and risk parity examples.

### Backtesting Strategies  
See `examples/02_backtesting.py` for momentum strategy backtest with performance analysis.

### Data Loading & Risk Management
See `examples/03_real_data_example.py` for:
- Loading data from multiple sources
- Data cleaning and preprocessing
- Portfolio optimization with risk limits
- Position sizing and VaR calculation

### Advanced Portfolio Construction
See `examples/04_advanced_portfolio.py` for:
- Black-Litterman optimization with investor views
- Performance comparison vs traditional mean-variance
- Factor attribution analysis

## Configuration

See `config/example_config.yaml` for a complete configuration template covering:
- Data sources and caching
- Risk management parameters
- Portfolio optimization settings
- Backtesting parameters
- Strategy configurations

## What's New in v0.2.0

### Major Additions:
1. **Data Module**: Multi-source data loading (Alpha Vantage API, CSV, synthetic)
2. **Risk Management**: Position sizing, risk limits, VaR/ES calculations
3. **Black-Litterman**: Advanced portfolio optimization with investor views
4. **Walk-Forward Optimization**: Robust strategy validation
5. **Performance Attribution**: Factor-based and Brinson attribution
6. **Enhanced Examples**: Real-world usage patterns
7. **Configuration Management**: YAML-based configuration

### Improvements:
- Python 3.11+ compatibility with `from __future__ import annotations`
- Comprehensive logging with loguru
- Better error handling and validation
- Extended test coverage


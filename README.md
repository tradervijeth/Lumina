# Lumina

**Institutional-Grade Quantitative Trading Platform**

Developed by Vijeth Ltd
Author: Vithushan Jeyapahan

---

## Overview

Lumina is a comprehensive quantitative trading platform that integrates state-of-the-art optimization techniques, machine learning models, derivatives pricing, and portfolio management tools. Built for institutional traders and quantitative researchers, Lumina provides a robust framework for alpha generation, risk management, and systematic trading strategy development.

## Key Features

### 🎯 Multi-Library Integration
- **cvxpy**: Convex optimization for portfolio construction
- **qlib**: Machine learning-based alpha factor research
- **QuantLib**: Advanced derivatives pricing and risk analytics
- **PyPortfolioOpt**: Modern portfolio theory implementations

### 📊 Core Capabilities
- **Alpha Generation**: ML-based factor models and signal generation
- **Portfolio Optimization**: Mean-variance, risk parity, Black-Litterman
- **Derivatives Pricing**: Options, bonds, and structured products
- **Backtesting**: High-performance event-driven backtesting engine
- **Risk Management**: VaR, CVaR, and comprehensive risk metrics

## Architecture

```
lumina/
├── src/lumina/
│   ├── alpha/          # ML models and factor research (qlib)
│   ├── optimization/   # Portfolio optimization (cvxpy)
│   ├── derivatives/    # Derivatives pricing (QuantLib)
│   ├── portfolio/      # Portfolio management (PyPortfolioOpt)
│   ├── backtest/       # Backtesting framework
│   └── utils/          # Shared utilities and helpers
├── tests/
│   ├── unit/           # Unit tests
│   └── integration/    # Integration tests
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

## Core Modules

### Optimization (`lumina.optimization`)
Advanced portfolio optimization techniques using convex optimization:
- Mean-Variance Optimization
- Risk Parity
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
- Transaction cost modeling
- Slippage simulation
- Performance analytics

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

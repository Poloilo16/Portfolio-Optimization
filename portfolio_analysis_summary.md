# Minimum Variance Portfolio Construction with Real-World Data

## Overview

This project successfully demonstrates the construction of a minimum variance portfolio using the `optPort` method from the existing `Minimum_variance.py` file with real-world stock market data.

## Key Results

### Portfolio Composition
The optimized minimum variance portfolio consists of:

| Asset | Weight | Percentage |
|-------|---------|------------|
| **Johnson & Johnson (JNJ)** | 0.6604 | **66.04%** |
| **JPMorgan Chase (JPM)** | 0.1385 | **13.85%** |
| **Microsoft (MSFT)** | 0.1114 | **11.14%** |
| **Apple (AAPL)** | 0.0631 | **6.31%** |
| **Alphabet (GOOGL)** | 0.0266 | **2.66%** |

### Portfolio Performance Metrics

- **Expected Annual Return**: 4.23%
- **Annual Volatility**: 14.33%
- **Portfolio Variance**: 0.020546
- **Sharpe Ratio**: 0.295

### Backtest Results (2022-2024)
- **Total Return**: 10.03%
- **Annualized Volatility**: 14.33%
- **Maximum Drawdown**: -14.41%

## Key Insights

### 1. Risk Minimization Strategy
The minimum variance portfolio heavily weights Johnson & Johnson (66.04%), which is a defensive healthcare stock with lower volatility. This demonstrates the algorithm's preference for stable, low-volatility assets when the objective is risk minimization.

### 2. Diversification Benefits
While the portfolio is concentrated in JNJ, it still maintains meaningful positions in other sectors:
- **Healthcare**: JNJ (66.04%)
- **Financial Services**: JPM (13.85%)
- **Technology**: MSFT (11.14%), AAPL (6.31%), GOOGL (2.66%)

### 3. Risk-Adjusted Performance
Comparing individual assets vs. portfolio:

| Asset | Return | Volatility | Sharpe Ratio |
|-------|---------|------------|--------------|
| AAPL | 15.15% | 27.11% | 0.559 |
| GOOGL | 14.77% | 32.76% | 0.451 |
| **JNJ** | **-1.77%** | **16.38%** | **-0.108** |
| JPM | 19.09% | 25.01% | 0.763 |
| MSFT | 12.66% | 27.59% | 0.459 |
| **Portfolio** | **4.23%** | **14.33%** | **0.295** |

The portfolio achieves its objective of **risk minimization** with the lowest volatility (14.33%) compared to any individual asset, though this comes at the cost of lower expected returns.

## Technical Implementation

### Method Used
- **Core Algorithm**: `optPort(cov, mu=None)` from `Minimum_variance.py`
- **Data Source**: Yahoo Finance via `yfinance` library
- **Period**: January 2022 - December 2024 (3 years)
- **Assets**: 5 major stocks across different sectors

### Mathematical Foundation
The minimum variance portfolio is found by solving:
```
minimize: w^T Σ w
subject to: w^T 1 = 1
```
Where:
- `w` = portfolio weights vector
- `Σ` = covariance matrix of asset returns
- `1` = vector of ones (portfolio weights sum to 100%)

### Code Structure
1. **Data Collection**: Robust fetching with error handling
2. **Return Calculation**: Daily returns from price data
3. **Covariance Estimation**: Annualized covariance matrix
4. **Optimization**: Using the existing `optPort` method
5. **Analysis**: Comprehensive performance metrics and visualization

## Practical Applications

### Investment Strategy
This minimum variance approach is suitable for:
- **Conservative investors** seeking capital preservation
- **Risk-averse portfolios** during market uncertainty
- **Defensive allocation** in volatile market conditions
- **Institutional investors** with strict risk constraints

### Limitations
- **Lower expected returns** compared to growth-oriented strategies
- **Concentration risk** in defensive sectors
- **Historical data dependency** - past correlations may not persist
- **Transaction costs** not considered in optimization

## Files Created

1. **`real_world_portfolio.py`**: Comprehensive portfolio construction class
2. **`simple_portfolio.py`**: Streamlined implementation with robust error handling
3. **`requirements.txt`**: Dependencies for the implementation
4. **`portfolio_analysis_summary.md`**: This summary document

## Conclusion

The project successfully demonstrates how to use the existing minimum variance portfolio method (`optPort`) with real-world data. The results show that the algorithm correctly identifies and weights assets to minimize portfolio risk, with Johnson & Johnson dominating the allocation due to its defensive characteristics during the analysis period.

**Key Achievement**: Reduced portfolio volatility to 14.33%, lower than any individual asset, while maintaining reasonable diversification across sectors.

This implementation serves as a practical foundation for building more sophisticated portfolio optimization strategies and risk management systems.
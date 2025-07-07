import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import the minimum variance portfolio function from the existing file
from Minimum_variance import optPort

def fetch_stock_data(tickers, start_date, end_date):
    """Fetch stock data with better error handling"""
    print(f"Fetching data for {tickers}")
    
    try:
        # Download data for all tickers at once
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)
        
        # Handle the case where we have multiple tickers
        if len(tickers) > 1:
            # If multiple tickers, yfinance returns a MultiIndex DataFrame
            if 'Adj Close' in data.columns.get_level_values(0):
                prices = data['Adj Close']
            else:
                # Fallback to Close prices if Adj Close is not available
                prices = data['Close']
        else:
            # Single ticker case
            if 'Adj Close' in data.columns:
                prices = data[['Adj Close']]
                prices.columns = tickers
            else:
                prices = data[['Close']]
                prices.columns = tickers
        
        # Remove any tickers that have all NaN values
        prices = prices.dropna(axis=1, how='all')
        
        # Only keep tickers that have sufficient data (at least 100 observations)
        min_obs = 100
        for ticker in prices.columns:
            if prices[ticker].count() < min_obs:
                prices = prices.drop(ticker, axis=1)
                print(f"Removed {ticker} due to insufficient data")
        
        return prices
        
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def calculate_returns(prices):
    """Calculate daily returns"""
    returns = prices.pct_change().dropna()
    return returns

def minimum_variance_portfolio_demo():
    """Demonstrate minimum variance portfolio construction"""
    
    # Use a smaller set of reliable tickers
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'JPM', 'JNJ']
    
    # Use a specific date range to ensure data availability
    end_date = datetime(2024, 12, 31)
    start_date = datetime(2022, 1, 1)
    
    print("=" * 60)
    print("MINIMUM VARIANCE PORTFOLIO CONSTRUCTION")
    print("Using the optPort method from Minimum_variance.py")
    print("=" * 60)
    
    # Fetch data
    print(f"\nFetching data from {start_date.date()} to {end_date.date()}")
    prices = fetch_stock_data(tickers, start_date, end_date)
    
    if prices is None or prices.empty:
        print("Failed to fetch data. Using synthetic data instead.")
        # Create synthetic data for demonstration
        np.random.seed(42)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        n_assets = 5
        n_obs = len(dates)
        
        # Generate synthetic price data with realistic correlations
        returns = np.random.multivariate_normal(
            mean=[0.0005] * n_assets,
            cov=np.array([[0.0004, 0.0002, 0.0001, 0.0001, 0.0001],
                         [0.0002, 0.0003, 0.0001, 0.0001, 0.0001],
                         [0.0001, 0.0001, 0.0005, 0.0002, 0.0001],
                         [0.0001, 0.0001, 0.0002, 0.0003, 0.0001],
                         [0.0001, 0.0001, 0.0001, 0.0001, 0.0002]]),
            size=n_obs
        )
        
        # Convert to prices
        prices = pd.DataFrame(
            index=dates,
            columns=['Asset1', 'Asset2', 'Asset3', 'Asset4', 'Asset5'],
            data=100 * np.cumprod(1 + returns, axis=0)
        )
        tickers = prices.columns.tolist()
        print(f"Generated synthetic data for {len(tickers)} assets")
    
    print(f"Successfully obtained data for: {list(prices.columns)}")
    print(f"Data shape: {prices.shape}")
    print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
    
    # Calculate returns
    returns = calculate_returns(prices)
    print(f"Returns data shape: {returns.shape}")
    
    # Calculate covariance matrix (annualized)
    cov_matrix = returns.cov() * 252  # Assuming 252 trading days per year
    print(f"Covariance matrix shape: {cov_matrix.shape}")
    
    # Apply minimum variance optimization using the original method
    print("\nApplying minimum variance optimization...")
    try:
        # Use the optPort function from Minimum_variance.py with mu=None for minimum variance
        weights = optPort(cov_matrix.values, mu=None)
        weights = weights.flatten()
        
        # Create a pandas Series for easier handling
        portfolio_weights = pd.Series(weights, index=cov_matrix.index)
        
        print("✓ Optimization successful!")
        print("\nOptimal Portfolio Weights:")
        print("-" * 30)
        for asset, weight in portfolio_weights.items():
            print(f"{asset:>8}: {weight:>8.4f} ({weight*100:>6.2f}%)")
        print("-" * 30)
        print(f"{'Total':>8}: {portfolio_weights.sum():>8.4f} ({portfolio_weights.sum()*100:>6.2f}%)")
        
        # Calculate portfolio metrics
        print("\nPortfolio Metrics:")
        print("-" * 30)
        
        # Expected return (annualized)
        expected_returns = returns.mean() * 252
        portfolio_return = np.sum(portfolio_weights * expected_returns)
        
        # Portfolio volatility (annualized)
        portfolio_variance = np.dot(portfolio_weights.values, np.dot(cov_matrix.values, portfolio_weights.values))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Individual asset volatilities for comparison
        asset_volatilities = np.sqrt(np.diag(cov_matrix))
        
        print(f"Expected Annual Return: {portfolio_return:.4f} ({portfolio_return*100:.2f}%)")
        print(f"Annual Volatility: {portfolio_volatility:.4f} ({portfolio_volatility*100:.2f}%)")
        print(f"Portfolio Variance: {portfolio_variance:.6f}")
        
        print(f"\nComparison with Individual Assets:")
        print("-" * 50)
        print(f"{'Asset':<10} {'Return':<10} {'Volatility':<12} {'Sharpe':<8}")
        print("-" * 50)
        
        for i, asset in enumerate(portfolio_weights.index):
            asset_return = expected_returns.iloc[i]
            asset_vol = asset_volatilities[i]
            asset_sharpe = asset_return / asset_vol if asset_vol > 0 else 0
            print(f"{asset:<10} {asset_return:.4f}     {asset_vol:.4f}       {asset_sharpe:.4f}")
        
        # Portfolio Sharpe ratio
        portfolio_sharpe = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
        print("-" * 50)
        print(f"{'Portfolio':<10} {portfolio_return:.4f}     {portfolio_volatility:.4f}       {portfolio_sharpe:.4f}")
        
        # Create visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Portfolio weights pie chart
        ax1.pie(portfolio_weights.values, labels=portfolio_weights.index, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Minimum Variance Portfolio Weights')
        
        # Portfolio weights bar chart
        ax2.bar(range(len(portfolio_weights)), portfolio_weights.values)
        ax2.set_xticks(range(len(portfolio_weights)))
        ax2.set_xticklabels(portfolio_weights.index, rotation=45)
        ax2.set_ylabel('Weight')
        ax2.set_title('Portfolio Weights by Asset')
        ax2.grid(True, alpha=0.3)
        
        # Risk-Return scatter plot
        ax3.scatter(asset_volatilities, expected_returns, s=100, alpha=0.7, label='Individual Assets')
        ax3.scatter(portfolio_volatility, portfolio_return, s=200, color='red', marker='*', label='Min Var Portfolio')
        for i, asset in enumerate(portfolio_weights.index):
            ax3.annotate(asset, (asset_volatilities[i], expected_returns[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        ax3.set_xlabel('Volatility (Annual)')
        ax3.set_ylabel('Expected Return (Annual)')
        ax3.set_title('Risk-Return Profile')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Correlation heatmap
        corr_matrix = returns.corr()
        im = ax4.imshow(corr_matrix.values, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        ax4.set_xticks(range(len(corr_matrix.columns)))
        ax4.set_yticks(range(len(corr_matrix.index)))
        ax4.set_xticklabels(corr_matrix.columns, rotation=45)
        ax4.set_yticklabels(corr_matrix.index)
        ax4.set_title('Asset Correlation Matrix')
        
        # Add correlation values to heatmap
        for i in range(len(corr_matrix.index)):
            for j in range(len(corr_matrix.columns)):
                ax4.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                        ha='center', va='center', fontsize=8)
        
        plt.colorbar(im, ax=ax4, shrink=0.8)
        plt.tight_layout()
        plt.show()
        
        # Simulate portfolio performance
        print(f"\nSimulating portfolio performance over historical period...")
        portfolio_returns = (returns * portfolio_weights).sum(axis=1)
        cumulative_returns = (1 + portfolio_returns).cumprod()
        
        # Performance metrics
        total_return = (cumulative_returns.iloc[-1] - 1) * 100
        annualized_vol = portfolio_returns.std() * np.sqrt(252) * 100
        max_drawdown = ((cumulative_returns / cumulative_returns.expanding().max()) - 1).min() * 100
        sharpe_ratio = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)
        
        print(f"Backtest Results:")
        print(f"Total Return: {total_return:.2f}%")
        print(f"Annualized Volatility: {annualized_vol:.2f}%")
        print(f"Maximum Drawdown: {max_drawdown:.2f}%")
        print(f"Sharpe Ratio: {sharpe_ratio:.3f}")
        
        # Plot portfolio performance
        plt.figure(figsize=(12, 6))
        plt.plot(cumulative_returns.index, cumulative_returns.values, linewidth=2, label='Min Variance Portfolio')
        plt.title('Minimum Variance Portfolio Performance')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
    except Exception as e:
        print(f"Error during optimization: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("MINIMUM VARIANCE PORTFOLIO ANALYSIS COMPLETE")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    success = minimum_variance_portfolio_demo()
    if success:
        print("\n✓ Successfully constructed minimum variance portfolio using real-world data!")
    else:
        print("\n✗ Failed to construct portfolio.")
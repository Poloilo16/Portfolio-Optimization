import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import the minimum variance portfolio function from the existing file
from Minimum_variance import optPort, corr2cov

class MinimumVariancePortfolio:
    def __init__(self, tickers, start_date=None, end_date=None):
        """
        Initialize the Minimum Variance Portfolio with stock tickers
        
        Parameters:
        tickers: list of stock symbols
        start_date: start date for historical data (default: 2 years ago)
        end_date: end date for historical data (default: today)
        """
        self.tickers = tickers
        self.end_date = end_date or datetime.now()
        self.start_date = start_date or (self.end_date - timedelta(days=2*365))
        self.data = None
        self.returns = None
        self.cov_matrix = None
        self.optimal_weights = None
        
    def fetch_data(self):
        """Fetch historical stock data from Yahoo Finance"""
        print(f"Fetching data for {len(self.tickers)} stocks from {self.start_date.date()} to {self.end_date.date()}")
        
        try:
            # Download data for all tickers
            self.data = yf.download(self.tickers, start=self.start_date, end=self.end_date)['Adj Close']
            
            # Handle single ticker case
            if len(self.tickers) == 1:
                self.data = pd.DataFrame(self.data)
                self.data.columns = self.tickers
                
            # Drop any tickers with insufficient data
            self.data = self.data.dropna(axis=1, how='any')
            self.tickers = list(self.data.columns)
            
            print(f"Successfully fetched data for {len(self.tickers)} stocks")
            print(f"Data shape: {self.data.shape}")
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            return False
        
        return True
    
    def calculate_returns(self):
        """Calculate daily returns"""
        if self.data is None:
            print("No data available. Please fetch data first.")
            return False
            
        # Calculate daily returns
        self.returns = self.data.pct_change().dropna()
        
        print(f"Calculated returns for {len(self.tickers)} stocks")
        print(f"Returns data shape: {self.returns.shape}")
        print(f"Sample period: {self.returns.index[0].date()} to {self.returns.index[-1].date()}")
        
        return True
    
    def calculate_covariance_matrix(self):
        """Calculate the covariance matrix of returns"""
        if self.returns is None:
            print("No returns data available. Please calculate returns first.")
            return False
        
        # Calculate annualized covariance matrix (multiply by 252 trading days)
        self.cov_matrix = self.returns.cov() * 252
        
        print("Calculated covariance matrix")
        print(f"Matrix shape: {self.cov_matrix.shape}")
        
        return True
    
    def optimize_portfolio(self):
        """Find the minimum variance portfolio weights using the existing method"""
        if self.cov_matrix is None:
            print("No covariance matrix available. Please calculate it first.")
            return False
        
        try:
            # Use the optPort function from the existing file with mu=None for minimum variance
            weights = optPort(self.cov_matrix.values, mu=None)
            
            # Store weights as a pandas Series
            self.optimal_weights = pd.Series(weights.flatten(), index=self.tickers)
            
            print("Portfolio optimization completed successfully!")
            print("Optimal weights:")
            for ticker, weight in self.optimal_weights.items():
                print(f"  {ticker}: {weight:.4f} ({weight*100:.2f}%)")
                
            return True
            
        except Exception as e:
            print(f"Error during optimization: {e}")
            return False
    
    def calculate_portfolio_metrics(self):
        """Calculate portfolio risk and return metrics"""
        if self.optimal_weights is None or self.returns is None:
            print("Portfolio not optimized or returns not available.")
            return None
        
        # Calculate portfolio return (annualized)
        portfolio_return = np.sum(self.optimal_weights * self.returns.mean() * 252)
        
        # Calculate portfolio volatility (annualized)
        portfolio_variance = np.dot(self.optimal_weights.values.T, 
                                  np.dot(self.cov_matrix.values, self.optimal_weights.values))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Calculate Sharpe ratio (assuming risk-free rate of 2%)
        risk_free_rate = 0.02
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
        
        metrics = {
            'Expected Return': portfolio_return,
            'Volatility': portfolio_volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Variance': portfolio_variance
        }
        
        print("\nPortfolio Metrics:")
        print(f"Expected Annual Return: {portfolio_return:.4f} ({portfolio_return*100:.2f}%)")
        print(f"Annual Volatility: {portfolio_volatility:.4f} ({portfolio_volatility*100:.2f}%)")
        print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
        print(f"Portfolio Variance: {portfolio_variance:.6f}")
        
        return metrics
    
    def plot_weights(self):
        """Plot the optimal portfolio weights"""
        if self.optimal_weights is None:
            print("Portfolio not optimized.")
            return
        
        plt.figure(figsize=(12, 8))
        
        # Create pie chart
        plt.subplot(2, 1, 1)
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.optimal_weights)))
        wedges, texts, autotexts = plt.pie(self.optimal_weights.values, 
                                          labels=self.optimal_weights.index,
                                          autopct='%1.1f%%', 
                                          colors=colors,
                                          startangle=90)
        plt.title('Minimum Variance Portfolio Weights', fontsize=14, fontweight='bold')
        
        # Create bar chart
        plt.subplot(2, 1, 2)
        bars = plt.bar(range(len(self.optimal_weights)), self.optimal_weights.values, 
                      color=colors, alpha=0.7)
        plt.xticks(range(len(self.optimal_weights)), self.optimal_weights.index, rotation=45)
        plt.ylabel('Weight')
        plt.title('Portfolio Weights by Asset', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.show()
    
    def plot_correlation_heatmap(self):
        """Plot correlation heatmap of the assets"""
        if self.returns is None:
            print("No returns data available.")
            return
        
        plt.figure(figsize=(10, 8))
        correlation_matrix = self.returns.corr()
        
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title('Asset Correlation Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def simulate_portfolio_performance(self):
        """Simulate portfolio performance over the historical period"""
        if self.optimal_weights is None or self.returns is None:
            print("Portfolio not optimized or returns not available.")
            return
        
        # Calculate portfolio daily returns
        portfolio_returns = (self.returns * self.optimal_weights).sum(axis=1)
        
        # Calculate cumulative returns
        cumulative_returns = (1 + portfolio_returns).cumprod()
        
        # Plot performance
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(cumulative_returns.index, cumulative_returns.values, 
                linewidth=2, color='navy', label='Min Variance Portfolio')
        plt.title('Minimum Variance Portfolio Performance', fontsize=14, fontweight='bold')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        plt.plot(portfolio_returns.index, portfolio_returns.values, 
                linewidth=1, color='darkred', alpha=0.7)
        plt.title('Daily Portfolio Returns', fontsize=12, fontweight='bold')
        plt.ylabel('Daily Return')
        plt.xlabel('Date')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print performance statistics
        total_return = (cumulative_returns.iloc[-1] - 1) * 100
        volatility = portfolio_returns.std() * np.sqrt(252) * 100
        max_drawdown = ((cumulative_returns / cumulative_returns.expanding().max()) - 1).min() * 100
        
        print(f"\nPortfolio Performance Summary:")
        print(f"Total Return: {total_return:.2f}%")
        print(f"Annualized Volatility: {volatility:.2f}%")
        print(f"Maximum Drawdown: {max_drawdown:.2f}%")
        
        return portfolio_returns, cumulative_returns

def main():
    """Main function to demonstrate minimum variance portfolio construction"""
    
    # Define a diversified set of stocks from different sectors
    tickers = [
        'AAPL',  # Technology
        'MSFT',  # Technology  
        'GOOGL', # Technology
        'JPM',   # Financial
        'JNJ',   # Healthcare
        'PG',    # Consumer Staples
        'XOM',   # Energy
        'GE',    # Industrial
        'KO',    # Consumer Staples
        'DIS'    # Communication Services
    ]
    
    print("=" * 60)
    print("MINIMUM VARIANCE PORTFOLIO CONSTRUCTION")
    print("Using real-world stock data")
    print("=" * 60)
    
    # Create portfolio instance
    portfolio = MinimumVariancePortfolio(tickers)
    
    # Step 1: Fetch data
    if not portfolio.fetch_data():
        print("Failed to fetch data. Exiting.")
        return
    
    # Step 2: Calculate returns
    if not portfolio.calculate_returns():
        print("Failed to calculate returns. Exiting.")
        return
    
    # Step 3: Calculate covariance matrix
    if not portfolio.calculate_covariance_matrix():
        print("Failed to calculate covariance matrix. Exiting.")
        return
    
    # Step 4: Optimize portfolio
    if not portfolio.optimize_portfolio():
        print("Failed to optimize portfolio. Exiting.")
        return
    
    # Step 5: Calculate metrics
    metrics = portfolio.calculate_portfolio_metrics()
    
    # Step 6: Visualizations
    print("\nGenerating visualizations...")
    portfolio.plot_weights()
    portfolio.plot_correlation_heatmap()
    
    # Step 7: Simulate performance
    print("\nSimulating portfolio performance...")
    portfolio.simulate_portfolio_performance()
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
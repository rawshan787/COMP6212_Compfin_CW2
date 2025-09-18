"""
Rolls-Royce Stock Analysis Script
---------------------------------
This script performs statistical analysis of Rolls-Royce (RR.L) stock returns, including:
- Distribution analysis with comparison to normal distribution
- Autocorrelation of returns and squared returns to detect patterns
- Calculation of key statistical measures (skewness, kurtosis)

Install packages using:
- pip install yfinance pandas numpy matplotlib scipy seaborn

Run it by executing this command into the terimal at the script directory:
- python task4.py
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns

# Set global plot parameters for consistency
plt.rcParams.update({'font.size': 16})
PLOT_LINEWIDTH = 2
FIGSIZE_LARGE = (12, 6)
FIGSIZE_MEDIUM = (10, 6)


def fetch_stock_data(ticker, years=10):
    """
    Fetch historical stock data for the specified ticker.
    
    Args:
        ticker (str): Stock ticker symbol
        years (int): Number of years of historical data to retrieve
        
    Returns:
        DataFrame: Historical stock data with OHLC prices and volume
    """
    end_date = pd.to_datetime("2025-05-05")
    start_date = end_date - timedelta(days=365 * years)
    
    print(f"Fetching {years} years of data for {ticker} ({start_date.date()} to {end_date.date()})")
    
    data = yf.download(
        ticker, 
        start=start_date, 
        end=end_date,
        progress=False,
    )
    
    return data


def calculate_log_returns(price_data):
    """
    Calculate logarithmic returns from price data.
    
    Args:
        price_data (Series): Time series of closing prices
        
    Returns:
        Series: Log returns (first observation is NaN and dropped)
    """
    return np.log(price_data / price_data.shift(1)).dropna()


def autocorrelation(time_series, max_lag):
    """
    Calculate autocorrelation function for time series data.
    
    Args:
        time_series (array-like): Time series data
        max_lag (int): Maximum lag to calculate autocorrelation for
        
    Returns:
        array: Autocorrelation values for each lag from 0 to max_lag
    """
    N = len(time_series)
    mean = np.mean(time_series)
    
    # Initialize array to store autocorrelation values
    acf = np.zeros(max_lag + 1)
    
    # Calculate autocorrelation for each lag tau
    for tau in range(max_lag + 1):
        # Numerator: sum of (r(t_k)-r_hat)(r(t_k+tau)-r_hat)
        numerator = 0
        for k in range(N - tau):
            numerator += (time_series[k] - mean) * (time_series[k + tau] - mean)
        
        # Denominator: sqrt(sum((r(t_k)-rr_hat)**2)) * sqrt(sum((r(t_k+tau)-r_hat)**2))
        sum1 = sum((time_series[k] - mean)**2 for k in range(N))
        sum2 = sum((time_series[k] - mean)**2 for k in range(tau, N))
        
        denominator = np.sqrt(sum1) * np.sqrt(sum2)
        
        # Complete autocorrelation calculation
        acf[tau] = numerator / denominator
    
    return acf


def calculate_excess_kurtosis(data):
    """
    Calculate excess kurtosis (kurtosis - 3).
    
    Args:
        data (array-like): Data to calculate kurtosis for
        
    Returns:
        float: Excess kurtosis value
    """
    mean = np.mean(data)
    std = np.std(data, ddof=0) 
    z = (data - mean) / std
    return float(np.mean(z**4) - 3)


def plot_return_distribution(returns, mu, sigma, skewness, excess_kurtosis):
    """
    Plot histogram of returns with KDE and normal distribution overlay.
    
    Args:
        returns (Series): Log returns
        mu (float): Mean of returns
        sigma (float): Standard deviation of returns
        skewness (float): Skewness of the distribution
        excess_kurtosis (float): Excess kurtosis of the distribution
    """
    plt.figure(figsize=FIGSIZE_LARGE)
    
    # Plot histogram
    sns.histplot(
        returns,  
        bins=100,
        kde=False,
        alpha=0.6,
        linewidth=0.5,
        stat="density",
        label='Density'
    )
    
    # Plot KDE
    sns.kdeplot(
        returns,
        linewidth=PLOT_LINEWIDTH,
        label="KDE"
    )
    
    # Add normal distribution overlay
    x = np.linspace(returns.min(), returns.max(), 100)
    plt.plot(x, 
             stats.norm.pdf(x, mu, sigma), 
             'r-', 
             linewidth=PLOT_LINEWIDTH,
             label='Normal Distribution')
    
    # Add textbox with statistics
    stats_text = f"Skewness = {skewness:,.5f}\nExcess Kurtosis = {excess_kurtosis:,.5f}"
    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
    
    plt.title("Distribution of Log Returns")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend(loc='upper right')  
    plt.grid(linestyle='--', alpha=0.7)
    plt.tight_layout()


def plot_autocorrelation(time_series, max_lag, title):
    """
    Plot autocorrelation function with confidence intervals.
    
    Args:
        time_series (array-like): Time series data
        max_lag (int): Maximum lag to calculate autocorrelation for
        title (str): Plot title
    """
    # Calculate autocorrelation
    acf_values = autocorrelation(time_series, max_lag)
    
    # Calculate confidence intervals
    n = len(time_series)
    conf_level = 1.96 / np.sqrt(n)  # 95% confidence interval
    
    # Create the plot
    plt.figure(figsize=FIGSIZE_MEDIUM)
    
    # Plot the acf values
    plt.bar(range(len(acf_values)), acf_values, width=0.3, color='blue', alpha=0.7)
    plt.axhline(y=0, linestyle='-', color='black', linewidth=0.5)
    
    # Add confidence intervals
    plt.axhline(y=conf_level, linestyle='--', color='red', linewidth=1)
    plt.axhline(y=-conf_level, linestyle='--', color='red', linewidth=1)
    
    # Create shaded confidence interval
    x = np.arange(-0.5, max_lag + 0.5)
    plt.fill_between(x, -conf_level, conf_level, color='gray', alpha=0.2)
    
    # Labels and formatting
    plt.title(title)
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.xlim([-0.5, max_lag + 0.5])
    if title == 'Autocorrelation of Returns':
        plt.ylim([-0.2, 1])
    plt.grid(True)
    plt.tight_layout()


def main():
    ticker = "RR.L"
    max_lag = 50
    
    # 1. Fetch Data
    data = fetch_stock_data(ticker)
    
    # 2. Calculate log returns
    log_returns = calculate_log_returns(data['Close'])
    
    # 3. Calculate statistics
    mu = float(log_returns.mean())
    sigma = float(log_returns.std())
    skewness = float(stats.skew(log_returns))
    excess_kurtosis = calculate_excess_kurtosis(log_returns)
    
    # 4. Print summary statistics
    print("\nSummary Statistics:")
    print(f"Mean of log returns: {mu}")
    print(f"Standard deviation of log returns: {sigma}")
    print(f"Skewness of log returns: {skewness}")
    print(f"Excess kurtosis of log returns: {excess_kurtosis}")
    
    # 5. Plot distribution of returns
    plot_return_distribution(log_returns, mu, sigma, skewness, excess_kurtosis)
    
    # 6. Plot autocorrelation of returns
    returns_series = log_returns.values
    plot_autocorrelation(returns_series, max_lag, 'Autocorrelation of Returns')
    
    # 7. Plot autocorrelation of squared returns (volatility clustering)
    squared_returns = returns_series**2
    plot_autocorrelation(squared_returns, max_lag, 'Autocorrelation of Squared Returns')
    
    plt.show()


if __name__ == "__main__":
    main()
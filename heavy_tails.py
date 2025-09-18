import yfinance as yf
import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter

def fetch_stock_data(ticker: str, end_date: str, years: int) -> pd.DataFrame:
    """
    Fetches historical stock data for a given ticker over a specified period.
    
    Args:
        ticker (str): Stock symbol (e.g., 'RR.L' for Rolls-Royce)
        end_date (str): End date in 'YYYY-MM-DD' format
        years (int): Number of years of historical data to fetch
    
    Returns:
        pd.DataFrame: OHLCV data with dates as index
    """
    end = pd.to_datetime(end_date)
    start = end - timedelta(days=365 * years)
    
    try:
        data = yf.download(
            ticker, 
            start=start, 
            end=end,
            progress=False,
        )
        return data
    
    except Exception as e:
        print(f"Error fetching data for {ticker}: {str(e)}")
        return pd.DataFrame()

def calculate_log_returns(data: pd.DataFrame) -> pd.Series:
    """
    Calculates daily log returns from closing prices.
    """
    return np.log(data['Close'] / data['Close'].shift(1))

def plot_log_return_distribution(log_returns: pd.Series, ticker: str) -> None:
    """
    Plots the distribution of log returns as a histogram.
    """
    plt.hist(log_returns, bins=50)
    plt.show()
    

def plot_closing_prices(data: pd.DataFrame, ticker: str) -> None:
    """
    Plots the adjusted closing prices of a stock.
    """
    if data.empty:
        print("No data to plot!")
        return
    
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['Close'], label='Adjusted Close Price', color='blue')
    
    plt.title(f'{ticker} Adjusted Closing Prices', fontsize=14)
    plt.ylabel('Price (£)')
    plt.xlabel('Date')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=45)
    
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    rr_data = fetch_stock_data("RR.L", "2025-05-05", 5)
    
    if not rr_data.empty:
        print(f"\nRolls-Royce (RR.L) data - last 5 years up to 2025-05-05:")
        print(f"Date range: {rr_data.index[0].date()} to {rr_data.index[-1].date()}")
        print(rr_data.tail())
        
        plot_closing_prices(rr_data, "RR.L")
        
        # Calculate and plot log returns
        log_returns = calculate_log_returns(rr_data)
        plot_log_return_distribution(log_returns, "RR.L")
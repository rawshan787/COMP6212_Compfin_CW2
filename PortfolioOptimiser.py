import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, linregress
from tqdm import tqdm


"""
To run the Portfolio optimiser the following libraries are needed:
pip install numpy pandas yfinance matplotlib seaborn scipy tqdm


You can run it with the following command:
python PortfolioOptimiser.py
"""


"""
Optimises a portfolio of stocks using a Brute Force Approach across a time period.
"""
class PortfolioOptimiser:

    def __init__(self, tickers):
        """
        Initialise the PortfolioOptimiser with a list of ticker symbols.

        Args:
            tickers (list): A list of stock ticker strings.
        """

        np.random.seed(78930123)
        self.tickers = tickers

    def get_data(self, start_date = '2018-01-01', end_date = '2020-12-31'):
        """
        Download historical closing price data and compute returns, then split into training and testing sets.

        Args:
            start_date (str): Start date for data download (YYYY-MM-DD).
            end_date (str): End date for data download (YYYY-MM-DD).

        Returns:
            tuple: (train_returns, test_returns) as two pandas DataFrames.
        """
        # Download data from closing time
        data = yf.download(self.tickers, start=start_date, end=end_date)['Close']

        # Calculate daily returns
        returns = self.compute_returns(data)

        # Split into training and testing
        split_idx = len(data) // 2
        train_returns = returns.iloc[:split_idx]
        test_returns = returns.iloc[split_idx:]

        return train_returns, test_returns

    def compute_returns(self, data):
        """
        Compute daily percentage returns from price data.

        Args:
            data (DataFrame): DataFrame of price data with dates as index and tickers as columns.

        Returns:
            DataFrame: DataFrame of daily returns.
        """

        prices = data.values
        T, N = prices.shape

        # Create empty array to store the daily returns
        returns = np.zeros((T - 1, N))

        # Calculate returns as: (P_t / P_{t-1}) - 1
        for t in range(1, T):
            for i in range(N):
                returns[t - 1, i] = (prices[t, i] / prices[t - 1, i]) - 1

        return pd.DataFrame(returns, columns=data.columns, index=data.index[1:])

    def compute_covariance_matrix(self, data):
        """
        Compute the sample covariance matrix for a set of returns.

        Args:
            data (DataFrame): DataFrame of daily returns.

        Returns:
            ndarray: Covariance matrix.
        """

        data = data.values
        T, N = data.shape
        means = np.mean(data, axis=0)
        cov = np.zeros((N, N))

        # Computes covariance between each item in the dataframe using the equation found in the report
        for i in range(N):
            for j in range(N):
                cov[i, j] = np.sum((data[:, i] - means[i]) * (data[:, j] - means[j])) / (T - 1)
        return cov

    def compute_mean(self, data):
        """
        Computes the mean of a set of data

        Args:
            data (DataFrame): DataFrame of daily returns.

        Returns:
            float: Mean value.
        """
        return data.mean()

    def brute_force_optimiser(self, train_data, n_points = 10000000):
        """
        Brute-force search to find optimal portfolio weights that maximise the Sharpe ratio.

        Args:
            train_data (DataFrame): Historical returns used for training.
            n_points (int): Number of random portfolios to generate.

        Returns:
            tuple: (best_weights, all_weights_tested)
        """

        # Stores all weights calculated
        results = []

        #Used to store best results
        best_sharpe = -np.inf
        best_weights = np.random.random(len(tickers))
        best_weights /= np.sum(best_weights)

        # Calculate mean returns and covariance matrix from training data
        mean_returns = self.compute_mean(train_data)
        cov_matrix = self.compute_covariance_matrix(train_data)

        for _ in tqdm(range(n_points), desc="Brute Force Portfolio", ncols=100):
            # weights = np.random.random(len(tickers))

            # Generating weights with limits seems to work well in ensuring that extreme values (e.g. having near 0 or
            # near 1 weights doesn't occur)
            weights = np.random.uniform(0.05, 0.2, size=len(tickers))
            weights /= np.sum(weights)

            # Calculate the return, volatility and sharpe ratio (assuming risk-free rate is 0)
            port_return = np.dot(weights, mean_returns)
            port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe_ratio = port_return / port_vol

            # Updates weights to best sharpe ratio
            if (sharpe_ratio > best_sharpe):
                best_sharpe = sharpe_ratio
                best_weights = weights

            results.append(weights)

        return best_weights, results

    def one_over_n_optimiser(self):
        """
        Returns equal-weighted (1/N) portfolio.

        Returns:
            ndarray: Equal weights for each asset.
        """
        return np.ones(len(tickers)) / len(tickers)

    def get_vol_return(self, weights, returns_data):
        """
        Compute the return and volatility from the given weights and data.

        Args:
            weights (ndarray): Portfolio weights.
            returns_data (DataFrame): Historical return data.

        Returns:
            tuple: (expected return, volatility)
        """
        port_return = np.dot(self.compute_mean(returns_data), weights)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(self.compute_covariance_matrix(returns_data), weights)))
        return port_return, port_vol

    def visualise_covariance_matrix(self, data):
        """
         Compute and plot the heatmap of the annualised return covariance matrix.

         Args:
             data (DataFrame): Daily returns data.
         """

        #  Aggregate returns annually
        yearly_returns = (1 + data).resample('Y').prod() - 1

        # Calculate covariance of annualised returns
        cov_matrix = yearly_returns.cov()

        # Plot covariance matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cov_matrix, annot=True, fmt='.5f', cmap='coolwarm', square=True, linewidths=0.5)
        plt.title("Covariance Matrix of Annualised Returns", fontsize=16)
        plt.xlabel("Stocks", fontsize=12)
        plt.ylabel("Stocks", fontsize=12)
        plt.tight_layout()
        plt.savefig('CovarianceMatrix.png')


    def visualise_weights(self, brute_weights, data, name="weights.png"):
        # Convert the weights to percentage form
        brute_weights_percentage = np.array(brute_weights) * 100

        # Create a bar chart
        plt.figure(figsize=(10, 6))
        plt.bar(data.columns.tolist(), brute_weights_percentage, color='deepskyblue')
        plt.title('Portfolio Allocation by Assets', fontsize=16)
        plt.xlabel('Stocks', fontsize=12)
        plt.ylabel('Weight (%)', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(name)


    def visualise_brute_force(self, weights, train_data, test_data):
        """
        Visualise all the calculated weights from the brute force optimiser on both the training and testing sets
        This function shouldn't be used.

        Args:
            weights (ndarray): Portfolio weights.
            train_data (DataFrame): Historical return data the optimiser was trained on.
            test_data (DataFrame): Historical return data the optimiser is tested on.
        """

        train_sharpe = []
        test_sharpe = []

        # Compute the necessary metrics for return and volatility calculations
        train_cov = self.compute_covariance_matrix(train_data)
        test_cov = self.compute_covariance_matrix(test_data)
        train_mean = self.compute_mean(train_data)
        test_mean = self.compute_mean(test_data)
        idx = 0

        for _ in tqdm(weights, desc="Brute Force Portfolio Visualisation", ncols=100):
            weight = weights[idx]

            # Calculate on Train Data
            train_ret = np.dot(weight, train_mean)
            train_vol = np.sqrt(np.dot(weight.T, np.dot(train_cov, weight)))

            # Calculate on Test Data
            test_ret = np.dot(weight, test_mean)
            test_vol = np.sqrt(np.dot(weight.T, np.dot(test_cov, weight)))

            # Calculate Sharpe
            train_sharpe.append(train_ret/train_vol)
            test_sharpe.append(test_ret/test_vol)
            idx+=1


        x_axis = range(1, len(train_sharpe) + 1)

        # Plot the training sharpe ratio and testing sharpe for calculated weights
        plt.figure(figsize=(10, 6))
        plt.plot(x_axis, train_sharpe, label='Train Sharpe Ratio', alpha=0.7)
        plt.plot(x_axis, test_sharpe, label='Test Sharpe Ratio', alpha=0.7)
        plt.xlabel("Brute Force Iterations")
        plt.ylabel("Sharpe Ratio")
        plt.title("Brute Force Portfolio Performance Over Time")
        plt.legend()
        plt.grid(True)
        plt.savefig("BruteForce.png")

        # Calculate Pearson correlation coefficient
        corr, p_value = pearsonr(train_sharpe, test_sharpe)
        print(f"Pearson Correlation: {corr:.4f}, p-value: {p_value}")

        # Create line of best fit
        slope, intercept, r_value, p_val, std_err = linregress(train_sharpe, test_sharpe)
        line = slope * np.array(train_sharpe) + intercept

        # Draw line of best fit for train sharpe vs test sharpe
        plt.figure(figsize=(8, 6))
        plt.scatter(train_sharpe, test_sharpe, alpha=0.3, s=1)
        plt.plot(train_sharpe, line, color='red', label='Best Fit Line')
        plt.title(f"Train vs Test Sharpe Ratios\nCorrelation: {corr:.4f} Using Brute Force Weights")
        plt.xlabel("Train Sharpe Ratio")
        plt.ylabel("Test Sharpe Ratio")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("BruteForceCorrelation.png")

    def visualise_results(self, train_data, test_data, brute_best_weights, over_n_weights):
        """
        Plots the retsults by comparing the risk-return performance of
        the brute-force optimised portfolio and the 1/N portfolio on both training and testing data.

        Args:
            train_data (DataFrame): Training returns.
            test_data (DataFrame): Testing returns.
            brute_best_weights (ndarray): Optimal weights from brute-force optimisation.
            over_n_weights (ndarray): Equal weights for the 1/N portfolio.
        """

        train_brute_return, train_brute_vol = portfolio.get_vol_return(brute_best_weights, train_data)
        test_brute_return, test_brute_vol = portfolio.get_vol_return(brute_best_weights, test_data)

        train_n_return, train_n_vol = portfolio.get_vol_return(over_n_weights, train_data)
        test_n_return, test_n_vol = portfolio.get_vol_return(over_n_weights, test_data)

        plt.figure(figsize=(10, 6))

        # Plot brute force portfolio
        plt.scatter(train_brute_vol, train_brute_return, color='blue', marker='x', s=200,
                    label='Brute Force Portfolio (Train)')
        plt.scatter(test_brute_vol, test_brute_return, color='blue', marker='o', s=200,
                    label='Brute Force Portfolio (Test)')

        # Plot 1/n portfolio
        plt.scatter(train_n_vol, train_n_return, color='red', marker='x', s=200, label='1/n Portfolio (Train)')
        plt.scatter(test_n_vol, test_n_return, color='red', marker='o', s=200, label='1/n Portfolio (Test)')

        plt.title('Risk vs Return on Training and Testing Portfolio')
        plt.xlabel('Volatility (Risk)')
        plt.ylabel('Return')
        plt.legend()
        plt.grid(True)
        plt.savefig("Results.png")

    def visualise_yearly_returns(self, data):
        """
        Calculate and display yearly returns in percentage format, including their average.

        Args:
            data (DataFrame): DataFrame of daily returns.
        """

        # Convert daily returns to cumulative yearly returns
        yearly_returns = (1 + data).resample('Y').prod() - 1

        # Convert to percentage and round
        yearly_returns_percentage = yearly_returns * 100
        yearly_returns_percentage = yearly_returns_percentage.round(3)

        # Transpose to show tickers as rows
        table = yearly_returns_percentage.T

        # Add average column
        table["Average %"] = table.mean(axis=1).round(3)

        # Show all columns in console
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print(table)
        pd.reset_option('display.max_columns')
        pd.reset_option('display.width')







if __name__ == "__main__":
    tickers = ['HSBA.L', 'LLOY.L', 'STAN.L', 'BARC.L', 'SGE.L', 'REL.L', 'PRTC.L', 'TSCO.L', 'ULVR.L', 'SBRY.L']
    start_date = '2016-01-01'
    end_date = '2019-12-29'

    # Get data from yahoo finance for each tickers and split into train/test
    portfolio = PortfolioOptimiser(tickers)
    train_data, test_data = portfolio.get_data(start_date, end_date)

    # Visualise the yearly returns and covariance matrix
    portfolio.visualise_yearly_returns(train_data)
    portfolio.visualise_covariance_matrix(train_data)

    # Compute weights for brute force optimiser and 1/n
    brute_best_weights, brute_weights = portfolio.brute_force_optimiser(train_data)
    over_n_weights = portfolio.one_over_n_optimiser()

    print(f"\nBrute force weights: {brute_best_weights}")

    # Compute the volatility and returns of both portfolios using weights
    train_brute_return, train_brute_vol = portfolio.get_vol_return(brute_best_weights, train_data)
    test_brute_return, test_brute_vol = portfolio.get_vol_return(brute_best_weights, test_data)

    train_n_return, train_n_vol = portfolio.get_vol_return(over_n_weights, train_data)
    test_n_return, test_n_vol = portfolio.get_vol_return(over_n_weights, test_data)

    # Output results
    print("\nBrute Force Portfolio:")
    print(f"  Train Return: {train_brute_return:.6f}., Train Volatility: {train_brute_vol:.6f}")
    print(f"  Test Return {test_brute_return:.6f}, Test Volatility {test_brute_vol:.6f}")

    print("\n1/n Portfolio:")
    print(f"  Train Return: {train_n_return:.6f}, Train Volatility: {train_n_vol:.6f}")
    print(f"  Test Return {test_n_return:.6f}, Test Volatility {test_n_vol:.6f}")

    # Visualise results
    # portfolio.visualise_brute_force(brute_weights, train_data, test_data)
    portfolio.visualise_weights(brute_best_weights, train_data, "Weights.png")
    portfolio.visualise_results(train_data, test_data, brute_best_weights, over_n_weights)

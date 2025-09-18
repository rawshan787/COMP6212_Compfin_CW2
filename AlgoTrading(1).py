import os
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from dateutil.relativedelta import relativedelta
from statsmodels.tools.tools import add_constant
from statsmodels.regression.linear_model import OLS

# === Pairs Trading Class ===
""" 
# This class implements a pairs trading strategy using various methods to identify and trade pairs of stocks.
# It fetches stock data, calculates signals, executes trades, and plots results.
"""
class PairsTrading():
    def __init__(self, capital, strategies, enhancement:bool, ticker_a=None, ticker_b=None, start_date=None, end_date=None):
        self.capital = capital
        self.strategies = strategies
        self.enhancement  = enhancement

        if ticker_a == None:

            self.start_date = "2023-01-01"
            self.end_date = "2024-01-01"

            ticker_count_1 = 0
            ticker_count_2 = 1
            tickers = [
              "AZN.L", "RKT.L", "HSBA.L", "SHEL.L", "ULVR.L", "REL.L", "BATS.L", "RR.L", "BP.L",
              "LSEG.L", "RIO.L", "GSK.L", "BA.L", "NG.L", "DGE.L", "BARC.L", "CPG.L",
              "LLOY.L", "III.L", "NWG.L", "HLN.L", "EXPN.L", "GLEN.L", "CCEP.L", "AAL.L", "STAN.L", "TSCO.L",
              "IMB.L", "PRU.L", "SSE.L", "AHT.L", "ANTO.L", "VOD.L", "BT.A.L",
              "AV.L", "ABF.L", "IAG.L", "NXT.L", "IHG.L", "LGEN.L", "CCH.L", "SGE.L", "SMT.L", "HLMA.L", "INF.L", "ADM.L", "SN.L",
              "SGRO.L", "RTO.L", "BNZL.L", "SVT.L", "ITRK.L", "PSON.L", "AUTO.L",
              "UU.L", "CNA.L", "FRES.L", "MKS.L", "PSH.L", "SMIN.L", "BTRW.L", "WPP.L", "SBRY.L", "WEIR.L", "AAF.L", "MRO.L", "PHNX.L",
              "ICG.L", "STJ.L", "DPLM.L", "RMV.L", "SDR.L", "BEZ.L", "KGF.L",
              "CTEC.L", "FCIT.L", "MNDI.L", "MNG.L", "EDV.L", "GAW.L", "DCC.L", "WTB.L", "IMI.L", "ALW.L", "SPX.L", "ENT.L", "JD.L",
              "HWDN.L", "LAND.L", "CRDA.L", "HIK.L", "PSN.L", "BAB.L", "BKG.L",
              "TW.L", "EZJ.L", "UTG.L", "LMP.L", "HSX.L", "PCT.L"
            ] # the ftse100 companies taken from: https://www.londonstockexchange.com/indices/ftse-100/constituents/table?page=1

            strategy_signals = {strategy: None for strategy in strategies}

            # === Finding suitable pairs ===
            while ticker_count_1 < len(tickers) - 1:
                self.ticker_a = tickers[ticker_count_1]
                self.ticker_b = tickers[ticker_count_2]

                self.stock_a, self.stock_b = self._fetch_data(self.ticker_a, self.ticker_b, self.start_date, self.end_date)

                print(self.ticker_a, self.ticker_b)

                success = True
                # === Calculate Signals ===
                for strategy in strategies:
                    if strategy == "mean_reversion_spread":
                        signals = self._mean_reversion_spread()
                    elif strategy == "mean_reversion_ratio":
                        signals = self._mean_reversion_ratio()
                    elif strategy == "ols":
                        signals = self._ols()
                    elif strategy == "ts_momentum_signals":
                        signals = self._ts_momentum_signals()

                    if signals is None:
                        success = False
                        print("Trying another pair...")
                        break
                    else:
                        strategy_signals[strategy] = signals

                if success:
                    break
                else:
                    if ticker_count_2 != len(tickers)-1:
                        ticker_count_2 += 1
                        if ticker_count_2 == ticker_count_1: ticker_count_2 += 1
                    else:
                        ticker_count_1 += 1
                        ticker_count_2 = 0

            # === Execute Strategy ===
            strategy_pnl = {}
            for strategy, signal in strategy_signals.items():
                if enhancement==True:
                    pnl = self._execute_strategy_with_enhancement(strategy, signal)
                else:
                    if strategy == "ts_momentum_signals":
                        pnl = self._execute_strategy_momentum(strategy, signal)
                    else:
                        pnl = self._execute_strategy(strategy, signal)
                strategy_pnl[strategy] = pnl

            # === Plot Results ===
            self._plot_returns(strategy_pnl)
            self._plot_zscore(strategy_signals)

        else:
            self.stock_a, self.stock_b = self.fetch_data(ticker_a, ticker_b, start_date, end_date)


    def _fetch_data(self, ticker_A, ticker_B, start_date, end_date, notVol:bool=True):
        '''
        Fetch data from Yahoo Finance API
        1) Find a suitable pair of stocks from the FTSE100 for pair trading. Motivate your approach
        IMPORTANT: Mention to truly eval you would have to test it under many different market conditions and stocks.
        '''

        data = yf.download([ticker_A, ticker_B], start=start_date, end=end_date)
        data = data.dropna()

        stock_A  = data['Close', ticker_A]
        stock_B = data['Close', ticker_B]

        if notVol:
            self.dates = stock_A.index.strftime('%Y-%m-%d').tolist()

        return np.array(stock_A), np.array(stock_B)


    def _execute_strategy(self, strategy, z_score):
        '''
        Execute the selected strategy.
        Plot and analyze the PnL including the ratio between them.
        '''
        entry_threshold = 1.0
        long_signal = z_score < -entry_threshold  # long A, short B
        short_signal = z_score > entry_threshold  # short A, long B

        prices_A = self.stock_a
        prices_B = self.stock_b

        position = None
        entry_A = entry_B = 0
        pnl = [0] * len(z_score)

        capital = self.capital
        capital_A = capital_B = capital / 2  # 50% allocation to each

        qty_A = qty_B = 0  # Number of shares (fractional)

        for t in range(len(z_score)):
            if position is None:
                if long_signal[t]:
                    position = 'long'
                    entry_A = prices_A[t]
                    entry_B = prices_B[t]
                    qty_A = capital_A / entry_A   # Buy A
                    qty_B = capital_B / entry_B   # Short B

                elif short_signal[t]:
                    position = 'short'
                    entry_A = prices_A[t]
                    entry_B = prices_B[t]
                    qty_A = capital_A / entry_A   # Short A
                    qty_B = capital_B / entry_B   # Buy B

            elif position == 'long':
                if t == (len(z_score)-1):
                    # Close position as trading period has ended.
                    profit = (prices_A[t] - entry_A) * qty_A + (entry_B - prices_B[t]) * qty_B
                    pnl[t] = profit
                    capital += profit
                    capital_A = capital_B = capital / 2
                    position = None

                else:
                    if z_score[t] >=0: # Could set to other buy position, but this Pairs Trading algo works on assumption of mean-reversion, not that it will go up, revert, down (could go up, revert, up, revert).
                        # Close long A, short B
                        profit = (prices_A[t] - entry_A) * qty_A + (entry_B - prices_B[t]) * qty_B
                        pnl[t] = profit
                        capital += profit
                        capital_A = capital_B = capital / 2
                        position = None

                    if short_signal[t]:
                        # Open short A, long B
                        position = 'short'
                        entry_A = prices_A[t]
                        entry_B = prices_B[t]
                        qty_A = capital_A / entry_A   # Short A
                        qty_B = capital_B / entry_B   # Buy B
    
            elif position == 'short':
                if t == (len(z_score)-1):
                    # Close position as trading period has ended
                    profit = (entry_A - prices_A[t]) * qty_A + (prices_B[t] - entry_B) * qty_B
                    pnl[t] = profit
                    capital += profit
                    capital_A = capital_B = capital / 2
                    position = None

                else:
                    if z_score[t] <= 0:
                        # Close short A, long B
                        profit = (entry_A - prices_A[t]) * qty_A + (prices_B[t] - entry_B) * qty_B
                        pnl[t] = profit
                        capital += profit
                        capital_A = capital_B = capital / 2
                        position = None

                    if long_signal[t]:
                        # Open long A, short B
                        position = 'long'
                        entry_A = prices_A[t]
                        entry_B = prices_B[t]
                        qty_A = capital_A / entry_A   # Buy A
                        qty_B = capital_B / entry_B   # Short B

        # Cumulative PnL over time
        cumulative_pnl = np.cumsum(pnl)
        print(f"{strategy} PnL: {cumulative_pnl[-1]:.2f} GBX")

        return cumulative_pnl

    def _execute_strategy_momentum(self, strategy, signals):
        prices_A = self.stock_a
        prices_B = self.stock_b

        entry_A = entry_B = 0
        pnl = [0] * len(signals)

        capital = self.capital
        capital_A = capital_B = capital / 2  # 50% allocation to each

        qty_A = qty_B = 0  # Number of shares (fractional)

        pos_a = pos_b = 0

        for t in range(len(signals)):
            if signals[t] == 2:
                # open long a
                entry_A = prices_A[t]
                qty_A = capital_A / entry_A
                pos_a = 1
                pnl[t] += 0

                # open short b
                entry_B = prices_B[t]
                qty_B = capital_B / entry_B
                pos_b = -1
                pnl[t] += 0
            elif signals[t] == 3:
                # open short a
                entry_A = prices_A[t]
                qty_A = capital_A / entry_A
                pos_a = -1
                pnl[t] += 0

                # open long b
                entry_B = prices_B[t]
                qty_B = capital_B / entry_B
                pos_b = 1
                pnl[t] += 0
            elif signals[t] == 1:
                if pos_a == 1:
                  # close long a
                  profit = (prices_A[t] - entry_A) * qty_A
                  pnl[t] += profit
                  capital_A += profit
                  pos_a = 0

                  # close short b
                  profit = (entry_B - prices_B[t]) * qty_B
                  pnl[t] += profit
                  capital_B += profit
                  pos_b = 0
                elif pos_a == -1:
                  # close short a
                  profit = (entry_A - prices_A[t]) * qty_A
                  pnl[t] += profit
                  capital_A += profit
                  pos_a = 0

                  # close long b
                  profit = (prices_B[t] - entry_B) * qty_B
                  pnl[t] += profit
                  capital_B += profit
                  pos_b = 0
            else: pnl[t] += 0

        cumulative_pnl = np.cumsum(pnl)
        print(f"{strategy} PnL: {cumulative_pnl[-1]:.2f} GBX")

        return cumulative_pnl

    # === Plotting ===
    def _plot_returns(self, strat_pnl_dict):
        if not os.path.exists("algo"):
            os.makedirs("algo")

        plt.figure(figsize=(12, 5))
        for strategy, cum_pnl in strat_pnl_dict.items():
            plt.plot(self.dates, cum_pnl/self.capital, label=f"{strategy} ({cum_pnl[-1]:.2f} GBP | {round(cum_pnl[-1]/10000)}%)")

        plt.xlabel("Days")
        plt.xticks([])
        plt.ylabel("Returns %")
        plt.title(f"Cumulative PnL of Strategies for {self.ticker_a} and {self.ticker_b} ({self.start_date} to {self.end_date})")
        plt.legend()
        plt.grid(True)
        plt.savefig("algo/pnl.png")


    def _plot_zscore(self, strategy_zscore):
        if not os.path.exists("algo"):
            os.makedirs("algo")

        plt.figure(figsize=(12, 5))
        for strategy, zscore in strategy_zscore.items():
            if strategy == "ts_momentum_signals":
                continue
            else:
                plt.plot(self.dates, zscore, label=f"{strategy}")

        # green dotted buy-zone lines (z = +1 and z = –1)
        plt.axhline( 1, color="green", linestyle="--", linewidth=1, label="Buy zone (+1)")
        plt.axhline(-1, color="green", linestyle="--", linewidth=1)

        # red dotted sell line (z = 0)
        plt.axhline( 0, color="red",   linestyle="--", linewidth=1, label="Sell line (0)")

        plt.xlabel("Days")
        plt.xticks([])
        plt.ylabel("Z score")
        plt.title(f"Strategies Z scores for {self.ticker_a} and {self.ticker_b} ({self.start_date} to {self.end_date})")
        plt.legend()
        plt.grid(True)
        plt.savefig("algo/zscores.png")


    # Strategy 1: Mean-reversion with spread
    def _mean_reversion_spread(self):
        '''
        Mean-reversion strategy using price spread.
        1) Calculate correlation between two stocks via Pearson
        2) Calculate the spread between the two stocks
        3) Check for stationarity using ADF-like test
        4) Generate z-score of the spread
        '''

        def compute_spread(A, B):
            norm_A = (A - np.mean(A)) / np.std(A)
            norm_B = (B - np.mean(B)) / np.std(B)
            return norm_A - norm_B

        def generate_z_score(spread):
            mean = np.mean(spread)
            std = np.std(spread)
            return (spread - mean) / std

        correlation = np.corrcoef(self.stock_a, self.stock_b)[0, 1]
        if correlation < 0.8:
            print(f"Mean reversion spread: {self.ticker_a}, {self.ticker_b} not correlated enough (corr={correlation:.2f}).")
            return None

        spread = compute_spread(self.stock_a, self.stock_b)

        adf_result = adfuller(spread)
        adf_stat = adf_result[0]
        p_value = adf_result[1]

        if p_value > 0.05:
            print(f"Mean reversion spread: ADF test failed (ADF stat = {adf_stat:.4f}, p = {p_value:.4f}).")
            return None

        z_score = generate_z_score(spread)
        return z_score


    # Strategy 2: Mean-reversion with ratio
    def _mean_reversion_ratio(self):
        '''
        Mean-reversion strategy using price ratio.
        Best for when prices are at different scales. Assumes price porpotional remains the same.
        1) Calculate correlation between two stocks via Pearson
        2) Calculate the ratio between the two stocks
        3) Check for stationarity using ADF-like test
        4) Generate z-score of the ratio
        '''

        def generate_z_score(ratio):
            mean = np.mean(ratio)
            std = np.std(ratio)
            return (ratio - mean) / std
        
        # Calculate correlation between the two stocks and check if it's above a threshold
        correlation = np.corrcoef(self.stock_a, self.stock_b)[0, 1]
        if correlation < 0.8:
            print(f"Mean reversion ratio: {self.ticker_a}, {self.ticker_b} not correlated enough (corr={correlation:.2f}).")
            return None

        ratio = self.stock_a / self.stock_b

        # Check for stationarity using ADF test
        adf_result = adfuller(ratio)
        adf_stat = adf_result[0]
        p_value = adf_result[1]

        if p_value > 0.05:
            print(f"Mean reversion ratio: ADF test failed (ADF stat = {adf_stat:.4f}, p = {p_value:.4f}).")
            return None

        z_score = generate_z_score(ratio)
        return z_score

    # Strategy 3: OLS Regression (Hedge Ratio)
    def _ols(self):
        """
        Calculate hedge ratio using OLS regression, compute spread, and return z-score of the spread.

        Parameters:
        - series_y (pd.Series): Dependent variable (e.g., stock A)
        - series_x (pd.Series): Independent variable (e.g., stock B)

        Returns:
        - zscore_series (pd.Series): Z-score of the spread
        """
        # Ensure series are aligned
        series_y = A = np.array(self.stock_a)
        series_x =B = np.array(self.stock_b)

        # Add constant for intercept
        X = add_constant(series_x)
        model = OLS(series_y, X).fit()

        # Hedge ratio is the slope
        hedge_ratio = model.params[1]

        # Calculate spread
        spread = series_y - hedge_ratio * series_x

        # Calculate z-score of spread
        mean_spread = spread.mean()
        std_spread = spread.std()
        zscore = (spread - mean_spread) / std_spread

        return zscore

    def _ts_optimize(self, end_date, short_window_range=[1,2,3,4,5,6,7,8,9,10], long_window_range=[20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]):
        best_score = -np.inf
        best_short = None
        best_long = None

        start_date = (datetime.strptime(end_date, "%Y-%m-%d") - relativedelta(months=12)).strftime("%Y-%m-%d")
        historical_prices = yf.download([self.ticker_a, self.ticker_b], start=start_date, end=end_date)['Close']
        historical_prices = historical_prices.dropna()

        stock_a_historical = historical_prices[self.ticker_a]
        stock_b_historical = historical_prices[self.ticker_b]

        # Calculate the ratio
        historical_ratio = stock_a_historical / stock_b_historical

        # Calculate returns of the ratio (this is your "TSMOM target")
        ratio_returns = historical_ratio.pct_change().fillna(0)

        # Grid search over windows
        for short_w in short_window_range:
            for long_w in long_window_range:
                if short_w >= long_w:
                    continue  # Invalid combination

                sma = historical_ratio.rolling(window=short_w).mean()
                lma = historical_ratio.rolling(window=long_w).mean()

                # Generate signals on the ratio
                signal = np.where(sma > lma, 1, -1)
                signal = pd.Series(signal, index=historical_ratio.index).shift(1).fillna(0)

                # Strategy returns on ratio returns
                strategy_returns = signal * ratio_returns

                # Evaluate using cumulative return
                cumulative_return = (1 + strategy_returns).prod() - 1

                if cumulative_return > best_score:
                    best_score = cumulative_return
                    best_short = short_w
                    best_long = long_w

        return best_short, best_long

    # Strategy 4: Time Series Momentum on Price Ratio
    def _ts_momentum_signals(self, short_window=None, long_window=None):

        data = np.array(self.stock_a) / np.array(self.stock_b)

        if short_window==None:
            # Optimize on past 1 year before self.start_date
            end_date = self.start_date
            short_window, long_window = self._ts_optimize(end_date=end_date)
            print(f"Found optimal short and long windows: ({short_window}, {long_window})")

        # Calculate moving averages on evaluation data only
        prices_series = pd.Series(data)
        sma_array = prices_series.rolling(window=short_window).mean().values
        lma_array = prices_series.rolling(window=long_window).mean().values

        action_array = np.zeros(len(data), dtype=int)
        current_position = 0

        for i in range(long_window, len(data)):
            sma = sma_array[i]
            lma = lma_array[i]

            if np.isnan(sma) or np.isnan(lma):
                continue

            if sma > lma:
                if current_position == 0:
                    action_array[i] = 3  # Open long / short
                    current_position = 1
                elif current_position == -1:
                    action_array[i] = 1  # Close short
                    current_position = 0
            elif sma < lma:
                if current_position == 0:
                    action_array[i] = 2  # Open short / long
                    current_position = -1
                elif current_position == 1:
                    action_array[i] = 1  # Close long
                    current_position = 0

        return action_array

# === Test the strategies ===
strategies = ["mean_reversion_spread", "mean_reversion_ratio", "ols", 'ts_momentum_signals']
PairsTrading(capital=1000000, enhancement=False, strategies=strategies)
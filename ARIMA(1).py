import os
import random
import numpy as np
import yfinance as yf
from tqdm import tqdm
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

random.seed(7)

# ================
# == Exercise 1 ==
# ================

class ARIMA():
    def __init__(self, p, d, q, phi_vals, theta_vals):
        self.p = p
        self.d = d
        self.q = q

        self.phi = phi_vals
        self.theta = theta_vals

    def difference(self, data):
        ''' Apply d-order differencing '''
        diffed = np.array(data)
        for _ in range(self.d):
            diffed = np.diff(diffed, prepend=diffed[0])
        return diffed

    def run(self, data):
        y = np.array(data)
        dy = self.difference(y)
        n = len(dy)

        dy_pred = np.zeros(n + 1)
        error = np.zeros(n + 1)

        # Main ARIMA loop (AR + MA terms)
        for t in range(max(self.p, self.q), n+1):
            ar_part = sum(self.phi[i] * dy[t - i - 1] for i in range(self.p))
            ma_part = sum(self.theta[i] * error[t - i - 1] for i in range(self.q))
            dy_pred[t] = ar_part + ma_part

            if t != n:
                error[t] = dy[t] - dy_pred[t]

        # Reverse differencing to get the predicted y_t+1
        prediction = y[-1] + dy_pred[-1]

        return prediction, dy[-1], dy_pred[-1], y[-1]

    def run_further(self, prev_pred, dy, dy_pred, y, past_dy, past_errors):
        # Update history
        past_dy = np.append(past_dy[-(self.p - 1):], np.diff([y, prev_pred])[0])
        past_errors = np.append(past_errors[-(self.q - 1):], np.diff([y, prev_pred])[0] - dy_pred)

        # Predict next dy
        ar_part = sum(self.phi[i] * past_dy[-i - 1] for i in range(self.p))
        ma_part = sum(self.theta[i] * past_errors[-i - 1] for i in range(self.q))
        dy_pred = ar_part + ma_part

        prediction = prev_pred + dy_pred

        return prediction, past_dy[-1], dy_pred, prev_pred, past_dy, past_errors

    def run_multiple(self, num_forecasts, data):
        predictions = []
        prev_pred, dy, dy_pred, y = self.run(data)

        # Initialize past_dy and past_errors for recursive forecasting
        dy_series = self.difference(data)
        past_dy = dy_series[-self.p:] if len(dy_series) >= self.p else np.pad(dy_series, (self.p - len(dy_series), 0), 'constant')
        past_errors = np.zeros(self.q)  # Start with zero errors

        predictions.append(prev_pred)
        for _ in range(num_forecasts-1):
            prev_pred, dy, dy_pred, y, past_dy, past_errors = self.run_further(prev_pred, dy, dy_pred, y, past_dy, past_errors)
            predictions.append(prev_pred)
        return predictions


# == Testing == 
y = np.array([15,10,12,17,25,23])
arima_model = ARIMA(p=2, d=1, q=1, phi_vals=[7,4], theta_vals=[1]) # Used Antonio student ID: 33174741
prediction = arima_model.run_multiple(2, y)
print("Predictions: ", prediction)

# ================
# == Exercise 2 ==
# ================

class TrainARIMA():
    def __init__(self, ticker, start_date, end_date):
        # Fetch data from Yahoo Finance
        data = yf.download(ticker, start=start_date, end=end_date)
        data = data.dropna()
        data  = data['Close', ticker]
        self.data = data
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date

        # Split data into train and test (do not randomize)
        train_size = int(len(data) * 0.67)
        self.train_data = data[:train_size]
        self.test_data = data[train_size:]
        self.train_size = train_size

        # Initialize model
        self.p,self.d,self.q = self._find_optimal_hyperparams()
        self.theta = None
        self.phi = None
        self.model = None

    def _find_optimal_hyperparams(self):
        data = self.train_data.values

        # Step 1: Determine d using ADF test
        d = 0
        p_value = adfuller(data)[1]
        differenced_data = data

        while p_value > 0.05:
            differenced_data = np.diff(differenced_data, n=1)
            d += 1
            p_value = adfuller(differenced_data)[1]

        print(f"Optimal d found: {d}")

        # Step 2: Use ACF and PACF to determine p and q
        fig, axes = plt.subplots(1, 2, figsize=(15, 4))
        plot_pacf(differenced_data, ax=axes[0], lags=20, method='ywm')
        axes[0].set_title('PACF Plot')
        plot_acf(differenced_data, ax=axes[1], lags=20)
        axes[1].set_title('ACF Plot')
        plt.savefig("pacf_acf.png")

        pacf_vals = pacf(differenced_data, nlags=20)
        acf_vals = acf(differenced_data, nlags=20)

        # Define threshold as approx. 95% confidence level
        threshold = 1.96 / np.sqrt(len(differenced_data))

        # Find where PACF crosses threshold
        p = np.where(np.abs(pacf_vals) > threshold)[0]
        p = p[1] if len(p) > 1 else 1  # exclude lag 0

        # Find where ACF crosses threshold
        q = np.where(np.abs(acf_vals) > threshold)[0]
        q = q[1] if len(q) > 1 else 1  # exclude lag 0

        print(f"Optimal p found: {p}")
        print(f"Optimal q found: {q}")

        self.p = 2
        self.d = 1
        self.q = 4

        return p, d, q

    def train(self, iterations=1000, learning_rate=0.01):
        train_data = self.train_data.values
        n = len(train_data)

        # Initialize randomly
        phi_vals = np.random.uniform(-0.5, 0.5, self.p)
        theta_vals = np.random.uniform(-0.5, 0.5, self.q)

        best_phi = phi_vals.copy()
        best_theta = theta_vals.copy()
        best_loss = np.inf

        # Optimization loop
        for i in tqdm(range(iterations)):
            model = ARIMA(p=self.p, d=self.d, q=self.q, phi_vals=phi_vals, theta_vals=theta_vals)

            # Rolling window forecast (1 step ahead)
            losses = []
            for window in range(max(self.p, self.q), n - 1):
                train_window = train_data[:window]
                true_value = train_data[window]

                prediction, _, _, _ = model.run(train_window)
                losses.append((prediction - true_value) ** 2) # regularizer (b) not used as GD not being use to loss insn't used for param update

            loss = np.mean(losses) # We are optimizing use MSE instead of SSE for hill even though it is the same practically its just custom preference

            # Hill climbing: try random small perturbation
            new_phi = phi_vals + np.random.normal(0, learning_rate, self.p)
            new_theta = theta_vals + np.random.normal(0, learning_rate, self.q)

            model_new = ARIMA(p=self.p, d=self.d, q=self.q, phi_vals=new_phi, theta_vals=new_theta)

            # Recalculate loss with new parameters
            new_losses = []
            for window in range(max(self.p, self.q), n - 1):
                train_window = train_data[:window]
                true_value = train_data[window]

                prediction, _, _, _ = model_new.run(train_window)
                new_losses.append((prediction - true_value) ** 2)

            new_loss = np.mean(new_losses)

            # Accept new parameters if they improved the loss
            if new_loss < loss:
                phi_vals = new_phi
                theta_vals = new_theta
                if new_loss < best_loss:
                    best_loss = new_loss
                    best_phi = new_phi
                    best_theta = new_theta

        print(f"Best phi: {best_phi}")
        print(f"Best theta: {best_theta}")
        self.phi = best_phi
        self.theta = best_theta
        self.model = ARIMA(p=self.p, d=self.d, q=self.q, phi_vals=best_phi, theta_vals=best_theta)
        return best_phi, best_theta

    def evaluate(self):
        arima_predictions = []
        for day_count in range(len(self.test_data)):
            limit = self.train_size + day_count
            train_data = np.array(self.data[:limit])
            prediction,_,_,_ = self.model.run(train_data)
            arima_predictions.append(prediction)

        test = np.array(self.test_data)
        avg_price = np.mean(test)

        arima_predictions = np.array(arima_predictions)
        arima_rse = ((arima_predictions - test) ** 2)
        arima_rmse = (np.mean(arima_rse))**0.5
        arima_mae = np.mean(np.abs(arima_predictions - test))
        print("Evaluation RMSE: ", arima_rmse)
        print("Evaluation MAE: ", arima_mae)
        print("Average Price: ", avg_price)
        # Using MAE because it gives the monetary amount by which it is off, which is relevant for the specific stock
        # RMSE and MAE is good to compare different models, but not different assets since different assets have different price scales
        # Continuing: a MAE of 2 is worse for an average price of 10 vs 200, hence MAPE would be better and it ratio's it to actual price.

        # --- Plotting ---
        if not os.path.exists("arima"):
            os.makedirs("arima")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Figure 1: Real vs Predicted Price
        ax1.plot(test, label=f'Actual Price', marker='o')
        ax1.plot(arima_predictions, label=f'ARIMA({self.p},{self.d},{self.q}) (RMSE: {arima_rmse:.2f}, MAE: {arima_mae:.2f}) | Avg Price: {avg_price:.2f}', marker='x')
        ax1.set_xlabel('Days ahead')
        ax1.set_ylabel('Price')
        ax1.set_title(f'Predicted vs Actual Price for {self.ticker}')
        ax1.legend()

        # Figure 2: RSE per Horizon
        # Using RSE instead of SE since RSE gives actual value by which it is off. Can't be mean or sum because we aren plotting each day forecast's error.
        # We are comparing performance on two different stocks so MSE is ideal for that.
        ax2.plot(arima_rse, label=f'ARIMA({self.p},{self.d},{self.q}) RSE (Avg price: {avg_price:.2f})', marker='d')
        ax2.set_xlabel('Days ahead')
        ax2.set_ylabel('RSE')
        ax2.set_title(f'RSE over Forecast Horizons for {self.ticker}')
        ax2.legend()

        # Adjust layout and save the combined figure
        plt.tight_layout()
        plt.savefig(f"arima/{self.ticker}_{self.start_date}_{self.end_date}_RMSE_Prices.png")
        plt.close()

# == training == 
model = TrainARIMA("WFC", start_date="2023-01-01", end_date="2024-01-01")
model.train()
model.evaluate()

# == Testing multiple tickers ==
tickers = ["AZN.L", "HSBA.L", "BP.L"]
for ticker in tickers:
    print(f"\n== Ticker: {ticker} ==\n")
    model = TrainARIMA(ticker, start_date="2023-01-01", end_date="2024-01-01")
    model.train()
    model.evaluate()

# Training one model per stock, as ARIMA is a univariate time series model, meaning it will model only one time series at a time, and each stock it is own time series. Makes no sense to predict BP price at t=5 based on AZN price at t=4. 

# Optimization method: hill climbing. Chosen due to simplicity. Gradient descent implementation was exploding (likely due to lack of normalization). Of course evaluate this option by mentioning that hill climbing fits for local optima so not ideal, but works for proof of concept. 

# As optimizer is hill climbing and not gradient descent, the model is not guaranteed to converge to a global minimum. Also, the data is not normalized as it is not used to make the param updates. If we were to use GD then normalization would be needed to avoid exploding gradients and other problems. 

# No training done for more than 1 horizon ahead. Training is done on subsets of the training data for one horizon ahead, starting from the first 10% of the training data. If only used the whole training dataset and last value for training it would overfit to that one value. 
# Testing done by predicting each point in the test set recursively. This means that for each point in the test set, we use all previous points to predict the next point. This is a common approach in time series forecasting, as it allows us to use all available information to make predictions.
# Reason for above is because (as lecture indicated) if we want to predict further horizons, we predict each horizon by running ARIMA recursively. Hence we want to optimize ARIMA hyperparams for 1 horizon ahead, but test its performance for multiple horizons.

# The model trains on the full historical data. This means that if n values are inputted, the model will use n-1 values for ARIMA modelling, and predict the nth value. Based on that nth value prediction error, the model will update the ARIMA hyperparameters. This means there is one prediction per epoch, and at each epoch the hyperparams are updated if the error decreases (hill climbing optimization). The reason for not using a window (prediction for t=4 using 0:t-3, etc.) is because ARIMA uses all the historical data, so we want to optimize the hyperparams based on the full historical data.

# RMSE is scale-dependent, so average price is presented to give context.

# Doubt: Unsure if the metrics chosen are the "most appropriate". Make sure you justify the choice by explaining what they mean in this context.
# Doubt: Update the ARIMA parameters. 

# Part 1: Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error
import math

# Part 2: Data Download
from statsmodels.datasets import get_rdataset
air = get_rdataset('AirPassengers', 'datasets')
data = air.data

# Convert to time series data
data['time'] = pd.date_range(start='1949-01-01', periods=len(data), freq='M')
data = data.set_index('time')
data.columns = ['passengers']
ts = data['passengers']

# Part 3: Plot Data
plt.figure(figsize=(12, 6))
plt.plot(ts)
plt.title('Monthly Air Passengers 1949-1960')
plt.xlabel('Date')
plt.ylabel('Number of Passengers')
plt.grid(True)
plt.show()

# Plot seasonality
plt.figure(figsize=(12, 8))
plt.subplot(211)
plt.title('Seasonal Plot')
for year in range(1949, 1961):
    if year == 1960:
        year_data = ts[f'{year}']
    else:
        year_data = ts[f'{year}']
    plt.plot(range(1, len(year_data) + 1), year_data, label=str(year))
plt.legend(loc='upper left')

# ACF and PACF plots
plt.figure(figsize=(12, 8))
plt.subplot(211)
plot_acf(ts, ax=plt.gca(), lags=40)
plt.subplot(212)
plot_pacf(ts, ax=plt.gca(), lags=40)
plt.tight_layout()
plt.show()

# Part 4: Data Split
# Split data into train and test sets (80% train, 20% test)
train_size = int(len(ts) * 0.8)
train, test = ts[:train_size], ts[train_size:]
print(f'Training data size: {len(train)}')
print(f'Testing data size: {len(test)}')

# Function to evaluate forecasts
def evaluate_forecast(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    rmse = math.sqrt(mse)
    print(f'Mean Squared Error: {mse:.3f}')
    print(f'Root Mean Squared Error: {rmse:.3f}')
   
    # Plot actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.plot(actual.index, actual, label='Actual')
    plt.plot(actual.index, predicted, color='red', label='Predicted')
    plt.title('Actual vs Predicted')
    plt.xlabel('Date')
    plt.ylabel('Number of Passengers')
    plt.legend()
    plt.grid(True)
    plt.show()

# Part 5: AR (AutoRegressive) Model
def fit_ar_model(lag=12):
    print("\n--- AR Model ---")
    # Fit AR model
    model = AutoReg(train, lags=lag)
    model_fit = model.fit()
    print(model_fit.summary())
    predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
    print("\nAR Model Evaluation:")
    evaluate_forecast(test, predictions)
   
    return model_fit, predictions

ar_model, ar_predictions = fit_ar_model(lag=13)

# Part 6: MA (Moving Average) Model
# MA models can be implemented through ARIMA(0,0,q)
def fit_ma_model(q=12):
    print("\n--- MA Model ---")
    # Fit MA model (using ARIMA with p=0, d=0)
    model = ARIMA(train, order=(0, 0, q))
    model_fit = model.fit()
    print(model_fit.summary())
    predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
    print("\nMA Model Evaluation:")
    evaluate_forecast(test, predictions)
    return model_fit, predictions

ma_model, ma_predictions = fit_ma_model(q=12)

# Part 7: ARIMA Model
def fit_arima_model(p=1, d=1, q=1):
    print("\n--- ARIMA Model ---")
    # Fit ARIMA model
    model = ARIMA(train, order=(p, d, q))
    model_fit = model.fit()
    print(model_fit.summary())
    predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
    print("\nARIMA Model Evaluation:")
    evaluate_forecast(test, predictions)
    return model_fit, predictions

arima_model, arima_predictions = fit_arima_model(p=2, d=1, q=2)

# Part 8: SARIMA Model
def fit_sarima_model(p=1, d=1, q=1, P=1, D=1, Q=1, s=12):
    print("\n--- SARIMA Model ---")
    # Fit SARIMA model
    model = SARIMAX(train, order=(p, d, q), seasonal_order=(P, D, Q, s))
    model_fit = model.fit(disp=False)
    print(model_fit.summary())
    predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
    print("\nSARIMA Model Evaluation:")
    evaluate_forecast(test, predictions)
    return model_fit, predictions

sarima_model, sarima_predictions = fit_sarima_model(p=1, d=1, q=1, P=1, D=1, Q=1, s=12)

# Compare all models
plt.figure(figsize=(12, 8))
plt.plot(test.index, test, label='Actual', color='black')
plt.plot(test.index, ar_predictions, label='AR', color='blue')
plt.plot(test.index, ma_predictions, label='MA', color='green')
plt.plot(test.index, arima_predictions, label='ARIMA', color='red')
plt.plot(test.index, sarima_predictions, label='SARIMA', color='purple')
plt.title('Model Comparison')
plt.xlabel('Date')
plt.ylabel('Number of Passengers')
plt.legend()
plt.grid(True)
plt.show()
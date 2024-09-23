import pandas as pd
import numpy as np
from pmdarima import auto_arima
from prophet import Prophet
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error

def time_series_models_comparison():
    random.seed(123)
    data_set = pd.read_csv("./data/temperatures.csv")
    # do not remove the lines above and write your solution here
    data_set['ds'] = pd.to_datetime(data_set['ds'])
    data_set = data_set.set_index('ds')
    date_range = pd.date_range(start='1980-01-01', end = '2012-12-01', freq = 'MS')
    data_set = data_set.reindex(date_range)
    data_set['y'] = data_set['y'].ffill()

    train = data_set[:-24]
    test = data_set[-24:]

    arima_model = pmdarima.auto_arima(train['y'])
    arima_forecast = arima_model.predict(n_periods =24)

    prophet_df = train.reset_index().rename(columns={'index': 'ds'})
    prophet_model = Prophet()
    prophet_model.fit(prophet_df)
    future = prophet_model.make_future_dataframe(periods=24, freq='MS')
    prophet_forecast_df = prophet_model.predict(future)
    prophet_forecast = prophet_forecast_df['yhat'][-24:].values

    ex_smooth_model = ExponentialSmoothing(train['y'], seasonal_periods=12).fit()
    ex_smooth_forecast = ex_smooth_model.forecast(steps=24)

    forecasts = pd.DataFrame({
    'ds': test.index,
    'arima': arima_forecast,
    'prophet': prophet_forecast,
    'ex_smooth': ex_smooth_forecast
    }).set_index('ds').round(3)
    print(forecasts)

    arima_mse = mean_squared_error(test['y'], arima_forecast)
    prophet_mse = mean_squared_error(test['y'], prophet_forecast)
    ex_smooth_mse = mean_squared_error(test['y'], ex_smooth_forecast)

    errors = pd.DataFrame({
    'model': ['arima', 'prophet', 'ex_smooth'],
    'mse': [arima_mse, prophet_mse, ex_smooth_mse]
    })
    return {
    'trainset': train,
    'testset': test,
    'forecasts': forecasts,
    'errors': errors
    }

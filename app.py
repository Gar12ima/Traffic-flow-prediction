import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import warnings

warnings.filterwarnings("ignore")


# 1. Load data first
@st.cache_data
def load_data():
    df = pd.read_csv("traffic.csv")
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df.set_index('DateTime', inplace=True)
    data = df[df['Junction'] == 1]['Vehicles']
    data = data.resample('H').sum()
    return data

data = load_data()
future_dates = pd.date_range(data.index[-1] + pd.Timedelta(hours=1), periods=24, freq='H')

# 2. UI setup
st.title("🚦 Traffic Flow Forecasting")
st.markdown("Forecast traffic for the next 24 hours using ARIMA, LSTM or Random Forest.")

model_option = st.selectbox("Choose Forecasting Model", ["ARIMA", "LSTM", "Random Forest"])
run = st.button("Run Forecast")

# 3. Forecast functions
def forecast_arima(data):
    model = ARIMA(data, order=(2, 1, 2))
    model_fit = model.fit()
    return model_fit.forecast(steps=24)

def forecast_lstm(data):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data.values.reshape(-1, 1))

    def create_sequences(data, seq_len):
        X, y = [], []
        for i in range(len(data) - seq_len):
            X.append(data[i:i+seq_len])
            y.append(data[i+seq_len])
        return np.array(X), np.array(y)

    X, y = create_sequences(scaled, 24)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=(24, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=3, batch_size=32, verbose=0)

    current_seq = scaled[-24:].reshape((1, 24, 1))
    preds = []
    for _ in range(24):
        pred = model.predict(current_seq)[0][0]
        preds.append(pred)
        current_seq = np.append(current_seq[:, 1:, :], [[[pred]]], axis=1)

    return scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()

def forecast_rf(data):
    def create_lag_features(data, lags):
        df_lag = pd.DataFrame()
        for lag in range(1, lags + 1):
            df_lag[f'lag_{lag}'] = data.shift(lag)
        df_lag['target'] = data.values
        df_lag.dropna(inplace=True)
        return df_lag

    df_lagged = create_lag_features(data, 24)
    X = df_lagged.drop('target', axis=1)
    y = df_lagged['target']

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    last_vals = data.values[-24:]
    preds = []
    current = last_vals.copy()
    for _ in range(24):
        input_df = pd.DataFrame([current[::-1]], columns=[f'lag_{i}' for i in range(1, 25)])
        pred = model.predict(input_df)[0]
        preds.append(pred)
        current = np.append(current[1:], pred)
    return preds

# 4. Run model only after button click
if run:
    with st.spinner("Forecasting... Please wait."):
        if model_option == "ARIMA":
            forecast = forecast_arima(data)
            label = "ARIMA Forecast"
            color = "orange"
        elif model_option == "LSTM":
            forecast = forecast_lstm(data)
            label = "LSTM Forecast"
            color = "red"
        else:
            forecast = forecast_rf(data)
            label = "Random Forest Forecast"
            color = "green"

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(data, label="Actual")
        ax.plot(future_dates, forecast, label=label, color=color)
        ax.set_title(label)
        ax.set_xlabel("Time")
        ax.set_ylabel("Traffic Volume")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        st.success("✅ Forecast complete!")

        forecast_df = pd.DataFrame({'DateTime': future_dates, 'Forecasted Traffic': forecast})
        st.dataframe(forecast_df)

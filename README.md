 🚦 Traffic Flow Forecasting App

This Streamlit application forecasts hourly traffic volume for the next 24 hours using time-series and machine learning models. Choose between **ARIMA**, **LSTM**, or **Random Forest** to generate accurate predictions based on historical data.

---

## 🔧 Features

* Upload and process traffic data from CSV
* Choose from 3 forecasting models: ARIMA, LSTM, or Random Forest
* Visualize actual vs predicted traffic data
* Hourly forecast for the next 24 hours
* Lightweight UI using Streamlit

---

## 🛠️ Requirements

* Python 3.8+
* Install dependencies via pip:

```bash
pip install streamlit pandas numpy matplotlib scikit-learn statsmodels tensorflow
```

---

## 🚀 Running the App

1. Ensure `traffic.csv` is in the root directory. It should include:
   - `DateTime`: timestamp column
   - `Vehicles`: numeric vehicle count
   - `Junction`: identifier for the traffic junction

2. Launch the Streamlit app:

```bash
streamlit run app.py
```

3. Interact with the app in your browser (default: `http://localhost:8501`):
   - Select model from the dropdown
   - Click "Run Forecast" to generate predictions and visual plots

---

## 📸 App Screenshots

### 🔹 Main Interface

![Screenshot 2025-06-03 111620](https://github.com/user-attachments/assets/83395bf7-c8a0-45eb-a86a-255798988963)

### 🔹 Forecast Output (ARIMA / LSTM / RF)

![Screenshot 2025-06-03 111104](https://github.com/user-attachments/assets/e244f83b-eb6c-4798-8c07-56aa48f265c9)
![Screenshot 2025-06-03 111131](https://github.com/user-attachments/assets/8ac185f7-b1b3-46b9-858f-b10dc2d88fc6)
![Screenshot 2025-06-03 111206](https://github.com/user-attachments/assets/d0c74a09-5579-4817-91c2-c94a3d1ef6c7)


> Add your screenshots to the `/screenshots/` folder.

---

## 📁 Folder Structure

```
├── app.py             # Main Streamlit app
├── traffic.csv        # Input dataset file
├── screenshots/       # Folder for storing UI screenshots
├── README.md          # This file
```

---

## 🙌 Credits

* Built with [Streamlit](https://streamlit.io/)
* Forecasting using [statsmodels](https://www.statsmodels.org/), [scikit-learn](https://scikit-learn.org/), [TensorFlow](https://www.tensorflow.org/)

---

## 📜 License

This project is licensed under the MIT License.

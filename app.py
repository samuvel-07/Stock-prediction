import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta

# -----------------------------
# CONFIG
# -----------------------------
SEQ_LEN = 60
PRED_DAYS = 7
FEATURES = ["Open", "High", "Low", "Close"]
MODEL_PATH = "lstm_multistep_ohlc.h5"
SCALER_PATH = "scaler_minmax.save"

# -----------------------------
# LOAD MODEL & SCALER
# -----------------------------
@st.cache_resource
def load_lstm_model():
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

model, scaler = load_lstm_model()

# -----------------------------
# APP UI
# -----------------------------
st.title("📈 Advanced LSTM Stock Predictor (7-Day Forecast)")
st.write("Predict the next 7 days of Open, High, Low, Close for any stock using a trained LSTM model.")

ticker = st.text_input("Enter Stock Symbol (e.g., AAPL, TSLA, RELIANCE.NS)", "AAPL")
start_date = st.date_input("Start Date", value=pd.to_datetime("2018-01-01"))
end_date = st.date_input("End Date", value=pd.to_datetime("today"))

if st.button("🔮 Predict Next 7 Days"):
    with st.spinner("Fetching data and predicting..."):
        # Fetch data
        df = yf.download(ticker, start=start_date, end=end_date)
        if df.empty:
            st.error("No data found. Try another ticker.")
            st.stop()

        data = df[FEATURES].dropna().reset_index()
        scaled_vals = scaler.transform(data[FEATURES].values)

        # Last 60 days for prediction
        if len(scaled_vals) < SEQ_LEN:
            st.error("Not enough data! Need at least 60 days.")
            st.stop()

        last_window = scaled_vals[-SEQ_LEN:, :].reshape(1, SEQ_LEN, len(FEATURES))

        # Predict
        next7_flat = model.predict(last_window)
        n_features = len(FEATURES)
        next7 = next7_flat.reshape((PRED_DAYS, n_features))
        next7_inv = scaler.inverse_transform(next7)

        # Create prediction DataFrame
        last_date = data["Date"].iloc[-1]
        next7_dates = pd.date_range(start=last_date + timedelta(days=1), periods=PRED_DAYS, freq="B")
        pred_df = pd.DataFrame(next7_inv, columns=FEATURES)
        pred_df.insert(0, "Date", next7_dates)

        st.subheader(f"📊 Predicted Prices for Next {PRED_DAYS} Days")
        st.dataframe(pred_df.style.format(precision=2))

        # Plot
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        axs = axs.flatten()
        for i, f in enumerate(FEATURES):
            axs[i].plot(data["Date"].iloc[-30:], data[f].iloc[-30:], label="Past 30 Days", color="blue")
            axs[i].plot(pred_df["Date"], pred_df[f], marker='o', label="Predicted", color="red")
            axs[i].set_title(f)
            axs[i].legend()
        plt.tight_layout()
        st.pyplot(fig)

        st.success("✅ Prediction complete!")

st.markdown("---")
st.caption("Created with ❤️ using Streamlit + TensorFlow LSTM")

"""
Advanced Multi-step Multivariate LSTM
Predict next 7 days of Open/High/Low/Close using past 60 days of OHLC.
"""

# Install required packages (uncomment if needed in Colab / notebook)
# !pip install yfinance tensorflow pandas numpy scikit-learn matplotlib

import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# -------------------------
# USER CONFIG
# -------------------------
TICKER = "AAPL"            # change to any ticker like "TSLA", "MSFT", "RELIANCE.NS"
START_DATE = "2018-01-01"
END_DATE = None            # None means till today
SEQ_LEN = 60               # look-back days
PRED_DAYS = 7              # predict next 7 days
FEATURES = ["Open", "High", "Low", "Close"]  # features to use & predict
BATCH_SIZE = 32
EPOCHS = 60
MODEL_SAVE_PATH = "lstm_multistep_ohlc.h5"
USE_CSV = False            # Set True if you want to read from a local CSV instead of yfinance
CSV_PATH = "AAPL.csv"      # if USE_CSV True, CSV must contain Date + OHLC columns

# -------------------------
# 1) LOAD DATA
# -------------------------
if USE_CSV:
    df = pd.read_csv(CSV_PATH, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
else:
    # yfinance
    yf_ticker = yf.Ticker(TICKER)
    if END_DATE:
        df = yf_ticker.history(start=START_DATE, end=END_DATE)
    else:
        df = yf_ticker.history(start=START_DATE)
    df.reset_index(inplace=True)

# Ensure required columns exist
for col in ["Open", "High", "Low", "Close"]:
    if col not in df.columns:
        raise ValueError(f"Column {col} not found in data")

data = df[["Date"] + FEATURES].copy()
data.dropna(inplace=True)
data = data.reset_index(drop=True)

print(f"Loaded {len(data)} rows from {data['Date'].iloc[0].date()} to {data['Date'].iloc[-1].date()}")

# -------------------------
# 2) SCALE DATA
# -------------------------
# We'll scale features to [0,1] using MinMaxScaler fitted on the whole dataset
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_vals = scaler.fit_transform(data[FEATURES].values)  # shape (n_samples, n_features)

# -------------------------
# 3) CREATE SEQUENCES
# -------------------------
def create_multistep_sequences(series, seq_len, pred_len):
    """
    series: numpy array shaped (n_samples, n_features)
    returns:
      X: (n_sequences, seq_len, n_features)
      y: (n_sequences, pred_len, n_features)
      dates_y: list of starting date indices for y (index in original dataframe for the first predicted day)
    """
    X, y, dates_y = [], [], []
    n_total = len(series)
    for i in range(seq_len, n_total - pred_len + 1):
        X.append(series[i - seq_len:i, :])
        y.append(series[i:i + pred_len, :])
        dates_y.append(i)  # index in original data that is the first day of the predicted block
    X = np.array(X)
    y = np.array(y)
    return X, y, dates_y

X, y, dates_y = create_multistep_sequences(scaled_vals, SEQ_LEN, PRED_DAYS)
print("X shape:", X.shape)   # (samples, seq_len, n_features)
print("y shape:", y.shape)   # (samples, pred_len, n_features)

# Flatten y targets to (samples, pred_len * n_features) for direct dense output
n_features = len(FEATURES)
y_flat = y.reshape((y.shape[0], PRED_DAYS * n_features))

# -------------------------
# 4) TRAIN/TEST SPLIT
# -------------------------
split_index = int(0.8 * len(X))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y_flat[:split_index], y_flat[split_index:]
dates_test = [data["Date"].iloc[idx] for idx in dates_y[split_index:]]  # starting dates for test predictions

print("Train samples:", X_train.shape[0], "Test samples:", X_test.shape[0])

# -------------------------
# 5) MODEL (Multistep Multi-feature output)
# -------------------------
tf.random.set_seed(42)
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(SEQ_LEN, n_features)),
    Dropout(0.2),
    LSTM(64, return_sequences=False),
    Dropout(0.15),
    Dense(64, activation="relu"),
    Dense(PRED_DAYS * n_features, activation="linear")  # outputs flattened predictions
])

model.compile(optimizer="adam", loss="mse")
model.summary()

# Callbacks
es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
mc = ModelCheckpoint(MODEL_SAVE_PATH, monitor="val_loss", save_best_only=True, verbose=1)

# -------------------------
# 6) TRAIN
# -------------------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[es, mc],
    verbose=1
)

# -------------------------
# 7) PREDICT on TEST
# -------------------------
y_pred_flat = model.predict(X_test)  # shape (samples, pred_len * n_features)

# Function to inverse transform flattened predictions back to original scale and shape
def invert_predictions(y_flattened, scaler, pred_len, n_features):
    # y_flattened: (samples, pred_len * n_features)
    samples = y_flattened.shape[0]
    # reshape to (samples * pred_len, n_features) so scaler.inverse_transform can be applied
    reshaped = y_flattened.reshape((samples * pred_len, n_features))
    inv = scaler.inverse_transform(reshaped)  # (samples * pred_len, n_features)
    # reshape back to (samples, pred_len, n_features)
    inv = inv.reshape((samples, pred_len, n_features))
    return inv

y_test_inv = invert_predictions(y_test, scaler, PRED_DAYS, n_features)
y_pred_inv = invert_predictions(y_pred_flat, scaler, PRED_DAYS, n_features)

# -------------------------
# 8) EVALUATION: RMSE per feature over all predicted days
# -------------------------
def rmse_per_feature(y_true3d, y_pred3d, feature_names):
    # y_true3d, y_pred3d: (samples, pred_len, n_features)
    rmses = {}
    for i, fname in enumerate(feature_names):
        # flatten across samples and time
        true_flat = y_true3d[:, :, i].reshape(-1)
        pred_flat = y_pred3d[:, :, i].reshape(-1)
        rmse = math.sqrt(mean_squared_error(true_flat, pred_flat))
        rmses[fname] = rmse
    return rmses

rmses = rmse_per_feature(y_test_inv, y_pred_inv, FEATURES)
print("\nRMSE per feature (over test set and all 7 predicted days):")
for k, v in rmses.items():
    print(f"  {k}: {v:.4f}")

# -------------------------
# 9) PLOT example: choose a test sample (the last test sample)
# -------------------------
sample_idx = -1  # last test sample
sample_date = dates_test[sample_idx]  # starting date for the prediction block
pred_dates = pd.date_range(start=sample_date, periods=PRED_DAYS, freq='B')  # business days (approx)

actual_block = y_test_inv[sample_idx]   # (pred_len, n_features)
pred_block = y_pred_inv[sample_idx]

plt.figure(figsize=(14, 10))
for i, fname in enumerate(FEATURES):
    plt.subplot(2, 2, i + 1)
    plt.plot(pred_dates, actual_block[:, i], marker='o', label="Actual")
    plt.plot(pred_dates, pred_block[:, i], marker='x', label="Predicted")
    plt.title(f"{fname} — Actual vs Predicted (7 days)")
    plt.xlabel("Date")
    plt.ylabel(fname)
    plt.legend()
plt.tight_layout()
plt.show()

# -------------------------
# 10) PRINT predicted next 7 days from the last available window in the dataset
# -------------------------
# Build the last input window
last_window = scaled_vals[-SEQ_LEN:, :]  # shape (SEQ_LEN, n_features)
last_window = last_window.reshape((1, SEQ_LEN, n_features))
next7_flat = model.predict(last_window)  # (1, pred_len*n_features)
next7_inv = invert_predictions(next7_flat, scaler, PRED_DAYS, n_features)[0]  # (pred_len, n_features)

next7_dates = pd.date_range(start=data["Date"].iloc[-1] + pd.Timedelta(days=1), periods=PRED_DAYS, freq='B')  # business days
pred_df = pd.DataFrame(next7_inv, columns=FEATURES)
pred_df.insert(0, "Date", next7_dates)
print("\nPredicted next 7 business days (Open, High, Low, Close):")
print(pred_df.to_string(index=False, float_format="%.4f"))

# -------------------------
# 11) Save model (already saved via ModelCheckpoint), optional save scaler
# -------------------------
import joblib
scaler_path = "scaler_minmax.save"
joblib.dump(scaler, scaler_path)
print(f"\nModel saved to: {MODEL_SAVE_PATH}  (checkpointed during training if val_loss improved)")
print(f"Scaler saved to: {scaler_path}")

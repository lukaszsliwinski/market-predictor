import os
import pickle
import sys

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date, timedelta

from sklearn.preprocessing import StandardScaler

CACHE_FILE = "es_futures_cache.pkl"

# -----------------------------
# CACHE CLEANUP
# -----------------------------

if "--clear-cache" in sys.argv:
  try:
    os.remove(CACHE_FILE)
  except FileNotFoundError:
    pass

def get_data(start_date, end_date):
  # =========================
  # LOAD FROM CACHE OR DOWNLOAD
  # =========================

  if os.path.exists(CACHE_FILE):
    print("Loading data from cache...")
    with open(CACHE_FILE, "rb") as f:
      data = pickle.load(f)

  else:
    print("Downloading data from yfinance...")

    data = yf.download(
      "ES=F",
      start=start_date,
      end=end_date,
      interval="1h",
      group_by="column",
      auto_adjust=False
    )

    if isinstance(data.columns, pd.MultiIndex):
      data.columns = data.columns.droplevel(1)

    data = data.reset_index()

    # save cache
    with open(CACHE_FILE, "wb") as f:
      pickle.dump(data, f)

  # =========================
  # 1. TIME
  # =========================

  # UTC (original)
  data["Datetime_og"] = pd.to_datetime(data["Datetime"], utc=True)

  # NY time (with DST)
  data["Datetime"] = data["Datetime_og"].dt.tz_convert("America/New_York")

  # extract date and hour from local time
  data["Date"] = data["Datetime"].dt.date
  data["Hour"] = data["Datetime"].dt.hour

  # =========================
  # 2. SESSION DAY (CME logic: start 18:00)
  # =========================

  data["Session_Day"] = data["Datetime"].dt.date

  mask = data["Datetime"].dt.hour < 18
  data.loc[mask, "Session_Day"] = (data.loc[mask, "Datetime"] - pd.Timedelta(days=1)).dt.date

  # =========================
  # 3. SESSION TYPE (RTH vs ETH)
  # =========================

  hour = data["Datetime"].dt.hour

  data["Session_Type"] = "ETH"

  rth_mask = (hour >= 9) & (hour < 16)
  data.loc[rth_mask, "Session_Type"] = "RTH"

  # =========================
  # 6. ORDER COLUMNS
  # =========================

  data = data[[
    "Datetime",
    "Datetime_og",
    "Date",
    "Hour",
    "Session_Day",
    "Session_Type",
    "Open",
    "Close",
    "High",
    "Low",
    "Volume",
  ]]

  # =========================
  # 5. FILL MISSING VOLUMES
  # =========================

  # Fill all 18:00 rows with the next available 19:00 row volume
  for idx in data[data['Hour'] == 18].index:
    # If the next row is 19:00, assign that volume
    if idx < len(data) - 1 and data.loc[idx + 1, 'Hour'] == 19:
      next_19_volume = data.loc[idx + 1, 'Volume']  # 19:00 volume
      data.loc[idx, 'Volume'] = round(next_19_volume * 0.9)  # round and assign
    else:
      # If this is the last 18:00 row, copy volume from the previous 18:00
      previous_18_index = data.loc[data['Hour'] == 18].index[-2]  # Ostatni wiersz 18:00 przed ostatnim
      data.loc[idx, 'Volume'] = data.loc[previous_18_index, 'Volume']

  # Replace zeros with NaN so missing volumes can be filled
  data['Volume'] = data['Volume'].replace(0, np.nan)

  # Fill missing volumes:
  # If volume is NaN, fill with the average of the previous and next hour
  data['Volume'] = data['Volume'].fillna((data['Volume'].shift(1) + data['Volume'].shift(-1)) / 2)

  # For the first and last row, if volume is NaN, copy from adjacent hours
  data.loc[0, 'Volume'] = data.loc[0, 'Volume'] if pd.notna(data.loc[0, 'Volume']) else data.loc[1, 'Volume']
  data.loc[len(data) - 1, 'Volume'] = data.loc[len(data) - 1, 'Volume'] if pd.notna(data.loc[len(data) - 1, 'Volume']) else data.loc[len(data) - 2, 'Volume']

  # Round volume to integer
  data['Volume'] = data['Volume'].round().astype(int)

  # =========================
  # 14. EXPORT
  # =========================

  data.to_csv("es_data.csv", index=False)

  return data




def create_features(start_date=date.today()-timedelta(days=729), end_date=date.today()+timedelta(days=1)):
  data = get_data(start_date=start_date, end_date=end_date)

  # =========================
  # 4. IS_RTH
  # =========================

  data["is_rth"] = (data["Session_Type"] == "RTH").astype(int)

  # =========================
  # 6. MOMENTUM (core)
  # =========================
  data["ret_1"] = data["Close"].pct_change(1)
  data["ret_3"] = data["Close"].pct_change(3)
  data["ret_6"] = data["Close"].pct_change(6)
  data["ret_24"] = data["Close"].pct_change(24)

  # acceleration / deceleration
  data["mom_acc_3_1"] = data["ret_1"] - data["ret_3"]
  data["mom_acc_6_3"] = data["ret_3"] - data["ret_6"]

  # =========================
  # 7. VOLATILITY REGIME
  # =========================
  data["vol_6"] = data["ret_1"].rolling(6).std()
  data["vol_24"] = data["ret_1"].rolling(24).std()
  data["vol_48"] = data["ret_1"].rolling(48).std()

  data["vol_regime"] = data["vol_24"] / data["vol_24"].rolling(100).mean()
  data["vol_trend"] = data["vol_24"] / data["vol_6"]

  # =========================
  # 8. TREND / MEAN REVERSION CONTEXT
  # =========================
  data["sma_10"] = data["Close"].rolling(10).mean()
  data["sma_24"] = data["Close"].rolling(24).mean()

  data["ema_10"] = data["Close"].ewm(span=10).mean()
  data["ema_24"] = data["Close"].ewm(span=24).mean()

  data["price_vs_sma10"] = (data["Close"] - data["sma_10"]) / data["Close"]
  data["price_vs_sma24"] = (data["Close"] - data["sma_24"]) / data["Close"]

  data["ema_slope_10"] = data["ema_10"] - data["ema_10"].shift(3)
  data["ema_slope_24"] = data["ema_24"] - data["ema_24"].shift(3)

  # =========================
  # 9. BREAKOUT / RANGE POSITION
  # =========================
  data["high_24"] = data["High"].rolling(24).max()
  data["low_24"] = data["Low"].rolling(24).min()

  data["breakout_up"] = (data["Close"] - data["high_24"]) / data["Close"]
  data["breakout_down"] = (data["Close"] - data["low_24"]) / data["Close"]

  data["range_pos_24"] = (
    (data["Close"] - data["low_24"]) / 
    (data["high_24"] - data["low_24"])
  )

  # =========================
  # 10. MICROSTRUCTURE
  # =========================
  data["hl_range"] = (data["High"] - data["Low"]) / data["Close"]
  data["body"] = (data["Close"] - data["Open"]) / data["Close"]

  data["upper_wick"] = (data["High"] - np.maximum(data["Open"], data["Close"])) / data["Close"]
  data["lower_wick"] = (np.minimum(data["Open"], data["Close"]) - data["Low"]) / data["Close"]

  data["wick_ratio"] = data["upper_wick"] / (data["hl_range"] + 1e-8)

  # =========================
  # 11. TIME FEATURES
  # =========================
  data["sin_hour"] = np.sin(2 * np.pi * data["Hour"] / 24)
  data["cos_hour"] = np.cos(2 * np.pi * data["Hour"] / 24)

  # =========================
  # 12. VOLUME (simple but important)
  # =========================
  data["vol_chg"] = data["Volume"].pct_change(1)
  data["vol_ma_24"] = data["Volume"].rolling(24).mean()
  data["vol_ratio"] = data["Volume"] / data["vol_ma_24"]



  # =========================
  # 12.5 ORDER FLOW PROXIES (from OHLCV only)
  # =========================

  # data["range"] = (data["High"] - data["Low"]) / data["Close"]
  # data["body_abs"] = np.abs(data["body"])

  # data["effort"] = data["range"] / (data["Volume"] + 1e-9)
  # data["absorption"] = data["Volume"] / (data["range"] + 1e-9)

  # data["direction"] = np.sign(data["Close"] - data["Open"])
  # data["signed_volume"] = data["direction"] * data["Volume"]
  # data["volume_imbalance"] = data["signed_volume"].rolling(10).sum()

  # data["volatility"] = data["range"].rolling(10).std()
  # data["volume_volatility_ratio"] = data["Volume"].rolling(10).mean() / (data["volatility"] + 1e-9)

  # data["rejection_up"] = (data["High"] - data["Close"]).rolling(10).mean()
  # data["rejection_down"] = (data["Close"] - data["Low"]).rolling(10).mean()


  # =========================
  # 13. TARGET (1 - direction continuation, 0 - direction reversal)
  # Target is taken from the next bar, then the last row is removed (because it has no target)
  # =========================

  data["target"] = (np.sign(data["body"]) == np.sign(data["body"].shift(1))).astype(int)
  data["target"] = data["target"].shift(-1)
  data = data.iloc[:-1]

  # =========================
  # SCALING
  # =========================

  scale_cols = [
    "vol_6",
    "vol_24",
    "vol_48",

    "sma_10",
    "sma_24",
    "ema_10",
    "ema_24",
    "high_24",
    "low_24",

    "vol_ma_24",

    "hl_range",
    "upper_wick",
    "lower_wick",
  ]

  scaler = StandardScaler()
  data[scale_cols] = scaler.fit_transform(data[scale_cols])

  # =========================
  # CLEANUP
  # =========================

  data = data.drop(columns=[
    "Datetime_og","Date","Hour",
    "Session_Day","Session_Type",
    "Open","Close","High","Low","Volume"
  ])

  data[data.columns.difference(["Datetime"])] = data[data.columns.difference(["Datetime"])].astype("float32")
  data = data.dropna().reset_index(drop=True)
  print(data['target'].value_counts())
  data.to_csv("es_features.csv", index=False)

create_features(end_date="2026-04-23")
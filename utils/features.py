import os
import yfinance as yf
import pandas as pd
import numpy as np
import exchange_calendars as xcals


def create_features(latest_open=0, latest_close=0, latest_high=0, latest_low=0, latest_volume=0):
  # =========================
  # YFINANCE DATA
  # =========================

  # Get backuped raw yfinance data from .csv
  raw_file_path = "data/raw.csv"

  data = pd.read_csv(raw_file_path)
  data["Datetime"] = pd.to_datetime(data["Datetime"], utc=True)
  data = data[["Datetime", "Open", "High", "Low", "Close", "Volume"]]
  raw_data = data.copy()

  # Import latest data w from yfinance
  data = yf.download(
    "ES=F",
    period="max",
    interval="1h",
  )

  # Remove MultiIndex if present
  if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.droplevel(1)
  
  data = data.reset_index()
  data["Datetime"] = pd.to_datetime(data["index"])
  data = data[["Datetime", "Open", "High", "Low", "Close", "Volume"]]
  data.columns.name = None

  # Aggregate latest and backuped data
  data["Datetime"] = pd.to_datetime(data["Datetime"], utc=True)
  last_raw_datetime = raw_data["Datetime"].max()
  new_data = data[data["Datetime"] >= last_raw_datetime]

  data = pd.concat([raw_data, new_data], ignore_index=True)
  data = data.drop_duplicates(subset=["Datetime"], keep="last")
  data = data.sort_values("Datetime").reset_index(drop=True)

  data.iloc[:-1].to_csv(raw_file_path, index=False)

  # Overwrite last row with manually entered data
  data.loc[data.index[-1], "Open"] = latest_open
  data.loc[data.index[-1], "High"] = latest_high
  data.loc[data.index[-1], "Low"] = latest_low
  data.loc[data.index[-1], "Close"] = latest_close
  data.loc[data.index[-1], "Volume"] = latest_volume

  # Add current hour due to yfinance delay
  last_datetime = pd.Timestamp(data.iloc[-1]["Datetime"])
  now_hour = pd.Timestamp.now(tz=last_datetime.tz).floor("h")

  if last_datetime.floor("h") != now_hour:
    data.loc[len(data)] = {
      "Datetime": now_hour,
      "Open": data.iloc[-1]["Close"],
      "High": 0.0,
      "Low": 0.0,
      "Close": 0.0,
      "Volume": 0.0
    }

  # =========================
  # TIME FEATURES
  # =========================

  # UTC (original)
  data["Datetime"] = pd.to_datetime(data["Datetime"], utc=True)

  # NY time (with DST)
  data["Datetime_NY"] = data["Datetime"].dt.tz_convert("America/New_York")

  # Extract date and hour from local time
  data["Date_NY"] = data["Datetime_NY"].dt.date
  data["Hour_NY"] = data["Datetime_NY"].dt.hour

  # Session day (CME logic: start 18:00)
  data["Session_Day"] = data["Datetime_NY"].dt.date

  # Shift bars before 18:00 into the previous CME session day
  mask = data["Datetime_NY"].dt.hour < 18
  data.loc[mask, "Session_Day"] = (data.loc[mask, "Datetime_NY"] - pd.Timedelta(days=1)).dt.date
  data["Session_weekday"] = pd.to_datetime(data["Session_Day"]).dt.dayofweek.map({6: 1, 0: 2, 1: 3, 2: 4, 3: 5})

  # Features
  data["Sin_hour"] = np.sin(2 * np.pi * data["Hour_NY"] / 24)
  data["Cos_hour"] = np.cos(2 * np.pi * data["Hour_NY"] / 24)

  data["Sin_weekday"] = np.sin(2 * np.pi * data["Session_weekday"] / 7)
  data["Cos_weekday"] = np.cos(2 * np.pi * data["Session_weekday"] / 7)



  # =========================
  # IMPUTE MISSING VOLUME AND CREATE PREV HOUR FEATURES
  # =========================

  # Impute missing volume values for session open bars and remaining gaps
  # Fill all 18:00 rows with the next available 19:00 row volume
  for idx in data[data["Hour_NY"] == 18].index:
    # If the next row is 19:00, assign that volume
    if idx < len(data) - 1 and data.loc[idx + 1, "Hour_NY"] == 19:
      next_19_volume = data.loc[idx + 1, "Volume"]  # 19:00 volume
      data.loc[idx, "Volume"] = round(next_19_volume * 0.9)  # round and assign
    else:
      # If this is the last 18:00 row, copy volume from the previous 18:00
      previous_18_index = data.loc[data["Hour_NY"] == 18].index[-2]  # Last 18:00 row before the final one
      data.loc[idx, "Volume"] = data.loc[previous_18_index, "Volume"]

  # Replace zeros with NaN so missing volumes can be filled
  data["Volume"] = data["Volume"].replace(0, np.nan)

  # If volume is NaN, fill with the average of the previous and next hour
  data["Volume"] = data["Volume"].fillna((data["Volume"].shift(1) + data["Volume"].shift(-1)) / 2)

  # For the first and last row, if volume is NaN, copy from adjacent hours
  data.loc[0, "Volume"] = data.loc[0, "Volume"] if pd.notna(data.loc[0, "Volume"]) else data.loc[1, "Volume"]
  data.loc[len(data) - 1, "Volume"] = data.loc[len(data) - 1, "Volume"] if pd.notna(data.loc[len(data) - 1, "Volume"]) else data.loc[len(data) - 2, "Volume"]

  # Round volume to integer
  data["Volume"] = data["Volume"].round().astype(int)

  # Previous bar values for price and volume-based features
  open_prev = data["Open"].shift(1)
  close_prev = data["Close"].shift(1)
  high_prev = data["High"].shift(1)
  low_prev = data["Low"].shift(1)
  volume_prev = data["Volume"].shift(1)
  session_open = data.groupby("Session_Day")["Open"].transform("first")


  # =========================
  # SENTIMENT / TREND / VOLATILITY FEATURES
  # =========================

  # Session type (RTH vs ETH) - for is_RTH
  hour = data["Datetime_NY"].dt.hour

  data["Session_Type"] = "ETH"

  rth_mask = (hour >= 9) & (hour < 16)
  data.loc[rth_mask, "Session_Type"] = "RTH"

  # move from session open to previous completed candle
  data["Early_move"] = (close_prev - session_open) / session_open

  # threshold
  THRESHOLD = data["Early_move"].abs().median()

  # session open per day (18:00) - used as the daily reference price for direction features
  data["Session_open"] = data.groupby("Session_Day")["Open"].transform("first")

  # Features
  data["Is_RTH"] = (data["Session_Type"] == "RTH").astype("float32")
  data["Is_trending"] = (data["Early_move"].abs() > THRESHOLD).astype("float32")

  data["Prev_volatility"] = (high_prev - low_prev) / open_prev   # hourly volatility (%)


  data["Session_prev_volatility"] = (
    data.groupby("Session_Day")["Prev_volatility"]
    .transform(lambda x: x.shift(1).expanding().mean())
  )
  # data["Session_prev_volatility"] = data.groupby("Session_Day")["Prev_volatility"].transform("mean") # average session volatility
  data["Rel_prev_volatility"] = data["Prev_volatility"] / data["Session_prev_volatility"] # relative volatility (context)

  data["Vol_prev_log"] = np.log1p(volume_prev)
  
  data["Prev_hour_dir"] = np.sign(data["Close"].shift(1) - data["Open"].shift(1)) # TODO: decide whether to delete
  data["Day_dir_till_hour"] = np.sign(close_prev - data["Session_open"])  # For target only



  # =========================
  # MOMENTUM / MEAN REVERSION FEATURES
  # =========================

  # Returns and momentum acceleration
  data["Ret_1h"] = close_prev.pct_change(1)
  data["Ret_2h"] = close_prev.pct_change(2)
  data["Ret_4h"] = close_prev.pct_change(4)  # For dependent features only
  data["Ret_8h"] = close_prev.pct_change(8)  # For dependent features only

  data["Momentum_accel"] = data["Ret_1h"] - data["Ret_4h"]

  # VWAP (session cumulative) & distance from VWAP
  typical_price_prev = (high_prev + low_prev + close_prev) / 3
  cum_vol_price = (typical_price_prev * volume_prev).groupby(data["Session_Day"]).cumsum()
  cum_volume = volume_prev.groupby(data["Session_Day"]).cumsum()

  data["VWAP"] = cum_vol_price / cum_volume
  data["Dist_VWAP"] = (close_prev - data["VWAP"]) / data["VWAP"]
  data["VWAP_log"] = np.log1p(data["VWAP"])

  # Rolling volatility
  returns_1h_prev = close_prev.pct_change()

  data["Vol_4h"] = returns_1h_prev.rolling(4).std()
  data["Vol_24h"] = returns_1h_prev.rolling(24).std()

  # Z-score moves (volatility-adjusted returns)
  rolling_std_20 = returns_1h_prev.rolling(20).std()
  rolling_std_20 = rolling_std_20.replace(0, np.nan)

  data["Zscore_4h"] = data["Ret_4h"] / rolling_std_20
  data["Zscore_8h"] = data["Ret_8h"] / rolling_std_20

  # Distance from moving average (last 50 candles)
  ma_50 = close_prev.rolling(50).mean()
  data["Dist_MA50"] = (close_prev - ma_50) / ma_50
 
  # Range position (where the current price is within the 12h high-low range)
  rolling_high_12 = high_prev.rolling(12).max()
  rolling_low_12 = low_prev.rolling(12).min()
  range_size = rolling_high_12 - rolling_low_12
  range_size = range_size.replace(0, np.nan)
  data["Range_position_12h"] = (close_prev - rolling_low_12) / range_size



  # =========================
  # RELATED INDEXES RTH (IN LOOP)
  # =========================
  for prefix, cal_code in {
    "Nikkei": "XTKS",
    "Taiex": "XTAI",
    "EuroStoxx": "XEUR"
  }.items():
    calendar = xcals.get_calendar(cal_code)
    data[f"{prefix}_is_RTH"] = (
      data["Datetime"]
      .apply(lambda ts: float(calendar.is_open_on_minute(ts)))
      .astype("float32")
    )



  # =========================
  # CLEANUP, TARGET, FEATURE SELECTION
  # =========================

  # remove everything until the first occurrence of hour 18
  idx = data.index[data["Hour_NY"] == 18]

  if len(idx) > 0:
    first_18_idx = idx[0]
    data = data.loc[first_18_idx:].reset_index(drop=True)

  # Target indicates whether the current hour maintains the session direction from the session open
  data["Dir"] = np.sign(data["Close"] - data["Open"]).astype("float32")
  data["Target"] = (data["Dir"] == data["Day_dir_till_hour"]).astype("float32")

  # Select the final feature set for training/export
  data = data[[
    "Datetime",
    "Date_NY",
    "Session_weekday",


    ### Using only in backtest ###
    "Open",
    "Close",
    "High",
    "Low",
    "Dir",
    "Day_dir_till_hour",
    ### ---------------------- ###

    "Hour_NY",
    "Sin_hour",
    "Cos_hour",
    "Sin_weekday",
    "Cos_weekday",
    "Is_RTH",
    "Is_trending",
    "Prev_volatility",
    "Session_prev_volatility",
    "Rel_prev_volatility",
    # "Prev_hour_dir",
    "Vol_prev_log",
    "Ret_1h",
    "Ret_2h",
    "VWAP_log",
    "Dist_VWAP",
    "Vol_4h",
    "Vol_24h",
    "Zscore_4h",
    "Zscore_8h",
    "Dist_MA50",
    "Momentum_accel",
    "Range_position_12h",
    "Nikkei_is_RTH",
    "Taiex_is_RTH",
    "EuroStoxx_is_RTH",
    "Target"
  ]]

  # Remove 17:00 and 18:00
  data = data[(data["Hour_NY"] != 17) & (data["Hour_NY"] != 18)]

  # Remove empty values
  data = data.dropna().reset_index(drop=True)

  # Save to CSV (append new data)
  csv_file_path = "data/data.csv"

  data["Datetime"] = pd.to_datetime(data["Datetime"], utc=True)

  if os.path.exists(csv_file_path):

    old = pd.read_csv(csv_file_path)
    old["Datetime"] = pd.to_datetime(old["Datetime"], utc=True)

    old_set = set(old["Datetime"])

    new_data = data[~data["Datetime"].isin(old_set)]

    new_data = new_data.reindex(columns=old.columns)

    if len(new_data) > 0:
      new_data.to_csv(csv_file_path, mode="a", header=False, index=False)
      print(f"Dodano {len(new_data)} nowych rekordów")
    else:
      print("Brak nowych danych do dodania")

  else:
    data.to_csv(csv_file_path, index=False)


create_features(latest_open=100, latest_close=200, latest_high=100, latest_low=100, latest_volume=1000000)

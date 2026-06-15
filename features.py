import os
import yfinance as yf
import pandas as pd
import numpy as np
import exchange_calendars as xcals


def create_features(open_lch=0, close_lch=0, high_lch=0, low_lch=0):
  # =========================
  # YFINANCE DATA
  # =========================

  # Get backuped raw yfinance data from .csv
  raw_file_path = "data/raw.csv"

  data = pd.read_csv(raw_file_path)
  data["Datetime"] = pd.to_datetime(data["Datetime"], utc=True)
  data = data[["Datetime", "Open", "High", "Low", "Close"]]
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
  data = data[["Datetime", "Open", "High", "Low", "Close"]]
  data.columns.name = None

  # Aggregate latest and backuped data
  data["Datetime"] = pd.to_datetime(data["Datetime"], utc=True)
  last_raw_datetime = raw_data["Datetime"].max()
  new_data = data[data["Datetime"] >= last_raw_datetime]

  data = pd.concat([raw_data, new_data], ignore_index=True)
  data = data.drop_duplicates(subset=["Datetime"], keep="last")
  data = data.sort_values("Datetime").reset_index(drop=True)

  data.iloc[:-2].to_csv(raw_file_path, index=False)

  # Data adaptation due to yfinance delay
  last_row_datime = pd.Timestamp(data.iloc[-1]["Datetime"])
  now_hour = pd.Timestamp.now(tz=last_row_datime.tz).floor("h")

  # if yfinance returned current hour
  if last_row_datime.floor("h") == now_hour:
    data = data.iloc[:-1].copy()  # remove last row (current hour) if present

  # Overwrite last closed hour with data from props
  data.loc[data.index[-1], "Open"] = open_lch
  data.loc[data.index[-1], "High"] = high_lch
  data.loc[data.index[-1], "Low"] = low_lch
  data.loc[data.index[-1], "Close"] = close_lch

  # Create a new row for the current hour with open price
  data.loc[len(data)] = {
    "Datetime": now_hour,
    "Open": data.iloc[-1]["Close"],
    "High": 0.0,
    "Low": 0.0,
    "Close": 0.0
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



  # Previous bar values for price
  open_prev = data["Open"].shift(1)
  close_prev = data["Close"].shift(1)
  high_prev = data["High"].shift(1)
  low_prev = data["Low"].shift(1)
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
  ) # average session volatility
  data["Rel_prev_volatility"] = data["Prev_volatility"] / data["Session_prev_volatility"] # relative volatility (context)

  data["Prev_hour_dir"] = np.sign(data["Close"].shift(1) - data["Open"].shift(1))
  data["Day_dir_till_hour"] = np.sign(close_prev - data["Session_open"])



  # =========================
  # MOMENTUM / MEAN REVERSION FEATURES
  # =========================

  # Returns and momentum acceleration
  data["Ret_1h"] = close_prev.pct_change(1)
  data["Ret_2h"] = close_prev.pct_change(2)
  data["Ret_4h"] = close_prev.pct_change(4)
  data["Ret_8h"] = close_prev.pct_change(8)

  data["Momentum_accel"] = data["Ret_1h"] - data["Ret_4h"]

  # Rolling volatility
  returns_1h_prev = close_prev.pct_change()

  data["Vol_2h"] = returns_1h_prev.rolling(2).std()
  data["Vol_4h"] = returns_1h_prev.rolling(4).std()
  data["Vol_8h"] = returns_1h_prev.rolling(8).std()
  data["Vol_24h"] = returns_1h_prev.rolling(24).std()

  # Z-score moves (volatility-adjusted returns)
  rolling_std_20 = returns_1h_prev.rolling(20).std()
  rolling_std_20 = rolling_std_20.replace(0, np.nan)

  data["Zscore_1h"] = data["Ret_1h"] / rolling_std_20
  data["Zscore_2h"] = data["Ret_2h"] / rolling_std_20
  data["Zscore_4h"] = data["Ret_4h"] / rolling_std_20
  data["Zscore_8h"] = data["Ret_8h"] / rolling_std_20

  # Distance from moving average
  ma_10 = close_prev.rolling(10).mean()
  ma_20 = close_prev.rolling(20).mean()
  ma_30 = close_prev.rolling(30).mean()
  ma_40 = close_prev.rolling(40).mean()
  ma_50 = close_prev.rolling(50).mean()

  data["Dist_MA10"] = (close_prev - ma_10) / ma_10
  data["Dist_MA20"] = (close_prev - ma_20) / ma_20
  data["Dist_MA30"] = (close_prev - ma_30) / ma_30
  data["Dist_MA40"] = (close_prev - ma_40) / ma_40
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

    "Open",
    "Close",
    "High",
    "Low",
    "Dir",
    "Day_dir_till_hour",

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
    "Prev_hour_dir",
    
    "Ret_1h",
    "Ret_2h",
    "Ret_4h",
    "Ret_8h",
    
    "Vol_2h",
    "Vol_4h",
    "Vol_8h",
    "Vol_24h",
    
    "Zscore_1h",
    "Zscore_2h",
    "Zscore_4h",
    "Zscore_8h",
    
    "Dist_MA10",
    "Dist_MA20",
    "Dist_MA30",
    "Dist_MA40",
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

    # physically remove last two non-empty lines from the CSV (keep header)
    with open(csv_file_path, "r", encoding="utf-8") as f:
      lines = f.read().splitlines()

    if len(lines) > 1:
      i = len(lines) - 1
      removed = 0
      while i > 0 and removed < 2:
        if lines[i].strip() != "":
          lines.pop(i)
          removed += 1
        i -= 1
      if removed > 0:
        with open(csv_file_path, "w", encoding="utf-8") as f:
          f.write("\n".join(lines) + "\n")

    # read existing datetimes only
    try:
      old_dt = pd.read_csv(csv_file_path, usecols=["Datetime"])['Datetime']
      old_dt = pd.to_datetime(old_dt, utc=True)
      old_set = set(old_dt)
    except Exception:
      old_set = set()

    # select rows missing in the file
    new_data = data[~data["Datetime"].isin(old_set)]

    # align columns to existing file header when possible
    try:
      header = pd.read_csv(csv_file_path, nrows=0).columns.tolist()
      new_data = new_data.reindex(columns=header)
    except Exception:
      pass

    if len(new_data) > 0:
      new_data.to_csv(csv_file_path, mode="a", header=False, index=False)
      print(f"Dodano {len(new_data)} nowych rekordów")
    else:
      print("Brak nowych danych do dodania")

  else:
    data.to_csv(csv_file_path, index=False)


create_features(open_lch=7614.25, close_lch=7623.25, high_lch=7625.0, low_lch=7605.5)


# create_features(open_lch=7614.25, close_lch=7623.25, high_lch=7625.0, low_lch=7605.5)


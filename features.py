import os
import pickle
import sys

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo
import exchange_calendars as xcals


def load_yf_data(full):
  """
  Download data from yfinance and save to cache.
  Subsequent calls will load from cache only.
  """

  if full:
    start_date = date.today()-timedelta(days=728)
  else:
    start_date = date.today()-timedelta(days=4)

  # tickers + names
  tickers = {
    "es": "ES=F",
    "nikkei": "^N225",
    "taiex": "^TWII",
    "euro_stoxx": "^STOXX50E"
  }

  if "--load-cache" in sys.argv:
    cache_file_path = "data/yf_cache_2026-04-29_09-03-00.pkl"

    if os.path.exists(cache_file_path) and full:
      print("Loading data from cache...")
      with open(cache_file_path, "rb") as f:
        data_dict = pickle.load(f)
    else:
      print("Unable to load cache data for current prediction!")
      pass

  else:
    print("Downloading data from yfinance...")
    data_dict = {}

    for name, ticker in tickers.items():
      df = yf.download(
        ticker,
        start=start_date,
        end=date.today()+timedelta(days=1),
        interval="1h",
        group_by="column",
        auto_adjust=False
      )

      # remove MultiIndex if present
      if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
      
      df = df.reset_index()
      df["Datetime"] = pd.to_datetime(df["Datetime"])
      df = df[["Datetime", "Open", "High", "Low", "Close", "Volume"]]

      # save to dictionary
      data_dict[name] = df


    # cache yfinance data
    cache_file_path = datetime.now().strftime("data/yf_cache_%Y-%m-%d_%H-%M-%S.pkl")

    if "--save-cache" in sys.argv:
      with open(cache_file_path, "wb") as f:
        pickle.dump(data_dict, f)

  return (data_dict["es"], data_dict["nikkei"], data_dict["taiex"], data_dict["euro_stoxx"])


def create_features(start_date=None, end_date=None, full=False):
  # Load the main ES dataset plus external index series from cache or yfinance
  try:
    data, nikkei, taiex, euro_stoxx = load_yf_data(full=full)

    # add current hour due to yfinance delay
    now_hour = pd.Timestamp.now("UTC").floor("h")

    if data.iloc[-1]["Datetime"].floor("h") != now_hour:
      data.loc[len(data)] = {
        "Datetime": now_hour,
        "Open": data.iloc[-1]["Close"]
      }

  except:
    print("Loading data error!")
    return


  # =========================
  # TIME FEATURES
  # =========================

  # UTC (original)
  data["Datetime"] = pd.to_datetime(data["Datetime"], utc=True)

  # NY time (with DST)
  data["Datetime_NY"] = data["Datetime"].dt.tz_convert("America/New_York")

  # extract date and hour from local time
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
  # INDEX FEATURES (IN LOOP)
  # =========================

  # Merge external index data and derive aligned RTH and direction features
  xtks = xcals.get_calendar("XTKS")
  xtai = xcals.get_calendar("XTAI")
  xeur = xcals.get_calendar("XEUR")

  indices = {
    "Nikkei": {
      "data": nikkei,
      "calendar": xtks
    },

    "Taiex": {
      "data": taiex,
      "calendar": xtai
    },

    "EuroStoxx": {
      "data": euro_stoxx,
      "calendar": xeur
    }
  }

  for prefix, config in indices.items():

    df_index = config["data"]
    calendar = config["calendar"]

    df = df_index.copy()

    df[f"{prefix}_open"] = df["Open"]
    df[f"{prefix}_close"] = df["Close"]
    df = df[["Datetime", f"{prefix}_open", f"{prefix}_close"]]

    df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True)

    data = data.merge(df, on="Datetime", how="left")

    # RTH from exchange_calendar
    data[f"{prefix}_is_RTH"] = data["Datetime"].apply(
      lambda ts: float(calendar.is_open_on_minute(ts))
    ).astype("float32")

    # fillna(0.0)
    data[f"{prefix}_prev_hour_dir"] = np.sign(
      data[f"{prefix}_close"].shift(1) - data[f"{prefix}_open"].shift(1)
    ).fillna(0.0)

    session_open = data.groupby("Session_Day")[f"{prefix}_open"].transform("first")

    # Use the previous session's open for the index only when the prior bar was in regular trading hours
    data[f"{prefix}_session_open"] = np.where(
      data[f"{prefix}_is_RTH"].shift(1) == 1.0,
      session_open,
      np.nan
    )

    data[f"{prefix}_day_dir_till_hour"] = np.sign(
      data[f"{prefix}_close"].shift(1) - data[f"{prefix}_session_open"]
    )

    # forward fill without altering initial NaN values
    data[f"{prefix}_day_dir_till_hour"] = data[f"{prefix}_day_dir_till_hour"].ffill()



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


  # Optional filter by session day range
  if start_date is not None:
    # Convert to date if a string or datetime was provided
    start_dt = pd.to_datetime(start_date).date()
    data = data[data["Session_Day"] >= start_dt]

  if end_date is not None:
    # Convert to date if a string or datetime was provided
    end_dt = pd.to_datetime(end_date).date()
    data = data[data["Session_Day"] <= end_dt]

  # After filtering, reset the index to avoid issues when selecting columns
  data = data.reset_index(drop=True)

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
    "Nikkei_prev_hour_dir",
    "Nikkei_day_dir_till_hour",
    
    "Taiex_is_RTH",
    "Taiex_prev_hour_dir",
    "Taiex_day_dir_till_hour",
    
    "EuroStoxx_is_RTH",
    "EuroStoxx_prev_hour_dir",
    "EuroStoxx_day_dir_till_hour",

    "Target"
  ]]

  # Remove 17:00 and 18:00
  data = data[(data["Hour_NY"] != 17) & (data["Hour_NY"] != 18)]

  # Export data to csv if not for predict
  if full:
    # Remove empty values    
    data = data.dropna().reset_index(drop=True)

    # Remove the last line if it concerns the current time
    if data.iloc[-1]["Hour_NY"] == datetime.now(ZoneInfo("America/New_York")).hour:
      data = data.iloc[:-1]

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
  
  else:
    # Leave only current hour
    data = data.tail(1)
    data.to_csv("data/predict.csv", index=False)

create_features(full="--predict" not in sys.argv)


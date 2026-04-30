import os
import pickle
import sys

import yfinance as yf
import pandas as pd
import numpy as np
import talib

from datetime import date
from functools import reduce

import lightgbm as lgb
import optuna
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, log_loss, confusion_matrix


CACHE_FILE = "yf_data_full_cache.pkl"


# -----------------------------
# CACHE CLEANUP
# -----------------------------

if "--clear-cache" in sys.argv:
  try:
    os.remove(CACHE_FILE)
  except FileNotFoundError:
    pass


# -----------------------------
# HELPER FUNCTIONS
# -----------------------------

def load_yf_data_full_cache():    
  """
  Download data from yfinance and save to cache.
  Subsequent calls will load from cache only.
  """

  # tickers + names
  tickers = {
    "spy": "SPY",               # od 1993-01-29
    "euro_stoxx": "^STOXX50E",  # od 2007-03-30
    "nikkei": "^N225",          # od 1965-01-05
    "taiex": "^TWII",           # od 1997-07-02
    "gold": "GC=F",             # od 2000-08-30
    "oil": "CL=F",              # od 2000-08-23
    "copper": "HG=F",           # od 2000-08-30
    "vix": "^VIX",              # od 1990-01-02
    "dxy": "DX-Y.NYB",          # od 1971-01-04
    "futures": "ES=F",          # od 2000-09-18
    "yield10y": "^TNX",         # od 1962-01-02
  }

  if os.path.exists(CACHE_FILE):
    print("Loading data from cache...")
    with open(CACHE_FILE, "rb") as f:
      data_dict = pickle.load(f)
  else:
    print("Downloading data from yfinance...")
    data_dict = {}

    for name, ticker in tickers.items():
      df = yf.download(
        ticker,
        start="2009-01-01",  # earliest safe date for which all data is available
        end=date.today(),
        interval="1d",
        group_by="column",
        auto_adjust=False
      )
      
      # remove MultiIndex if present
      if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
      
      df = df.reset_index()
      df["Date"] = pd.to_datetime(df["Date"])
      df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
           
      # save to dictionary
      data_dict[name] = df
    
    with open(CACHE_FILE, "wb") as f:
      pickle.dump(data_dict, f)
  
  return (data_dict["spy"], data_dict["euro_stoxx"], data_dict["nikkei"], data_dict["taiex"], data_dict["gold"], data_dict["oil"], data_dict["copper"], data_dict["vix"], data_dict["dxy"], data_dict["futures"], data_dict["yield10y"])



def dir(close_price, open_price):
  """
  Return a 2-class daily direction feature (0, 1) from Open and Close prices.
  """

  # daily return
  ret = (close_price - open_price) / open_price
  
  # 3-class edge case handled by threshold
  return np.where(ret > 0, 1, 0)





# -----------------------------
# MAIN FUNCTION
# -----------------------------

def get_features(start_date, end_date):
  # -----------------------------
  # Load data from yfinance or cache
  # -----------------------------

  spy, euro_stoxx, nikkei, taiex, gold, oil, copper, vix, dxy, futures, yield10y = load_yf_data_full_cache()


  ########## ------------------------------------------------- ##########
  ##########      CORRELATED INDEX AND COMMODITY FEATURES      ##########
  ########## ------------------------------------------------- ##########

  # Euro Stoxx 50 - opening gap on the prediction day
  euro_stoxx["Close_prev"] = euro_stoxx["Close"].shift(1)
  euro_stoxx["gap_EU"] = (euro_stoxx["Open"] - euro_stoxx["Close_prev"]) / euro_stoxx["Close_prev"]

  euro_stoxx_feature = euro_stoxx[["Date", "gap_EU"]]
  euro_stoxx_feature.dropna().reset_index(drop=True)

  # Nikkei 225 - opening gap and direction on the prediction day
  nikkei["Close_prev"] = nikkei["Close"].shift(1)
  nikkei["gap_NIK"] = (nikkei["Open"] - nikkei["Close_prev"]) / nikkei["Close_prev"]
  nikkei["dir_NIK"] = dir(nikkei["Close"], nikkei["Open"])

  nikkei_feature = nikkei[["Date", "gap_NIK", "dir_NIK"]]
  nikkei_feature.dropna().reset_index(drop=True)

  # Taiex - opening gap and direction on the prediction day
  taiex["Close_prev"] = taiex["Close"].shift(1)
  taiex["gap_AS"] = (taiex["Open"] - taiex["Close_prev"]) / taiex["Close_prev"]
  taiex["dir_AS"] = dir(taiex["Close"], taiex["Open"])

  taiex_feature = taiex[["Date", "gap_AS", "dir_AS"]]
  taiex_feature.dropna().reset_index(drop=True)

  # Commodities - 5-day trend, direction and opening gap from previous day to prediction day
  commodities = {
    "gold": gold,
    "oil": oil,
    "copper": copper
  }

  commodity_features = []

  for name, df in commodities.items():
    df[f"Close_prev_{name}"] = df["Close"].shift(1)

    # 1. GAP (overnight move)
    df[f"gap_{name}"] = (df["Open"] - df[f"Close_prev_{name}"]) / df[f"Close_prev_{name}"]

    # 2. MOMENTUM (log return 5D – stabilny cross-asset feature)
    df[f"mom_{name}"] = np.log(df[f"Close_prev_{name}"] / df[f"Close_prev_{name}"].shift(5))

    # 3. (opcjonalnie, ale bardzo polecane) volatility-adjusted momentum
    vol = df[f"Close_prev_{name}"].pct_change().rolling(20).std()
    df[f"mom_vol_{name}"] = df[f"mom_{name}"] / vol

    # feature selection
    features = df[[
        "Date",
        f"gap_{name}",
        f"mom_{name}",
        f"mom_vol_{name}"
    ]]

    features = features.dropna().reset_index(drop=True)
    commodity_features.append(features)

  # Merge all commodities into a single DataFrame by Date
  commodities = reduce(lambda left, right: pd.merge(left, right, on="Date", how="left"), commodity_features)

  
  ########## --------------------------------------------- ##########
  ##########            MARKET REGIME INDICATORS          ##########
  ########## --------------------------------------------- ##########


  # VIX - change in fear regime + opening gap (consistent with other features)
  
  vix["Close_prev"] = vix["Close"].shift(1)

  # VIX opening gap (overnight reaction)
  vix["gap_VIX"] = (vix["Open"] - vix["Close_prev"]) / vix["Close_prev"]

  # fear level change (VIX momentum) to the day before prediction
  vix["mom_VIX"] = (vix["Close_prev"] - vix["Close_prev"].shift(3)) / vix["Close_prev"].shift(3)

  vix_feature = vix[["Date", "gap_VIX", "mom_VIX"]]
  vix_feature = vix_feature.dropna().reset_index(drop=True)


  # DXY - dollar strength change + opening gap (liquidity / risk-off proxy)

  dxy["Close_prev"] = dxy["Close"].shift(1)

  # opening gap (overnight reaction)
  dxy["gap_DXY"] = (dxy["Open"] - dxy["Close_prev"]) / dxy["Close_prev"]

  # dollar level change (DXY momentum) to the day before prediction
  dxy["mom_DXY"] = (dxy["Close_prev"] - dxy["Close_prev"].shift(3)) / dxy["Close_prev"].shift(3)

  dxy_feature = dxy[["Date", "gap_DXY", "mom_DXY"]]
  dxy_feature = dxy_feature.dropna().reset_index(drop=True)


  # ES FUTURES (S&P500 futures) - market regime / SPY expectations
  
  futures["Close_prev"] = futures["Close"].shift(1)

  # futures opening gap (overnight sentiment)
  futures["gap_FUT"] = (futures["Open"] - futures["Close_prev"]) / futures["Close_prev"]

  # contract change (risk-on / risk-off momentum) to the day before prediction
  futures["mom_FUT"] = (futures["Close_prev"] - futures["Close_prev"].shift(3)) / futures["Close_prev"].shift(3)

  futures_feature = futures[["Date", "gap_FUT", "mom_FUT"]]
  futures_feature = futures_feature.dropna().reset_index(drop=True)


  # US 10Y YIELD (^TNX) - interest rates / macro regime
  
  yield10y["Close_prev"] = yield10y["Close"].shift(1)

  # opening gap (overnight reaction)
  yield10y["gap_10Y"] = (yield10y["Open"] - yield10y["Close_prev"]) / yield10y["Close_prev"]

  # rate level change (TNX momentum) to the day before prediction
  yield10y["mom_10Y"] = (yield10y["Close_prev"] - yield10y["Close_prev"].shift(3)) / yield10y["Close_prev"].shift(3)

  yield10y_feature = yield10y[["Date", "gap_10Y", "mom_10Y"]]
  yield10y_feature = yield10y_feature.dropna().reset_index(drop=True)


  ########## ---------------------------------------------- ##########
  ##########          S&P500 FEATURES ON THE PREDICTION DAY           ##########
  ##########      (for target and derived features)               ##########
  ########## ---------------------------------------------- ##########

  # Previous day columns
  spy["Open_prev"] = spy["Open"].shift(1)
  spy["High_prev"] = spy["High"].shift(1)
  spy["Low_prev"] = spy["Low"].shift(1)
  spy["Close_prev"] = spy["Close"].shift(1)
  spy["Volume_prev"] = spy["Volume"].shift(1)

  # Opening gap
  spy["gap"] = spy["Open"] - spy["Close_prev"]
  spy["gap_pct"] = spy["gap"] / spy["Close_prev"]

  # Target
  spy["Target"] = dir(spy["Close"], spy["Open"])


  ########## ------------------------------------------------------------- ##########
  ##########      S&P500 FEATURES FROM THE DAY BEFORE PREDICTION      ##########
  ########## ------------------------------------------------------------- ##########


  # -----------------------------
  # Previous day candlestick with 3 classes
  # -----------------------------

  # real body = Close - Open
  spy["real_body_prev"] = spy["Close_prev"] - spy["Open_prev"]

  # candle direction
  spy["candle_dir_prev"] = dir(spy["Close_prev"], spy["Open_prev"])

  # upper shadow = High - max(Open, Close)
  spy["upper_shadow_prev"] = spy["High_prev"] - spy[["Open_prev", "Close_prev"]].max(axis=1)

  # lower shadow = min(Open, Close) - Low
  spy["lower_shadow_prev"] = spy[["Open_prev", "Close_prev"]].min(axis=1) - spy["Low_prev"]

  # ratio of lower shadow to body
  spy["lower_shadow_ratio"] = spy["lower_shadow_prev"] / (spy["real_body_prev"].abs().replace(0, 1e-6))

  # ratio of upper shadow to body
  spy["upper_shadow_ratio"] = spy["upper_shadow_prev"] / (spy["real_body_prev"].abs().replace(0, 1e-6))



  # -----------------------------
  # Trend indicators and moving averages (TA-Lib)
  # -----------------------------

  # short-term EMAs (fast reaction to changes)
  spy["ema_5"] = talib.EMA(spy["Close_prev"], timeperiod=5)
  spy["ema_10"] = talib.EMA(spy["Close_prev"], timeperiod=10)

  # medium-term SMAs (stable trend)
  spy["sma_20"] = talib.SMA(spy["Close_prev"], timeperiod=20)
  spy["sma_50"] = talib.SMA(spy["Close_prev"], timeperiod=50)

  # crossover features: short-term MA minus medium-term MA
  spy["ema5_minus_sma20"] = spy["ema_5"] - spy["sma_20"]
  spy["ema10_minus_sma20"] = spy["ema_10"] - spy["sma_20"]

  # deviation of opening price from moving averages (relative to MA)

  spy["open_minus_sma20_pct"] = (spy["Open"] - spy["sma_20"]) / spy["sma_20"]
  spy["open_minus_sma50_pct"] = (spy["Open"] - spy["sma_50"]) / spy["sma_50"]

  # additionally, add EMA/SMA relation features (percent-based)
  spy["ema5_minus_sma20_pct"] = (spy["ema_5"] - spy["sma_20"]) / spy["sma_20"]
  spy["ema10_minus_sma20_pct"] = (spy["ema_10"] - spy["sma_20"]) / spy["sma_20"]

  # ADX (Average Directional Index) - trend strength coefficient
  for p in [7, 14, 28]:
    spy[f'adx_{p}'] = talib.ADX(spy["High_prev"], spy["Low_prev"], spy["Close_prev"], timeperiod=p)

  # MACD - relation between two moving averages, used to identify trend direction and momentum
  macd, macdsignal, macdhist = talib.MACD(spy["Close_prev"], fastperiod=12, slowperiod=26, signalperiod=9)
  spy["macd"] = macd
  spy["macd_signal"] = macdsignal
  spy["macd_hist"] = macdhist

  # -----------------------------
  # Momentum and momentum indicators
  # -----------------------------

  # Momentum: close price difference
  for p in [5, 10, 20]:
    spy[f'momentum_{p}'] = spy["Close_prev"] - spy["Close_prev"].shift(p)
  spy["momentum_5_diff"] = spy["momentum_5"] - spy["momentum_5"].shift(1)

  # RSI - shows overbought or oversold market conditions based on recent gains and losses
  # base periods + dynamics
  for p in [7, 14, 21]:
    spy[f'rsi_{p}'] = talib.RSI(spy["Close_prev"], timeperiod=p)
    spy[f'rsi_{p}_diff'] = spy[f'rsi_{p}'] - spy[f'rsi_{p}'].shift(1)

  spy["rsi_14_centered"] = spy["rsi_14"] - 50
  spy["rsi_14_overbought"] = (spy["rsi_14"] > 70).astype(int)
  spy["rsi_14_oversold"] = (spy["rsi_14"] < 30).astype(int)

  # CCI (Commodity Channel Index) - deviation from the mean over time
  # base periods + dynamics
  for p in [10, 20, 40]:
    spy[f'cci_{p}'] = talib.CCI(spy["High_prev"], spy["Low_prev"], spy["Close_prev"], timeperiod=p)
    spy[f'cci_{p}_diff'] = spy[f'cci_{p}'] - spy[f'cci_{p}'].shift(1)

  spy["cci_20_centered"] = spy["cci_20"] / 100
  spy["cci_20_overbought"] = (spy["cci_20"] > 100).astype(int)
  spy["cci_20_oversold"] = (spy["cci_20"] < -100).astype(int)

  # Stochastic Oscillator (%K, %D) - how close the current price is to the period high or low
  spy["stoch_k"], spy["stoch_d"] = talib.STOCH(spy["High_prev"], spy["Low_prev"], spy["Close_prev"], fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)

  spy["stoch_k_minus_d"] = spy["stoch_k"] - spy["stoch_d"]
  spy["stoch_k_diff"] = spy["stoch_k"] - spy["stoch_k"].shift(1)
  spy["stoch_cross_up"] = (
    (spy["stoch_k"].shift(1) < spy["stoch_d"].shift(1)) &
    (spy["stoch_k"] > spy["stoch_d"])
  ).astype(int)

  spy["stoch_cross_down"] = (
    (spy["stoch_k"].shift(1) > spy["stoch_d"].shift(1)) &
    (spy["stoch_k"] < spy["stoch_d"])
  ).astype(int)

  spy["stoch_overbought"] = (spy["stoch_k"] > 80).astype(int)
  spy["stoch_oversold"] = (spy["stoch_k"] < 20).astype(int)


  # -----------------------------
  # Volume indicators (no TA-Lib function)
  # -----------------------------

  # Basic volume ratio
  spy["vol_ratio_5"] = spy["Volume_prev"] / spy["Volume_prev"].rolling(5).mean()

  # Volume ROC (Rate of Change) - measures percent volume change compared to N periods ago, showing pace (momentum) of market movement
  spy["vol_roc_5"] = spy["Volume_prev"].pct_change(5)

  # Volume spike - signal indicating an unusual sudden volume increase versus recent days
  spy["vol_spike"] = (spy["vol_ratio_5"] > 1.5).astype(int)

  # OBV (On-Balance Volume) with logarithm - shows capital flow direction by summing volume according to price moves
  # Vectorized OBV
  obv_raw = np.zeros(len(spy), dtype=np.float32)
  price_diff = np.sign(spy["Close_prev"].diff().fillna(0).values)
  volumes = spy["Volume_prev"].values

  # OBV accumulation
  obv_raw[0] = 0
  for i in range(1, len(spy)):
    obv_raw[i] = obv_raw[i-1] + price_diff[i] * volumes[i]

  # signed logarithm
  spy["obv"] = np.sign(obv_raw) * np.log1p(np.abs(obv_raw))

  # log-transformed OBV difference
  spy["obv_diff"] = spy["obv"].diff().fillna(0)

  # VWAP (daily Volume Weighted Average Price) - shows the volume-weighted average price traded during the day
  typical_price = (spy["High_prev"] + spy["Low_prev"] + spy["Close_prev"]) / 3
  vwap = (typical_price * spy["Volume_prev"]).cumsum() / spy["Volume_prev"].cumsum()

  # VWAP relation to price
  spy["price_vs_vwap"] = (spy["Close_prev"] - vwap) / vwap

  # volume confirmation
  spy["volume_trend_confirm"] = (
    (spy["Close_prev"] > spy["Close_prev"].shift(1)) &
    (spy["Volume_prev"] > spy["Volume_prev"].shift(1))
  ).astype(int)


  # -----------------------------
  # Volatility and ATR + Bollinger Bands indicators (TA-Lib)
  # -----------------------------

  # Basic volatility (standard deviation)
  for p in [5, 10, 20]:
    spy[f'volatility_{p}'] = np.log1p(spy["Close_prev"].rolling(p, min_periods=1).std())

  # ATR (Average True Range) 14 days - measures average price volatility over the period, i.e., typical market move size
  for p in [7, 14, 28]:
    spy[f'atr_{p}'] = np.log1p(talib.ATR(spy["High_prev"], spy["Low_prev"], spy["Close_prev"], timeperiod=p))
  
  # Bollinger Bands (20 days, 2 std) - show market volatility using upper and lower bands around a moving average, indicating potential overbought/oversold zones
  bb_upper, bb_middle, bb_lower = talib.BBANDS(spy["Close_prev"], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
  spy["bb_upper"] = bb_upper
  spy["bb_lower"] = bb_lower
  spy["bb_open_deviation"] = (spy["Open"] - bb_middle) / (bb_middle.replace(0, 1e-6))

  # -----------------------------
  # One-hot encoding of weekday, numbering from 1 (1=Monday, 5=Friday)
  # -----------------------------
  weekday_onehot = pd.get_dummies(spy["Date"].dt.weekday + 1, prefix="weekday")
  spy = pd.concat([spy, weekday_onehot], axis=1)


  ########## ------------------------------------ ##########
  ##########      MERGE AND DATA CLEANING         ##########
  ########## ------------------------------------ ##########

  # Drop columns to avoid leakage
  spy = spy.drop(columns=["High", "Low", "Close", "Volume"])

  # Log transform volume (done last because other indicators use volume)
  spy["Volume_prev"] = np.log1p(spy["Volume_prev"])

  # Drop NA values and reset index
  data = spy.dropna().reset_index(drop=True)

  # Merge indices, commodities and indicators into the SPY dataset
  data = data.merge(euro_stoxx_feature, on="Date", how="left")
  data = data.merge(nikkei_feature, on="Date", how="left")
  data = data.merge(taiex_feature, on="Date", how="left")
  data = data.merge(commodities, on="Date", how="left")
  data = data.merge(vix_feature, on="Date", how="left")
  data = data.merge(dxy_feature, on="Date", how="left")
  data = data.merge(futures_feature, on="Date", how="left")
  data = data.merge(yield10y_feature, on="Date", how="left")

  # Filter data by the requested date range
  data = data[
    (data["Date"] >= pd.to_datetime(start_date)) & 
    (data["Date"] <= pd.to_datetime(end_date))
  ]

  # Fill missing index and commodity data
  # - momentum and gap: forward fill, then fill zeros if the first row is NaN
  # - direction (3-class): forward fill
  commodities_cols = [c for c in data.columns if c.endswith(("_EU", "_NIK", "_AS", "_gold", "_oil", "_copper"))]
  data[commodities_cols] = data[commodities_cols].fillna(0)

  # Convert all features to float32
  data[data.columns.difference(["Date"])] = data[data.columns.difference(["Date"])].astype("float32")

  print(data['Target'].value_counts().reindex([0.0,1.0],fill_value=0))

  return data

data = get_features(start_date="2009-01-01", end_date="2026-04-15")
data.to_csv("data.csv", index=False)







###### Model training process (run only if --train is provided on the command line) ######

if "--train" in sys.argv:

  # --- 1. DEFINICJA FUNKCJI CELU DLA OPTUNA ---
  def objective(trial):
    df_trial = get_features(start_date="2010-01-01", end_date="2025-12-31")
    df_trial = df_trial.sort_values("Date").reset_index(drop=True)
    
    X_trial = df_trial.drop(columns=["Target", "Date"])
    y_trial = df_trial["Target"]

    param_grid = {
      "objective": "binary",
      "metric": "binary_logloss",
      "verbosity": -1,
      "boosting_type": "gbdt",
      "random_state": 42,
      "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
      "num_leaves": trial.suggest_int("num_leaves", 15, 127),
      "max_depth": trial.suggest_int("max_depth", 3, 12),
      "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 0.9),
      "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 0.9),
      "bagging_freq": trial.suggest_int("bagging_freq", 1, 5),
      "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
      "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
      "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
    }

    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = []

    for train_index, test_index in tscv.split(X_trial):
      X_train, X_test = X_trial.iloc[train_index], X_trial.iloc[test_index]
      y_train, y_test = y_trial.iloc[train_index], y_trial.iloc[test_index]

      # check whether both classes are present
      if len(np.unique(y_train)) < 2:
        return 999.0 

      train_data = lgb.Dataset(X_train, label=y_train)
      valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

      model = lgb.train(
        param_grid,
        train_data,
        num_boost_round=1500,
        valid_sets=[valid_data],
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=0)]
      )
      
      preds = model.predict(X_test)
      cv_scores.append(log_loss(y_test, preds))

    return np.mean(cv_scores)

  # --- 2. RUN OPTIMIZATION ---
  study = optuna.create_study(direction="minimize")
  study.optimize(objective, n_trials=100)

  print(f"\nBest parameters: {study.best_params}")

  # --- 3. FINAL TRAINING ---
  df_final = get_features(start_date="2010-01-01", end_date="2025-12-31")
  df_final = df_final.sort_values("Date").reset_index(drop=True)

  X_final = df_final.drop(columns=["Target", "Date"])
  y_final = df_final["Target"]

  best_model_params = study.best_params.copy()
  best_model_params.update({
    "objective": "binary",
    "metric": "binary_logloss",
    "verbosity": -1,
    "random_state": 42
  })

  split_idx = int(len(X_final) * 0.8)
  X_train_f, X_test_f = X_final.iloc[:split_idx], X_final.iloc[split_idx:]
  y_train_f, y_test_f = y_final.iloc[:split_idx], y_final.iloc[split_idx:]
  dates_test = df_final["Date"].iloc[split_idx:]

  final_train_ds = lgb.Dataset(X_train_f, label=y_train_f)
  final_valid_ds = lgb.Dataset(X_test_f, label=y_test_f, reference=final_train_ds)

  final_model = lgb.train(
    best_model_params,
    final_train_ds,
    num_boost_round=3000,
    valid_sets=[final_train_ds, final_valid_ds],
    valid_names=['train', 'valid'],
    callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(period=100)]
  )

  # --- 4. REPORTING AND MODEL EVALUATION ---

  # A. Predykcje
  y_prob = final_model.predict(X_test_f)
  y_pred = (y_prob > 0.5).astype(int)

  print("\n" + "="*30)
  print("CLASSIFICATION REPORT (Out-of-Sample Test Set)")
  print("="*30)
  target_names = ['Down (0)', 'Up (1)']
  print(classification_report(y_test_f, y_pred, target_names=target_names))

  # B. Confusion matrix
  plt.figure(figsize=(6, 5))
  cm = confusion_matrix(y_test_f, y_pred)
  sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
  plt.title('Confusion Matrix')
  plt.xlabel('Prediction')
  plt.ylabel('Actual')
  plt.show()

  # C. Feature importance
  plt.figure(figsize=(10, 8))
  lgb.plot_importance(final_model, max_num_features=20, importance_type='gain', title='Top 20 Features (Gain)')
  plt.show()

  # D. Simple directional backtest
  test_results = pd.DataFrame({
    'Date': dates_test,
    'Actual_Target': y_test_f,
    'Pred_Target': y_pred
  })

  test_results['Is_Correct'] = (test_results['Actual_Target'] == test_results['Pred_Target']).astype(int)

  plt.figure(figsize=(12, 6))
  plt.plot(test_results['Date'], test_results['Is_Correct'].rolling(50).mean(), label='Accuracy (Rolling 50)')
  plt.axhline(y=0.5, color='r', linestyle='--', label='Random Chance (50%)')
  plt.title('Rolling Model Accuracy')
  plt.legend()
  plt.show()

  print("\nDecision insights:")
  print(f"Class balance in the test set: \n{y_test_f.value_counts(normalize=True).sort_index()}")
  print("If Precision for class 'Up' or 'Down' is > 50%, the model has predictive potential.")

  # Save the model
  final_model.save_model('lgbm_sp500_model_binary.txt')


if "--test" in sys.argv:

  def manual_test_model(model=None, start_date="2026-01-01", end_date="2026-04-15", threshold=0.4):
    """Manual testing of a trained model.

    - loads data via `get_features(...)`
    - iterates over rows and makes predictions (loop required)
    - returns a DataFrame with columns: Date, Pred_prob, Pred, Target, Match
    - prints a percent summary of hits and misses

    If `model` is None, it uses the global `final_model` if available.
    """
    if model is None:
      try:
        model = final_model
      except NameError:
        raise ValueError("Model was not passed and `final_model` does not exist.")

    data = get_features(start_date=start_date, end_date=end_date)
    data = data.sort_values("Date").reset_index(drop=True)

    # feature columns (everything except Date and Target)
    feature_cols = [c for c in data.columns if c not in ("Date", "Target")]

    rows = []
    for _, row in data.iterrows():
      X = row[feature_cols].astype("float32").values.reshape(1, -1)
      # LightGBM predictor returns the probability for class 1
      pred_prob = float(model.predict(X)[0])
      pred_class = int(pred_prob > threshold)
      target = int(row["Target"]) if not pd.isna(row["Target"]) else None
      match = int(pred_class == target) if target is not None else 0

      dir_as = int(row["dir_AS"]) if ("dir_AS" in row.index and not pd.isna(row["dir_AS"])) else None

      rows.append({
        "Date": row["Date"],
        "dir_AS": dir_as,
        "prediction": pred_class,
        "actual_target": target,
        "prediction_result": match
      })

    results = pd.DataFrame(rows)

    # Overall statistics
    total_all = len(results)
    correct_all = int(results["prediction_result"].sum())
    wrong_all = total_all - correct_all
    pct_correct_all = (correct_all / total_all * 100) if total_all > 0 else 0.0
    print("\n--- Manual Test Summary ---")
    print(f"Total (all): {total_all} | Correct: {correct_all} ({pct_correct_all:.2f}%) | Wrong: {wrong_all}")
    print("Breakdown by prediction:")
    for p in [0, 1]:
      df_p = results[results["prediction"] == p]
      total_p = len(df_p)
      correct_p = int(df_p["prediction_result"].sum()) if total_p > 0 else 0
      wrong_p = total_p - correct_p
      pct_correct_p = (correct_p / total_p * 100) if total_p > 0 else 0.0
      pct_wrong_p = (wrong_p / total_p * 100) if total_p > 0 else 0.0
      print(f" Prediction={p}: Total: {total_p} | Correct: {correct_p} ({pct_correct_p:.2f}%) | Wrong: {wrong_p} ({pct_wrong_p:.2f}%)")
    print("---------------------------\n")

    # Return the results table with required columns (including `dir_AS`)
    cols = ["Date", "dir_AS", "prediction", "actual_target", "prediction_result"]
    # ensure columns exist (avoid error if missing)
    cols = [c for c in cols if c in results.columns]
    return results[cols]


  if __name__ == "__main__":
    # Load the model from file and run manual test
    loaded_model = lgb.Booster(model_file='lgbm_sp500_model_binary.txt')
    results = manual_test_model(model=loaded_model, start_date="2026-01-01", end_date="2026-04-15")
    results.to_csv("manual_test_results.csv", index=False)

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime, timedelta
from model import train

from constants import FEATURES

# =========================
# LOAD DATA AND CREATE PREDICTIONS WITH RETRAIN EVERY MONTH
# =========================
data = pd.DataFrame()

for date in [
  "2026-04-30",
  "2026-05-31"
]:
  month_data = pd.read_csv("data/data.csv")
  month_data["Datetime"] = pd.to_datetime(month_data["Datetime"])

  start_date = (datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
  end_date = (((datetime.strptime(start_date, "%Y-%m-%d").replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)).strftime("%Y-%m-%d"))

  month_data = month_data[
    (month_data["Datetime"] >= start_date) &
    (month_data["Datetime"] <= end_date)
  ]

  train(end_train=date) # retrain every month with new data and different prop value for testing
  model = joblib.load("models/lgbm_model.pkl")

  X = month_data[FEATURES]

  month_data["Pred"] = model.predict(X)
  month_data = month_data[["Date_NY", "Hour_NY", "Session_weekday", "Open", "Close", "High", "Low", "Dir", "Target", "Pred", "Day_dir_till_hour"]]

  data = pd.concat([data, month_data], ignore_index=True)


# =========================
# PREPARE BACKTEST
# =========================


# Test only selected hours
data = data[data["Hour_NY"].isin([11])].reset_index(drop=True)

data["Pred"] = (data["Pred"]).astype(np.float32)
data["Pred_dir"] = np.where(
  data["Day_dir_till_hour"] == data["Pred"],
  1.0,
  -1.0
).astype(np.float32)
data["Pct_diff"] = (((data["Close"] - data["Open"]) / data["Open"]) * 100).round(2)

data["High_pct"] = np.where(
  data["Close"] > data["Open"],
  ((data["High"] - data["Close"]) / data["Close"]) * 100,
  ((data["High"] - data["Open"]) / data["Open"]) * 100
).round(2)

data["Low_pct"] = np.where(
  data["Close"] > data["Open"],
  ((data["Open"] - data["Low"]) / data["Open"]) * 100,
  ((data["Close"] - data["Low"]) / data["Close"]) * 100
).round(2)

# SL/TP with leverage & spread (%)
tp = 100.0 # no TP in current strategy
sl = 6.0

spread = 0.3 # avg. value

data["Early_stop_value"] = np.where(
  (data["Pred_dir"] == 1),
  data["Low_pct"]*20,
  data["High_pct"]*20,
)

data["Early_take_value"] = np.where(
  (data["Pred_dir"] == 1),
  data["High_pct"]*20,
  data["Low_pct"]*20,
)

data["Abs_ret_pct_x_lev"] = (
  np.where(
    data["Early_stop_value"] >= sl,
    -sl,
    np.where(
      data["Early_take_value"] >= tp,
      tp,
      np.where(
        (data["Pred_dir"] == data["Dir"]),
        np.minimum(data["Pct_diff"].abs() * 20, tp),
        -np.minimum(data["Pct_diff"].abs() * 20, sl)
      )
    )
  ) / 100
).astype(np.float32)

data["Ret_m_spread_x_lev"] = (data["Abs_ret_pct_x_lev"] - spread/100)

"""
Min cost formula based on XTB inf.

1 lot value - price * 50
min. position - 0.005
leverage - 0.05

min. unit cost (0.005) - price * 50 * 0.05 * 0.005

"""

data["Min_cost"] = (data["Open"]*0.0125)



# Hourly capital return loop
initial_capital = 156.0

data["Capital"] = 0.0
data["Invested"] = 0.0
data["Unused"] = 0.0
data["Units"] = 0

capital = initial_capital

for i in range(len(data)):

  min_cost = data.loc[i, "Min_cost"]
  ret = data.loc[i, "Ret_m_spread_x_lev"]

  # how many units I can buy
  units = int(capital // min_cost)

  # invested capital
  invested = units * min_cost

  # not invested capital
  unused = capital - invested

  # full capital after trade
  capital = invested * (1 + ret) + unused

  data.loc[i, "Capital"] = capital


# Select data to report
data = data[[
  "Date_NY",
  "Hour_NY",
  "Session_weekday",
  "Pred",
  "Pred_dir",
  "Dir",
  "Pct_diff",
  "Ret_m_spread_x_lev",
  "Capital"
]]


print(data.tail(60))

print("=========================")
print(f"Accuracy: {(data['Pred_dir'] == data['Dir']).mean() * 100:.2f}%")
print(f"Plus days %: {(data['Ret_m_spread_x_lev'] > 0).mean() * 100:.2f}%")

# =========================
# PLOT BACKTEST RESULTS
# =========================

plt.figure(figsize=(12, 6))
plt.plot(data["Date_NY"], data["Capital"], marker="o")
plt.xlabel("Data")
plt.ylabel("Kapitał po transakcji")
plt.title(f"Całkowity kapitał od 2026-01-01")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()